from __future__ import annotations

import asyncio
import json
import time
import uuid
from pathlib import Path
from typing import Any

from src.config import AppConfig
from src.layout.preprocessor import preprocess_page
from src.layout.detector import LayoutDetector
from src.layout.reading_order import xy_cut_order
from src.layout.cropper import crop_and_save
from src.extraction.vlm_client import VLMClient
from src.extraction.chunker import Chunker
from src.extraction.table_parser import (
    parse_table_response,
    reclassify_figure_as_table,
    detect_table_continuation,
)
from src.extraction.validator import Validator
from src.models import (
    Document,
    DocumentChunk,
    RawRegion,
    RegionType,
    ExtractionSchema,
    CURRENT_SCHEMA_VERSION,
)
from src.router import SmartRouter
from src.observability import (
    get_logger,
    DOCS_PROCESSED,
    PIPELINE_LATENCY,
    DEDUP_HITS,
    CONFIDENCE_HIST,
    LAYOUT_REGIONS,
    timed,
    VLM_LATENCY,
)
from src.storage.base import StorageAdapter

logger = get_logger(__name__)


class ExtractionPipeline:
    """Orchestrates the full document extraction pipeline.

    Pipeline stages (in order):
    1. Dedup check — sha256 content_hash against all active adapters.
    2. Ingest — rasterise document to page images.
    3. Preprocess — deskew/denoise/contrast each page image.
    4. Layout detect — DocLayout-YOLO per page.
    5. Reading order — XY-cut reorder per page.
    6. Crop — save region images to IMAGE_STORE_PATH.
    7. Chunk — group regions into DocumentChunk objects.
    8. VLM extract — parallel async calls with concurrency_limit.
    9. Validate + correct — per-chunk validation loop.
    10. Assemble Document model.
    11. Write to active storage adapters (manifest-tracked).

    All write failures are recorded per-adapter in a manifest file alongside
    the document in IMAGE_STORE_PATH so failed adapters can be re-run without
    re-extracting.
    """

    def __init__(
        self,
        config: AppConfig,
        adapters: list[StorageAdapter],
    ):
        self._config = config
        self._adapters = adapters
        self._router = SmartRouter()
        self._detector = LayoutDetector(config.layout)
        self._vlm = VLMClient.from_config(config.vlm)
        self._validator = Validator(self._vlm, max_passes=config.vlm.correction_max_passes)
        self._chunker = Chunker()

    @classmethod
    def from_env(cls) -> "ExtractionPipeline":
        """Construct pipeline and all adapters from environment configuration.

        Reads AppConfig from environment, creates only the adapters whose
        connection strings are configured (non-None).

        Returns:
            ExtractionPipeline: fully initialised pipeline instance.
        """
        config = AppConfig()
        adapters: list[StorageAdapter] = []

        if config.database.mongodb_uri:
            from src.storage.mongodb import MongoAdapter
            adapters.append(MongoAdapter.from_config(config.database))

        if config.database.postgres_dsn:
            from src.storage.postgres import PostgresAdapter
            adapters.append(
                PostgresAdapter.from_config(config.database, config.embedding)
            )

        if config.database.neo4j_uri:
            from src.storage.neo4j_store import Neo4jAdapter
            adapters.append(Neo4jAdapter.from_config(config.database))

        from src.storage.vector import VectorStore
        vec = VectorStore.from_config(
            config.database, config.embedding, config.visual_embedding
        )
        adapters.append(vec)

        return cls(config, adapters)

    async def run(
        self,
        source_path: str,
        extraction_schema: ExtractionSchema | None = None,
        targets: list[str] | None = None,
    ) -> Document:
        """Execute the full extraction pipeline for a single document.

        Computes sha256 of source file and checks all active adapters for an
        existing document with that hash. Returns the cached Document immediately
        if found (dedup hit). Otherwise runs the full pipeline and writes to all
        adapters in targets (or all adapters if targets is None).

        Args:
            source_path: absolute path to the source document.
            extraction_schema: optional user-defined field extraction schema.
            targets: list of adapter class names to write to. None = all adapters.

        Returns:
            Document: fully extracted and stored document.
        """
        start_time = time.perf_counter()
        document_id = str(uuid.uuid4())
        log = logger.bind(document_id=document_id, source_path=source_path)

        # --- Stage 1: Route + Dedup ----------------------------------------
        decision = self._router.route(source_path)
        content_hash = decision.content_hash

        active_adapters = self._select_adapters(targets)
        for adapter in active_adapters:
            existing = await adapter.find_by_hash(content_hash)
            if existing:
                DEDUP_HITS.inc()
                log.info("dedup_hit", existing_document_id=existing.document_id)
                return existing

        # --- Stage 2: Ingest -------------------------------------------------
        log.info("pipeline_stage", stage="ingest", format=decision.format)
        ingestor = decision.ingestor_cls()
        pages = ingestor.ingest(source_path)
        for page in pages:
            page.quality_tier = decision.quality_tier

        # --- Stages 3–6: Layout + Crop per page ------------------------------
        all_regions: list[RawRegion] = []

        for page_result in pages:
            preprocessed = preprocess_page(page_result.image)

            with timed(VLM_LATENCY, ["layout"]):
                raw_regions = self._detector.detect(
                    preprocessed, page_result.page_number, document_id
                )

            ordered = xy_cut_order(raw_regions)

            for region in ordered:
                region = crop_and_save(preprocessed, region, self._config.database)
                LAYOUT_REGIONS.labels(region.region_type.value).inc()
                all_regions.append(region)

        # --- Stage 7: Chunk --------------------------------------------------
        log.info("pipeline_stage", stage="chunk", region_count=len(all_regions))
        chunks = self._chunker.chunk_regions(all_regions, document_id)

        # --- Stages 8–9: VLM extract + Validate (bounded concurrency) --------
        log.info("pipeline_stage", stage="extract", chunk_count=len(chunks))
        self._regions_cache = all_regions  # used by _find_region_for_chunk
        sem = asyncio.Semaphore(self._config.vlm.concurrency_limit)
        tasks = [
            self._extract_chunk(chunk, extraction_schema, sem)
            for chunk in chunks
        ]
        chunks = await asyncio.gather(*tasks)

        # --- Stage 10: Assemble Document -------------------------------------
        review_required = any(c.confidence_score < 0.5 for c in chunks)
        document = Document(
            document_id=document_id,
            source_path=source_path,
            content_hash=content_hash,
            format=decision.format,
            total_pages=len(pages),
            processing_status="review_required" if review_required else "complete",
            chunks=list(chunks),
            regions=all_regions,
            metadata={
                "quality_tier": decision.quality_tier,
                "ingestor": decision.ingestor_cls.__name__,
            },
            schema_version=CURRENT_SCHEMA_VERSION,
        )

        # --- Stage 11: Write to adapters (manifest-tracked) ------------------
        manifest = await self._write_to_adapters(document, active_adapters)
        await self._save_manifest(document_id, manifest)

        elapsed = time.perf_counter() - start_time
        PIPELINE_LATENCY.observe(elapsed)
        DOCS_PROCESSED.labels(document.processing_status).inc()

        log.info(
            "pipeline_complete",
            status=document.processing_status,
            chunks=len(chunks),
            elapsed_s=round(elapsed, 2),
        )
        return document

    async def _extract_chunk(
        self,
        chunk: DocumentChunk,
        schema: ExtractionSchema | None,
        sem: asyncio.Semaphore,
    ) -> DocumentChunk:
        """Run VLM extraction and validation for a single chunk.

        Skips VLM for HEADER/FOOTER chunks tagged metadata_only=True.
        Populates chunk fields from the validated VLM response.

        Returns:
            DocumentChunk: chunk with all extraction fields populated.
        """
        if chunk.metadata.get("metadata_only"):
            return chunk

        region = self._find_region_for_chunk(chunk)
        if region is None:
            return chunk

        async with sem:
            with timed(VLM_LATENCY, [chunk.content_type.value]):
                raw_response = await self._vlm.extract(
                    region,
                    schema,
                    self._config.vlm.extraction_max_tokens,
                )

            corrected, correction_applied, confidence = (
                await self._validator.validate_and_correct(region, raw_response, schema)
            )

        chunk = self._populate_chunk_from_response(chunk, corrected)
        chunk.correction_applied = correction_applied
        chunk.confidence_score = confidence
        CONFIDENCE_HIST.observe(confidence)

        return chunk

    def _find_region_for_chunk(self, chunk: DocumentChunk) -> RawRegion | None:
        """Locate the source RawRegion for a chunk by matching cropped_image_path.

        Returns:
            RawRegion if found, None if the chunk has no associated region.
        """
        for region in self._all_regions_cache:
            if region.cropped_image_path == chunk.cropped_image_path:
                return region
        return None

    def _populate_chunk_from_response(
        self, chunk: DocumentChunk, response: dict
    ) -> DocumentChunk:
        """Write VLM response fields into the appropriate DocumentChunk attributes.

        Handles type-specific field mapping:
          TABLE → parse_table_response, with figure reclassification check.
          FIGURE → image_description from response["description"].
          FORMULA → formula_latex from response["latex"].
          All types → raw_text, overview, structured_data, entities.

        Returns:
            DocumentChunk: updated chunk.
        """
        chunk.raw_text = response.get("text", response.get("raw_text", ""))
        chunk.overview = response.get("overview", "")
        chunk.entities = response.get("entities", [])

        if chunk.content_type == RegionType.TABLE or (
            chunk.content_type == RegionType.FIGURE
            and reclassify_figure_as_table(response)
        ):
            chunk.content_type = RegionType.TABLE
            chunk.table_data = parse_table_response(response)

        elif chunk.content_type == RegionType.FIGURE:
            chunk.image_description = response.get("description", "")

        elif chunk.content_type == RegionType.FORMULA:
            chunk.formula_latex = response.get("latex", "")

        if "fields" in response:
            chunk.structured_data = response["fields"]
        else:
            chunk.structured_data = {
                k: v
                for k, v in response.items()
                if k not in {"text", "raw_text", "overview", "entities", "markdown",
                             "headers", "rows", "cells", "description", "latex",
                             "plain_text", "image_type", "embedded_text", "confidence",
                             "metadata_only"}
            }

        return chunk

    def _select_adapters(self, targets: list[str] | None) -> list[StorageAdapter]:
        """Return the subset of adapters matching the targets list.

        Returns:
            list[StorageAdapter]: active adapters to write to.
        """
        if not targets:
            return self._adapters
        return [
            a for a in self._adapters
            if type(a).__name__.lower().replace("adapter", "").replace("store", "")
            in [t.lower() for t in targets]
        ]

    async def _write_to_adapters(
        self,
        document: Document,
        adapters: list[StorageAdapter],
    ) -> dict[str, str]:
        """Write document to all adapters. Record success/failure per adapter.

        Returns:
            dict[str, str]: mapping of adapter_name → "ok" or error message.
        """
        manifest: dict[str, str] = {}
        for adapter in adapters:
            name = type(adapter).__name__
            try:
                await adapter.write_document(document)
                manifest[name] = "ok"
                logger.info("adapter_write_ok", adapter=name, document_id=document.document_id)
            except Exception as exc:
                manifest[name] = str(exc)
                from src.observability import WRITE_FAILURES
                WRITE_FAILURES.labels(name).inc()
                logger.error(
                    "adapter_write_failed",
                    adapter=name,
                    document_id=document.document_id,
                    error=str(exc),
                )
        return manifest

    async def _save_manifest(
        self, document_id: str, manifest: dict[str, str]
    ) -> None:
        """Save the write-result manifest as JSON alongside the image store.

        Returns:
            None
        """
        manifest_dir = Path(self._config.database.image_store_path) / document_id
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / "manifest.json"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2))
        except OSError as exc:
            logger.warning("manifest_save_failed", error=str(exc))

    @property
    def _all_regions_cache(self) -> list[RawRegion]:
        """Lazy cache of all_regions built during the current run.

        Note: this is set by run() and only valid within a single pipeline execution.
        Returns empty list if accessed outside a run context.

        Returns:
            list[RawRegion]
        """
        return getattr(self, "_regions_cache", [])
