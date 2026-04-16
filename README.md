# docintel

Production-grade Python document intelligence package. Ingests documents of any format, detects layout regions with DocLayout-YOLO, extracts structured data with DeepSeek-OCR-2 (end-to-end VLM), validates and auto-corrects extractions, and stores results across MongoDB, PostgreSQL, Neo4j, and a vector database. Includes fused retrieval with Reciprocal Rank Fusion.

---

## Overview

The pipeline stages for every document:

1. **Smart Router** — detects format, scores quality, computes sha256 dedup hash
2. **Ingestor** — rasterises pages to images (PDF, DOCX, XLS, image, text, email)
3. **Preprocessor** — deskew, denoise, CLAHE contrast normalisation per page
4. **Layout Detector** — DocLayout-YOLO detects all regions per page with bounding boxes
5. **Reading Order** — XY-cut algorithm corrects multi-column reading order
6. **Cropper** — saves each region as a PNG crop under `IMAGE_STORE_PATH`
7. **Chunker** — groups regions into chunks, links captions, bridges page breaks
8. **VLM Extraction** — DeepSeek-OCR-2 called per chunk with typed prompts
9. **Validator** — semantic field rules; up to 2 targeted correction re-prompts
10. **Storage** — writes to all active database adapters (manifest-tracked)

---

## Setup

Requirements: Docker and Docker Compose.

```
cp .env .env.local
# Edit .env.local with your VLM and embedding endpoints
docker compose up -d
```

Run database migrations (PostgreSQL only):

```
docker compose exec celery-worker alembic upgrade head
```

Download DocLayout-YOLO weights (required before first run):

```
# Place the .pt file at configs/doclayout_yolo.pt
# Download from: https://huggingface.co/opendatalab/DocLayout-YOLO
```

---

## Configuration

All configuration is via environment variables loaded from `.env`. The `.env` file shipped in this repository contains safe defaults with empty secrets.

### VLM (DeepSeek-OCR-2)

| Variable | Description | Default |
|---|---|---|
| `VLM_ENDPOINT` | OpenAI-compatible chat completions URL | `http://localhost:8000/v1/chat/completions` |
| `VLM_MODEL` | Model name sent in request body | `deepseek-ocr-2` |
| `VLM_API_KEY` | Bearer token (empty if no auth) | — |
| `VLM_CONCURRENCY_LIMIT` | Max parallel VLM calls per document | `8` |
| `VLM_CORRECTION_MAX_PASSES` | Max validation correction re-prompts | `2` |

### Text Embedding

| Variable | Description | Default |
|---|---|---|
| `EMBED_ENDPOINT` | OpenAI-compatible embeddings URL | `http://localhost:8001/v1/embeddings` |
| `EMBED_MODEL` | Embedding model name | `BAAI/bge-m3` |
| `EMBED_DIMENSIONS` | Vector dimensions (must match DB index) | `1024` |

### Visual Embedding (optional — ColPali / ColQwen2)

Set `VISUAL_EMBED_ENABLED=true` to send FIGURE chunk crops to a multimodal embedding endpoint instead of embedding the text overview. Required for image-heavy document corpora.

| Variable | Description |
|---|---|
| `VISUAL_EMBED_ENABLED` | Enable visual embedding for FIGURE chunks |
| `VISUAL_EMBED_ENDPOINT` | ColPali/ColQwen2 embeddings URL |
| `VISUAL_EMBED_MODEL` | e.g. `vidore/colqwen2-v1.0` |
| `VISUAL_EMBED_DIMENSIONS` | 128 for ColPali |

### Layout Detector

| Variable | Description | Default |
|---|---|---|
| `LAYOUT_MODEL_PATH` | Path to DocLayout-YOLO .pt weights | `configs/doclayout_yolo.pt` |
| `LAYOUT_CONF_THRESHOLD` | Minimum detection confidence | `0.25` |
| `LAYOUT_DEVICE` | `cpu`, `cuda`, or `cuda:0` | `cpu` |

---

## Supported Formats

| Format | Extension(s) | Notes |
|---|---|---|
| PDF | `.pdf` | Digital-native and scanned; mixed-mode per-page |
| Word | `.docx`, `.doc` | Converted to PDF via docx2pdf |
| Excel | `.xlsx`, `.xls`, `.xlsm` | Each sheet is one page; cell data preserved as TableData |
| Images | `.jpg`, `.png`, `.tiff`, `.bmp`, `.webp` | Treated as single-page scanned |
| Plain text | `.txt`, `.md`, `.csv`, `.html` | Rendered to page images |
| Email | `.eml`, `.msg` | Body extracted; attachments queued as separate tasks |

---

## Databases

### MongoDB

NoSQL document store. Best for: flexible querying by metadata, full-text search over `overview` fields, storing documents with evolving schemas.

Collections: `documents`, `chunks`, `regions`.

- `overview` field has a text index — supports `$text` full-text search.
- `content_hash` is unique-indexed for deduplication.
- `table_data` stored as nested BSON, queryable via `$elemMatch` on cells.

Configure: `DB_MONGODB_URI=mongodb://user:pass@host:27017/docintel`

### PostgreSQL

Relational store with JSONB for structured extraction results. Best for: SQL queries on chunk metadata, JSONB path operators on `structured_data`, and as the pgvector backend for vector embeddings without a separate vector DB.

Tables: `documents`, `regions`, `chunks`.

- `structured_data JSONB` has a GIN index — supports `@>`, `?`, `jsonb_path_exists`.
- `table_data JSONB` similarly indexed.
- `embedding VECTOR(N)` column is added automatically when pgvector is the active vector backend.
- Schema migrations are managed by Alembic. Run `alembic upgrade head` after deploy.

Configure: `DB_POSTGRES_DSN=postgresql://user:pass@host:5432/dbname`

### Neo4j

Graph database. Best for: cross-document entity relationship queries, traversing which documents mention the same organisation or person, table row graph traversal.

Node labels: `:Document`, `:Chunk`, `:Region`, `:Entity`, `:TableRow`

Relationships:
- `(:Document)-[:HAS_CHUNK]->(:Chunk)`
- `(:Chunk)-[:NEXT_CHUNK]->(:Chunk)` (reading order chain)
- `(:Chunk)-[:MENTIONS]->(:Entity)` (NER output)
- `(:Chunk)-[:HAS_ROW]->(:TableRow)`

Entity nodes are MERGEd by `text + type` — the same organisation appearing in 100 documents creates one `:Entity` node with 100 `MENTIONS` edges. This enables:

```cypher
MATCH (e:Entity {text: "Acme Corp"})<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
RETURN d.source_path, c.overview, c.page_number
ORDER BY d.source_path
```

Configure: `DB_NEO4J_URI=bolt://host:7687`, `DB_NEO4J_USER`, `DB_NEO4J_PASSWORD`

### Vector Database

Semantic similarity search over chunk embeddings.

**pgvector (default):** embeddings stored in `chunks.embedding VECTOR(N)` column. Uses `<=>` cosine distance operator. No extra service required.

**Qdrant:** standalone vector DB with support for hybrid search (ANN + payload metadata filter). Recommended for large corpora (>1M chunks).

Select backend: `DB_VECTOR_BACKEND=pgvector` or `DB_VECTOR_BACKEND=qdrant`

For Qdrant: `DB_QDRANT_HOST=qdrant`, `DB_QDRANT_PORT=6333`

Image handling in vector store: FIGURE chunks are embedded using the VLM-generated `image_description` text (or `overview` as fallback). When `VISUAL_EMBED_ENABLED=true`, the cropped image itself is sent to the visual embedding endpoint. The `cropped_image_path` is always stored in the vector payload so a search hit immediately returns the image file reference without a second database lookup.

---

## Retrieval

### Per-adapter retrieval

```python
from src.retrieval.mongo_retrieval import MongoRetrieval
from src.retrieval.vector_retrieval import VectorRetrieval
from src.retrieval.neo4j_retrieval import Neo4jRetrieval

# MongoDB — filter by content type and full-text search
result = await mongo_ret.query(
    content_types=["table"],
    full_text="quarterly revenue",
    limit=20,
)

# Vector — semantic ANN search
result = await vector_ret.search(
    query_text="total operating expenses by division",
    top_k=10,
    metadata_filter={"content_type": "table"},
)

# Neo4j — entity-based graph traversal
result = await neo4j_ret.traverse(
    start_entity="Acme Corp",
    entity_type="ORG",
    depth=2,
    limit=15,
)

# Paginate with cursor
next_result = await mongo_ret.query(
    full_text="quarterly revenue",
    limit=20,
    cursor=result.next_cursor,
)
```

### Fused retrieval (Reciprocal Rank Fusion)

Queries multiple adapters in parallel and merges ranked results using RRF (k=60). Each result chunk gets `metadata["retrieval_sources"]` and `metadata["rrf_score"]`.

```python
from src.retrieval.fused import FusedRetrieval

fused = FusedRetrieval(adapters={
    "vector": vector_ret,
    "neo4j": neo4j_ret,
    "mongo": mongo_ret,
})

result = fused.search(
    query_text="operating expenses breakdown Q3 2025",
    top_k=10,
    include_adapters=["vector", "neo4j"],
)

for chunk in result.chunks:
    print(chunk.overview)
    print(chunk.metadata["rrf_score"], chunk.metadata["retrieval_sources"])
```

---

## Submitting Documents

```python
from src.worker import submit
from src.models import ExtractionSchema

schema = ExtractionSchema(
    fields={
        "invoice_number": "The invoice ID or reference number",
        "total_amount": "The total amount due including tax",
        "vendor_name": "Name of the issuing vendor or supplier",
    },
    extract_entities=True,
)

document_id = submit(
    source_path="/data/invoices/invoice_2025_001.pdf",
    extraction_schema=schema,
    targets=["mongodb", "neo4j", "vector"],
)
```

Fire-and-forget (non-blocking):

```python
from src.worker import process_document_task

result = process_document_task.delay(
    "/data/contracts/contract_abc.pdf",
    schema.model_dump(),
    ["postgres"],
)
# Check later:
document_id = result.get(timeout=300)
```

---

## Observability

Prometheus metrics are exposed at `http://celery-worker:9090/metrics`.

| Metric | Type | Description |
|---|---|---|
| `docintel_documents_processed_total` | Counter | Documents by final status |
| `docintel_vlm_latency_seconds` | Histogram | VLM call latency by region_type |
| `docintel_storage_write_failures_total` | Counter | Write failures by adapter name |
| `docintel_chunk_confidence` | Histogram | Distribution of chunk confidence scores |
| `docintel_correction_passes_total` | Counter | Correction passes by region_type and pass number |
| `docintel_layout_regions_total` | Counter | Detected regions by type |
| `docintel_pipeline_latency_seconds` | Histogram | End-to-end pipeline latency |
| `docintel_dedup_hits_total` | Counter | Documents skipped by deduplication |

Structured JSON logs include `document_id`, `stage`, `duration_ms`, and `chunk_count` on every pipeline event.

OpenTelemetry distributed tracing: set `OBS_OTEL_ENDPOINT=http://collector:4317` to export spans to any OTLP-compatible collector (Jaeger, Tempo, Honeycomb).

---

## Schema Migration

When the `DocumentChunk` model evolves, increment `CURRENT_SCHEMA_VERSION` in `src/models.py` and add a migration path in `StorageAdapter._adapt_chunk()` in `src/storage/base.py`.

For PostgreSQL:

```
alembic revision --autogenerate -m "add_new_column"
alembic upgrade head
```

Older records at `schema_version=N` are transparently upgraded on read via `_adapt_chunk()`. The minimum supported version is `SUPPORTED_SCHEMA_VERSIONS = {1, 2}` — records below this raise `UnsupportedSchemaVersion`.

---

## Deployment Notes

**Distributed deployment:** `IMAGE_STORE_PATH` is a local filesystem path. If the Celery worker and a database service run on different machines, replace the `crop_and_save()` call in `src/layout/cropper.py` with an object-store write (S3/MinIO) and store the object key as `cropped_image_path`. Set `DB_IMAGE_BASE_URL` to the public HTTP prefix so callers can construct image URLs from the stored path.

**Vector backend choice:**
- `pgvector`: simpler deployment (no extra service), good to ~500k chunks.
- `qdrant`: hybrid search (ANN + payload filter in one query), better at scale.
- pgvector and Qdrant cannot be active simultaneously — set `DB_VECTOR_BACKEND` to select one.

**GPU acceleration:** Set `LAYOUT_DEVICE=cuda` for DocLayout-YOLO inference. The Celery worker container must have the NVIDIA container runtime configured.
