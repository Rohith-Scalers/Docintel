"""Fused multi-source retrieval using Reciprocal Rank Fusion."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.models import DocumentChunk, RetrievalResult

logger = logging.getLogger(__name__)

_RRF_K_DEFAULT = 60


class FusedRetrieval:
    """Multi-source retrieval with Reciprocal Rank Fusion (RRF).

    Queries all configured retrieval adapters in parallel using asyncio.gather.
    Each adapter returns an independent ranked list. RRF merges them:
        score(chunk) = sum(1 / (k + rank_i)) for each adapter i that ranked it
    where k=60 (standard RRF constant from Cormack et al. 2009).

    Results are de-duplicated by chunk_id. Each returned chunk gains:
      metadata["retrieval_sources"]: list of adapter names that returned it
      metadata["rrf_score"]: float RRF score

    Note: FusedRetrieval does not support cursor pagination (top_k is the
    final result count). Use individual adapters for paginated access.
    """

    def __init__(self, adapters: dict[str, Any], rrf_k: int = _RRF_K_DEFAULT) -> None:
        """Initialise with a mapping of adapter_name → retrieval instance.

        adapters: dict mapping name strings to retrieval instances. Each
            instance must have a search() or query() method returning RetrievalResult.
        rrf_k: RRF smoothing constant. Default 60 follows Cormack et al.
        """
        self._adapters = adapters
        self._rrf_k = rrf_k

    async def search(
        self,
        query_text: str,
        top_k: int = 20,
        metadata_filter: dict | None = None,
        include_adapters: list[str] | None = None,
    ) -> RetrievalResult:
        """Query all (or selected) adapters in parallel and fuse results with RRF.

        include_adapters: if provided, only these adapter names are queried.

        Returns:
            RetrievalResult with top_k chunks sorted by RRF score descending.
            Each chunk has metadata["retrieval_sources"] and metadata["rrf_score"].
        """
        active: dict[str, Any] = (
            {k: v for k, v in self._adapters.items() if k in include_adapters}
            if include_adapters
            else self._adapters
        )

        tasks = {
            name: asyncio.create_task(
                self._query_adapter(name, adapter, query_text, top_k, metadata_filter)
            )
            for name, adapter in active.items()
        }

        results: dict[str, list[DocumentChunk]] = {}
        for name, task in tasks.items():
            try:
                ranked = await task
                results[name] = ranked
            except Exception:
                logger.warning(
                    "Adapter '%s' failed during fused retrieval; excluding from fusion",
                    name,
                    exc_info=True,
                )

        fused = _reciprocal_rank_fusion(results, self._rrf_k)
        top = fused[:top_k]

        return RetrievalResult(chunks=top, total=len(top), next_cursor=None)

    @staticmethod
    async def _query_adapter(
        name: str,
        adapter: Any,
        query_text: str,
        top_k: int,
        metadata_filter: dict | None,
    ) -> list[DocumentChunk]:
        """Call search() or query() on an adapter and return its ranked chunk list.

        Adapters with a search() method are called with query_text and metadata_filter.
        Adapters with only a query() method are called without query_text
        (metadata_filter is passed as the filters argument).

        Returns:
            Ordered list of DocumentChunk objects from this adapter.
        """
        if hasattr(adapter, "search"):
            result: RetrievalResult = await adapter.search(
                query_text=query_text,
                top_k=top_k,
                metadata_filter=metadata_filter,
                limit=top_k,
            )
        elif hasattr(adapter, "query"):
            result = await adapter.query(
                filters=metadata_filter,
                limit=top_k,
            )
        else:
            raise AttributeError(
                f"Adapter '{name}' has neither search() nor query() method"
            )
        return result.chunks


def _reciprocal_rank_fusion(
    ranked_lists: dict[str, list[DocumentChunk]],
    k: int,
) -> list[DocumentChunk]:
    """Merge multiple ranked chunk lists using Reciprocal Rank Fusion.

    For each chunk appearing in one or more lists:
        rrf_score = sum(1 / (k + rank_i))  where rank_i is 1-based position

    Chunks are de-duplicated by chunk_id. The chunk object from the highest-
    scoring adapter (lowest rank) is retained in the output.

    Returns:
        List of DocumentChunk sorted by rrf_score descending.
    """
    scores: dict[str, float] = {}
    sources: dict[str, list[str]] = {}
    best_chunk: dict[str, DocumentChunk] = {}

    for adapter_name, chunks in ranked_lists.items():
        for rank_zero, chunk in enumerate(chunks):
            cid = chunk.chunk_id
            rank_one = rank_zero + 1
            contribution = 1.0 / (k + rank_one)

            scores[cid] = scores.get(cid, 0.0) + contribution
            sources.setdefault(cid, []).append(adapter_name)

            if cid not in best_chunk or rank_one < _get_rank(
                best_chunk[cid], ranked_lists
            ):
                best_chunk[cid] = chunk

    for cid, chunk in best_chunk.items():
        chunk.metadata["rrf_score"] = scores[cid]
        chunk.metadata["retrieval_sources"] = sources[cid]

    return sorted(best_chunk.values(), key=lambda c: c.metadata["rrf_score"], reverse=True)


def _get_rank(chunk: DocumentChunk, ranked_lists: dict[str, list[DocumentChunk]]) -> int:
    """Return the best (lowest) 1-based rank of chunk across all adapter lists.

    Returns:
        Lowest rank position found; len(all chunks) + 1 if not found.
    """
    best = sum(len(v) for v in ranked_lists.values()) + 1
    for chunks in ranked_lists.values():
        for i, c in enumerate(chunks):
            if c.chunk_id == chunk.chunk_id:
                best = min(best, i + 1)
    return best
