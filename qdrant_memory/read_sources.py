"""Recall over the provider's own store plus read-only ``read_sources``.

The read sources are other profiles' collections. This module only ever calls
``collect_candidates`` on them, which issues read requests only (search, plus the
sparse lane's scroll) and never touches access counters; the Qdrant client
additionally refuses every mutating request against them
(``QdrantClient.read_only_collections``), so a read source stays untouched even if a
call site here or elsewhere gets it wrong.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MergedRecall:
    """Search the own store and each read source, then score the union once.

    ``own`` and every source are ``MemoryRetriever`` or ``LearningStore`` objects
    (``collect_candidates(query, *, top_k, **filters)``, ``rank_candidates(pools)``).
    The candidates of all collections are scored together by ``own.rank_candidates``:
    one min-max normalization over every collection's dense scores, then the same
    final scoring and ``min_final_score`` cut the single-collection search applies, so
    scores from different collections are comparable. A read source that cannot be
    searched (for example a collection that does not exist) is skipped with a debug
    log; it is never created.
    """

    def __init__(self, own: Any, own_collection: str, sources: list[tuple[str, Any]]):
        self.own = own
        self.own_collection = own_collection
        self.sources = sources

    def search_labelled(self, query: str, top_k: int = 5, update_access: bool = True, **kwargs: Any) -> list[tuple[str, Any]]:
        limit = max(1, min(20, int(top_k)))
        pools = [self.own.collect_candidates(query, top_k=limit, **kwargs)]
        for collection, source in self.sources:
            try:
                pools.append(source.collect_candidates(query, top_k=limit, **kwargs))
            except Exception:
                logger.debug("Qdrant read source %s skipped: search failed", collection, exc_info=True)
        # Sorted best first; the sort is stable, so on a tie the own collection keeps
        # precedence. A point id present in several collections keeps its best hit.
        selected: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for collection, chunk in self.own.rank_candidates(pools):
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            selected.append((collection, chunk))
            if len(selected) >= limit:
                break
        if update_access:
            # Only the own store's counters, and only for own hits the merge kept: what
            # the own store alone would have done for the hits it returned.
            own_hits = [chunk for collection, chunk in selected if collection == self.own_collection]
            if own_hits:
                self.own.update_access_metadata(own_hits)
        return selected

    def search(self, query: str, top_k: int = 5, **kwargs: Any) -> list[Any]:
        return [chunk for _, chunk in self.search_labelled(query, top_k=top_k, **kwargs)]
