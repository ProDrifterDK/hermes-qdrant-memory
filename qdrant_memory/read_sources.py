"""Recall over the provider's own store plus read-only ``read_sources``.

The read sources are other profiles' collections. This module only ever calls
``search(..., update_access=False)`` on them; the Qdrant client additionally refuses
every mutating request against them (``QdrantClient.read_only_collections``), so a
read source stays untouched even if a call site here or elsewhere gets it wrong.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MergedRecall:
    """Search the own store and each read source, then merge the hits by score.

    ``own`` and every source expose ``search(query, *, top_k, update_access, **filters)``
    and return chunks with ``id`` and ``final_score`` (``MemoryRetriever`` or
    ``LearningStore``). A read source that cannot be searched (for example a collection
    that does not exist) is skipped with a debug log; it is never created.
    """

    def __init__(self, own: Any, own_collection: str, sources: list[tuple[str, Any]]):
        self.own = own
        self.own_collection = own_collection
        self.sources = sources

    def search_labelled(self, query: str, top_k: int = 5, update_access: bool = True, **kwargs: Any) -> list[tuple[str, Any]]:
        limit = max(1, min(20, int(top_k)))
        labelled = [
            (self.own_collection, chunk)
            for chunk in self.own.search(query, top_k=limit, update_access=False, **kwargs)
        ]
        for collection, source in self.sources:
            try:
                hits = source.search(query, top_k=limit, update_access=False, **kwargs)
            except Exception:
                logger.debug("Qdrant read source %s skipped: search failed", collection, exc_info=True)
                continue
            labelled.extend((collection, chunk) for chunk in hits)
        # final_score is normalized per collection, so each collection's best hit tends to
        # tie; the raw vector score (same embedding model everywhere) breaks those ties.
        # Stable sort: on a full tie the own collection keeps precedence.
        labelled.sort(key=lambda item: (float(item[1].final_score), float(item[1].qdrant_score)), reverse=True)
        selected: list[tuple[str, Any]] = []
        seen: set[str] = set()
        for collection, chunk in labelled:
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
