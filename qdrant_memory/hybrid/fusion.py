"""Pure fusion helpers for the Phase 5 hybrid retrieve path.

These helpers are intentionally stdlib-only. They operate on already-ranked
fragments from the three retrieval lanes (dense+sparse, graph, RAPTOR) and
produce a stable, dedupe-by-point-id output without ever mutating Qdrant.

Public functions:

* :func:`rrf_fuse` — Reciprocal Rank Fusion over a sequence of ranked
  candidate lists. Each input list contributes ``1 / (k + rank)`` per
  point id. Output is sorted by fused score descending.
* :func:`deduplicate_by_point_id` — deterministic dedupe that keeps the
  first occurrence of each point id and replaces duplicates with their
  first-seen values.

No global state, no I/O, no logging. The function return types are plain
dicts/lists so callers can serialize them to JSON without touching the
plugin infrastructure.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Any


_RRF_K_DEFAULT = 60.0


def rrf_fuse(
    ranked_lists: Iterable[Iterable[Mapping[str, Any]]],
    *,
    k: float = _RRF_K_DEFAULT,
    point_id_key: str = "point_id",
    score_key: str = "_rrf_score",
) -> list[dict[str, Any]]:
    """Reciprocal Rank Fusion over a sequence of ranked candidate lists.

    Each input list contributes ``1 / (k + rank)`` to the fused score of
    every point id present in that list. ``rank`` is 1-based (rank 1
    receives the largest contribution).

    Returns a new list of dicts, sorted by fused score descending. Each
    output dict is a **shallow copy** of the first-seen occurrence of a
    point id, extended with ``score_key`` (the fused score) and
    ``"_rrf_ranks"`` (the ranks where it appeared in each input list).
    Duplicate occurrences from later lists do not overwrite earlier values.
    """
    if k <= 0:
        raise ValueError("rrf k must be positive")
    fused: dict[str, dict[str, Any]] = {}
    contributions: dict[str, list[float]] = {}
    ranks_seen: dict[str, list[int]] = {}

    for lane_index, ranked_list in enumerate(ranked_lists):
        for rank, item in enumerate(ranked_list, start=1):
            if not isinstance(item, Mapping):
                continue
            point_id = str(item.get(point_id_key) or item.get("id") or "").strip()
            if not point_id:
                continue
            contribution = 1.0 / (float(k) + float(rank))
            contributions.setdefault(point_id, [0.0] * lane_index)
            contributions[point_id].append(contribution)
            ranks_seen.setdefault(point_id, [])
            ranks_seen[point_id].append(rank)
            if point_id not in fused:
                fused[point_id] = dict(item)
                fused[point_id][point_id_key] = point_id

    out: list[dict[str, Any]] = []
    for point_id, base in fused.items():
        score = sum(contributions.get(point_id, []))
        out_item = dict(base)
        out_item[score_key] = float(score)
        out_item["_rrf_ranks"] = list(ranks_seen.get(point_id, []))
        out_item["_rrf_lanes"] = len(ranks_seen.get(point_id, []))
        out.append(out_item)

    out.sort(key=lambda item: item.get(score_key, 0.0), reverse=True)
    return out


def deduplicate_by_point_id(
    items: Iterable[Mapping[str, Any]],
    *,
    point_id_key: str = "point_id",
) -> list[dict[str, Any]]:
    """Stable dedupe of an iterable of mappings, keyed by point id.

    The first occurrence of each point id wins. Subsequent occurrences are
    silently dropped. Non-mapping items and items without a non-empty point
    id are dropped.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, Mapping):
            continue
        point_id = str(item.get(point_id_key) or item.get("id") or "").strip()
        if not point_id:
            continue
        if point_id in seen:
            continue
        seen.add(point_id)
        out.append(dict(item))
    return out


__all__ = [
    "deduplicate_by_point_id",
    "rrf_fuse",
]
