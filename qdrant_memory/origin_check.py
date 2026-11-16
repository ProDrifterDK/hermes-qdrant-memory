"""Read-only check of ``session_origin`` citations (C'4, design 5.3 and 5.9).

``check_origins`` scrolls one page of the points that carry ``session_origin`` in one
collection, re-verifies every citation against ``state.db`` with C'2's
``verify_origin_citation`` (``verify_citation`` for spans, a presence check for
``tool_position``), and reports counts per current status and the point ids per status.

It never writes: no payload update, no access counters (it scrolls, it does not search), no
payload index, no collection creation, and ``state.db`` is only opened read-only by the
verifier. Every status is informational. ``expired`` is host retention, not a retraction,
and nothing is flagged or demoted because of any status.
"""
from __future__ import annotations

from typing import Any, Mapping

from qdrant_memory.session_origin import SessionOriginRefused, validate_session_origin, verify_origin_citation

DEFAULT_LIMIT = 100
MAX_LIMIT = 500
DEFAULT_MAX_IDS = 100
MAX_IDS = 500
NOTE = ("Read-only: nothing was changed. expired means the host pruned the session (retention), "
        "not that the fact was retracted; no status flags or demotes a point.")


def _bounded(value: Any, default: int, cap: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(cap, parsed))


def parse_offset(value: Any) -> Any:
    """A scroll continuation as the tool received it: ``None``, an unsigned int or a UUID string."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    text = str(value).strip()
    return int(text) if text.isdigit() else (text or None)


def origin_filter(scope: Mapping[str, str]) -> dict[str, Any]:
    """Points with a non-empty ``session_origin``, inside the provider's scope."""
    flt: dict[str, Any] = {"must_not": [{"is_empty": {"key": "session_origin"}}]}
    must = [{"key": key, "match": {"value": value}} for key, value in sorted(scope.items()) if value]
    if must:
        flt["must"] = must
    return flt


def _add(bucket: dict[str, list[str]], key: str, point_id: str) -> None:
    ids = bucket.setdefault(key, [])
    if point_id not in ids:
        ids.append(point_id)


def _capped(bucket: dict[str, list[str]], max_ids: int) -> tuple[dict[str, list[str]], list[str]]:
    truncated = sorted(key for key, ids in bucket.items() if len(ids) > max_ids)
    return {key: ids[:max_ids] for key, ids in sorted(bucket.items())}, truncated


def check_origins(qdrant: Any, collection_name: str, *, state_db: str, scope: Mapping[str, str] | None = None,
                  limit: Any = DEFAULT_LIMIT, offset: Any = None, max_ids: Any = DEFAULT_MAX_IDS) -> dict[str, Any]:
    """One bounded page of the check over ``collection_name``.

    ``qdrant`` needs ``get_collections`` and ``scroll_page`` only. ``counts`` are per citation
    (the status belongs to the citation); ``point_counts`` and ``point_ids`` count and list
    each point once under every status one of its citations has (``point_counts`` is never
    truncated). A valid mapping with no citation (``unresolved:<reason>``) is
    reported under ``uncited_*``, a mapping that fails the v1 validator under ``invalid_*``;
    neither aborts the page. Raises only what the Qdrant client raises.
    """
    page_limit = _bounded(limit, DEFAULT_LIMIT, MAX_LIMIT)
    id_cap = _bounded(max_ids, DEFAULT_MAX_IDS, MAX_IDS)
    start = parse_offset(offset)
    result: dict[str, Any] = {"read_only": True, "collection_name": collection_name, "limit": page_limit,
                              "offset": start}
    if collection_name not in set(qdrant.get_collections()):
        result.update({"collection_missing": True, "points_checked": 0, "citations_checked": 0, "counts": {},
                       "point_counts": {}, "point_ids": {}, "point_ids_truncated": [], "uncited_counts": {}, "uncited_point_ids": {},
                       "invalid_count": 0, "invalid_point_ids": [], "next_offset": None, "complete": True,
                       "note": NOTE})
        return result
    points, next_offset = qdrant.scroll_page(collection_name, origin_filter(scope or {}), limit=page_limit,
                                             offset=start, with_payload=["session_origin"], with_vector=False)
    counts: dict[str, int] = {}
    by_status: dict[str, list[str]] = {}
    uncited_counts: dict[str, int] = {}
    uncited: dict[str, list[str]] = {}
    invalid: list[str] = []
    citations_checked = 0
    for point in points:
        point_id = str(point.get("id"))
        payload = point.get("payload") if isinstance(point.get("payload"), Mapping) else {}
        mapping = payload.get("session_origin")
        try:
            validate_session_origin(mapping)
        except SessionOriginRefused:
            if point_id not in invalid:
                invalid.append(point_id)
            continue
        assert isinstance(mapping, Mapping)  # validate_session_origin refuses anything else
        if not mapping["citations"]:
            status = str(mapping["citation_status"])
            uncited_counts[status] = uncited_counts.get(status, 0) + 1
            _add(uncited, status, point_id)
            continue
        for citation in mapping["citations"]:
            current = verify_origin_citation(citation, state_db=state_db)["status"]
            citations_checked += 1
            counts[current] = counts.get(current, 0) + 1
            _add(by_status, current, point_id)
    point_ids, truncated = _capped(by_status, id_cap)
    uncited_ids, uncited_truncated = _capped(uncited, id_cap)
    result.update({
        "points_checked": len(points),
        "citations_checked": citations_checked,
        "counts": dict(sorted(counts.items())),
        "point_counts": {status: len(ids) for status, ids in sorted(by_status.items())},
        "point_ids": point_ids,
        "point_ids_truncated": sorted(set(truncated) | {f"uncited:{key}" for key in uncited_truncated}),
        "uncited_counts": dict(sorted(uncited_counts.items())),
        "uncited_point_ids": uncited_ids,
        "invalid_count": len(invalid),
        "invalid_point_ids": invalid[:id_cap],
        "next_offset": None if next_offset is None else str(next_offset),
        "complete": next_offset is None,
        "note": NOTE,
    })
    return result


__all__ = ["check_origins", "origin_filter", "parse_offset", "DEFAULT_LIMIT", "MAX_LIMIT", "DEFAULT_MAX_IDS",
           "MAX_IDS"]
