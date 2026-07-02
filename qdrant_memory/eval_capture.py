"""Phase 6B read-only capture core for RAPTOR/hybrid retrieval variants.

This module is the **capture** half of the Phase 6B eval slice. Unlike
Phase 6A's :mod:`qdrant_memory.evaluation` (which is stdlib-only and
offline), this module MAY contact live local Qdrant/embeddings — but
**only when the operator explicitly invokes the capture command**, and
**never mutates** Qdrant:

* Every retrieval call passes ``update_access=False``.
* No ``upsert``, ``delete_ids``, ``delete_filter``, ``update_payload``,
  or access-metadata write path is reachable.
* The dense lane suppresses sparse scroll unless the ``dense+sparse``
  variant explicitly enables it (still read-only Qdrant ``scroll``).

The capture core accepts an **already-initialized provider** (so the CLI
can own provider construction) and a list of validated Phase 6A eval
cases, then emits JSONL run rows whose ``packet`` shape is compatible
with :mod:`qdrant_memory.evaluation` scoring.

Hard privacy/safety rules enforced here:

* Run rows identify a capture by ``case_id``, ``variant``, ``packet``,
  ``latency_ms``, and a sanitized ``capture`` metadata dict only.
* **Raw query text is NEVER serialized** into run rows. The cases file
  already contains query text by operator design; the runs file must
  not duplicate it so a persisted runs file cannot leak queries.
* Captured errors are sanitized to ``<redacted>`` plus a stable error
  kind, never raw exception strings (which can embed query text,
  Qdrant request details, or packet snippets).
* This module does not register a Hermes tool. The CLI command is the
  only entry point; ``build_tool_call`` raises ``CliUsageError`` for
  ``eval-capture`` so provider dispatch fails closed.
"""

from __future__ import annotations

import time
from typing import Any, Mapping

from qdrant_memory.hybrid.router import (
    _dense_to_exact_hits,
    _graph_to_relations,
)


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

# Canonical variant identifiers. The lowercase forms
# ``raptor-only`` and ``hybrid-no-raptor`` are the canonical JSON keys
# (the design brief used mixed-case names for readability; JSON output
# uses these lowercase forms).
DEFAULT_CAPTURE_VARIANTS: tuple[str, ...] = (
    "dense-only",
    "dense+sparse",
    "graph",
    "raptor-only",
    "hybrid",
    "hybrid-no-graph",
    "hybrid-no-raptor",
)

_ALL_ALIAS = "all"

# Sentinel used in run-row ``error`` and ``capture.error_kind`` fields.
# The raw captured exception text is NEVER serialized.
_REDACTED_ERROR = "<redacted>"


# --------------------------------------------------------------------------- #
# Variant parsing
# --------------------------------------------------------------------------- #


def parse_variants(value: str | list[str] | None) -> list[str]:
    """Parse a variant selector into an ordered list of canonical variants.

    Accepts:

    * ``None`` → all default variants.
    * ``"all"`` (case-insensitive) → all default variants.
    * A comma-separated string like ``"dense-only,graph,hybrid"``.
    * A list of strings.

    Each token is stripped and lowercased for comparison. Unknown
    variants raise :class:`ValueError` so a typo surfaces immediately
    rather than silently producing an incomplete capture.

    Order follows the canonical :data:`DEFAULT_CAPTURE_VARIANTS`
    sequence for determinism, regardless of the input order.
    """

    if value is None:
        return list(DEFAULT_CAPTURE_VARIANTS)
    if isinstance(value, str):
        tokens = [token.strip() for token in value.split(",")]
    else:
        tokens = [str(token).strip() for token in value]
    normalized: set[str] = set()
    for token in tokens:
        if not token:
            continue
        lowered = token.lower()
        normalized.add(lowered)
    if not normalized:
        return list(DEFAULT_CAPTURE_VARIANTS)
    if _ALL_ALIAS in normalized:
        return list(DEFAULT_CAPTURE_VARIANTS)
    unknown = sorted(normalized - set(DEFAULT_CAPTURE_VARIANTS))
    if unknown:
        raise ValueError(
            f"unknown capture variant(s): {', '.join(unknown)}"
        )
    # Preserve canonical ordering for deterministic output.
    return [variant for variant in DEFAULT_CAPTURE_VARIANTS if variant in normalized]


# --------------------------------------------------------------------------- #
# Error sanitization
# --------------------------------------------------------------------------- #


def _error_kind(exc: BaseException) -> str:
    """Return a stable, sanitized error kind for a captured exception.

    Never includes the raw exception message (which may embed query
    text, Qdrant request details, or packet snippets). Uses the
    exception class name as the primary signal.
    """

    name = type(exc).__name__ or "Exception"
    # Collapse common Qdrant/HTTP/Connection errors to stable kinds so
    # the runs file stays free of raw message text while still giving
    # the operator an actionable category.
    lowered = name.lower()
    if "timeout" in lowered:
        return "timeout"
    if "connection" in lowered:
        return "connection_error"
    return name


def _sanitized_error_row(case_id: str, variant: str, exc: BaseException,
                         capture_meta: dict[str, Any]) -> dict[str, Any]:
    """Build a run row for a capture that raised an exception.

    The row carries the redacted sentinel plus the error kind, never
    the raw exception string. It also exposes top-level numeric
    ``latency_ms`` so the shape matches the success rows (which carry
    a top-level ``latency_ms`` derived from the same ``capture_meta``).
    The raw exception/query text is never serialized into either the
    row body or the ``capture`` metadata block.
    """

    kind = _error_kind(exc)
    safe_meta = _safe_capture_metadata(capture_meta)
    latency_ms = safe_meta.get("latency_ms")
    # Defensive: coerce to a numeric when possible. The capture core
    # writes ``latency_ms`` as ``round(..., 1)`` (a float). If for any
    # reason the value is missing or non-numeric, surface ``None``
    # rather than letting a string leak through. This preserves the
    # success-row shape so the Phase 6A evaluator can treat both
    # success and error rows uniformly on ``latency_ms``.
    if isinstance(latency_ms, bool):
        # bool is an int subclass; exclude it explicitly so True/False
        # never masquerade as a numeric latency.
        latency_ms = None
    elif latency_ms is not None and not isinstance(latency_ms, (int, float)):
        try:
            latency_ms = float(latency_ms)
        except (TypeError, ValueError):
            latency_ms = None
    return {
        "case_id": case_id,
        "variant": variant,
        "packet": {
            "results": {
                "exact_hits": [],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            },
        },
        "latency_ms": latency_ms,
        "error": _REDACTED_ERROR,
        "capture": {
            **safe_meta,
            "error_kind": kind,
            "errored": True,
        },
    }


# --------------------------------------------------------------------------- #
# Capture metadata sanitization
# --------------------------------------------------------------------------- #

# Keys that are safe to echo back in the ``capture`` metadata block of
# a run row. These are small, non-sensitive config labels. Query text
# is never included.
_SAFE_CAPTURE_KEYS: frozenset[str] = frozenset({
    "top_k",
    "mode",
    "max_depth",
    "max_children",
    "max_source_chars",
    "candidate_seed_top_k",
    "max_graph_results",
    "include_fact_history",
    "include_metadata",
    "variant",
    "latency_ms",
    "errored",
    "error_kind",
    "collection_name",
})


def _safe_capture_metadata(capture: Mapping[str, Any]) -> dict[str, Any]:
    """Project a capture metadata dict down to safe keys only."""

    if not isinstance(capture, Mapping):
        return {}
    return {
        key: capture[key]
        for key in _SAFE_CAPTURE_KEYS
        if key in capture
    }


# --------------------------------------------------------------------------- #
# Packet shape helpers
# --------------------------------------------------------------------------- #


def _empty_results() -> dict[str, list]:
    """Return the canonical empty Phase 5 grouped results dict."""

    return {
        "exact_hits": [],
        "summaries": [],
        "cited_leaves": [],
        "graph_relations": [],
    }


def _packet_from_results(
    *,
    exact_hits: list[dict[str, Any]] | None = None,
    summaries: list[dict[str, Any]] | None = None,
    cited_leaves: list[dict[str, Any]] | None = None,
    graph_relations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a Phase 5 grouped-retrieve packet shape from lane lists."""

    results = _empty_results()
    if exact_hits:
        results["exact_hits"] = list(exact_hits)
    if summaries:
        results["summaries"] = list(summaries)
    if cited_leaves:
        results["cited_leaves"] = list(cited_leaves)
    if graph_relations:
        results["graph_relations"] = list(graph_relations)
    return {"results": results}


# --------------------------------------------------------------------------- #
# Dense projection (local, preserves evaluator fields)
# --------------------------------------------------------------------------- #


def _dense_chunk_to_exact_hit(chunk: Any) -> dict[str, Any]:
    """Local projection kept for back-compat / single-chunk callers.

    Prefer :func:`qdrant_memory.hybrid.router._dense_to_exact_hits` for
    any new code: that helper applies secret-bearing text/id/payload
    redaction, unsafe-status demotion, per-hit text truncation, and a
    hard context-char budget. The Phase 6B capture core reuses that
    helper directly via :func:`_dense_projection` below so captured
    rows never carry secret-shaped text or point ids through the dense
    lane. This single-chunk projection is intentionally NOT a public
    re-export and is only here for symmetry with the graph projection
    helper. **Do not** call it directly when writing a packet row:
    use :func:`_dense_projection` instead.
    """

    payload: dict[str, Any] = {}
    if hasattr(chunk, "payload"):
        try:
            payload = dict(getattr(chunk, "payload") or {})
        except Exception:
            payload = {}
    point_id = str(getattr(chunk, "id", "") or "")
    text = str(getattr(chunk, "text", "") or "")
    score: float = 0.0
    for key in ("final_score", "qdrant_score"):
        value = getattr(chunk, key, None)
        if value is not None:
            try:
                score = float(value)
                break
            except (TypeError, ValueError):
                pass
    item: dict[str, Any] = {
        "point_id": point_id,
        "text": text,
        "score": score,
        "source_uri": str(payload.get("source_uri") or payload.get("source") or ""),
        "file_path": str(payload.get("file_path") or payload.get("path") or ""),
        "heading": str(payload.get("heading") or ""),
    }
    return item


# --------------------------------------------------------------------------- #
# Sanitized dense projection (re-uses router's secret-bearing filter)
# --------------------------------------------------------------------------- #


def _dense_projection(
    dense_chunks: list[Any],
    *,
    query: str | None = None,
    include_fact_history: bool = False,
    max_source_chars: int,
) -> list[dict[str, Any]]:
    """Project dense chunks through the router's sanitized helper.

    Re-uses :func:`qdrant_memory.hybrid.router._dense_to_exact_hits` so
    secret-bearing content (point id, text, payload projection,
    ranking_debug) is dropped before the packet is emitted, and so
    unsafe-status markers (stale / review / quarantined /
    raptor-excluded / unsafe fact_status) are demoted from active
    ``results.exact_hits`` — unless ``include_fact_history=True`` is
    forwarded through this function to the router helper. The
    per-hit ``max_source_chars`` truncation and hard context-char
    budget are also applied by the router helper, matching the
    behavior of the live :class:`HybridRouter`.

    The ``warnings`` list is intentionally not collected: the capture
    runs file is the persisted output, and the router's per-hit
    warnings would add operator-only noise. Secret-bearing drops are
    still enforced (they happen inside the router helper regardless
    of whether ``warnings`` is provided).
    """

    return _dense_to_exact_hits(
        dense_chunks,
        query=query,
        warnings=None,
        include_fact_history=include_fact_history,
        max_source_chars=max_source_chars,
        # The router's hard context char budget is the read-only
        # invariant the live system enforces; the capture variant
        # preserves it so the captured runs file is comparable to
        # the live system output.
        hard_context_char_budget=None,
    ) or []


# --------------------------------------------------------------------------- #
# Graph projection (local, preserves evaluator fields)
# --------------------------------------------------------------------------- #


def _graph_result_to_relations(graph_result: Any) -> list[dict[str, Any]]:
    """Local projection kept for back-compat / single-candidate callers.

    Prefer :func:`qdrant_memory.hybrid.router._graph_to_relations` for
    any new code: that helper applies the secret-bearing filter on
    ``point_id``, every element of ``path``, and every element of
    ``relation_path``. The Phase 6B capture core reuses that helper
    directly via :func:`_graph_projection` below so captured rows
    never carry secret-shaped ids or relation paths through the graph
    lane. This local projection is intentionally NOT a public
    re-export and is only here for symmetry with the dense projection
    helper. **Do not** call it directly when writing a packet row:
    use :func:`_graph_projection` instead.
    """

    if graph_result is None:
        return []
    final = getattr(graph_result, "final", None) or []
    out: list[dict[str, Any]] = []
    for candidate in final:
        if candidate is None:
            continue
        point_id = str(getattr(candidate, "point_id", "") or "")
        relation: dict[str, Any] = {
            "point_id": point_id,
            "graph_distance": int(getattr(candidate, "graph_distance", 0) or 0),
            "final_score": float(getattr(candidate, "final_score", 0.0) or 0.0),
            "path": list(getattr(candidate, "path", []) or []),
            "relation_path": list(getattr(candidate, "relation_path", []) or []),
        }
        # Surface text if the candidate carries it so the evaluator's
        # term-matching has something to match against.
        text = getattr(candidate, "text", None)
        if isinstance(text, str) and text:
            relation["text"] = text
        source_uri = getattr(candidate, "source_uri", None)
        if isinstance(source_uri, str) and source_uri:
            relation["source_uri"] = source_uri
        file_path = getattr(candidate, "file_path", None)
        if isinstance(file_path, str) and file_path:
            relation["file_path"] = file_path
        out.append(relation)
    return out


# --------------------------------------------------------------------------- #
# Sanitized graph projection (re-uses router's secret-bearing filter)
# --------------------------------------------------------------------------- #


def _graph_projection(
    graph_result: Any,
    *,
    query: str | None = None,
    max_source_chars: int | None = None,
    include_fact_history: bool = False,
) -> list[dict[str, Any]]:
    """Project graph candidates through the router's sanitized helper.

    Re-uses :func:`qdrant_memory.hybrid.router._graph_to_relations`
    so secret-bearing candidates (point id, ``path`` element,
    ``relation_path`` element, OR any payload-derived ``source_uri`` /
    ``file_path`` / ``heading`` / raw ``text`` matches
    ``contains_secret``) are dropped before the packet is emitted.
    ``warnings`` is intentionally not collected: the capture runs
    file is the persisted output, and the router's per-hit warnings
    would add operator-only noise.

    Phase 6E: ``max_source_chars`` and ``include_fact_history`` are
    forwarded so the eval-capture rows carry the same
    ``source_uri`` / ``file_path`` / ``heading`` / bounded ``text``
    handle projection (and the same unsafe-status demotion gate)
    that the live HybridRouter path now provides.
    """

    return _graph_to_relations(
        graph_result,
        query=query,
        warnings=None,
        max_source_chars=max_source_chars,
        include_fact_history=include_fact_history,
    ) or []


# --------------------------------------------------------------------------- #
# Provider component extraction
# --------------------------------------------------------------------------- #


def _get_attr(obj: Any, name: str, default: Any = None) -> Any:
    """Safe attribute getter that returns ``default`` on any failure."""

    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _build_components(provider: Any) -> dict[str, Any]:
    """Extract the reusable components from an initialized provider.

    Returns a dict with:

    * ``qdrant`` — the Qdrant client.
    * ``embeddings`` — the embedding client.
    * ``collection_name`` — the memory collection name.
    * ``scope`` — the provider scope filter values.
    * ``base_retriever`` — the provider's ``MemoryRetriever``.
    * ``graph_retriever`` — the provider's ``GraphMemoryRetriever`` (may be None).
    * ``raptor_searcher`` — the provider's ``RaptorSearcher`` (may be None).

    The capture core uses these to build variant-specific
    retrievers/routers. It never reaches private mutable state beyond
    what is needed for read-only retrieval.
    """

    # Collection name: prefer provider config, fall back to "memory".
    config = _get_attr(provider, "_config", {}) or {}
    collection_name = str(config.get("collection_name") or "memory")

    # The provider lazily builds graph/raptor components. Accessing the
    # ensure_* methods triggers the lazy construction without mutation.
    graph_retriever = None
    method = _get_attr(provider, "_ensure_graph_retriever")
    if callable(method):
        try:
            graph_retriever = method(collection_name)
        except Exception:
            graph_retriever = None
        if graph_retriever is None:
            # Some providers store it directly on the attribute.
            graph_retriever = _get_attr(provider, "_graph_retriever", None)

    raptor_searcher = None
    method = _get_attr(provider, "_ensure_raptor_searcher")
    if callable(method):
        try:
            raptor_searcher = method(collection_name)
        except Exception:
            raptor_searcher = None
        if raptor_searcher is None:
            raptor_searcher = _get_attr(provider, "_raptor_searcher", None)

    return {
        "qdrant": _get_attr(provider, "_qdrant", None),
        "embeddings": _get_attr(provider, "_embeddings", None),
        "collection_name": collection_name,
        "scope": _get_attr(provider, "_scope_filter_values", lambda: {})() or {},
        "base_retriever": _get_attr(provider, "_retriever", None),
        "graph_retriever": graph_retriever,
        "raptor_searcher": raptor_searcher,
    }


# --------------------------------------------------------------------------- #
# Variant runners
# --------------------------------------------------------------------------- #
#
# Each runner takes the components dict + query + params and returns a
# Phase 5 grouped-retrieve packet (dict with ``results``). All
# retrieval calls pass ``update_access=False``. The dense-only variant
# suppresses sparse scroll; the dense+sparse variant allows it.
#
# Runners never serialize raw query text into the returned packet.


def _run_dense_only(
    components: dict[str, Any],
    query: str,
    *,
    top_k: int,
    max_source_chars: int,
    include_fact_history: bool,
) -> dict[str, Any]:
    """Dense-only: use ``MemoryRetriever`` with sparse scroll suppressed.

    Calls ``.search(query, top_k=..., update_access=False,
    allow_sparse_scroll=False, ...)`` so neither sparse scroll nor
    access-metadata writes fire.
    """

    base_retriever = components["base_retriever"]
    if base_retriever is None:
        raise RuntimeError("provider base retriever is not initialized")
    chunks = base_retriever.search(
        query,
        top_k=top_k,
        update_access=False,
        allow_sparse_scroll=False,
        include_fact_history=include_fact_history,
    )
    # Re-use the router's sanitized dense projection so secret-bearing
    # text/point id/payload and unsafe-status markers are dropped
    # before the run row is serialized.
    exact_hits = _dense_projection(
        chunks,
        query=query,
        include_fact_history=include_fact_history,
        max_source_chars=max_source_chars,
    )
    return _packet_from_results(exact_hits=exact_hits)


def _run_dense_sparse(
    components: dict[str, Any],
    query: str,
    *,
    top_k: int,
    max_source_chars: int,
    include_fact_history: bool,
) -> dict[str, Any]:
    """Dense+sparse: allow sparse scroll so the literal sparse baseline runs.

    Still ``update_access=False`` (read-only). The sparse lane uses
    Qdrant ``scroll``, which is a read operation, so this is safe.
    """

    base_retriever = components["base_retriever"]
    if base_retriever is None:
        raise RuntimeError("provider base retriever is not initialized")
    chunks = base_retriever.search(
        query,
        top_k=top_k,
        update_access=False,
        allow_sparse_scroll=True,
        include_fact_history=include_fact_history,
    )
    exact_hits = _dense_projection(
        chunks,
        query=query,
        include_fact_history=include_fact_history,
        max_source_chars=max_source_chars,
    )
    return _packet_from_results(exact_hits=exact_hits)


def _run_graph(
    components: dict[str, Any],
    query: str,
    *,
    top_k: int,
    candidate_seed_top_k: int,
    max_graph_results: int,
    max_depth: int,
    max_source_chars: int,
    include_fact_history: bool,
) -> dict[str, Any]:
    """Graph-only: use ``GraphMemoryRetriever.search`` read-only.

    Graph expansion is enabled (``allow_graph_scroll=True``) because
    this variant is specifically measuring graph expansion. The seed
    dense search still suppresses sparse scroll and access updates.
    """

    graph_retriever = components["graph_retriever"]
    if graph_retriever is None:
        raise RuntimeError("provider graph retriever is not initialized")
    graph_result = graph_retriever.search(
        query,
        top_k=top_k,
        candidate_seed_top_k=candidate_seed_top_k,
        max_graph_results=max_graph_results,
        max_depth=max_depth,
        include_fact_history=include_fact_history,
        debug=True,
        allow_sparse_scroll=False,
        allow_graph_scroll=True,
    )
    # Re-use the router's sanitized graph projection so secret-bearing
    # point ids, path entries, relation-path entries, AND payload-
    # derived ``source_uri`` / ``file_path`` / ``heading`` / raw
    # ``text`` are dropped before the run row is serialized.
    # ``max_source_chars`` is forwarded so the per-relation text
    # cap is enforced identically to the live HybridRouter path.
    relations = _graph_projection(
        graph_result,
        query=query,
        max_source_chars=max_source_chars,
        include_fact_history=include_fact_history,
    )
    return _packet_from_results(graph_relations=relations)


def _run_raptor_only(
    components: dict[str, Any],
    query: str,
    *,
    top_k: int,
    max_depth: int,
    max_children: int,
    max_source_chars: int,
    include_fact_history: bool,
    include_metadata: bool,
) -> dict[str, Any]:
    """RAPTOR-only: use ``RaptorSearcher.search`` and wrap into grouped packet.

    Summaries and cited_leaves populate their lanes; exact_hits and
    graph_relations are empty.
    """

    raptor_searcher = components["raptor_searcher"]
    if raptor_searcher is None:
        raise RuntimeError("provider raptor searcher is not initialized")
    raptor_result = raptor_searcher.search(
        query,
        top_k=top_k,
        max_depth=max_depth,
        max_children=max_children,
        max_source_chars=max_source_chars,
        include_fact_history=include_fact_history,
    )
    summaries = [
        s.to_dict(include_metadata=include_metadata)
        for s in (raptor_result.summaries or [])
    ]
    cited_leaves = [
        leaf.to_dict(include_metadata=include_metadata)
        for leaf in (raptor_result.cited_leaves or [])
    ]
    return _packet_from_results(summaries=summaries, cited_leaves=cited_leaves)


def _run_hybrid(
    components: dict[str, Any],
    query: str,
    *,
    mode: str,
    top_k: int,
    max_depth: int,
    max_children: int,
    max_source_chars: int,
    candidate_seed_top_k: int,
    max_graph_results: int,
    include_fact_history: bool,
    include_metadata: bool,
    with_graph: bool,
    with_raptor: bool,
) -> dict[str, Any]:
    """Hybrid (or hybrid-no-graph / hybrid-no-raptor): use ``HybridRouter``.

    ``with_graph=False`` passes ``graph_retriever=None``; same for
    ``with_raptor`` and ``raptor_searcher=None``.
    """

    from qdrant_memory.hybrid import HybridRouter

    graph_retriever = components["graph_retriever"] if with_graph else None
    raptor_searcher = components["raptor_searcher"] if with_raptor else None

    router = HybridRouter(
        qdrant=components["qdrant"],
        embeddings=components["embeddings"],
        collection_name=components["collection_name"],
        base_retriever=components["base_retriever"],
        graph_retriever=graph_retriever,
        raptor_searcher=raptor_searcher,
        scope=components["scope"],
    )
    result = router.retrieve(
        query,
        mode=mode,
        top_k=top_k,
        include_fact_history=include_fact_history,
        include_metadata=include_metadata,
        max_depth=max_depth,
        max_children=max_children,
        max_source_chars=max_source_chars,
        candidate_seed_top_k=candidate_seed_top_k,
        max_graph_results=max_graph_results,
    )
    # HybridRouteResult.to_dict() never echoes raw query text.
    return result.to_dict(include_metadata=include_metadata)


# --------------------------------------------------------------------------- #
# Dispatch table
# --------------------------------------------------------------------------- #


def _dispatch_variant(
    variant: str,
    components: dict[str, Any],
    query: str,
    *,
    mode: str,
    top_k: int,
    max_depth: int,
    max_children: int,
    max_source_chars: int,
    candidate_seed_top_k: int,
    max_graph_results: int,
    include_fact_history: bool,
    include_metadata: bool,
) -> dict[str, Any]:
    """Run a single variant and return its packet dict."""

    if variant == "dense-only":
        return _run_dense_only(
            components, query,
            top_k=top_k,
            max_source_chars=max_source_chars,
            include_fact_history=include_fact_history,
        )
    if variant == "dense+sparse":
        return _run_dense_sparse(
            components, query,
            top_k=top_k,
            max_source_chars=max_source_chars,
            include_fact_history=include_fact_history,
        )
    if variant == "graph":
        return _run_graph(
            components, query,
            top_k=top_k,
            candidate_seed_top_k=candidate_seed_top_k,
            max_graph_results=max_graph_results,
            max_depth=max_depth,
            max_source_chars=max_source_chars,
            include_fact_history=include_fact_history,
        )
    if variant == "raptor-only":
        return _run_raptor_only(
            components, query,
            top_k=top_k,
            max_depth=max_depth,
            max_children=max_children,
            max_source_chars=max_source_chars,
            include_fact_history=include_fact_history,
            include_metadata=include_metadata,
        )
    if variant == "hybrid":
        return _run_hybrid(
            components, query,
            mode=mode,
            top_k=top_k,
            max_depth=max_depth,
            max_children=max_children,
            max_source_chars=max_source_chars,
            candidate_seed_top_k=candidate_seed_top_k,
            max_graph_results=max_graph_results,
            include_fact_history=include_fact_history,
            include_metadata=include_metadata,
            with_graph=True,
            with_raptor=True,
        )
    if variant == "hybrid-no-graph":
        return _run_hybrid(
            components, query,
            mode=mode,
            top_k=top_k,
            max_depth=max_depth,
            max_children=max_children,
            max_source_chars=max_source_chars,
            candidate_seed_top_k=candidate_seed_top_k,
            max_graph_results=max_graph_results,
            include_fact_history=include_fact_history,
            include_metadata=include_metadata,
            with_graph=False,
            with_raptor=True,
        )
    if variant == "hybrid-no-raptor":
        return _run_hybrid(
            components, query,
            mode=mode,
            top_k=top_k,
            max_depth=max_depth,
            max_children=max_children,
            max_source_chars=max_source_chars,
            candidate_seed_top_k=candidate_seed_top_k,
            max_graph_results=max_graph_results,
            include_fact_history=include_fact_history,
            include_metadata=include_metadata,
            with_graph=True,
            with_raptor=False,
        )
    # Should be unreachable due to parse_variants validation.
    raise ValueError(f"unknown capture variant: {variant}")


# --------------------------------------------------------------------------- #
# Capture core
# --------------------------------------------------------------------------- #


def capture_eval_runs(
    provider: Any,
    cases: list[dict[str, Any]],
    variants: str | list[str] | None = None,
    *,
    top_k: int = 5,
    mode: str = "hybrid",
    max_depth: int = 2,
    max_children: int = 8,
    max_source_chars: int = 1200,
    candidate_seed_top_k: int = 20,
    max_graph_results: int = 20,
    include_fact_history: bool = False,
    include_metadata: bool = False,
) -> dict[str, Any]:
    """Capture read-only run packets for each ``(case, variant)`` pair.

    Parameters
    ----------
    provider
        An initialized Qdrant memory provider. The caller (CLI) owns
        construction. This function extracts read-only retrieval
        components and never mutates Qdrant.
    cases
        A list of validated Phase 6A eval case dicts (must each carry
        ``case_id`` and ``query``). Only ``case_id`` and ``query`` are
        read from each case; no other case field is serialized into
        run rows.
    variants
        Variant selector; see :func:`parse_variants`.
    top_k, mode, max_depth, max_children, max_source_chars, ...
        Retrieval parameters passed through to each variant runner.
    include_fact_history
        Whether to include deprecated/superseded history in retrieval.
    include_metadata
        Whether to include raw RAPTOR metadata in summaries/leaves.

    Returns
    -------
    dict
        A JSON-serializable dict with:

        * ``rows`` — list of run-row dicts (``case_id``, ``variant``,
          ``packet``, ``latency_ms``, ``capture``).
        * ``summary`` — compact aggregate (no raw queries, no raw
          packets): ``total_rows``, ``errored_rows``, ``variants``,
          ``cases``.

    Run rows NEVER carry raw query text. The ``capture`` metadata block
    contains only safe config labels and (on error) a redacted sentinel
    plus error kind.
    """

    variant_list = parse_variants(variants)
    components = _build_components(provider)
    base_capture_meta: dict[str, Any] = {
        "top_k": int(top_k),
        "mode": str(mode),
        "max_depth": int(max_depth),
        "max_children": int(max_children),
        "max_source_chars": int(max_source_chars),
        "candidate_seed_top_k": int(candidate_seed_top_k),
        "max_graph_results": int(max_graph_results),
        "include_fact_history": bool(include_fact_history),
        "include_metadata": bool(include_metadata),
        "collection_name": components["collection_name"],
    }

    rows: list[dict[str, Any]] = []
    errored_count = 0
    for case in cases:
        case_id = str(case.get("case_id") or "")
        query = str(case.get("query") or "")
        if not case_id or not query:
            continue
        for variant in variant_list:
            capture_meta = {**base_capture_meta, "variant": variant}
            start = time.perf_counter()
            try:
                packet = _dispatch_variant(
                    variant,
                    components,
                    query,
                    mode=str(mode),
                    top_k=int(top_k),
                    max_depth=int(max_depth),
                    max_children=int(max_children),
                    max_source_chars=int(max_source_chars),
                    candidate_seed_top_k=int(candidate_seed_top_k),
                    max_graph_results=int(max_graph_results),
                    include_fact_history=bool(include_fact_history),
                    include_metadata=bool(include_metadata),
                )
                latency_ms = round((time.perf_counter() - start) * 1000.0, 1)
                capture_meta["latency_ms"] = latency_ms
                rows.append({
                    "case_id": case_id,
                    "variant": variant,
                    "packet": packet,
                    "latency_ms": latency_ms,
                    "capture": _safe_capture_metadata(capture_meta),
                })
            except Exception as exc:  # noqa: BLE001 — sanitize all errors
                latency_ms = round((time.perf_counter() - start) * 1000.0, 1)
                capture_meta["latency_ms"] = latency_ms
                row = _sanitized_error_row(case_id, variant, exc, capture_meta)
                rows.append(row)
                errored_count += 1

    # Build a compact summary. No raw queries, no raw packets.
    variant_counts: dict[str, dict[str, int]] = {}
    for variant in variant_list:
        variant_counts[variant] = {"rows": 0, "errored": 0}
    for row in rows:
        variant = row.get("variant") or ""
        if variant in variant_counts:
            variant_counts[variant]["rows"] += 1
            if row.get("error"):
                variant_counts[variant]["errored"] += 1

    return {
        "rows": rows,
        "summary": {
            "total_rows": len(rows),
            "errored_rows": errored_count,
            "cases": len(cases),
            "variants": variant_counts,
        },
    }


# --------------------------------------------------------------------------- #
# JSONL writer
# --------------------------------------------------------------------------- #


def write_runs_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    """Write capture run rows to a JSONL file.

    Each row is serialized as one compact JSON line. The file is
    opened with UTF-8 encoding. Existing files are overwritten.
    """

    import json

    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")


__all__ = [
    "DEFAULT_CAPTURE_VARIANTS",
    "parse_variants",
    "capture_eval_runs",
    "write_runs_jsonl",
]
