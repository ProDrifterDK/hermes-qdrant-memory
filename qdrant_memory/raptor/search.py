"""Phase 5: read-only RAPTOR search / zoom lane.

This module is **purely read-only**. It never calls ``upsert``,
``delete_ids``, ``delete_filter``, ``update_payload``, ``scroll_by_filter``,
or any other mutating Qdrant method. All I/O is restricted to:

* ``MemoryRetriever.search(..., update_access=False)`` for dense+sparse seeds.
* ``QdrantClient.retrieve(collection_name, ids, with_payload=True)`` for
  fetching explicit child/parent/root RAPTOR node payloads. ``retrieve()``
  does not accept a filter, so every payload returned by it MUST be
  post-filtered against the provider scope before it can be promoted into a
  RAPTOR result.

Public dataclasses:

* :class:`RaptorSummaryHit` — a parent summary match retrieved by dense/sparse
  search.
* :class:`RaptorLeafHit` — a child leaf promoted alongside its parent. The
  raw text is redacted if the payload carries secret-shaped content.
* :class:`RaptorSearchResult` — the read-only result of one RAPTOR
  search/zoom pass, organized into summaries / cited_leaves / a debug dict.

Safety invariants enforced here:

1. No Qdrant mutation. The only methods called on the client are
   ``search`` (via MemoryRetriever) and ``retrieve``.
2. ``retrieve()`` results are defensively post-filtered against the scope
   (``profile_id`` / ``user_id_hash`` / ``chat_id_hash``) so a payload from
   a different scope cannot leak.
3. Every child leaf goes through :func:`raptor.assess_leaf_safety` so the
   unsafe markers (``fact_status in {stale, deprecated, superseded,
   disputed, review_required}``, ``consolidation_quarantined``,
   ``requires_review``, ``stale``, ``raptor_excluded``, ``raptor_forgotten``,
   secrets) demote the leaf to a warning-only entry rather than a promoted
   result.
4. Budgets: ``max_depth``, ``max_children``, ``max_source_chars``, and the
   retrieved leaf text length are all clamped to hard caps.
5. Stable point IDs are used as dedupe keys; payload dicts carry
   ``source_uri`` / ``locator`` / ``content_hash`` so callers can trace the
   underlying memory point without re-running dense retrieval.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping

from qdrant_memory.lesson_extractor import contains_secret
from qdrant_memory.raptor.apply import (
    _UNSAFE_LEAF_STATUSES,
    assess_leaf_safety,
    assess_parent_status,
)
from qdrant_memory.raptor.builder import _safe_handle_for_point_id
from qdrant_memory.raptor.schema import RAPTOR_DERIVATION_TYPE


# ---------------------------------------------------------------------------
# Hard caps (non-negotiable)
# ---------------------------------------------------------------------------

HARD_MAX_DEPTH: int = 3
HARD_MAX_CHILDREN: int = 16
HARD_MAX_SOURCE_CHARS: int = 2400
HARD_CONTEXT_CHAR_BUDGET: int = 16000
HARD_SEED_TOP_K: int = 32


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class RaptorSummaryHit:
    """A parent RAPTOR summary surfaced by the search/zoom lane."""

    point_id: str
    raptor_node_id: str
    raptor_root_id: str
    raptor_level: int
    raptor_tree_id: str
    raptor_build_id: str
    raptor_cluster_id: str
    raptor_child_ids: list[str] = field(default_factory=list)
    raptor_parent_ids: list[str] = field(default_factory=list)
    raptor_summary_of: list[str] = field(default_factory=list)
    text: str = ""
    source_hashes: list[str] = field(default_factory=list)
    derived_from: list[dict[str, Any]] = field(default_factory=list)
    parent_status: str = "active"
    parent_assessment: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def _metadata_secret_free(self) -> bool:
        """Return True iff derived_from / extra / parent_assessment are safe.

        ``include_metadata=true`` exposes raw ``derived_from``,
        ``parent_assessment``, and ``extra`` payloads to the caller. If
        any of those carry a secret-shaped value (credential URI,
        bearer token, etc.) we must fail closed and refuse to expose the
        raw metadata — even when the caller explicitly asked for it.
        """

        import json as _json

        for field_name in ("derived_from", "parent_assessment", "extra"):
            value = getattr(self, field_name, None)
            if value in (None, "", [], {}):
                continue
            try:
                blob = _json.dumps(value, sort_keys=True, default=str)
            except Exception:
                blob = str(value)
            if contains_secret(blob):
                return False
        return True

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "point_id": self.point_id,
            "raptor_node_id": self.raptor_node_id,
            "raptor_root_id": self.raptor_root_id,
            "raptor_level": self.raptor_level,
            "raptor_tree_id": self.raptor_tree_id,
            "raptor_build_id": self.raptor_build_id,
            "raptor_cluster_id": self.raptor_cluster_id,
            "raptor_child_ids": list(self.raptor_child_ids),
            "raptor_parent_ids": list(self.raptor_parent_ids),
            "raptor_summary_of": list(self.raptor_summary_of),
            "text": self.text,
            "source_hashes": list(self.source_hashes),
            "parent_status": self.parent_status,
        }
        if include_metadata:
            if self._metadata_secret_free():
                out["derived_from"] = list(self.derived_from)
                out["extra"] = dict(self.extra)
                out["parent_assessment"] = dict(self.parent_assessment)
            else:
                # Fail closed: surface a sentinel so callers can detect
                # that metadata was withheld, without leaking the raw
                # secret-bearing values.
                out["derived_from"] = []
                out["extra"] = {}
                out["parent_assessment"] = {
                    "parent_status": self.parent_status,
                    "metadata_redacted": True,
                    "reason": "secret_bearing_metadata",
                }
        return out


@dataclass
class RaptorLeafHit:
    """A child leaf memory point promoted next to its parent summary."""

    point_id: str
    parent_raptor_node_id: str
    parent_point_id: str
    text: str = ""
    source_uri: str = ""
    file_path: str = ""
    heading: str = ""
    content_hash: str = ""
    source_type: str = ""
    locator: dict[str, Any] = field(default_factory=dict)
    safety: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def _metadata_secret_free(self) -> bool:
        """Return True iff ``locator`` / ``extra`` are safe to emit.

        ``include_metadata=true`` exposes raw ``locator`` and ``extra``
        payloads to the caller. If any of those carry a secret-shaped
        value (credential URI, bearer token, etc.) we must fail closed
        and refuse to expose the raw metadata — even when the caller
        explicitly asked for it.
        """

        import json as _json

        for field_name in ("locator", "extra"):
            value = getattr(self, field_name, None)
            if value in (None, "", [], {}):
                continue
            try:
                blob = _json.dumps(value, sort_keys=True, default=str)
            except Exception:
                blob = str(value)
            if contains_secret(blob):
                return False
        return True

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "point_id": self.point_id,
            "parent_raptor_node_id": self.parent_raptor_node_id,
            "parent_point_id": self.parent_point_id,
            "source_uri": self.source_uri,
            "file_path": self.file_path,
            "heading": self.heading,
            "source_type": self.source_type,
            "content_hash": self.content_hash,
        }
        if include_metadata:
            if self._metadata_secret_free():
                out["text"] = self.text
                out["locator"] = dict(self.locator)
                out["safety"] = dict(self.safety)
                out["extra"] = dict(self.extra)
            else:
                # Fail closed: do NOT leak the raw secret-bearing values.
                # We still surface ``text`` and ``safety`` if those are
                # safe so the caller can see *why* the leaf is being
                # surfaced, but only after redaction.
                out["text"] = self.text
                out["locator"] = {}
                out["safety"] = dict(self.safety)
                out["extra"] = {"metadata_redacted": True, "reason": "secret_bearing_metadata"}
        return out


@dataclass
class RaptorSearchResult:
    """Read-only result of a RAPTOR search/zoom pass."""

    query: str
    summaries: list[RaptorSummaryHit] = field(default_factory=list)
    cited_leaves: list[RaptorLeafHit] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)
    unsafe_summary_ids: set[str] = field(default_factory=set)
    unsafe_leaf_ids: set[str] = field(default_factory=set)

    def to_dict(self, *, include_metadata: bool = False) -> dict[str, Any]:
        # Phase 5 fix12 (final10 P3 finding): never echo the raw
        # ``self.query`` into the JSON envelope. ``HybridRouteResult``
        # already projects a safe metadata block; we mirror it here
        # via the local :func:`_redact_query_metadata` helper so
        # the standalone RAPTOR public API cannot leak a secret
        # the user accidentally pasted in as the query. We do NOT
        # import the hybrid router's helper to avoid a circular
        # import.
        safe_query = _redact_query_metadata(self.query)
        return {
            "query_length": safe_query["query_length"],
            "query_digest": safe_query["query_digest"],
            "query_redacted": safe_query["query_redacted"],
            "summaries": [s.to_dict(include_metadata=include_metadata) for s in self.summaries],
            "cited_leaves": [l.to_dict(include_metadata=include_metadata) for l in self.cited_leaves],
            "warnings": list(self.warnings),
            "debug": dict(self.debug),
            # IDs may be secret-shaped; always emit redacted handles so
            # the JSON envelope cannot be used to leak raw point IDs.
            "unsafe_summary_ids": sorted(_safe_handle(s) for s in self.unsafe_summary_ids),
            "unsafe_leaf_ids": sorted(_safe_handle(s) for s in self.unsafe_leaf_ids),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SCOPE_KEYS: tuple[str, ...] = ("profile_id", "user_id_hash", "chat_id_hash")


def _clamp_int(value: int | None, default: int, lo: int, hi: int) -> int:
    try:
        candidate = int(value) if value is not None else default
    except Exception:
        candidate = default
    return max(lo, min(int(candidate), hi))


def _payload_scope(payload: Mapping[str, Any] | None) -> dict[str, str]:
    """Pull scope fields from a payload in a defensive way."""
    if not isinstance(payload, Mapping):
        return {}
    return {
        key: str(payload.get(key) or "")
        for key in _SCOPE_KEYS
    }


def _payload_matches_scope(
    payload: Mapping[str, Any] | None,
    scope: Mapping[str, str] | None,
) -> bool:
    """Return True iff the payload matches the requested scope.

    Fail-closed: any non-empty scope key must match the payload value exactly.
    A payload that lacks a scope key when the scope expects one is rejected.
    Empty/missing scope keys are not enforced.
    """
    if not scope:
        return True
    payload_scope = _payload_scope(payload)
    for key in _SCOPE_KEYS:
        expected = str(scope.get(key) or "")
        if not expected:
            continue
        if str(payload_scope.get(key) or "") != expected:
            return False
    return True


def _safe_handle(point_id: str) -> str:
    """Return a deterministic, secret-free handle for *point_id*.

    Used in :data:`RaptorSearchResult.warnings` so secret-shaped handles never
    leak through error messages. Falls back to the builder's
    :func:`_safe_handle_for_point_id`.
    """
    try:
        return _safe_handle_for_point_id(point_id)
    except Exception:
        return ""


def _redact_leaf_text(text: str) -> str:
    text = str(text or "")
    if contains_secret(text):
        return "[redacted: possible secret-bearing memory]"
    return text


def _truncate(text: str, max_chars: int) -> str:
    text = str(text or "")
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _is_raptor_parent_payload(payload: Mapping[str, Any]) -> bool:
    """Return True iff *payload* is a RAPTOR parent summary (level >= 1)."""
    if not isinstance(payload, Mapping):
        return False
    if payload.get("derivation_type") != RAPTOR_DERIVATION_TYPE:
        return False
    try:
        level = int(payload.get("raptor_level") or 0)
    except Exception:
        level = 0
    return level >= 1


# Default-emitted core fields on a RAPTOR summary. These are the
# fields the public :meth:`RaptorSummaryHit.to_dict` always serializes,
# even when ``include_metadata=false``. Every value in this tuple is
# covered by the secret-bearing pre-promotion check so the dense seed
# path cannot smuggle raw credential-shaped IDs or provenance hashes
# into the warning / JSON envelope.
_DEFAULT_EMITTED_SUMMARY_FIELDS: tuple[str, ...] = (
    "point_id",
    "raptor_node_id",
    "raptor_root_id",
    "raptor_tree_id",
    "raptor_build_id",
    "raptor_cluster_id",
    "raptor_child_ids",
    "raptor_parent_ids",
    "raptor_summary_of",
    "source_hashes",
)


def _summary_default_emitted_secret_bearing(
    point: Mapping[str, Any],
    payload: Mapping[str, Any],
) -> bool:
    """Return True iff any default-emitted summary field carries a secret.

    The default (:meth:`RaptorSummaryHit.to_dict`) always emits
    ``point_id``, ``raptor_node_id``, ``raptor_root_id``,
    ``raptor_tree_id``, ``raptor_build_id``, ``raptor_cluster_id``,
    ``raptor_child_ids``, ``raptor_parent_ids``, ``raptor_summary_of``,
    and ``source_hashes``. We project those values exactly as they
    would land on the wire and run :func:`contains_secret` over a
    stable JSON blob. ``source_hashes`` entries like ``"a"*64`` (hex
    digests) are not considered secret-bearing; only values that match
    one of the secret-scanner regexes fire.
    """

    if not isinstance(point, Mapping):
        point = {}
    if not isinstance(payload, Mapping):
        payload = {}

    projection: dict[str, Any] = {
        "point_id": str(point.get("id") or ""),
        "raptor_node_id": str(payload.get("raptor_node_id") or ""),
        "raptor_root_id": str(payload.get("raptor_root_id") or ""),
        "raptor_tree_id": str(payload.get("raptor_tree_id") or ""),
        "raptor_build_id": str(payload.get("raptor_build_id") or ""),
        "raptor_cluster_id": str(payload.get("raptor_cluster_id") or ""),
        "raptor_child_ids": _normalize_id_list(payload.get("raptor_child_ids")),
        "raptor_parent_ids": _normalize_id_list(payload.get("raptor_parent_ids")),
        "raptor_summary_of": _normalize_id_list(payload.get("raptor_summary_of")),
        "source_hashes": [
            str(item) for item in (payload.get("source_hashes") or []) if str(item)
        ],
    }
    try:
        import json as _json

        blob = _json.dumps(projection, sort_keys=True, default=str)
    except Exception:
        blob = " ".join(str(v) for v in projection.values())
    return contains_secret(blob)


def _is_raptor_leaf_ref_payload(payload: Mapping[str, Any]) -> bool:
    """Return True iff *payload* is a RAPTOR leaf-ref node (level == 0)."""
    if not isinstance(payload, Mapping):
        return False
    if payload.get("derivation_type") != RAPTOR_DERIVATION_TYPE:
        return False
    try:
        level = int(payload.get("raptor_level") or 0)
    except Exception:
        return False
    return level == 0


# -------------------------------------------------------------------
# Phase 5 fix12 (final10 P2 finding): parent trust gate.
#
# RAPTOR parent payloads carry trust flags at construction time
# (``canonical=False``, ``requires_review=True``,
# ``raptor_review_status="review_required"``). The downstream
# :func:`assess_parent_status` recomputation only inspects *child*
# safety/missing-child accounting; if the children are clean, a
# parent that itself carries ``requires_review=True``,
# ``raptor_review_status="review_required"``, ``stale=True``,
# ``consolidation_quarantined=True``, ``raptor_excluded=True``,
# ``raptor_forgotten=True``, or an unsafe ``fact_status`` could
# otherwise keep its raw text and remain ``active`` even though
# the parent payload explicitly marks it as non-active. This gate
# runs over the *parent* payload (mirror of
# :func:`_dense_payload_unsafe_for_active_context` in the hybrid
# router) so a trust-flagged parent never reaches the caller as
# active context. The reuses :data:`_UNSAFE_LEAF_STATUSES` so both
# the leaf and parent gates speak the same vocabulary.
# -------------------------------------------------------------------
_PARENT_TRUST_FLAG_KEYS: tuple[str, ...] = (
    "requires_review",
    "stale",
    "consolidation_quarantined",
    "raptor_excluded",
    "raptor_forgotten",
)
# Status markers the parent trust gate treats as non-active when
# the parent payload carries them in :attr:`fact_status`. Reuses
# :data:`_UNSAFE_LEAF_STATUSES` so leaf/parent stay aligned. The
# gate additionally fires on
# ``raptor_review_status == "review_required"`` (the canonical
# Phase 3 marker emitted by :mod:`qdrant_memory.raptor.schema`).
_PARENT_REVIEW_STATUS_VALUES: frozenset[str] = frozenset({"review_required"})


def _parent_trust_gate_reasons(payload: Mapping[str, Any] | None) -> list[str]:
    """Return a list of parent-payload trust reasons (empty == safe).

    Mirrors the dense-payload vocabulary used by the hybrid router
    so both lanes agree on what makes a node non-active. A non-empty
    reasons list means the parent payload itself carries a marker
    that should demote it regardless of child safety.
    """

    if not isinstance(payload, Mapping):
        return ["payload_not_dict"]

    reasons: list[str] = []
    for key in _PARENT_TRUST_FLAG_KEYS:
        try:
            if payload.get(key) is True:
                reasons.append(key)
        except Exception:
            continue
    fact_status = str(payload.get("fact_status") or "").strip().lower()
    if fact_status and fact_status in _UNSAFE_LEAF_STATUSES:
        reasons.append(f"fact_status:{fact_status}")
    review_status = str(payload.get("raptor_review_status") or "").strip().lower()
    if review_status and review_status in _PARENT_REVIEW_STATUS_VALUES:
        reasons.append(f"raptor_review_status:{review_status}")
    return reasons


def _parent_payload_unsafe_for_active_context(payload: Mapping[str, Any] | None) -> bool:
    """Return True iff the parent payload itself is non-active.

    Phase 5 fix12 (final10 P2): the parent trust gate must run
    BOTH at promotion time (so a parent that already carries
    ``requires_review=True`` / ``raptor_review_status=review_required``
    never lands in the active ``summaries`` list) AND during the
    post-demotion recompute (defense-in-depth: a payload that
    changes between dense-seed retrieval and child walk must still
    be honored). A non-empty ``_parent_trust_gate_reasons`` output
    is sufficient to demote the parent; we deliberately do NOT
    require ``canonical=False`` here because canonical flipped
    parents are still safe to surface — only the unsafe-status
    markers should drive the demotion.
    """

    return bool(_parent_trust_gate_reasons(payload))


# Local query-redaction helper for ``RaptorSearchResult``. Mirrors
# :func:`qdrant_memory.hybrid.router._redact_query_metadata` so the
# top-level ``qdrant_memory_retrieve`` envelope and the standalone
# RAPTOR envelope stay aligned. We deliberately re-implement the
# helper here (instead of importing from ``qdrant_memory.hybrid``)
# to avoid a circular import: ``hybrid.router`` already imports
# :class:`RaptorSearchResult` from this module.
_PARENT_QUERY_REDACTED_SENTINEL = "[redacted: query omitted from retrieve output]"


def _redact_query_metadata(query: Any) -> dict[str, Any]:
    """Return the safe query metadata block for RAPTOR search output.

    The raw query is NEVER echoed. We project
    ``query_length``, ``query_digest`` (sha256[:16]), and
    ``query_redacted`` (fixed sentinel) so a secret-shaped query
    can never leak through ``RaptorSearchResult.to_dict``.
    """

    import hashlib

    raw = str(query or "")
    length = len(raw)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return {
        "query_length": length,
        "query_digest": digest,
        "query_redacted": _PARENT_QUERY_REDACTED_SENTINEL,
    }


def _stringify_node_id(value: Any) -> str:
    return str(value or "").strip()


def _normalize_id_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in value:
        sid = _stringify_node_id(item)
        if sid and sid not in seen:
            seen.add(sid)
            out.append(sid)
    return out


def _normalize_leaf_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Strip the leaf down to a compact redacted projection."""
    if not isinstance(payload, Mapping):
        return {}
    return {
        "text": str(payload.get("text") or payload.get("lesson") or ""),
        "source_uri": str(payload.get("source_uri") or ""),
        "file_path": str(payload.get("file_path") or ""),
        "heading": str(payload.get("heading") or ""),
        "content_hash": str(payload.get("content_hash") or ""),
        "source_type": str(payload.get("source_type") or ""),
        "locator": payload.get("locator") if isinstance(payload.get("locator"), Mapping) else {},
        "profile_id": str(payload.get("profile_id") or ""),
        "user_id_hash": str(payload.get("user_id_hash") or ""),
        "chat_id_hash": str(payload.get("chat_id_hash") or ""),
    }


def _leaf_payload_visible(payload: Mapping[str, Any] | None) -> bool:
    """Defensive: refuse to project a payload that is secret-bearing."""
    if not isinstance(payload, Mapping):
        return False
    text = str(payload.get("text") or payload.get("lesson") or "")
    if contains_secret(text):
        return False
    try:
        import json as _json  # local import keeps top of module lean

        if contains_secret(_json.dumps(payload, sort_keys=True, default=str)):
            return False
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# RaptorSearcher
# ---------------------------------------------------------------------------


@dataclass
class RaptorSearcher:
    """Read-only RAPTOR search / zoom helper.

    Wraps an existing :class:`MemoryRetriever` for dense+sparse seeding and
    explicitly addresses RAPTOR parent/child IDs with ``retrieve()`` (with
    defensive scope post-filtering). It never calls ``upsert``, ``delete_*``,
    ``update_payload``, or ``scroll_by_filter`` directly.
    """

    qdrant: Any
    retriever: Any  # MemoryRetriever-like; must accept ``update_access=False``
    collection_name: str
    scope: dict[str, str] = field(default_factory=dict)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        max_depth: int = 2,
        max_children: int = 8,
        max_source_chars: int = 1200,
        include_fact_history: bool = False,
        source_type: Any = None,
        tags: list[str] | None = None,
        source: str | None = None,
        file_path: str | None = None,
        project_path: str | None = None,
        since: str | None = None,
        until: str | None = None,
    ) -> RaptorSearchResult:
        """Execute a read-only RAPTOR search/zoom pass.

        Phases:

        1. Run dense+sparse search via :attr:`retriever.search` (always
           ``update_access=False`` so retrieval is non-mutating).
        2. For every seed that carries ``raptor_node_id`` info, walk
           ancestors (parents → root) and descendants (``raptor_child_ids``
           and ``raptor_summary_of``) up to *max_depth* hops, bounded to
           ``max_children``.
        3. Project each candidate parent into :class:`RaptorSummaryHit`.
        4. Project child leaves into :class:`RaptorLeafHit` after running
           :func:`assess_leaf_safety`; unsafe leaves demoted to warnings.
        5. Dedupe by point ID and clamp the final result to ``top_k``
           summaries plus a hard cap on total context characters
           (``max_source_chars`` × ``top_k``).
        """
        safe_top_k = _clamp_int(top_k, 5, 1, 20)
        safe_max_depth = _clamp_int(max_depth, 2, 1, HARD_MAX_DEPTH)
        safe_max_children = _clamp_int(max_children, 8, 1, HARD_MAX_CHILDREN)
        safe_max_source_chars = _clamp_int(
            max_source_chars, 1200, 1, HARD_MAX_SOURCE_CHARS
        )

        warnings: list[str] = []
        debug: dict[str, Any] = {
            "mode": "raptor_search",
            "top_k": safe_top_k,
            "max_depth": safe_max_depth,
            "max_children": safe_max_children,
            "max_source_chars": safe_max_source_chars,
            "stages": {},
        }

        # Stage 1: dense+sparse seed search.
        #
        # Phase 5 fix5: pass ``allow_sparse_scroll=False`` so the RAPTOR
        # seed search never invokes ``scroll_by_filter`` through the
        # ``MemoryRetriever.search`` path. The hybrid retrieve
        # contract forbids any scroll-by-filter call (read-only
        # invariant), and strong-signal retrieve queries would
        # otherwise trigger the sparse lane because
        # ``MemoryRetriever.search`` defaults ``allow_sparse_scroll``
        # to ``True``. We also keep ``update_access=False`` so RAPTOR
        # seed search never bumps access metadata. ``MemoryRetriever``
        # itself accepts the kwarg; a ``TypeError`` here means the
        # retriever does NOT enforce the read-only invariant, so we
        # fail closed (empty seeds + warning) rather than silently
        # falling back to the legacy behaviour.
        seed_call_top_k = max(safe_top_k, min(HARD_SEED_TOP_K, safe_top_k * 4))
        search_kwargs: dict[str, Any] = {
            "source_type": source_type,
            "tags": tags,
            "source": source,
            "file_path": file_path,
            "project_path": project_path,
            "since": since,
            "until": until,
            "include_fact_history": include_fact_history,
            "update_access": False,
        }
        try:
            seeds = self.retriever.search(
                query,
                top_k=seed_call_top_k,
                allow_sparse_scroll=False,
                **search_kwargs,
            )
        except TypeError:
            # The retriever does not accept ``allow_sparse_scroll``.
            # We refuse to silently fall back: the read-only
            # invariant requires that scroll_by_filter is never
            # invoked from the RAPTOR seed search path, so a missing
            # kwarg means the caller wired a non-conforming
            # retriever. Surface a sanitized warning and fail closed
            # (empty seeds) so the rest of the read-only retrieval
            # can still proceed without violating the contract.
            #
            # Phase 5 fix7: do NOT interpolate ``{exc}``. The
            # exception's ``__str__`` can echo the requested query
            # (which may be secret-shaped) or other raw backend
            # strings into the JSON envelope via HybridRouter.
            # Operators correlate via server-side debug logs (the
            # ``debug.stages.seed_search`` entry below carries a
            # stable ``error`` code without leaking the exception).
            warnings.append(
                "raptor seed search: retriever does not accept "
                "allow_sparse_scroll=False; failing closed "
                "(no raw exception leaked; see server logs)"
            )
            debug["stages"].setdefault("seed_search", {})["error"] = "type_error"
            seeds = []
        except Exception:
            # Phase 5 fix7: emit a sanitized warning, never the raw
            # ``{exc}``. The dense+sparse retriever backend can echo
            # the query string (which may carry a secret-shaped
            # token) or other raw backend text into the exception
            # ``__str__``; surfacing that into ``warnings`` would let
            # a secret reach the JSON envelope downstream through
            # ``qdrant_memory_retrieve``. A stable error code is
            # recorded in ``debug.stages.seed_search.error`` so
            # operators can correlate via server-side debug logs.
            warnings.append(
                "raptor seed search failed "
                "(no raw exception leaked; see server logs)"
            )
            debug["stages"].setdefault("seed_search", {})["error"] = "exception"
            seeds = []

        # Preserve any ``error`` code set by a sanitized ``except`` arm
        # above (Phase 5 fix7) so operators can correlate via debug
        # without the raw exception text leaking into the warning
        # channel.
        _seed_existing = debug["stages"].get("seed_search") or {}
        _seed_error = _seed_existing.get("error") if isinstance(_seed_existing, Mapping) else None
        debug["stages"]["seed_search"] = {
            "requested": seed_call_top_k,
            "returned": len(seeds),
        }
        if _seed_error:
            debug["stages"]["seed_search"]["error"] = _seed_error

        # Stage 2: collect candidate node IDs from seeds.
        parent_node_ids: list[str] = []
        seen_node_ids: set[str] = set()

        for seed in seeds:
            payload = getattr(seed, "payload", None) or {}
            node_id = _stringify_node_id(payload.get("raptor_node_id"))
            if node_id and node_id not in seen_node_ids:
                seen_node_ids.add(node_id)
                parent_node_ids.append(node_id)

        # Stage 3: walk ancestors up to safe_max_depth hops and dereference
        # children IDs bounded by safe_max_children. We deliberately call
        # ``retrieve`` only — no scroll, no upsert.
        parent_queue: list[tuple[str, int]] = [
            (nid, 0) for nid in parent_node_ids
        ]
        summaries: list[RaptorSummaryHit] = []
        summary_by_node_id: dict[str, RaptorSummaryHit] = {}
        all_leaf_ids: list[str] = []
        seen_leaf_ids: set[str] = set()
        unsafe_summary_ids: set[str] = set()
        retrieved_node_ids: set[str] = set()
        # Phase 5 fix8 (final6 finding #2): ``parents_for_leaf``
        # tracks EVERY parent that references a given child id, not
        # just the first one (``setdefault``-wins attribution that
        # fix7 used). When a leaf is unsafe / missing, every parent
        # in this list must absorb that outcome so a parent whose
        # children are silently shared with another parent cannot
        # remain ``active`` while its evidence is demoted. The
        # retrieval-pass dedupe (which fetches each unique leaf
        # exactly once) is still controlled by ``seen_leaf_ids``;
        # ``parents_for_leaf`` is purely a per-parent attribution
        # map used during safety accounting.
        parents_for_leaf: dict[str, list[RaptorSummaryHit]] = {}
        # Per-parent **referenced** child set (deduped across
        # ``raptor_child_ids`` + ``raptor_summary_of``). The
        # post-demotion recomputation subtracts the actually-retrieved
        # child population to derive the *missing* child count and
        # treats missing children as unsafe so a parent whose leaves
        # were silently dropped by the backend
        # (missing/deleted/scope-filtered) cannot remain ``active``.
        # We keep a SET (not just a count) so per-parent referenced
        # children survive shared-child deduplication: a parent that
        # shares one leaf with another parent still counts that
        # shared leaf as referenced, so its missing/safe accounting
        # is computed against its own referenced set, not against a
        # global dedupe set. This is the core of the fix8
        # finding-#2 contract.
        per_parent_referenced_children: dict[int, set[str]] = {}

        # Bidirectional travel: forward (parents) and up (parents of parents).
        iterations = 0
        while parent_queue and len(summaries) < safe_top_k and iterations < HARD_MAX_DEPTH * 2:
            iterations += 1
            batch_ids: list[str] = []
            depths: dict[str, int] = {}
            for nid, depth in parent_queue:
                if nid in retrieved_node_ids:
                    continue
                if nid not in depths:
                    depths[nid] = depth
                if nid not in batch_ids:
                    batch_ids.append(nid)
                if len(batch_ids) >= safe_max_children:
                    break
            parent_queue = parent_queue[safe_max_children:]
            if not batch_ids:
                # Avoid infinite loops: drop everything in the queue once
                # we've exhausted a batch.
                parent_queue = []
                continue
            retrieved_node_ids.update(batch_ids)
            retrieved = self._safe_retrieve(batch_ids, warnings)
            for point in retrieved:
                payload = point.get("payload") if isinstance(point, Mapping) else None
                if not _is_raptor_parent_payload(payload or {}):
                    continue
                node_id = _stringify_node_id(payload.get("raptor_node_id"))
                if not node_id or node_id in summary_by_node_id:
                    continue
                text = str(payload.get("text") or "")
                # Pre-promotion secret-bearing check over the exact
                # default-emitted core fields (point_id, node/root/
                # tree/build/cluster IDs, child/parent/summary_of IDs,
                # source_hashes). Any one of those carrying a secret
                # must NOT reach ``summaries`` or the warning channel,
                # because the raw credential would otherwise be
                # echoed through ``point_id`` /
                # ``raptor_node_id`` / ``source_hashes`` even with
                # ``include_metadata=false``.
                if _summary_default_emitted_secret_bearing(point, payload or {}):
                    unsafe_summary_ids.add(node_id)
                    warnings.append(
                        "raptor summary skipped: secret-bearing core field "
                        f"(handle={_safe_handle(node_id)})"
                    )
                    continue
                if contains_secret(text):
                    unsafe_summary_ids.add(node_id)
                    warnings.append(
                        f"raptor summary skipped: secret-bearing text "
                        f"(handle={_safe_handle(node_id)})"
                    )
                    continue

                child_ids = _normalize_id_list(payload.get("raptor_child_ids"))
                parent_ids = _normalize_id_list(payload.get("raptor_parent_ids"))
                summary_of = _normalize_id_list(payload.get("raptor_summary_of"))
                source_hashes = [
                    str(item) for item in (payload.get("source_hashes") or []) if str(item)
                ]
                derived_from: list[dict[str, Any]] = []
                for edge in payload.get("derived_from") or []:
                    if isinstance(edge, Mapping):
                        derived_from.append(dict(edge))

                # Conservative parent assessment: no actual child payloads yet.
                # The default ``parent_status`` is ``"active"``; the
                # no-child-guard (Phase 5 fix15) overwrites both
                # ``parent_status`` and ``parent_assessment`` to fail
                # closed BEFORE the post-demotion recompute ever sees
                # the hit, so a childless RAPTOR parent cannot be
                # re-promoted to ``"active"`` later.
                parent_assessment = {
                    "parent_status": "active",
                    "unsafe_children": [],
                    "safe_children_count": 0,
                    "total_children": 0,
                }
                summary_status = "active"
                # Phase 5 fix15 (final12 P2 finding): a RAPTOR
                # parent with no source-backed child refs (both
                # ``raptor_child_ids`` AND ``raptor_summary_of``
                # empty) must fail closed by default. The previous
                # implementation set ``summary_status = "excluded"``
                # and emitted the downgrade warning, but did NOT
                # record a marker on ``parent_assessment``; the
                # post-demotion recompute then called
                # ``assess_parent_status([])`` (because no children
                # were ever enqueued), which returns ``"active"``
                # and silently re-promoted the childless parent to
                # active context. We now (1) build the assessment
                # block with ``parent_status == "excluded"`` and a
                # bounded ``missing_child_reasons`` marker so the
                # post-demotion recompute can replay the demotion
                # (mirroring how ``trust_gate_reasons`` replays
                # the parent trust gate), and (2) keep the existing
                # redacted warning so operators can correlate
                # via the audit envelope. The marker is bounded
                # to a short token vocabulary (``"no_child_refs"``)
                # so the recompute JSON envelope cannot echo raw
                # payload content either.
                has_no_child_refs = (
                    not child_ids and not summary_of
                )
                if has_no_child_refs:
                    summary_status = "excluded"
                    parent_assessment = {
                        "parent_status": "excluded",
                        "unsafe_children": [],
                        "safe_children_count": 0,
                        "total_children": 0,
                        "missing_child_reasons": ["no_child_refs"],
                        "no_child_refs": True,
                    }
                    warnings.append(
                        "raptor summary has no child IDs; downgraded "
                        f"(handle={_safe_handle(node_id)})"
                    )

                # Phase 5 fix12 (final10 P2 finding): the parent
                # trust gate runs BEFORE the hit is appended to
                # ``summaries``. A parent payload that already
                # carries an unsafe-status marker
                # (``requires_review=True``,
                # ``raptor_review_status="review_required"``,
                # ``stale=True``, ``consolidation_quarantined``,
                # ``raptor_excluded``, ``raptor_forgotten``, or an
                # unsafe ``fact_status`` value) is demoted here so
                # its raw text never reaches the caller as active
                # context. The post-demotion recompute loop below
                # applies the same gate again as defense-in-depth
                # in case the payload was updated between dense
                # seed retrieval and child walk.
                #
                # Phase 5 fix13 (final11 P2 regression): the
                # trust-gated hit MUST still flow through the same
                # cap-bounded child enqueue / accounting block as an
                # active parent. The parent's *summary* text is
                # unsafe-by-trust, but its referenced child leaves
                # are independently source-backed evidence: a clean
                # child must remain retrievable / citable for
                # evidence-mode traces and downstream prompts. The
                # demoted parent therefore falls through to the
                # unified child-enqueue / parent-ascent blocks
                # below — its `text` stays `""`, its
                # `parent_status` stays non-active, and
                # ``trust_gate_reasons`` is carried in
                # ``parent_assessment`` so the post-demotion
                # recompute forces ``review_required`` even when
                # every child is clean.
                trust_reasons = _parent_trust_gate_reasons(payload or {})
                if trust_reasons:
                    # Phase 5 fix16 (final13 P2 finding): when a
                    # parent carries BOTH the no-source-evidence
                    # marker (captured above by the ``has_no_child_refs``
                    # branch) AND a trust flag, childless/excluded
                    # must win over generic review_required. The
                    # dataclass-level ``parent_status`` is forced to
                    # ``"excluded"`` so a production-shaped parent
                    # that is review-required-by-default AND has no
                    # source-backed children reports the stricter
                    # no-evidence state. The no-child marker is
                    # preserved on the trust-gated
                    # ``parent_assessment`` block (alongside the
                    # trust reasons) so the post-demotion replay
                    # can demote to ``excluded`` again even if
                    # ``assess_parent_status`` would otherwise
                    # return ``"active"`` on an empty surviving
                    # child set.
                    no_child_marker = (
                        parent_assessment.get("no_child_refs") is True
                    ) if isinstance(parent_assessment, Mapping) else False
                    no_child_reasons_marker = (
                        parent_assessment.get("missing_child_reasons")
                        if isinstance(parent_assessment, Mapping) else None
                    )
                    if no_child_marker:
                        # Childless/no-evidence wins: the parent has
                        # nothing to cite even if the trust gate
                        # were cleared.
                        summary_status = "excluded"
                    else:
                        summary_status = "review_required"
                    unsafe_summary_ids.add(node_id)
                    # Build the trust-gated ``parent_assessment``
                    # block while preserving the no-child marker
                    # captured by ``has_no_child_refs`` above. The
                    # trust-only branch (no children) gets the
                    # legacy shape; the overlap branch keeps
                    # ``no_child_refs`` / ``missing_child_reasons``
                    # so the post-demotion replay can keep the
                    # parent excluded.
                    trust_assessment: dict[str, Any] = {
                        "parent_status": summary_status,
                        "unsafe_children": [],
                        "safe_children_count": 0,
                        "total_children": len(child_ids),
                        "trust_gate_reasons": list(trust_reasons),
                    }
                    if no_child_marker:
                        trust_assessment["no_child_refs"] = True
                        trust_assessment["missing_child_reasons"] = (
                            list(no_child_reasons_marker)
                            if isinstance(no_child_reasons_marker, list)
                            else ["no_child_refs"]
                        )
                    hit = RaptorSummaryHit(
                        point_id=str(point.get("id") or node_id),
                        raptor_node_id=node_id,
                        raptor_root_id=str(payload.get("raptor_root_id") or ""),
                        raptor_level=int(payload.get("raptor_level") or 1),
                        raptor_tree_id=str(payload.get("raptor_tree_id") or ""),
                        raptor_build_id=str(payload.get("raptor_build_id") or ""),
                        raptor_cluster_id=str(payload.get("raptor_cluster_id") or ""),
                        raptor_child_ids=child_ids,
                        raptor_parent_ids=parent_ids,
                        raptor_summary_of=summary_of,
                        text="",
                        source_hashes=source_hashes,
                        derived_from=derived_from,
                        parent_status=summary_status,
                        parent_assessment=trust_assessment,
                        extra={
                            "memory_kind": str(payload.get("memory_kind") or ""),
                            "derivation_type": str(payload.get("derivation_type") or ""),
                            "requires_review": bool(payload.get("requires_review")),
                            "canonical": bool(payload.get("canonical")),
                            "fact_status": str(payload.get("fact_status") or ""),
                            "raptor_review_status": str(
                                payload.get("raptor_review_status") or ""
                            ),
                            "depth": depths.get(node_id, 0),
                        },
                    )
                    summaries.append(hit)
                    summary_by_node_id[node_id] = hit
                    # Redacted warning: handle only, no raw id / text /
                    # payload fields. The reason list is bounded to
                    # short unaccented tokens so it cannot echo the
                    # raw payload either.
                    warnings.append(
                        "raptor summary demoted: parent trust gate "
                        f"[{', '.join(trust_reasons)}] "
                        f"(handle={_safe_handle(node_id)})"
                    )
                    # Do NOT ``continue`` here (fix13 / final11 P2):
                    # the trust-gated parent still needs the same
                    # cap-bounded child enqueue / parent-ascent
                    # accounting as an active parent so a clean
                    # child can be retrieved and cited. The
                    # demoted parent's own text / status is preserved
                    # by the post-demotion recompute, which replays
                    # the trust gate via ``parent_assessment[
                    # "trust_gate_reasons"]`` and forces
                    # ``review_required`` + ``text == ""`` again
                    # after the child walk finishes.
                else:
                    hit = RaptorSummaryHit(
                        point_id=str(point.get("id") or node_id),
                        raptor_node_id=node_id,
                        raptor_root_id=str(payload.get("raptor_root_id") or ""),
                        raptor_level=int(payload.get("raptor_level") or 1),
                        raptor_tree_id=str(payload.get("raptor_tree_id") or ""),
                        raptor_build_id=str(payload.get("raptor_build_id") or ""),
                        raptor_cluster_id=str(payload.get("raptor_cluster_id") or ""),
                        raptor_child_ids=child_ids,
                        raptor_parent_ids=parent_ids,
                        raptor_summary_of=summary_of,
                        text=_truncate(text, safe_max_source_chars),
                        source_hashes=source_hashes,
                        derived_from=derived_from,
                        parent_status=summary_status,
                        parent_assessment=parent_assessment,
                        extra={
                            "memory_kind": str(payload.get("memory_kind") or ""),
                            "derivation_type": str(payload.get("derivation_type") or ""),
                            "requires_review": bool(payload.get("requires_review")),
                            "canonical": bool(payload.get("canonical")),
                            "fact_status": str(payload.get("fact_status") or ""),
                            "raptor_review_status": str(payload.get("raptor_review_status") or ""),
                            "depth": depths.get(node_id, 0),
                        },
                    )
                    summaries.append(hit)
                    summary_by_node_id[node_id] = hit

                # Stage 4a: enqueue child IDs (limit fanout, dedupe).
                # We also populate the leaf-id → parent-summary mapping
                # here so the cited-leaves stage can attribute each leaf
                # back to its actual parent rather than always falling
                # back to the first parent. When a leaf is shared across
                # multiple parents the first parent encountered wins; this
                # is deterministic by iteration order.
                #
                # Phase 5 fix7 + fix8: also track the **referenced**
                # child count per parent (deduped across
                # ``raptor_child_ids`` + ``raptor_summary_of``) so the
                # parent-status recomputation can demote parents whose
                # children are missing / deleted / scope-filtered by
                # the backend, not just demoted for safety reasons.
                #
                # Phase 5 fix9 (final7 finding #1): the referenced set
                # MUST only count children the searcher actually
                # intended to retrieve — i.e. children that fit inside
                # the ``safe_max_children`` fanout cap. Children beyond
                # the cap are intentionally not retrieved because of
                # the fanout budget; they are NOT missing / deleted /
                # scope-filtered evidence. Counting them in
                # ``referenced_set`` and subtracting ``retrieved_set``
                # below would falsely inflate the missing count and
                # demote a perfectly safe parent. We therefore add a
                # cid to ``referenced_for_parent`` only when we have
                # fanout budget for it (or it was already enqueued by
                # another parent sharing the cap).
                enqueued = 0
                enqueued_unique: set[str] = set()
                referenced_for_parent: set[str] = set()
                for cid in child_ids + summary_of:
                    if not cid:
                        continue
                    # Cap accounting: children beyond ``safe_max_children``
                    # are budget-skipped, not missing. Only count cids
                    # that the searcher intended to retrieve (i.e. had
                    # fanout budget for) as referenced for
                    # missing-evidence purposes. A cid that was
                    # already enqueued by a previous parent still
                    # counts for this parent: the searcher DID intend
                    # to see it, it just was not fetched twice.
                    already_enqueued = cid in seen_leaf_ids
                    if enqueued >= safe_max_children and not already_enqueued:
                        # Pure fanout budget skip: do not mark as
                        # referenced. The parent is not responsible
                        # for retrieving it.
                        continue
                    referenced_for_parent.add(cid)
                    # Every parent that references a leaf is appended
                    # to ``parents_for_leaf`` so safety accounting
                    # below can demote EVERY parent that depends on a
                    # shared unsafe / missing child, not just the
                    # first parent encountered. We append
                    # unconditionally (no ``setdefault``-wins
                    # attribution) because accounting must reach all
                    # parents, while the leaf-row output below still
                    # uses deterministic first-seen-wins to keep the
                    # ``cited_leaves`` projection stable.
                    parents_for_leaf.setdefault(cid, []).append(hit)
                    if already_enqueued:
                        continue
                    seen_leaf_ids.add(cid)
                    all_leaf_ids.append(cid)
                    enqueued_unique.add(cid)
                    enqueued += 1
                # Use the **referenced** child set (not just the
                # enqueued slice) so missing-children accounting is
                # derived against the parent's own evidence graph
                # within the cap, not against the global dedupe set.
                # Children beyond the fanout cap are excluded from
                # the referenced set so the missing-count
                # ``referenced - retrieved`` accurately reflects
                # actual missing evidence, not budget-skipped
                # children. A parent that shares every child with
                # another parent will still see all its children as
                # referenced (the ``already_enqueued`` branch above
                # adds them) even though none of them was enqueued
                # via this parent.
                per_parent_referenced_children[id(hit)] = referenced_for_parent

                # Stage 4b: enqueue parent IDs so we can zoom up to the root
                # within safe_max_depth hops. Only ascend to parents we have
                # not already visited and not already in flight.
                if depths.get(node_id, 0) < safe_max_depth:
                    for pid in parent_ids:
                        if pid and pid not in retrieved_node_ids:
                            parent_queue.append((pid, depths[node_id] + 1))

        debug["stages"]["parents_visited"] = len(summary_by_node_id)
        debug["stages"]["leaf_ids_collected"] = len(all_leaf_ids)
        debug["stages"]["retrieval_iterations"] = iterations

        # Stage 5: retrieve child leaves, project redacted view, run safety.
        cited_leaves: list[RaptorLeafHit] = []
        unsafe_leaf_ids: set[str] = set()
        # Phase 5 fix6: per-parent child safety tracking. We track two
        # parallel structures so the post-demotion parent
        # recomputation knows both:
        # * which children SURVIVED the demotion (so the parent can
        #   see its real, post-safety child population), and
        # * which children WERE demoted for safety reasons (so a
        #   parent with zero surviving-but-some-demoted children is
        #   still classified as non-active rather than as "no
        #   children at all").
        #
        # Keyed by ``RaptorSummaryHit`` identity so two parents at
        # different depths with the same node_id do not collide.
        per_parent_safe_payloads: dict[int, list[Mapping[str, Any]]] = {}
        per_parent_unsafe_count: dict[int, int] = {}
        # Phase 5 fix8 (final6 #2): also track per-parent
        # **retrieved** child sets so missing-count is computed
        # against the parent's own referenced set, not against a
        # global dedupe set. ``parents_for_leaf`` (declared above
        # next to ``per_parent_referenced_children``) is used to
        # resolve the list of parents that referenced each leaf
        # so a shared unsafe / missing child demotes every parent
        # that depended on it.
        per_parent_retrieved_children: dict[int, set[str]] = {}

        if all_leaf_ids:
            chunks: list[list[str]] = []
            chunk_size = max(1, min(32, safe_max_children * 2))
            for start in range(0, len(all_leaf_ids), chunk_size):
                chunks.append(all_leaf_ids[start : start + chunk_size])
            for chunk in chunks:
                retrieved = self._safe_retrieve(chunk, warnings)
                for point in retrieved:
                    payload = point.get("payload") if isinstance(point, Mapping) else None
                    if not isinstance(payload, Mapping):
                        continue
                    pid = str(point.get("id") or "")
                    if not pid:
                        continue
                    # Phase 5 fix8 (final6 #2): every parent that
                    # referenced this leaf must absorb the safety
                    # outcome, not just the first one. We resolve
                    # the attribution list once and reuse it for
                    # both safe and unsafe bookkeeping so a shared
                    # unsafe / missing child demotes every parent
                    # that depended on it.
                    attributed_parents = parents_for_leaf.get(pid, [])
                    if not attributed_parents:
                        # The leaf was retrieved but no parent in
                        # the searcher attributed it (e.g. it was
                        # reached via an explicit-id branch). Drop
                        # the leaf and warn with a redacted handle.
                        unsafe_leaf_ids.add(pid)
                        warnings.append(
                            "leaf dropped: no parent attribution found "
                            f"(handle={_safe_handle(pid)})"
                        )
                        continue
                    if not _leaf_payload_visible(payload):
                        unsafe_leaf_ids.add(pid)
                        # Phase 5 fix6 + fix8 (final6 #2): bump the
                        # unsafe-child counter for EVERY parent
                        # that referenced this shared unsafe child.
                        for attributed in attributed_parents:
                            key = id(attributed)
                            per_parent_unsafe_count[key] = (
                                per_parent_unsafe_count.get(key, 0) + 1
                            )
                            per_parent_retrieved_children.setdefault(
                                key, set()
                            ).add(pid)
                        warnings.append(
                            f"leaf skipped: secret-bearing text "
                            f"(handle={_safe_handle(pid)})"
                        )
                        continue
                    safety = assess_leaf_safety(payload)
                    if not safety.get("safe", False):
                        unsafe_leaf_ids.add(pid)
                        # Phase 5 fix6 + fix8 (final6 #2): track the
                        # unsafe child count for EVERY parent that
                        # referenced this leaf so the post-demotion
                        # recomputation can demote each parent that
                        # shared this unsafe child.
                        for attributed in attributed_parents:
                            key = id(attributed)
                            per_parent_unsafe_count[key] = (
                                per_parent_unsafe_count.get(key, 0) + 1
                            )
                            per_parent_retrieved_children.setdefault(
                                key, set()
                            ).add(pid)
                        reasons = ", ".join(safety.get("reasons") or [])
                        warnings.append(
                            f"leaf demoted: unsafe reasons=[{reasons}] "
                            f"(handle={_safe_handle(pid)})"
                        )
                        continue
                    # Phase 5 fix8 (final6 #2): the leaf-row
                    # ``cited_leaves`` output stays first-seen-wins
                    # (deterministic attribution) but per-parent
                    # safety bookkeeping adds the safe payload to
                    # EVERY parent that referenced the leaf so a
                    # shared safe child can keep all sharing
                    # parents active (when no unsafe / missing
                    # children exist for those parents).
                    normalized = _normalize_leaf_payload(payload)
                    # First-seen-wins parent for the leaf-row
                    # projection. This is the same parent that
                    # ``deduped_leaves`` will surface downstream.
                    leaf_row_parent = attributed_parents[0]
                    for attributed in attributed_parents:
                        key = id(attributed)
                        per_parent_retrieved_children.setdefault(
                            key, set()
                        ).add(pid)
                    # Phase 5 fix6: track per-parent surviving
                    # children so we can recompute
                    # ``parent_status`` after child demotion. The
                    # key is the ``RaptorSummaryHit`` instance id
                    # (avoid using a fragile string comparison
                    # against ``node_id`` because two parents may
                    # share the same node id at different depths).
                    # Phase 5 fix8 (final6 #2): add the safe payload
                    # to EVERY parent that referenced the leaf so a
                    # shared safe child can keep all sharing parents
                    # active (the per-parent missing-count is
                    # derived from referenced-set-minus-retrieved-set
                    # below).
                    for attributed in attributed_parents:
                        key = id(attributed)
                        try:
                            per_parent_safe_payloads.setdefault(
                                key, []
                            ).append(payload)
                        except Exception:
                            pass
                    cited_leaves.append(
                        RaptorLeafHit(
                            point_id=pid,
                            parent_raptor_node_id=getattr(leaf_row_parent, "raptor_node_id", "") or "",
                            parent_point_id=getattr(leaf_row_parent, "point_id", "") or "",
                            text=_truncate(
                                _redact_leaf_text(normalized.get("text", "") or ""),
                                safe_max_source_chars,
                            ),
                            source_uri=str(normalized.get("source_uri") or ""),
                            file_path=str(normalized.get("file_path") or ""),
                            heading=str(normalized.get("heading") or ""),
                            content_hash=str(normalized.get("content_hash") or ""),
                            source_type=str(normalized.get("source_type") or ""),
                            locator=normalized.get("locator") or {},
                            safety=safety,
                        )
                    )

        # Dedupe leaves by point ID (in case multiple parents share a child).
        deduped_leaves: list[RaptorLeafHit] = []
        seen_leaf_point_ids: set[str] = set()
        for leaf in cited_leaves:
            if leaf.point_id in seen_leaf_point_ids:
                continue
            seen_leaf_point_ids.add(leaf.point_id)
            deduped_leaves.append(leaf)

        # Sort leaves deterministically (parent, then point_id).
        deduped_leaves.sort(
            key=lambda leaf: (leaf.parent_point_id, leaf.point_id)
        )

        # Phase 5 fix6 / fix7: recompute ``parent_status`` after child
        # safety demotion AND child-evidence loss so a parent never
        # remains ``active`` while its child evidence has been
        # dropped. We seed the assessment with the surviving safe
        # children; if any unsafe child was seen for that parent or
        # any referenced child is missing (deleted / scope-filtered /
        # backend-removed), we ALSO inject a synthetic unsafe
        # payload into the assessment list so ``assess_parent_status``
        # returns the right value (it would otherwise treat the
        # parent as childless-active when every child was dropped
        # for safety or never made it back from the backend).
        # Summary entries whose ``parent_status`` is now non-active
        # are demoted: their ``text`` is cleared and the parent is
        # added to ``unsafe_summary_ids`` with a redacted warning.
        # The summary dataclass itself stays in ``summaries`` (so the
        # caller can still see *which* parent is unsafe and why), but
        # its text never reaches the caller as active context.
        for hit in summaries:
            key = id(hit)
            surviving = list(per_parent_safe_payloads.get(key, []))
            unsafe_seen = int(per_parent_unsafe_count.get(key, 0))
            # Phase 5 fix8 (final6 #2): missing children are
            # computed against THIS parent's referenced set, NOT
            # against the global dedupe set. The previous fix7
            # implementation used ``referenced_count`` (an int)
            # derived from the per-parent enqueued slice, which
            # silently dropped shared children that an earlier
            # parent had already enqueued. We now have the full
            # per-parent referenced set, so we compute
            # ``missing = referenced - retrieved`` as a set
            # difference: any referenced child that was never
            # returned (regardless of whether it was enqueued by
            # this parent or another sharing parent) counts as
            # missing.
            referenced_set = set(per_parent_referenced_children.get(key, set()))
            retrieved_set = set(per_parent_retrieved_children.get(key, set()))
            missing_count = max(0, len(referenced_set - retrieved_set))
            total_unsafe = unsafe_seen + missing_count
            # Build the assessment list. If the parent had unsafe
            # or missing children, inject a synthetic unsafe payload
            # so the conservative ``assess_parent_status``
            # vocabulary can classify the parent as non-active
            # (``stale`` or ``excluded``) without re-running the
            # actual leaf safety logic against payloads we no
            # longer carry.
            assessment_children: list[Mapping[str, Any]] = list(surviving)
            if total_unsafe > 0:
                for _ in range(total_unsafe):
                    assessment_children.append({
                        "_demoted_marker": True,
                        "requires_review": True,
                    })
            assessment = assess_parent_status(assessment_children)
            new_status = str(assessment.get("parent_status") or "active")

            # Phase 5 fix12 (final10 P2) defense-in-depth: re-run
            # the parent trust gate at recompute time so a payload
            # that was demoted at promotion cannot be re-promoted
            # to ``active`` here. We capture the trust reasons
            # from the OLD ``parent_assessment`` BEFORE we
            # overwrite it on the hit. When the parent was
            # promoted clean (``trust_gate_reasons`` empty or
            # absent) we let the new child-safety recompute run.
            # When it was demoted at promotion time we FORCE
            # ``review_required`` here so even an all-clean child
            # population cannot resurrect an unsafe-by-trust
            # parent.
            trust_reasons_replay: list[str] = []
            existing_assessment = getattr(hit, "parent_assessment", None)
            if isinstance(existing_assessment, Mapping):
                stored = existing_assessment.get("trust_gate_reasons")
                if isinstance(stored, list):
                    trust_reasons_replay = [str(s) for s in stored if str(s)]
            hit.parent_assessment = dict(assessment)
            # Phase 5 fix15 (final12 P2 finding): replay the
            # childless-marker captured at construction time so a
            # parent that was originally childless cannot be
            # re-promoted to ``"active"`` by
            # ``assess_parent_status([])`` after the child walk
            # (which has no children to walk — every leaf id was
            # budget-skipped because the cap-bounded enqueue saw
            # an empty ``child_ids + summary_of`` list). The marker
            # is also force-set to ``"excluded"`` rather than
            # ``"review_required"`` so the demotion status aligns
            # with the original no-child downgrade fired at
            # promotion time, satisfying the security reviewer's
            # acceptance criterion that ``parent_assessment[
            # "parent_status"]`` agree with the dataclass-level
            # ``hit.parent_status``.
            #
            # Phase 5 fix16 (final13 P2 finding): the no-child
            # replay MUST run BEFORE the trust replay so a parent
            # that carries both markers (the realistic production
            # overlap of ``requires_review=True`` with empty child
            # refs) ends up ``"excluded"`` rather than
            # ``"review_required"``. When the no-child marker is
            # present we (1) force ``parent_status = "excluded"``,
            # (2) preserve the ``trust_gate_reasons`` list from the
            # existing assessment so operators can still see why
            # the trust gate fired, and (3) emit a warning that
            # surfaces BOTH reason families (no-evidence wins +
            # trust reasons) without leaking raw payload content.
            # The legacy trust-only path (no children) runs only
            # when the no-child marker is absent.
            missing_reasons_replay: list[str] = []
            missing_marker_present = bool(
                existing_assessment.get("no_child_refs")
            ) if isinstance(existing_assessment, Mapping) else False
            if (
                not missing_marker_present
                and isinstance(existing_assessment, Mapping)
            ):
                stored_missing = existing_assessment.get("missing_child_reasons")
                if isinstance(stored_missing, list) and any(
                    str(s) for s in stored_missing
                ):
                    missing_marker_present = True
                    missing_reasons_replay = [
                        str(s) for s in stored_missing if str(s)
                    ]
            if missing_marker_present:
                if not missing_reasons_replay:
                    missing_reasons_replay = ["no_child_refs"]
                # Phase 5 fix16: when the trust gate ALSO fired
                # (the realistic production overlap of
                # ``requires_review=True`` + empty child refs),
                # preserve the trust reasons on the assessment
                # so the audit envelope can show *both* signals.
                # The dataclass-level ``parent_status`` stays
                # ``"excluded"`` because childless / no-source-
                # evidence is the stricter of the two demotions.
                preserved_trust_reasons = trust_reasons_replay
                hit.parent_assessment["parent_status"] = "excluded"
                hit.parent_assessment["missing_child_reasons"] = list(
                    missing_reasons_replay
                )
                hit.parent_assessment["no_child_refs"] = True
                if preserved_trust_reasons:
                    hit.parent_assessment["trust_gate_reasons"] = list(
                        preserved_trust_reasons
                    )
                unsafe_summary_ids.add(hit.raptor_node_id)
                hit.parent_status = "excluded"
                hit.text = ""
                # Combine the warning reasons so operators can
                # see both the no-child evidence loss and the
                # trust-gate markers; both reason families are
                # bounded to short unaccented tokens so the
                # warning channel cannot echo raw payload
                # content.
                combined_reasons: list[str] = list(missing_reasons_replay)
                if preserved_trust_reasons:
                    combined_reasons.extend(preserved_trust_reasons)
                warnings.append(
                    "raptor summary demoted: no source-backed child refs "
                    "(replay) "
                    f"[{', '.join(combined_reasons)}] "
                    f"(handle={_safe_handle(hit.raptor_node_id)})"
                )
                continue
            if trust_reasons_replay:
                # Phase 5 fix14: the dataclass-level ``hit.parent_status``
                # is forced to ``review_required`` below, so the nested
                # ``parent_assessment`` projection must agree. Without
                # this sync, a trust-gated parent whose all-clean
                # children made ``assess_parent_status`` return
                # ``active`` would surface
                # ``summary["parent_status"] == "review_required"``
                # alongside ``summary["parent_assessment"]
                # ["parent_status"] == "active"`` under
                # ``include_metadata=True`` — an avoidable output
                # inconsistency. We mirror the demoted value into
                # ``parent_assessment`` while preserving the
                # ``trust_gate_reasons`` list, the safe/unsafe child
                # counters, and every other field returned by
                # ``assess_parent_status`` (replayed via the
                # ``dict(assessment)`` snapshot above).
                #
                # Phase 5 fix16: this branch only runs when the
                # no-child marker is ABSENT (i.e. the parent has
                # source-backed child refs). When the parent has
                # children, trust-only demotion semantics are
                # unchanged — the parent stays ``review_required``
                # with cleared text but the cited clean children
                # remain accessible via ``cited_leaves``.
                hit.parent_assessment["parent_status"] = "review_required"
                hit.parent_assessment["trust_gate_reasons"] = list(
                    trust_reasons_replay
                )
                unsafe_summary_ids.add(hit.raptor_node_id)
                hit.parent_status = "review_required"
                hit.text = ""
                warnings.append(
                    "raptor summary demoted: parent trust gate (replay) "
                    f"[{', '.join(trust_reasons_replay)}] "
                    f"(handle={_safe_handle(hit.raptor_node_id)})"
                )
                continue
            if new_status == "active":
                # No demotion needed: keep ``parent_status`` and
                # ``text`` as-is so the caller's safe path is
                # untouched.
                hit.parent_status = "active"
                continue
            # Non-active parent: clear the summary text so the unsafe
            # parent never reaches the caller as active context, and
            # add a redacted warning so operators can correlate via
            # the audit envelope. ``unsafe_summary_ids`` tracks the
            # parent_node_id (not the raw payload) so the dense
            # envelope can render it through the same redacted handle
            # discipline.
            unsafe_summary_ids.add(hit.raptor_node_id)
            hit.parent_status = new_status
            hit.text = ""
            warnings.append(
                "raptor summary demoted after child-safety review: "
                f"status={new_status} "
                f"(handle={_safe_handle(hit.raptor_node_id)})"
            )

        # Final cap: top_k summaries and the per-summary top max_children
        # leaves, bounded by the cumulative context budget.
        context_used = 0
        truncated_summary_ids: set[str] = set()
        final_summaries: list[RaptorSummaryHit] = []
        for hit in summaries:
            # Phase 5 fix6: a demoted parent summary has empty text
            # (zero char cost). Still respect the top_k summary cap
            # but never count non-active parent text against
            # ``context_used`` so unrelated safe parents are not
            # starved.
            char_cost = len(hit.text or "")
            context_used += char_cost
            if context_used > HARD_CONTEXT_CHAR_BUDGET or len(final_summaries) >= safe_top_k:
                truncated_summary_ids.add(hit.raptor_node_id)
                continue
            final_summaries.append(hit)

        truncated_leaf_ids: set[str] = set()
        final_leaves: list[RaptorLeafHit] = []
        for leaf in deduped_leaves:
            context_used += len(leaf.text)
            if context_used > HARD_CONTEXT_CHAR_BUDGET:
                truncated_leaf_ids.add(leaf.point_id)
                continue
            final_leaves.append(leaf)

        debug["stages"]["context_used_chars"] = context_used
        debug["truncated_summary_ids"] = sorted(_safe_handle(s) for s in truncated_summary_ids)
        debug["truncated_leaf_ids"] = sorted(_safe_handle(s) for s in truncated_leaf_ids)

        # Sort summaries deterministically (level asc, then node_id asc).
        final_summaries.sort(
            key=lambda hit: (int(hit.raptor_level or 0), hit.raptor_node_id)
        )

        return RaptorSearchResult(
            query=query,
            summaries=final_summaries,
            cited_leaves=final_leaves,
            warnings=warnings,
            debug=debug,
            unsafe_summary_ids=unsafe_summary_ids | truncated_summary_ids,
            unsafe_leaf_ids=unsafe_leaf_ids | truncated_leaf_ids,
        )

    # -- Private helpers -------------------------------------------------

    def _safe_retrieve(
        self,
        point_ids: Iterable[str],
        warnings: list[str],
    ) -> list[dict[str, Any]]:
        """Call ``QdrantClient.retrieve`` defensively.

        * No filter is passed (Qdrant does not support one here).
        * Every returned payload is post-filtered against :attr:`scope`.
        * Per-call failures are caught and surfaced as a warning so partial
          results still flow.
        * Duplicate point IDs are dropped defensively.

        Phase 5 fix6: warning text does **not** include raw exception
        text or any of the requested point ids. The exception
        ``__str__`` can echo secret-shaped values (e.g. a backend
        error message that quotes the requested ``raptor_node_id``),
        and the point ids themselves can be secret-shaped. The warning
        carries only the requested count so the caller can correlate
        with server logs if needed.
        """
        ids = []
        seen: set[str] = set()
        for raw in point_ids or []:
            sid = str(raw or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            ids.append(sid)
        if not ids:
            return []
        try:
            retrieved = self.qdrant.retrieve(
                self.collection_name,
                ids,
                with_payload=True,
                with_vector=False,
            )
        except Exception:
            # Phase 5 fix6: emit a sanitized warning. Do NOT include
            # ``{exc}`` because the exception message can echo the
            # requested ids (which may be secret-shaped) or other
            # raw backend strings. Operators correlate via server logs.
            warnings.append(
                f"retrieve({len(ids)} ids) failed (no raw exception "
                "leaked; see server logs)"
            )
            return []
        safe: list[dict[str, Any]] = []
        for point in retrieved or []:
            if not isinstance(point, Mapping):
                continue
            payload = (
                point.get("payload")
                if isinstance(point.get("payload"), Mapping)
                else {}
            )
            if not _payload_matches_scope(payload, self.scope):
                continue
            safe.append(point)
        return safe


def _pick_parent_for_leaf(
    leaf_ids: list[str],
    summary_by_node_id: dict[str, RaptorSummaryHit],
) -> dict[str, str]:
    """Return a best-effort ``leaf_id -> parent_node_id`` mapping.

    The default mapping falls back to the first parent RAPTOR node so
    leaves from a same retrieve() batch anchor to a stable parent even if
    their exact ``raptor_parent_ids`` were not provided (legacy Phase 3
    payloads). Callers override this only when they can resolve the exact
    link.
    """
    fallback = next(iter(summary_by_node_id), "")
    if not fallback:
        return {}
    return {leaf_id: fallback for leaf_id in leaf_ids}


__all__ = [
    "HARD_CONTEXT_CHAR_BUDGET",
    "HARD_MAX_CHILDREN",
    "HARD_MAX_DEPTH",
    "HARD_MAX_SOURCE_CHARS",
    "HARD_SEED_TOP_K",
    "RaptorLeafHit",
    "RaptorSearcher",
    "RaptorSearchResult",
    "RaptorSummaryHit",
]
