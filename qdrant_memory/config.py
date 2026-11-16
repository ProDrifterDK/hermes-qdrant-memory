from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "qdrant_url": "http://127.0.0.1:6333",
    "embedding_url": "http://127.0.0.1:8080/v1",
    "embedding_model": "bge-m3",
    "embedding_max_input_chars": 12000,
    "embedding_max_chunks": 16,
    "vector_size": 1024,
    "distance": "Cosine",
    "collection_name": "hermes_memory",
    "learning_collection_name": "hermes_learnings",
    "auto_recall": True,
    "auto_recall_top_k": 5,
    "search_candidates": 20,
    "decay_rate": 0.001,
    "max_chunk_tokens": 512,
    "display_tokens": 300,
    "sync_turns": True,
    "sync_subagents": False,
    "learning_enabled": True,
    "consolidation_enabled": False,
    "consolidation_report_max_points": 200,
    "consolidation_report_max_groups": 20,
    "consolidation_duplicate_threshold": 0.92,
    "consolidation_stale_days": 90,
    "consolidation_min_importance_for_keep": 4,
    "consolidation_persist_reports": True,
    "consolidation_artifact_dir": "",
    "proposal_artifact_dir": "",
    "obsidian_adapter_enabled": False,
    "obsidian_vault_root": "",
    "obsidian_proposal_dir": "",
    "consolidation_apply_dry_run_default": True,
    "guarded_auto_max_actions": 10,
    "guarded_auto_duplicate_min_confidence": 0.98,
    "guarded_auto_duplicate_max_cluster_size": 20,
    "guarded_auto_learning_min_confidence": 0.90,
    "guarded_auto_stale_quarantine_enabled": True,
    "guarded_auto_quarantine_days": 30,
    "backup_artifact_dir": "",
    "reconsolidation_enabled": False,
    "reconsolidation_report_only": True,
    "reconsolidation_include_by_default": False,
    "reconsolidation_min_confidence": 0.6,
    "reconsolidation_max_candidates": 10,
    "query_prefix": "search_query: ",
    "document_prefix": "search_document: ",
    "scope_mode": "profile",
    "min_raw_score": 0.0,
    "min_final_score": 0.0,
    # Automatic recall only (the legacy block): memories below these are not injected; with none
    # left the block is empty. Tool searches use min_raw_score/min_final_score. Off by default
    # ("off", "", null or 0 disables); docs/session-recall-prefetch.md "Relevance floor" gives
    # values measured for one embedding/reranker pair, to be re-derived for other models.
    "auto_recall_min_final_score": "off",
    "auto_recall_min_raw_score": "off",
    # Messages with fewer letters/digits than this skip automatic recall (legacy, hybrid and
    # lane). Off by default ("off", "", null or 0).
    "auto_recall_min_word_chars": "off",
    "provenance_ranking_enabled": True,
    "qdrant_api_key": "",
    "embedding_api_key": "",
    "index_dirs": [],
    "index_extensions": [".md", ".txt"],
    "index_exclude_dirs": [
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        "dist",
        "build",
        "target",
        ".next",
        ".cache",
    ],
    "index_max_files": 500,
    "index_dry_run_default": True,
    "manual_store_duplicate_threshold": 0.92,
    "manual_store_duplicate_top_k": 3,
    # Phase 6H: runtime shadow mode for hybrid auto-recall path.
    # Disabled by default; never alters prompt context.
    "auto_recall_shadow_enabled": False,
    "auto_recall_shadow_max_per_session": 20,
    "auto_recall_shadow_artifact_dir": "",
    "auto_recall_shadow_mode": "hybrid",
    # Phase 6I: opt-in hybrid auto-recall prompt injection.
    # ``legacy`` (default) keeps dense-only auto-recall.
    # ``hybrid`` routes the prefetch prompt through a dedicated prompt-safe
    # hybrid formatter. Invalid values fail closed to ``legacy``.
    "auto_recall_mode": "legacy",
    # Lineage (W0): supported values are exactly off, capture, reconcile.
    # Default/fail-closed value is off. In W0 no writer is enabled: capture
    # becomes operational in W1 and reconcile stays closed until W2.
    "lineage_mode": "off",
    "lineage_lock_dir": "",
    "lineage_lock_timeout_seconds": 5.0,
    "lineage_invalidation_max_depth": 8,
    "lineage_invalidation_max_points": 4096,
    "lineage_invalidation_max_edges": 8192,
    "lineage_trace_downstream_max_results": 100,
    # Session recall: raw conversation windows from state.db in their own collection.
    # See README "Session recall". `$HERMES_HOME` in the state.db path is replaced by
    # the provider's Hermes home. An empty rerank_url means dense order only (the tool
    # says so in its output); an empty embedding_url means the plugin's embedding_url.
    "session_recall_enabled": True,
    "session_recall_collection": "hermes_session_windows",
    "session_recall_state_db": "$HERMES_HOME/state.db",
    "session_recall_window_chars": 1000,
    "session_recall_overlap_chars": 128,
    "session_recall_candidates": 128,
    "session_recall_top_k": 15,
    "session_recall_max_chars": 16000,
    "session_recall_rerank_url": "",
    "session_recall_rerank_timeout_s": 30.0,
    "session_recall_embedding_url": "",
    "session_recall_auto_index": True,
    "session_recall_exclude_sources": [],
    "session_recall_lag_warn_messages": 500,
    "session_recall_inline_catchup_seconds": 8.0,
    "session_recall_dedup": True,
    # Session-recall lane in automatic recall (C'5), opt-in. Each prefetch also searches
    # the session windows with the latest user message and appends verbatim windows that
    # are no longer in the active prompt. Fail-open: any error or timeout returns the
    # legacy prefetch unchanged. See docs/session-recall-prefetch.md.
    "session_prefetch_enabled": False,
    "session_prefetch_scope": "current",
    "session_prefetch_top_k": 5,
    "session_prefetch_max_chars": 1200,
    # 0 = the legacy block's cap (max(800, display_tokens * 4)) plus session_prefetch_max_chars.
    "session_prefetch_total_chars": 0,
    "session_prefetch_rerank": True,
    # Opt-in: drop a recalled window prompted only by a one-shot run (capture_reader.window_is_lane_noise),
    # e.g. a test harness's "Reply with exactly: OK" turn. Agent-prompted windows are kept.
    "session_prefetch_exclude_machine": False,
    # Automatic capture (docs/capture.md). ``off`` (default): no capture tools, ``capture run``
    # refuses. ``dry_run``: runs only report. ``on``: ``capture run --apply`` queues, writes the
    # auto and mirror routes and advances the cursor. Review items expire after N days.
    "capture_mode": "off",
    "capture_review_expiry_days": 21,
    # Gateway sender tags (``[Name] ``) that are the owner. Empty: no row of a shared channel
    # qualifies; a row without a sender tag in a one-to-one conversation is the owner's.
    "capture_owner_senders": [],
    # Auto route (lane A): off by default (review only). When on, only the design's auto rule
    # (heuristic AND a second signal >= 0.9, kind preference/invariant/correction/decision,
    # not task-scoped, no reversal, not identity/money/security), at most N per day.
    "capture_auto_apply": False,
    "capture_daily_auto_cap": 5,
    # Dense near-duplicate threshold among stored captures (cosine); a match merges the citation.
    "capture_dedupe_threshold": 0.92,
    "session_prefetch_candidates": 32,
    "session_prefetch_timeout_s": 2.0,
    # Lane relevance floors: a reranked excerpt needs at least min_rerank_score, a dense-ordered
    # one (no reranker, or it failed) at least min_dense_score (cosine). Off by default ("off",
    # "" or null; 0 is a real threshold on these scales). Values are model-specific: see
    # docs/session-recall-prefetch.md "Relevance floor".
    "session_prefetch_min_rerank_score": "off",
    "session_prefetch_min_dense_score": "off",
    # Opt-in read-only tool qdrant_memory_session_expand (messages around a msg id).
    "session_expand_enabled": False,
    # C' session origin: ``off`` (default) keeps manual-store payloads, schemas and outputs
    # as they were; ``recorded`` adds a digest-pinned state.db citation (``session_origin``)
    # to manual stores and the optional ``cite`` store argument. ``recorded`` never blocks a
    # store because state.db cannot resolve the citation: it records ``unresolved:<reason>``.
    # Unknown values fall back to ``off``. See README "Session origin on manual stores".
    "session_origin_mode": "off",
    # Tool surface: ``full`` exposes every tool (default); ``recall`` exposes only the
    # read-mostly recall tools in tools.RECALL_TOOL_NAMES. Unknown values fall back to ``full``.
    "tool_surface": "full",
    # Extra read-only sources merged into recall: a list of
    # {collection, profile_id (null = no profile filter), kind: memory|learning}.
    # The client refuses every mutating request against these collections.
    "read_sources": [],
}

# Keys of the session-hook capture paths removed in capture CP1 (the learning heuristic and the
# source-line hook extractor). Configs that still set them keep loading: the keys are ignored
# and the first load that sees one logs a single warning for the process.
DEPRECATED_KEY_PREFIXES: tuple[str, ...] = ("learning_auto_extract_", "source_extraction_")
_deprecation_warned = False


def deprecated_keys_set(*sources: Mapping[str, Any] | None) -> list[str]:
    """The removed capture keys present in ``sources`` (config mappings or ``os.environ``)."""
    found: set[str] = set()
    for source in sources:
        if not isinstance(source, Mapping):
            continue
        for key in source:
            name = str(key)
            if name.startswith("HERMES_QDRANT_MEMORY_"):
                name = name[len("HERMES_QDRANT_MEMORY_"):].lower()
            if name.startswith(DEPRECATED_KEY_PREFIXES):
                found.add(name)
    return sorted(found)


def _warn_deprecated_once(keys: list[str]) -> None:
    global _deprecation_warned
    if not keys or _deprecation_warned:
        return
    _deprecation_warned = True
    logger.warning(
        "qdrant_memory: ignoring removed capture settings %s; the session-hook learning and "
        "source extraction paths no longer exist (see docs/capture.md). Remove them from your config.",
        ", ".join(keys),
    )


TOOL_SURFACES: frozenset[str] = frozenset({"full", "recall"})
READ_SOURCE_KINDS: frozenset[str] = frozenset({"memory", "learning"})

_BOOL_KEYS = {
    "enabled",
    "auto_recall",
    "sync_turns",
    "sync_subagents",
    "learning_enabled",
    "consolidation_enabled",
    "consolidation_persist_reports",
    "obsidian_adapter_enabled",
    "consolidation_apply_dry_run_default",
    "guarded_auto_stale_quarantine_enabled",
    "reconsolidation_enabled",
    "reconsolidation_report_only",
    "reconsolidation_include_by_default",
    "index_dry_run_default",
    "provenance_ranking_enabled",
    "auto_recall_shadow_enabled",
    "session_recall_enabled",
    "session_recall_auto_index",
    "session_recall_dedup",
    "session_prefetch_enabled",
    "session_prefetch_rerank",
    "session_prefetch_exclude_machine",
    "session_expand_enabled",
    "capture_auto_apply",
}
_INT_KEYS = {
    "vector_size",
    "embedding_max_input_chars",
    "embedding_max_chunks",
    "auto_recall_top_k",
    "search_candidates",
    "max_chunk_tokens",
    "display_tokens",
    "index_max_files",
    "consolidation_report_max_points",
    "consolidation_report_max_groups",
    "consolidation_stale_days",
    "consolidation_min_importance_for_keep",
    "guarded_auto_max_actions",
    "guarded_auto_duplicate_max_cluster_size",
    "guarded_auto_quarantine_days",
    "reconsolidation_max_candidates",
    "manual_store_duplicate_top_k",
    "auto_recall_shadow_max_per_session",
    "lineage_invalidation_max_depth",
    "lineage_invalidation_max_points",
    "lineage_invalidation_max_edges",
    "lineage_trace_downstream_max_results",
    "session_recall_window_chars",
    "session_recall_overlap_chars",
    "session_recall_candidates",
    "session_recall_top_k",
    "session_recall_max_chars",
    "session_recall_lag_warn_messages",
    "session_prefetch_top_k",
    "session_prefetch_max_chars",
    "session_prefetch_total_chars",
    "session_prefetch_candidates",
    "capture_daily_auto_cap",
}
_CAPTURE_MODES = frozenset({"off", "dry_run", "on"})
_FLOAT_KEYS = {"capture_review_expiry_days", "capture_dedupe_threshold", "session_prefetch_timeout_s", "session_recall_rerank_timeout_s", "session_recall_inline_catchup_seconds", "decay_rate", "min_raw_score", "min_final_score", "consolidation_duplicate_threshold", "guarded_auto_duplicate_min_confidence", "guarded_auto_learning_min_confidence", "reconsolidation_min_confidence", "manual_store_duplicate_threshold", "lineage_lock_timeout_seconds"}
_LIST_KEYS = {"index_dirs", "index_extensions", "index_exclude_dirs", "session_recall_exclude_sources", "capture_owner_senders"}
# Automatic-recall relevance floors. "off" (also "", null, false) disables every one of them.
# On the legacy floors 0 disables too (their scores are never negative); on the lane floors 0
# is a real threshold (rerank logits and cosine can be negative).
_LEGACY_FLOOR_KEYS = {"auto_recall_min_final_score", "auto_recall_min_raw_score"}
_LANE_FLOOR_KEYS = {"session_prefetch_min_rerank_score", "session_prefetch_min_dense_score"}
_OFF_WORDS = {"", "off", "none", "null", "false", "no", "disabled"}


def relevance_floor(value: Any, *, zero_is_off: bool = False) -> float | None:
    """A relevance floor from config: a number, or None when it is off.

    "off", "", null, a boolean or an unparseable value are off (no floor, the behaviour
    without the key). With ``zero_is_off`` a 0 is off too.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str) and value.strip().lower() in _OFF_WORDS:
        return None
    try:
        number = float(value)
    except Exception:
        logger.warning("qdrant_memory: unreadable relevance floor %r, treated as off", value)
        return None
    if number != number or (zero_is_off and number == 0.0):  # NaN, or 0 on a legacy floor
        return None
    return number


def min_word_chars(value: Any) -> int:
    """``auto_recall_min_word_chars`` as a count; 0 when off ("off", "", null, 0 or garbage)."""
    if value is None or isinstance(value, bool):
        return 0
    if isinstance(value, str) and value.strip().lower() in _OFF_WORDS:
        return 0
    try:
        return max(0, int(float(value)))
    except Exception:
        logger.warning("qdrant_memory: unreadable auto_recall_min_word_chars %r, treated as off", value)
        return 0


def _canonical_floor(key: str, value: Any) -> Any:
    """Resolved config form of a floor key: a number, or the string "off"."""
    if key == "auto_recall_min_word_chars":
        return min_word_chars(value) or "off"
    floor = relevance_floor(value, zero_is_off=key in _LEGACY_FLOOR_KEYS)
    return "off" if floor is None else floor


# Phase 6I: allowed values for ``auto_recall_mode``.
# Invalid values fail closed to ``legacy``.
_AUTO_RECALL_MODES: frozenset[str] = frozenset({"legacy", "hybrid"})

# Lineage (W0): allowed values for ``lineage_mode``.
# Invalid values fail closed to ``off``.
_LINEAGE_MODES: frozenset[str] = frozenset({"off", "capture", "reconcile"})

# C' (session_origin_mode): invalid values fail closed to ``off``.
_SESSION_ORIGIN_MODES: frozenset[str] = frozenset({"off", "recorded"})


def _as_lineage_mode(value: Any) -> str:
    """Coerce ``lineage_mode`` to off, capture, or reconcile.

    Non-string values and unrecognized strings collapse to ``off``
    (fail-closed default).
    """
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _LINEAGE_MODES:
            return candidate
    return "off"


def _as_tool_surface(value: Any) -> str:
    """Coerce ``tool_surface``; anything unrecognized falls back to ``full``."""
    if isinstance(value, str) and value.strip().lower() in TOOL_SURFACES:
        return value.strip().lower()
    logger.warning("qdrant_memory.tool_surface=%r is not one of %s; using 'full'", value, sorted(TOOL_SURFACES))
    return "full"


def _as_read_sources(value: Any) -> list[dict[str, Any]]:
    """Normalize ``read_sources``; malformed entries are dropped with a warning."""
    if isinstance(value, str):
        try:
            value = json.loads(value) if value.strip() else []
        except Exception:
            logger.warning("qdrant_memory.read_sources is not valid JSON; ignoring it")
            return []
    if not isinstance(value, (list, tuple)):
        logger.warning("qdrant_memory.read_sources must be a list; ignoring it")
        return []
    sources: list[dict[str, Any]] = []
    seen: set[tuple[str, Any, str]] = set()
    for entry in value:
        if not isinstance(entry, Mapping):
            logger.warning("qdrant_memory.read_sources entry %r is not a mapping; skipping it", entry)
            continue
        collection = str(entry.get("collection") or "").strip()
        kind = str(entry.get("kind") or "memory").strip().lower()
        raw_profile = entry.get("profile_id")
        profile_id = str(raw_profile).strip() if raw_profile is not None else ""
        if not collection or kind not in READ_SOURCE_KINDS:
            logger.warning("qdrant_memory.read_sources entry %r needs a collection and kind memory|learning; skipping it", dict(entry))
            continue
        source = {"collection": collection, "profile_id": profile_id or None, "kind": kind}
        key = (collection, source["profile_id"], kind)
        if key not in seen:
            seen.add(key)
            sources.append(source)
    return sources


def read_source_collections(config: Mapping[str, Any]) -> frozenset[str]:
    """Collection names the Qdrant client must treat as read-only."""
    return frozenset(
        str(source.get("collection") or "")
        for source in config.get("read_sources") or []
        if isinstance(source, Mapping) and source.get("collection")
    )


def _as_auto_recall_mode(value: Any) -> str:
    """Coerce ``auto_recall_mode`` to ``legacy`` or ``hybrid``.

    Non-string values and unrecognized strings collapse to ``legacy``
    (fail-closed default) so a typo or a malicious value cannot activate
    the hybrid auto-recall path.
    """
    if isinstance(value, str):
        candidate = value.strip().lower()
        if candidate in _AUTO_RECALL_MODES:
            return candidate
    return "legacy"


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "no", "n", "off"}:
            return False
    return default


def _coerce(key: str, value: Any) -> Any:
    default = DEFAULTS[key]
    if value is None:
        return default
    if key in _BOOL_KEYS:
        return _as_bool(value, bool(default))
    if key in _LEGACY_FLOOR_KEYS or key in _LANE_FLOOR_KEYS or key == "auto_recall_min_word_chars":
        return _canonical_floor(key, value)
    if key in _INT_KEYS:
        try:
            return int(value)
        except Exception:
            return default
    if key in _FLOAT_KEYS:
        try:
            return float(value)
        except Exception:
            return default
    if key in _LIST_KEYS:
        if isinstance(value, list):
            return [str(item) for item in value]
        if isinstance(value, tuple):
            return [str(item) for item in value]
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return []
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except Exception:
                pass
            return [item.strip() for item in stripped.split(",") if item.strip()]
        return list(default)
    if key == "auto_recall_mode":
        return _as_auto_recall_mode(value)
    if key == "lineage_mode":
        return _as_lineage_mode(value)
    if key == "tool_surface":
        return _as_tool_surface(value)
    if key == "session_origin_mode":
        candidate = value.strip().lower() if isinstance(value, str) else ""
        return candidate if candidate in _SESSION_ORIGIN_MODES else "off"
    if key == "read_sources":
        return _as_read_sources(value)
    if key == "capture_mode":
        if isinstance(value, bool):  # YAML 1.1 reads a bare on/off as a boolean
            return "on" if value else "off"
        candidate = value.strip().lower().replace("-", "_") if isinstance(value, str) else ""
        return candidate if candidate in _CAPTURE_MODES else "off"
    if isinstance(default, str):
        return str(value)
    return value


def _load_json_config(hermes_home: str | None) -> dict[str, Any]:
    if not hermes_home:
        return {}
    path = Path(hermes_home) / "qdrant_memory.json"
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return raw if isinstance(raw, dict) else {}
    except Exception:
        return {}


def _load_hermes_config() -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config  # type: ignore

        cfg = load_config()
        return cfg if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def load_config(*, hermes_home: str | None = None, hermes_config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Load qdrant_memory config with safe defaults.

    Precedence: DEFAULTS < $HERMES_HOME/qdrant_memory.json < config.yaml qdrant_memory section <
    environment variables named HERMES_QDRANT_MEMORY_<KEY>.
    """
    merged: dict[str, Any] = dict(DEFAULTS)
    json_config = _load_json_config(hermes_home)
    merged.update({k: v for k, v in json_config.items() if k in DEFAULTS})
    root_cfg = dict(hermes_config) if hermes_config is not None else _load_hermes_config()
    section = root_cfg.get("qdrant_memory", {}) if isinstance(root_cfg, Mapping) else {}
    if isinstance(section, Mapping):
        merged.update({k: v for k, v in section.items() if k in DEFAULTS})
    plugin_env = {k: v for k, v in os.environ.items() if k.startswith("HERMES_QDRANT_MEMORY_")}
    _warn_deprecated_once(deprecated_keys_set(json_config, section if isinstance(section, Mapping) else None,
                                              plugin_env))
    for key in DEFAULTS:
        env_key = f"HERMES_QDRANT_MEMORY_{key.upper()}"
        if env_key in os.environ:
            merged[key] = os.environ[env_key]
    resolved = {key: _coerce(key, merged.get(key)) for key in DEFAULTS}
    # The learning store upserts without the lineage overwrite guard, so it is only
    # safe while its collection cannot hold a memory point. Fail closed at load
    # rather than depend on the two names happening to differ.
    memory_collection = str(resolved.get("collection_name") or "").strip()
    learning_collection = str(resolved.get("learning_collection_name") or "").strip()
    if memory_collection and memory_collection == learning_collection:
        raise ValueError(
            "collection_name and learning_collection_name must differ: the learning "
            "store writes without the lineage overwrite guard, so one shared "
            "collection would let a learning upsert replace a protected payload"
        )
    # Session recall upserts and range-deletes windows with no payload predicate at
    # all, so its collection may never be one that holds memories or learnings.
    session_collection = str(resolved.get("session_recall_collection") or "").strip()
    if session_collection and session_collection in {memory_collection, learning_collection}:
        raise ValueError(
            "session_recall_collection must differ from collection_name and "
            "learning_collection_name: the session recall indexer rewrites and deletes "
            "windows in its collection without any payload check"
        )
    # A read source is guarded read-only in the client, so it may never be one of the
    # collections this provider writes: that would either block the provider's own
    # writes or, worse, mean the "shared" source is really the provider's own.
    own_collections = {memory_collection, learning_collection, session_collection} - {""}
    clashing = sorted(read_source_collections(resolved) & own_collections)
    if clashing:
        raise ValueError(
            f"read_sources must not name collections this provider writes ({', '.join(clashing)}); "
            "set collection_name, learning_collection_name and session_recall_collection to "
            "collections of this profile's own"
        )
    return resolved
