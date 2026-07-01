from __future__ import annotations

# ruff: noqa: E402 - user plugins patch sys.path before importing Hermes/runtime modules.

import hashlib
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

# User plugins may be imported outside Hermes' package context during tests.
_HERMES_AGENT = Path.home() / ".hermes" / "hermes-agent"
_PLUGIN_DIR = Path(__file__).resolve().parent
for _path in (_PLUGIN_DIR, _HERMES_AGENT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

try:
    from agent.memory_provider import MemoryProvider  # type: ignore
except Exception:  # pragma: no cover - used only for standalone tests without Hermes installed
    class MemoryProvider:  # type: ignore[no-redef]
        """Minimal fallback base class for standalone test/lint environments.

        Hermes provides the real `agent.memory_provider.MemoryProvider` at runtime.
        """

        pass

from qdrant_memory.graph_retriever import GraphExpansionPolicy, GraphMemoryRetriever
from qdrant_memory.client import QdrantClient
from qdrant_memory.backup import create_backup
from qdrant_memory.config import load_config
from qdrant_memory.context import ContextTemplateError, build_context_packet, default_context_top_k
from qdrant_memory.consolidation import (
    _point_requires_manual_review,
    build_consolidation_report,
    expected_action_for_proposal,
    find_proposal,
    load_consolidation_report,
    make_filter,
    parse_bool_arg,
    persist_application_record,
    persist_consolidation_report,
    points_from_qdrant,
)
from qdrant_memory.embeddings import EmbeddingClient
from qdrant_memory.indexer import FileIndexer
from qdrant_memory.learning import LearningStore, build_learning_payload, classify_learning_type
from qdrant_memory.lesson_extractor import LearningCandidate, candidate_to_learning_args, contains_secret, extract_learning_candidates_from_messages
from qdrant_memory.recipes import get_recipe
from qdrant_memory.ranking import RankingPolicy
from qdrant_memory.retriever import MemoryRetriever, format_for_prompt
from qdrant_memory.sources import expand_point, inspect_point, source_status_for_point, trace_point
from qdrant_memory.proposals import proposal_draft_metadata, write_proposal_draft
from qdrant_memory.source_extraction import (
    build_source_extraction_proposal,
    evaluate_source_extraction_candidate,
    extract_source_candidates_from_messages,
    preview_source_extraction_candidates,
)
from qdrant_memory.improve import (
    IMPROVE_MAX_CANDIDATES_DEFAULT,
    IMPROVE_MAX_CANDIDATES_HARD_CAP,
    REPORT_ID_RE,
    build_improve_report,
    extract_improve_candidates_from_point,
    extract_improve_candidates_from_text,
    is_candidate_applied,
    is_identity_bearing_graph_candidate,
    is_identity_bearing_value,
    load_improve_report,
    make_candidate_digest,
    persist_improve_report,
    record_candidate_applied,
)
from qdrant_memory.write_gate import evaluate_write_candidate
from qdrant_memory.tools import TOOL_SCHEMAS
from qdrant_memory.writer import ConversationWriter

logger = logging.getLogger(__name__)


def _json_error(message: str) -> str:
    return json.dumps({"error": message})


def _hash_value(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def _tool_tag_filters(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        return []
    tags: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            tag = item.strip()
            if tag:
                tags.append(tag)
    return tags


def _tool_search_filters(args: dict[str, Any]) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    tags = _tool_tag_filters(args.get("tags"))
    if tags:
        filters["tags"] = tags
    for key in ("source", "file_path", "project_path", "since", "until"):
        if key not in args:
            continue
        value = str(args.get(key) or "").strip()
        if value:
            filters[key] = value
    return filters


def _context_filter_values(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    else:
        values = [value]
    output: list[str] = []
    for raw in values:
        text = str(raw or "").strip()
        if text:
            output.append(text)
    return output


def _context_bool_filter(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    return None


def _context_status_search_kwargs(status_filters: dict[str, Any]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    raw_fact_status = status_filters.get("fact_status")
    fact_status: dict[str, Any] = raw_fact_status if isinstance(raw_fact_status, dict) else {}
    excludes = _context_filter_values(fact_status.get("exclude"))
    if excludes:
        kwargs["fact_status_exclude"] = excludes
    for key in ("stale", "requires_review", "canonical"):
        rule = status_filters.get(key)
        if not isinstance(rule, dict) or "include" not in rule:
            continue
        include = rule.get("include")
        bool_value = _context_bool_filter(include)
        if bool_value is not None:
            kwargs[key] = bool_value
    return kwargs


def _context_include_fact_history(status_filters: dict[str, Any]) -> bool:
    raw_fact_status = status_filters.get("fact_status")
    fact_status: dict[str, Any] = raw_fact_status if isinstance(raw_fact_status, dict) else {}
    included = set(_context_filter_values(fact_status.get("include")))
    review_statuses = {"disputed", "superseded", "deprecated"}
    return bool(included & review_statuses)


def _context_recipe_collections(recipe: dict[str, Any]) -> list[str]:
    collections = [str(item).strip() for item in recipe.get("collections") or []]
    collections = [item for item in collections if item in {"memory", "learning"}]
    return collections or ["memory"]


def _context_search_kwargs(collection: str, recipe: dict[str, Any], top_k: int) -> dict[str, Any]:
    raw_filters = recipe.get("filters")
    filters: dict[str, Any] = raw_filters if isinstance(raw_filters, dict) else {}
    raw_status_filters = recipe.get("status_filters")
    status_filters: dict[str, Any] = raw_status_filters if isinstance(raw_status_filters, dict) else {}
    kwargs: dict[str, Any] = {"top_k": top_k, "update_access": False}
    tags = _context_filter_values(filters.get("tags"))
    if tags:
        kwargs["tags"] = tags
    kwargs.update(_context_status_search_kwargs(status_filters))
    if collection == "learning":
        learning_type = _context_filter_values(filters.get("learning_type"))
        if learning_type:
            kwargs["learning_type"] = learning_type
        return kwargs
    memory_kind = _context_filter_values(filters.get("memory_kind"))
    if memory_kind:
        kwargs["memory_kind"] = memory_kind
    source_type = _context_filter_values(filters.get("source_type"))
    if source_type:
        kwargs["source_type"] = source_type
    kwargs["include_fact_history"] = _context_include_fact_history(status_filters)
    return kwargs


def _result_id(result: Any) -> str:
    if isinstance(result, dict):
        return str(result.get("id") or result.get("point_id") or "").strip()
    return str(getattr(result, "id", "") or "").strip()


def _merge_context_results(groups: list[list[Any]], top_k: int) -> list[Any]:
    selected: list[Any] = []
    seen: set[str] = set()
    max_len = max((len(group) for group in groups), default=0)
    limit = max(1, min(20, int(top_k)))
    for index in range(max_len):
        for group in groups:
            if index >= len(group):
                continue
            result = group[index]
            point_id = _result_id(result)
            if point_id and point_id in seen:
                continue
            if point_id:
                seen.add(point_id)
            selected.append(result)
            if len(selected) >= limit:
                return selected
    return selected


class QdrantMemoryProvider(MemoryProvider):
    def __init__(self):
        self._config = load_config()
        self._qdrant: Optional[QdrantClient] = None
        self._embeddings: Optional[EmbeddingClient] = None
        self._retriever: Optional[MemoryRetriever] = None
        self._learning_store: Optional[LearningStore] = None
        self._writer: Optional[ConversationWriter] = None
        self._executor: Optional[ThreadPoolExecutor] = None
        self._prefetch_cache: dict[str, str] = {}
        self._prefetch_lock = threading.Lock()
        self._session_id = ""
        self._hermes_home = ""
        self._profile_id = "default"
        self._platform = "cli"
        self._user_id_hash = ""
        self._chat_id_hash = ""
        self._active = False
        self._write_enabled = True
        self._pending_learning_candidates: dict[str, LearningCandidate] = {}
        self._pending_extraction_candidates: dict[str, Any] = {}
        self._reviewed_extraction_candidate_ids: set[str] = set()
        self._pending_improve_reports: dict[str, dict[str, Any]] = {}
        self._reviewed_improve_candidate_keys: set[str] = set()

    @property
    def name(self) -> str:
        return "qdrant"

    def is_available(self) -> bool:
        cfg = load_config()
        return bool(cfg.get("enabled", True))

    def initialize(self, session_id: str, **kwargs) -> None:
        try:
            from hermes_constants import get_hermes_home  # type: ignore

            default_home = str(get_hermes_home())
        except Exception:
            default_home = str(Path.home() / ".hermes")
        self._hermes_home = kwargs.get("hermes_home") or default_home
        self._config = load_config(hermes_home=self._hermes_home)
        self._session_id = session_id
        self._profile_id = str(kwargs.get("agent_identity") or "default")
        self._platform = str(kwargs.get("platform") or "cli")
        self._user_id_hash = _hash_value(str(kwargs.get("user_id") or ""))
        self._chat_id_hash = _hash_value(str(kwargs.get("chat_id") or kwargs.get("thread_id") or ""))
        agent_context = str(kwargs.get("agent_context") or "primary")
        self._write_enabled = agent_context != "subagent" or bool(self._config.get("sync_subagents"))
        if agent_context in {"cron", "flush"}:
            self._write_enabled = False
        self._active = bool(self._config.get("enabled", True))
        if not self._active:
            return
        self._qdrant = QdrantClient(self._config["qdrant_url"], api_key=self._config.get("qdrant_api_key", ""))
        self._embeddings = EmbeddingClient(
            self._config["embedding_url"],
            self._config["embedding_model"],
            query_prefix=self._config["query_prefix"],
            document_prefix=self._config["document_prefix"],
            api_key=self._config.get("embedding_api_key", ""),
            max_input_chars=self._config.get("embedding_max_input_chars", 12000),
            max_chunks=self._config.get("embedding_max_chunks", 16),
        )
        scope = self._scope_filter_values()
        self._retriever = MemoryRetriever(
            qdrant=self._qdrant,
            embeddings=self._embeddings,
            collection_name=self._config["collection_name"],
            search_candidates=self._config["search_candidates"],
            decay_rate=self._config["decay_rate"],
            scope=scope,
            min_raw_score=self._config.get("min_raw_score", 0.0),
            min_final_score=self._config.get("min_final_score", 0.0),
            ranking_policy=RankingPolicy(
                enabled=bool(self._config.get("provenance_ranking_enabled", True))
            ),
        )
        self._writer = ConversationWriter(
            qdrant=self._qdrant,
            embeddings=self._embeddings,
            collection_name=self._config["collection_name"],
            profile_id=self._profile_id,
            platform=self._platform,
            session_id=self._session_id,
            user_id_hash=self._user_id_hash,
            chat_id_hash=self._chat_id_hash,
            model=self._config["embedding_model"],
        )
        self._learning_store = LearningStore(
            qdrant=self._qdrant,
            embeddings=self._embeddings,
            collection_name=self._config["learning_collection_name"],
            profile_id=self._profile_id,
            platform=self._platform,
            session_id=self._session_id,
            user_id_hash=self._user_id_hash,
            chat_id_hash=self._chat_id_hash,
            model=self._config["embedding_model"],
            scope=scope,
            search_candidates=self._config["search_candidates"],
            decay_rate=self._config["decay_rate"],
            min_raw_score=self._config.get("min_raw_score", 0.0),
            min_final_score=self._config.get("min_final_score", 0.0),
        )
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="qdrant-memory")
        try:
            self._qdrant.ensure_collection(self._config["collection_name"], self._config["vector_size"], self._config["distance"])
            self._qdrant.ensure_collection(self._config["learning_collection_name"], self._config["vector_size"], self._config["distance"])
        except Exception:
            logger.debug("Qdrant collection setup failed", exc_info=True)

    def _scope_filter_values(self) -> dict[str, str]:
        mode = str(self._config.get("scope_mode") or "profile")
        scope = {"profile_id": self._profile_id}
        if mode in {"user", "chat"} and self._user_id_hash:
            scope["user_id_hash"] = self._user_id_hash
        if mode == "chat" and self._chat_id_hash:
            scope["chat_id_hash"] = self._chat_id_hash
        if mode == "global":
            return {}
        return scope

    def system_prompt_block(self) -> str:
        if not self._active:
            return ""
        return (
            "# Qdrant Memory\n"
            "Active local long-term semantic memory. Use qdrant_memory_context, qdrant_memory_search, "
            "qdrant_memory_inspect, qdrant_memory_trace, qdrant_memory_expand, "
            "qdrant_memory_source_status, qdrant_memory_store, qdrant_memory_index, "
            "qdrant_learning_search, qdrant_learning_store, and qdrant_memory_status "
            "for explicit memory operations."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._active or not self._config.get("auto_recall") or not query.strip() or not self._retriever:
            return ""
        sid = session_id or self._session_id or "default"
        with self._prefetch_lock:
            cached = self._prefetch_cache.pop(sid, "")
        if cached:
            return cached
        try:
            chunks = self._retriever.search(query, top_k=int(self._config["auto_recall_top_k"]))
            return format_for_prompt(chunks, int(self._config["display_tokens"]))
        except Exception:
            logger.debug("Qdrant prefetch failed", exc_info=True)
            return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._active or not self._executor or not self._retriever or not self._config.get("auto_recall") or not query.strip():
            return
        sid = session_id or self._session_id or "default"

        def _run() -> None:
            try:
                chunks = self._retriever.search(query, top_k=int(self._config["auto_recall_top_k"]))
                formatted = format_for_prompt(chunks, int(self._config["display_tokens"]))
                with self._prefetch_lock:
                    self._prefetch_cache[sid] = formatted
            except Exception:
                logger.debug("Qdrant queued prefetch failed", exc_info=True)

        self._executor.submit(_run)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        if not self._active or not self._write_enabled or not self._config.get("sync_turns") or not self._writer:
            return
        if session_id:
            self._writer.session_id = session_id

        def _run() -> None:
            try:
                self._writer.store_turn(user_content, assistant_content)
            except Exception:
                logger.debug("Qdrant sync_turn failed", exc_info=True)

        if self._executor:
            self._executor.submit(_run)
        else:
            _run()

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        self._session_id = new_session_id or ""
        if self._writer:
            self._writer.session_id = self._session_id
        if self._learning_store:
            self._learning_store.session_id = self._session_id
        if reset:
            with self._prefetch_lock:
                self._prefetch_cache.clear()
            self._pending_learning_candidates.clear()
            self._pending_extraction_candidates.clear()
            self._reviewed_extraction_candidate_ids.clear()
            self._pending_improve_reports.clear()
            self._reviewed_improve_candidate_keys.clear()

    def _auto_learning_enabled(self) -> bool:
        return bool(self._config.get("learning_enabled", True)) and bool(self._config.get("learning_auto_extract_enabled", False))

    def _candidate_limits(self) -> tuple[float, int]:
        try:
            min_confidence = float(self._config.get("learning_auto_extract_min_confidence", 0.85))
        except Exception:
            min_confidence = 0.85
        try:
            max_candidates = int(self._config.get("learning_auto_extract_max_candidates_per_session", 3))
        except Exception:
            max_candidates = 3
        return min_confidence, max(1, max_candidates)

    def _is_semantic_duplicate_learning_candidate(self, candidate: LearningCandidate) -> bool:
        if not self._config.get("learning_auto_extract_semantic_dedupe_enabled", True):
            return False
        store = self._ensure_learning_store()
        if not store:
            return False
        try:
            threshold = float(self._config.get("learning_auto_extract_semantic_dedupe_threshold", 0.9))
        except Exception:
            threshold = 0.9
        try:
            top_k = int(self._config.get("learning_auto_extract_semantic_dedupe_top_k", 3))
        except Exception:
            top_k = 3
        query = "\n".join(part for part in [candidate.lesson, candidate.trigger, candidate.correction] if part)
        try:
            return bool(
                store.find_semantic_duplicate(
                    query,
                    learning_type=candidate.learning_type,
                    threshold=threshold,
                    top_k=top_k,
                )
            )
        except Exception:
            logger.debug("Qdrant learning semantic dedupe failed open", exc_info=True)
            return False

    def _collect_learning_candidates(self, messages: list[Any], *, source_hook: str) -> list[LearningCandidate]:
        if not self._auto_learning_enabled():
            return []
        mode = str(self._config.get("learning_auto_extract_mode") or "preview")
        if mode not in {"preview", "store"}:
            return []
        min_confidence, max_candidates = self._candidate_limits()
        remaining = max_candidates - len(self._pending_learning_candidates)
        if remaining <= 0:
            return []
        candidates = extract_learning_candidates_from_messages(
            messages,
            source_hook=source_hook,
            min_confidence=min_confidence,
            max_candidates=remaining,
        )
        if self._config.get("learning_auto_extract_require_evidence", True):
            candidates = [candidate for candidate in candidates if candidate.evidence.strip()]
        accepted: list[LearningCandidate] = []
        for candidate in candidates:
            if len(self._pending_learning_candidates) >= max_candidates:
                break
            if candidate.candidate_id in self._pending_learning_candidates:
                continue
            if self._is_semantic_duplicate_learning_candidate(candidate):
                continue
            self._pending_learning_candidates[candidate.candidate_id] = candidate
            accepted.append(candidate)
        if mode == "store":
            # Intentionally conservative: M7.1 still routes through approval/dry-run.
            logger.info("qdrant learning auto_extract_mode=store is not active yet; candidates kept pending")
        return accepted

    def _source_extraction_enabled(self) -> bool:
        return bool(self._config.get("source_extraction_enabled", False))

    def _source_extraction_limits(self) -> tuple[float, int]:
        try:
            min_confidence = float(self._config.get("source_extraction_min_confidence", 0.0))
        except Exception:
            min_confidence = 0.0
        try:
            max_candidates = int(self._config.get("source_extraction_max_candidates_per_session", 8))
        except Exception:
            max_candidates = 8
        return max(0.0, min(1.0, min_confidence)), max(1, max_candidates)

    def _collect_source_extraction_candidates(self, messages: list[Any], *, source_hook: str) -> list[Any]:
        if not self._source_extraction_enabled():
            return []
        mode = str(self._config.get("source_extraction_mode") or "preview")
        if mode not in {"preview", "store"}:
            return []
        min_confidence, max_candidates = self._source_extraction_limits()
        remaining = max_candidates - len(self._pending_extraction_candidates)
        if remaining <= 0:
            return []
        source_uri = f"session://{self._session_id or 'current'}/{source_hook}"
        candidates = extract_source_candidates_from_messages(
            messages,
            source_uri=source_uri,
            lifecycle_id=f"{self._session_id or 'current'}:{source_hook}",
            min_confidence=min_confidence,
            max_candidates=remaining,
        )
        accepted: list[Any] = []
        for candidate in candidates:
            if len(self._pending_extraction_candidates) >= max_candidates:
                break
            if candidate.candidate_id in self._pending_extraction_candidates:
                continue
            decision = evaluate_source_extraction_candidate(candidate)
            if decision.decision == "reject":
                continue
            self._pending_extraction_candidates[candidate.candidate_id] = candidate
            accepted.append(candidate)
        if mode == "store":
            # Source extraction never auto-mutates; exact candidate approval remains required.
            logger.info("qdrant source_extraction_mode=store is not active; candidates kept pending")
        return accepted

    def on_pre_compress(self, messages: list[Any]) -> str:
        learning_candidates = self._collect_learning_candidates(messages, source_hook="on_pre_compress")
        source_candidates = self._collect_source_extraction_candidates(messages, source_hook="on_pre_compress")
        blocks: list[str] = []
        if learning_candidates:
            lines = [
                "# Qdrant Learning Candidates",
                "Potential procedural lessons detected but not stored. Review with qdrant_learning_preview and approve explicitly.",
            ]
            for candidate in learning_candidates:
                lines.append(f"- {candidate.candidate_id} [{candidate.learning_type}] {candidate.lesson[:220]}")
            blocks.append("\n".join(lines))
        if source_candidates:
            lines = [
                "# Qdrant Source Extraction Candidates",
                "Potential source-backed memory/assertion candidates detected but not stored. Review with qdrant_memory_extraction_preview and approve explicitly.",
            ]
            for candidate in source_candidates:
                text = str(candidate.proposed_payload.get("text") or "")[:220]
                lines.append(f"- {candidate.candidate_id} [{candidate.candidate_type}] {text}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)

    def on_session_end(self, messages: list[Any]) -> None:
        self._collect_learning_candidates(messages, source_hook="on_session_end")
        self._collect_source_extraction_candidates(messages, source_hook="on_session_end")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def _tool_status(self) -> str:
        qdrant_ok = False
        embedding_ok = False
        collections: list[str] = []
        point_count = None
        learning_point_count = None
        collection_vector_size = None
        learning_collection_vector_size = None
        if self._qdrant:
            qdrant_ok = self._qdrant.health()
            try:
                collections = self._qdrant.get_collections()
            except Exception:
                collections = []
            try:
                point_count = self._qdrant.count(self._config["collection_name"])
            except Exception:
                point_count = None
            try:
                learning_point_count = self._qdrant.count(self._config["learning_collection_name"])
            except Exception:
                learning_point_count = None
            try:
                collection_vector_size = self._qdrant.collection_vector_size(self._config["collection_name"])
            except Exception:
                collection_vector_size = None
            try:
                learning_collection_vector_size = self._qdrant.collection_vector_size(self._config["learning_collection_name"])
            except Exception:
                learning_collection_vector_size = None
        if self._embeddings:
            embedding_ok = self._embeddings.health()
        payload = {
            "provider": self.name,
            "active": self._active,
            "qdrant_url": self._config["qdrant_url"],
            "qdrant_ok": qdrant_ok,
            "embedding_url": self._config["embedding_url"],
            "embedding_model": self._config["embedding_model"],
            "embedding_ok": embedding_ok,
            "collection_name": self._config["collection_name"],
            "collection_exists": self._config["collection_name"] in collections,
            "vector_size": self._config["vector_size"],
            "collection_vector_size": collection_vector_size,
            "collection_vector_size_matches": collection_vector_size == self._config["vector_size"] if collection_vector_size is not None else False,
            "point_count": point_count,
            "learning_collection_name": self._config["learning_collection_name"],
            "learning_collection_exists": self._config["learning_collection_name"] in collections,
            "learning_collection_vector_size": learning_collection_vector_size,
            "learning_collection_vector_size_matches": learning_collection_vector_size == self._config["vector_size"] if learning_collection_vector_size is not None else False,
            "learning_point_count": learning_point_count,
            "learning_enabled": self._config.get("learning_enabled", True),
            "learning_auto_extract_enabled": self._config.get("learning_auto_extract_enabled", False),
            "learning_auto_extract_mode": self._config.get("learning_auto_extract_mode", "preview"),
            "pending_learning_candidate_count": len(self._pending_learning_candidates),
            "source_extraction_enabled": self._config.get("source_extraction_enabled", False),
            "source_extraction_mode": self._config.get("source_extraction_mode", "preview"),
            "pending_extraction_candidate_count": len(self._pending_extraction_candidates),
            "consolidation_enabled": self._config.get("consolidation_enabled", False),
            "consolidation_persist_reports": self._config.get("consolidation_persist_reports", True),
            "consolidation_apply_enabled": True,
            "consolidation_supported_actions": ["merge", "delete", "quarantine", "promote_to_skill", "draft_review"],
            "reconsolidation_enabled": self._config.get("reconsolidation_enabled", False),
            "reconsolidation_report_only": self._config.get("reconsolidation_report_only", True),
            "reconsolidation_supported_actions": ["draft_review"],
            "consolidation_report_only": False,
            "auto_recall": self._config["auto_recall"],
            "sync_turns": self._config["sync_turns"],
        }
        return json.dumps(payload)

    def _tool_store(self, args: dict) -> str:
        if not self._writer:
            return _json_error("Qdrant memory provider is not initialized")
        text = str(args.get("text") or "").strip()
        if not text:
            return _json_error("text is required")
        source_type = str(args.get("source_type") or "manual")
        try:
            importance = max(1, min(10, int(args.get("importance", 5))))
        except Exception:
            importance = 5
        tags = args.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags]
        dry_run = parse_bool_arg(args.get("dry_run"), default=True)
        approve = parse_bool_arg(args.get("approve"), default=False)
        duplicate_preview = parse_bool_arg(args.get("duplicate_preview"), default=False)
        try:
            raw_threshold = args.get("duplicate_threshold")
            if raw_threshold is None:
                raw_threshold = self._config.get("manual_store_duplicate_threshold", 0.92)
            duplicate_threshold = max(0.0, min(1.0, float(raw_threshold)))
        except Exception:
            duplicate_threshold = 0.92
        try:
            raw_top_k = args.get("duplicate_top_k")
            if raw_top_k is None:
                raw_top_k = self._config.get("manual_store_duplicate_top_k", 3)
            duplicate_top_k = max(1, min(20, int(raw_top_k)))
        except Exception:
            duplicate_top_k = 3
        try:
            preview = self._writer.preview_text(text, source_type=source_type, importance=importance, tags=tags)
            if not preview.get("id"):
                return _json_error("text is required")
            duplicate = None
            if duplicate_preview:
                duplicate = self._writer.find_semantic_duplicate(
                    text,
                    source_type=source_type,
                    threshold=duplicate_threshold,
                    top_k=duplicate_top_k,
                )
            base = {
                "id": preview.get("id"),
                "source_type": source_type,
                "collection_name": self._config["collection_name"],
                "duplicate_preview": duplicate_preview,
                "duplicate_found": bool(duplicate),
            }
            if duplicate:
                base["duplicate"] = duplicate
            write_decision = evaluate_write_candidate(
                text=text,
                target="memory",
                source_type=source_type,
                confidence=1.0,
                duplicate=duplicate,
                metadata={"importance": importance},
            )
            base["write_decision"] = write_decision.to_dict()
            if dry_run:
                return json.dumps({"dry_run": True, "saved": False, "would_store": write_decision.decision == "store", **base})
            if not approve:
                return _json_error("approve=true is required when dry_run=false")
            if write_decision.decision == "reject":
                return _json_error("write gate rejected memory candidate")
            if write_decision.decision == "skip":
                return json.dumps({"dry_run": False, "saved": False, "would_store": False, **base})
            if write_decision.decision != "store":
                return _json_error("write gate requires review before storing memory candidate")
            point_id = self._writer.store_text(text, source_type=source_type, importance=importance, tags=tags)
            return json.dumps({"dry_run": False, "saved": bool(point_id), "id": point_id, "source_type": source_type, "collection_name": self._config["collection_name"], "write_decision": write_decision.to_dict()})
        except Exception as exc:
            return _json_error(f"Failed to store memory: {exc}")

    def _tool_context(self, args: dict[str, Any]) -> str:
        template = str(args.get("template") or "source_backed_answer").strip()
        topic = str(args.get("topic") or "").strip()
        if not topic:
            return _json_error("topic is required")
        try:
            recipe = get_recipe(template)
            default_top_k = default_context_top_k(template)
            top_k = max(1, min(20, int(args.get("top_k") or default_top_k)))
        except (ContextTemplateError, KeyError) as exc:
            return _json_error(str(exc).strip("'"))
        except Exception:
            top_k = 6
            recipe = get_recipe(template)
        collections = _context_recipe_collections(recipe)
        if "memory" in collections and not self._retriever:
            return _json_error("Qdrant memory provider is not initialized")
        try:
            result_groups: list[list[Any]] = []
            for collection in collections:
                search_kwargs = _context_search_kwargs(collection, recipe, top_k)
                if collection == "learning":
                    if not self._config.get("learning_enabled", True):
                        continue
                    store = self._ensure_learning_store()
                    if not store:
                        continue
                    result_groups.append(list(store.search(topic, **search_kwargs)))
                    continue
                if self._retriever:
                    result_groups.append(list(self._retriever.search(topic, **search_kwargs)))
            chunks = _merge_context_results(result_groups, top_k)
            packet = build_context_packet(template=template, topic=topic, results=chunks)
            return json.dumps(packet)
        except ContextTemplateError as exc:
            return _json_error(str(exc))
        except Exception as exc:
            return _json_error(f"Context packet failed: {exc}")

    def _tool_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return _json_error("query is required")
        collection = str(args.get("collection") or "memory").strip().lower()
        if collection not in {"memory", "learning"}:
            return _json_error("collection must be one of: memory, learning")
        if collection == "learning":
            routed_args = dict(args)
            routed_args.pop("collection", None)
            routed_args.pop("source_type", None)
            return self._tool_learning_search(routed_args)
        if not self._retriever:
            return _json_error("Qdrant memory provider is not initialized")
        try:
            top_k = max(1, min(20, int(args.get("top_k", 5))))
        except Exception:
            top_k = 5
        include_metadata = bool(args.get("include_metadata", False))
        include_fact_history = parse_bool_arg(args.get("include_fact_history"), default=False)
        source_type = args.get("source_type") or None
        try:
            chunks = self._retriever.search(
                query,
                top_k=top_k,
                source_type=source_type,
                include_fact_history=include_fact_history,
                **_tool_search_filters(args),
            )
            results = []
            for chunk in chunks:
                item: dict[str, Any] = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "score": round(chunk.final_score, 6),
                    "vector_score": round(chunk.qdrant_score, 6),
                }
                if include_metadata:
                    item["metadata"] = chunk.payload
                    item["ranking"] = chunk.ranking_debug
                results.append(item)
            return json.dumps({"results": results, "count": len(results)})
        except Exception as exc:
            return _json_error(f"Search failed: {exc}")

    def _tool_index(self, args: dict) -> str:
        configured_paths = self._config.get("index_dirs") or []
        paths = args.get("paths") or configured_paths
        if isinstance(paths, str):
            paths = [paths]
        if not isinstance(paths, list) or not [p for p in paths if str(p).strip()]:
            return _json_error("paths are required when qdrant_memory.index_dirs is empty")
        if "dry_run" in args:
            dry_run = bool(args.get("dry_run"))
        else:
            dry_run = bool(self._config.get("index_dry_run_default", True))
        force = bool(args.get("force", False))
        max_files = args.get("max_files") or None
        try:
            max_files = int(max_files) if max_files is not None else None
        except Exception:
            max_files = None
        indexer = FileIndexer(
            qdrant=self._qdrant,
            embeddings=self._embeddings,
            collection_name=self._config["collection_name"],
            config=self._config,
            profile_id=self._profile_id,
            platform=self._platform,
            session_id=self._session_id,
            user_id_hash=self._user_id_hash,
            chat_id_hash=self._chat_id_hash,
            model=self._config.get("embedding_model", ""),
        )
        try:
            summary = indexer.index([str(p) for p in paths if str(p).strip()], dry_run=dry_run, force=force, max_files=max_files)
            return json.dumps(summary)
        except Exception as exc:
            return _json_error(f"Index failed: {exc}")

    def _tool_forget(self, args: dict) -> str:
        ids = args.get("ids") or []
        if isinstance(ids, str):
            ids = [ids]
        ids = [str(item).strip() for item in ids if str(item).strip()] if isinstance(ids, list) else []
        if not ids:
            return _json_error("ids are required")
        dry_run = bool(args.get("dry_run", True))
        if dry_run:
            return json.dumps({"dry_run": True, "ids": ids, "deleted": 0})
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        try:
            self._qdrant.delete_ids(self._config["collection_name"], ids)
            return json.dumps({"dry_run": False, "ids": ids, "deleted": len(ids)})
        except Exception as exc:
            return _json_error(f"Forget failed: {exc}")

    def _collection_name_from_tool_args(self, args: dict[str, Any]) -> tuple[str, str] | None:
        collection = str(args.get("collection") or "memory").strip().lower()
        if collection not in {"memory", "learning"}:
            return None
        collection_name = self._config["learning_collection_name"] if collection == "learning" else self._config["collection_name"]
        return collection, collection_name

    def _tool_inspect(self, args: dict[str, Any]) -> str:
        point_id = str(args.get("point_id") or "").strip()
        if not point_id:
            return _json_error("point_id is required")
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        collection_pair = self._collection_name_from_tool_args(args)
        if not collection_pair:
            return _json_error("collection must be one of: memory, learning")
        collection, collection_name = collection_pair
        try:
            payload = inspect_point(self._qdrant, collection_name, point_id, collection=collection)
            if not payload.get("found"):
                return _json_error("Point not found")
            return json.dumps(payload)
        except Exception as exc:
            return _json_error(f"Inspect failed: {exc}")

    def _tool_trace(self, args: dict[str, Any]) -> str:
        point_id = str(args.get("point_id") or "").strip()
        if not point_id:
            return _json_error("point_id is required")
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        collection_pair = self._collection_name_from_tool_args(args)
        if not collection_pair:
            return _json_error("collection must be one of: memory, learning")
        collection, collection_name = collection_pair
        direction = str(args.get("direction") or "upstream").strip().lower()
        if direction not in {"upstream", "downstream", "both"}:
            return _json_error("direction must be one of: upstream, downstream, both")
        try:
            payload = trace_point(self._qdrant, collection_name, point_id, collection=collection, direction=direction, config=self._config)
            if not payload.get("found"):
                return _json_error("Point not found")
            return json.dumps(payload)
        except Exception as exc:
            return _json_error(f"Trace failed: {exc}")

    def _tool_expand(self, args: dict[str, Any]) -> str:
        point_id = str(args.get("point_id") or "").strip()
        if not point_id:
            return _json_error("point_id is required")
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        collection_pair = self._collection_name_from_tool_args(args)
        if not collection_pair:
            return _json_error("collection must be one of: memory, learning")
        collection, collection_name = collection_pair
        mode = str(args.get("mode") or "excerpt").strip().lower()
        if mode not in {"excerpt", "source", "neighbors"}:
            return _json_error("mode must be one of: excerpt, source, neighbors")
        try:
            max_chars = max(1, min(100000, int(args.get("max_chars") or 8000)))
        except Exception:
            max_chars = 8000
        try:
            payload = expand_point(self._qdrant, collection_name, point_id, collection=collection, mode=mode, max_chars=max_chars, config=self._config)
            if not payload.get("found", True):
                return _json_error("Point not found")
            return json.dumps(payload)
        except Exception as exc:
            return _json_error(f"Expand failed: {exc}")

    def _tool_source_status(self, args: dict[str, Any]) -> str:
        point_id = str(args.get("point_id") or "").strip()
        if not point_id:
            return _json_error("point_id is required")
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        collection_pair = self._collection_name_from_tool_args(args)
        if not collection_pair:
            return _json_error("collection must be one of: memory, learning")
        collection, collection_name = collection_pair
        try:
            payload = source_status_for_point(self._qdrant, collection_name, point_id, collection=collection, config=self._config)
            if not payload.get("found", True):
                return _json_error("Point not found")
            return json.dumps(payload)
        except Exception as exc:
            return _json_error(f"Source status failed: {exc}")

    def _tool_consolidate(self, args: dict) -> str:
        if not parse_bool_arg(args.get("dry_run", True), default=True):
            return _json_error("qdrant_memory_consolidate is report-only; use qdrant_memory_consolidation_apply with proposal_id for live actions")
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        scope = str(args.get("scope") or "both").strip().lower()
        if scope not in {"memory", "learning", "both"}:
            return _json_error("scope must be one of: memory, learning, both")
        try:
            max_points = max(1, min(1000, int(args.get("max_points") or self._config.get("consolidation_report_max_points", 200))))
        except Exception:
            max_points = 200
        try:
            max_groups = max(1, min(100, int(args.get("max_groups") or self._config.get("consolidation_report_max_groups", 20))))
        except Exception:
            max_groups = 20
        include_examples = bool(args.get("include_examples", False))
        include_reconsolidation = parse_bool_arg(args.get("include_reconsolidation"), default=bool(self._config.get("reconsolidation_include_by_default", False)))
        try:
            reconsolidation_max_candidates = max(1, min(100, int(args.get("reconsolidation_max_candidates") or self._config.get("reconsolidation_max_candidates", 10))))
        except Exception:
            reconsolidation_max_candidates = 10
        base_scope = self._scope_filter_values()
        memory_points = []
        learning_points = []
        try:
            if scope in {"memory", "both"}:
                raw_memory = self._qdrant.scroll_by_filter(
                    self._config["collection_name"],
                    make_filter(base_scope),
                    limit=max_points,
                    with_payload=True,
                    with_vector=False,
                    max_total=max_points,
                )
                memory_points = points_from_qdrant(raw_memory, collection_name=self._config["collection_name"])
            if scope in {"learning", "both"}:
                raw_learning = self._qdrant.scroll_by_filter(
                    self._config["learning_collection_name"],
                    make_filter(base_scope, source_type="learning"),
                    limit=max_points,
                    with_payload=True,
                    with_vector=False,
                    max_total=max_points,
                )
                learning_points = points_from_qdrant(raw_learning, collection_name=self._config["learning_collection_name"])
            report = build_consolidation_report(
                memory_points=memory_points,
                learning_points=learning_points,
                collection_name=self._config["collection_name"],
                learning_collection_name=self._config["learning_collection_name"],
                scope=scope,
                include_examples=include_examples,
                max_groups=max_groups,
                stale_days=int(self._config.get("consolidation_stale_days", 90)),
                min_importance_for_keep=int(self._config.get("consolidation_min_importance_for_keep", 4)),
                duplicate_threshold=float(self._config.get("consolidation_duplicate_threshold", 0.92)),
                consolidation_enabled=bool(self._config.get("consolidation_enabled", False)),
                reconsolidation_enabled=bool(self._config.get("reconsolidation_enabled", False)),
                include_reconsolidation=include_reconsolidation,
                reconsolidation_max_candidates=reconsolidation_max_candidates,
                reconsolidation_min_confidence=float(self._config.get("reconsolidation_min_confidence", 0.6)),
            )
            report.update(
                {
                    "profile_id": self._profile_id,
                    "platform": self._platform,
                    "session_id": self._session_id,
                    "user_id_hash": self._user_id_hash,
                    "chat_id_hash": self._chat_id_hash,
                }
            )
            if parse_bool_arg(args.get("persist"), default=bool(self._config.get("consolidation_persist_reports", True))):
                report = persist_consolidation_report(
                    report,
                    hermes_home=self._hermes_home,
                    configured_dir=str(self._config.get("consolidation_artifact_dir") or ""),
                )
            else:
                report["persisted"] = False
            return json.dumps(report)
        except Exception as exc:
            return _json_error(f"Consolidation report failed: {exc}")

    def _collection_for_proposal(self, proposal: dict[str, Any]) -> str:
        collection = str(proposal.get("collection_name") or "")
        if collection in {"memory", self._config.get("collection_name")}:
            return self._config["collection_name"]
        if collection in {"learning", "learnings", self._config.get("learning_collection_name")}:
            return self._config["learning_collection_name"]
        return collection

    def _retrieve_consolidation_points(self, collection_name: str, ids: list[str]) -> list[Any]:
        if not self._qdrant:
            return []
        raw = self._qdrant.retrieve(collection_name, ids, with_payload=True, with_vector=False)
        return points_from_qdrant(raw, collection_name=collection_name)

    def _proposal_write_decision(self, proposal: dict[str, Any], action: str, points: list[Any]) -> Any:
        proposal_type = str(proposal.get("proposal_type") or "")
        confidence = proposal.get("confidence")
        if action == "draft_review":
            return evaluate_write_candidate(
                text="\n".join(str(getattr(point, "text", "") or "") for point in points),
                derivation_type="reconsolidation",
                confidence=confidence,
                metadata={"proposal_type": proposal_type},
            )
        if action == "promote_to_skill" and points:
            point = points[0]
            payload = getattr(point, "payload", {}) or {}
            return evaluate_write_candidate(
                text=str(getattr(point, "text", "") or ""),
                target="learning",
                source_type=str(payload.get("source_type") or "learning"),
                confidence=payload.get("confidence", confidence),
                promote_to_skill_candidate=True,
                metadata={"importance": payload.get("importance"), "proposal_type": proposal_type},
            )
        return None

    def _proposal_apply_plan(self, report: dict[str, Any], proposal: dict[str, Any], action: str, points: list[Any]) -> dict[str, Any]:
        affected_ids = [str(item) for item in proposal.get("affected_ids") or [] if str(item)]
        plan = {
            "report_id": report.get("report_id"),
            "proposal_id": proposal.get("proposal_id"),
            "proposal_type": proposal.get("proposal_type"),
            "action": action,
            "affected_ids": affected_ids,
            "collection_name": self._collection_for_proposal(proposal),
        }
        for key in (
            "risk",
            "confidence",
            "source_snippets",
            "proposed_status_changes",
            "manual_review_required",
            "manual_review_reason",
            "candidate_statement",
            "fact_key",
            "current_or_newer_id",
            "superseded_candidate_ids",
        ):
            if key in proposal:
                plan[key] = proposal[key]
        if action == "merge" and points:
            canonical = self._select_canonical_point(points)
            plan.update({"canonical_id": canonical.id, "delete_ids": [p.id for p in points if p.id != canonical.id]})
        elif action == "delete":
            plan.update({"delete_ids": affected_ids})
        elif action == "quarantine":
            plan.update({"quarantine_ids": affected_ids})
        elif action == "promote_to_skill" and points:
            write_decision = self._proposal_write_decision(proposal, action, points)
            draft = proposal_draft_metadata(report=report, proposal=proposal, points=[points[0]], hermes_home=self._hermes_home, config=self._config, write_decision=write_decision)
            plan.update({"proposal_draft_path": draft["path"], "skill_draft_path": draft["path"], "write_decision": write_decision.to_dict() if write_decision else None})
        elif action == "draft_review" and points:
            write_decision = self._proposal_write_decision(proposal, action, points)
            draft = proposal_draft_metadata(report=report, proposal=proposal, points=points, hermes_home=self._hermes_home, config=self._config, write_decision=write_decision)
            plan.update({"proposal_draft_path": draft["path"], "reconsolidation_draft_path": draft["path"], "write_decision": write_decision.to_dict() if write_decision else None})
        return plan

    def _select_canonical_point(self, points: list[Any]) -> Any:
        def key(point: Any) -> tuple[float, float, str, str]:
            payload = point.payload or {}
            try:
                importance = float(payload.get("importance", 5))
            except Exception:
                importance = 5.0
            try:
                confidence = float(payload.get("confidence", 0.0))
            except Exception:
                confidence = 0.0
            return (importance, confidence, str(payload.get("created_at") or ""), point.id)

        return max(points, key=key)

    def _tool_consolidation_apply(self, args: dict) -> str:
        proposal_id = str(args.get("proposal_id") or "").strip()
        if not proposal_id:
            return _json_error("proposal_id is required")
        report_id = str(args.get("report_id") or "").strip()
        if not report_id:
            return _json_error("report_id is required")
        if not self._qdrant:
            return _json_error("Qdrant memory provider is not initialized")
        dry_run = parse_bool_arg(args.get("dry_run"), default=bool(self._config.get("consolidation_apply_dry_run_default", True)))
        if not dry_run and not parse_bool_arg(args.get("approve"), default=False):
            return _json_error("approve=true is required when dry_run=false")
        if not dry_run and not str(args.get("action") or "").strip():
            return _json_error("action is required when dry_run=false")
        try:
            report = load_consolidation_report(report_id, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
            if str(report.get("profile_id") or self._profile_id) != self._profile_id:
                return _json_error("proposal belongs to a different profile scope")
            proposal = find_proposal(report, proposal_id)
            proposal_type = str(proposal.get("proposal_type") or "")
            expected_action = expected_action_for_proposal(proposal_type)
            if not expected_action:
                return _json_error("proposal requires manual review and cannot be applied automatically")
            action = str(args.get("action") or expected_action).strip()
            allowed_actions = {expected_action}
            if proposal_type == "stale_low_value":
                allowed_actions.add("quarantine")
            if action not in allowed_actions:
                return _json_error("action mismatch for proposal type")
            affected_ids = [str(item) for item in proposal.get("affected_ids") or [] if str(item)]
            if not affected_ids:
                return _json_error("proposal has no explicit affected_ids")
            collection_name = self._collection_for_proposal(proposal)
            points = self._retrieve_consolidation_points(collection_name, affected_ids)
            if len(points) != len(set(affected_ids)):
                return _json_error("affected point missing; rerun consolidation")
            if str(proposal.get("preauthorized_policy") or "").startswith("guarded-auto:") and action in {"merge", "delete", "quarantine"}:
                if any(contains_secret(p.text) or contains_secret(json.dumps(p.payload or {}, sort_keys=True, default=str)) for p in points):
                    return _json_error("secret-bearing point requires manual review")
                if any(_point_requires_manual_review(p) for p in points):
                    return _json_error("profile or fact-like memory requires manual review")
            plan = self._proposal_apply_plan(report, proposal, action, points)
            if dry_run:
                return json.dumps({"dry_run": True, "would_apply": True, **plan})
            pre_apply: dict[str, Any] = {}
            if parse_bool_arg(args.get("backup_first"), default=False):
                backup = create_backup(self._qdrant, self._config, hermes_home=self._hermes_home, scope="both")
                pre_apply["pre_apply_backup_id"] = backup.get("backup_id")
            if action == "draft_review":
                write_decision = evaluate_write_candidate(
                    text="\n".join(point.text for point in points),
                    derivation_type="reconsolidation",
                    confidence=proposal.get("confidence"),
                    metadata={"proposal_type": proposal_type},
                )
                draft = write_proposal_draft(
                    report=report,
                    proposal=proposal,
                    points=points,
                    hermes_home=self._hermes_home,
                    config=self._config,
                    write_decision=write_decision,
                )
                draft_path = str(draft["path"])
                record = persist_application_record({"applied": True, **plan, **pre_apply, "proposal_draft_path": draft_path, "reconsolidation_draft_path": draft_path, "write_decision": write_decision.to_dict()}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, **pre_apply, "proposal_draft_path": draft_path, "reconsolidation_draft_path": draft_path, "write_decision": write_decision.to_dict(), "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "delete":
                self._qdrant.delete_ids(collection_name, affected_ids)
                record = persist_application_record({"applied": True, **plan, **pre_apply, "deleted_ids": affected_ids}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, **pre_apply, "deleted_ids": affected_ids, "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "quarantine":
                try:
                    quarantine_days = max(1, min(365, int(args.get("quarantine_days", self._config.get("guarded_auto_quarantine_days", 30)))))
                except Exception:
                    quarantine_days = 30
                quarantined_at = datetime.utcnow().isoformat() + "Z"
                quarantine_until = (datetime.utcnow() + timedelta(days=quarantine_days)).isoformat() + "Z"
                payload_update = {
                    "consolidation_quarantined": True,
                    "consolidation_quarantine_reason": "guarded-auto stale_low_value",
                    "consolidation_quarantined_at": quarantined_at,
                    "consolidation_quarantine_until": quarantine_until,
                    "consolidation_proposal_id": proposal_id,
                    "consolidation_report_id": report_id,
                }
                for point_id in affected_ids:
                    self._qdrant.update_payload(collection_name, point_id, payload_update)
                record = persist_application_record({"applied": True, **plan, **pre_apply, "quarantined_ids": affected_ids, "quarantine_until": quarantine_until}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, **pre_apply, "quarantined_ids": affected_ids, "quarantine_until": quarantine_until, "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "merge":
                if len(points) < 2:
                    return _json_error("merge requires at least two affected points")
                if any(contains_secret(p.text) for p in points):
                    return _json_error("secret-bearing duplicate cluster requires manual review")
                canonical = self._select_canonical_point(points)
                delete_ids = [p.id for p in points if p.id != canonical.id]
                payload_update = {
                    "consolidated_from": delete_ids,
                    "consolidation_proposal_id": proposal_id,
                    "consolidation_report_id": report_id,
                    "consolidated_at": datetime.utcnow().isoformat() + "Z",
                    "duplicate_count": len(points),
                }
                self._qdrant.update_payload(collection_name, canonical.id, payload_update)
                if delete_ids:
                    self._qdrant.delete_ids(collection_name, delete_ids)
                record = persist_application_record({"applied": True, **plan, **pre_apply, "canonical_id": canonical.id, "deleted_ids": delete_ids}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, **pre_apply, "canonical_id": canonical.id, "deleted_ids": delete_ids, "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "promote_to_skill":
                if collection_name != self._config["learning_collection_name"]:
                    return _json_error("promote_to_skill requires a learning collection proposal")
                point = points[0]
                if contains_secret(point.text) or contains_secret(json.dumps(point.payload or {}, sort_keys=True, default=str)):
                    return _json_error("secret-bearing learning requires manual review")
                write_decision = evaluate_write_candidate(
                    text=point.text,
                    target="learning",
                    source_type=str((point.payload or {}).get("source_type") or "learning"),
                    confidence=(point.payload or {}).get("confidence", proposal.get("confidence")),
                    promote_to_skill_candidate=True,
                    metadata={"importance": (point.payload or {}).get("importance"), "proposal_type": proposal_type},
                )
                if write_decision.decision not in {"skill_candidate", "draft_review"}:
                    return _json_error("learning promotion requires review before drafting")
                draft = write_proposal_draft(
                    report=report,
                    proposal=proposal,
                    points=[point],
                    hermes_home=self._hermes_home,
                    config=self._config,
                    write_decision=write_decision,
                )
                draft_path = str(draft["path"])
                self._qdrant.update_payload(
                    collection_name,
                    point.id,
                    {
                        "promoted_to_skill_draft": True,
                        "proposal_draft_path": draft_path,
                        "skill_draft_path": draft_path,
                        "promoted_at": datetime.utcnow().isoformat() + "Z",
                        "consolidation_proposal_id": proposal_id,
                    },
                )
                record = persist_application_record({"applied": True, **plan, **pre_apply, "proposal_draft_path": draft_path, "skill_draft_path": draft_path, "write_decision": write_decision.to_dict()}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, **pre_apply, "proposal_draft_path": draft_path, "skill_draft_path": draft_path, "write_decision": write_decision.to_dict(), "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            return _json_error("unsupported consolidation action")
        except Exception as exc:
            return _json_error(f"Consolidation apply failed: {exc}")

    def _ensure_learning_store(self) -> Optional[LearningStore]:
        if self._learning_store:
            return self._learning_store
        if not self._qdrant or not self._embeddings:
            return None
        self._learning_store = LearningStore(
            qdrant=self._qdrant,
            embeddings=self._embeddings,
            collection_name=self._config["learning_collection_name"],
            profile_id=self._profile_id,
            platform=self._platform,
            session_id=self._session_id,
            user_id_hash=self._user_id_hash,
            chat_id_hash=self._chat_id_hash,
            model=self._config.get("embedding_model", ""),
            scope=self._scope_filter_values(),
            search_candidates=self._config.get("search_candidates", 20),
            decay_rate=self._config.get("decay_rate", 0.001),
            min_raw_score=self._config.get("min_raw_score", 0.0),
            min_final_score=self._config.get("min_final_score", 0.0),
        )
        return self._learning_store

    def _tool_learning_store(self, args: dict) -> str:
        if not self._config.get("learning_enabled", True):
            return _json_error("Qdrant learning tools are disabled by qdrant_memory.learning_enabled")
        store = self._ensure_learning_store()
        if not store:
            return _json_error("Qdrant learning store is not initialized")
        lesson = str(args.get("lesson") or "").strip()
        if not lesson:
            return _json_error("lesson is required")
        tags = args.get("tags") or []
        if not isinstance(tags, list):
            tags = []
        try:
            importance = max(1, min(10, int(args.get("importance", 7))))
        except Exception:
            importance = 7
        try:
            confidence = max(0.0, min(1.0, float(args.get("confidence", 0.8))))
        except Exception:
            confidence = 0.8
        learning_type = str(args.get("learning_type") or "")
        trigger = str(args.get("trigger") or "")
        mistake = str(args.get("mistake") or "")
        correction = str(args.get("correction") or "")
        evidence = str(args.get("evidence") or "")
        tool_name = str(args.get("tool_name") or "")
        command = str(args.get("command") or "")
        gate_text = "\n".join(
            part
            for part in (
                f"Lesson: {lesson}",
                f"Trigger: {trigger}" if trigger else "",
                f"Mistake: {mistake}" if mistake else "",
                f"Correction: {correction}" if correction else "",
                f"Evidence: {evidence}" if evidence else "",
                f"Tool: {tool_name}" if tool_name else "",
                f"Command: {command}" if command else "",
            )
            if part
        )
        try:
            write_decision = evaluate_write_candidate(
                text=gate_text,
                target="learning",
                source_type="learning",
                confidence=confidence,
                promote_to_skill_candidate=bool(args.get("promote_to_skill_candidate", False)),
                metadata={"importance": importance, "learning_type": learning_type, "trigger": trigger, "mistake": mistake, "correction": correction, "evidence": evidence, "tool_name": tool_name, "command": command, "tags": [str(t) for t in tags]},
            )
            if write_decision.decision == "reject":
                return _json_error("write gate rejected learning candidate")
            if write_decision.decision == "skip":
                return json.dumps({"saved": False, "id": None, "collection_name": self._config["learning_collection_name"], "write_decision": write_decision.to_dict()})
            if write_decision.decision not in {"learning_candidate", "store", "skill_candidate"}:
                return _json_error("learning candidate requires manual draft review before storing")
            point_id = store.store(
                lesson=lesson,
                learning_type=str(args.get("learning_type") or ""),
                trigger=str(args.get("trigger") or ""),
                mistake=str(args.get("mistake") or ""),
                correction=str(args.get("correction") or ""),
                evidence=str(args.get("evidence") or ""),
                tool_name=str(args.get("tool_name") or ""),
                command=str(args.get("command") or ""),
                importance=importance,
                confidence=confidence,
                tags=[str(t) for t in tags],
                promote_to_skill_candidate=bool(args.get("promote_to_skill_candidate", False)),
            )
            return json.dumps({"saved": bool(point_id), "id": point_id, "collection_name": self._config["learning_collection_name"], "write_decision": write_decision.to_dict()})
        except Exception as exc:
            return _json_error(f"Failed to store learning: {exc}")

    def _tool_learning_search(self, args: dict) -> str:
        if not self._config.get("learning_enabled", True):
            return _json_error("Qdrant learning tools are disabled by qdrant_memory.learning_enabled")
        store = self._ensure_learning_store()
        if not store:
            return _json_error("Qdrant learning store is not initialized")
        query = str(args.get("query") or "").strip()
        if not query:
            return _json_error("query is required")
        try:
            top_k = max(1, min(20, int(args.get("top_k", 5))))
        except Exception:
            top_k = 5
        include_metadata = bool(args.get("include_metadata", False))
        learning_type = args.get("learning_type") or None
        try:
            chunks = store.search(query, top_k=top_k, learning_type=learning_type, **_tool_search_filters(args))
            results = []
            for chunk in chunks:
                item: dict[str, Any] = {
                    "id": chunk.id,
                    "text": chunk.text,
                    "score": round(chunk.final_score, 6),
                    "vector_score": round(chunk.qdrant_score, 6),
                }
                payload = chunk.payload or {}
                if payload.get("learning_type"):
                    item["learning_type"] = payload.get("learning_type")
                if include_metadata:
                    item["metadata"] = payload
                results.append(item)
            return json.dumps({"results": results, "count": len(results), "collection_name": self._config["learning_collection_name"]})
        except Exception as exc:
            return _json_error(f"Learning search failed: {exc}")

    def _tool_learning_preview(self, args: dict) -> str:
        include_metadata = bool(args.get("include_metadata", False))
        candidates = [candidate.to_dict(include_metadata=include_metadata) for candidate in self._pending_learning_candidates.values()]
        return json.dumps({"candidates": candidates, "count": len(candidates), "dry_run": True})

    def _tool_learning_approve(self, args: dict) -> str:
        if not self._config.get("learning_enabled", True):
            return _json_error("Qdrant learning tools are disabled by qdrant_memory.learning_enabled")
        candidate_id = str(args.get("candidate_id") or "").strip()
        if not candidate_id:
            return _json_error("candidate_id is required")
        candidate = self._pending_learning_candidates.get(candidate_id)
        if not candidate:
            return _json_error(f"Unknown learning candidate: {candidate_id}")
        dry_run = bool(args.get("dry_run", True))
        learning_args = candidate_to_learning_args(candidate)
        store = self._ensure_learning_store()
        learning_type = str(learning_args.get("learning_type") or "")
        if not learning_type:
            learning_type = classify_learning_type(
                str(learning_args.get("trigger") or ""),
                str(learning_args.get("correction") or ""),
                tool_name=str(learning_args.get("tool_name") or ""),
            )
        persisted_payload = build_learning_payload(
            lesson=str(learning_args.get("lesson") or ""),
            learning_type=learning_type,
            trigger=str(learning_args.get("trigger") or ""),
            mistake=str(learning_args.get("mistake") or ""),
            correction=str(learning_args.get("correction") or ""),
            evidence=str(learning_args.get("evidence") or ""),
            tool_name=str(learning_args.get("tool_name") or ""),
            command=str(learning_args.get("command") or ""),
            project_path=store.project_path if store else "",
            profile_id=store.profile_id if store else self._profile_id,
            platform=store.platform if store else self._platform,
            user_id_hash=store.user_id_hash if store else self._user_id_hash,
            chat_id_hash=store.chat_id_hash if store else self._chat_id_hash,
            session_id=store.session_id if store else self._session_id,
            model=store.model if store else self._config.get("embedding_model", ""),
            importance=int(learning_args.get("importance") or 7),
            confidence=float(learning_args.get("confidence") or 0.8),
            tags=[str(t) for t in learning_args.get("tags") or []],
            promote_to_skill_candidate=bool(learning_args.get("promote_to_skill_candidate", False)),
        )
        write_decision = evaluate_write_candidate(
            text=str(persisted_payload.get("text") or ""),
            target="learning",
            source_type=str(persisted_payload.get("source_type") or "learning"),
            confidence=learning_args.get("confidence"),
            promote_to_skill_candidate=bool(learning_args.get("promote_to_skill_candidate", False)),
            metadata=persisted_payload,
        )
        if dry_run:
            return json.dumps({"dry_run": True, "saved": False, "candidate": candidate.to_dict(include_metadata=True), "learning_args": learning_args, "write_decision": write_decision.to_dict()})
        if not store:
            return _json_error("Qdrant learning store is not initialized")
        try:
            if write_decision.decision == "reject":
                return _json_error("write gate rejected learning candidate")
            if write_decision.decision not in {"learning_candidate", "store", "skill_candidate"}:
                return _json_error("learning candidate requires manual draft review before storing")
            point_id = store.store(**learning_args)
            self._pending_learning_candidates.pop(candidate_id, None)
            return json.dumps({"dry_run": False, "saved": bool(point_id), "id": point_id, "collection_name": self._config["learning_collection_name"], "write_decision": write_decision.to_dict()})
        except Exception as exc:
            return _json_error(f"Learning approval failed: {exc}")

    def _tool_extraction_preview(self, args: dict) -> str:
        candidates = list(self._pending_extraction_candidates.values())
        preview = preview_source_extraction_candidates(candidates)
        self._reviewed_extraction_candidate_ids.update(str(candidate.candidate_id) for candidate in candidates)
        return json.dumps(preview)

    def _source_extraction_approval_payload(self, candidate: Any) -> dict[str, Any]:
        payload = dict(candidate.proposed_payload or {})
        payload.update(
            {
                "profile_id": self._profile_id,
                "platform": self._platform,
                "session_id": self._session_id,
                "user_id_hash": self._user_id_hash,
                "chat_id_hash": self._chat_id_hash,
                "model": self._config.get("embedding_model", ""),
                "provider": "qdrant",
                "source_extraction_candidate_id": candidate.candidate_id,
                "source_extraction_approved_at": datetime.utcnow().isoformat() + "Z",
            }
        )
        return payload

    def _tool_extraction_approve(self, args: dict) -> str:
        candidate_id = str(args.get("candidate_id") or "").strip()
        if not candidate_id:
            return _json_error("candidate_id is required")
        candidate = self._pending_extraction_candidates.get(candidate_id)
        if not candidate:
            return _json_error(f"Unknown extraction candidate: {candidate_id}")
        dry_run = parse_bool_arg(args.get("dry_run"), default=True)
        approve = parse_bool_arg(args.get("approve"), default=False)
        write_decision = evaluate_source_extraction_candidate(candidate)
        proposal_payload: dict[str, Any] = {}
        if write_decision.decision == "draft_review":
            _, proposal, _ = build_source_extraction_proposal(candidate, write_decision)
            proposal_payload["proposal_id"] = proposal["proposal_id"]
        base = {
            "candidate": candidate.to_dict(),
            "write_decision": write_decision.to_dict(),
            "would_store": write_decision.decision == "store",
            "would_create_proposal": write_decision.decision == "draft_review",
            "collection_name": self._config["collection_name"],
            **proposal_payload,
        }
        if dry_run:
            self._reviewed_extraction_candidate_ids.add(candidate_id)
            return json.dumps({"dry_run": True, "saved": False, "proposal_created": False, **base})
        if not approve:
            return _json_error("approve=true is required when dry_run=false")
        if candidate_id not in self._reviewed_extraction_candidate_ids:
            return _json_error("dry-run preview is required before live extraction approval")
        if write_decision.decision == "reject":
            reasons = ", ".join(write_decision.reasons)
            return _json_error(f"write gate rejected extraction candidate: {reasons}")
        if write_decision.decision == "draft_review":
            report, proposal, points = build_source_extraction_proposal(candidate, write_decision)
            draft = write_proposal_draft(
                report=report,
                proposal=proposal,
                points=points,
                hermes_home=self._hermes_home,
                config=self._config,
                write_decision=write_decision,
            )
            self._pending_extraction_candidates.pop(candidate_id, None)
            self._reviewed_extraction_candidate_ids.discard(candidate_id)
            draft_path = str(draft["path"])
            return json.dumps(
                {
                    "dry_run": False,
                    "saved": False,
                    "proposal_created": True,
                    "proposal_draft_path": draft_path,
                    "proposal_id": proposal["proposal_id"],
                    **base,
                }
            )
        if write_decision.decision != "store":
            return _json_error("extraction candidate requires manual review before storing")
        if not self._qdrant or not self._embeddings:
            return _json_error("Qdrant memory provider is not initialized")
        try:
            payload = self._source_extraction_approval_payload(candidate)
            persisted_decision = evaluate_source_extraction_candidate(candidate, persisted_payload=payload)
            persisted_base = {
                **base,
                "write_decision": persisted_decision.to_dict(),
                "would_store": persisted_decision.decision == "store",
                "would_create_proposal": persisted_decision.decision == "draft_review",
            }
            if persisted_decision.decision == "reject":
                reasons = ", ".join(persisted_decision.reasons)
                return _json_error(f"write gate rejected extraction candidate: {reasons}")
            if persisted_decision.decision == "draft_review":
                report, proposal, points = build_source_extraction_proposal(candidate, persisted_decision)
                draft = write_proposal_draft(
                    report=report,
                    proposal=proposal,
                    points=points,
                    hermes_home=self._hermes_home,
                    config=self._config,
                    write_decision=persisted_decision,
                )
                self._pending_extraction_candidates.pop(candidate_id, None)
                self._reviewed_extraction_candidate_ids.discard(candidate_id)
                draft_path = str(draft["path"])
                return json.dumps(
                    {
                        "dry_run": False,
                        "saved": False,
                        "proposal_created": True,
                        "proposal_draft_path": draft_path,
                        "proposal_id": proposal["proposal_id"],
                        **persisted_base,
                    }
                )
            if persisted_decision.decision != "store":
                return _json_error("extraction candidate requires manual review before storing")
            text = str(payload.get("text") or payload.get("claim_text") or "")
            vector = self._embeddings.embed_document(text)
            self._qdrant.upsert(self._config["collection_name"], [{"id": candidate.candidate_id, "vector": vector, "payload": payload}])
            self._pending_extraction_candidates.pop(candidate_id, None)
            self._reviewed_extraction_candidate_ids.discard(candidate_id)
            return json.dumps(
                {
                    "dry_run": False,
                    "saved": True,
                    "proposal_created": False,
                    "id": candidate.candidate_id,
                    **persisted_base,
                }
            )
        except Exception as exc:
            return _json_error(f"Extraction approval failed: {exc}")

    def _tool_improve_preview(self, args: dict) -> str:
        """Preview improve candidates as a dry-run report. No Qdrant writes."""
        source_scope = str(args.get("source_scope") or "").strip()
        source_text = str(args.get("source_text") or "")
        source_uri_arg = str(args.get("source_uri") or "").strip()
        point_ids = args.get("point_ids") or []
        session_id = str(args.get("session_id") or self._session_id or "")
        persist = parse_bool_arg(args.get("persist"), default=True)
        include_metadata = parse_bool_arg(args.get("include_metadata"), default=False)
        try:
            max_candidates = max(1, min(IMPROVE_MAX_CANDIDATES_HARD_CAP, int(args.get("max_candidates") or IMPROVE_MAX_CANDIDATES_DEFAULT)))
        except Exception:
            max_candidates = IMPROVE_MAX_CANDIDATES_DEFAULT

        # Determine effective scope
        if not source_scope:
            if source_text:
                source_scope = "source_text"
            elif point_ids:
                source_scope = "point_ids"
            else:
                source_scope = "pending_session"

        profile_id = self._profile_id
        all_candidates: list[Any] = []
        source_handles: list[str] = []

        if source_scope == "source_text":
            if not source_text.strip():
                return _json_error("source_text is required when source_scope=source_text")
            uri = source_uri_arg or f"session://{session_id or 'current'}/improve-source"
            all_candidates = extract_improve_candidates_from_text(
                source_text,
                source_uri=uri,
                source_type="source_text",
                derivation_type="source_text",
                confidence=0.85,
                profile_id=profile_id,
                lifecycle_id=session_id,
                max_candidates=max_candidates,
            )
            source_handles = [uri]

        elif source_scope == "point_ids":
            if not point_ids or not isinstance(point_ids, list):
                return _json_error("point_ids is required when source_scope=point_ids")
            if not self._qdrant:
                return _json_error("Qdrant memory provider is not initialized")
            for pid in point_ids[:max_candidates]:
                pid_str = str(pid).strip()
                if not pid_str:
                    continue
                try:
                    # Read-only retrieve
                    results = self._qdrant.scroll(
                        self._config["collection_name"],
                        limit=1,
                        with_payload=True,
                        with_vectors=False,
                        _filter={"must": [{"key": "id", "match": {"value": pid_str}}]},
                    )
                    points = results[0] if isinstance(results, (list, tuple)) and results else []
                    for point in (points if isinstance(points, list) else []):
                        cands = extract_improve_candidates_from_point(
                            point,
                            confidence=0.75,
                            profile_id=profile_id,
                            lifecycle_id=session_id,
                            max_candidates=max_candidates,
                        )
                        all_candidates.extend(cands)
                        source_handles.append(f"qdrant://memory/{pid_str}")
                except Exception:
                    pass
            # Dedupe
            seen_ids: set[str] = set()
            deduped: list[Any] = []
            for c in all_candidates:
                if c.candidate_id not in seen_ids:
                    seen_ids.add(c.candidate_id)
                    deduped.append(c)
            all_candidates = deduped[:max_candidates]

        elif source_scope == "pending_session":
            # Use already-pending extraction candidates from session hooks
            all_candidates = list(self._pending_extraction_candidates.values())[:max_candidates]
            source_handles = [f"session://{session_id or 'current'}"]

        else:
            return _json_error(f"unsupported source_scope: {source_scope}")

        report = build_improve_report(
            all_candidates,
            profile_id=profile_id,
            session_id=session_id,
            source_scope=source_scope,
            source_handles=source_handles,
            persist=persist,
            include_metadata=include_metadata,
        )

        if persist:
            try:
                persist_improve_report(report, hermes_home=self._hermes_home)
            except Exception as exc:
                logger.warning("Failed to persist improve report: %s", exc)

        # Store in provider-local pending state
        self._pending_improve_reports[report["report_id"]] = report

        return json.dumps(report)

    def _tool_improve_apply(self, args: dict) -> str:
        """Preview or apply exactly one candidate from one report."""
        # Blocker 1: validate raw report_id BEFORE any normalization.
        # Leading/trailing whitespace must be rejected, not stripped.
        raw_report_id = str(args.get("report_id") or "")
        if not raw_report_id:
            return _json_error("report_id is required")
        if raw_report_id != raw_report_id.strip():
            return _json_error("report_id must match canonical format improve-<12hex>")
        report_id = raw_report_id.strip()
        if not REPORT_ID_RE.match(report_id):
            return _json_error("report_id must match canonical format improve-<12hex>")
        candidate_id = str(args.get("candidate_id") or "").strip()
        if not candidate_id:
            return _json_error("candidate_id is required")
        dry_run = parse_bool_arg(args.get("dry_run"), default=True)
        approve = parse_bool_arg(args.get("approve"), default=False)

        # Load report from in-memory state or persisted artifact
        report = self._pending_improve_reports.get(report_id)
        if not report:
            report = load_improve_report(report_id, hermes_home=self._hermes_home)
            if report:
                self._pending_improve_reports[report_id] = report
        if not report:
            return _json_error(f"Unknown improve report: {report_id}")

        # Profile scope check
        if str(report.get("profile_id") or self._profile_id) != self._profile_id:
            return _json_error("improve report belongs to a different profile scope")

        # Find exactly one candidate
        candidates_list = report.get("candidates") or []
        match_item: dict[str, Any] | None = None
        for item in candidates_list:
            if not isinstance(item, dict):
                continue
            if str(item.get("candidate_id") or "") == candidate_id:
                if match_item is not None:
                    return _json_error("duplicate candidate_id in report")
                match_item = item
        if not match_item:
            return _json_error(f"Unknown candidate_id in report: {candidate_id}")

        candidate_digest = str(match_item.get("candidate_digest") or "")
        review_key = f"{report_id}:{candidate_id}:{candidate_digest}"

        # Reconstruct candidate from report item for write-gate evaluation
        from qdrant_memory.extraction_candidates import ExtractionCandidate as _EC
        candidate = _EC(
            candidate_id=str(match_item.get("candidate_id") or ""),
            candidate_type=str(match_item.get("candidate_type") or ""),
            source_uri=str(match_item.get("source_uri") or ""),
            locator=match_item.get("locator") or {},
            derived_from=match_item.get("derived_from") or [],
            proposed_payload=match_item.get("proposed_payload") or {},
            reason=str(match_item.get("reason") or ""),
            confidence=float(match_item.get("confidence") or 0.0),
            risk=str(match_item.get("risk") or "unknown"),
            requires_review=bool(match_item.get("requires_review", True)),
            created_at=str(match_item.get("created_at") or ""),
        )

        # Evaluate write gate
        write_decision = evaluate_source_extraction_candidate(candidate)

        base_result: dict[str, Any] = {
            "report_id": report_id,
            "candidate_id": candidate_id,
            "candidate_digest": candidate_digest,
            "target_point_id": match_item.get("target_point_id") or "",
            "target_memory_kind": match_item.get("target_memory_kind") or "",
            "write_decision": write_decision.to_dict(),
        }

        # Dry-run path
        if dry_run:
            self._reviewed_improve_candidate_keys.add(review_key)
            return json.dumps({
                "dry_run": True,
                "applied": False,
                "saved": False,
                "would_store": write_decision.decision == "store",
                "would_create_proposal": write_decision.decision == "draft_review",
                **base_result,
            })

        # Live apply path
        if not approve:
            return _json_error("approve=true is required when dry_run=false")
        if review_key not in self._reviewed_improve_candidate_keys:
            return _json_error("dry-run preview is required before live improve apply")

        # Check digest hasn't changed (stale report)
        fresh_digest = make_candidate_digest(match_item)
        if fresh_digest != candidate_digest:
            return _json_error("candidate digest mismatch; report may be stale")

        # Route by write decision
        if write_decision.decision == "reject":
            reasons = ", ".join(write_decision.reasons)
            return _json_error(f"write gate rejected improve candidate: {reasons}")

        if write_decision.decision == "draft_review":
            # Write a local proposal draft only; no Qdrant mutation
            from qdrant_memory.proposals import write_proposal_draft
            report_meta = {
                "report_id": report_id,
                "report_type": "improve_draft_review",
                "source_uri": candidate.source_uri,
                "dry_run_first": True,
            }
            proposal_meta = {
                "proposal_id": candidate_id,
                "proposal_type": "improve_candidate",
                "candidate_type": candidate.candidate_type,
                "suggested_action": "manual_review",
                "affected_ids": [candidate_id],
                "confidence": candidate.confidence,
                "write_decision": write_decision.to_dict(),
            }
            point_data = {
                "id": match_item.get("target_point_id") or candidate_id,
                "text": str(candidate.proposed_payload.get("text") or ""),
                "payload": {**candidate.proposed_payload, "improve_candidate_id": candidate_id},
            }
            try:
                draft = write_proposal_draft(
                    report=report_meta,
                    proposal=proposal_meta,
                    points=[point_data],
                    hermes_home=self._hermes_home,
                    config=self._config,
                    write_decision=write_decision,
                )
            except Exception as exc:
                return _json_error(f"Failed to write improve draft: {exc}")
            # Remove from pending state
            self._reviewed_improve_candidate_keys.discard(review_key)
            self._remove_candidate_from_report(report_id, candidate_id)
            return json.dumps({
                "dry_run": False,
                "applied": True,
                "saved": False,
                "proposal_created": True,
                "proposal_id": candidate_id,
                "proposal_draft_path": str(draft["path"]),
                **base_result,
            })

        if write_decision.decision != "store":
            return _json_error("improve candidate requires manual review before storing")

        # Blocker 2: Idempotency check — if this candidate was already applied
        # for this report, return already_applied without embedding/upserting.
        applied_record = is_candidate_applied(
            report_id, candidate_id, hermes_home=self._hermes_home
        )
        if applied_record is not None:
            self._reviewed_improve_candidate_keys.discard(review_key)
            return json.dumps({
                "dry_run": False,
                "applied": True,
                "saved": False,
                "already_applied": True,
                "id": str(applied_record.get("target_point_id") or ""),
                "application_record": applied_record,
                **base_result,
            })

        # Store path: embed and upsert
        if not self._qdrant or not self._embeddings:
            return _json_error("Qdrant memory provider is not initialized")
        try:
            # Build persisted payload with provider metadata
            payload = dict(candidate.proposed_payload)
            payload.update({
                "profile_id": self._profile_id,
                "platform": self._platform,
                "session_id": self._session_id,
                "user_id_hash": self._user_id_hash,
                "chat_id_hash": self._chat_id_hash,
                "model": self._config.get("embedding_model", ""),
                "provider": "qdrant",
                "improve_candidate_id": candidate_id,
                "improve_report_id": report_id,
                "improve_applied_at": datetime.utcnow().isoformat() + "Z",
            })
            # Re-run write gate with persisted payload
            persisted_decision = evaluate_source_extraction_candidate(candidate, persisted_payload=payload)
            if persisted_decision.decision == "reject":
                reasons = ", ".join(persisted_decision.reasons)
                return _json_error(f"write gate rejected improve candidate: {reasons}")
            if persisted_decision.decision != "store":
                return _json_error("improve candidate requires manual review after payload enrichment")
            base_result["write_decision"] = persisted_decision.to_dict()

            target_pid = str(match_item.get("target_point_id") or candidate_id)

            # Blocker 3: Conflict detection — retrieve existing point before upsert
            # Fail closed on retrieval errors: cannot verify target absence.
            try:
                existing_points = self._qdrant.retrieve(
                    self._config["collection_name"], [target_pid], with_payload=True
                )
            except Exception:
                return _json_error(
                    "Unable to verify target point absence; refusing to overwrite "
                    "(Qdrant retrieval error)"
                )
            if existing_points:
                existing = existing_points[0]
                existing_payload = existing.get("payload") if isinstance(existing, dict) else {}
                # Check if it's an exact replay (same candidate/provenance marker)
                if isinstance(existing_payload, dict):
                    existing_candidate_id = str(existing_payload.get("improve_candidate_id") or "")
                    existing_report_id = str(existing_payload.get("improve_report_id") or "")
                    if existing_candidate_id == candidate_id and existing_report_id == report_id:
                        # Exact replay — already applied
                        record_candidate_applied(
                            report_id, candidate_id,
                            hermes_home=self._hermes_home,
                            target_point_id=target_pid,
                            candidate_digest=candidate_digest,
                        )
                        self._reviewed_improve_candidate_keys.discard(review_key)
                        self._remove_candidate_from_report(report_id, candidate_id)
                        return json.dumps({
                            "dry_run": False,
                            "applied": True,
                            "saved": False,
                            "already_applied": True,
                            "id": target_pid,
                            "collection_name": self._config["collection_name"],
                            **base_result,
                        })
                    # Existing point differs — fail closed, do NOT overwrite
                    return _json_error(
                        f"Target point {target_pid} already exists with different "
                        f"payload/provenance; refusing to overwrite (use manual review)"
                    )

            text = str(payload.get("text") or payload.get("claim_text") or "")
            vector = self._embeddings.embed_document(text)
            self._qdrant.upsert(self._config["collection_name"], [{"id": target_pid, "vector": vector, "payload": payload}])

            # Blocker 2: Persist application record for idempotent repeat
            record_candidate_applied(
                report_id, candidate_id,
                hermes_home=self._hermes_home,
                target_point_id=target_pid,
                candidate_digest=candidate_digest,
            )

            # Remove from pending state after successful apply
            self._reviewed_improve_candidate_keys.discard(review_key)
            self._remove_candidate_from_report(report_id, candidate_id)

            return json.dumps({
                "dry_run": False,
                "applied": True,
                "saved": True,
                "already_applied": False,
                "id": target_pid,
                "collection_name": self._config["collection_name"],
                **base_result,
            })
        except Exception as exc:
            return _json_error(f"Improve apply failed: {exc}")

    def _remove_candidate_from_report(self, report_id: str, candidate_id: str) -> None:
        """Remove a candidate from a pending report after successful apply."""
        report = self._pending_improve_reports.get(report_id)
        if not report:
            return
        candidates = report.get("candidates") or []
        report["candidates"] = [c for c in candidates if isinstance(c, dict) and str(c.get("candidate_id") or "") != candidate_id]
        counts = report.get("counts") or {}
        if isinstance(counts, dict):
            counts["total"] = max(0, int(counts.get("total") or 0) - 1)
        # If no candidates left, remove the entire report
        if not report["candidates"]:
            self._pending_improve_reports.pop(report_id, None)

    def _tool_graph_search(self, args: dict) -> str:
        query = str(args.get("query") or "").strip()
        if not query:
            return _json_error("query is required")
        collection = str(args.get("collection") or "memory").strip().lower()
        if collection not in {"memory", "learning"}:
            return _json_error("collection must be one of: memory, learning")
        if not getattr(self, "_qdrant", None) or not getattr(self, "_embeddings", None):
            return _json_error("Qdrant memory provider is not initialized")
        collection_name = self._config["learning_collection_name"] if collection == "learning" else self._config["collection_name"]
        try:
            top_k = max(1, min(20, int(args.get("top_k", 5))))
            candidate_seed_top_k = max(1, min(50, int(args.get("candidate_seed_top_k", 20))))
            max_graph_results = max(1, min(50, int(args.get("max_graph_results", 20))))
            max_depth = max(1, min(3, int(args.get("max_depth", 2))))
        except Exception:
            top_k, candidate_seed_top_k, max_graph_results, max_depth = 5, 20, 20, 2

        include_fact_history = parse_bool_arg(args.get("include_fact_history"), default=False)
        debug = parse_bool_arg(args.get("debug"), default=True)
        entity_types = args.get("entity_types") or None
        relation_types = args.get("relation_types") or None

        retriever = GraphMemoryRetriever(
            qdrant=self._qdrant,
            embeddings=self._embeddings,
            collection_name=collection_name,
            expansion_policy=GraphExpansionPolicy(max_depth=max_depth),
            scope=self._scope_filter_values(),
        )
        try:
            result = retriever.search(
                query,
                top_k=top_k,
                candidate_seed_top_k=candidate_seed_top_k,
                max_graph_results=max_graph_results,
                entity_types=entity_types,
                relation_types=relation_types,
                include_fact_history=include_fact_history,
                debug=debug,
            )
        except ValueError as exc:
            return _json_error(str(exc))
        except Exception as exc:
            return _json_error(f"Graph search failed: {exc}")

        results = []
        for candidate in result.final:
            item: dict[str, Any] = {
                "point_id": candidate.point_id,
                "final_score": round(candidate.final_score, 6),
                "graph_distance": candidate.graph_distance,
                "text": str(candidate.payload.get("text") or ""),
                "path": candidate.path,
                "relation_path": candidate.relation_path,
            }
            results.append(item)
        return json.dumps({
            "results": results,
            "count": len(results),
            "debug": result.debug,
        })

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        args = args or {}
        if tool_name == "qdrant_memory_status":
            return self._tool_status()
        if tool_name == "qdrant_memory_store":
            return self._tool_store(args)
        if tool_name == "qdrant_memory_search":
            return self._tool_search(args)
        if tool_name == "qdrant_memory_context":
            return self._tool_context(args)
        if tool_name == "qdrant_memory_index":
            return self._tool_index(args)
        if tool_name == "qdrant_memory_forget":
            return self._tool_forget(args)
        if tool_name == "qdrant_memory_inspect":
            return self._tool_inspect(args)
        if tool_name == "qdrant_memory_trace":
            return self._tool_trace(args)
        if tool_name == "qdrant_memory_expand":
            return self._tool_expand(args)
        if tool_name == "qdrant_memory_source_status":
            return self._tool_source_status(args)
        if tool_name == "qdrant_memory_consolidate":
            return self._tool_consolidate(args)
        if tool_name == "qdrant_memory_consolidation_apply":
            return self._tool_consolidation_apply(args)
        if tool_name == "qdrant_learning_store":
            return self._tool_learning_store(args)
        if tool_name == "qdrant_learning_search":
            return self._tool_learning_search(args)
        if tool_name == "qdrant_learning_preview":
            return self._tool_learning_preview(args)
        if tool_name == "qdrant_learning_approve":
            return self._tool_learning_approve(args)
        if tool_name == "qdrant_memory_extraction_preview":
            return self._tool_extraction_preview(args)
        if tool_name == "qdrant_memory_extraction_approve":
            return self._tool_extraction_approve(args)
        if tool_name == "qdrant_memory_graph_search":
            return self._tool_graph_search(args)
        if tool_name == "qdrant_memory_improve_preview":
            return self._tool_improve_preview(args)
        if tool_name == "qdrant_memory_improve_apply":
            return self._tool_improve_apply(args)
        return _json_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None


def register(ctx) -> None:
    ctx.register_memory_provider(QdrantMemoryProvider())
