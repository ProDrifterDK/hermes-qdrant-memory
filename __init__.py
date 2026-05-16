from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
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

from qdrant_memory.client import QdrantClient
from qdrant_memory.config import load_config
from qdrant_memory.consolidation import (
    artifact_root,
    build_consolidation_report,
    build_reconsolidation_draft_text,
    build_skill_draft_text,
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
from qdrant_memory.learning import LearningStore
from qdrant_memory.lesson_extractor import LearningCandidate, candidate_to_learning_args, extract_learning_candidates_from_messages
from qdrant_memory.retriever import MemoryRetriever, format_for_prompt
from qdrant_memory.tools import TOOL_SCHEMAS
from qdrant_memory.writer import ConversationWriter

logger = logging.getLogger(__name__)


def _json_error(message: str) -> str:
    return json.dumps({"error": message})


def _hash_value(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


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
            "Active local long-term semantic memory. Use qdrant_memory_search, "
            "qdrant_memory_store, qdrant_memory_index, qdrant_learning_search, "
            "qdrant_learning_store, and qdrant_memory_status for explicit memory operations."
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

    def on_pre_compress(self, messages: list[Any]) -> str:
        candidates = self._collect_learning_candidates(messages, source_hook="on_pre_compress")
        if not candidates:
            return ""
        lines = ["# Qdrant Learning Candidates", "Potential procedural lessons detected but not stored. Review with qdrant_learning_preview and approve explicitly."]
        for candidate in candidates:
            lines.append(f"- {candidate.candidate_id} [{candidate.learning_type}] {candidate.lesson[:220]}")
        return "\n".join(lines)

    def on_session_end(self, messages: list[Any]) -> None:
        self._collect_learning_candidates(messages, source_hook="on_session_end")

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return TOOL_SCHEMAS

    def _tool_status(self) -> str:
        qdrant_ok = False
        embedding_ok = False
        collections: list[str] = []
        point_count = None
        learning_point_count = None
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
            "point_count": point_count,
            "learning_collection_name": self._config["learning_collection_name"],
            "learning_collection_exists": self._config["learning_collection_name"] in collections,
            "learning_point_count": learning_point_count,
            "learning_enabled": self._config.get("learning_enabled", True),
            "learning_auto_extract_enabled": self._config.get("learning_auto_extract_enabled", False),
            "learning_auto_extract_mode": self._config.get("learning_auto_extract_mode", "preview"),
            "pending_learning_candidate_count": len(self._pending_learning_candidates),
            "consolidation_enabled": self._config.get("consolidation_enabled", False),
            "consolidation_persist_reports": self._config.get("consolidation_persist_reports", True),
            "consolidation_apply_enabled": True,
            "consolidation_supported_actions": ["merge", "delete", "promote_to_skill", "draft_review"],
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
        try:
            point_id = self._writer.store_text(text, source_type=source_type, importance=importance, tags=[str(t) for t in tags])
            return json.dumps({"saved": bool(point_id), "id": point_id, "source_type": source_type})
        except Exception as exc:
            return _json_error(f"Failed to store memory: {exc}")

    def _tool_search(self, args: dict) -> str:
        if not self._retriever:
            return _json_error("Qdrant memory provider is not initialized")
        query = str(args.get("query") or "").strip()
        if not query:
            return _json_error("query is required")
        try:
            top_k = max(1, min(20, int(args.get("top_k", 5))))
        except Exception:
            top_k = 5
        include_metadata = bool(args.get("include_metadata", False))
        source_type = args.get("source_type") or None
        try:
            chunks = self._retriever.search(query, top_k=top_k, source_type=source_type)
            results = []
            for chunk in chunks:
                item: dict[str, Any] = {"id": chunk.id, "text": chunk.text, "score": round(chunk.final_score, 6)}
                if include_metadata:
                    item["metadata"] = chunk.payload
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
        if action == "merge" and points:
            canonical = self._select_canonical_point(points)
            plan.update({"canonical_id": canonical.id, "delete_ids": [p.id for p in points if p.id != canonical.id]})
        elif action == "delete":
            plan.update({"delete_ids": affected_ids})
        elif action == "promote_to_skill" and points:
            root = artifact_root(self._hermes_home, str(self._config.get("consolidation_artifact_dir") or "")) / "skill_drafts"
            plan.update({"skill_draft_path": str(root / f"{proposal.get('proposal_id')}.md")})
        elif action == "draft_review" and points:
            root = artifact_root(self._hermes_home, str(self._config.get("consolidation_artifact_dir") or "")) / "reconsolidation_drafts"
            plan.update({"reconsolidation_draft_path": str(root / f"{proposal.get('proposal_id')}.md")})
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
            if action != expected_action:
                return _json_error("action mismatch for proposal type")
            affected_ids = [str(item) for item in proposal.get("affected_ids") or [] if str(item)]
            if not affected_ids:
                return _json_error("proposal has no explicit affected_ids")
            collection_name = self._collection_for_proposal(proposal)
            points = self._retrieve_consolidation_points(collection_name, affected_ids)
            if len(points) != len(set(affected_ids)):
                return _json_error("affected point missing; rerun consolidation")
            plan = self._proposal_apply_plan(report, proposal, action, points)
            if dry_run:
                return json.dumps({"dry_run": True, "would_apply": True, **plan})
            if action == "draft_review":
                draft_root = artifact_root(self._hermes_home, str(self._config.get("consolidation_artifact_dir") or "")) / "reconsolidation_drafts"
                draft_root.mkdir(parents=True, exist_ok=True)
                draft_path = draft_root / f"{proposal_id}.md"
                draft_text = build_reconsolidation_draft_text(points, proposal=proposal, report_id=report_id)
                draft_path.write_text(draft_text, encoding="utf-8")
                record = persist_application_record({"applied": True, **plan, "reconsolidation_draft_path": str(draft_path)}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, "reconsolidation_draft_path": str(draft_path), "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "delete":
                self._qdrant.delete_ids(collection_name, affected_ids)
                record = persist_application_record({"applied": True, **plan, "deleted_ids": affected_ids}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, "deleted_ids": affected_ids, "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "merge":
                if len(points) < 2:
                    return _json_error("merge requires at least two affected points")
                if any("bearer" in p.text.lower() or "secret" in p.text.lower() for p in points):
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
                record = persist_application_record({"applied": True, **plan, "canonical_id": canonical.id, "deleted_ids": delete_ids}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, "canonical_id": canonical.id, "deleted_ids": delete_ids, "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
            if action == "promote_to_skill":
                if collection_name != self._config["learning_collection_name"]:
                    return _json_error("promote_to_skill requires a learning collection proposal")
                point = points[0]
                if "bearer" in point.text.lower() or "secret" in point.text.lower():
                    return _json_error("secret-bearing learning requires manual review")
                draft_root = artifact_root(self._hermes_home, str(self._config.get("consolidation_artifact_dir") or "")) / "skill_drafts"
                draft_root.mkdir(parents=True, exist_ok=True)
                draft_path = draft_root / f"{proposal_id}.md"
                draft_text = build_skill_draft_text(point, proposal_id=proposal_id, report_id=report_id)
                draft_path.write_text(draft_text, encoding="utf-8")
                self._qdrant.update_payload(
                    collection_name,
                    point.id,
                    {
                        "promoted_to_skill_draft": True,
                        "skill_draft_path": str(draft_path),
                        "promoted_at": datetime.utcnow().isoformat() + "Z",
                        "consolidation_proposal_id": proposal_id,
                    },
                )
                record = persist_application_record({"applied": True, **plan, "skill_draft_path": str(draft_path)}, hermes_home=self._hermes_home, configured_dir=str(self._config.get("consolidation_artifact_dir") or ""))
                return json.dumps({"dry_run": False, "applied": True, **plan, "skill_draft_path": str(draft_path), "application_id": record.get("application_id"), "application_artifact": record.get("artifact_path")})
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
        try:
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
            return json.dumps({"saved": bool(point_id), "id": point_id, "collection_name": self._config["learning_collection_name"]})
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
            chunks = store.search(query, top_k=top_k, learning_type=learning_type)
            results = []
            for chunk in chunks:
                item: dict[str, Any] = {"id": chunk.id, "text": chunk.text, "score": round(chunk.final_score, 6)}
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
        if dry_run:
            return json.dumps({"dry_run": True, "saved": False, "candidate": candidate.to_dict(include_metadata=True), "learning_args": learning_args})
        store = self._ensure_learning_store()
        if not store:
            return _json_error("Qdrant learning store is not initialized")
        try:
            point_id = store.store(**learning_args)
            self._pending_learning_candidates.pop(candidate_id, None)
            return json.dumps({"dry_run": False, "saved": bool(point_id), "id": point_id, "collection_name": self._config["learning_collection_name"]})
        except Exception as exc:
            return _json_error(f"Learning approval failed: {exc}")

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        args = args or {}
        if tool_name == "qdrant_memory_status":
            return self._tool_status()
        if tool_name == "qdrant_memory_store":
            return self._tool_store(args)
        if tool_name == "qdrant_memory_search":
            return self._tool_search(args)
        if tool_name == "qdrant_memory_index":
            return self._tool_index(args)
        if tool_name == "qdrant_memory_forget":
            return self._tool_forget(args)
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
        return _json_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None


def register(ctx) -> None:
    ctx.register_memory_provider(QdrantMemoryProvider())
