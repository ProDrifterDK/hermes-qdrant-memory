from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
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
from qdrant_memory.embeddings import EmbeddingClient
from qdrant_memory.indexer import FileIndexer
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
            "qdrant_memory_store, qdrant_memory_index, and qdrant_memory_status for explicit memory operations."
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
        if reset:
            with self._prefetch_lock:
                self._prefetch_cache.clear()

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
            "learning_point_count": learning_point_count,
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
        return _json_error(f"Unknown tool: {tool_name}")

    def shutdown(self) -> None:
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None


def register(ctx) -> None:
    ctx.register_memory_provider(QdrantMemoryProvider())
