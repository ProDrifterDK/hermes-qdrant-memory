"""Phase 6I: hybrid auto-recall prompt injection tests.

Covers:

- Config defaults/coercion/env for the new ``auto_recall_mode`` key.
- ``format_hybrid_for_prompt`` output shape, privacy contract, and
  fallback-to-empty on failure.
- Prefetch hybrid path: when ``auto_recall_mode=hybrid``, prefetch
  returns hybrid-formatted context and falls back to legacy on failure
  or empty hybrid result.
- Queue prefetch primes the selected mode (hybrid or legacy) without
  emitting shadow events.
- Shadow dedup: when active mode is hybrid and shadow mode is hybrid,
  the shadow emission is skipped to avoid duplicate work.
- Privacy: secret-shaped values in source/path/debug/warnings/IDs do
  not appear in the final prompt string.
- Status exposes ``auto_recall_mode`` and ``auto_recall_effective_mode``.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from qdrant_memory.config import DEFAULTS, load_config
from qdrant_memory.retriever import format_hybrid_for_prompt


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------


class TestAutoRecallModeConfig:
    def test_default_is_legacy(self, tmp_path):
        cfg = load_config(hermes_home=str(tmp_path), hermes_config={})
        assert "auto_recall_mode" in DEFAULTS
        assert cfg["auto_recall_mode"] == "legacy"

    def test_hybrid_via_config(self, tmp_path):
        cfg = load_config(
            hermes_home=str(tmp_path),
            hermes_config={
                "qdrant_memory": {
                    "auto_recall_mode": "hybrid",
                }
            },
        )
        assert cfg["auto_recall_mode"] == "hybrid"

    def test_invalid_value_fails_closed_to_legacy(self, tmp_path):
        for bad in ("raptor", "dense", "", "HYBRID!", "promote", "all", None, 42, True):
            cfg = load_config(
                hermes_home=str(tmp_path),
                hermes_config={
                    "qdrant_memory": {
                        "auto_recall_mode": bad,
                    }
                },
            )
            assert cfg["auto_recall_mode"] == "legacy", f"Failed for value: {bad!r}"

    def test_env_override(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_QDRANT_MEMORY_AUTO_RECALL_MODE", "hybrid")
        cfg = load_config(hermes_home=str(tmp_path), hermes_config={})
        assert cfg["auto_recall_mode"] == "hybrid"

    def test_env_override_invalid_falls_back(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HERMES_QDRANT_MEMORY_AUTO_RECALL_MODE", "nonsense")
        cfg = load_config(hermes_home=str(tmp_path), hermes_config={})
        assert cfg["auto_recall_mode"] == "legacy"

    def test_case_insensitive(self, tmp_path):
        cfg = load_config(
            hermes_home=str(tmp_path),
            hermes_config={
                "qdrant_memory": {
                    "auto_recall_mode": "Hybrid",
                }
            },
        )
        assert cfg["auto_recall_mode"] == "hybrid"


# ---------------------------------------------------------------------------
# format_hybrid_for_prompt tests
# ---------------------------------------------------------------------------


class _FakeResult:
    """Minimal duck-typed result with the four lanes."""

    def __init__(self, **kwargs):
        self.summaries = kwargs.get("summaries", [])
        self.cited_leaves = kwargs.get("cited_leaves", [])
        self.exact_hits = kwargs.get("exact_hits", [])
        self.graph_relations = kwargs.get("graph_relations", [])


class TestFormatHybridForPrompt:
    def test_empty_result_returns_empty(self):
        assert format_hybrid_for_prompt(None) == ""
        assert format_hybrid_for_prompt(_FakeResult()) == ""

    def test_basic_output_shape(self):
        result = _FakeResult(
            summaries=[{"text": "Summary of cluster A", "point_id": "secret-id"}],
            exact_hits=[{"text": "Exact hit text", "point_id": "pid-1"}],
        )
        out = format_hybrid_for_prompt(result)
        assert out
        assert "Hybrid" in out
        assert "context with provenance" in out.lower()
        assert "Summary of cluster A" in out
        assert "Exact hit text" in out

    def test_no_point_ids_in_output(self):
        result = _FakeResult(
            exact_hits=[{"text": "safe text", "point_id": "uuid-123-abc"}],
            summaries=[{"text": "parent summary", "point_id": "parent-456"}],
        )
        out = format_hybrid_for_prompt(result)
        assert "uuid-123-abc" not in out
        assert "parent-456" not in out

    def test_no_source_uri_file_path_heading_in_output(self):
        result = _FakeResult(
            exact_hits=[{
                "text": "safe text body",
                "point_id": "pid",
                "source_uri": "https://secret.example.com/path",
                "file_path": "/home/user/secret/file.md",
                "heading": "Secret Heading",
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert "https://secret.example.com/path" not in out
        assert "/home/user/secret/file.md" not in out
        assert "Secret Heading" not in out

    def test_no_query_digest_debug_warnings_in_output(self):
        # These fields are on HybridRouteResult, not on item dicts.
        # But let's simulate items that might carry them as keys.
        result = _FakeResult(
            exact_hits=[{
                "text": "safe text",
                "query_digest": "abcdef0123456789",
                "debug": {"mode": "hybrid", "secret_key": "value"},
                "warnings": ["dense exact hit redacted"],
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert "abcdef0123456789" not in out
        assert "secret_key" not in out
        assert "dense exact hit redacted" not in out

    def test_secret_text_is_excluded(self):
        # Runtime-constructed secret-shaped text
        secret_val = "Bearer " + "sk_live_" + "a" * 24
        result = _FakeResult(
            exact_hits=[{"text": secret_val}],
        )
        out = format_hybrid_for_prompt(result)
        assert secret_val not in out
        assert "Bearer" not in out
        assert "sk_live" not in out

    def test_all_sections_present(self):
        result = _FakeResult(
            summaries=[{"text": "S1"}],
            exact_hits=[{"text": "E1"}],
            cited_leaves=[{"text": "L1"}],
            graph_relations=[{"text": "G1"}],
        )
        out = format_hybrid_for_prompt(result)
        assert "S1" in out
        assert "E1" in out
        assert "L1" in out
        assert "G1" in out

    def test_char_budget_bounded(self):
        # Create many large texts
        big_texts = [{"text": "x" * 500} for _ in range(50)]
        result = _FakeResult(exact_hits=big_texts)
        out = format_hybrid_for_prompt(result)
        # Should be bounded by display_tokens * 4 = 1200 chars
        assert len(out) <= 2000

    def test_exception_returns_empty(self):
        class _Bad:
            @property
            def summaries(self):
                raise RuntimeError("boom")
            exact_hits = []
            cited_leaves = []
            graph_relations = []

        assert format_hybrid_for_prompt(_Bad()) == ""

    def test_non_dict_items_handled(self):
        result = _FakeResult(
            exact_hits=["not a dict", 42, None, {"text": "valid"}],
        )
        out = format_hybrid_for_prompt(result)
        assert "valid" in out
        assert "not a dict" not in out

    def test_context_authority_language(self):
        result = _FakeResult(exact_hits=[{"text": "some text"}])
        out = format_hybrid_for_prompt(result)
        assert "not instructions" in out.lower()


# ---------------------------------------------------------------------------
# Privacy: comprehensive secret-shaped field exclusion
# ---------------------------------------------------------------------------


class TestHybridPromptPrivacy:
    """Guarantee that secret-shaped values from source/path/debug/IDs
    never appear in the final prompt string."""

    # Runtime-constructed secrets (avoids literal-secret-fixture scan)
    @staticmethod
    def _make_secret():
        return "ghp_" + "a" * 36

    @staticmethod
    def _make_token():
        return "Bearer " + "sk_" + "b" * 24

    @staticmethod
    def _make_path():
        return "/home/user/.aws/credentials"

    def test_secret_in_text_excluded(self):
        s = self._make_secret()
        result = _FakeResult(exact_hits=[{"text": f"my token is {s}"}])
        out = format_hybrid_for_prompt(result)
        assert s not in out

    def test_token_in_text_excluded(self):
        t = self._make_token()
        result = _FakeResult(exact_hits=[{"text": f"auth header {t}"}])
        out = format_hybrid_for_prompt(result)
        assert t not in out
        assert "Bearer" not in out

    def test_secret_in_source_uri_never_emitted(self):
        s = self._make_secret()
        result = _FakeResult(
            exact_hits=[{
                "text": "clean text",
                "source_uri": f"https://api.example.com?key={s}",
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert s not in out

    def test_secret_in_file_path_never_emitted(self):
        p = self._make_path()
        result = _FakeResult(
            exact_hits=[{
                "text": "clean text",
                "file_path": p,
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert p not in out

    def test_secret_in_point_id_never_emitted(self):
        s = self._make_secret()
        result = _FakeResult(
            exact_hits=[{"text": "clean", "point_id": s}],
        )
        out = format_hybrid_for_prompt(result)
        assert s not in out

    def test_secret_in_debug_never_emitted(self):
        s = self._make_secret()
        result = _FakeResult(
            exact_hits=[{
                "text": "clean",
                "ranking_debug": {"source_hash_current": s},
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert s not in out

    def test_secret_in_warnings_never_emitted(self):
        s = self._make_secret()
        result = _FakeResult(
            exact_hits=[{
                "text": "clean",
                "warnings": [f"warning with {s}"],
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert s not in out

    def test_secret_in_graph_relation_path_never_emitted(self):
        s = self._make_secret()
        result = _FakeResult(
            graph_relations=[{
                "text": "clean relation",
                "path": [s, "another-id"],
                "relation_path": ["relation_with_" + s],
            }],
        )
        out = format_hybrid_for_prompt(result)
        assert s not in out


# ---------------------------------------------------------------------------
# Provider-level prefetch tests
# ---------------------------------------------------------------------------


def _build_provider(
    *,
    auto_recall_mode: str = "legacy",
    shadow_enabled: bool = False,
    shadow_dir: str = "",
    retriever_results: list | None = None,
    hybrid_result: Any = None,
):
    """Build a minimal QdrantMemoryProvider with stubs for prefetch/queue_prefetch."""
    from __init__ import QdrantMemoryProvider  # type: ignore  # noqa: PLC0415

    class _StubProvider(QdrantMemoryProvider):
        def __init__(self):  # noqa: D401 - test stub
            self._active = True
            self._config = {
                "auto_recall": True,
                "auto_recall_top_k": 5,
                "display_tokens": 300,
                "collection_name": "memory",
                "auto_recall_shadow_enabled": shadow_enabled,
                "auto_recall_shadow_max_per_session": 20,
                "auto_recall_shadow_artifact_dir": shadow_dir,
                "auto_recall_shadow_mode": "hybrid",
                "auto_recall_mode": auto_recall_mode,
                # Keys read by _tool_status:
                "qdrant_url": "http://127.0.0.1:6333",
                "embedding_url": "http://127.0.0.1:8080/v1",
                "embedding_model": "bge-m3",
                "vector_size": 1024,
                "learning_collection_name": "learnings",
                "learning_enabled": True,
                "learning_auto_extract_enabled": False,
                "learning_auto_extract_mode": "preview",
                "source_extraction_enabled": False,
                "source_extraction_mode": "preview",
                "consolidation_enabled": False,
                "consolidation_persist_reports": True,
                "reconsolidation_enabled": False,
                "reconsolidation_report_only": True,
                "sync_turns": True,
            }
            self._retriever = MagicMock()
            self._retriever.search = MagicMock(
                return_value=retriever_results if retriever_results is not None else []
            )
            self._prefetch_cache = {}
            self._prefetch_lock = threading.Lock()
            self._session_id = "test-session"
            self._executor = None
            self._hermes_home = ""
            self._shadow_recorder = None
            self._hybrid_router = None
            self._qdrant = None
            self._embeddings = None
            self._raptor_searcher = None
            self._graph_retriever = None
            self._pending_learning_candidates = {}
            self._pending_extraction_candidates = {}

        def _ensure_hybrid_router(self, collection_name):  # noqa: D401
            if hybrid_result is not None:
                router = MagicMock()
                router.retrieve = MagicMock(return_value=hybrid_result)
                return router
            return None

        def _scope_filter_values(self):  # noqa: D401
            return {"profile_id": "default"}

    provider = _StubProvider()
    provider._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="test-phase6i")
    if shadow_enabled:
        from qdrant_memory.shadow_runtime import ShadowRecorder

        provider._shadow_recorder = ShadowRecorder(
            hermes_home=provider._hermes_home or "/tmp",
            max_per_session=20,
            artifact_dir=shadow_dir,
        )
    return provider


class TestPrefetchLegacyMode:
    """When auto_recall_mode=legacy (default), prefetch behaves exactly
    like the pre-Phase-6I code: dense search + legacy format, no hybrid."""

    def test_legacy_mode_uses_dense_path(self, tmp_path):
        provider = _build_provider(
            auto_recall_mode="legacy",
            retriever_results=[],
        )
        result = provider.prefetch("test query", session_id="s1")
        assert isinstance(result, str)
        # Legacy retriever.search was called
        provider._retriever.search.assert_called_once()

    def test_legacy_mode_no_hybrid_router_call(self, tmp_path):
        hybrid = _FakeResult(exact_hits=[{"text": "hybrid text"}])
        provider = _build_provider(
            auto_recall_mode="legacy",
            hybrid_result=hybrid,
        )
        result = provider.prefetch("test query", session_id="s1")
        # Hybrid text should NOT appear in legacy mode result
        assert "hybrid text" not in result


class TestPrefetchHybridMode:
    """When auto_recall_mode=hybrid, prefetch uses the hybrid path."""

    def test_hybrid_mode_returns_hybrid_formatted(self, tmp_path):
        hybrid = _FakeResult(
            summaries=[{"text": "hybrid summary content"}],
            exact_hits=[{"text": "hybrid exact hit"}],
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=hybrid,
        )
        result = provider.prefetch("test query", session_id="s1")
        assert "hybrid summary content" in result
        assert "hybrid exact hit" in result
        assert "Hybrid" in result

    def test_hybrid_fallback_to_legacy_on_empty(self, tmp_path):
        """If hybrid result is empty, prefetch falls back to legacy."""
        from qdrant_memory.retriever import RetrievedMemory

        # Empty hybrid result
        hybrid = _FakeResult()
        # Legacy retriever returns a hit
        legacy_chunk = RetrievedMemory(
            id="legacy-1",
            text="legacy dense result text",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=hybrid,
            retriever_results=[legacy_chunk],
        )
        result = provider.prefetch("test query", session_id="s1")
        # Should fall back to legacy formatted output
        assert "legacy dense result text" in result

    def test_hybrid_fallback_to_legacy_on_router_none(self, tmp_path):
        """If hybrid router is unavailable, prefetch falls back to legacy."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="legacy-2",
            text="legacy fallback text",
            payload={"source_type": "manual"},
            qdrant_score=0.8,
            final_score=0.8,
        )
        # No hybrid_result → _ensure_hybrid_router returns None
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=None,
            retriever_results=[legacy_chunk],
        )
        result = provider.prefetch("test query", session_id="s1")
        assert "legacy fallback text" in result

    def test_hybrid_fallback_to_legacy_on_exception(self, tmp_path):
        """If hybrid retrieve raises, prefetch falls back to legacy."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="legacy-3",
            text="legacy exception fallback",
            payload={"source_type": "manual"},
            qdrant_score=0.7,
            final_score=0.7,
        )
        # Build a provider with a crashing hybrid router
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=None,
            retriever_results=[legacy_chunk],
        )
        # Override _ensure_hybrid_router to return a router that crashes
        bad_router = MagicMock()
        bad_router.retrieve = MagicMock(side_effect=RuntimeError("crash"))
        provider._ensure_hybrid_router = MagicMock(return_value=bad_router)

        result = provider.prefetch("test query", session_id="s1")
        assert "legacy exception fallback" in result

    def test_hybrid_mode_no_secret_in_prompt(self, tmp_path):
        """Hybrid formatted prompt must not contain secret-shaped values."""
        secret = "Bearer " + "sk_test_" + "x" * 24
        hybrid = _FakeResult(
            exact_hits=[{
                "text": f"clean text without secret",
                "point_id": secret,  # secret in point_id — must NOT leak
                "source_uri": f"https://api?key={secret}",
                "file_path": f"/home/secret/{secret}",
            }],
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=hybrid,
        )
        result = provider.prefetch("test query", session_id="s1")
        assert secret not in result
        assert "Bearer" not in result


class TestQueuePrefetchHybridMode:
    """queue_prefetch primes the selected mode without shadow emission."""

    def test_hybrid_queue_primes_hybrid_cache(self, tmp_path):
        hybrid = _FakeResult(exact_hits=[{"text": "queued hybrid content"}])
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=hybrid,
        )
        provider.queue_prefetch("queued query", session_id="s1")
        provider._executor.shutdown(wait=True)
        cached = provider._prefetch_cache.get("s1", "")
        assert "queued hybrid content" in cached

    def test_hybrid_queue_falls_back_to_legacy(self, tmp_path):
        """If hybrid returns empty, queue_prefetch primes legacy."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="legacy-q",
            text="legacy queue fallback",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        hybrid = _FakeResult()  # empty
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=hybrid,
            retriever_results=[legacy_chunk],
        )
        provider.queue_prefetch("queued query", session_id="s1")
        provider._executor.shutdown(wait=True)
        cached = provider._prefetch_cache.get("s1", "")
        assert "legacy queue fallback" in cached

    def test_queue_prefetch_no_shadow_event(self, tmp_path):
        """queue_prefetch must NEVER emit a shadow event."""
        hybrid = _FakeResult(exact_hits=[{"text": "some text"}])
        provider = _build_provider(
            auto_recall_mode="hybrid",
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=hybrid,
        )
        provider.queue_prefetch("queued query", session_id="s1")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        if events_file.exists():
            lines = [l for l in events_file.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        else:
            lines = []
        assert len(lines) == 0


# ---------------------------------------------------------------------------
# Regression: update_access=False invariant for all auto-recall dense search
# ---------------------------------------------------------------------------


class TestUpdateAccessFalseInvariant:
    """Blocker B1 regression: every dense search inside prefetch() and
    queue_prefetch() must pass ``update_access=False`` so auto-recall
    never mutates Qdrant access metadata.

    Covers:
      - legacy mode prefetch
      - hybrid fallback prefetch (empty hybrid result)
      - hybrid fallback prefetch (router unavailable)
      - hybrid fallback prefetch (router exception)
      - legacy queue_prefetch
      - hybrid fallback queue_prefetch (empty hybrid result)
    """

    @staticmethod
    def _assert_all_calls_read_only(mock_search: MagicMock) -> None:
        """Assert every recorded call to ``_retriever.search`` passed
        ``update_access=False``."""
        assert mock_search.called, "Expected at least one retriever.search call"
        for i, call in enumerate(mock_search.call_args_list):
            # update_access can be passed as a keyword arg or positional
            # (keyword-only after top_k, but we check kwargs too for safety).
            ua = call.kwargs.get("update_access")
            assert ua is False, (
                f"retriever.search call #{i} did not pass update_access=False "
                f"(got {ua!r}). Auto-recall must never mutate access metadata."
            )

    def test_legacy_prefetch_update_access_false(self, tmp_path):
        """Legacy-mode prefetch must call retriever.search with update_access=False."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-legacy-1",
            text="legacy update_access test",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        provider = _build_provider(
            auto_recall_mode="legacy",
            retriever_results=[legacy_chunk],
        )
        provider.prefetch("query-ua-legacy", session_id="s1")
        self._assert_all_calls_read_only(provider._retriever.search)

    def test_hybrid_fallback_prefetch_update_access_false_on_empty(self, tmp_path):
        """Hybrid fallback (empty hybrid result) must use update_access=False."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-fallback-empty",
            text="fallback for empty hybrid",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=_FakeResult(),  # empty
            retriever_results=[legacy_chunk],
        )
        provider.prefetch("query-ua-fallback-empty", session_id="s1")
        self._assert_all_calls_read_only(provider._retriever.search)

    def test_hybrid_fallback_prefetch_update_access_false_on_router_none(self, tmp_path):
        """Hybrid fallback (router unavailable) must use update_access=False."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-fallback-none",
            text="fallback for no router",
            payload={"source_type": "manual"},
            qdrant_score=0.8,
            final_score=0.8,
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=None,  # router returns None
            retriever_results=[legacy_chunk],
        )
        provider.prefetch("query-ua-fallback-none", session_id="s1")
        self._assert_all_calls_read_only(provider._retriever.search)

    def test_hybrid_fallback_prefetch_update_access_false_on_exception(self, tmp_path):
        """Hybrid fallback (router raises) must use update_access=False."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-fallback-exc",
            text="fallback for crashed router",
            payload={"source_type": "manual"},
            qdrant_score=0.7,
            final_score=0.7,
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=None,
            retriever_results=[legacy_chunk],
        )
        bad_router = MagicMock()
        bad_router.retrieve = MagicMock(side_effect=RuntimeError("crash-ua"))
        provider._ensure_hybrid_router = MagicMock(return_value=bad_router)
        provider.prefetch("query-ua-fallback-exc", session_id="s1")
        self._assert_all_calls_read_only(provider._retriever.search)

    def test_legacy_queue_prefetch_update_access_false(self, tmp_path):
        """Legacy-mode queue_prefetch must call retriever.search with update_access=False."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-queue-legacy",
            text="queue legacy update_access",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        provider = _build_provider(
            auto_recall_mode="legacy",
            retriever_results=[legacy_chunk],
        )
        provider.queue_prefetch("queue-query-ua-legacy", session_id="s1")
        provider._executor.shutdown(wait=True)
        self._assert_all_calls_read_only(provider._retriever.search)

    def test_hybrid_fallback_queue_prefetch_update_access_false(self, tmp_path):
        """Hybrid fallback queue_prefetch must use update_access=False."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-queue-fallback",
            text="queue hybrid fallback update_access",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=_FakeResult(),  # empty → fallback to legacy
            retriever_results=[legacy_chunk],
        )
        provider.queue_prefetch("queue-query-ua-fallback", session_id="s1")
        provider._executor.shutdown(wait=True)
        self._assert_all_calls_read_only(provider._retriever.search)

    def test_legacy_prefetch_does_not_call_update_access_metadata(self, tmp_path):
        """Ensure retriever.search with update_access=False does not call
        update_access_metadata on the retriever."""
        from qdrant_memory.retriever import RetrievedMemory

        legacy_chunk = RetrievedMemory(
            id="ua-no-mutate-1",
            text="no mutation check",
            payload={"source_type": "manual"},
            qdrant_score=0.9,
            final_score=0.9,
        )
        provider = _build_provider(
            auto_recall_mode="legacy",
            retriever_results=[legacy_chunk],
        )
        # Spy on update_access_metadata — it should never be called when
        # update_access=False.
        provider._retriever.update_access_metadata = MagicMock()
        provider.prefetch("query-no-mutate", session_id="s1")
        provider._retriever.update_access_metadata.assert_not_called()


# ---------------------------------------------------------------------------
# Shadow dedup tests
# ---------------------------------------------------------------------------


class TestShadowDedup:
    """Phase 6I: when active mode is hybrid and shadow mode is hybrid,
    the shadow emission is skipped to avoid duplicate expensive work."""

    def test_hybrid_active_hybrid_shadow_skips_shadow(self, tmp_path):
        hybrid = _FakeResult(
            summaries=[{"text": "s"}],
            exact_hits=[{"text": "e"}],
        )
        provider = _build_provider(
            auto_recall_mode="hybrid",
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=hybrid,
        )
        result = provider.prefetch("dedup test", session_id="s1")
        provider._executor.shutdown(wait=True)
        # No shadow event should be written because active=hybrid and shadow=hybrid
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert not events_file.exists() or not events_file.read_text(encoding="utf-8").strip()

    def test_legacy_active_hybrid_shadow_emits_shadow(self, tmp_path):
        """When active mode is legacy and shadow is hybrid, shadow still fires
        (the original Phase 6H behavior: compare legacy-vs-hybrid)."""
        hybrid = _FakeResult(
            exact_hits=[{"text": "e"}],
        )
        provider = _build_provider(
            auto_recall_mode="legacy",
            shadow_enabled=True,
            shadow_dir=str(tmp_path / "shadow"),
            hybrid_result=hybrid,
        )
        result = provider.prefetch("shadow test", session_id="s1")
        provider._executor.shutdown(wait=True)
        events_file = tmp_path / "shadow" / "shadow_events.jsonl"
        assert events_file.exists()
        lines = [l for l in events_file.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
        assert len(lines) == 1


# ---------------------------------------------------------------------------
# Status tests
# ---------------------------------------------------------------------------


class TestStatusAutoRecallMode:
    def test_status_exposes_auto_recall_mode(self, tmp_path):
        provider = _build_provider(auto_recall_mode="legacy")
        status = json.loads(provider._tool_status())
        assert status["auto_recall_mode"] == "legacy"
        assert status["auto_recall_effective_mode"] == "legacy"

    def test_status_hybrid_mode(self, tmp_path):
        provider = _build_provider(auto_recall_mode="hybrid")
        status = json.loads(provider._tool_status())
        assert status["auto_recall_mode"] == "hybrid"
        assert status["auto_recall_effective_mode"] == "hybrid"

    def test_status_auto_recall_false_forces_legacy(self, tmp_path):
        provider = _build_provider(auto_recall_mode="hybrid")
        provider._config["auto_recall"] = False
        status = json.loads(provider._tool_status())
        assert status["auto_recall_mode"] == "hybrid"  # configured value
        assert status["auto_recall_effective_mode"] == "legacy"  # but effective is legacy
        assert status["auto_recall"] is False

    def test_status_allowlisted_values_only(self, tmp_path):
        """auto_recall_mode must only expose 'legacy' or 'hybrid'."""
        for bad in ("raptor", "dense", "all", "promote"):
            provider = _build_provider(auto_recall_mode=bad)
            status = json.loads(provider._tool_status())
            assert status["auto_recall_mode"] in ("legacy", "hybrid")
            assert status["auto_recall_effective_mode"] in ("legacy", "hybrid")


# ---------------------------------------------------------------------------
# Auto-recall kill switch test
# ---------------------------------------------------------------------------


class TestAutoRecallKillSwitch:
    def test_auto_recall_false_no_prefetch(self, tmp_path):
        """auto_recall=false is the hard kill switch for all prompt auto-recall."""
        provider = _build_provider(
            auto_recall_mode="hybrid",
            hybrid_result=_FakeResult(exact_hits=[{"text": "should not appear"}]),
        )
        provider._config["auto_recall"] = False
        result = provider.prefetch("test query", session_id="s1")
        assert result == ""
