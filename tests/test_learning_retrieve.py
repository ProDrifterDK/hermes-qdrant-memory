"""Phase 5 fix4 regression tests for the learning retrieve tool path.

These tests cover the narrow residual P2 findings left over after the
final2 reviewer/security pass:

* ``_learning_hit_secret_bearing`` must include the learning chunk's
  ``chunk.id`` (the value echoed back into ``results.exact_hits[i].point_id``)
  in its secret scan. Secret-shaped ids with clean text/projection are
  dropped to warning-only and the raw id never reaches warnings,
  ``exact_hits``, ``debug``, or the JSON envelope.
* ``LearningStore.search(..., update_access=False)`` is still preserved
  (no access-metadata mutation).
* The learning path never wires through the memory hybrid router
  (no memory-router build, no ``base_retriever.search`` call).
"""

from __future__ import annotations

import json
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _Chunk:
    def __init__(self, pid, text, payload, final_score=0.6, qdrant_score=0.6):
        self.id = pid
        self.text = text
        self.payload = payload
        self.final_score = final_score
        self.qdrant_score = qdrant_score


class FakeLearningStore:
    """Minimal stand-in for ``LearningStore`` capturing call args."""

    def __init__(self, chunks=None):
        self._chunks = chunks or []
        self.calls: list[dict[str, Any]] = []

    def search(self, query, *, top_k=5, update_access=True, **kwargs):
        self.calls.append({
            "query": query,
            "top_k": top_k,
            "update_access": update_access,
            "kwargs": kwargs,
        })
        return list(self._chunks)


class _ProviderBase:
    """Lightweight stand-in for ``QdrantMemoryProvider``.

    The real ``__init__`` calls ``load_config``; we don't want that to
    read or mutate ``~/.hermes/qdrant.yaml``. We re-import the real
    provider class so ``_tool_retrieve_learning`` runs against an
    instance whose method actually lives on the real class.
    """

    def __init__(self):
        self._config = {
            "learning_enabled": True,
            "learning_collection_name": "learnings",
        }
        # Stub the runtime checks inside ``_tool_retrieve_learning``.
        self._qdrant = object()
        self._embeddings = object()
        self._scope_calls: list[dict[str, Any]] = []

    def _scope_filter_values(self) -> dict[str, str]:
        self._scope_calls.append({})
        return {"profile_id": "default"}

    def _ensure_learning_store(self):
        return self._injected_store


@pytest.fixture
def provider_cls():
    """Return a class that exposes ``_tool_retrieve_learning`` for tests.

    We dynamically mix the test-only helpers in ``_ProviderBase`` with
    the real ``QdrantMemoryProvider`` so ``_tool_retrieve_learning``
    runs against the same implementation as production while skipping
    ``__init__``'s config load.
    """
    from __init__ import QdrantMemoryProvider  # type: ignore  # noqa: PLC0415

    class _Provider(_ProviderBase):
        # Bind the bound method from the real provider so the body of
        # ``_tool_retrieve_learning`` runs unchanged.
        _tool_retrieve_learning = QdrantMemoryProvider._tool_retrieve_learning

    return _Provider


# ---------------------------------------------------------------------------
# Adversarial: secret-shaped learning point id
# ---------------------------------------------------------------------------


class TestLearningSecretShapedPointId:
    def test_secret_shaped_id_dropped(self, provider_cls):
        bad_id = "".join(["Bearer ", "1" * 24])
        store = FakeLearningStore(chunks=[
            _Chunk(bad_id, "plain clean learning text",
                   {"profile_id": "default", "learning_type": "fact"}),
            _Chunk("clean-learning-id", "another clean learning",
                   {"profile_id": "default", "learning_type": "fact"}),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)

        # The secret-shaped id must NOT appear in exact_hits.
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert bad_id not in ids
        assert "clean-learning-id" in ids

        # The full envelope must not echo the raw id anywhere.
        assert bad_id not in raw
        assert bad_id not in json.dumps(d, default=str)

        # The dropped id must be reported only via a redacted handle in
        # the warning channel.
        redact_warnings = [w for w in d["warnings"] if "learning exact hit" in w]
        assert redact_warnings
        for w in redact_warnings:
            assert bad_id not in w
            assert "redacted:" in w

        # Debug envelope should expose only the redacted handle, not the
        # raw id.
        dropped = d["debug"].get("dropped_exact_hit_ids", [])
        for handle in dropped:
            assert bad_id not in handle

        # update_access invariant preserved.
        assert store.calls
        assert all(c["update_access"] is False for c in store.calls)

    def test_clean_id_passes_through(self, provider_cls):
        store = FakeLearningStore(chunks=[
            _Chunk("plain-learning-id", "clean text",
                   {"profile_id": "default", "learning_type": "fact"}),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)

        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert ids == ["plain-learning-id"]
        # No spurious learning-exact-hit redaction warning for clean ids.
        assert not any("learning exact hit redacted" in w for w in d["warnings"])


class TestLearningNoMemoryRouterBuild:
    def test_learning_retrieve_does_not_build_memory_router(self, provider_cls):
        # The learning path must NOT trigger ``_ensure_hybrid_router``
        # (which would build a memory ``MemoryRetriever``/``HybridRouter``
        # chain and potentially poison the memory cache).
        store = FakeLearningStore(chunks=[
            _Chunk("clean-id", "clean text",
                   {"profile_id": "default", "learning_type": "fact"}),
        ])
        provider = provider_cls()
        provider._injected_store = store

        # Spy: ``_ensure_hybrid_router`` must never be called.
        called: list[Any] = []
        provider._ensure_hybrid_router = lambda *a, **kw: called.append((a, kw)) or None  # type: ignore[attr-defined]

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        assert d["results"]["exact_hits"]
        assert called == []

# ---------------------------------------------------------------------------
# Phase 5 fix7: learning retrieve failure must NOT leak raw exception text
#
# Regression coverage for finding #3 from the final5 reviewer/security
# pass. ``_tool_retrieve_learning`` used to interpolate ``{exc}`` into
# the JSON error envelope on ``LearningStore.search`` failure. The
# exception ``__str__`` can echo the requested query (a secret-shaped
# token) or other raw backend strings into the JSON response,
# reaching the LLM context downstream through
# ``qdrant_memory_retrieve``. We now return a sanitized error.
# ---------------------------------------------------------------------------


class TestLearningRetrieveFailureNoRawExceptionLeak:
    def test_learning_store_exception_query_not_leaked_in_error(self, provider_cls):
        # Construct the secret-shaped query at runtime so the
        # scanner doesn't trip on a literal in the source file.
        bad_query = "".join(["Bearer ", "u" * 24])

        class _EchoingStore:
            """Fake LearningStore whose ``search`` raises and whose
            exception ``__str__`` echoes the requested query verbatim.
            """

            def __init__(self):
                self.calls: list[dict[str, Any]] = []

            def search(self, query, *, top_k=5, update_access=True, **kwargs):
                self.calls.append({
                    "query": query,
                    "top_k": top_k,
                    "update_access": update_access,
                    "kwargs": kwargs,
                })
                # Echo the full query into the exception ``__str__``
                # so a leak via ``f"...{exc}"`` would surface the
                # secret-shaped token into the JSON error envelope.
                raise RuntimeError(
                    "learning store refused to process query=" + repr(query)
                )

        store = _EchoingStore()
        provider = provider_cls()
        provider._injected_store = store  # type: ignore[attr-defined]

        raw = provider._tool_retrieve_learning({"query": bad_query, "top_k": 5})
        # Raw response is JSON; parse it.
        d = json.loads(raw)
        # The standard error envelope shape from ``_json_error``.
        assert "error" in d
        error_value = d["error"]
        assert isinstance(error_value, str)
        assert bad_query not in error_value, (
            "learning retrieve error leaked the secret-shaped query"
        )
        # Sanitized warning shape from fix7.
        assert "no raw exception leaked" in error_value
        assert "Learning retrieve failed" in error_value
        # Serialized envelope (in case future changes add sibling
        # fields): still no raw secret-shaped query anywhere.
        serialized = json.dumps(d, default=str)
        assert bad_query not in serialized

        # The original store call STILL happened with
        # ``update_access=False`` (read-only invariant preserved).
        assert store.calls
        assert all(c["update_access"] is False for c in store.calls)


# ---------------------------------------------------------------------------
# Phase 5 fix10 (final8 finding #2): the learning retrieve path must
# apply the same active-context status vocabulary, ``max_source_chars``
# per-hit cap, and hard context char budget as the dense memory lane.
# Pre-fix10 the learning path bypassed the safety gate and the budget
# enforcement so a 5000-char learning hit with ``requires_review=true``
# could become a normal active ``results.exact_hit`` despite the
# caller's ``max_source_chars=10``.
# ---------------------------------------------------------------------------


class TestLearningActiveContextStatusSafety:
    """Regression for final8 finding #2 — active-context safety
    vocabulary on the learning path.

    The learning lane MUST demote unsafe-status hits from active
    ``results.exact_hits`` exactly the way the dense memory lane
    does. ``stale=True``, ``requires_review=True``,
    ``consolidation_quarantined=True``, ``raptor_excluded=True`` /
    ``raptor_forgotten=True``, or unsafe ``fact_status`` values
    (``stale``, ``review_required``, ``disputed``, ``deprecated``,
    ``superseded``) all qualify.
    """

    def test_requires_review_learning_hit_not_in_active_exact_hits(self, provider_cls):
        store = FakeLearningStore(chunks=[
            _Chunk(
                "review-pid-1",
                "this learning needs review",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "requires_review": True,
                    "fact_status": "review_required",
                },
            ),
            _Chunk(
                "clean-pid-1",
                "this learning is clean",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        # The review-required hit MUST NOT be active context.
        assert "review-pid-1" not in ids
        assert "clean-pid-1" in ids
        # Warning channel MUST surface a demotion with the
        # redacted handle and the unsafe reasons (no raw id).
        demote_warnings = [
            w for w in d["warnings"]
            if "learning exact hit demoted" in w
        ]
        assert demote_warnings
        for w in demote_warnings:
            assert "review-pid-1" not in w
            assert "redacted:" in w
            # The reasons must mention requires_review OR fact_status:review_required.
            assert (
                "requires_review" in w
                or "fact_status:review_required" in w
            )

    def test_stale_learning_hit_not_in_active_exact_hits(self, provider_cls):
        store = FakeLearningStore(chunks=[
            _Chunk(
                "stale-pid-1",
                "this learning is stale",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "stale": True,
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert "stale-pid-1" not in ids
        demote_warnings = [
            w for w in d["warnings"]
            if "learning exact hit demoted" in w and "stale" in w
        ]
        assert demote_warnings

    def test_quarantined_learning_hit_not_in_active_exact_hits(self, provider_cls):
        store = FakeLearningStore(chunks=[
            _Chunk(
                "quar-pid-1",
                "this learning is quarantined",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "consolidation_quarantined": True,
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert "quar-pid-1" not in ids
        demote_warnings = [
            w for w in d["warnings"]
            if "learning exact hit demoted" in w and "quarantined" in w
        ]
        assert demote_warnings

    def test_unsafe_fact_status_learning_hit_not_in_active_exact_hits(self, provider_cls):
        # Unsafe ``fact_status`` values: stale, review_required,
        # disputed, deprecated, superseded. Use a value that
        # matches the dense-lane vocabulary exactly.
        for unsafe_status in (
            "stale", "review_required", "disputed",
            "deprecated", "superseded",
        ):
            store = FakeLearningStore(chunks=[
                _Chunk(
                    f"unsafe-fs-{unsafe_status}",
                    f"this learning has fact_status={unsafe_status}",
                    {
                        "profile_id": "default",
                        "learning_type": "fact",
                        "fact_status": unsafe_status,
                    },
                ),
            ])
            provider = provider_cls()
            provider._injected_store = store

            raw = provider._tool_retrieve_learning(
                {"query": "anything", "top_k": 5}
            )
            d = json.loads(raw)
            ids = [h["point_id"] for h in d["results"]["exact_hits"]]
            assert f"unsafe-fs-{unsafe_status}" not in ids, (
                f"learning hit with fact_status={unsafe_status!r} leaked "
                f"into active exact_hits; final8 finding #2 regression"
            )

    def test_clean_learning_hit_still_active(self, provider_cls):
        # Sanity: a clean learning hit MUST still reach active
        # exact_hits. The safety gate must not over-trigger.
        store = FakeLearningStore(chunks=[
            _Chunk(
                "clean-pid-A",
                "clean learning text",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "fact_status": "active",
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert "clean-pid-A" in ids
        # No spurious demotion warning.
        assert not any(
            "learning exact hit demoted" in w for w in d["warnings"]
        )


class TestLearningMaxSourceCharsEnforcement:
    """Regression for final8 finding #2 — ``max_source_chars`` on
    the learning path.

    A long learning hit MUST be truncated to ``max_source_chars``
    and the truncated length MUST count against the cumulative
    hard context budget. Pre-fix10 the learning path emitted
    chunk.text verbatim.
    """

    def test_long_learning_hit_truncated_to_max_source_chars(self, provider_cls):
        long_text = "L" * 5000
        store = FakeLearningStore(chunks=[
            _Chunk(
                "long-pid-1",
                long_text,
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning(
            {"query": "anything", "top_k": 5, "max_source_chars": 10}
        )
        d = json.loads(raw)
        hits = d["results"]["exact_hits"]
        assert len(hits) == 1
        hit = hits[0]
        # Per-hit truncation: text must be <= 10 chars.
        assert len(hit["text"]) <= 10, (
            f"learning exact hit text length {len(hit['text'])} but "
            f"max_source_chars=10; final8 finding #2 regression"
        )
        # context_used_chars MUST include this hit (was 0 pre-fix10).
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx > 0
        assert ctx <= 10
        # debug.max_source_chars reflects the cap.
        assert int(d["debug"].get("max_source_chars") or 0) == 10

    def test_max_source_chars_clamped_to_hard_cap(self, provider_cls):
        # A caller asking for ``max_source_chars=100000`` MUST be
        # clamped to the hard ceiling (2400) so the per-hit cap
        # cannot be bypassed.
        long_text = "X" * 5000
        store = FakeLearningStore(chunks=[
            _Chunk(
                "clamp-pid-1",
                long_text,
                {"profile_id": "default", "learning_type": "fact"},
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning(
            {"query": "anything", "top_k": 5, "max_source_chars": 100000}
        )
        d = json.loads(raw)
        # The cap is clamped to HARD_MAX_SOURCE_CHARS=2400; the
        # text is then truncated to <=2400.
        hit = d["results"]["exact_hits"][0]
        assert len(hit["text"]) <= 2400
        assert int(d["debug"].get("max_source_chars") or 0) == 2400


class TestLearningHardContextBudgetEnforcement:
    """Regression for final8 finding #2 — cumulative hard context
    char budget on the learning path.

    The union of all emitted ``exact_hits`` MUST NOT exceed
    ``HARD_CONTEXT_CHAR_BUDGET`` (16000). Overflow hits are
    dropped first-seen-wins with a sanitized warning.
    """

    def test_many_learning_hits_dropped_at_hard_budget(self, provider_cls):
        n_hits = 30
        per_hit = 600  # caller-provided cap; stays 600 after clamp
        # Without the global budget enforcer, 30 × 600 = 18000
        # chars > 16000. The enforcer MUST drop 3 hits to fit.
        store = FakeLearningStore(chunks=[
            _Chunk(
                pid=f"learn-many-{i:02d}",
                text=("Y" * 1000),  # truncated to 600
                payload={"profile_id": "default", "learning_type": "fact"},
                final_score=0.9 - i * 0.001,
            )
            for i in range(n_hits)
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning(
            {"query": "anything", "top_k": 20, "max_source_chars": per_hit}
        )
        d = json.loads(raw)
        hits = d["results"]["exact_hits"]
        # The hard cap holds.
        hard_budget = 16000
        ctx = int(d.get("debug", {}).get("context_used_chars") or 0)
        assert ctx <= hard_budget, (
            f"learning context_used_chars={ctx} exceeds "
            f"HARD_CONTEXT_CHAR_BUDGET={hard_budget}"
        )
        # At least one drop warning must be present, sanitized
        # (no raw ids leaking through). The per-hit drop warnings
        # MUST carry the redacted handle; the trailing summary
        # warning counts the drops but is anonymized on its own.
        budget_warnings = [
            w for w in d["warnings"]
            if "hard context budget" in w
        ]
        assert budget_warnings
        per_hit_warnings = [
            w for w in budget_warnings
            if "hard context budget exceeded" in w
        ]
        assert per_hit_warnings
        for w in per_hit_warnings:
            assert "learn-many-" not in w, (
                "hard budget warning leaked raw point id: " + w
            )
            assert "redacted:" in w
        # First-seen-wins determinism: the lowest-indexed pids
        # must survive.
        emitted_pids = [h["point_id"] for h in hits]
        assert emitted_pids[0] == "learn-many-00"
        # And the tail (highest-indexed) was dropped.
        assert "learn-many-29" not in emitted_pids

    def test_hard_budget_zero_hits_zero_context(self, provider_cls):
        # Sanity: no hits → 0 context used, no warnings.
        store = FakeLearningStore(chunks=[])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        assert d["results"]["exact_hits"] == []
        assert int(d["debug"].get("context_used_chars") or 0) == 0
        assert not any(
            "hard context budget" in w for w in d["warnings"]
        )


class TestLearningWarningNoRawSecretLeak:
    """Regression for final8 finding #2 — warnings must not echo
    raw ids or secret-shaped values.

    Every learning warning that mentions a point id MUST use the
    redacted handle (``redacted:<sha256[:16]>``), never the raw
    id and never the raw text. This is the same invariant the
    dense lane enforces (final4 finding #4). The unsafe-status
    warnings added by fix10 carry the unsafe reasons
    (e.g. ``stale``, ``requires_review``) but NEVER the raw
    payload field values or the raw point id.
    """

    def test_unsafe_status_warning_uses_redacted_handle(self, provider_cls):
        bad_id = "plain-pid-not-secret-but-redacted-by-rule"
        store = FakeLearningStore(chunks=[
            _Chunk(
                bad_id,
                "needs review",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "requires_review": True,
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        # The bad id MUST NOT appear in any warning or in the
        # dropped_exact_hit_ids debug field.
        for w in d["warnings"]:
            assert bad_id not in w, (
                "unsafe-status warning leaked raw point id: " + w
            )
        for handle in d["debug"].get("dropped_exact_hit_ids", []):
            assert bad_id not in handle
        # And it MUST NOT be in the exact_hits list (the gate
        # already enforces that, but the warning channel is the
        # easy place for a regression to slip through).
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert bad_id not in ids

    def test_secret_shaped_text_in_unsafe_status_warning_not_leaked(self, provider_cls):
        # Even when the unsafe-status payload ALSO carries a
        # secret-shaped value (e.g. inside a ``notes`` field),
        # the warning channel MUST NOT echo that value. The
        # gate drops the hit (so it never becomes an active
        # exact_hit) and the warning only carries the unsafe
        # reasons + redacted handle.
        bad_id = "secret-in-unsafe-pid"
        secret_value = "".join(["Bearer ", "k" * 24])
        store = FakeLearningStore(chunks=[
            _Chunk(
                bad_id,
                "this learning has a secret in a metadata field",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "requires_review": True,
                    "notes": "the secret is: " + secret_value,
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning({"query": "anything", "top_k": 5})
        d = json.loads(raw)
        # The hit was dropped (both for status AND for being
        # secret-bearing).
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        assert bad_id not in ids
        # The raw secret-shaped value MUST NOT appear in any
        # warning or in the serialized envelope.
        serialized = json.dumps(d, default=str)
        assert secret_value not in serialized
        for w in d["warnings"]:
            assert secret_value not in w



# ---------------------------------------------------------------------------
# Phase 5 fix11 (final9 finding #1, learning branch): the learning
# retrieve output MUST NOT echo the raw query text. The new contract
# matches the memory hybrid lane:
#   * ``query_length`` — int length of the raw query.
#   * ``query_digest`` — sha256(query)[:16].
#   * ``query_redacted`` — fixed sentinel.
# ---------------------------------------------------------------------------


class TestLearningRetrieveNoRawQueryEcho:
    """Regression for final9 finding #1, learning branch.

    The learning path's ``_tool_retrieve_learning`` used to return
    ``"query": query`` in its output envelope. A caller-supplied
    secret-shaped query (e.g. accidentally pasted Bearer token) would
    echo back through the LLM-facing JSON. The fix projects the
    query into a safe metadata block via the shared
    ``_redact_query_metadata`` helper.
    """

    def test_learning_retrieve_never_echoes_raw_query(self, provider_cls):
        # Runtime-constructed secret-shaped query so the scanner
        # doesn't trip on a literal in the source file.
        bad_query = "".join(["Bearer ", "e" * 24])
        store = FakeLearningStore(chunks=[])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning(
            {"query": bad_query, "top_k": 5}
        )
        d = json.loads(raw)
        # Raw query MUST NOT appear anywhere in the envelope.
        serialized = json.dumps(d, default=str)
        assert bad_query not in serialized, (
            "learning retrieve leaked the raw query into the JSON "
            "envelope; final9 finding #1 regression"
        )
        # The new contract keys are present.
        assert "query_length" in d
        assert "query_digest" in d
        assert "query_redacted" in d
        # The legacy raw ``query`` key is gone.
        assert "query" not in d
        # Length matches the raw input.
        assert d["query_length"] == len(bad_query)
        # Digest is sha256[:16].
        import hashlib
        assert d["query_digest"] == (
            hashlib.sha256(bad_query.encode("utf-8")).hexdigest()[:16]
        )
        # The sentinel is fixed and never carries the raw value.
        assert d["query_redacted"] == (
            "[redacted: query omitted from retrieve output]"
        )

    def test_learning_retrieve_serialized_envelope_no_secret(self, provider_cls):
        # Different secret shape to verify the redaction is shape-
        # independent. The serialized JSON MUST NOT carry the raw
        # secret-shaped query anywhere.
        bad_query = "".join(["api_key=", "f" * 24])
        store = FakeLearningStore(chunks=[])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning(
            {"query": bad_query, "top_k": 5}
        )
        assert bad_query not in raw, (
            "learning retrieve JSON leaked a secret-shaped query; "
            "final9 finding #1 regression"
        )


# ---------------------------------------------------------------------------
# Phase 5 fix11 (final9 finding #3): the learning path's
# active-context status safety gate MUST NOT be short-circuited by
# ``include_fact_history=True``. Pre-fix11 the learning path passed
# the caller's ``include_fact_history`` through to
# ``_dense_payload_unsafe_for_active_context``, which returns
# ``False`` immediately when the flag is set — so a learning hit
# with ``requires_review=True`` and ``fact_status=review_required``
# was emitted as a normal active ``results.exact_hits`` even with
# the fact-history opt-in. The fix forces
# ``include_fact_history=False`` for the safety gate regardless of
# what the caller asked.
# ---------------------------------------------------------------------------


class TestLearningIncludeFactHistoryDoesNotBypassSafetyGate:
    """Regression for final9 finding #3.

    The learning path has no separate non-active history bucket, so
    the only safe behavior is to enforce the active-context safety
    gate unconditionally. A caller-supplied
    ``include_fact_history=True`` MUST NOT cause unsafe-status
    learning hits to flow into the active ``results.exact_hits``
    context.
    """

    def test_include_fact_history_true_still_demotes_review_required(
        self, provider_cls,
    ):
        store = FakeLearningStore(chunks=[
            _Chunk(
                "learning-review-pid-1",
                "this learning needs review",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "requires_review": True,
                    "fact_status": "review_required",
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        raw = provider._tool_retrieve_learning(
            {"query": "anything", "top_k": 5, "include_fact_history": True}
        )
        d = json.loads(raw)
        ids = [h["point_id"] for h in d["results"]["exact_hits"]]
        # The unsafe-status hit MUST NOT be active context even when
        # the caller passed include_fact_history=True. Pre-fix11 the
        # gate short-circuited and let it through.
        assert "learning-review-pid-1" not in ids, (
            "learning hit with requires_review=True and "
            "fact_status=review_required leaked into active "
            "exact_hits when caller passed include_fact_history=True; "
            "final9 finding #3 regression"
        )
        # The demotion warning must surface the unsafe reasons.
        demote_warnings = [
            w for w in d["warnings"]
            if "learning exact hit demoted" in w
        ]
        assert demote_warnings
        for w in demote_warnings:
            assert "learning-review-pid-1" not in w
            assert (
                "requires_review" in w
                or "fact_status:review_required" in w
            )

    def test_include_fact_history_true_demotion_warning_ignored_notice(
        self, provider_cls,
    ):
        # When the caller passes include_fact_history=True on the
        # learning path, the fix surfaces a warning that the flag
        # was ignored so the unsafe status gate could hold. This is
        # operator-facing traceability; the raw query is NEVER
        # echoed in that warning.
        store = FakeLearningStore(chunks=[
            _Chunk(
                "learning-stale-pid-1",
                "this learning is stale",
                {
                    "profile_id": "default",
                    "learning_type": "fact",
                    "stale": True,
                },
            ),
        ])
        provider = provider_cls()
        provider._injected_store = store

        # Build a runtime-constructed secret-shaped query so we can
        # also assert the traceback warning never echoes it.
        bad_query = "".join(["Bearer ", "g" * 24])
        raw = provider._tool_retrieve_learning(
            {"query": bad_query, "top_k": 5, "include_fact_history": True}
        )
        d = json.loads(raw)
        # The stale hit was demoted; the caller-visible
        # ``include_fact_history ignored on learning retrieve``
        # warning is present, and never carries the raw query.
        ignored_warnings = [
            w for w in d["warnings"]
            if "include_fact_history ignored on learning retrieve" in w
        ]
        assert ignored_warnings, (
            "expected a warning that include_fact_history was ignored "
            "on the learning path; final9 finding #3"
        )
        for w in ignored_warnings:
            assert bad_query not in w, (
                "ignored-fact-history warning leaked the raw query"
            )

    def test_include_fact_history_true_still_demotes_unsafe_fact_status(
        self, provider_cls,
    ):
        # Sweep the unsafe ``fact_status`` vocabulary the dense
        # memory lane uses; the learning path MUST honor the same
        # gate even with ``include_fact_history=True``.
        for unsafe_status in (
            "stale", "review_required", "disputed",
            "deprecated", "superseded",
        ):
            store = FakeLearningStore(chunks=[
                _Chunk(
                    f"learning-fs-{unsafe_status}",
                    f"this learning has fact_status={unsafe_status}",
                    {
                        "profile_id": "default",
                        "learning_type": "fact",
                        "fact_status": unsafe_status,
                    },
                ),
            ])
            provider = provider_cls()
            provider._injected_store = store

            raw = provider._tool_retrieve_learning(
                {
                    "query": "anything",
                    "top_k": 5,
                    "include_fact_history": True,
                }
            )
            d = json.loads(raw)
            ids = [h["point_id"] for h in d["results"]["exact_hits"]]
            assert f"learning-fs-{unsafe_status}" not in ids, (
                f"learning hit with fact_status={unsafe_status!r} "
                f"leaked into active exact_hits even with "
                f"include_fact_history=True; final9 finding #3 "
                f"regression"
            )
