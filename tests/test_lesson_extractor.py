from __future__ import annotations

from qdrant_memory.lesson_extractor import (
    LearningCandidate,
    candidate_to_learning_args,
    contains_secret,
    extract_learning_candidates_from_messages,
    extract_learning_candidates_from_turn,
)


def test_extract_user_correction_candidate_requires_explicit_correction():
    candidates = extract_learning_candidates_from_turn(
        "Actually, my surname is Gárate, not Garate.",
        "Thanks, I will spell it as Gárate from now on.",
        source_hook="sync_turn",
        min_confidence=0.8,
    )

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.learning_type == "user_correction"
    assert "Gárate" in candidate.lesson
    assert "Garate" in candidate.mistake
    assert "Gárate" in candidate.correction
    assert candidate.confidence >= 0.8
    assert "auto_candidate" in candidate.tags
    assert candidate.gated is True


def test_no_candidate_for_generic_conversation():
    candidates = extract_learning_candidates_from_turn(
        "Tell me about Qdrant collections.",
        "Qdrant collections hold vectors and payloads.",
    )

    assert candidates == []


def test_extract_tool_failure_candidate_only_with_resolution():
    messages = [
        {"role": "user", "content": "Run the tests."},
        {"role": "tool", "name": "terminal", "content": "pytest: command not found"},
        {"role": "assistant", "content": "The terminal command failed. Correction: use /home/prodrifterdk/.hermes/hermes-agent/venv/bin/python -m pytest tests -q."},
    ]

    candidates = extract_learning_candidates_from_messages(messages, source_hook="on_session_end", min_confidence=0.8)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.learning_type == "tool_failure_lesson"
    assert candidate.tool_name == "terminal"
    assert "pytest" in candidate.trigger
    assert "venv/bin/python -m pytest" in candidate.correction
    assert candidate.confidence >= 0.8


def test_tool_failure_without_correction_is_ignored():
    messages = [
        {"role": "tool", "name": "terminal", "content": "curl failed with HTTP 500"},
        {"role": "assistant", "content": "The command failed."},
    ]

    assert extract_learning_candidates_from_messages(messages, source_hook="on_session_end") == []


def test_secret_bearing_candidate_is_blocked():
    candidates = extract_learning_candidates_from_turn(
        "Actually, remember this API key: sk-1234567890abcdef and use it for tests.",
        "I will use that key.",
    )

    assert candidates == []


def test_bearer_github_aws_and_jwt_secrets_are_blocked():
    secret_inputs = [
        "Actually, use Authorization: Bearer ghp_1234567890abcdef1234567890abcdef123456 instead.",
        "Actually, use AWS key AKIA1234567890ABCDEF for deployment.",
        "Actually, remember token eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signature.",
    ]

    for text in secret_inputs:
        assert extract_learning_candidates_from_turn(text, "ok") == []


def test_contains_secret_preserves_credential_detection_while_ignoring_token_concepts():
    positive_inputs = [
        "Actually, remember token eyJhbG...ture.",
        "Use access token abc1234 for the call.",
        "password hunter2",
        "secret abc123",
        "api_key abc123",
        "api_key: abcdefghijklmno",
        "TOKEN=***",
        "Authorization: " + "Bearer " + "".join(["abc", "def", "ghi", "jkl", "mnop"]),
    ]
    negative_inputs = [
        "Token Budget Enforcement keeps summaries bounded.",
        "Token Counting Cache improves tokenizer estimates.",
        "The summary token overhead is four tokens per message.",
        "JWT validation strategy uses HS256 for POC and RS256 for prod.",
    ]

    for text in positive_inputs:
        assert contains_secret(text), text
    for text in negative_inputs:
        assert not contains_secret(text), text


def test_candidate_ids_are_stable_and_args_match_learning_store_shape():
    one = LearningCandidate(
        lesson="Use the Hermes venv for pytest.",
        learning_type="environment_quirk",
        trigger="pytest command not found",
        correction="run venv/bin/python -m pytest",
        source_hook="unit_test",
    )
    two = LearningCandidate(
        lesson="Use the Hermes venv for pytest.",
        learning_type="environment_quirk",
        trigger="pytest command not found",
        correction="run venv/bin/python -m pytest",
        source_hook="unit_test",
    )

    assert one.candidate_id == two.candidate_id
    args = candidate_to_learning_args(one)
    assert args["lesson"] == one.lesson
    assert args["learning_type"] == "environment_quirk"
    assert args["trigger"] == "pytest command not found"
    assert args["correction"] == "run venv/bin/python -m pytest"
    assert "auto_candidate" in args["tags"]


def test_candidate_limit_is_enforced():
    messages = []
    for idx in range(10):
        messages.extend(
            [
                {"role": "tool", "name": "terminal", "content": f"command {idx} failed with exit code 1"},
                {"role": "assistant", "content": f"Correction: rerun command {idx} with --fixed."},
            ]
        )

    candidates = extract_learning_candidates_from_messages(messages, max_candidates=3, min_confidence=0.8)

    assert len(candidates) == 3
