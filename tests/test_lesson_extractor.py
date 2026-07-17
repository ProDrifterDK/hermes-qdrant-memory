from __future__ import annotations

from qdrant_memory.lesson_extractor import (
    LearningCandidate,
    candidate_to_learning_args,
    contains_secret,
    extract_learning_candidates_from_messages,
    extract_learning_candidates_from_turn,
)


def _credential_assignment(key: str, value: str, *, separator: str = "=") -> str:
    """Build scanner-shaped fixtures without literal credentials in source."""
    return key + " " + separator + " " + value


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
        {"role": "assistant", "content": "The terminal command failed. Correction: use <hermes-venv>/bin/python -m pytest tests -q."},
    ]

    candidates = extract_learning_candidates_from_messages(messages, source_hook="on_session_end", min_confidence=0.8)

    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.learning_type == "tool_failure_lesson"
    assert candidate.tool_name == "terminal"
    assert "pytest" in candidate.trigger
    assert "<hermes-venv>/bin/python -m pytest" in candidate.correction
    assert candidate.confidence >= 0.8


def test_tool_failure_without_correction_is_ignored():
    messages = [
        {"role": "tool", "name": "terminal", "content": "curl failed with HTTP 500"},
        {"role": "assistant", "content": "The command failed."},
    ]

    assert extract_learning_candidates_from_messages(messages, source_hook="on_session_end") == []


def test_secret_bearing_candidate_is_blocked():
    openai_like = "".join(["s", "k", "-", "1234567890abcdef"])
    candidates = extract_learning_candidates_from_turn(
        f"Actually, remember this API key: {openai_like} and use it for tests.",
        "I will use that key.",
    )

    assert candidates == []


def test_bearer_github_aws_and_jwt_secrets_are_blocked():
    bearer_like = " ".join(["Authorization:", "Bearer", "".join(["abc", "def", "ghi", "jkl", "mnop"])])
    aws_like = "".join(["AK", "IA", "1234567890ABCDEF"])
    jwt_like = ".".join([
        "".join(["ey", "J", "hbGciOiJIUzI1NiJ9"]),
        "payloadpayload",
        "signaturesignature",
    ])
    secret_inputs = [
        f"Actually, use {bearer_like} instead.",
        f"Actually, use AWS key {aws_like} for deployment.",
        f"Actually, remember token {jwt_like}.",
    ]

    for text in secret_inputs:
        assert extract_learning_candidates_from_turn(text, "ok") == []


def test_contains_secret_preserves_credential_detection_while_ignoring_token_concepts():
    jwt_like = ".".join([
        "".join(["ey", "J", "hbGciOiJIUzI1NiJ9"]),
        "payloadpayload",
        "signaturesignature",
    ])
    # Build secret-shaped values at runtime so the scanner does not flag
    # this test file as containing literal credentials.
    openai_like = "".join(["s", "k", "-", "abcdef1234567890ABCDE"])
    github_like = "".join(["gh", "p_", "abcdef1234567890ABCDE"])
    aws_like = "".join(["AK", "IA", "1234567890ABCDEF"])
    bearer_token = "".join(["abc", "def", "ghi", "jkl", "mnop", "qrst"])
    long_value = "".join(["abc", "def", "ghi", "jkl", "123", "456", "789"])
    long_alpha = "".join(["abc", "def", "ghi", "jkl", "mno", "pqr", "stu"])
    hunter_long = "".join(["hunter2", "hunter3", "hunter4"])
    bearer_like = " ".join(["Authorization:", "Bearer", bearer_token])
    # Realistic, plausible secret-shaped values that satisfy the
    # ``_looks_like_secret_value`` filter (≥8 chars and contain a digit
    # or a non-alphanumeric character, or are ≥16 chars of pure alpha).
    # We split the keyword, separator, and value across Python string
    # concatenations so the literal-secret scanner does not flag this file.
    secret_kw = "secret"
    api_kw = "api_key"
    pw_kw = "password"
    tok_kw = "token"
    sep_colon = " " + chr(58) + " "
    sep_eq = " " + chr(61) + " "
    quote_dq = chr(34)  # "
    quote_sq = chr(39)  # '
    positive_inputs = [
        f"Actually, remember token {jwt_like}.",
        f"Use access token {long_value} for the call.",
        # Inline assignment shapes (patterns 6, 7)
        api_kw + sep_eq + long_alpha,
        api_kw + sep_eq + long_value,
        pw_kw + sep_eq + hunter_long,
        secret_kw + sep_eq + long_value,
        tok_kw + sep_eq + long_value,
        api_kw + sep_colon + long_alpha,
        pw_kw + sep_colon + long_value,
        secret_kw + sep_colon + ("x" * 18),
        # Quoted inline assignment values — regression fix for quoted RHS.
        pw_kw + sep_eq + quote_dq + hunter_long + quote_dq,
        tok_kw + sep_eq + quote_sq + long_value + quote_sq,
        # Provider / header / marker / URL basic-auth shapes
        openai_like,
        github_like,
        aws_like,
        bearer_like,
    ]
    negative_inputs = [
        "Token Budget Enforcement keeps summaries bounded.",
        "Token Counting Cache improves tokenizer estimates.",
        "The summary token overhead is four tokens per message.",
        "JWT validation strategy uses HS256 for POC and RS256 for prod.",
        # New: ordinary English / placeholder prose should not be flagged.
        "Token \u00d7 cost analysis from benchmark report.",
        "Token Bucket algorithm is unrelated to OAuth tokens.",
        "secret-bearing memory",
        "manual secret review",
        "secret detection is hard",
        "secret redaction works",
        "password detection is heuristic",
        "api key detection in code",
        "api key patterns we look for",
        "password: <REDACTED>",
        "password: <input required>",
        "password: (empty)",
        "when using sudo, prompts for password before continuing",
        "requires a password to login",
        "context token discussion",
    ]

    for text in positive_inputs:
        assert contains_secret(text), text
    for text in negative_inputs:
        assert not contains_secret(text), text


def test_contains_secret_fails_closed_for_explicit_credential_assignments():
    quote = chr(34)
    short_value = "".join(["p", "i", "n"])
    alpha_passphrase = "".join(["correct", "horse", "battery"])
    assignment_keys = [
        "password",
        "api_key",
        "secret",
        "token",
        "passwd",
        "credential",
        "private_key",
    ]

    for key in assignment_keys:
        assert contains_secret(_credential_assignment(key, short_value)), key
    assert contains_secret(_credential_assignment("password", quote + short_value + quote))
    assert contains_secret(_credential_assignment("password", alpha_passphrase, separator=":"))
    assert contains_secret(_credential_assignment("api-key", quote + "two words" + quote))


def test_contains_secret_assignment_value_must_start_on_same_line():
    separator = chr(58)
    fence = chr(96) * 3
    prose_without_value = "La salida termina sin " + "password" + separator
    fenced_output = "\n".join([prose_without_value, fence + "text", "$ command --status", "ok", fence])

    assert not contains_secret(fenced_output)

    quote = chr(34)
    short_value = "".join(["p", "i", "n"])
    for horizontal_whitespace in ("", " ", "\t", " \t"):
        same_line = (
            "password"
            + horizontal_whitespace
            + separator
            + horizontal_whitespace
            + quote
            + short_value
            + quote
        )
        assert contains_secret(same_line), repr(same_line)


def test_contains_secret_allows_only_exact_placeholder_assignment_values():
    placeholders = [
        "<REDACTED>",
        "[REDACTED]",
        "<input required>",
        "(empty)",
        "***",
        "[redacted: possible secret-bearing value]",
    ]

    for placeholder in placeholders:
        assert not contains_secret(_credential_assignment("password", placeholder)), placeholder

    quoted_with_sentence_punctuation = chr(34) + "<REDACTED>" + chr(34) + "."
    assert not contains_secret(_credential_assignment("token", quoted_with_sentence_punctuation))
    assert not contains_secret(_credential_assignment("token", "***."))
    assert contains_secret(_credential_assignment("password", "redacted-value"))


def test_contains_secret_checks_every_loose_pattern_match():
    later_short_value = "".join(["n", "e", "w", "p", "i", "n"])
    assignments = " then ".join(
        [
            _credential_assignment("password", "<REDACTED>"),
            _credential_assignment("password", later_short_value),
        ]
    )
    token_contexts = "token planet then token alpha123"
    credential_contexts = "password planet then password alpha123"

    assert contains_secret(assignments)
    assert contains_secret(token_contexts)
    assert contains_secret(credential_contexts)


def test_write_gate_rejects_short_explicit_credential_assignment():
    from qdrant_memory.write_gate import evaluate_write_candidate

    text = "store this " + _credential_assignment("password", "".join(["p", "i", "n"]))

    decision = evaluate_write_candidate(text=text, source_type="manual")

    assert decision.decision == "reject"
    assert decision.requires_review is True
    assert "possible_secret" in decision.reasons


def test_contains_secret_looks_like_secret_value_helper():
    from qdrant_memory.lesson_extractor import _looks_like_secret_value

    # Real secret shapes — must be classified as secret-like.
    assert _looks_like_secret_value("hunter2hunter3hunter4")
    assert _looks_like_secret_value("mySecretPass123!")
    assert _looks_like_secret_value("abcdefghij1234567890")
    assert _looks_like_secret_value("x" * 18)
    assert _looks_like_secret_value("p4ssw0rd!withSpecial")

    # Placeholder / redaction markers — must be rejected.
    assert not _looks_like_secret_value("<REDACTED>")
    assert not _looks_like_secret_value("<input required>")
    assert not _looks_like_secret_value("(empty)")
    assert not _looks_like_secret_value("[placeholder]")
    assert not _looks_like_secret_value("`code`")
    assert not _looks_like_secret_value("***")

    # Short or pure-alphabetic single English words — must be rejected.
    assert not _looks_like_secret_value("discussion")
    assert not _looks_like_secret_value("discussion!")
    assert not _looks_like_secret_value("short-pass")
    assert not _looks_like_secret_value("Bucket")
    assert not _looks_like_secret_value("before")
    assert not _looks_like_secret_value("hunter2")  # 7 chars
    assert not _looks_like_secret_value("abc1234")  # 7 chars
    assert not _looks_like_secret_value("password")  # 8 chars, pure alpha

    # Empty / None.
    assert not _looks_like_secret_value("")


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
