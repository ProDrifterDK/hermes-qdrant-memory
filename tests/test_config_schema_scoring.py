from __future__ import annotations

from datetime import datetime, timedelta, timezone

from qdrant_memory.config import DEFAULTS, load_config
from qdrant_memory.schema import build_payload, clean_text_for_memory, make_point_id, score_importance
from qdrant_memory.scoring import final_memory_score, normalize_minmax, recency_score


def test_config_defaults_when_no_file(tmp_path):
    cfg = load_config(hermes_home=str(tmp_path), hermes_config={})
    assert cfg["enabled"] is True
    assert cfg["qdrant_url"] == "http://127.0.0.1:6333"
    assert cfg["embedding_model"] == "bge-m3"
    assert cfg["vector_size"] == 1024
    assert cfg["learning_auto_extract_semantic_dedupe_enabled"] is True
    assert cfg["learning_auto_extract_semantic_dedupe_threshold"] == 0.9
    assert cfg["learning_auto_extract_semantic_dedupe_top_k"] == 3
    assert set(DEFAULTS).issubset(cfg)


def test_config_overrides_from_qdrant_memory_section(tmp_path):
    cfg = load_config(
        hermes_home=str(tmp_path),
        hermes_config={
            "qdrant_memory": {
                "enabled": "false",
                "auto_recall_top_k": "9",
                "distance": "Dot",
                "learning_auto_extract_semantic_dedupe_enabled": "false",
                "learning_auto_extract_semantic_dedupe_threshold": "0.93",
                "learning_auto_extract_semantic_dedupe_top_k": "7",
            }
        },
    )
    assert cfg["enabled"] is False
    assert cfg["auto_recall_top_k"] == 9
    assert cfg["distance"] == "Dot"
    assert cfg["learning_auto_extract_semantic_dedupe_enabled"] is False
    assert cfg["learning_auto_extract_semantic_dedupe_threshold"] == 0.93
    assert cfg["learning_auto_extract_semantic_dedupe_top_k"] == 7


def test_point_id_is_deterministic_and_uuid_compatible():
    one = make_point_id("manual", "remember alpha")
    two = make_point_id("manual", "remember alpha")
    assert one == two
    assert len(one) == 36


def test_payload_contains_scope_and_defaults():
    payload = build_payload(
        text="hello",
        source="unit",
        source_type="manual",
        profile_id="coder",
        platform="cli",
        session_id="s1",
    )
    assert payload["text"] == "hello"
    assert payload["importance"] == 5
    assert payload["profile_id"] == "coder"
    assert payload["platform"] == "cli"
    assert payload["session_id"] == "s1"
    assert payload["access_count"] == 0


def test_importance_scoring_prefers_explicit_memory():
    assert score_importance("please remember this important decision", "manual") >= 8
    assert score_importance("ok", "conversation") <= 3


def test_clean_text_strips_injected_memory_blocks():
    text = "before\n# Relevant Long-Term Memory\nsecret retrieved memory\n# Actual Answer\nafter\n# Past Learnings\nold lesson\n"
    cleaned = clean_text_for_memory(text)
    assert "secret retrieved memory" not in cleaned
    assert "old lesson" not in cleaned
    assert "before" in cleaned


def test_normalize_equal_scores_to_one():
    assert normalize_minmax([0.5, 0.5]) == [1.0, 1.0]


def test_recency_older_is_lower():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    new = now.isoformat()
    old = (now - timedelta(hours=10)).isoformat()
    assert recency_score(new, 0.01, now=now) > recency_score(old, 0.01, now=now)


def test_final_score_importance_matters():
    now = datetime.now(timezone.utc).isoformat()
    assert final_memory_score(1.0, 10, now, 0.001) > final_memory_score(1.0, 1, now, 0.001)
