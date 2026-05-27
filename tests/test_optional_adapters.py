from __future__ import annotations

from qdrant_memory.proposals import proposal_root
from qdrant_memory.sources import default_registry, expand_point, source_status_for_point


class _FakeQdrant:
    def __init__(self, point: dict):
        self.point = point

    def retrieve(self, collection_name, ids, *, with_payload=True, with_vector=False):
        if self.point.get("id") in ids:
            return [self.point]
        return []


def _point(source_uri: str, locator: dict | None = None) -> dict:
    return {
        "id": "p1",
        "payload": {
            "text": "fallback text",
            "source_uri": source_uri,
            "locator": locator or {"line_start": 2, "line_end": 2},
        },
    }


def test_obsidian_scheme_is_unsupported_by_default():
    expanded = default_registry().expand("obsidian://Note.md", max_chars=100)

    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["scheme"] == "obsidian"


def test_enabled_obsidian_adapter_expands_bounded_note_from_temp_vault(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note.md").write_text("line one\nline two has useful content\nline three\n", encoding="utf-8")
    config = {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(vault)}

    expanded = expand_point(_FakeQdrant(_point("obsidian://Note.md")), "memory", "p1", config=config, max_chars=12)

    assert expanded["supported"] is True
    assert expanded["status"] == "exists"
    assert expanded["text"] == "line two has"
    assert expanded["truncated"] is True
    assert expanded["locator"] == {"line_start": 2, "line_end": 2}


def test_enabled_obsidian_adapter_reports_source_status(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "Note.md").write_text("content\n", encoding="utf-8")
    config = {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(vault)}

    status = source_status_for_point(_FakeQdrant(_point("obsidian://Note.md")), "memory", "p1", config=config)

    assert status["supported"] is True
    assert status["exists"] is True
    assert status["scheme"] == "obsidian"


def test_enabled_obsidian_adapter_rejects_path_traversal(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(vault)}

    expanded = expand_point(_FakeQdrant(_point("obsidian://../secret.md")), "memory", "p1", config=config)

    assert expanded["supported"] is False
    assert expanded["status"] == "unsupported"
    assert expanded["reason"] == "unsafe_obsidian_uri"


def test_obsidian_proposal_routing_requires_explicit_config(tmp_path):
    vault = tmp_path / "vault"
    generic_root = proposal_root(str(tmp_path / "hermes"), {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(vault)})
    obsidian_root = proposal_root(
        str(tmp_path / "hermes"),
        {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(vault), "obsidian_proposal_dir": "Reviews/Qdrant"},
    )

    assert generic_root == tmp_path / "hermes" / "qdrant_memory" / "proposals"
    assert obsidian_root == vault / "Reviews" / "Qdrant"
