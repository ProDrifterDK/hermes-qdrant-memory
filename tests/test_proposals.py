from __future__ import annotations

from qdrant_memory.consolidation import ConsolidationPoint
from qdrant_memory.proposals import list_proposal_drafts, load_proposal_draft, proposal_root, render_proposal_markdown, write_proposal_draft
from qdrant_memory.write_gate import WriteDecision


def _point(point_id="p1"):
    return ConsolidationPoint(
        id=point_id,
        collection_name="memory",
        text="A useful memory snippet with no secret material.",
        payload={
            "text": "A useful memory snippet with no secret material.",
            "source_uri": "file:///tmp/source.md",
            "source_type": "markdown",
            "locator": {"line_start": 4, "line_end": 6, "heading": "Architecture"},
            "content_hash": "sha256:abc123",
            "derivation_type": "indexed_chunk",
            "derived_from": [{"source_uri": "session://abc", "derivation_type": "completed_turn"}],
            "canonical": True,
            "requires_review": False,
        },
    )


def test_neutral_proposal_writer_uses_profile_safe_generic_root(tmp_path):
    root = proposal_root(str(tmp_path / "hermes"), {})

    assert root == tmp_path / "hermes" / "qdrant_memory" / "proposals"
    assert root.exists()


def test_render_proposal_markdown_includes_chain_of_custody_and_redacts_secrets():
    secret_key = "api" + "_key"
    point = _point()
    point.payload[secret_key] = "secret" + "-value"
    markdown = render_proposal_markdown(
        report={"report_id": "report-1"},
        proposal={"proposal_id": "proposal-1", "proposal_type": "duplicate_cluster", "affected_ids": ["p1"], "suggested_action": "merge_review_only"},
        points=[point],
        write_decision=WriteDecision("draft_review", ["missing_provenance"], 0.6, True),
    )

    assert "report_id: report-1" in markdown
    assert "proposal_id: proposal-1" in markdown
    assert "p1" in markdown
    assert "file:///tmp/source.md" in markdown
    assert "derived_from" in markdown
    assert "secret-value" not in markdown
    assert "neutral review artifact" in markdown


def test_write_list_and_load_proposal_draft_round_trip(tmp_path):
    result = write_proposal_draft(
        report={"report_id": "report-1"},
        proposal={"proposal_id": "proposal-1", "proposal_type": "duplicate_cluster", "affected_ids": ["p1"]},
        points=[_point()],
        hermes_home=str(tmp_path / "hermes"),
        config={},
        write_decision={"decision": "draft_review", "requires_review": True},
    )

    assert result["draft_id"].startswith("report-1-proposal-1-")
    assert result["source_point_ids"] == ["p1"]
    assert result["requires_review"] is True
    assert (tmp_path / "hermes" / "qdrant_memory" / "proposals").exists()
    assert result in list_proposal_drafts(hermes_home=str(tmp_path / "hermes"), config={})

    loaded = load_proposal_draft(result["draft_id"], hermes_home=str(tmp_path / "hermes"), config={})
    assert loaded["draft_id"] == result["draft_id"]
    assert "## Source points" in loaded["markdown"]


def test_proposal_writer_routes_to_obsidian_only_when_explicitly_configured(tmp_path):
    vault = tmp_path / "vault"
    config = {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(vault), "obsidian_proposal_dir": "Qdrant Proposals"}

    root = proposal_root(str(tmp_path / "hermes"), config)

    assert root == vault / "Qdrant Proposals"
    assert root.exists()


def test_proposal_writer_rejects_obsidian_proposal_dir_escape(tmp_path):
    config = {"obsidian_adapter_enabled": True, "obsidian_vault_root": str(tmp_path / "vault"), "obsidian_proposal_dir": "../escape"}

    try:
        proposal_root(str(tmp_path / "hermes"), config)
    except ValueError as exc:
        assert "inside obsidian_vault_root" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected path escape to be rejected")
