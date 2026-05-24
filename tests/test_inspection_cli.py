import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_plugin_cli_module():
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_inspection_test", ROOT / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser():
    cli = _load_plugin_cli_module()
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    qdrant_parser = subparsers.add_parser("qdrant")
    cli.register_cli(qdrant_parser)
    qdrant_parser.set_defaults(func=cli.qdrant_command)
    return parser


class FakeQdrant:
    def __init__(self, points=None):
        self.points = points or []
        self.retrieve_calls = []
        self.upsert_calls = []
        self.delete_ids_calls = []
        self.delete_filter_calls = []
        self.update_payload_calls = []
        self.ensure_collection_calls = []

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        self.retrieve_calls.append(
            {
                "name": name,
                "ids": ids,
                "with_payload": with_payload,
                "with_vector": with_vector,
            }
        )
        requested = {str(point_id) for point_id in ids}
        return [point for point in self.points if str(point.get("id")) in requested]

    def upsert(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.upsert_calls.append((args, kwargs))

    def delete_ids(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.delete_ids_calls.append((args, kwargs))

    def delete_filter(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.delete_filter_calls.append((args, kwargs))

    def update_payload(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.update_payload_calls.append((args, kwargs))

    def ensure_collection(self, *args, **kwargs):  # pragma: no cover - should never be reached
        self.ensure_collection_calls.append((args, kwargs))


def _write_config(hermes_home, **overrides):
    hermes_home.mkdir(parents=True, exist_ok=True)
    config = {
        "collection_name": "hermes_memory_test",
        "learning_collection_name": "hermes_learning_test",
        "qdrant_url": "http://127.0.0.1:6333",
        "embedding_url": "http://127.0.0.1:8080/v1",
    }
    config.update(overrides)
    (hermes_home / "qdrant_memory.json").write_text(json.dumps(config), encoding="utf-8")
    return config


def _write_report(hermes_home, report_id="report-ok", proposal_id="duplicate-1"):
    root = hermes_home / "qdrant_memory" / "consolidation"
    root.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "report_id": report_id,
        "created_at": "2026-05-24T06:10:00Z",
        "scope": "both",
        "summary": {"duplicate_cluster": 1},
        "artifact": {"path": str(root / f"report-{report_id}.json"), "proposal_count": 1},
        "proposals": [
            {
                "proposal_id": proposal_id,
                "proposal_type": "duplicate_cluster",
                "collection_name": "hermes_memory_test",
                "affected_ids": ["point-a", "point-b"],
                "suggested_action": "merge_review_only",
                "confidence": 0.95,
                "risk": "medium",
                "evidence": [
                    {"id": "point-a", "reason": "identical normalized text"},
                    {"id": "point-b", "reason": "identical normalized text"},
                ],
                "requires_explicit_approval": True,
            }
        ],
    }
    path = root / f"report-{report_id}.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return report, path


def test_register_cli_adds_inspection_subcommands():
    parser = _parser()

    show = parser.parse_args(["qdrant", "show", "point-1", "--collection", "memory", "--include-vector", "--include-payload", "--json"])
    assert show.qdrant_subcommand == "show"
    assert show.point_id == "point-1"
    assert show.collection == "memory"
    assert show.include_vector is True
    assert show.include_payload is True
    assert show.json is True

    reports_list = parser.parse_args(["qdrant", "reports", "list", "--limit", "5"])
    assert reports_list.qdrant_subcommand == "reports"
    assert reports_list.reports_subcommand == "list"
    assert reports_list.limit == 5

    reports_show = parser.parse_args(["qdrant", "reports", "show", "report-ok", "--json"])
    assert reports_show.reports_subcommand == "show"
    assert reports_show.report_id == "report-ok"
    assert reports_show.json is True

    proposal = parser.parse_args(["qdrant", "proposals", "show", "report-ok", "duplicate-1"])
    assert proposal.qdrant_subcommand == "proposals"
    assert proposal.proposals_subcommand == "show"
    assert proposal.report_id == "report-ok"
    assert proposal.proposal_id == "duplicate-1"


def test_show_point_is_exact_read_only_and_omits_payload_and_vector_by_default(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command
    import qdrant_memory.backup as backup

    hermes_home = tmp_path / "hermes"
    _write_config(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    token_key = "api" + "_key"
    fake_qdrant = FakeQdrant(
        [
            {
                "id": "point-1",
                "payload": {
                    "text": "Durable operational memory about exact report review.",
                    token_key: "secret" + "-value",
                    "source_type": "manual",
                    "created_at": "2026-05-24T06:00:00Z",
                    "importance": 8,
                    "tags": ["ops", "memory"],
                },
                "vector": [0.1, 0.2, 0.3],
            }
        ]
    )
    monkeypatch.setattr(backup, "qdrant_client_from_config", lambda config: fake_qdrant)
    parser = _parser()
    args = parser.parse_args(["qdrant", "show", "point-1", "--collection", "memory"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    assert fake_qdrant.retrieve_calls == [
        {"name": "hermes_memory_test", "ids": ["point-1"], "with_payload": True, "with_vector": False}
    ]
    assert not fake_qdrant.upsert_calls
    assert not fake_qdrant.delete_ids_calls
    assert not fake_qdrant.delete_filter_calls
    assert not fake_qdrant.update_payload_calls
    assert not fake_qdrant.ensure_collection_calls
    human = capsys.readouterr().out
    assert "Point: point-1" in human
    assert "collection: memory (hermes_memory_test)" in human
    assert "source_type: manual" in human
    assert "vector_included: False" in human
    assert "payload_included: False" in human
    assert "secret-value" not in human
    assert "[0.1" not in human
    with pytest.raises(json.JSONDecodeError):
        json.loads(human)


def test_show_point_json_can_include_redacted_payload_and_vector_only_when_requested(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command
    import qdrant_memory.backup as backup

    hermes_home = tmp_path / "hermes"
    _write_config(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    token_key = "access" + "_token"
    fake_qdrant = FakeQdrant(
        [
            {
                "id": "learning-1",
                "payload": {"lesson": "Use exact IDs for report apply.", token_key: "token" + "-value", "learning_type": "workflow_lesson"},
                "vector": [0.1, 0.2, 0.3],
            }
        ]
    )
    monkeypatch.setattr(backup, "qdrant_client_from_config", lambda config: fake_qdrant)
    parser = _parser()
    args = parser.parse_args(["qdrant", "show", "learning-1", "--collection", "learning", "--include-payload", "--include-vector", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert fake_qdrant.retrieve_calls == [
        {"name": "hermes_learning_test", "ids": ["learning-1"], "with_payload": True, "with_vector": True}
    ]
    assert payload["found"] is True
    assert payload["collection"] == "learning"
    assert payload["payload_included"] is True
    assert payload["vector_included"] is True
    assert payload["vector"] == [0.1, 0.2, 0.3]
    assert payload["payload"][token_key] == "[redacted: possible secret-bearing value]"
    assert "token-value" not in json.dumps(payload)


def test_show_point_missing_returns_found_false(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command
    import qdrant_memory.backup as backup

    hermes_home = tmp_path / "hermes"
    _write_config(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    fake_qdrant = FakeQdrant([])
    monkeypatch.setattr(backup, "qdrant_client_from_config", lambda config: fake_qdrant)
    parser = _parser()
    args = parser.parse_args(["qdrant", "show", "missing", "--collection", "memory", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["found"] is False
    assert payload["point_id"] == "missing"
    assert payload["collection_name"] == "hermes_memory_test"


def test_reports_list_and_show_are_local_read_only(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    _write_config(hermes_home)
    report, path = _write_report(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()

    list_args = parser.parse_args(["qdrant", "reports", "list", "--json"])
    assert execute_command(list_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    listing = json.loads(capsys.readouterr().out)
    assert listing["count"] == 1
    assert listing["reports"][0]["report_id"] == report["report_id"]
    assert listing["reports"][0]["proposal_count"] == 1
    assert listing["reports"][0]["path"] == str(path)

    show_args = parser.parse_args(["qdrant", "reports", "show", report["report_id"]])
    assert execute_command(show_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    human = capsys.readouterr().out
    assert f"Report: {report['report_id']}" in human
    assert "scope: both" in human
    assert "duplicate_cluster: 1" in human
    assert "duplicate-1 [duplicate_cluster] action=merge_review_only" in human
    with pytest.raises(json.JSONDecodeError):
        json.loads(human)


def test_proposals_show_returns_exact_proposal_with_expected_action(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    _write_config(hermes_home)
    report, _path = _write_report(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "proposals", "show", report["report_id"], "duplicate-1", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["report_id"] == report["report_id"]
    assert payload["proposal_id"] == "duplicate-1"
    assert payload["expected_action"] == "merge"
    assert payload["proposal"]["proposal_type"] == "duplicate_cluster"
    assert payload["proposal"]["affected_ids"] == ["point-a", "point-b"]


def test_reports_and_proposals_reject_path_traversal_ids(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    _write_config(hermes_home)
    _write_report(hermes_home)
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()

    invalid_cases = [
        ["qdrant", "reports", "show", "../x"],
        ["qdrant", "reports", "show", ".."],
        ["qdrant", "reports", "show", "/tmp/x"],
        ["qdrant", "reports", "show", "a/b"],
        ["qdrant", "proposals", "show", "report-ok", "a/b"],
        ["qdrant", "proposals", "show", "../x", "duplicate-1"],
    ]
    for argv in invalid_cases:
        args = parser.parse_args(argv + ["--json"])
        exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))
        captured = capsys.readouterr()
        assert exit_code == 2
        assert captured.out == ""
        error = json.loads(captured.err)
        assert error["error"] is True
        assert "invalid" in error["message"] or "not found" in error["message"]
