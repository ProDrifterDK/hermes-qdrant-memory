from __future__ import annotations

import argparse
import importlib.util
import json
import stat
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_plugin_cli_module():
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_backup_test", ROOT / "cli.py")
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
    def __init__(self, by_collection=None, vector_size=2):
        self.by_collection = by_collection or {}
        self.vector_size: Any = vector_size
        self.scrolls = []
        self.retrieves = []
        self.upserts = []
        self.deleted_ids = []
        self.deleted_filters = []
        self.ensure_calls = []

    def scroll_by_filter(self, name, filter, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.scrolls.append(
            {
                "name": name,
                "filter": filter,
                "limit": limit,
                "with_payload": with_payload,
                "with_vector": with_vector,
                "max_total": max_total,
            }
        )
        points = list(self.by_collection.get(name, []))
        must = filter.get("must", []) if isinstance(filter, dict) else []

        def matches(point):
            payload = point.get("payload") or {}
            for clause in must:
                if "nested" in clause:
                    return False
                actual = payload.get(clause.get("key"))
                match = clause.get("match") or {}
                if "value" in match:
                    expected = match["value"]
                    if actual != expected and not (
                        isinstance(actual, list) and expected in actual
                    ):
                        return False
                if "any" in match and actual not in match["any"]:
                    return False
            return True

        points = [point for point in points if matches(point)]
        return points[:max_total] if max_total is not None else points

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        self.retrieves.append({"name": name, "ids": list(ids), "with_payload": with_payload, "with_vector": with_vector})
        wanted = {str(item) for item in ids}
        return [point for point in self.by_collection.get(name, []) if str(point.get("id")) in wanted]

    def upsert(self, name, points):
        self.upserts.append((name, list(points)))

    def delete_ids(self, name, ids):
        self.deleted_ids.append((name, list(ids)))

    def delete_filter(self, name, filter):
        self.deleted_filters.append((name, filter))

    def ensure_collection(self, name, vector_size, distance):
        self.ensure_calls.append((name, vector_size, distance))

    def collection_vector_size(self, name):
        if isinstance(self.vector_size, dict):
            return self.vector_size.get(name)
        return self.vector_size

    def collection_info(self, name):
        return {"config": {"params": {"vectors": {"size": self.collection_vector_size(name), "distance": "Cosine"}}}}


class FailingScrollQdrant(FakeQdrant):
    def __init__(self, failure_message):
        super().__init__({})
        self.failure_message = failure_message

    def scroll_by_filter(self, *args, **kwargs):
        raise RuntimeError(self.failure_message)


def _point(point_id, text, vector=None, **payload):
    return {"id": point_id, "vector": vector or [0.1, 0.2], "payload": {"text": text, **payload}}


def _write_config(monkeypatch, tmp_path, **overrides):
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    config = {
        "qdrant_url": "http://local-qdrant.invalid:6333",
        "collection_name": "memory",
        "learning_collection_name": "learnings",
        "vector_size": 2,
        "distance": "Cosine",
    }
    config.update(overrides)
    (hermes_home / "qdrant_memory.json").write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    return hermes_home


def _install_fake_qdrant(monkeypatch, fake):
    import qdrant_memory.backup as backup

    monkeypatch.setattr(backup, "QdrantClient", lambda *args, **kwargs: fake)
    return fake


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_register_cli_exposes_export_backup_restore_and_apply_backup_first():
    parser = _parser()

    export_memory = parser.parse_args(["qdrant", "export", "memory", "--out", "memory.jsonl", "--json"])
    assert export_memory.qdrant_subcommand == "export"
    assert export_memory.export_scope == "memory"
    assert export_memory.out == "memory.jsonl"
    assert export_memory.overwrite is False
    assert export_memory.json is True

    export_learning = parser.parse_args(["qdrant", "export", "learning", "--out", "learning.jsonl", "--overwrite"])
    assert export_learning.export_scope == "learning"
    assert export_learning.overwrite is True

    backup_create = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert backup_create.qdrant_subcommand == "backup"
    assert backup_create.backup_subcommand == "create"
    assert backup_create.scope == "memory"

    backup_list = parser.parse_args(["qdrant", "backup", "list", "--json"])
    assert backup_list.backup_subcommand == "list"

    backup_inspect = parser.parse_args(["qdrant", "backup", "inspect", "backup-20260101T000000Z-abcdef12", "--json"])
    assert backup_inspect.backup_subcommand == "inspect"
    assert backup_inspect.backup_id == "backup-20260101T000000Z-abcdef12"

    restore_default = parser.parse_args(["qdrant", "restore", "--backup", "backup-20260101T000000Z-abcdef12"])
    assert restore_default.qdrant_subcommand == "restore"
    assert restore_default.dry_run is True
    assert restore_default.approve is False
    assert restore_default.backup_first is False

    restore_live = parser.parse_args(["qdrant", "restore", "--backup", "backup-20260101T000000Z-abcdef12", "--no-dry-run", "--approve", "--backup-first"])
    assert restore_live.dry_run is False
    assert restore_live.approve is True
    assert restore_live.backup_first is True

    apply_live = parser.parse_args(
        [
            "qdrant",
            "apply",
            "--report-id",
            "report-1",
            "--proposal-id",
            "proposal-1",
            "--action",
            "delete",
            "--no-dry-run",
            "--approve",
            "--backup-first",
        ]
    )
    assert apply_live.backup_first is True
    from qdrant_memory.cli_core import build_tool_call

    tool_name, tool_args = build_tool_call(apply_live)
    assert tool_name == "qdrant_memory_consolidation_apply"
    assert tool_args["backup_first"] is True


def test_restore_live_approval_gate_exits_before_provider_or_qdrant(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant())
    parser = _parser()
    args = parser.parse_args(["qdrant", "restore", "--backup", "backup-20260101T000000Z-abcdef12", "--no-dry-run", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 2
    assert "--approve is required" in capsys.readouterr().err
    assert fake.scrolls == []
    assert fake.retrieves == []
    assert fake.upserts == []


def test_export_memory_writes_jsonl_with_vectors_and_safe_stdout(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    raw_text = "raw memory text alpha"
    _write_config(monkeypatch, tmp_path)
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point("m1", raw_text, source_type="manual")]}))
    out_path = tmp_path / "memory-export.jsonl"
    parser = _parser()
    args = parser.parse_args(["qdrant", "export", "memory", "--out", str(out_path), "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    stdout = capsys.readouterr().out
    assert raw_text not in stdout
    assert "0.1" not in stdout
    payload = json.loads(stdout)
    assert payload["scope"] == "memory"
    assert payload["count"] == 1
    assert payload["path"] == str(out_path)
    assert _mode(out_path) == 0o600
    lines = out_path.read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["artifact_type"] == "qdrant_memory_export"
    record = json.loads(lines[1])
    assert record["id"] == "m1"
    assert record["payload"]["text"] == raw_text
    assert record["vector"] == [0.1, 0.2]
    assert len(record["point_sha256"]) == 64
    assert fake.scrolls == [
        {"name": "memory", "filter": {}, "limit": 256, "with_payload": True, "with_vector": True, "max_total": None}
    ]
    assert fake.upserts == []
    assert fake.deleted_ids == []
    assert fake.deleted_filters == []
    assert fake.ensure_calls == []


def test_export_default_human_summary_includes_checksum_without_payload_leak(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    raw_text = "raw export default text"
    export_qdrant_url = "http://" + "export-user" + ":" + "export-pass" + "@local-qdrant.invalid:6333"
    _write_config(monkeypatch, tmp_path, qdrant_url=export_qdrant_url)
    _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point("m1", raw_text, [0.1, 0.2], source_type="manual")]}))
    out_path = tmp_path / "memory-export.jsonl"
    parser = _parser()
    args = parser.parse_args(["qdrant", "export", "memory", "--out", str(out_path)])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "Exported memory collection" in stdout
    assert f"path: {out_path}" in stdout
    assert "count: 1" in stdout
    assert "sha256:" in stdout
    assert raw_text not in stdout
    assert "0.1" not in stdout
    assert "export-pass" not in stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)


def test_export_refuses_existing_file_without_overwrite(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point("m1", "raw memory text beta")]}))
    out_path = tmp_path / "memory-export.jsonl"
    out_path.write_text("existing", encoding="utf-8")
    parser = _parser()
    args = parser.parse_args(["qdrant", "export", "memory", "--out", str(out_path), "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 2
    assert "already exists" in capsys.readouterr().err
    assert out_path.read_text(encoding="utf-8") == "existing"
    assert fake.scrolls == []


def test_backup_create_list_inspect_use_safe_artifacts_and_redacted_summaries(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    memory_text = "raw backup memory text"
    learning_text = "raw backup learning text"
    qdrant_url = "http://" + "backup-user" + ":" + "backup-pass" + "@local-qdrant.invalid:6333"
    hermes_home = _write_config(monkeypatch, tmp_path, qdrant_url=qdrant_url)
    fake = _install_fake_qdrant(
        monkeypatch,
        FakeQdrant({"memory": [_point("m1", memory_text)], "learnings": [_point("l1", learning_text, source_type="learning")]}),
    )
    parser = _parser()

    create_args = parser.parse_args(["qdrant", "backup", "create", "--json"])
    create_exit = execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    create_stdout = capsys.readouterr().out
    assert create_exit == 0
    assert memory_text not in create_stdout
    assert learning_text not in create_stdout
    assert "backup-pass" not in create_stdout
    created = json.loads(create_stdout)
    backup_id = created["backup_id"]
    backup_root = hermes_home / "qdrant_memory" / "backups"
    backup_dir = backup_root / backup_id
    assert _mode(backup_root) == 0o700
    assert _mode(backup_dir) == 0o700
    manifest_path = backup_dir / "manifest.json"
    memory_path = backup_dir / "memory.jsonl"
    learning_path = backup_dir / "learning.jsonl"
    assert _mode(manifest_path) == 0o600
    assert _mode(memory_path) == 0o600
    assert _mode(learning_path) == 0o600
    manifest_text = manifest_path.read_text(encoding="utf-8")
    assert memory_text not in manifest_text
    assert learning_text not in manifest_text
    assert "backup-pass" not in manifest_text
    manifest = json.loads(manifest_text)
    assert manifest["contains_raw_memory_text"] is True
    assert manifest["contains_vectors"] is True
    assert manifest["collections"]["memory"]["count"] == 1
    assert manifest["collections"]["learning"]["count"] == 1
    assert fake.upserts == []
    assert fake.deleted_ids == []
    assert fake.ensure_calls == []

    list_args = parser.parse_args(["qdrant", "backup", "list", "--json"])
    assert execute_command(list_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    list_stdout = capsys.readouterr().out
    assert memory_text not in list_stdout
    listed = json.loads(list_stdout)
    assert listed["count"] == 1
    assert listed["backups"][0]["backup_id"] == backup_id

    inspect_args = parser.parse_args(["qdrant", "backup", "inspect", backup_id, "--json"])
    assert execute_command(inspect_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    inspect_stdout = capsys.readouterr().out
    assert memory_text not in inspect_stdout
    inspected = json.loads(inspect_stdout)
    assert inspected["backup_id"] == backup_id
    assert inspected["checksum_ok"] is True
    assert inspected["collections"]["memory"]["count"] == 1

def test_backup_default_human_summaries_include_safe_artifact_metadata(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    raw_text = "raw default backup text"
    qdrant_url = "http://" + "human-user" + ":" + "human-pass" + "@local-qdrant.invalid:6333"
    hermes_home = _write_config(monkeypatch, tmp_path, qdrant_url=qdrant_url)
    _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point("m1", raw_text, [0.1, 0.2])]}))
    parser = _parser()

    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    create_stdout = capsys.readouterr().out
    assert "Created backup:" in create_stdout
    assert "backup_id:" in create_stdout
    assert "scope: memory" in create_stdout
    assert "memory: 1 point" in create_stdout
    assert "sha256:" in create_stdout
    assert raw_text not in create_stdout
    assert "0.1" not in create_stdout
    assert "human-pass" not in create_stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(create_stdout)

    backup_id = next((hermes_home / "qdrant_memory" / "backups").iterdir()).name

    list_args = parser.parse_args(["qdrant", "backup", "list"])
    assert execute_command(list_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    list_stdout = capsys.readouterr().out
    assert "Backups: 1" in list_stdout
    assert backup_id in list_stdout
    assert "memory: 1 point" in list_stdout
    assert "sha256:" in list_stdout
    assert raw_text not in list_stdout
    assert "human-pass" not in list_stdout

    inspect_args = parser.parse_args(["qdrant", "backup", "inspect", backup_id])
    assert execute_command(inspect_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    inspect_stdout = capsys.readouterr().out
    assert "Backup:" in inspect_stdout
    assert backup_id in inspect_stdout
    assert "checksum_ok: True" in inspect_stdout
    assert "memory: 1 point" in inspect_stdout
    assert "sha256:" in inspect_stdout
    assert raw_text not in inspect_stdout
    assert "human-pass" not in inspect_stdout



def test_restore_dry_run_reports_same_changed_missing_without_mutation(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    backup_points = {
        "memory": [
            _point("m1", "restore memory alpha", [0.1, 0.2]),
            _point("m2", "restore memory beta", [0.2, 0.3]),
            _point("m3", "restore memory gamma", [0.3, 0.4]),
        ],
        "learnings": [],
    }
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant(backup_points))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection = {
        "memory": [
            _point("m1", "restore memory alpha", [0.1, 0.2]),
            _point("m2", "restore memory beta changed", [0.2, 0.3]),
        ]
    }

    restore_args = parser.parse_args(["qdrant", "restore", "--backup", backup_id, "--json"])
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "restore memory" not in stdout
    result = json.loads(stdout)
    assert result["dry_run"] is True
    assert result["validated"] is True
    assert result["collections"]["memory"]["same"] == 1
    assert result["collections"]["memory"]["changed"] == 1
    assert result["collections"]["memory"]["missing"] == 1
    assert result["collections"]["memory"]["would_upsert"] == 2
    assert fake.retrieves == [{"name": "memory", "ids": ["m1", "m2", "m3"], "with_payload": True, "with_vector": True}]
    assert fake.upserts == []
    assert fake.deleted_ids == []
    assert fake.deleted_filters == []
    assert fake.ensure_calls == []


def test_restore_default_human_summary_reports_counts_without_payload_leak(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    restore_qdrant_url = "http://" + "restore-user" + ":" + "restore-pass" + "@local-qdrant.invalid:6333"
    _write_config(monkeypatch, tmp_path, qdrant_url=restore_qdrant_url)
    backup_points = {
        "memory": [
            _point("m1", "restore default alpha", [0.1, 0.2]),
            _point("m2", "restore default beta", [0.2, 0.3]),
        ]
    }
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant(backup_points))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection = {"memory": [_point("m1", "restore default alpha", [0.1, 0.2])]}

    restore_args = parser.parse_args(["qdrant", "restore", "--backup", backup_id])
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert f"Restore dry-run: {backup_id}" in stdout
    assert "memory: total=2 same=1 changed=0 missing=1 would_upsert=1" in stdout
    assert "restore default" not in stdout
    assert "0.1" not in stdout
    assert "restore-pass" not in stdout
    with pytest.raises(json.JSONDecodeError):
        json.loads(stdout)
    assert fake.upserts == []


def test_restore_dry_run_matches_numeric_point_ids(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point(7, "numeric restore alpha", [0.1, 0.2])]}))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]

    restore_args = parser.parse_args(["qdrant", "restore", "--backup", backup_id, "--json"])
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    result = json.loads(capsys.readouterr().out)
    assert result["collections"]["memory"]["same"] == 1
    assert result["collections"]["memory"]["missing"] == 0
    assert result["collections"]["memory"]["would_upsert"] == 0
    assert fake.upserts == []


def test_restore_rejects_tampered_backup_before_qdrant_lookup(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = _write_config(monkeypatch, tmp_path)
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point("m1", "restore memory tamper")]}))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    memory_path = hermes_home / "qdrant_memory" / "backups" / backup_id / "memory.jsonl"
    with memory_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    restore_args = parser.parse_args(["qdrant", "restore", "--backup", backup_id, "--json"])
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 2
    assert "checksum" in capsys.readouterr().err.lower()
    assert fake.retrieves == []
    assert fake.upserts == []


def test_restore_live_upserts_changed_and_missing_only_with_backup_first(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = _write_config(monkeypatch, tmp_path)
    backup_points = {
        "memory": [
            _point("m1", "live restore alpha", [0.1, 0.2]),
            _point("m2", "live restore beta", [0.2, 0.3]),
            _point("m3", "live restore gamma", [0.3, 0.4]),
        ]
    }
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant(backup_points))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    original_backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection = {
        "memory": [
            _point("m1", "live restore alpha", [0.1, 0.2]),
            _point("m2", "live restore beta changed", [0.2, 0.3]),
        ],
        "learnings": [],
    }

    restore_args = parser.parse_args(
        ["qdrant", "restore", "--backup", original_backup_id, "--no-dry-run", "--approve", "--backup-first", "--json"]
    )
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    stdout = capsys.readouterr().out
    assert exit_code == 0
    assert "live restore" not in stdout
    result = json.loads(stdout)
    assert result["dry_run"] is False
    assert result["applied"] is True
    assert result["pre_restore_backup_id"] != original_backup_id
    assert result["collections"]["memory"]["upserted"] == 2
    assert fake.upserts == [("memory", [backup_points["memory"][1], backup_points["memory"][2]])]
    assert fake.deleted_ids == []
    assert fake.deleted_filters == []
    assert fake.ensure_calls == []
    backup_root = hermes_home / "qdrant_memory" / "backups"
    assert (backup_root / result["pre_restore_backup_id"] / "manifest.json").exists()


def test_restore_lineage_fence_refuses_actual_scoped_dependent(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    scoped = {
        "profile_id": "architect",
        "user_id_hash": "user-scope",
        "chat_id_hash": "chat-scope",
    }
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant({
        "memory": [_point("root", "backup", [0.1, 0.2], **scoped)],
    }))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection["memory"] = [
        _point("root", "changed", [0.1, 0.2], **scoped),
        _point("dependent", "dependent", [0.2, 0.3], **scoped),
        _point(
            "edge", "", {}, **scoped, memory_kind="graph_edge",
            source_point_id="dependent", target_point_id="root",
            relation_type="DERIVED_FROM",
        ),
    ]
    restore_args = parser.parse_args([
        "qdrant", "restore", "--backup", backup_id,
        "--no-dry-run", "--approve", "--backup-first", "--json",
    ])

    assert execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 2
    assert fake.upserts == []


@pytest.mark.parametrize(
    "impact",
    [
        {"complete": False, "errors": [], "dependent_ids": []},
        {"complete": True, "errors": ["lookup failed"], "dependent_ids": []},
        {"complete": True, "errors": [], "dependent_ids": ["dependent"]},
    ],
)
def test_restore_lineage_fence_refuses_incomplete_errors_and_scoped_dependents(
    monkeypatch, tmp_path, capsys, impact,
):
    from qdrant_memory.cli_core import execute_command
    import qdrant_memory.backup as backup

    _write_config(monkeypatch, tmp_path)
    scoped = {
        "profile_id": "architect",
        "user_id_hash": "user-scope",
        "chat_id_hash": "chat-scope",
    }
    backup_points = {"memory": [_point("m1", "backup", [0.1, 0.2], **scoped)]}
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant(backup_points))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection["memory"] = [_point("m1", "changed", [0.1, 0.2], **scoped)]
    calls = []

    def impact_lookup(**kwargs):
        calls.append(kwargs)
        return {"schema_version": 1, "root_ids": ["m1"], "snapshot_digests": {}, **impact}

    monkeypatch.setattr(backup, "build_lineage_impact_snapshot", impact_lookup)
    restore_args = parser.parse_args([
        "qdrant", "restore", "--backup", backup_id,
        "--no-dry-run", "--approve", "--backup-first", "--json",
    ])

    assert execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 2
    assert calls[0]["profile_id"] == "architect"
    assert calls[0]["user_id_hash"] == "user-scope"
    assert calls[0]["chat_id_hash"] == "chat-scope"
    assert fake.upserts == []


def test_backup_url_redaction_fails_closed_if_cli_redactor_breaks(monkeypatch):
    import qdrant_memory.backup as backup
    import qdrant_memory.cli_core as cli_core

    password = "".join(["fallback", "pass"])
    raw_url = "http://" + "fallback-user" + ":" + password + "@local-qdrant.invalid:6333"

    def broken_redactor(_config):
        raise RuntimeError("redactor unavailable")

    monkeypatch.setattr(cli_core, "redact_config", broken_redactor)

    redacted = backup._redacted_qdrant_url({"qdrant_url": raw_url})

    assert password not in redacted
    assert "fallback-user" not in redacted
    assert "<redacted>" in redacted
    assert "@local-qdrant.invalid" in redacted


def test_backup_list_and_inspect_reredact_stale_manifest_urls(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = _write_config(monkeypatch, tmp_path)
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant({"memory": [_point("m1", "safe manifest redaction")]}))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    backup_dir = hermes_home / "qdrant_memory" / "backups" / backup_id
    manifest_path = backup_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    password = "".join(["stale", "manifest", "pass"])
    raw_url = "http://" + "stale-user" + ":" + password + "@local-qdrant.invalid:6333?" + "api_key=" + password
    manifest["qdrant_url"] = raw_url
    manifest["redacted_qdrant_url"] = raw_url
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    list_args = parser.parse_args(["qdrant", "backup", "list", "--json"])
    assert execute_command(list_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    list_stdout = capsys.readouterr().out
    assert password not in list_stdout
    assert "stale-user" not in list_stdout
    listed = json.loads(list_stdout)["backups"][0]
    assert "<redacted>" in listed["qdrant_url"]
    assert "<redacted>" in listed["redacted_qdrant_url"]

    inspect_args = parser.parse_args(["qdrant", "backup", "inspect", backup_id, "--json"])
    assert execute_command(inspect_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    inspect_stdout = capsys.readouterr().out
    assert password not in inspect_stdout
    assert "stale-user" not in inspect_stdout
    inspected = json.loads(inspect_stdout)
    assert "<redacted>" in inspected["qdrant_url"]
    assert "<redacted>" in inspected["redacted_qdrant_url"]
    assert fake.upserts == []


def test_export_qdrant_runtime_error_is_sanitized_json_boundary(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    marker = "".join(["service", "body", "marker"])
    raw_url = "http://" + "svc-user" + ":" + marker + "@qdrant.invalid:6333"
    failure_message = "Qdrant HTTP 500: " + json.dumps({"status": {"error": marker, "url": raw_url}})
    _install_fake_qdrant(monkeypatch, FailingScrollQdrant(failure_message))
    parser = _parser()
    args = parser.parse_args(["qdrant", "export", "memory", "--out", str(tmp_path / "memory.jsonl"), "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert marker not in captured.err
    assert "svc-user" not in captured.err
    assert "Traceback" not in captured.err
    payload = json.loads(captured.err)
    assert payload["error"] is True
    assert "Qdrant" in payload["message"]


def _direct_memory_backup(tmp_path):
    from qdrant_memory.backup import create_backup

    hermes_home = tmp_path / "hermes-direct"
    hermes_home.mkdir()
    config = {
        "qdrant_url": "http://local-qdrant.invalid:6333",
        "collection_name": "memory",
        "learning_collection_name": "learnings",
        "vector_size": 2,
        "distance": "Cosine",
    }
    fake = FakeQdrant({"memory": [_point("m1", "malformed artifact", [0.1, 0.2])]})
    summary = create_backup(fake, config, hermes_home=hermes_home, scope="memory")
    return hermes_home, config, str(summary["backup_id"])


def test_malformed_manifest_schema_version_raises_backup_error(tmp_path):
    from qdrant_memory.backup import BackupError, inspect_backup

    hermes_home, config, backup_id = _direct_memory_backup(tmp_path)
    manifest_path = hermes_home / "qdrant_memory" / "backups" / backup_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = "not-an-int"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(BackupError, match="schema_version"):
        inspect_backup(backup_id, config, hermes_home=hermes_home)


def test_malformed_collection_header_schema_version_raises_backup_error(tmp_path):
    from qdrant_memory.backup import BackupError, file_sha256, inspect_backup

    hermes_home, config, backup_id = _direct_memory_backup(tmp_path)
    backup_dir = hermes_home / "qdrant_memory" / "backups" / backup_id
    manifest_path = backup_dir / "manifest.json"
    memory_path = backup_dir / "memory.jsonl"
    lines = memory_path.read_text(encoding="utf-8").splitlines()
    header = json.loads(lines[0])
    header["schema_version"] = "not-an-int"
    lines[0] = json.dumps(header, sort_keys=True)
    memory_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collections"]["memory"]["sha256"] = file_sha256(memory_path)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(BackupError, match="schema_version"):
        inspect_backup(backup_id, config, hermes_home=hermes_home)


def test_malformed_collection_count_raises_backup_error(tmp_path):
    from qdrant_memory.backup import BackupError, inspect_backup

    hermes_home, config, backup_id = _direct_memory_backup(tmp_path)
    manifest_path = hermes_home / "qdrant_memory" / "backups" / backup_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["collections"]["memory"]["count"] = "not-an-int"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")

    with pytest.raises(BackupError, match="count"):
        inspect_backup(backup_id, config, hermes_home=hermes_home)


def test_restore_live_preflights_all_target_vectors_before_any_upsert(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    _write_config(monkeypatch, tmp_path)
    backup_points = {
        "memory": [_point("m1", "preflight memory", [0.1, 0.2])],
        "learnings": [_point("l1", "preflight learning", [0.3, 0.4], source_type="learning")],
    }
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant(backup_points, vector_size=2))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection = {"memory": [], "learnings": []}
    fake.vector_size = {"memory": 2, "learnings": 3}

    restore_args = parser.parse_args(["qdrant", "restore", "--backup", backup_id, "--no-dry-run", "--approve", "--json"])
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert "vector size" in captured.err.lower()
    assert "learning" in captured.err.lower()
    assert fake.upserts == []
    assert fake.deleted_ids == []
    assert fake.deleted_filters == []


def test_restore_live_creates_pre_restore_backup_by_default(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = _write_config(monkeypatch, tmp_path)
    backup_points = {"memory": [_point("m1", "auto backup live restore", [0.1, 0.2])]}
    fake = _install_fake_qdrant(monkeypatch, FakeQdrant(backup_points, vector_size=2))
    parser = _parser()
    create_args = parser.parse_args(["qdrant", "backup", "create", "--scope", "memory", "--json"])
    assert execute_command(create_args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    original_backup_id = json.loads(capsys.readouterr().out)["backup_id"]
    fake.by_collection = {"memory": []}

    restore_args = parser.parse_args(["qdrant", "restore", "--backup", original_backup_id, "--no-dry-run", "--approve", "--json"])
    exit_code = execute_command(restore_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    stdout = capsys.readouterr().out
    assert exit_code == 0
    result = json.loads(stdout)
    assert result["applied"] is True
    assert result["pre_restore_backup_id"] != original_backup_id
    assert (hermes_home / "qdrant_memory" / "backups" / result["pre_restore_backup_id"] / "manifest.json").exists()
    assert fake.upserts == [("memory", [backup_points["memory"][0]])]

def test_restore_refuses_missing_chunk_retired_by_committed_event(tmp_path, monkeypatch):
    from qdrant_memory import backup as backup_module

    chunk = _point(
        "chunk-retired", "retired chunk", [0.1, 0.2],
        lineage_schema_version=1, file_version_id="version-retired",
    )
    version = _point(
        "version-retired", "", {}, lineage_record=True,
        lineage_role="file_version", fact_status="active",
    )
    source = FakeQdrant({"memory": [chunk, version], "learnings": []})
    config = {
        "collection_name": "memory", "learning_collection_name": "learnings",
        "vector_size": 2, "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    backup_id = backup_module.create_backup(
        source, config, hermes_home=hermes_home, scope="both",
    )["backup_id"]
    event = _point(
        "event-retired", "", {}, lineage_record=True,
        lineage_role="change_event", event_state="committed",
        retired_point_ids=["chunk-retired"],
    )
    target = FakeQdrant({"memory": [event, version], "learnings": []})
    monkeypatch.setattr(
        backup_module, "create_backup", lambda *args, **kwargs: {"backup_id": "pre-restore"},
    )

    with pytest.raises(backup_module.BackupError, match="resurrect retired"):
        backup_module.restore_backup(
            target, config, hermes_home=hermes_home, backup_id=backup_id,
            dry_run=False, backup_first=True,
        )
    assert target.upserts == []


def test_restore_refuses_missing_lineage_point_with_retired_status(tmp_path, monkeypatch):
    from qdrant_memory import backup as backup_module

    chunk = _point(
        "chunk-status-retired", "retired chunk", [0.1, 0.2],
        lineage_schema_version=1, fact_status="superseded",
    )
    source = FakeQdrant({"memory": [chunk], "learnings": []})
    config = {
        "collection_name": "memory", "learning_collection_name": "learnings",
        "vector_size": 2, "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    backup_id = backup_module.create_backup(
        source, config, hermes_home=hermes_home, scope="both",
    )["backup_id"]
    target = FakeQdrant({"memory": [], "learnings": []})
    monkeypatch.setattr(
        backup_module, "create_backup", lambda *args, **kwargs: {"backup_id": "pre-restore"},
    )

    with pytest.raises(backup_module.BackupError, match="resurrect retired"):
        backup_module.restore_backup(
            target, config, hermes_home=hermes_home, backup_id=backup_id,
            dry_run=False, backup_first=True,
        )
    assert target.upserts == []


def test_restore_refuses_missing_chunk_whose_version_is_superseded(tmp_path, monkeypatch):
    from qdrant_memory import backup as backup_module

    chunk = _point(
        "chunk-superseded", "superseded chunk", [0.1, 0.2],
        lineage_schema_version=1, file_version_id="version-superseded",
    )
    version = _point(
        "version-superseded", "", {}, lineage_record=True,
        lineage_role="file_version", fact_status="superseded",
    )
    source = FakeQdrant({"memory": [chunk, version], "learnings": []})
    config = {
        "collection_name": "memory", "learning_collection_name": "learnings",
        "vector_size": 2, "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    backup_id = backup_module.create_backup(
        source, config, hermes_home=hermes_home, scope="both",
    )["backup_id"]
    target = FakeQdrant({"memory": [version], "learnings": []})
    monkeypatch.setattr(
        backup_module, "create_backup", lambda *args, **kwargs: {"backup_id": "pre-restore"},
    )

    with pytest.raises(backup_module.BackupError, match="resurrect retired"):
        backup_module.restore_backup(
            target, config, hermes_home=hermes_home, backup_id=backup_id,
            dry_run=False, backup_first=True,
        )
    assert target.upserts == []


def test_restore_allows_lineage_chunk_that_was_merely_lost(tmp_path, monkeypatch):
    from qdrant_memory import backup as backup_module

    chunk = _point(
        "chunk-lost", "lost chunk", [0.1, 0.2],
        lineage_schema_version=1, file_version_id="version-active",
    )
    version = _point(
        "version-active", "", {}, lineage_record=True,
        lineage_role="file_version", fact_status="active",
    )
    source = FakeQdrant({"memory": [chunk, version], "learnings": []})
    config = {
        "collection_name": "memory", "learning_collection_name": "learnings",
        "vector_size": 2, "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    backup_id = backup_module.create_backup(
        source, config, hermes_home=hermes_home, scope="both",
    )["backup_id"]
    target = FakeQdrant({"memory": [version], "learnings": []})
    monkeypatch.setattr(
        backup_module, "create_backup", lambda *args, **kwargs: {"backup_id": "pre-restore"},
    )

    result = backup_module.restore_backup(
        target, config, hermes_home=hermes_home, backup_id=backup_id,
        dry_run=False, backup_first=True,
    )
    assert result["collections"]["memory"]["upserted"] == 1
    assert target.upserts == [("memory", [chunk])]


def test_restore_preflights_every_collection_fence_before_first_write(tmp_path, monkeypatch):
    from qdrant_memory import backup as backup_module

    source = FakeQdrant({
        "memory": [_point("memory-changed", "backup memory")],
        "learnings": [_point("learning-missing", "restore me")],
    })
    config = {
        "collection_name": "memory", "learning_collection_name": "learnings",
        "vector_size": 2, "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    backup_id = backup_module.create_backup(
        source, config, hermes_home=hermes_home, scope="both",
    )["backup_id"]
    target = FakeQdrant({
        "memory": [_point(
            "memory-changed", "live memory",
            lineage_schema_version=1,
        )],
        "learnings": [],
    })
    monkeypatch.setattr(
        backup_module, "create_backup", lambda *args, **kwargs: {"backup_id": "pre-restore"},
    )

    with pytest.raises(backup_module.BackupError, match="overwrite lineage-managed"):
        backup_module.restore_backup(
            target, config, hermes_home=hermes_home, backup_id=backup_id,
            dry_run=False, backup_first=True,
        )
    assert target.upserts == []


# --- restore is the third in-place overwrite route ---------------------------
#
# Its predicate used to be local (three identity fields plus the structural marker),
# so it allowed the review-only target the store and extraction routes refuse, and
# targets carrying only lineage_role / lineage_scope_key / lineage_source_key. The
# rows below are the shared overwrite predicate, so a future narrowing of one route
# cannot pass by testing one field.

_RESTORE_PROTECTED_FIELDS = ("file_version_id", "lineage_entity_id", "lineage_schema_version",
                             "lineage_scope_key", "lineage_source_key", "lineage_role",
                             "lineage_review_event_ids")


def _restore_fixture(tmp_path, monkeypatch, live_payload):
    """A backup whose memory point changed, against a target holding ``live_payload``."""
    from qdrant_memory import backup as backup_module

    source = FakeQdrant({
        "memory": [_point("memory-changed", "backup memory")],
        "learnings": [_point("learning-missing", "restore me")],
    })
    config = {
        "collection_name": "memory", "learning_collection_name": "learnings",
        "vector_size": 2, "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    backup_id = backup_module.create_backup(
        source, config, hermes_home=hermes_home, scope="both",
    )["backup_id"]
    target = FakeQdrant({"memory": [_point("memory-changed", "live memory", **live_payload)],
                         "learnings": []})
    monkeypatch.setattr(
        backup_module, "create_backup", lambda *args, **kwargs: {"backup_id": "pre-restore"},
    )
    return backup_module, target, config, hermes_home, backup_id


@pytest.mark.parametrize("field", _RESTORE_PROTECTED_FIELDS)
def test_restore_refuses_to_overwrite_every_protected_payload(field, tmp_path, monkeypatch):
    backup_module, target, config, hermes_home, backup_id = _restore_fixture(
        tmp_path, monkeypatch, {field: ["event-1"] if field == "lineage_review_event_ids" else "identity-1"},
    )

    with pytest.raises(backup_module.BackupError, match="overwrite lineage-managed"):
        backup_module.restore_backup(
            target, config, hermes_home=hermes_home, backup_id=backup_id,
            dry_run=False, backup_first=True,
        )
    assert target.upserts == []


def test_restore_refuses_the_review_only_shape(tmp_path, monkeypatch):
    """The exact shape the second review reproduced on the other two routes: no
    identity field, only the state a transition wrote. Replacing it erases
    ``requires_review``, the gate the retriever reads."""
    backup_module, target, config, hermes_home, backup_id = _restore_fixture(
        tmp_path, monkeypatch,
        {"requires_review": True, "fact_status": "review_required",
         "lineage_review_event_ids": ["event-1"]},
    )

    with pytest.raises(backup_module.BackupError, match="overwrite lineage-managed"):
        backup_module.restore_backup(
            target, config, hermes_home=hermes_home, backup_id=backup_id,
            dry_run=False, backup_first=True,
        )
    assert target.upserts == []


def test_restore_still_overwrites_an_ordinary_payload(tmp_path, monkeypatch):
    """Negative control: restore keeps restoring. The refusal is about the live
    payload's lineage state, not about the route."""
    backup_module, target, config, hermes_home, backup_id = _restore_fixture(
        tmp_path, monkeypatch, {},
    )

    result = backup_module.restore_backup(
        target, config, hermes_home=hermes_home, backup_id=backup_id,
        dry_run=False, backup_first=True,
    )

    assert result["applied"] is True
    assert result["upserted"] == 2
    # Both scopes were written; the order restore walks them is not part of the
    # contract, so the control asserts the set rather than the sequence.
    assert sorted(name for name, _ in target.upserts) == ["learnings", "memory"]


# ===========================================================================
# W0: payload-only points (vector={}) round-trip exactly
# ===========================================================================

def test_export_and_restore_roundtrip_payload_only_points_exactly(tmp_path, monkeypatch):
    """Metadata-only structural records export with ``vector={}`` and restore
    byte-exact: same id, empty vector map, identical payload."""
    qdrant = FakeQdrant(by_collection={
        "memory": [
            {
                "id": "3ef33041-ae69-62c4-f736-c033ce20c8b7",
                "vector": {},
                "payload": {
                    "memory_kind": "graph_entity",
                    "entity_id": "entity-0123456789abcdef",
                    "entity_type": "source",
                    "lineage_record": True,
                    "lineage_schema_version": 1,
                    "lineage_operation": "index_capture",
                    "lineage_identity_digest": "a" * 64,
                    "lineage_role": "file_source",
                    "label": "file-source-abc123",
                    "text": "file-source-abc123",
                    "canonical": False,
                    "requires_review": True,
                    "profile_id": "p1",
                    "user_id_hash": "",
                    "chat_id_hash": "",
                },
            },
            {
                "id": "ordinary-1",
                "vector": [0.1, 0.2],
                "payload": {"text": "ordinary memory"},
            },
        ],
        "learnings": [],
    })
    config = {
        "qdrant_url": "http://local-qdrant.invalid:6333",
        "collection_name": "memory",
        "learning_collection_name": "learnings",
        "vector_size": 2,
        "distance": "Cosine",
    }
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    out_file = tmp_path / "export" / "memory.jsonl"

    from qdrant_memory import backup as backup_module

    export_info = backup_module.export_collection(
        qdrant, config, scope="memory", out_file=str(out_file), overwrite=False
    )
    assert export_info.get("count") == 2 or export_info.get("point_count") == 2

    # The exported JSONL must keep the empty vector map verbatim.
    lines = [json.loads(line) for line in out_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    # First line is the collection header; points carry an explicit "id".
    records = [record for record in lines if isinstance(record, dict) and "id" in record]
    by_id = {record["id"]: record for record in records}
    assert by_id["3ef33041-ae69-62c4-f736-c033ce20c8b7"]["vector"] == {}
    assert by_id["ordinary-1"]["vector"] == [0.1, 0.2]

    backup = backup_module.create_backup(qdrant, config, hermes_home=str(hermes_home), scope="both")
    backup_id = backup["backup_id"]

    # Restore into a second, empty fake collection set.
    target = FakeQdrant(by_collection={"memory": [], "learnings": []})
    monkeypatch.setattr(backup_module, "create_backup", lambda *a, **k: {"backup_id": "pre-restore-skip"})
    result = backup_module.restore_backup(target, config, hermes_home=str(hermes_home), backup_id=backup_id, dry_run=False, backup_first=False)
    assert result["applied"] is True
    upserted = {point["id"]: point for _name, points in target.upserts for point in points}
    structural = upserted["3ef33041-ae69-62c4-f736-c033ce20c8b7"]
    assert structural["vector"] == {}
    assert structural["payload"]["lineage_record"] is True
    assert structural["payload"] == by_id["3ef33041-ae69-62c4-f736-c033ce20c8b7"]["payload"]
    ordinary = upserted["ordinary-1"]
    assert ordinary["vector"] == [0.1, 0.2]


def test_validate_vector_size_accepts_empty_vector_map():
    from qdrant_memory.backup import BackupError, _validate_vector_size

    record = {"id": "p1", "vector": {}, "payload": {}}
    _validate_vector_size(record, 1024, scope="memory")  # must not raise
    with pytest.raises(BackupError):
        _validate_vector_size({"id": "p2", "vector": [0.1], "payload": {}}, 1024, scope="memory")
