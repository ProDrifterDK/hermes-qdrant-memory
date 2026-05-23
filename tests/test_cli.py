import argparse
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_plugin_cli_module():
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_test", ROOT / "cli.py")
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


class FakeProvider:
    def __init__(self):
        self.calls = []

    def handle_tool_call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return json.dumps({"tool": tool_name, "args": args})


class FakeStatusProvider:
    def __init__(self, status):
        self.status = status
        self.calls = []

    def handle_tool_call(self, tool_name, args):
        self.calls.append((tool_name, args))
        return json.dumps(self.status)


def test_register_cli_adds_mvp_subcommands_and_safe_defaults():
    parser = _parser()

    status = parser.parse_args(["qdrant", "status"])
    assert status.qdrant_subcommand == "status"

    doctor = parser.parse_args(["qdrant", "doctor", "--json"])
    assert doctor.qdrant_subcommand == "doctor"
    assert doctor.json is True

    search = parser.parse_args(["qdrant", "search", "agent memory", "--top-k", "3", "--json"])
    assert search.query == "agent memory"
    assert search.top_k == 3
    assert search.json is True

    index = parser.parse_args(["qdrant", "index", "docs", "README.md"])
    assert index.paths == ["docs", "README.md"]
    assert index.dry_run is True
    assert index.approve is False

    live_index = parser.parse_args(["qdrant", "index", "docs", "--no-dry-run", "--approve"])
    assert live_index.dry_run is False
    assert live_index.approve is True

    consolidate = parser.parse_args(["qdrant", "consolidate", "--scope", "both", "--persist"])
    assert consolidate.scope == "both"
    assert consolidate.dry_run is True
    assert consolidate.persist is True

    with pytest.raises(SystemExit):
        parser.parse_args(["qdrant", "search", "agent memory", "--top-k", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["qdrant", "learning", "search", "agent memory", "--top-k", "21"])


def test_build_tool_call_maps_read_only_commands():
    from qdrant_memory.cli_core import build_tool_call

    parser = _parser()

    status = parser.parse_args(["qdrant", "status"])
    assert build_tool_call(status) == ("qdrant_memory_status", {})

    search = parser.parse_args(["qdrant", "search", "agent memory", "--source-type", "manual", "--include-metadata"])
    assert build_tool_call(search) == (
        "qdrant_memory_search",
        {"query": "agent memory", "top_k": 5, "source_type": "manual", "include_metadata": True},
    )

    learning = parser.parse_args(["qdrant", "learning", "search", "tool failure", "--top-k", "2"])
    assert build_tool_call(learning) == (
        "qdrant_learning_search",
        {"query": "tool failure", "top_k": 2, "learning_type": None, "include_metadata": False},
    )

    preview = parser.parse_args(["qdrant", "learning", "preview", "--include-metadata"])
    assert build_tool_call(preview) == ("qdrant_learning_preview", {"include_metadata": True})


def test_build_tool_call_preserves_dry_run_and_approval_gates():
    from qdrant_memory.cli_core import CliUsageError, build_tool_call

    parser = _parser()

    index = parser.parse_args(["qdrant", "index", "docs", "--max-files", "10"])
    assert build_tool_call(index) == (
        "qdrant_memory_index",
        {"paths": ["docs"], "dry_run": True, "force": False, "max_files": 10},
    )

    live_index = parser.parse_args(["qdrant", "index", "docs", "--no-dry-run", "--approve"])
    assert build_tool_call(live_index)[1]["dry_run"] is False

    unapproved_forget = parser.parse_args(["qdrant", "forget", "point-1", "--no-dry-run"])
    with pytest.raises(CliUsageError, match="--approve is required"):
        build_tool_call(unapproved_forget)

    forget = parser.parse_args(["qdrant", "forget", "point-1", "--no-dry-run", "--approve"])
    assert build_tool_call(forget) == ("qdrant_memory_forget", {"ids": ["point-1"], "dry_run": False})

    consolidate = parser.parse_args(["qdrant", "consolidate", "--scope", "learning", "--persist"])
    assert build_tool_call(consolidate) == (
        "qdrant_memory_consolidate",
        {
            "dry_run": True,
            "scope": "learning",
            "persist": True,
            "include_reconsolidation": False,
        },
    )

    live_apply = parser.parse_args(
        [
            "qdrant",
            "apply",
            "--report-id",
            "report-1",
            "--proposal-id",
            "proposal-1",
            "--action",
            "merge",
            "--no-dry-run",
        ]
    )
    with pytest.raises(CliUsageError, match="--approve is required"):
        build_tool_call(live_apply)


def test_execute_command_invokes_provider_and_prints_json(capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    provider = FakeProvider()
    args = parser.parse_args(["qdrant", "search", "agent memory", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: provider)

    assert exit_code == 0
    assert provider.calls == [
        (
            "qdrant_memory_search",
            {"query": "agent memory", "top_k": 5, "source_type": None, "include_metadata": False},
        )
    ]
    output = json.loads(capsys.readouterr().out)
    assert output["tool"] == "qdrant_memory_search"


def test_execute_command_reports_usage_errors_without_provider(capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    args = parser.parse_args(["qdrant", "forget", "point-1", "--no-dry-run"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 2
    error = json.loads(capsys.readouterr().err)
    assert error["error"] is True
    assert "--approve is required" in error["message"]


def test_cli_module_loads_when_plugin_dir_is_not_on_sys_path(tmp_path):
    plugin_dir = tmp_path / "qdrant"
    plugin_dir.mkdir()
    shutil.copy(ROOT / "cli.py", plugin_dir / "cli.py")
    shutil.copytree(ROOT / "qdrant_memory", plugin_dir / "qdrant_memory")

    probe = f"""
import importlib.util
import sys
from pathlib import Path
plugin_dir = Path({str(plugin_dir)!r})
sys.path = [p for p in sys.path if Path(p or '.').resolve() != plugin_dir.resolve()]
spec = importlib.util.spec_from_file_location('_hermes_memory_cli_qdrant_probe', plugin_dir / 'cli.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert callable(module.register_cli)
assert str(plugin_dir) in sys.path
"""
    subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, check=True)


def test_default_provider_factory_initializes_provider(monkeypatch):
    from qdrant_memory import cli_core

    class Provider:
        def __init__(self):
            self.initialized_with = None

        def initialize(self, session_id, **kwargs):
            self.initialized_with = (session_id, kwargs)

    monkeypatch.setattr(cli_core, "_load_provider_class", lambda: Provider)

    provider = cli_core.default_provider_factory()

    assert provider.initialized_with == ("cli", {"platform": "cli", "agent_context": "cli"})


def test_execute_command_returns_nonzero_when_provider_returns_json_error(capsys):
    from qdrant_memory.cli_core import execute_command

    class ErrorProvider:
        def handle_tool_call(self, tool_name, args):
            return json.dumps({"error": "Qdrant memory provider is not initialized"})

    parser = _parser()
    args = parser.parse_args(["qdrant", "search", "agent memory"])

    exit_code = execute_command(args, provider_factory=ErrorProvider)

    assert exit_code == 1
    assert json.loads(capsys.readouterr().out)["error"] == "Qdrant memory provider is not initialized"


def test_qdrant_command_exits_with_execute_command_code(monkeypatch):
    cli = _load_plugin_cli_module()
    parser = _parser()
    args = parser.parse_args(["qdrant", "forget", "point-1", "--no-dry-run"])

    monkeypatch.setattr(cli, "execute_command", lambda received: 2)

    with pytest.raises(SystemExit) as exc:
        cli.qdrant_command(args)

    assert exc.value.code == 2


def test_config_show_prints_redacted_config_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "qdrant_memory.json").write_text(
        json.dumps({"qdrant_api_key": "real-qdrant-key", "embedding_api_key": "real-embedding-key", "collection_name": "custom"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "config", "show", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["collection_name"] == "custom"
    assert payload["qdrant_api_key"] == "<redacted>"
    assert payload["embedding_api_key"] == "<redacted>"


def test_config_show_redacts_credentialed_urls_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    raw_qdrant_url = "http://" + "user-name" + ":" + "pass-word" + "@example.local:6333"
    raw_embedding_url = "https://" + "embed-user" + ":" + "embed-pass" + "@embeddings.local/v1"
    (hermes_home / "qdrant_memory.json").write_text(
        json.dumps(
            {
                "qdrant_url": raw_qdrant_url,
                "embedding_url": raw_embedding_url,
                "qdrant_api_key": "MARKER_QDRANT_VALUE",
                "embedding_api_key": "MARKER_EMBEDDING_VALUE",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "config", "show", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert raw_qdrant_url not in output
    assert raw_embedding_url not in output
    payload = json.loads(output)
    assert payload["qdrant_url"] == "http://<redacted>@example.local:6333"
    assert payload["embedding_url"] == "https://<redacted>@embeddings.local/v1"
    assert payload["qdrant_api_key"] == "<redacted>"
    assert payload["embedding_api_key"] == "<redacted>"


def test_config_show_redacts_url_query_and_fragment_credentials_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    marker_query = "MARKER_QUERY_VALUE"
    marker_fragment = "MARKER_FRAGMENT_VALUE"
    marker_password = "MARKER_PASSWORD_VALUE"
    sensitive_query_key = "_".join(["api", "key"])
    sensitive_fragment_key = "_".join(["access", "token"])
    sensitive_token_key = "".join(["to", "ken"])
    raw_qdrant_url = f"https://user:{marker_password}@example.local:6333/path?{sensitive_query_key}={marker_query}&safe=kept#{sensitive_fragment_key}={marker_fragment}"
    raw_embedding_url = f"https://embeddings.local/v1?{sensitive_token_key}={marker_query}"
    (hermes_home / "qdrant_memory.json").write_text(
        json.dumps({"qdrant_url": raw_qdrant_url, "embedding_url": raw_embedding_url}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "config", "show", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    output = capsys.readouterr().out
    assert marker_query not in output
    assert marker_fragment not in output
    assert marker_password not in output
    payload = json.loads(output)
    redacted_value = "%3Credacted%3E"
    expected_qdrant_url = f"https://<redacted>@example.local:6333/path?{sensitive_query_key}={redacted_value}&safe=kept#{sensitive_fragment_key}={redacted_value}"
    expected_embedding_url = f"https://embeddings.local/v1?{sensitive_token_key}={redacted_value}"
    assert payload["qdrant_url"] == expected_qdrant_url
    assert payload["embedding_url"] == expected_embedding_url


def test_doctor_returns_structured_checks_and_success_exit(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    diag_qdrant_url = "http://" + "diag-user" + ":" + "diag-pass" + "@example.local:6333"
    diag_embedding_url = "https://" + "embed-user" + ":" + "embed-pass" + "@embeddings.local/v1"
    (hermes_home / "qdrant_memory.json").write_text(
        json.dumps(
            {
                "qdrant_url": diag_qdrant_url,
                "embedding_url": diag_embedding_url,
                "qdrant_api_key": "MARKER_QDRANT_VALUE",
                "embedding_api_key": "MARKER_EMBEDDING_VALUE",
                "collection_name": "memory",
                "learning_collection_name": "learnings",
                "vector_size": 1024,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    provider = FakeStatusProvider(
        {
            "provider": "qdrant",
            "active": True,
            "qdrant_ok": True,
            "embedding_ok": True,
            "collection_name": "memory",
            "collection_exists": True,
            "collection_vector_size": 1024,
            "learning_collection_name": "learnings",
            "learning_collection_exists": True,
            "learning_collection_vector_size": 1024,
            "vector_size": 1024,
        }
    )
    parser = _parser()
    args = parser.parse_args(["qdrant", "doctor", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: provider)

    assert exit_code == 0
    assert provider.calls == [("qdrant_memory_status", {})]
    output = capsys.readouterr().out
    assert "diag-pass" not in output
    assert "embed-pass" not in output
    assert "MARKER_QDRANT_VALUE" not in output
    payload = json.loads(output)
    assert payload["ok"] is True
    assert payload["summary"]["failed_critical"] == 0
    checks = {check["name"]: check for check in payload["checks"]}
    for name in {
        "active_provider",
        "plugin_discovery",
        "metadata_version",
        "qdrant_reachable",
        "embedding_reachable",
        "collection_vector_size",
        "memory_collection",
        "learning_collection",
        "watcher_artifacts",
        "config_redaction",
    }:
        assert checks[name]["ok"] is True


def test_doctor_returns_nonzero_for_critical_failures(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    provider = FakeStatusProvider(
        {
            "provider": "qdrant",
            "active": False,
            "qdrant_ok": False,
            "embedding_ok": True,
            "collection_name": "memory",
            "collection_exists": True,
            "collection_vector_size": 768,
            "learning_collection_name": "learnings",
            "learning_collection_exists": False,
            "learning_collection_vector_size": None,
            "vector_size": 1024,
        }
    )
    parser = _parser()
    args = parser.parse_args(["qdrant", "doctor", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: provider)

    assert exit_code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    failed = {check["name"] for check in payload["checks"] if check["critical"] and not check["ok"]}
    assert {"active_provider", "qdrant_reachable", "collection_vector_size", "learning_collection"}.issubset(failed)


def test_build_tool_call_maps_store_with_comma_friendly_tags():
    from qdrant_memory.cli_core import CliUsageError, build_tool_call

    parser = _parser()
    args = parser.parse_args(["qdrant", "store", "remember this", "--source-type", "manual", "--importance", "8", "--tag", "alpha,beta", "--tag", "gamma"])
    assert build_tool_call(args) == (
        "qdrant_memory_store",
        {"text": "remember this", "source_type": "manual", "importance": 8, "tags": ["alpha", "beta", "gamma"]},
    )

    with pytest.raises(CliUsageError, match="text is required"):
        build_tool_call(parser.parse_args(["qdrant", "store", "   "]))
    with pytest.raises(SystemExit):
        parser.parse_args(["qdrant", "store", "x", "--importance", "11"])


def test_build_tool_call_maps_learning_store_and_approve_gate():
    from qdrant_memory.cli_core import CliUsageError, build_tool_call

    parser = _parser()
    store = parser.parse_args([
        "qdrant",
        "learning",
        "store",
        "Prefer ctx_execute_file for large files",
        "--learning-type",
        "workflow_lesson",
        "--trigger",
        "large file",
        "--mistake",
        "cat huge file",
        "--correction",
        "use ctx_execute_file",
        "--evidence",
        "tool guidance",
        "--tool-name",
        "ctx_execute_file",
        "--command",
        "python",
        "--importance",
        "9",
        "--confidence",
        "0.9",
        "--tag",
        "cli,learning",
        "--promote-to-skill-candidate",
    ])
    assert build_tool_call(store) == (
        "qdrant_learning_store",
        {
            "lesson": "Prefer ctx_execute_file for large files",
            "learning_type": "workflow_lesson",
            "trigger": "large file",
            "mistake": "cat huge file",
            "correction": "use ctx_execute_file",
            "evidence": "tool guidance",
            "tool_name": "ctx_execute_file",
            "command": "python",
            "importance": 9,
            "confidence": 0.9,
            "tags": ["cli", "learning"],
            "promote_to_skill_candidate": True,
        },
    )

    dry_run = parser.parse_args(["qdrant", "learning", "approve", "candidate-1"])
    assert build_tool_call(dry_run) == ("qdrant_learning_approve", {"candidate_id": "candidate-1", "dry_run": True})

    unapproved = parser.parse_args(["qdrant", "learning", "approve", "candidate-1", "--no-dry-run"])
    with pytest.raises(CliUsageError, match="--approve is required"):
        build_tool_call(unapproved)

    live = parser.parse_args(["qdrant", "learning", "approve", "candidate-1", "--no-dry-run", "--approve"])
    assert build_tool_call(live) == ("qdrant_learning_approve", {"candidate_id": "candidate-1", "dry_run": False})


def test_watcher_status_missing_state_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "status", "--json"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["configured"] is True
    assert payload["state_exists"] is False
    assert payload["state"] == {}
    assert payload["state_path"].endswith("qdrant_memory/consolidation/watcher_state.json")


def test_watcher_run_maps_to_report_only_consolidation():
    from qdrant_memory.cli_core import build_tool_call

    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "run", "--scope", "memory", "--max-points", "123", "--max-groups", "7", "--reconsolidation-max-candidates", "4", "--no-include-reconsolidation"])

    assert build_tool_call(args) == (
        "qdrant_memory_consolidate",
        {
            "dry_run": True,
            "scope": "memory",
            "persist": True,
            "include_reconsolidation": False,
            "include_examples": False,
            "max_points": 123,
            "max_groups": 7,
            "reconsolidation_max_candidates": 4,
        },
    )
