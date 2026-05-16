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


def test_register_cli_adds_mvp_subcommands_and_safe_defaults():
    parser = _parser()

    status = parser.parse_args(["qdrant", "status"])
    assert status.qdrant_subcommand == "status"

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
