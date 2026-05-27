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


class StaticJsonProvider:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def handle_tool_call(self, tool_name, args):
        self.calls.append((tool_name, args))
        payload = self.payloads[tool_name]
        return payload if isinstance(payload, str) else json.dumps(payload)


def test_register_cli_adds_mvp_subcommands_and_safe_defaults():
    parser = _parser()

    status = parser.parse_args(["qdrant", "status"])
    assert status.qdrant_subcommand == "status"
    assert getattr(status, "json", False) is False

    status_json = parser.parse_args(["qdrant", "status", "--json"])
    assert status_json.qdrant_subcommand == "status"
    assert status_json.json is True

    doctor = parser.parse_args(["qdrant", "doctor", "--json"])
    assert doctor.qdrant_subcommand == "doctor"
    assert doctor.json is True

    search = parser.parse_args(["qdrant", "search", "agent memory", "--top-k", "3", "--json"])
    assert search.query == "agent memory"
    assert search.top_k == 3
    assert search.json is True

    store = parser.parse_args(["qdrant", "store", "manual memory"])
    assert store.text == "manual memory"
    assert store.dry_run is True
    assert store.approve is False
    assert store.preview_duplicates is False

    live_store = parser.parse_args(["qdrant", "store", "manual memory", "--preview-duplicates", "--no-dry-run", "--approve"])
    assert live_store.dry_run is False
    assert live_store.approve is True
    assert live_store.preview_duplicates is True

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

    history_search = parser.parse_args(["qdrant", "search", "agent memory", "--include-fact-history"])
    assert build_tool_call(history_search) == (
        "qdrant_memory_search",
        {"query": "agent memory", "top_k": 5, "source_type": None, "include_metadata": False, "include_fact_history": True},
    )

    learning = parser.parse_args(["qdrant", "learning", "search", "tool failure", "--top-k", "2"])
    assert build_tool_call(learning) == (
        "qdrant_learning_search",
        {"query": "tool failure", "top_k": 2, "learning_type": None, "include_metadata": False},
    )

    preview = parser.parse_args(["qdrant", "learning", "preview", "--include-metadata"])
    assert build_tool_call(preview) == ("qdrant_learning_preview", {"include_metadata": True})

    inspect = parser.parse_args(["qdrant", "inspect", "point-1"])
    assert build_tool_call(inspect) == ("qdrant_memory_inspect", {"point_id": "point-1", "collection": "memory"})

    trace = parser.parse_args(["qdrant", "trace", "point-1", "--direction", "both", "--collection", "learning"])
    assert build_tool_call(trace) == (
        "qdrant_memory_trace",
        {"point_id": "point-1", "direction": "both", "collection": "learning"},
    )

    expand = parser.parse_args(["qdrant", "expand", "point-1", "--mode", "source", "--max-chars", "123"])
    assert build_tool_call(expand) == (
        "qdrant_memory_expand",
        {"point_id": "point-1", "mode": "source", "max_chars": 123, "collection": "memory"},
    )

    source_status = parser.parse_args(["qdrant", "source-status", "point-1"])
    assert build_tool_call(source_status) == ("qdrant_memory_source_status", {"point_id": "point-1", "collection": "memory"})


def test_build_tool_call_maps_richer_search_filters_and_collection():
    from qdrant_memory.cli_core import build_tool_call

    parser = _parser()

    search = parser.parse_args(
        [
            "qdrant",
            "search",
            "agent memory",
            "--source-type",
            "project_doc",
            "--tag",
            "api",
            "--tag",
            "v0.7,release",
            "--source",
            "api.md",
            "--path",
            "/repo/docs/api.md",
            "--project-path",
            "/repo",
            "--since",
            "2026-01-01T00:00:00Z",
            "--until",
            "2026-01-31T23:59:59Z",
            "--collection",
            "learning",
        ]
    )
    assert build_tool_call(search) == (
        "qdrant_memory_search",
        {
            "query": "agent memory",
            "top_k": 5,
            "source_type": "project_doc",
            "include_metadata": False,
            "tags": ["api", "v0.7", "release"],
            "source": "api.md",
            "file_path": "/repo/docs/api.md",
            "project_path": "/repo",
            "since": "2026-01-01T00:00:00Z",
            "until": "2026-01-31T23:59:59Z",
            "collection": "learning",
        },
    )

    learning = parser.parse_args(
        [
            "qdrant",
            "learning",
            "search",
            "tool failure",
            "--tag",
            "workflow",
            "--source",
            "hermes_learning",
            "--file-path",
            "/repo/lessons.md",
            "--project-path",
            "/repo",
            "--since",
            "2026-02-01T00:00:00Z",
            "--until",
            "2026-02-28T23:59:59Z",
        ]
    )
    assert build_tool_call(learning) == (
        "qdrant_learning_search",
        {
            "query": "tool failure",
            "top_k": 5,
            "learning_type": None,
            "include_metadata": False,
            "tags": ["workflow"],
            "source": "hermes_learning",
            "file_path": "/repo/lessons.md",
            "project_path": "/repo",
            "since": "2026-02-01T00:00:00Z",
            "until": "2026-02-28T23:59:59Z",
        },
    )


def test_build_tool_call_preserves_dry_run_and_approval_gates():
    from qdrant_memory.cli_core import CliUsageError, build_tool_call

    parser = _parser()

    index = parser.parse_args(["qdrant", "index", "docs", "--max-files", "10"])
    assert build_tool_call(index) == (
        "qdrant_memory_index",
        {"paths": ["docs"], "dry_run": True, "force": False, "max_files": 10},
    )

    store = parser.parse_args(["qdrant", "store", "manual memory", "--preview-duplicates"])
    assert build_tool_call(store) == (
        "qdrant_memory_store",
        {
            "text": "manual memory",
            "source_type": "manual",
            "importance": 5,
            "tags": [],
            "dry_run": True,
            "duplicate_preview": True,
        },
    )

    unapproved_store = parser.parse_args(["qdrant", "store", "manual memory", "--no-dry-run"])
    with pytest.raises(CliUsageError, match="--approve is required"):
        build_tool_call(unapproved_store)

    live_store = parser.parse_args(["qdrant", "store", "manual memory", "--no-dry-run", "--approve"])
    assert build_tool_call(live_store)[1]["dry_run"] is False
    assert build_tool_call(live_store)[1]["approve"] is True

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


def test_status_defaults_to_human_output_and_json_preserves_raw_payload(capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    raw_status = json.dumps(
        {
            "provider": "qdrant",
            "active": True,
            "qdrant_ok": True,
            "embedding_ok": False,
            "collection_name": "memory",
            "collection_exists": True,
            "point_count": 2,
            "learning_collection_name": "learnings",
            "learning_collection_exists": True,
            "learning_point_count": 1,
            "qdrant_url": "http://status-user:status-pass@example.local:6333",
            "embedding_url": "https://embed-user:embed-pass@example.local/v1",
        }
    )
    provider = StaticJsonProvider({"qdrant_memory_status": raw_status})

    human_args = parser.parse_args(["qdrant", "status"])
    assert execute_command(human_args, provider_factory=lambda: provider) == 0
    human = capsys.readouterr().out
    assert "Qdrant memory provider: active" in human
    assert "[OK] qdrant" in human
    assert "[WARN] embeddings" in human
    assert "memory: 2" in human
    assert "learnings: 1" in human
    assert "status-pass" not in human
    assert "embed-pass" not in human
    with pytest.raises(json.JSONDecodeError):
        json.loads(human)

    json_args = parser.parse_args(["qdrant", "status", "--json"])
    assert execute_command(json_args, provider_factory=lambda: provider) == 0
    assert capsys.readouterr().out == raw_status + "\n"


def test_config_show_defaults_to_redacted_human_output_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    qdrant_url = "http://" + "config-user" + ":" + "config-pass" + "@example.local:6333"
    sensitive_query_key = "_".join(["api", "key"])
    query_marker = "secret" + "-query"
    embedding_url = "https://" + "embed-user" + ":" + "embed-pass" + f"@example.local/v1?{sensitive_query_key}={query_marker}"
    (hermes_home / "qdrant_memory.json").write_text(
        json.dumps(
            {
                "qdrant_url": qdrant_url,
                "embedding_url": embedding_url,
                "qdrant_api_key": "real-qdrant-key",
                "embedding_api_key": "real-embedding-key",
                "collection_name": "custom",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "config", "show"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    assert exit_code == 0
    human = capsys.readouterr().out
    assert "Qdrant memory configuration" in human
    assert "collection_name: custom" in human
    assert "qdrant_api_key: <redacted>" in human
    assert "embedding_api_key: <redacted>" in human
    assert "config-pass" not in human
    assert "embed-pass" not in human
    assert "secret-query" not in human
    with pytest.raises(json.JSONDecodeError):
        json.loads(human)


def test_execute_command_reports_usage_errors_without_provider(capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    args = parser.parse_args(["qdrant", "forget", "point-1", "--no-dry-run"])

    exit_code = execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

    captured = capsys.readouterr()
    assert exit_code == 2
    assert captured.out == ""
    assert "Error: --approve is required" in captured.err

    json_args = parser.parse_args(["qdrant", "forget", "point-1", "--no-dry-run", "--json"])
    exit_code = execute_command(json_args, provider_factory=lambda: pytest.fail("provider should not be constructed"))

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


def test_execute_command_formats_provider_json_errors_by_output_mode(capsys):
    from qdrant_memory.cli_core import execute_command

    raw_error = json.dumps({"error": "Qdrant memory provider is not initialized"})

    class ErrorProvider:
        def handle_tool_call(self, tool_name, args):
            return raw_error

    parser = _parser()
    human_args = parser.parse_args(["qdrant", "search", "agent memory"])

    exit_code = execute_command(human_args, provider_factory=ErrorProvider)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert "Error: Qdrant memory provider is not initialized" in captured.err
    with pytest.raises(json.JSONDecodeError):
        json.loads(captured.err)

    json_args = parser.parse_args(["qdrant", "search", "agent memory", "--json"])
    exit_code = execute_command(json_args, provider_factory=ErrorProvider)

    captured = capsys.readouterr()
    assert exit_code == 1
    assert captured.out == ""
    assert captured.err == raw_error + "\n"
    assert json.loads(captured.err)["error"] == "Qdrant memory provider is not initialized"


def test_execute_command_json_mode_fails_closed_on_non_object_provider_payload(capsys):
    from qdrant_memory.cli_core import execute_command

    class BadProvider:
        def __init__(self, result):
            self.result = result

        def handle_tool_call(self, tool_name, args):
            return self.result

    parser = _parser()
    cases = ["not-json", "[]"]
    for result in cases:
        args = parser.parse_args(["qdrant", "search", "agent memory", "--json"])
        exit_code = execute_command(args, provider_factory=lambda result=result: BadProvider(result))

        captured = capsys.readouterr()
        assert exit_code == 1
        assert captured.out == ""
        error = json.loads(captured.err)
        assert error["error"] is True
        assert "JSON object" in error["message"]


def test_provider_backed_commands_default_to_human_summaries_and_json_can_remain_raw(capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    cases = [
        (
            ["qdrant", "search", "alpha"],
            "qdrant_memory_search",
            '{"results":[{"id":"m1","text":"alpha memory","score":0.988}],"count":1}',
            ["Found 1 memory", "1. m1 score=0.988 alpha memory"],
        ),
        (
            ["qdrant", "store", "alpha memory"],
            "qdrant_memory_store",
            '{"saved":true,"id":"m1","source_type":"manual"}',
            ["Saved memory: m1", "source_type: manual"],
        ),
        (
            ["qdrant", "store", "alpha memory"],
            "qdrant_memory_store",
            '{"dry_run":true,"saved":false,"would_store":true,"id":"m-dry","source_type":"manual","duplicate_preview":true,"duplicate_found":false}',
            ["Memory store dry-run: m-dry", "would_store: true", "duplicate_preview: true", "duplicate_found: false"],
        ),
        (
            ["qdrant", "store", "alpha memory"],
            "qdrant_memory_store",
            '{"dry_run":false,"saved":false,"would_store":false,"id":"m-new","source_type":"manual","duplicate_preview":true,"duplicate_found":true,"duplicate":{"id":"m-existing","score":0.93}}',
            ["Memory not saved: duplicate found", "duplicate_id: m-existing", "duplicate_score: 0.930"],
        ),
        (
            ["qdrant", "learning", "preview"],
            "qdrant_learning_preview",
            '{"candidates":[{"candidate_id":"c1","learning_type":"workflow","lesson":"Prefer exact IDs"}],"count":1,"dry_run":true}',
            ["Pending learning candidates: 1", "1. c1 [workflow] Prefer exact IDs"],
        ),
        (
            ["qdrant", "watcher", "run", "--scope", "memory"],
            "qdrant_memory_consolidate",
            '{"dry_run":true,"report_only":true,"report_id":"r1","scope":"memory","proposals":[{"proposal_id":"p1"},{"proposal_id":"p2"}],"summary":{"duplicate_cluster":2}}',
            ["Consolidation watcher (report-only): 2 proposals, guarded-auto applied=0, errors=0", "report_id: r1", "scope: memory"],
        ),
        (
            ["qdrant", "apply", "--report-id", "r1", "--proposal-id", "p1", "--action", "merge"],
            "qdrant_memory_consolidation_apply",
            '{"dry_run":true,"would_apply":true,"report_id":"r1","proposal_id":"p1","action":"merge","affected_ids":["m1","m2"]}',
            ["Apply dry-run: action=merge proposal=p1 affected=2", "report_id: r1"],
        ),
        (
            ["qdrant", "inspect", "m1"],
            "qdrant_memory_inspect",
            '{"found":true,"point_id":"m1","collection":"memory","collection_name":"mem","payload":{"text":"alpha","source_uri":"file:///tmp/a.md"},"source":{"source_uri":"file:///tmp/a.md"}}',
            ["Point: m1", "collection: memory (mem)", "source_uri: file:///tmp/a.md"],
        ),
        (
            ["qdrant", "trace", "m1"],
            "qdrant_memory_trace",
            '{"point_id":"m1","direction":"upstream","upstream":[{"source_uri":"memory://point/root","status":"exists"}]}',
            ["Trace: m1", "upstream: 1", "memory://point/root"],
        ),
        (
            ["qdrant", "expand", "m1", "--max-chars", "20"],
            "qdrant_memory_expand",
            '{"point_id":"m1","mode":"excerpt","status":"exists","source_uri":"file:///tmp/a.md","text":"expanded source","truncated":false}',
            ["Expansion: m1", "status: exists", "expanded source"],
        ),
        (
            ["qdrant", "source-status", "m1"],
            "qdrant_memory_source_status",
            '{"point_id":"m1","status":"changed","source_uri":"file:///tmp/a.md","changed":true}',
            ["Source status: m1", "status: changed", "changed: True"],
        ),
    ]

    for argv, tool_name, raw_payload, expected_lines in cases:
        provider = StaticJsonProvider({tool_name: raw_payload})
        args = parser.parse_args(argv)
        assert execute_command(args, provider_factory=lambda provider=provider: provider) == 0
        human = capsys.readouterr().out
        for expected in expected_lines:
            assert expected in human
        with pytest.raises(json.JSONDecodeError):
            json.loads(human)

    raw_store = '{"saved":true,"id":"m-json","source_type":"manual"}'
    json_provider = StaticJsonProvider({"qdrant_memory_store": raw_store})
    json_args = parser.parse_args(["qdrant", "store", "alpha memory", "--json"])
    assert execute_command(json_args, provider_factory=lambda: json_provider) == 0
    assert capsys.readouterr().out == raw_store + "\n"


def test_search_human_summary_uses_memories_plural(capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    raw_payload = '{"results":[{"id":"m1","text":"alpha","score":0.1},{"id":"m2","text":"beta","score":0.2}],"count":2}'
    provider = StaticJsonProvider({"qdrant_memory_search": raw_payload})
    args = parser.parse_args(["qdrant", "search", "alpha"])

    assert execute_command(args, provider_factory=lambda: provider) == 0
    human = capsys.readouterr().out
    assert "Found 2 memories" in human
    assert "memorys" not in human

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


def test_doctor_defaults_to_human_checklist_and_nonzero_failures(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    (hermes_home / "qdrant_memory.json").write_text(
        json.dumps(
            {
                "qdrant_url": "http://doctor-user:doctor-pass@example.local:6333",
                "embedding_url": "https://embed-user:embed-pass@example.local/v1",
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
            "active": False,
            "qdrant_ok": False,
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
    args = parser.parse_args(["qdrant", "doctor"])

    exit_code = execute_command(args, provider_factory=lambda: provider)

    assert exit_code == 1
    human = capsys.readouterr().out
    assert "Diagnostics:" in human
    assert "[OK] plugin_discovery" in human
    assert "[FAIL] active_provider" in human
    assert "[FAIL] qdrant_reachable" in human
    assert "doctor-pass" not in human
    assert "embed-pass" not in human
    with pytest.raises(json.JSONDecodeError):
        json.loads(human)


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
        {
            "text": "remember this",
            "source_type": "manual",
            "importance": 8,
            "tags": ["alpha", "beta", "gamma"],
            "dry_run": True,
            "duplicate_preview": False,
        },
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
