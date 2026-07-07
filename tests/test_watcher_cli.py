import argparse
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_plugin_cli_module():
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_watcher_test", ROOT / "cli.py")
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


class StaticJsonProvider:
    def __init__(self, payloads):
        self.payloads = payloads
        self.calls = []

    def handle_tool_call(self, tool_name, args):
        self.calls.append((tool_name, args))
        payload = self.payloads[tool_name]
        return payload if isinstance(payload, str) else json.dumps(payload)


class ApplyingProvider:
    def __init__(self, report):
        self.report = report
        self.calls = []

    def handle_tool_call(self, tool_name, args):
        self.calls.append((tool_name, args))
        if tool_name == "qdrant_memory_consolidate":
            return json.dumps(self.report)
        if tool_name == "qdrant_memory_consolidation_apply":
            return json.dumps({"dry_run": False, "applied": True, "proposal_id": args["proposal_id"], "action": args["action"], "affected_ids": ["x"]})
        raise AssertionError(f"unexpected tool: {tool_name}")


@pytest.fixture
def fake_crontab(monkeypatch):
    from qdrant_memory import cli_core

    state = {"text": "MAILTO=\"\"\n15 2 * * * /usr/bin/true\n"}

    def read_crontab():
        return state["text"]

    def write_crontab(text):
        state["text"] = text

    monkeypatch.setattr(cli_core, "_read_user_crontab", read_crontab)
    monkeypatch.setattr(cli_core, "_write_user_crontab", write_crontab)
    return state


def test_register_cli_adds_watcher_lifecycle_subcommands():
    parser = _parser()

    status = parser.parse_args(["qdrant", "watcher", "status", "--verbose", "--json"])
    assert status.qdrant_subcommand == "watcher"
    assert status.watcher_subcommand == "status"
    assert status.verbose is True
    assert status.json is True

    run = parser.parse_args(["qdrant", "watcher", "run", "--force-alert", "--scope", "learning", "--autonomy-mode", "guarded-auto"])
    assert run.watcher_subcommand == "run"
    assert run.force_alert is True
    assert run.scope == "learning"
    assert run.autonomy_mode == "guarded-auto"

    install = parser.parse_args(["qdrant", "watcher", "install", "--schedule", "0 3 * * *", "--json"])
    assert install.watcher_subcommand == "install"
    assert install.schedule == "0 3 * * *"
    assert install.approve is False
    assert install.json is True

    uninstall = parser.parse_args(["qdrant", "watcher", "uninstall", "--approve", "--json"])
    assert uninstall.watcher_subcommand == "uninstall"
    assert uninstall.approve is True

    logs = parser.parse_args(["qdrant", "watcher", "logs", "--tail", "5", "--json"])
    assert logs.watcher_subcommand == "logs"
    assert logs.tail == 5

    inspect_state = parser.parse_args(["qdrant", "watcher", "inspect-state", "--json"])
    assert inspect_state.watcher_subcommand == "inspect-state"

    reset = parser.parse_args(["qdrant", "watcher", "reset-signature", "--approve", "--json"])
    assert reset.watcher_subcommand == "reset-signature"
    assert reset.approve is True


def test_watcher_run_force_alert_still_maps_to_report_only_consolidation():
    from qdrant_memory.cli_core import build_tool_call

    parser = _parser()
    args = parser.parse_args(
        [
            "qdrant",
            "watcher",
            "run",
            "--force-alert",
            "--scope",
            "memory",
            "--max-points",
            "123",
            "--max-groups",
            "7",
            "--reconsolidation-max-candidates",
            "4",
            "--no-include-reconsolidation",
        ]
    )

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


def test_watcher_local_lifecycle_commands_do_not_construct_provider(monkeypatch, tmp_path, fake_crontab):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    parser = _parser()
    commands = [
        ["qdrant", "watcher", "status"],
        ["qdrant", "watcher", "status", "--verbose"],
        ["qdrant", "watcher", "logs"],
        ["qdrant", "watcher", "inspect-state"],
        ["qdrant", "watcher", "install"],
        ["qdrant", "watcher", "uninstall", "--approve"],
        ["qdrant", "watcher", "reset-signature", "--approve"],
    ]

    for argv in commands:
        args = parser.parse_args(argv)
        exit_code = execute_command(args, provider_factory=lambda: pytest.fail(f"provider constructed for {argv}"))
        assert exit_code == 0


def test_watcher_status_verbose_includes_paths_state_schedule_and_log_tail(monkeypatch, tmp_path, fake_crontab, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    state_dir = hermes_home / "qdrant_memory" / "consolidation"
    state_dir.mkdir(parents=True)
    (state_dir / "watcher_state.json").write_text(
        json.dumps({"installed": True, "last_report_id": "r1", "last_proposal_signature": "sig1"}),
        encoding="utf-8",
    )
    (state_dir / "watcher.log").write_text(
        json.dumps({"event": "old"}) + "\n" + json.dumps({"event": "new", "report_id": "r1"}) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "status", "--verbose", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["state_exists"] is True
    assert payload["state"]["last_report_id"] == "r1"
    assert payload["state"]["last_proposal_signature"] == "sig1"
    assert payload["state_path"].endswith("qdrant_memory/consolidation/watcher_state.json")
    assert payload["log_path"].endswith("qdrant_memory/consolidation/watcher.log")
    assert payload["schedule"]["installed"] is False
    assert payload["recent_log_events"][-1]["report_id"] == "r1"


def test_watcher_logs_reads_tail_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    state_dir = hermes_home / "qdrant_memory" / "consolidation"
    state_dir.mkdir(parents=True)
    (state_dir / "watcher.log").write_text(
        "\n".join(json.dumps({"index": index}) for index in range(5)) + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "logs", "--tail", "2", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["log_exists"] is True
    assert payload["count"] == 2
    assert [event["index"] for event in payload["events"]] == [3, 4]


def test_watcher_inspect_state_redacts_sensitive_keys_without_provider(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    state_dir = hermes_home / "qdrant_memory" / "consolidation"
    state_dir.mkdir(parents=True)
    sensitive_key = "api" + "_key"
    sensitive_value = "MARKER" + "_VALUE_TO_REDACT"
    (state_dir / "watcher_state.json").write_text(
        json.dumps({"last_report_id": "r1", sensitive_key: sensitive_value}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "inspect-state", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    output = capsys.readouterr().out
    assert sensitive_value not in output
    payload = json.loads(output)
    assert payload["state"]["last_report_id"] == "r1"
    assert payload["state"][sensitive_key] == "<redacted>"


def test_watcher_reset_signature_requires_approve_and_preserves_other_state(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    state_dir = hermes_home / "qdrant_memory" / "consolidation"
    state_dir.mkdir(parents=True)
    state_path = state_dir / "watcher_state.json"
    state_path.write_text(
        json.dumps(
            {
                "installed": True,
                "last_report_id": "r1",
                "last_proposal_signature": "sig1",
                "last_alert_signature": "sig1",
                "last_alert_at": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()

    without_approval = parser.parse_args(["qdrant", "watcher", "reset-signature", "--json"])
    assert execute_command(without_approval, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 2
    assert json.loads(state_path.read_text(encoding="utf-8"))["last_proposal_signature"] == "sig1"
    assert "--approve" in capsys.readouterr().err

    approved = parser.parse_args(["qdrant", "watcher", "reset-signature", "--approve", "--json"])
    assert execute_command(approved, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    payload = json.loads(capsys.readouterr().out)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert payload["reset"] is True
    assert state["installed"] is True
    assert state["last_report_id"] == "r1"
    assert "last_proposal_signature" not in state
    assert "last_alert_signature" not in state
    assert "last_alert_at" not in state


def test_watcher_install_writes_managed_cron_block_and_state_without_provider(monkeypatch, tmp_path, fake_crontab, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "install", "--schedule", "0 3 * * *", "--max-points", "42", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installed"] is True
    assert payload["changed"] is True
    assert "15 2 * * * /usr/bin/true" in fake_crontab["text"]
    assert "BEGIN HERMES_QDRANT_WATCHER" in fake_crontab["text"]
    assert "hermes qdrant watcher run" in fake_crontab["text"]
    assert "--max-points 42" in fake_crontab["text"]
    state = json.loads((hermes_home / "qdrant_memory" / "consolidation" / "watcher_state.json").read_text(encoding="utf-8"))
    assert state["installed"] is True
    assert state["schedule"] == "0 3 * * *"


def test_watcher_install_refuses_to_replace_different_managed_block_without_approve(monkeypatch, tmp_path, fake_crontab, capsys):
    from qdrant_memory.cli_core import execute_command

    fake_crontab["text"] += "# BEGIN HERMES_QDRANT_WATCHER\n0 1 * * * old-command\n# END HERMES_QDRANT_WATCHER\n"
    before = fake_crontab["text"]
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "install", "--schedule", "0 3 * * *", "--json"])

    assert execute_command(args, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 2
    assert fake_crontab["text"] == before
    assert "--approve" in capsys.readouterr().err


def test_watcher_uninstall_requires_approve_and_removes_only_managed_block(monkeypatch, tmp_path, fake_crontab, capsys):
    from qdrant_memory.cli_core import execute_command

    fake_crontab["text"] += "# BEGIN HERMES_QDRANT_WATCHER\n0 3 * * * hermes qdrant watcher run --json\n# END HERMES_QDRANT_WATCHER\n"
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    parser = _parser()

    unapproved = parser.parse_args(["qdrant", "watcher", "uninstall", "--json"])
    assert execute_command(unapproved, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 2
    assert "BEGIN HERMES_QDRANT_WATCHER" in fake_crontab["text"]
    assert "--approve" in capsys.readouterr().err

    approved = parser.parse_args(["qdrant", "watcher", "uninstall", "--approve", "--json"])
    assert execute_command(approved, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["installed"] is False
    assert payload["changed"] is True
    assert "BEGIN HERMES_QDRANT_WATCHER" not in fake_crontab["text"]
    assert "15 2 * * * /usr/bin/true" in fake_crontab["text"]


def test_watcher_uninstall_preserves_surrounding_crontab_lines(monkeypatch, tmp_path, fake_crontab):
    from qdrant_memory.cli_core import execute_command

    fake_crontab["text"] = (
        "MAILTO=\"\"\n"
        "# BEGIN HERMES_QDRANT_WATCHER\n"
        "0 3 * * * hermes qdrant watcher run --json\n"
        "# END HERMES_QDRANT_WATCHER\n"
        "15 2 * * * /usr/bin/true\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    parser = _parser()
    approved = parser.parse_args(["qdrant", "watcher", "uninstall", "--approve", "--json"])

    assert execute_command(approved, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    assert fake_crontab["text"] == "MAILTO=\"\"\n15 2 * * * /usr/bin/true\n"


def test_watcher_install_approve_replaces_middle_block_without_merging_neighbors(monkeypatch, tmp_path, fake_crontab):
    from qdrant_memory.cli_core import execute_command

    fake_crontab["text"] = (
        "MAILTO=\"\"\n"
        "# BEGIN HERMES_QDRANT_WATCHER\n"
        "0 1 * * * old-command\n"
        "# END HERMES_QDRANT_WATCHER\n"
        "15 2 * * * /usr/bin/true\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    parser = _parser()
    approved = parser.parse_args(["qdrant", "watcher", "install", "--schedule", "0 3 * * *", "--approve", "--json"])

    assert execute_command(approved, provider_factory=lambda: pytest.fail("provider should not be constructed")) == 0
    assert "MAILTO=\"\"\n15 2 * * * /usr/bin/true\n" in fake_crontab["text"]
    assert "MAILTO=\"\"15" not in fake_crontab["text"]
    assert "BEGIN HERMES_QDRANT_WATCHER" in fake_crontab["text"]
    assert "old-command" not in fake_crontab["text"]


def test_watcher_run_updates_signature_state_and_appends_log(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    report = {
        "dry_run": True,
        "report_only": True,
        "report_id": "r1",
        "scope": "both",
        "proposals": [
            {"proposal_id": "p2", "proposal_type": "stale", "suggested_action": "delete", "affected_ids": ["b"]},
            {"proposal_id": "p1", "proposal_type": "duplicate", "suggested_action": "merge", "affected_ids": ["a", "c"]},
        ],
    }
    provider = StaticJsonProvider({"qdrant_memory_consolidate": report})
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "run", "--force-alert", "--json"])

    assert execute_command(args, provider_factory=lambda: provider) == 0
    assert provider.calls[0][0] == "qdrant_memory_consolidate"
    assert provider.calls[0][1]["dry_run"] is True
    assert provider.calls[0][1]["persist"] is True
    payload = json.loads(capsys.readouterr().out)
    assert payload["report_id"] == "r1"
    state_path = hermes_home / "qdrant_memory" / "consolidation" / "watcher_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_report_id"] == "r1"
    assert state["last_proposal_count"] == 2
    assert state["last_signature_changed"] is True
    assert state["last_alerted"] is True
    assert state["last_force_alert"] is True
    assert state["last_proposal_signature"]
    log_path = hermes_home / "qdrant_memory" / "consolidation" / "watcher.log"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert events[-1]["event"] == "watcher_run"
    assert events[-1]["report_id"] == "r1"
    assert events[-1]["alerted"] is True


def test_watcher_run_unchanged_signature_suppresses_alert_unless_forced(monkeypatch, tmp_path):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    report = {
        "dry_run": True,
        "report_only": True,
        "report_id": "r1",
        "scope": "both",
        "proposals": [{"proposal_id": "p1", "proposal_type": "duplicate", "suggested_action": "merge", "affected_ids": ["a"]}],
    }
    parser = _parser()

    first = parser.parse_args(["qdrant", "watcher", "run", "--json"])
    assert execute_command(first, provider_factory=lambda: StaticJsonProvider({"qdrant_memory_consolidate": report})) == 0

    second = parser.parse_args(["qdrant", "watcher", "run", "--json"])
    assert execute_command(second, provider_factory=lambda: StaticJsonProvider({"qdrant_memory_consolidate": report})) == 0
    state_path = tmp_path / "hermes" / "qdrant_memory" / "consolidation" / "watcher_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_signature_changed"] is False
    assert state["last_alerted"] is False

    forced = parser.parse_args(["qdrant", "watcher", "run", "--force-alert", "--json"])
    assert execute_command(forced, provider_factory=lambda: StaticJsonProvider({"qdrant_memory_consolidate": report})) == 0
    state = json.loads(state_path.read_text(encoding="utf-8"))
    assert state["last_signature_changed"] is False
    assert state["last_alerted"] is True
    assert state["last_force_alert"] is True


def test_watcher_run_guarded_auto_applies_eligible_proposals_and_records_state(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    report = {
        "dry_run": True,
        "report_only": True,
        "report_id": "r-auto",
        "scope": "both",
        "proposals": [
            {
                "proposal_id": "dup1",
                "proposal_type": "duplicate_cluster",
                "suggested_action": "merge_review_only",
                "affected_ids": ["a", "b"],
                "confidence": 0.99,
                "risk": "low",
                "match_kind": "exact_normalized",
                "guarded_auto_eligible": True,
                "preauthorized_policy": "guarded-auto:exact-duplicate-merge",
            },
            {
                "proposal_id": "secret1",
                "proposal_type": "quality_warning",
                "suggested_action": "manual_secret_review_only",
                "affected_ids": ["s"],
                "risk": "high",
            },
        ],
    }
    provider = ApplyingProvider(report)
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "run", "--autonomy-mode", "guarded-auto", "--json"])

    assert execute_command(args, provider_factory=lambda: provider) == 0

    assert provider.calls[0][0] == "qdrant_memory_consolidate"
    assert provider.calls[1][0] == "qdrant_memory_consolidation_apply"
    assert provider.calls[1][1]["proposal_id"] == "dup1"
    assert provider.calls[1][1]["action"] == "merge"
    payload = json.loads(capsys.readouterr().out)
    assert payload["guarded_auto"]["enabled"] is True
    assert len(payload["guarded_auto"]["applied"]) == 1
    state = json.loads((hermes_home / "qdrant_memory" / "consolidation" / "watcher_state.json").read_text(encoding="utf-8"))
    assert state["last_guarded_auto_mode"] == "guarded-auto"
    assert state["last_guarded_auto_applied_count"] == 1
    assert state["last_alerted"] is True


def test_watcher_guarded_auto_skips_near_duplicate_even_if_marked_eligible(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    report = {
        "dry_run": True,
        "report_only": True,
        "report_id": "r-near",
        "scope": "memory",
        "proposals": [
            {
                "proposal_id": "dup-near",
                "proposal_type": "duplicate_cluster",
                "affected_ids": ["a", "b"],
                "confidence": 0.99,
                "risk": "medium",
                "match_kind": "near_duplicate",
                "guarded_auto_eligible": True,
                "preauthorized_policy": "guarded-auto:exact-duplicate-merge",
            }
        ],
    }
    provider = ApplyingProvider(report)
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "run", "--autonomy-mode", "guarded-auto", "--json"])

    assert execute_command(args, provider_factory=lambda: provider) == 0

    assert [call[0] for call in provider.calls] == ["qdrant_memory_consolidate"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["guarded_auto"]["enabled"] is True
    assert payload["guarded_auto"]["applied"] == []
    assert payload["guarded_auto"]["skipped"][0]["proposal_id"] == "dup-near"


def test_watcher_guarded_auto_skips_manual_review_and_reconsolidation(monkeypatch, tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes"))
    report = {
        "dry_run": True,
        "report_only": True,
        "report_id": "r-manual",
        "scope": "memory",
        "proposals": [
            {
                "proposal_id": "heading-generic",
                "proposal_type": "heading_noise",
                "affected_ids": ["h"],
                "risk": "medium",
                "guarded_auto_eligible": False,
                "manual_review_required": True,
            },
            {
                "proposal_id": "fact-conflict",
                "proposal_type": "reconsolidation_candidate",
                "affected_ids": ["a", "b"],
                "risk": "high",
                "manual_review_required": True,
            },
        ],
    }
    provider = ApplyingProvider(report)
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "run", "--autonomy-mode", "guarded-auto", "--json"])

    assert execute_command(args, provider_factory=lambda: provider) == 0

    assert [call[0] for call in provider.calls] == ["qdrant_memory_consolidate"]
    payload = json.loads(capsys.readouterr().out)
    assert payload["guarded_auto"]["applied"] == []
    assert len(payload["guarded_auto"]["skipped"]) == 2


def test_watcher_run_zero_proposal_clears_stale_counts_and_artifact(monkeypatch, tmp_path):
    """Zero-proposal clean run must overwrite stale last_counts, last_artifact, last_signature."""
    from qdrant_memory.cli_core import execute_command

    hermes_home = tmp_path / "hermes"
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    state_dir = hermes_home / "qdrant_memory" / "consolidation"
    state_dir.mkdir(parents=True)

    # Pre-write stale state with legacy fields
    stale_state = {
        "last_run_at": "2025-01-01T00:00:00",
        "last_counts": {"quality_warning": 9},
        "last_total_proposals": 3,
        "last_artifact": {"path": "/old/path/report-old.json", "proposal_count": 3},
        "last_signature": "deadbeef",
        "last_proposal_signature": "deadbeef",
        "last_report_id": "old-report",
    }
    (state_dir / "watcher_state.json").write_text(
        json.dumps(stale_state), encoding="utf-8"
    )

    # Clean report — no proposals, empty summary
    report = {
        "dry_run": True,
        "report_only": True,
        "report_id": "clean-r1",
        "scope": "both",
        "summary": {},
        "artifact": {"path": str(state_dir / "report-clean-r1.json"), "proposal_count": 0},
        "proposals": [],
    }
    provider = StaticJsonProvider({"qdrant_memory_consolidate": report})
    parser = _parser()
    args = parser.parse_args(["qdrant", "watcher", "run", "--json"])

    assert execute_command(args, provider_factory=lambda: provider) == 0

    state = json.loads(
        (state_dir / "watcher_state.json").read_text(encoding="utf-8")
    )
    # last_counts must reflect current summary (empty)
    assert state["last_counts"] == {}, f"expected empty dict, got {state['last_counts']}"
    # last_total_proposals must reflect current proposal count (0)
    assert state["last_total_proposals"] == 0
    # last_artifact must reflect current artifact
    assert state["last_artifact"] == report["artifact"]
    # last_signature must be a 16-char short signature
    assert isinstance(state["last_signature"], str) and len(state["last_signature"]) == 16
    # Existing fields preserved
    assert state["last_report_id"] == "clean-r1"
    assert state["last_proposal_count"] == 0
    assert state["last_signature_changed"] is True
