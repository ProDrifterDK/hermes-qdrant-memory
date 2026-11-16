"""Watcher closing fixes: a failed pre-mutation backup holds the rotating cursor.

Failure modes (written before the code):
C1 a night whose pre-mutation backup failed still advances the rotating cursor, so that
   window's actions wait a whole cycle (~60 nights) instead of being retried the next night;
C2 the hold is so broad that an ordinary night (report-only, or guarded-auto with only
   per-proposal refusals) stops advancing, and the watcher reads the same window forever;
C3 the held night is silent (no alert, nothing in the state or the log says why).
"""
from __future__ import annotations

import json

from test_consolidation_apply import _point, _provider
from test_watcher_cp6 import WindowQdrant, _run_watcher, _state

IDS = [f"p{index:02d}" for index in range(30)]


def _window_fixture():
    # p12 is a known heading-noise chunk guarded-auto quarantines; the rest are distinct notes.
    return [_point(pid, "Tareas" if pid == "p12" else f"distinct memory number {pid} about topic {pid}")
            for pid in IDS]


def _rotating(home, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_QDRANT_MEMORY_COLLECTION_NAME", "memory")
    monkeypatch.setenv("HERMES_QDRANT_MEMORY_LEARNING_COLLECTION_NAME", "learnings")
    provider = _provider(home)
    provider._qdrant = WindowQdrant({"memory": _window_fixture(), "learnings": []})
    return provider


ARGV = ["qdrant", "watcher", "run", "--scope", "memory", "--max-points", "10", "--scan-window", "rotating", "--json"]
GUARDED = ARGV + ["--autonomy-mode", "guarded-auto"]


def test_a_failed_backup_holds_the_cursor_and_the_window_is_retried_next_night(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    provider = _rotating(home, monkeypatch)
    real_backup = provider.guarded_auto_backup

    assert _run_watcher(GUARDED, provider) == 0  # night 1: p00-p09, nothing eligible
    assert _state(home)["scan_window"]["memory"]["next_offset"] == "p10"

    def failing_backup(scope):
        raise OSError("disk full")

    monkeypatch.setattr(provider, "guarded_auto_backup", failing_backup)
    assert _run_watcher(GUARDED, provider) == 0  # night 2: p10-p19, the backup fails
    night2 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert night2["guarded_auto"]["errors"] == [{"code": "backup_failed", "scope": "memory"}]
    state = _state(home)
    assert state["scan_window"]["memory"]["next_offset"] == "p10"  # C1: held
    assert state["scan_window_held"]["reason"] == "backup_failed"  # C3
    assert state["last_alerted"] is True and state["last_guarded_auto_error_count"] == 1
    log = (home / "qdrant_memory" / "consolidation" / "watcher.log").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[-1])["scan_window_held"] == "backup_failed"

    monkeypatch.setattr(provider, "guarded_auto_backup", real_backup)
    assert _run_watcher(GUARDED, provider) == 0  # night 3: p10-p19 again, applied
    night3 = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert [entry["action"] for entry in night3["guarded_auto"]["applied"]] == ["quarantine"]
    state = _state(home)
    assert state["scan_window"]["memory"]["next_offset"] == "p20"
    assert "scan_window_held" not in state

    windows = [window[0] + ".." + window[-1] for window in provider._qdrant.windows]
    assert windows == ["p00..p09", "p10..p19", "p10..p19"]


def test_report_only_and_rejected_proposals_still_advance(monkeypatch, tmp_path, capsys):
    """C2 control: only a night that could not attempt its actions is held."""
    home = tmp_path / "home"
    provider = _rotating(home, monkeypatch)
    assert _run_watcher(ARGV, provider) == 0
    assert _run_watcher(ARGV, provider) == 0  # report-only over the heading window
    assert _state(home)["scan_window"]["memory"]["next_offset"] == "p20"

    home2 = tmp_path / "home2"
    provider2 = _rotating(home2, monkeypatch)
    original2 = provider2.handle_tool_call
    monkeypatch.setattr(provider2, "handle_tool_call",
                        lambda name, args: json.dumps({"error": "refused for the test"})
                        if name == "qdrant_memory_consolidation_apply" else original2(name, args))
    assert _run_watcher(GUARDED, provider2) == 0
    assert _run_watcher(GUARDED, provider2) == 0  # p10-p19: the heading proposal is refused
    errors = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["guarded_auto"]["errors"]
    assert [e["code"] for e in errors] == ["provider_rejected"]
    assert _state(home2)["scan_window"]["memory"]["next_offset"] == "p20"
