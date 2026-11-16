"""Watcher closing fixes: cursor hold on a failed backup, backup retention.

Failure modes (written before the code):
C1 a night whose pre-mutation backup failed still advances the rotating cursor, so that
   window's actions wait a whole cycle (~60 nights) instead of being retried the next night;
C2 the hold is so broad that an ordinary night (report-only, or guarded-auto with only
   per-proposal refusals) stops advancing, and the watcher reads the same window forever;
C3 the held night is silent (no alert, nothing in the state or the log says why).

Backup retention (one ~264 MB full backup per mutating night, no retention):
R1 a manual backup (``hermes qdrant backup create``) is deleted;
R2 a per-action point snapshot (the undo of one action) is deleted;
R3 a watcher backup an application record references inside the window is deleted;
R4 nothing is ever deleted (unbounded growth): watcher backups beyond the newest N that no
   recent record references must go;
R5 an unreadable application record (cannot prove a backup unreferenced) still deletes;
R6 a directory whose manifest cannot be read is deleted;
R7 a report-only night deletes anything;
R8 the configured counts are ignored;
R9 a deletion leaves no trace in the run output or the watcher log.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

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


# --- backup retention -----------------------------------------------------------------------

NOW = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)


def _iso(days_ago: float) -> str:
    return (NOW - timedelta(days=days_ago)).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _fake_backup(root, backup_id, days_ago, *, created_by=None, partial=False, broken=False):
    """A backup directory in the plugin's layout (manifest + one collection file)."""
    from qdrant_memory.backup import SCHEMA_VERSION

    path = root / backup_id
    path.mkdir(parents=True)
    (path / "memory.jsonl").write_text('{"schema_version": 1}\n', encoding="utf-8")
    manifest = {"schema_version": SCHEMA_VERSION, "artifact_type": "qdrant_memory_backup", "backup_id": backup_id,
                "created_at": _iso(days_ago), "scope": "memory",
                "collections": {"memory": {"collection_name": "memory", "file": "memory.jsonl", "count": 0}}}
    if created_by:
        manifest["created_by"] = created_by
    if partial:
        manifest["partial"] = True
    (path / "manifest.json").write_text("{not json" if broken else json.dumps(manifest), encoding="utf-8")
    return path


def _record(home, name, days_ago, **refs):
    applications = home / "qdrant_memory" / "consolidation" / "applications"
    applications.mkdir(parents=True, exist_ok=True)
    (applications / f"application-{name}.json").write_text(
        json.dumps({"applied": True, "created_at": _iso(days_ago), **refs}), encoding="utf-8")


def _seed_retention(home):
    from qdrant_memory.backup_retention import WATCHER_BACKUP_ORIGIN

    root = home / "qdrant_memory" / "backups"
    for day in range(1, 11):  # w01 (1 day old) .. w10 (10 days old)
        _fake_backup(root, f"backup-w{day:02d}", day, created_by=WATCHER_BACKUP_ORIGIN)
    _fake_backup(root, "backup-manual", 100)
    _fake_backup(root, "backup-snapshot", 90, created_by=WATCHER_BACKUP_ORIGIN, partial=True)
    _fake_backup(root, "backup-broken", 80, broken=True)
    _record(home, "recent", 10, pre_apply_backup_id="backup-w08",
            pre_mutation_snapshot={"backup_id": "backup-snapshot"})
    _record(home, "old", 45, pre_apply_backup_id="backup-w09")
    return root


def _listing(root):
    return sorted(path.name for path in root.iterdir())


def test_retention_keeps_the_newest_referenced_manual_snapshot_and_unreadable_backups(tmp_path):
    from qdrant_memory.backup_retention import prune_watcher_backups

    home = tmp_path / "home"
    root = _seed_retention(home)
    result = prune_watcher_backups({}, hermes_home=str(home), keep_last=3, keep_days=30, now=NOW)

    deleted = {"backup-w04", "backup-w05", "backup-w06", "backup-w07", "backup-w09", "backup-w10"}
    assert set(result["deleted"]) == deleted, result  # R4 (w09: its only record is 45 days old)
    assert _listing(root) == sorted({"backup-w01", "backup-w02", "backup-w03", "backup-w08",  # R3
                                     "backup-manual", "backup-snapshot", "backup-broken"})  # R1 R2 R6
    assert result["protected"] == ["backup-w08"]
    assert "error" not in result


def test_retention_counts_come_from_the_arguments(tmp_path):
    from qdrant_memory.backup_retention import prune_watcher_backups

    home = tmp_path / "home"
    root = _seed_retention(home)
    result = prune_watcher_backups({}, hermes_home=str(home), keep_last=1, keep_days=50, now=NOW)  # R8
    assert set(result["deleted"]) == {f"backup-w{d:02d}" for d in (2, 3, 4, 5, 6, 7, 10)}
    assert "backup-w09" in _listing(root) and "backup-w08" in _listing(root)


def test_an_unreadable_application_record_stops_retention(tmp_path):
    from qdrant_memory.backup_retention import prune_watcher_backups

    home = tmp_path / "home"
    root = _seed_retention(home)
    (home / "qdrant_memory" / "consolidation" / "applications" / "application-bad.json").write_text("{", encoding="utf-8")
    before = _listing(root)
    result = prune_watcher_backups({}, hermes_home=str(home), keep_last=3, keep_days=30, now=NOW)  # R5
    assert result["deleted"] == [] and "application record" in result["error"]
    assert _listing(root) == before


def _guarded_night(monkeypatch, tmp_path, argv_extra, **config):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = _provider(home)
    provider._config.update(config)
    provider._qdrant = WindowQdrant({"memory": [_point("h-ok", "Tareas"), _point("n-1", "distinct ordinary note")],
                                     "learnings": []})
    root = home / "qdrant_memory" / "backups"
    from qdrant_memory.backup_retention import WATCHER_BACKUP_ORIGIN

    _fake_backup(root, "backup-old-a", 60, created_by=WATCHER_BACKUP_ORIGIN)
    _fake_backup(root, "backup-old-b", 50, created_by=WATCHER_BACKUP_ORIGIN)
    _fake_backup(root, "backup-manual", 90)
    argv = ["qdrant", "watcher", "run", "--scope", "memory", "--json", *argv_extra]
    return home, root, provider, argv


def test_a_guarded_auto_night_prunes_old_watcher_backups_and_says_so(monkeypatch, tmp_path, capsys):
    home, root, provider, argv = _guarded_night(
        monkeypatch, tmp_path, ["--autonomy-mode", "guarded-auto"],
        guarded_auto_backup_keep_last=2, guarded_auto_backup_keep_days=30)
    assert _run_watcher(argv, provider) == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["guarded_auto"]
    tonight = summary["backup"]["backup_id"]
    assert [entry["action"] for entry in summary["applied"]] == ["quarantine"]
    # The full backup taken tonight is tagged as the watcher's and counts toward the newest N.
    manifest = json.loads((root / tonight / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["created_by"] == "watcher_guarded_auto"
    assert summary["backup_retention"]["deleted"] == ["backup-old-a"]  # R9
    names = set(_listing(root))
    assert {tonight, "backup-old-b", "backup-manual"} <= names and "backup-old-a" not in names
    log = (home / "qdrant_memory" / "consolidation" / "watcher.log").read_text(encoding="utf-8").splitlines()
    assert json.loads(log[-1])["backup_retention_deleted"] == ["backup-old-a"]  # R9


def test_a_report_only_night_deletes_nothing(monkeypatch, tmp_path, capsys):
    home, root, provider, argv = _guarded_night(monkeypatch, tmp_path, [], guarded_auto_backup_keep_last=1)
    before = _listing(root)
    assert _run_watcher(argv, provider) == 0  # R7
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])["guarded_auto"]
    assert "backup_retention" not in summary
    assert _listing(root) == before


# --- quarantine expiry wording (decision: quarantine is permanent until restored) -------------
#
# X1 the CLI help or the tool schema promises a later hard deletion that nothing performs;
# X2 something starts acting on ``consolidation_quarantine_until``: a guarded-auto night
#    hard-deletes, or silently un-quarantines, a point whose date has passed.


def _help(argv, capsys):
    import pytest

    from test_watcher_cp6 import _parser

    with pytest.raises(SystemExit):
        _parser().parse_args(argv + ["--help"])
    return " ".join(capsys.readouterr().out.split())


def test_quarantine_days_text_promises_no_deletion(capsys):
    from qdrant_memory.tools import CONSOLIDATION_APPLY_SCHEMA

    texts = [_help(["qdrant", "watcher", "run"], capsys), _help(["qdrant", "watcher", "install"], capsys),
             _help(["qdrant", "apply"], capsys),
             CONSOLIDATION_APPLY_SCHEMA["parameters"]["properties"]["quarantine_days"]["description"]]
    for text in texts:
        assert "deletion" not in text.lower(), text  # X1
    for text in texts[:2] + texts[3:]:
        assert "until restored" in text, text


def test_a_quarantined_point_past_its_date_stays_quarantined_and_is_never_deleted(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = _provider(home)
    past = {"consolidation_quarantined": True, "consolidation_quarantined_at": _iso(400),
            "consolidation_quarantine_until": _iso(370), "consolidation_quarantine_reason": "guarded-auto heading_noise"}
    provider._qdrant = WindowQdrant({"memory": [_point("q-old", "Tareas", **past), _point("h-new", "Notas")],
                                     "learnings": []})
    argv = ["qdrant", "watcher", "run", "--scope", "memory", "--autonomy-mode", "guarded-auto", "--json"]
    assert _run_watcher(argv, provider) == 0
    capsys.readouterr()
    qdrant = provider._qdrant
    assert qdrant.deleted_ids == []  # X2
    assert all(point_id != "q-old" for _name, point_id, _update in qdrant.payload_updates)
    stored = next(p for p in qdrant.by_collection["memory"] if p["id"] == "q-old")
    assert stored["payload"]["consolidation_quarantined"] is True
    assert {point_id for _name, point_id, _update in qdrant.payload_updates} == {"h-new"}  # control: the path is live


def test_retention_defaults_are_seven_and_thirty():
    from qdrant_memory.config import load_config

    config = load_config(hermes_home="/nonexistent", hermes_config={})
    assert config["guarded_auto_backup_keep_last"] == 7
    assert config["guarded_auto_backup_keep_days"] == 30
