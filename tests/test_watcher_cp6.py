"""CP6 watcher: rotating scan window and guarded-auto safety.

Failure modes these tests target (written before the code, see the CP6 report):
W1/W2 the cursor is not persisted or never reaches the scroll; W3 a failed run advances it;
W4 no wrap at the end; G1 a session_origin point is touched; G2 identity/money/security
items are applied; G3 a reversible cleanup deletes; G4/G8 a failed or missing backup still
mutates; G6/G7 an applied action has no audit record or digest-bound snapshot.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from test_consolidation_apply import FakeQdrant, _point, _provider


def _parser():
    import argparse

    import cli

    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    qdrant_parser = subparsers.add_parser("qdrant")
    cli.register_cli(qdrant_parser)
    return parser


class WindowQdrant(FakeQdrant):
    """FakeQdrant plus Qdrant's scroll semantics: point-ID order, offset is a lower bound."""

    def __init__(self, by_collection=None):
        super().__init__(by_collection)
        self.windows: list[list[str]] = []
        self.fail_next_window = False

    def scroll_window(self, name, filter, *, start_offset=None, max_total, page_limit=256, with_payload=True, with_vector=False):
        if self.fail_next_window:
            self.fail_next_window = False
            raise RuntimeError("qdrant unavailable")
        ordered = sorted(self.by_collection.get(name, []), key=lambda point: str(point["id"]))
        if start_offset is not None:
            ordered = [point for point in ordered if str(point["id"]) >= str(start_offset)]
        page = ordered[:max_total]
        rest = ordered[max_total:]
        self.windows.append([str(point["id"]) for point in page])
        return page, (str(rest[0]["id"]) if rest else None)


def _run_watcher(argv, provider):
    from qdrant_memory.cli_core import execute_command

    return execute_command(_parser().parse_args(argv), provider_factory=lambda: provider)


def _state(home: Path) -> dict:
    return json.loads((home / "qdrant_memory" / "consolidation" / "watcher_state.json").read_text(encoding="utf-8"))


def test_rotating_window_covers_the_collection_over_nights_then_wraps(monkeypatch, tmp_path, capsys):
    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_QDRANT_MEMORY_COLLECTION_NAME", "memory")
    monkeypatch.setenv("HERMES_QDRANT_MEMORY_LEARNING_COLLECTION_NAME", "learnings")
    ids = [f"p{index:02d}" for index in range(25)]
    provider = _provider(home)
    provider._qdrant = WindowQdrant({"memory": [_point(point_id, f"distinct memory number {point_id}") for point_id in ids], "learnings": []})
    argv = ["qdrant", "watcher", "run", "--scope", "memory", "--max-points", "10", "--scan-window", "rotating", "--json"]

    # Oracle from the fixture alone: nights read consecutive 10-ID slices in ID order.
    expected = [set(ids[0:10]), set(ids[10:20]), set(ids[20:25]), set(ids[0:10])]

    assert _run_watcher(argv, provider) == 0
    assert _run_watcher(argv, provider) == 0
    # A failed night must not advance the cursor (W3).
    provider._qdrant.fail_next_window = True
    assert _run_watcher(argv, provider) == 1
    assert _state(home)["scan_window"]["memory"]["next_offset"] == "p20"
    assert _run_watcher(argv, provider) == 0
    assert _state(home)["scan_window"]["memory"]["next_offset"] is None
    assert _state(home)["scan_window"]["memory"]["cycle"] == 1
    assert _state(home)["scan_window"]["memory"]["last_completed_cycle_scanned"] == 25
    assert _run_watcher(argv, provider) == 0
    capsys.readouterr()

    nights = [set(window) for window in provider._qdrant.windows]
    assert nights == expected
    union_sizes = []
    seen: set[str] = set()
    for night in nights:
        seen |= night
        union_sizes.append(len(seen))
    assert union_sizes == [10, 20, 25, 25]


def _old(days: int = 400) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()


_CITED = {"captured_session_id": "20261101_000000_abcdef", "citation_status": "verified", "citations": []}
_STALE = {"importance": 1, "confidence": 0.3, "access_count": 0}


def _maintenance_fixture() -> list[dict]:
    return [
        _point("h-ok", "Tareas"),
        _point("h-cited", "Notas", session_origin=_CITED),
        _point("d-1", "Always dry-run before live vault indexing", importance=6, created_at=_old(10)),
        _point("d-2", "Always dry-run before live vault indexing", importance=5, created_at=_old(9)),
        _point("c-1", "Prefer ripgrep over grep for repository search", created_at=_old(10)),
        _point("c-2", "Prefer ripgrep over grep for repository search", created_at=_old(9), session_origin=_CITED),
        _point("s-ok", "Old scratch note about a renamed folder layout", created_at=_old(), **_STALE),
        _point("s-money", "Factura del hosting pagada en marzo", created_at=_old(), **_STALE),
        _point("s-security", "Rotated the ssh deploy key on the build box", created_at=_old(), **_STALE),
        _point("s-identity", "Lives near the old train station", created_at=_old(), source_type="user_memory", **_STALE),
        _point("s-cited", "Scratch remark from a chat about lunch plans", created_at=_old(), session_origin=_CITED, **_STALE),
    ]


def _touched(qdrant: FakeQdrant) -> set[str]:
    touched = {str(point_id) for _name, point_id, _payload in qdrant.payload_updates}
    for _name, ids in qdrant.deleted_ids:
        touched |= {str(point_id) for point_id in ids}
    return touched


def test_guarded_auto_applies_only_low_risk_points_reversibly_with_backup_and_snapshots(monkeypatch, tmp_path, capsys):
    from qdrant_memory.backup import read_backup_manifest

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = _provider(home)
    provider._qdrant = WindowQdrant({"memory": _maintenance_fixture(), "learnings": []})

    argv = ["qdrant", "watcher", "run", "--scope", "memory", "--autonomy-mode", "guarded-auto", "--json"]
    assert _run_watcher(argv, provider) == 0
    payload = json.loads(capsys.readouterr().out)
    summary = payload["guarded_auto"]
    qdrant = provider._qdrant

    # Exactly the uncited, non-sensitive, low-risk points changed (G1, G2) ...
    assert _touched(qdrant) == {"h-ok", "d-1", "d-2", "s-ok"}
    # ... and nothing was deleted: cleanups and duplicate retirement are quarantines (G3).
    assert qdrant.deleted_ids == []
    quarantined = {point_id for _name, point_id, update in qdrant.payload_updates if update.get("consolidation_quarantined")}
    assert quarantined == {"h-ok", "d-2", "s-ok"}
    assert summary["errors"] == []
    assert sorted(entry["action"] for entry in summary["applied"]) == ["merge", "quarantine", "quarantine"]

    # One full backup before the first mutation, and per action an audit record plus a
    # digest-bound snapshot that is itself a restorable partial backup (G6, G7).
    backup_id = summary["backup"]["backup_id"]
    full = read_backup_manifest(provider._config, hermes_home=str(home), backup_id=backup_id)
    assert full.get("partial") is not True
    assert full["collections"]["memory"]["count"] == len(_maintenance_fixture())
    report = json.loads(Path(payload["artifact"]["path"]).read_text(encoding="utf-8"))
    sealed = {proposal["proposal_id"]: proposal for proposal in report["proposals"]}
    records = [json.loads(path.read_text(encoding="utf-8")) for path in (home / "qdrant_memory" / "consolidation" / "applications").glob("application-*.json")]
    assert len(records) == len(summary["applied"]) == 3
    for record in records:
        assert record["guarded_auto"] is True
        assert record["pre_apply_backup_id"] == backup_id
        snapshot = record["pre_mutation_snapshot"]
        manifest = read_backup_manifest(provider._config, hermes_home=str(home), backup_id=snapshot["backup_id"])
        assert manifest["partial"] is True
        assert manifest["snapshot"]["proposal_id"] == record["proposal_id"]
        assert snapshot["point_digests"] == sealed[record["proposal_id"]]["guarded_auto_snapshot"]["point_digests"]
        assert sorted(snapshot["point_ids"]) == sorted(record["affected_ids"])

    state = _state(home)
    assert state["last_guarded_auto_applied_count"] == 3
    assert state["last_alerted"] is True


def test_watcher_apply_without_a_verified_full_backup_writes_nothing(monkeypatch, tmp_path):
    """Provider layer (G8): the exact-ID apply path refuses a watcher mutation on its own."""
    from qdrant_memory.backup import create_point_snapshot_backup

    home = tmp_path / "home"
    monkeypatch.setenv("HERMES_HOME", str(home))
    provider = _provider(home)
    provider._qdrant = WindowQdrant({"memory": [_point("h-ok", "Tareas")], "learnings": []})
    report = json.loads(provider.handle_tool_call("qdrant_memory_consolidate", {"scope": "memory", "persist": True}))
    proposal = next(p for p in report["proposals"] if p["proposal_type"] == "heading_noise")
    assert proposal["guarded_auto_eligible"] is True
    snapshot = create_point_snapshot_backup(
        [_point("h-ok", "Tareas")], provider._config, hermes_home=str(home), scope="memory",
        collection_name="memory", snapshot={"kind": "test"},
    )
    base = {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"], "action": "quarantine",
            "dry_run": False, "approve": True, "_guarded_auto": True}

    for extra in ({}, {"guarded_auto_backup_id": "backup-does-not-exist"}, {"guarded_auto_backup_id": snapshot["backup_id"]}):
        result = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {**base, **extra}))
        assert "verified pre-mutation backup" in result["error"]
    assert provider._qdrant.payload_updates == []
    assert provider._qdrant.deleted_ids == []

    full = provider.guarded_auto_backup(scope="memory")
    applied = json.loads(provider.handle_tool_call("qdrant_memory_consolidation_apply", {**base, "guarded_auto_backup_id": full["backup_id"]}))
    assert applied["applied"] is True
    assert _touched(provider._qdrant) == {"h-ok"}


class _ObservingProvider:
    """Applies whatever it is asked to: observes whether the caller still mutates."""

    def __init__(self, report, *, backup_error: bool):
        self.report = report
        self.backup_error = backup_error
        self.live_applies: list[dict] = []

    def guarded_auto_backup(self, scope):
        if self.backup_error:
            raise OSError("disk full")
        return {"backup_id": "backup-ok-0001", "scope": scope, "collections": {}}

    def handle_tool_call(self, tool_name, args):
        if tool_name == "qdrant_memory_consolidate":
            return json.dumps(self.report)
        if args.get("dry_run"):
            return json.dumps({"dry_run": True, "would_apply": True})
        self.live_applies.append(args)
        return json.dumps({"dry_run": False, "applied": True})


@pytest.mark.parametrize("backup_error", [True, False])
def test_watcher_caller_applies_nothing_when_the_backup_fails(monkeypatch, tmp_path, capsys, backup_error):
    """Caller layer (G4); the False case is the control that proves the path is live."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "home"))
    report = {
        "report_id": "r-backup",
        "scope": "memory",
        "collections": {"memory": "memory", "learning": "learnings"},
        "proposals": [
            {
                "proposal_id": "heading-1",
                "proposal_type": "heading_noise",
                "collection_name": "memory",
                "affected_ids": ["h-ok"],
                "risk": "low",
                "guarded_auto_eligible": True,
                "preauthorized_policy": "guarded-auto:heading-noise",
            }
        ],
    }
    provider = _ObservingProvider(report, backup_error=backup_error)
    argv = ["qdrant", "watcher", "run", "--scope", "memory", "--autonomy-mode", "guarded-auto", "--json"]
    assert _run_watcher(argv, provider) == 0
    summary = json.loads(capsys.readouterr().out)["guarded_auto"]
    if backup_error:
        assert provider.live_applies == []
        assert summary["errors"] == [{"code": "backup_failed", "scope": "memory"}]
        assert summary["applied"] == []
    else:
        assert [call["guarded_auto_backup_id"] for call in provider.live_applies] == ["backup-ok-0001"]
        assert len(summary["applied"]) == 1
