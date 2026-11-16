"""Retention for the full backups the watcher takes before a guarded-auto mutation.

A guarded-auto night that mutates takes one full plugin backup of the touched collections
(about 264 MB for a 17.7k-point memory collection). Without retention that is roughly 11 GB
per 60-night rotating cycle. ``prune_watcher_backups`` deletes only backups that are all of:

- tagged ``created_by: watcher_guarded_auto`` in their manifest (``create_backup`` writes the
  tag for the watcher's backup only), so manual backups and backups from before the tag are
  never candidates;
- full backups (``partial`` is not true): the per-action point snapshots are the undo of one
  action, a few KB each, and are kept;
- not among the newest ``keep_last`` watcher backups (by manifest ``created_at``);
- not referenced (``pre_apply_backup_id`` or ``pre_mutation_snapshot.backup_id``) by an
  application record created within the last ``keep_days`` days.

It deletes nothing when any application record cannot be read (it could not prove a backup
unreferenced), and never touches a directory whose manifest cannot be read.
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from qdrant_memory.backup import BackupError, _load_manifest_from_dir, backup_dir_for_id, backup_root
from qdrant_memory.consolidation import artifact_root_path

WATCHER_BACKUP_ORIGIN = "watcher_guarded_auto"
DEFAULT_KEEP_LAST = 7
DEFAULT_KEEP_DAYS = 30


def _parse_time(value: Any) -> datetime | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def referenced_backup_ids(applications_dir: Path, *, since: datetime) -> set[str]:
    """Backup ids an application record created at or after ``since`` refers to.

    A record whose time cannot be read counts as recent. Raises ``ValueError`` when a record
    file cannot be read or parsed.
    """
    referenced: set[str] = set()
    if not applications_dir.is_dir():
        return referenced
    for path in sorted(applications_dir.glob("application-*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"application record {path.name} is unreadable: {type(exc).__name__}") from exc
        if not isinstance(record, dict):
            raise ValueError(f"application record {path.name} is not an object")
        created = _parse_time(record.get("created_at"))
        if created is not None and created < since:
            continue
        snapshot = record.get("pre_mutation_snapshot")
        for value in (record.get("pre_apply_backup_id"),
                      snapshot.get("backup_id") if isinstance(snapshot, dict) else None):
            if isinstance(value, str) and value.strip():
                referenced.add(value.strip())
    return referenced


def prune_watcher_backups(config: dict[str, Any], *, hermes_home: str, keep_last: int = DEFAULT_KEEP_LAST,
                          keep_days: int = DEFAULT_KEEP_DAYS, now: datetime | None = None) -> dict[str, Any]:
    """Delete the watcher's full backups outside the retention rule; returns what it did."""
    keep_last = max(1, int(keep_last))
    keep_days = max(0, int(keep_days))
    clock = now or datetime.now(timezone.utc)
    result: dict[str, Any] = {"keep_last": keep_last, "keep_days": keep_days, "deleted": [], "kept": [],
                              "protected": []}
    root = backup_root(config, hermes_home)
    if not root.is_dir():
        return result
    applications = artifact_root_path(hermes_home, str(config.get("consolidation_artifact_dir") or "")) / "applications"
    try:
        referenced = referenced_backup_ids(applications, since=clock - timedelta(days=keep_days))
    except ValueError as exc:
        result["error"] = str(exc)
        return result
    candidates: list[tuple[str, str]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        try:
            manifest = _load_manifest_from_dir(child)
        except BackupError:
            continue  # unreadable or not a plugin backup: never ours to delete
        if manifest.get("created_by") != WATCHER_BACKUP_ORIGIN or manifest.get("partial") is True:
            continue
        backup_id = str(manifest.get("backup_id") or "")
        if backup_id != child.name:
            continue
        candidates.append((str(manifest.get("created_at") or ""), backup_id))
    candidates.sort(reverse=True)
    for index, (_created, backup_id) in enumerate(candidates):
        if index < keep_last:
            result["kept"].append(backup_id)
            continue
        if backup_id in referenced:
            result["protected"].append(backup_id)
            continue
        try:
            shutil.rmtree(backup_dir_for_id(root, backup_id))
        except (OSError, BackupError) as exc:
            result.setdefault("errors", []).append({"backup_id": backup_id, "error": type(exc).__name__})
            continue
        result["deleted"].append(backup_id)
    return result
