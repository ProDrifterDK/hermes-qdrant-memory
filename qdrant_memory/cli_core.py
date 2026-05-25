from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from qdrant_memory.guarded_auto import GuardedAutoPolicy, apply_guarded_auto


class CliUsageError(ValueError):
    """Raised when CLI arguments violate the plugin safety contract."""


def _require_live_approval(args: Namespace) -> None:
    if getattr(args, "dry_run", True) is False and not getattr(args, "approve", False):
        raise CliUsageError("--approve is required when using --no-dry-run")


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _split_tags(values: Any) -> list[str]:
    if not values:
        return []
    if isinstance(values, str):
        values = [values]
    if not isinstance(values, list):
        return []
    tags: list[str] = []
    for raw in values:
        for item in str(raw).split(","):
            tag = item.strip()
            if tag:
                tags.append(tag)
    return tags


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _search_filter_tool_args(args: Namespace, *, include_collection: bool = False) -> dict[str, Any]:
    tool_args: dict[str, Any] = {}
    tags = _split_tags(getattr(args, "tag", []))
    if tags:
        tool_args["tags"] = tags
    for key in ("source", "file_path", "project_path", "since", "until"):
        value = _optional_text(getattr(args, key, None))
        if value:
            tool_args[key] = value
    if include_collection:
        collection = _optional_text(getattr(args, "collection", None))
        if collection:
            tool_args["collection"] = collection
    return tool_args


def _non_empty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CliUsageError(f"{name} is required")
    return text


_ARTIFACT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _safe_artifact_id(value: Any, name: str) -> str:
    text = _non_empty(value, name)
    if text in {".", ".."} or "/" in text or "\\" in text or not _ARTIFACT_ID_RE.match(text):
        raise CliUsageError(f"invalid {name}")
    return text


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or (Path.home() / ".hermes"))


_SECRET_CONFIG_KEYWORDS = (
    "api_key",
    "apikey",
    "token",
    "password",
    "passwd",
    "secret",
    "authorization",
    "bearer",
    "credential",
    "private_key",
)


def _is_secret_config_key(key: str) -> bool:
    normalized = key.lower().replace("-", "_")
    return any(keyword in normalized for keyword in _SECRET_CONFIG_KEYWORDS)


def _redact_url_component_secrets(component: str) -> str:
    if not component:
        return component
    try:
        pairs = parse_qsl(component, keep_blank_values=True)
    except Exception:
        pairs = []
    if pairs:
        redacted_pairs = [(key, "<redacted>" if _is_secret_config_key(key) else val) for key, val in pairs]
        return urlencode(redacted_pairs)
    if _is_secret_config_key(component):
        return "<redacted>"
    return component


def _redact_credentialed_url(value: str) -> str:
    try:
        parts = urlsplit(value)
    except Exception:
        return value
    if not parts.scheme or not parts.netloc:
        return value
    netloc = parts.netloc
    if parts.username is not None or parts.password is not None:
        host = parts.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        netloc = f"<redacted>@{host}"
        try:
            if parts.port is not None:
                netloc = f"{netloc}:{parts.port}"
        except ValueError:
            pass
    query = _redact_url_component_secrets(parts.query)
    fragment = _redact_url_component_secrets(parts.fragment)
    if netloc == parts.netloc and query == parts.query and fragment == parts.fragment:
        return value
    return urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def _redact_config_value(key: str, value: Any) -> Any:
    if _is_secret_config_key(key):
        return "<redacted>" if value else ""
    if isinstance(value, str):
        return _redact_credentialed_url(value)
    if isinstance(value, dict):
        return {str(item_key): _redact_config_value(str(item_key), item_value) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [_redact_config_value(key, item) for item in value]
    return value


def redact_config(config: dict[str, Any]) -> dict[str, Any]:
    return {str(key): _redact_config_value(str(key), value) for key, value in config.items()}


def _redacted_config() -> dict[str, Any]:
    from qdrant_memory.config import load_config

    config = load_config(hermes_home=str(_hermes_home()))
    return redact_config(dict(config))


_WATCHER_CRON_BEGIN = "# BEGIN HERMES_QDRANT_WATCHER"
_WATCHER_CRON_END = "# END HERMES_QDRANT_WATCHER"
_WATCHER_SIGNATURE_KEYS = {
    "last_proposal_signature",
    "last_alert_signature",
    "last_alert_at",
    "last_signature_changed",
    "last_alerted",
    "last_force_alert",
}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _watcher_dir() -> Path:
    return _hermes_home() / "qdrant_memory" / "consolidation"


def _watcher_state_path() -> Path:
    return _watcher_dir() / "watcher_state.json"


def _watcher_log_path() -> Path:
    return _watcher_dir() / "watcher.log"


def _read_watcher_state() -> tuple[dict[str, Any], bool, str | None]:
    state_path = _watcher_state_path()
    if not state_path.exists():
        return {}, False, None
    try:
        parsed = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"error": f"failed to read watcher state: {exc}"}, True, str(exc)
    return (parsed if isinstance(parsed, dict) else {"raw": parsed}), True, None


def _write_watcher_state(state: dict[str, Any]) -> None:
    state_path = _watcher_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_watcher_log(event: dict[str, Any]) -> None:
    log_path = _watcher_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _read_watcher_log_tail(limit: int = 20) -> dict[str, Any]:
    log_path = _watcher_log_path()
    events: list[Any] = []
    exists = log_path.exists()
    if exists:
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            return {"log_path": str(log_path), "log_exists": True, "count": 0, "events": [], "error": str(exc)}
        for line in lines[-max(1, int(limit or 20)) :]:
            if not line.strip():
                continue
            try:
                parsed = json.loads(line)
            except Exception:
                parsed = {"raw": line}
            events.append(redact_config(parsed) if isinstance(parsed, dict) else parsed)
    return {"log_path": str(log_path), "log_exists": exists, "count": len(events), "events": events}


def _read_user_crontab() -> str:
    result = subprocess.run(["crontab", "-l"], capture_output=True, text=True, check=False)
    if result.returncode == 0:
        return result.stdout
    if result.returncode == 1:
        return ""
    raise RuntimeError((result.stderr or "failed to read crontab").strip())


def _write_user_crontab(text: str) -> None:
    result = subprocess.run(["crontab", "-"], input=text, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or "failed to write crontab").strip())


def _extract_managed_cron_block(crontab_text: str) -> tuple[str | None, str]:
    lines = crontab_text.splitlines(keepends=True)
    begin_index: int | None = None
    for index, line in enumerate(lines):
        if line.strip() == _WATCHER_CRON_BEGIN:
            begin_index = index
            break
    if begin_index is None:
        return None, crontab_text
    end_index: int | None = None
    for index in range(begin_index + 1, len(lines)):
        if lines[index].strip() == _WATCHER_CRON_END:
            end_index = index
            break
    if end_index is None:
        raise CliUsageError("managed watcher crontab block is missing its END sentinel")
    block = "".join(lines[begin_index : end_index + 1]).rstrip("\n")
    remaining = "".join(lines[:begin_index] + lines[end_index + 1 :])
    return block, remaining


def _watcher_schedule_status() -> dict[str, Any]:
    try:
        crontab_text = _read_user_crontab()
    except Exception as exc:
        return {"installed": False, "error": str(exc)}
    block, _ = _extract_managed_cron_block(crontab_text)
    return {"installed": block is not None, "block": block or ""}


def _validate_cron_schedule(schedule: str) -> str:
    text = str(schedule or "").strip()
    if "\n" in text or "\r" in text:
        raise CliUsageError("schedule must be one line")
    parts = text.split()
    if len(parts) != 5:
        raise CliUsageError("schedule must contain exactly five cron fields")
    return " ".join(parts)


def _watcher_run_command(args: Namespace) -> str:
    parts = [
        "hermes",
        "qdrant",
        "watcher",
        "run",
        "--scope",
        args.scope,
        "--max-points",
        str(args.max_points),
        "--max-groups",
        str(args.max_groups),
        "--reconsolidation-max-candidates",
        str(args.reconsolidation_max_candidates),
    ]
    if not getattr(args, "include_reconsolidation", True):
        parts.append("--no-include-reconsolidation")
    if getattr(args, "autonomy_mode", "report-only") != "report-only":
        parts.extend(["--autonomy-mode", args.autonomy_mode])
        parts.extend(["--max-auto-actions", str(getattr(args, "max_auto_actions", 10))])
        parts.extend(["--quarantine-days", str(getattr(args, "quarantine_days", 30))])
    parts.append("--json")
    return " ".join(shlex.quote(str(part)) for part in parts)


def _managed_cron_block(args: Namespace) -> str:
    schedule = _validate_cron_schedule(args.schedule)
    log_path = _watcher_log_path()
    command = _watcher_run_command(args)
    return "\n".join(
        [
            _WATCHER_CRON_BEGIN,
            f"# schedule: {schedule}",
            f"{schedule} {command} >> {shlex.quote(str(log_path))} 2>&1",
            _WATCHER_CRON_END,
        ]
    )


def _install_watcher(args: Namespace) -> dict[str, Any]:
    new_block = _managed_cron_block(args)
    crontab_text = _read_user_crontab()
    existing_block, remaining = _extract_managed_cron_block(crontab_text)
    changed = existing_block != new_block
    if existing_block and changed and not getattr(args, "approve", False):
        raise CliUsageError("--approve is required to replace an existing managed watcher entry")
    if changed:
        prefix = remaining.rstrip("\n")
        updated = (prefix + "\n" if prefix else "") + new_block + "\n"
        _write_user_crontab(updated)
    state, _, _ = _read_watcher_state()
    if state.get("error"):
        state = {}
    timestamp = _now_iso()
    state.update(
        {
            "installed": True,
            "schedule": _validate_cron_schedule(args.schedule),
            "command": _watcher_run_command(args),
            "installed_at": timestamp,
            "updated_at": timestamp,
        }
    )
    _write_watcher_state(state)
    return {
        "installed": True,
        "changed": changed,
        "schedule": state["schedule"],
        "command": state["command"],
        "state_path": str(_watcher_state_path()),
        "log_path": str(_watcher_log_path()),
    }


def _uninstall_watcher(args: Namespace) -> dict[str, Any]:
    if not getattr(args, "approve", False):
        raise CliUsageError("--approve is required to uninstall the managed watcher entry")
    crontab_text = _read_user_crontab()
    existing_block, remaining = _extract_managed_cron_block(crontab_text)
    changed = existing_block is not None
    if changed:
        _write_user_crontab(remaining.rstrip("\n") + ("\n" if remaining.strip() else ""))
    state, _, _ = _read_watcher_state()
    if state.get("error"):
        state = {}
    timestamp = _now_iso()
    state.update({"installed": False, "uninstalled_at": timestamp, "updated_at": timestamp})
    _write_watcher_state(state)
    return {"installed": False, "changed": changed, "state_path": str(_watcher_state_path()), "log_path": str(_watcher_log_path())}


def _inspect_watcher_state_payload() -> dict[str, Any]:
    state, exists, error = _read_watcher_state()
    payload = {"state_path": str(_watcher_state_path()), "state_exists": exists, "state": redact_config(state)}
    if error:
        payload["error"] = error
    return payload


def _reset_watcher_signature(args: Namespace) -> dict[str, Any]:
    if not getattr(args, "approve", False):
        raise CliUsageError("--approve is required to reset watcher signatures")
    state, exists, _ = _read_watcher_state()
    if state.get("error"):
        state = {}
    removed = sorted(key for key in _WATCHER_SIGNATURE_KEYS if key in state)
    for key in removed:
        state.pop(key, None)
    state["signature_reset_at"] = _now_iso()
    _write_watcher_state(state)
    return {"reset": True, "removed_keys": removed, "state_exists_before": exists, "state_path": str(_watcher_state_path())}


def _proposal_signature(report: dict[str, Any]) -> str:
    proposals = report.get("proposals")
    proposal_list = proposals if isinstance(proposals, list) else []
    normalized: list[dict[str, Any]] = []
    for proposal in proposal_list:
        if not isinstance(proposal, dict):
            continue
        affected = proposal.get("affected_ids")
        affected_list = affected if isinstance(affected, list) else []
        normalized.append(
            {
                "proposal_id": proposal.get("proposal_id"),
                "proposal_type": proposal.get("proposal_type"),
                "suggested_action": proposal.get("suggested_action"),
                "affected_ids": sorted(str(item) for item in affected_list),
            }
        )
    normalized.sort(key=lambda item: json.dumps(item, sort_keys=True))
    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _guarded_auto_policy_from_args(args: Namespace) -> GuardedAutoPolicy:
    return GuardedAutoPolicy(
        mode=str(getattr(args, "autonomy_mode", "report-only") or "report-only"),
        max_actions=int(getattr(args, "max_auto_actions", 10) or 10),
        quarantine_days=int(getattr(args, "quarantine_days", 30) or 30),
    )


def _record_watcher_run(args: Namespace, report: dict[str, Any]) -> None:
    if getattr(args, "qdrant_subcommand", None) != "watcher" or getattr(args, "watcher_subcommand", None) != "run":
        return
    state, _, _ = _read_watcher_state()
    if state.get("error"):
        state = {}
    signature = _proposal_signature(report)
    previous_signature = state.get("last_proposal_signature")
    signature_changed = previous_signature != signature
    force_alert = bool(getattr(args, "force_alert", False))
    alerted = bool(signature_changed or force_alert)
    proposals = report.get("proposals")
    proposal_list = proposals if isinstance(proposals, list) else []
    timestamp = _now_iso()
    guarded_auto = report.get("guarded_auto") if isinstance(report.get("guarded_auto"), dict) else {}
    guarded_auto_applied = len(guarded_auto.get("applied", []) or []) if guarded_auto else 0
    guarded_auto_errors = len(guarded_auto.get("errors", []) or []) if guarded_auto else 0
    if guarded_auto_applied or guarded_auto_errors:
        alerted = True
    state.update(
        {
            "last_run_at": timestamp,
            "last_report_id": report.get("report_id"),
            "last_scope": report.get("scope") or getattr(args, "scope", None),
            "last_proposal_count": len(proposal_list),
            "last_proposal_signature": signature,
            "last_signature_changed": signature_changed,
            "last_alerted": alerted,
            "last_force_alert": force_alert,
            "last_guarded_auto_mode": guarded_auto.get("mode") if guarded_auto else getattr(args, "autonomy_mode", "report-only"),
            "last_guarded_auto_applied_count": guarded_auto_applied,
            "last_guarded_auto_error_count": guarded_auto_errors,
            "updated_at": timestamp,
        }
    )
    if alerted:
        state["last_alert_signature"] = signature
        state["last_alert_at"] = timestamp
    _write_watcher_state(state)
    _append_watcher_log(
        {
            "event": "watcher_run",
            "timestamp": timestamp,
            "report_id": report.get("report_id"),
            "scope": report.get("scope") or getattr(args, "scope", None),
            "proposal_count": len(proposal_list),
            "signature_changed": signature_changed,
            "force_alert": force_alert,
            "alerted": alerted,
            "guarded_auto_mode": guarded_auto.get("mode") if guarded_auto else getattr(args, "autonomy_mode", "report-only"),
            "guarded_auto_applied_count": guarded_auto_applied,
            "guarded_auto_error_count": guarded_auto_errors,
        }
    )


def _watcher_status_payload(*, verbose: bool = False) -> dict[str, Any]:
    state, exists, error = _read_watcher_state()
    payload: dict[str, Any] = {
        "configured": True,
        "state_path": str(_watcher_state_path()),
        "state_exists": exists,
        "state": redact_config(state),
    }
    if error:
        payload["state_error"] = error
    if verbose:
        payload["log_path"] = str(_watcher_log_path())
        payload["schedule"] = _watcher_schedule_status()
        payload["recent_log_events"] = _read_watcher_log_tail(5).get("events", [])
    return payload


def _configured_artifact_dir(config: dict[str, Any], hermes_home: Path) -> Path:
    configured = str(config.get("consolidation_artifact_dir") or "").strip()
    return Path(configured).expanduser() if configured else hermes_home / "qdrant_memory" / "consolidation"


def _safe_report_path(config: dict[str, Any], hermes_home: Path, report_id: str) -> Path:
    safe_id = _safe_artifact_id(report_id, "report_id")
    root = _configured_artifact_dir(config, hermes_home).resolve()
    path = (root / f"report-{safe_id}.json").resolve()
    if path.parent != root:
        raise CliUsageError("invalid report_id path")
    return path


def _load_report_artifact(config: dict[str, Any], hermes_home: Path, report_id: str) -> dict[str, Any]:
    path = _safe_report_path(config, hermes_home, report_id)
    if not path.exists() or not path.is_file():
        raise CliUsageError(f"consolidation report not found: {_safe_artifact_id(report_id, 'report_id')}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise CliUsageError(f"failed to read consolidation report: {exc}") from exc
    if not isinstance(data, dict):
        raise CliUsageError("invalid consolidation report")
    return data


def _report_proposal_count(report: dict[str, Any]) -> int:
    proposals = report.get("proposals")
    if isinstance(proposals, list):
        return len(proposals)
    artifact_raw = report.get("artifact")
    artifact = artifact_raw if isinstance(artifact_raw, dict) else {}
    try:
        return int(artifact.get("proposal_count", 0))
    except Exception:
        return 0


def _report_summary_item(path: Path, report: dict[str, Any]) -> dict[str, Any]:
    return {
        "report_id": str(report.get("report_id") or path.stem.removeprefix("report-")),
        "created_at": report.get("created_at"),
        "scope": report.get("scope"),
        "proposal_count": _report_proposal_count(report),
        "summary": report.get("summary") if isinstance(report.get("summary"), dict) else {},
        "path": str(path),
    }


def _list_report_artifacts(config: dict[str, Any], hermes_home: Path, *, limit: int = 20) -> dict[str, Any]:
    root = _configured_artifact_dir(config, hermes_home)
    reports: list[dict[str, Any]] = []
    if root.exists() and root.is_dir():
        for path in root.glob("report-*.json"):
            if not path.is_file():
                continue
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    reports.append(_report_summary_item(path, data))
            except Exception:
                continue
    reports.sort(key=lambda item: str(item.get("created_at") or ""), reverse=True)
    max_reports = max(1, int(limit or 20))
    return {
        "artifact_root": str(root),
        "count": len(reports),
        "reports": reports[:max_reports],
        "truncated": len(reports) > max_reports,
    }


def _json_flag(args: Namespace) -> bool:
    return bool(getattr(args, "json", False))


def _one_line(value: Any, *, max_chars: int = 220) -> str:
    text = " ".join(str(value if value is not None else "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _plural(count: Any, singular: str, plural: str | None = None) -> str:
    try:
        return singular if int(count) == 1 else (plural or f"{singular}s")
    except Exception:
        return plural or f"{singular}s"


def _safe_scalar(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return _one_line(json.dumps(value, sort_keys=True, default=str), max_chars=220)
        except Exception:
            return _one_line(value)
    return _one_line(value)


def _ordered_collections(collections: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(collections, dict):
        return []
    order = {"memory": 0, "learning": 1}
    items: list[tuple[str, dict[str, Any]]] = []
    for scope, details in collections.items():
        if isinstance(details, dict):
            items.append((str(scope), details))
    return sorted(items, key=lambda item: (order.get(item[0], 99), item[0]))


def _collection_count_line(scope: str, details: dict[str, Any], *, indent: str = "") -> str:
    count = details.get("count")
    try:
        count_text = str(int(count))
    except Exception:
        count_text = "unknown"
    line = f"{indent}{scope}: {count_text} {_plural(count, 'point')}"
    sha = details.get("sha256")
    if sha:
        line += f" sha256: {_one_line(sha, max_chars=80)}"
    return line


def _format_export_summary(payload: dict[str, Any]) -> str:
    lines = [f"Exported {payload.get('scope')} collection"]
    if payload.get("collection_name"):
        lines.append(f"collection: {payload.get('collection_name')}")
    lines.append(f"path: {payload.get('path')}")
    lines.append(f"count: {payload.get('count')}")
    if payload.get("sha256"):
        lines.append(f"sha256: {payload.get('sha256')}")
    return "\n".join(lines)


def _format_backup_create_summary(payload: dict[str, Any]) -> str:
    lines = ["Created backup:", f"backup_id: {payload.get('backup_id')}", f"scope: {payload.get('scope')}"]
    if payload.get("backup_dir"):
        lines.append(f"path: {payload.get('backup_dir')}")
    for scope, details in _ordered_collections(payload.get("collections")):
        lines.append(_collection_count_line(scope, details))
    return "\n".join(lines)


def _format_backup_list_summary(payload: dict[str, Any]) -> str:
    backups = payload.get("backups") if isinstance(payload.get("backups"), list) else []
    lines = [f"Backups: {payload.get('count', len(backups))}", f"backup_root: {payload.get('backup_root')}"]
    for index, backup in enumerate(backups[:20], start=1):
        if not isinstance(backup, dict):
            continue
        line = f"{index}. {backup.get('backup_id')} scope={backup.get('scope')}"
        if backup.get("backup_dir"):
            line += f" path={backup.get('backup_dir')}"
        lines.append(line)
        for scope, details in _ordered_collections(backup.get("collections")):
            lines.append(_collection_count_line(scope, details, indent="   "))
    if len(backups) > 20:
        lines.append(f"... {len(backups) - 20} more backups not shown")
    return "\n".join(lines)


def _format_backup_inspect_summary(payload: dict[str, Any]) -> str:
    lines = [f"Backup: {payload.get('backup_id')}", f"scope: {payload.get('scope')}"]
    if payload.get("backup_dir"):
        lines.append(f"path: {payload.get('backup_dir')}")
    if "checksum_ok" in payload:
        lines.append(f"checksum_ok: {payload.get('checksum_ok')}")
    for scope, details in _ordered_collections(payload.get("collections")):
        lines.append(_collection_count_line(scope, details))
    return "\n".join(lines)


def _format_restore_summary(payload: dict[str, Any]) -> str:
    mode = "dry-run" if payload.get("dry_run", True) else "applied"
    lines = [f"Restore {mode}: {payload.get('backup_id')}"]
    if "validated" in payload:
        lines.append(f"validated: {payload.get('validated')}")
    if payload.get("pre_restore_backup_id"):
        lines.append(f"pre_restore_backup_id: {payload.get('pre_restore_backup_id')}")
    for scope, details in _ordered_collections(payload.get("collections")):
        parts = [
            f"total={details.get('total')}",
            f"same={details.get('same')}",
            f"changed={details.get('changed')}",
            f"missing={details.get('missing')}",
            f"would_upsert={details.get('would_upsert')}",
        ]
        if not payload.get("dry_run", True) or details.get("upserted"):
            parts.append(f"upserted={details.get('upserted')}")
        lines.append(f"{scope}: " + " ".join(parts))
    return "\n".join(lines)


def _print_payload(payload: dict[str, Any], args: Namespace, stdout, formatter: Callable[[dict[str, Any]], str]) -> None:
    if _json_flag(args):
        print(json.dumps(payload, sort_keys=True), file=stdout)
    else:
        print(formatter(payload), file=stdout)


def _local_service_error_message(exc: RuntimeError) -> str:
    message = str(exc)
    http_match = re.search(r"\bQdrant HTTP\s+([0-9]{3})\b", message)
    if http_match:
        return f"Qdrant service request failed (HTTP {http_match.group(1)})"
    if "qdrant" in message.lower():
        return "Qdrant service request failed"
    return "Local qdrant command failed"


def _local_config() -> tuple[Path, dict[str, Any]]:
    from qdrant_memory.config import load_config

    hermes_home = _hermes_home()
    return hermes_home, load_config(hermes_home=str(hermes_home))


def _redact_for_output(value: Any) -> Any:
    from qdrant_memory.consolidation import redact_secrets

    return redact_secrets(value)


def _collection_name_for_cli_scope(config: dict[str, Any], scope: str) -> str:
    if scope == "memory":
        return str(config.get("collection_name") or "hermes_memory")
    if scope == "learning":
        return str(config.get("learning_collection_name") or "hermes_learnings")
    raise CliUsageError("collection must be one of: memory, learning")


def _point_vector(point: dict[str, Any]) -> Any:
    if "vector" in point:
        return point.get("vector")
    if "vectors" in point:
        return point.get("vectors")
    return None


def _point_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    keys = ("source_type", "created_at", "updated_at", "importance", "confidence", "learning_type", "file_path", "source", "session_id", "tags")
    return {key: payload.get(key) for key in keys if payload.get(key) not in (None, "")}


def _point_snippet(payload: dict[str, Any]) -> str:
    text = payload.get("text") or payload.get("lesson") or ""
    return _one_line(_redact_for_output(text), max_chars=180) if text else ""


def _build_point_payload(raw_point: dict[str, Any] | None, *, point_id: str, collection: str, collection_name: str, include_payload: bool, include_vector: bool) -> dict[str, Any]:
    result: dict[str, Any] = {
        "found": raw_point is not None,
        "point_id": point_id,
        "collection": collection,
        "collection_name": collection_name,
        "payload_included": bool(include_payload),
        "vector_included": bool(include_vector),
    }
    if raw_point is None:
        return result
    payload_raw = raw_point.get("payload") if isinstance(raw_point.get("payload"), dict) else {}
    payload = _redact_for_output(payload_raw)
    payload = payload if isinstance(payload, dict) else {}
    result["metadata"] = _point_metadata(payload)
    if include_payload:
        snippet = _point_snippet(payload)
        if snippet:
            result["snippet"] = snippet
        result["payload"] = payload
    if include_vector:
        vector = _point_vector(raw_point)
        if vector is not None:
            result["vector"] = vector
    return result


def _execute_inspection_command(args: Namespace, stdout) -> int | None:
    subcommand = getattr(args, "qdrant_subcommand", None)
    if subcommand not in {"show", "reports", "proposals"}:
        return None

    hermes_home, config = _local_config()
    if subcommand == "show":
        from qdrant_memory.backup import qdrant_client_from_config

        point_id = _non_empty(getattr(args, "point_id", ""), "point_id")
        collection = str(getattr(args, "collection", ""))
        collection_name = _collection_name_for_cli_scope(config, collection)
        include_vector = bool(getattr(args, "include_vector", False))
        qdrant = qdrant_client_from_config(config)
        points = qdrant.retrieve(collection_name, [point_id], with_payload=True, with_vector=include_vector)
        raw_point = points[0] if points else None
        payload = _build_point_payload(
            raw_point,
            point_id=point_id,
            collection=collection,
            collection_name=collection_name,
            include_payload=bool(getattr(args, "include_payload", False)),
            include_vector=include_vector,
        )
        _print_payload(payload, args, stdout, _format_point_summary)
        return 0

    if subcommand == "reports":
        reports_subcommand = getattr(args, "reports_subcommand", None)
        if reports_subcommand == "list":
            payload = _list_report_artifacts(config, hermes_home, limit=int(getattr(args, "limit", 20) or 20))
            payload = _redact_for_output(payload)
            _print_payload(payload if isinstance(payload, dict) else {}, args, stdout, _format_reports_list_summary)
            return 0
        if reports_subcommand == "show":
            report_id = _safe_artifact_id(getattr(args, "report_id", ""), "report_id")
            report = _load_report_artifact(config, hermes_home, report_id)
            payload = _redact_for_output(report)
            _print_payload(payload if isinstance(payload, dict) else {}, args, stdout, _format_report_show_summary)
            return 0
        raise CliUsageError("unsupported qdrant reports command")

    if subcommand == "proposals":
        proposals_subcommand = getattr(args, "proposals_subcommand", None)
        if proposals_subcommand == "show":
            from qdrant_memory.consolidation import expected_action_for_proposal, find_proposal

            report_id = _safe_artifact_id(getattr(args, "report_id", ""), "report_id")
            proposal_id = _safe_artifact_id(getattr(args, "proposal_id", ""), "proposal_id")
            report = _load_report_artifact(config, hermes_home, report_id)
            try:
                proposal = find_proposal(report, proposal_id)
            except KeyError as exc:
                raise CliUsageError(f"proposal_id not found: {proposal_id}") from exc
            proposal = _redact_for_output(proposal)
            proposal = proposal if isinstance(proposal, dict) else {}
            payload = {
                "report_id": report_id,
                "proposal_id": proposal_id,
                "expected_action": expected_action_for_proposal(str(proposal.get("proposal_type") or "")),
                "proposal": proposal,
            }
            _print_payload(payload, args, stdout, _format_proposal_show_summary)
            return 0
        raise CliUsageError("unsupported qdrant proposals command")

    return None


def _execute_backup_export_restore_command(args: Namespace, stdout) -> int | None:
    subcommand = getattr(args, "qdrant_subcommand", None)
    if subcommand not in {"export", "backup", "restore"}:
        return None

    from qdrant_memory.backup import (
        create_backup,
        export_collection,
        inspect_backup,
        list_backups,
        qdrant_client_from_config,
        restore_backup,
    )

    hermes_home, config = _local_config()
    if subcommand == "export":
        qdrant = qdrant_client_from_config(config)
        payload = export_collection(qdrant, config, scope=str(getattr(args, "export_scope", "")), out_file=getattr(args, "out", ""), overwrite=bool(getattr(args, "overwrite", False)))
        _print_payload(payload, args, stdout, _format_export_summary)
        return 0

    if subcommand == "backup":
        backup_subcommand = getattr(args, "backup_subcommand", None)
        if backup_subcommand == "create":
            qdrant = qdrant_client_from_config(config)
            payload = create_backup(qdrant, config, hermes_home=hermes_home, scope=str(getattr(args, "scope", "both")))
            _print_payload(payload, args, stdout, _format_backup_create_summary)
            return 0
        if backup_subcommand == "list":
            payload = list_backups(config, hermes_home=hermes_home)
            _print_payload(payload, args, stdout, _format_backup_list_summary)
            return 0
        if backup_subcommand == "inspect":
            payload = inspect_backup(str(getattr(args, "backup_id", "")), config, hermes_home=hermes_home)
            _print_payload(payload, args, stdout, _format_backup_inspect_summary)
            return 0
        raise CliUsageError("unsupported qdrant backup command")

    if subcommand == "restore":
        _require_live_approval(args)
        qdrant = qdrant_client_from_config(config)
        payload = restore_backup(
            qdrant,
            config,
            hermes_home=hermes_home,
            backup_id=str(getattr(args, "backup_id", "")),
            dry_run=bool(getattr(args, "dry_run", True)),
            backup_first=bool(getattr(args, "backup_first", False)),
        )
        _print_payload(payload, args, stdout, _format_restore_summary)
        return 0

    return None


def _doctor_check(name: str, ok: bool, summary: str, *, critical: bool = True, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "name": name,
        "ok": bool(ok),
        "critical": bool(critical),
        "summary": summary,
        "details": details or {},
    }


def _plugin_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _read_simple_yaml(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        data[key.strip()] = value.strip().strip('"').strip("'")
    return data


def _plugin_discovery_check(plugin_root: Path | None = None) -> dict[str, Any]:
    plugin_root = plugin_root or _plugin_root()
    cli_path = plugin_root / "cli.py"
    metadata_path = plugin_root / "plugin.yaml"
    package_path = plugin_root / "qdrant_memory" / "cli_core.py"
    missing = [str(path) for path in (cli_path, metadata_path, package_path) if not path.exists()]
    metadata: dict[str, str] = {}
    if metadata_path.exists():
        try:
            metadata = _read_simple_yaml(metadata_path)
        except Exception:
            metadata = {}
    ok = not missing and metadata.get("name") == "qdrant" and metadata.get("category") == "memory"
    summary = "plugin CLI and memory-provider metadata are discoverable" if ok else "plugin CLI discovery files or metadata are invalid"
    return _doctor_check(
        "plugin_discovery",
        ok,
        summary,
        details={
            "plugin_root": str(plugin_root),
            "cli_path": str(cli_path),
            "metadata_path": str(metadata_path),
            "missing": missing,
            "metadata_name": metadata.get("name"),
            "metadata_category": metadata.get("category"),
        },
    )


def _first_match(pattern: str, text: str) -> str:
    match = re.search(pattern, text, flags=re.MULTILINE)
    return match.group(1) if match else ""


def _metadata_version_check(plugin_root: Path | None = None) -> dict[str, Any]:
    plugin_root = plugin_root or _plugin_root()
    metadata_path = plugin_root / "plugin.yaml"
    release_notes_path = plugin_root / "RELEASE_NOTES.md"
    changelog_path = plugin_root / "CHANGELOG.md"
    details: dict[str, Any] = {
        "plugin_yaml": str(metadata_path),
        "release_notes": str(release_notes_path),
        "changelog": str(changelog_path),
    }
    try:
        metadata_version = _read_simple_yaml(metadata_path).get("version", "")
        release_version = _first_match(r"Release Notes:\s*v([0-9]+\.[0-9]+\.[0-9]+)", release_notes_path.read_text(encoding="utf-8"))
        changelog_version = _first_match(r"^## \[([0-9]+\.[0-9]+\.[0-9]+)\]", changelog_path.read_text(encoding="utf-8"))
        details.update(
            {
                "plugin_version": metadata_version,
                "release_notes_version": release_version,
                "changelog_version": changelog_version,
            }
        )
        ok = bool(metadata_version) and metadata_version == release_version == changelog_version
        summary = f"metadata version {metadata_version} matches release docs" if ok else "plugin metadata version does not match release docs"
    except Exception as exc:
        ok = False
        summary = f"failed to read metadata/release docs: {exc}"
    return _doctor_check("metadata_version", ok, summary, details=details)


def _artifact_dir_from_config(config: dict[str, Any], hermes_home: Path) -> Path:
    configured = str(config.get("consolidation_artifact_dir") or "").strip()
    return Path(configured).expanduser() if configured else hermes_home / "qdrant_memory" / "consolidation"


def _watcher_artifacts_check(config: dict[str, Any]) -> dict[str, Any]:
    hermes_home = _hermes_home()
    watcher = _watcher_status_payload()
    state = watcher.get("state") if isinstance(watcher.get("state"), dict) else {}
    state_ok = not (watcher.get("state_exists") and isinstance(state, dict) and state.get("error"))
    artifact_dir = _artifact_dir_from_config(config, hermes_home)
    writable = False
    readable = False
    error = ""
    try:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        readable = os.access(artifact_dir, os.R_OK | os.X_OK)
        with tempfile.NamedTemporaryFile(prefix=".doctor-", suffix=".tmp", dir=artifact_dir, delete=True) as handle:
            handle.write(b"ok")
            handle.flush()
        writable = True
    except Exception as exc:
        error = str(exc)
    ok = bool(state_ok and readable and writable)
    summary = "watcher state is readable and artifact directory is writable" if ok else "watcher state or artifact directory is not usable"
    return _doctor_check(
        "watcher_artifacts",
        ok,
        summary,
        details={
            "state_path": watcher.get("state_path"),
            "state_exists": watcher.get("state_exists"),
            "artifact_dir": str(artifact_dir),
            "artifact_dir_readable": readable,
            "artifact_dir_writable": writable,
            "error": error,
        },
    )


def _config_redaction_check(config: dict[str, Any]) -> dict[str, Any]:
    markers = {
        "api_key": "MARKER_DOCTOR_API",
        "nested": "MARKER_DOCTOR_NESTED",
        "user": "MARKER_DOCTOR_USER",
        "password": "MARKER_DOCTOR_PASSWORD",
        "query": "MARKER_DOCTOR_QUERY",
        "fragment": "MARKER_DOCTOR_FRAGMENT",
    }
    sample = {
        "qdrant_api_key": markers["api_key"],
        "nested": {"access_token": markers["nested"]},
        "embedding_url": f"https://{markers['user']}:{markers['password']}@example.invalid/v1?api_key={markers['query']}#access_token={markers['fragment']}",
    }
    sample_redacted = redact_config(sample)
    sample_text = json.dumps(sample_redacted, sort_keys=True)
    redacted_config = redact_config(dict(config))
    leaked = [marker for marker in markers.values() if marker in sample_text]
    ok = not leaked and "@example.invalid" in sample_text and "<redacted>" in sample_text
    summary = "API-key fields and credentialed URLs are redacted" if ok else "config redaction self-test leaked sensitive markers"
    return _doctor_check(
        "config_redaction",
        ok,
        summary,
        details={
            "redacted_config_keys": sorted(redacted_config),
            "redacted_config_preview": {
                "qdrant_url": redacted_config.get("qdrant_url"),
                "embedding_url": redacted_config.get("embedding_url"),
                "qdrant_api_key": redacted_config.get("qdrant_api_key"),
                "embedding_api_key": redacted_config.get("embedding_api_key"),
            },
            "leaked_markers": leaked,
            "self_test_passed": ok,
        },
    )


def _provider_status(provider_factory: Callable[[], Any]) -> tuple[dict[str, Any], str]:
    provider = provider_factory()
    raw = provider.handle_tool_call("qdrant_memory_status", {})
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError("qdrant_memory_status did not return a JSON object")
    return parsed, raw


def _status_bool(status: dict[str, Any], key: str) -> bool:
    return bool(status.get(key))


def _status_check(name: str, ok: bool, summary_ok: str, summary_fail: str, *, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return _doctor_check(name, ok, summary_ok if ok else summary_fail, details=details)


def _collection_vector_size_check(status: dict[str, Any]) -> dict[str, Any]:
    configured = status.get("vector_size")
    memory_size = status.get("collection_vector_size")
    learning_size = status.get("learning_collection_vector_size")
    memory_ok = memory_size == configured
    learning_ok = learning_size == configured
    ok = bool(memory_ok and learning_ok)
    if memory_size is None or learning_size is None:
        ok = False
    return _doctor_check(
        "collection_vector_size",
        ok,
        "collection vector sizes match configured embedding size" if ok else "collection vector sizes do not match configured embedding size",
        details={
            "configured_vector_size": configured,
            "collection_vector_size": memory_size,
            "learning_collection_vector_size": learning_size,
            "collection_name": status.get("collection_name"),
            "learning_collection_name": status.get("learning_collection_name"),
        },
    )


def build_doctor_report(provider_factory: Callable[[], Any]) -> dict[str, Any]:
    from qdrant_memory.config import load_config

    config = load_config(hermes_home=str(_hermes_home()))
    checks: list[dict[str, Any]] = [
        _plugin_discovery_check(),
        _metadata_version_check(),
        _watcher_artifacts_check(config),
        _config_redaction_check(config),
    ]
    status: dict[str, Any] = {}
    try:
        status, _raw_status = _provider_status(provider_factory)
        checks.extend(
            [
                _status_check(
                    "active_provider",
                    status.get("provider") == "qdrant" and _status_bool(status, "active"),
                    "active qdrant memory provider is initialized",
                    "qdrant memory provider is not active",
                    details={"provider": status.get("provider"), "active": status.get("active")},
                ),
                _status_check(
                    "qdrant_reachable",
                    _status_bool(status, "qdrant_ok"),
                    "Qdrant endpoint is reachable",
                    "Qdrant endpoint is not reachable",
                    details={"qdrant_url": redact_config({"qdrant_url": status.get("qdrant_url", "")}).get("qdrant_url")},
                ),
                _status_check(
                    "embedding_reachable",
                    _status_bool(status, "embedding_ok"),
                    "embedding endpoint is reachable",
                    "embedding endpoint is not reachable",
                    details={
                        "embedding_url": redact_config({"embedding_url": status.get("embedding_url", "")}).get("embedding_url"),
                        "embedding_model": status.get("embedding_model"),
                    },
                ),
                _collection_vector_size_check(status),
                _status_check(
                    "memory_collection",
                    _status_bool(status, "collection_exists"),
                    "memory collection exists",
                    "memory collection does not exist",
                    details={"collection_name": status.get("collection_name")},
                ),
                _status_check(
                    "learning_collection",
                    _status_bool(status, "learning_collection_exists"),
                    "learning collection exists",
                    "learning collection does not exist",
                    details={"learning_collection_name": status.get("learning_collection_name")},
                ),
            ]
        )
    except Exception as exc:
        checks.append(
            _doctor_check(
                "active_provider",
                False,
                "failed to construct or query qdrant memory provider",
                details={"provider_error": type(exc).__name__},
            )
        )
        checks.extend(
            [
                _doctor_check("qdrant_reachable", False, "provider status unavailable"),
                _doctor_check("embedding_reachable", False, "provider status unavailable"),
                _doctor_check("collection_vector_size", False, "provider status unavailable"),
                _doctor_check("memory_collection", False, "provider status unavailable"),
                _doctor_check("learning_collection", False, "provider status unavailable"),
            ]
        )
    failed_critical = [check for check in checks if check.get("critical") and not check.get("ok")]
    ok = not failed_critical
    return {
        "ok": ok,
        "summary": {
            "total_checks": len(checks),
            "passed": len([check for check in checks if check.get("ok")]),
            "failed_critical": len(failed_critical),
        },
        "checks": checks,
    }


def _format_config_summary(payload: dict[str, Any]) -> str:
    lines = ["Qdrant memory configuration"]
    for key in sorted(payload):
        lines.append(f"{key}: {_safe_scalar(payload.get(key))}")
    return "\n".join(lines)


def _format_watcher_status_summary(payload: dict[str, Any]) -> str:
    lines = ["Watcher state"]
    lines.append(f"configured: {payload.get('configured')}")
    lines.append(f"state_exists: {payload.get('state_exists')}")
    lines.append(f"state_path: {payload.get('state_path')}")
    if payload.get("log_path"):
        lines.append(f"log_path: {payload.get('log_path')}")
    schedule = payload.get("schedule") if isinstance(payload.get("schedule"), dict) else {}
    if schedule:
        lines.append(f"installed: {schedule.get('installed')}")
        if schedule.get("error"):
            lines.append(f"schedule_error: {_one_line(schedule.get('error'))}")
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    if state.get("last_report_id"):
        lines.append(f"last_report_id: {state.get('last_report_id')}")
    if state.get("last_proposal_signature"):
        lines.append(f"last_proposal_signature: {state.get('last_proposal_signature')}")
    if state.get("error"):
        lines.append(f"state_error: {_one_line(state.get('error'))}")
    events = payload.get("recent_log_events") if isinstance(payload.get("recent_log_events"), list) else []
    if events:
        lines.append(f"recent_log_events: {len(events)}")
    return "\n".join(lines)


def _format_watcher_logs_summary(payload: dict[str, Any]) -> str:
    lines = [f"Watcher log events: {payload.get('count', 0)}", f"log_path: {payload.get('log_path')}"]
    if not payload.get("log_exists"):
        lines.append("log_exists: False")
    events = payload.get("events") if isinstance(payload.get("events"), list) else []
    for index, event in enumerate(events, start=1):
        if isinstance(event, dict):
            summary = event.get("event") or event.get("report_id") or event
            lines.append(f"{index}. {_safe_scalar(summary)}")
    if payload.get("error"):
        lines.append(f"error: {_one_line(payload.get('error'))}")
    return "\n".join(lines)


def _format_watcher_inspect_summary(payload: dict[str, Any]) -> str:
    lines = ["Watcher state inspect", f"state_exists: {payload.get('state_exists')}", f"state_path: {payload.get('state_path')}"]
    state = payload.get("state") if isinstance(payload.get("state"), dict) else {}
    for key in sorted(state):
        lines.append(f"{key}: {_safe_scalar(state.get(key))}")
    if payload.get("error"):
        lines.append(f"error: {_one_line(payload.get('error'))}")
    return "\n".join(lines)


def _format_watcher_install_summary(payload: dict[str, Any]) -> str:
    action = "installed" if payload.get("installed") else "uninstalled"
    lines = [f"Watcher {action}: changed={payload.get('changed')}"]
    if payload.get("schedule"):
        lines.append(f"schedule: {payload.get('schedule')}")
    if payload.get("state_path"):
        lines.append(f"state_path: {payload.get('state_path')}")
    if payload.get("log_path"):
        lines.append(f"log_path: {payload.get('log_path')}")
    return "\n".join(lines)


def _format_watcher_reset_summary(payload: dict[str, Any]) -> str:
    removed = payload.get("removed_keys") if isinstance(payload.get("removed_keys"), list) else []
    lines = [f"Watcher signature reset: {payload.get('reset')}", f"removed_keys: {', '.join(str(key) for key in removed)}"]
    if payload.get("state_path"):
        lines.append(f"state_path: {payload.get('state_path')}")
    return "\n".join(lines)


def _format_point_summary(payload: dict[str, Any]) -> str:
    found = bool(payload.get("found"))
    point_id = payload.get("point_id") or "<unknown>"
    lines = [f"Point: {point_id}" if found else f"Point not found: {point_id}"]
    lines.append(f"collection: {payload.get('collection')} ({payload.get('collection_name')})")
    lines.append(f"payload_included: {payload.get('payload_included')}")
    lines.append(f"vector_included: {payload.get('vector_included')}")
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    for key in ("source_type", "created_at", "updated_at", "importance", "confidence", "learning_type", "file_path", "session_id"):
        if metadata.get(key) not in (None, ""):
            lines.append(f"{key}: {_safe_scalar(metadata.get(key))}")
    if payload.get("snippet"):
        lines.append(f"snippet: {_one_line(payload.get('snippet'), max_chars=180)}")
    vector = payload.get("vector")
    if isinstance(vector, list):
        lines.append(f"vector_size: {len(vector)}")
    elif isinstance(vector, dict):
        lines.append(f"vector_names: {', '.join(sorted(str(key) for key in vector))}")
    payload_data = payload.get("payload") if isinstance(payload.get("payload"), dict) else {}
    if payload_data:
        lines.append("payload:")
        for key in sorted(payload_data)[:20]:
            lines.append(f"  {key}: {_safe_scalar(payload_data.get(key))}")
        if len(payload_data) > 20:
            lines.append(f"  ... {len(payload_data) - 20} more payload fields not shown")
    return "\n".join(lines)


def _format_reports_list_summary(payload: dict[str, Any]) -> str:
    reports = payload.get("reports") if isinstance(payload.get("reports"), list) else []
    lines = [f"Reports: {payload.get('count', len(reports))}", f"artifact_root: {payload.get('artifact_root')}"]
    for index, report in enumerate(reports, start=1):
        if not isinstance(report, dict):
            continue
        line = f"{index}. {report.get('report_id')} proposals={report.get('proposal_count')} scope={report.get('scope')}"
        if report.get("created_at"):
            line += f" created_at={report.get('created_at')}"
        lines.append(line)
        summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
        for key in sorted(summary):
            lines.append(f"   {key}: {summary.get(key)}")
    if payload.get("truncated"):
        lines.append("... additional reports not shown")
    return "\n".join(lines)


def _format_report_show_summary(payload: dict[str, Any]) -> str:
    lines = [f"Report: {payload.get('report_id')}"]
    if payload.get("created_at"):
        lines.append(f"created_at: {payload.get('created_at')}")
    if payload.get("scope"):
        lines.append(f"scope: {payload.get('scope')}")
    lines.append(f"proposal_count: {_proposal_count(payload)}")
    artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
    if artifact.get("path"):
        lines.append(f"artifact: {artifact.get('path')}")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    for key in sorted(summary):
        lines.append(f"{key}: {summary.get(key)}")
    proposals = payload.get("proposals") if isinstance(payload.get("proposals"), list) else []
    for index, proposal in enumerate(proposals[:20], start=1):
        if not isinstance(proposal, dict):
            continue
        lines.append(
            f"{index}. {proposal.get('proposal_id')} [{proposal.get('proposal_type')}] action={proposal.get('suggested_action')} affected={len(proposal.get('affected_ids') or [])}"
        )
    if len(proposals) > 20:
        lines.append(f"... {len(proposals) - 20} more proposals not shown")
    return "\n".join(lines)


def _format_proposal_show_summary(payload: dict[str, Any]) -> str:
    proposal = payload.get("proposal") if isinstance(payload.get("proposal"), dict) else {}
    affected = proposal.get("affected_ids") if isinstance(proposal.get("affected_ids"), list) else []
    lines = [f"Proposal: {payload.get('proposal_id')}", f"report_id: {payload.get('report_id')}"]
    lines.append(f"type: {proposal.get('proposal_type')}")
    lines.append(f"expected_action: {payload.get('expected_action')}")
    lines.append(f"suggested_action: {proposal.get('suggested_action')}")
    lines.append(f"risk: {proposal.get('risk')}")
    lines.append(f"confidence: {_format_score(proposal.get('confidence'))}")
    lines.append(f"affected_ids: {', '.join(str(item) for item in affected)}")
    evidence = proposal.get("evidence") if isinstance(proposal.get("evidence"), list) else []
    for item in evidence[:10]:
        if isinstance(item, dict):
            lines.append(f"evidence: {_safe_scalar(item)}")
    if len(evidence) > 10:
        lines.append(f"... {len(evidence) - 10} more evidence items not shown")
    return "\n".join(lines)


def _format_doctor_summary(report: dict[str, Any]) -> str:
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    total = summary.get("total_checks", 0)
    passed = summary.get("passed", 0)
    failed = summary.get("failed_critical", 0)
    if failed:
        lines = [f"Diagnostics: {passed}/{total} checks passed, {failed} critical failed"]
    else:
        lines = [f"Diagnostics: {passed}/{total} checks passed"]
    checks = report.get("checks") if isinstance(report.get("checks"), list) else []
    for check in checks:
        if not isinstance(check, dict):
            continue
        ok = bool(check.get("ok"))
        critical = bool(check.get("critical", True))
        marker = "OK" if ok else ("FAIL" if critical else "WARN")
        lines.append(f"[{marker}] {check.get('name')}: {_one_line(check.get('summary'))}")
    return "\n".join(lines)


def _format_status_summary(payload: dict[str, Any]) -> str:
    active = "active" if payload.get("active") else "inactive"
    lines = [f"Qdrant memory provider: {active}"]
    qdrant_marker = "OK" if payload.get("qdrant_ok") else "WARN"
    qdrant_text = "reachable" if payload.get("qdrant_ok") else "unavailable"
    lines.append(f"[{qdrant_marker}] qdrant: {qdrant_text}")
    embedding_marker = "OK" if payload.get("embedding_ok") else "WARN"
    embedding_text = "reachable" if payload.get("embedding_ok") else "unavailable"
    lines.append(f"[{embedding_marker}] embeddings: {embedding_text}")
    if payload.get("collection_name"):
        count = payload.get("point_count")
        count_text = "unknown" if count is None else str(count)
        lines.append(f"memory: {count_text} (collection: {payload.get('collection_name')})")
    if payload.get("learning_collection_name"):
        count = payload.get("learning_point_count")
        count_text = "unknown" if count is None else str(count)
        lines.append(f"learnings: {count_text} (collection: {payload.get('learning_collection_name')})")
    if "pending_learning_candidate_count" in payload:
        lines.append(f"pending_learning_candidates: {payload.get('pending_learning_candidate_count')}")
    return "\n".join(lines)


def _format_score(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return _one_line(value, max_chars=40)


def _format_result_list(payload: dict[str, Any], *, label: str, plural_label: str | None = None) -> str:
    results = payload.get("results") if isinstance(payload.get("results"), list) else []
    count = payload.get("count", len(results))
    noun = label if count == 1 else (plural_label or f"{label}s")
    lines = [f"Found {count} {noun}"]
    for index, item in enumerate(results[:10], start=1):
        if not isinstance(item, dict):
            continue
        item_id = item.get("id") or "<unknown>"
        score = _format_score(item.get("score")) if "score" in item else ""
        text = _one_line(item.get("text") or item.get("lesson") or "", max_chars=180)
        prefix = f"{index}. {item_id}"
        if score:
            prefix += f" score={score}"
        if item.get("learning_type"):
            prefix += f" [{item.get('learning_type')}]"
        lines.append(f"{prefix} {text}".rstrip())
    if len(results) > 10:
        lines.append(f"... {len(results) - 10} more results not shown")
    return "\n".join(lines)


def _format_store_summary(payload: dict[str, Any], *, learning: bool = False) -> str:
    label = "learning" if learning else "memory"
    item_id = payload.get("id") or "<unknown>"
    if payload.get("saved"):
        lines = [f"Saved {label}: {item_id}"]
    elif payload.get("dry_run", False):
        lines = [f"{label.capitalize()} store dry-run: {item_id}"]
        if payload.get("would_store") is not None:
            lines.append(f"would_store: {str(bool(payload.get('would_store'))).lower()}")
    elif payload.get("duplicate_found"):
        lines = [f"{label.capitalize()} not saved: duplicate found"]
    else:
        lines = [f"{label.capitalize()} not saved"]
    if payload.get("source_type"):
        lines.append(f"source_type: {payload.get('source_type')}")
    if payload.get("collection_name"):
        lines.append(f"collection: {payload.get('collection_name')}")
    if payload.get("duplicate_preview") is not None:
        lines.append(f"duplicate_preview: {str(bool(payload.get('duplicate_preview'))).lower()}")
    if payload.get("duplicate_found") is not None:
        lines.append(f"duplicate_found: {str(bool(payload.get('duplicate_found'))).lower()}")
    duplicate = payload.get("duplicate") if isinstance(payload.get("duplicate"), dict) else None
    if duplicate:
        if duplicate.get("id"):
            lines.append(f"duplicate_id: {duplicate.get('id')}")
        try:
            lines.append(f"duplicate_score: {float(duplicate.get('score', 0.0)):.3f}")
        except Exception:
            pass
    return "\n".join(lines)


def _format_index_summary(payload: dict[str, Any]) -> str:
    mode = "dry-run" if payload.get("dry_run", True) else "applied"
    lines = [f"Index {mode}: files={payload.get('files_indexed', 0)}/{payload.get('files_seen', 0)} chunks_prepared={payload.get('chunks_prepared', 0)}"]
    if payload.get("chunks_upserted"):
        lines.append(f"chunks_upserted: {payload.get('chunks_upserted')}")
    if payload.get("stale_count"):
        lines.append(f"stale_chunks: {payload.get('stale_count')} delete_mode={payload.get('delete_mode')}")
    if payload.get("errors"):
        lines.append(f"errors: {len(payload.get('errors') or [])}")
    return "\n".join(lines)


def _format_forget_summary(payload: dict[str, Any]) -> str:
    ids = payload.get("ids") if isinstance(payload.get("ids"), list) else []
    if payload.get("dry_run", True):
        return f"Forget dry-run: {len(ids)} explicit {_plural(len(ids), 'id')}"
    return f"Forgot memories: deleted={payload.get('deleted', 0)}"


def _format_learning_preview_summary(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") if isinstance(payload.get("candidates"), list) else []
    count = payload.get("count", len(candidates))
    lines = [f"Pending learning candidates: {count}"]
    for index, item in enumerate(candidates[:10], start=1):
        if not isinstance(item, dict):
            continue
        candidate_id = item.get("candidate_id") or item.get("id") or "<unknown>"
        learning_type = item.get("learning_type") or "learning"
        lesson = _one_line(item.get("lesson") or item.get("text") or "", max_chars=180)
        lines.append(f"{index}. {candidate_id} [{learning_type}] {lesson}".rstrip())
    if len(candidates) > 10:
        lines.append(f"... {len(candidates) - 10} more candidates not shown")
    return "\n".join(lines)


def _proposal_count(payload: dict[str, Any]) -> int:
    proposals = payload.get("proposals")
    if isinstance(proposals, list):
        return len(proposals)
    if isinstance(payload.get("artifact"), dict) and payload["artifact"].get("proposal_count") is not None:
        try:
            return int(payload["artifact"].get("proposal_count"))
        except Exception:
            return 0
    if payload.get("proposal_count") is not None:
        try:
            return int(payload.get("proposal_count"))
        except Exception:
            return 0
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    total = 0
    for value in summary.values():
        try:
            total += int(value)
        except Exception:
            pass
    return total


def _format_consolidate_summary(payload: dict[str, Any]) -> str:
    count = _proposal_count(payload)
    guarded_auto = payload.get("guarded_auto") if isinstance(payload.get("guarded_auto"), dict) else {}
    guarded_auto_applied = len(guarded_auto.get("applied", []) or []) if guarded_auto else 0
    guarded_auto_errors = len(guarded_auto.get("errors", []) or []) if guarded_auto else 0
    if guarded_auto:
        mode = guarded_auto.get("mode", "report-only")
        lines = [
            f"Consolidation watcher ({mode}): {count} {_plural(count, 'proposal')}, "
            f"guarded-auto applied={guarded_auto_applied}, errors={guarded_auto_errors}"
        ]
    else:
        lines = [f"Report-only consolidation: {count} {_plural(count, 'proposal')}"]
    if payload.get("report_id"):
        lines.append(f"report_id: {payload.get('report_id')}")
    if payload.get("scope"):
        lines.append(f"scope: {payload.get('scope')}")
    if payload.get("persisted") is not None:
        lines.append(f"persisted: {payload.get('persisted')}")
    artifact = payload.get("artifact") if isinstance(payload.get("artifact"), dict) else {}
    if artifact.get("path"):
        lines.append(f"artifact: {artifact.get('path')}")
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    for key in sorted(summary):
        lines.append(f"{key}: {summary.get(key)}")
    return "\n".join(lines)


def _format_apply_summary(payload: dict[str, Any]) -> str:
    affected = payload.get("affected_ids") if isinstance(payload.get("affected_ids"), list) else []
    if not affected and isinstance(payload.get("delete_ids"), list):
        affected = payload.get("delete_ids")
    mode = "dry-run" if payload.get("dry_run", True) else "applied"
    lines = [f"Apply {mode}: action={payload.get('action')} proposal={payload.get('proposal_id')} affected={len(affected)}"]
    if payload.get("report_id"):
        lines.append(f"report_id: {payload.get('report_id')}")
    if payload.get("application_id"):
        lines.append(f"application_id: {payload.get('application_id')}")
    if payload.get("pre_apply_backup_id"):
        lines.append(f"pre_apply_backup_id: {payload.get('pre_apply_backup_id')}")
    return "\n".join(lines)


def _format_learning_approve_summary(payload: dict[str, Any]) -> str:
    if payload.get("dry_run", True):
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        return f"Learning approval dry-run: candidate={candidate.get('candidate_id') or candidate.get('id') or '<unknown>'}"
    return _format_store_summary(payload, learning=True)


def _format_generic_summary(payload: Any, args: Namespace, tool_name: str = "") -> str:
    subcommand = getattr(args, "qdrant_subcommand", None) or tool_name or "command"
    if isinstance(payload, dict):
        keys = ", ".join(sorted(str(key) for key in payload.keys())[:12])
        return f"{subcommand}: completed" + (f" ({keys})" if keys else "")
    return f"{subcommand}: completed"


def _format_human_payload(payload: Any, args: Namespace, tool_name: str) -> str:
    if not isinstance(payload, dict):
        return _format_generic_summary(payload, args, tool_name)
    if tool_name == "qdrant_memory_status":
        return _format_status_summary(payload)
    if tool_name == "qdrant_memory_search":
        return _format_result_list(payload, label="memory", plural_label="memories")
    if tool_name == "qdrant_learning_search":
        return _format_result_list(payload, label="learning", plural_label="learnings")
    if tool_name == "qdrant_memory_store":
        return _format_store_summary(payload)
    if tool_name == "qdrant_learning_store":
        return _format_store_summary(payload, learning=True)
    if tool_name == "qdrant_memory_index":
        return _format_index_summary(payload)
    if tool_name == "qdrant_memory_forget":
        return _format_forget_summary(payload)
    if tool_name == "qdrant_learning_preview":
        return _format_learning_preview_summary(payload)
    if tool_name == "qdrant_learning_approve":
        return _format_learning_approve_summary(payload)
    if tool_name == "qdrant_memory_consolidate":
        return _format_consolidate_summary(payload)
    if tool_name == "qdrant_memory_consolidation_apply":
        return _format_apply_summary(payload)
    return _format_generic_summary(payload, args, tool_name)


def _provider_error_message(payload: Any) -> str:
    if isinstance(payload, dict):
        message = payload.get("message")
        if not message:
            error = payload.get("error")
            message = error if error is not True else "Provider returned an error"
        return _one_line(message or "Provider returned an error")
    return "Provider returned an error"


def _print_cli_error(message: str, args: Namespace, stderr) -> None:
    if _json_flag(args):
        print(json.dumps({"error": True, "message": message}, sort_keys=True), file=stderr)
    else:
        print(f"Error: {message}", file=stderr)


def _execute_local_command(args: Namespace, stdout) -> int | None:
    inspection_exit = _execute_inspection_command(args, stdout)
    if inspection_exit is not None:
        return inspection_exit
    backup_exit = _execute_backup_export_restore_command(args, stdout)
    if backup_exit is not None:
        return backup_exit
    subcommand = getattr(args, "qdrant_subcommand", None)
    if subcommand == "config" and getattr(args, "config_subcommand", None) == "show":
        _print_payload(_redacted_config(), args, stdout, _format_config_summary)
        return 0
    if subcommand == "watcher":
        watcher_subcommand = getattr(args, "watcher_subcommand", None)
        if watcher_subcommand == "status":
            _print_payload(_watcher_status_payload(verbose=getattr(args, "verbose", False)), args, stdout, _format_watcher_status_summary)
            return 0
        if watcher_subcommand == "logs":
            _print_payload(_read_watcher_log_tail(getattr(args, "tail", 20)), args, stdout, _format_watcher_logs_summary)
            return 0
        if watcher_subcommand == "inspect-state":
            _print_payload(_inspect_watcher_state_payload(), args, stdout, _format_watcher_inspect_summary)
            return 0
        if watcher_subcommand == "install":
            _print_payload(_install_watcher(args), args, stdout, _format_watcher_install_summary)
            return 0
        if watcher_subcommand == "uninstall":
            _print_payload(_uninstall_watcher(args), args, stdout, _format_watcher_install_summary)
            return 0
        if watcher_subcommand == "reset-signature":
            _print_payload(_reset_watcher_signature(args), args, stdout, _format_watcher_reset_summary)
            return 0
    return None


def build_tool_call(args: Namespace) -> tuple[str, dict[str, Any]]:
    """Convert parsed qdrant CLI args into the existing Hermes tool surface."""

    subcommand = getattr(args, "qdrant_subcommand", None)

    if subcommand == "status":
        return "qdrant_memory_status", {}
    if subcommand == "doctor":
        raise CliUsageError("doctor is handled by execute_command diagnostics")

    if subcommand == "config":
        raise CliUsageError("unsupported qdrant config command")

    if subcommand == "store":
        _require_live_approval(args)
        text = _non_empty(args.text, "text")
        tool_args = {
            "text": text,
            "source_type": args.source_type or "manual",
            "importance": args.importance,
            "tags": _split_tags(getattr(args, "tag", [])),
            "dry_run": args.dry_run,
            "duplicate_preview": bool(getattr(args, "preview_duplicates", False)),
        }
        if getattr(args, "approve", False):
            tool_args["approve"] = True
        return "qdrant_memory_store", tool_args

    if subcommand == "search":
        tool_args = {
            "query": args.query,
            "top_k": args.top_k,
            "source_type": args.source_type,
            "include_metadata": args.include_metadata,
        }
        tool_args.update(_search_filter_tool_args(args, include_collection=True))
        return "qdrant_memory_search", tool_args

    if subcommand == "index":
        _require_live_approval(args)
        tool_args: dict[str, Any] = {
            "paths": args.paths,
            "dry_run": args.dry_run,
            "force": args.force,
        }
        max_files = _optional_int(args.max_files)
        if max_files is not None:
            tool_args["max_files"] = max_files
        return "qdrant_memory_index", tool_args

    if subcommand == "forget":
        _require_live_approval(args)
        if not args.ids:
            raise CliUsageError("at least one point id is required")
        return "qdrant_memory_forget", {"ids": args.ids, "dry_run": args.dry_run}

    if subcommand == "consolidate":
        return "qdrant_memory_consolidate", {
            "dry_run": True,
            "scope": args.scope,
            "persist": args.persist,
            "include_reconsolidation": args.include_reconsolidation,
        }

    if subcommand == "apply":
        _require_live_approval(args)
        tool_args = {
            "report_id": args.report_id,
            "proposal_id": args.proposal_id,
            "action": args.action,
            "dry_run": args.dry_run,
            "approve": args.approve,
        }
        if getattr(args, "backup_first", False):
            tool_args["backup_first"] = True
        if getattr(args, "action", None) == "quarantine" and getattr(args, "quarantine_days", None):
            tool_args["quarantine_days"] = args.quarantine_days
        return "qdrant_memory_consolidation_apply", tool_args

    if subcommand == "learning":
        learning_subcommand = getattr(args, "learning_subcommand", None)
        if learning_subcommand == "search":
            tool_args = {
                "query": args.query,
                "top_k": args.top_k,
                "learning_type": args.learning_type,
                "include_metadata": args.include_metadata,
            }
            tool_args.update(_search_filter_tool_args(args))
            return "qdrant_learning_search", tool_args
        if learning_subcommand == "preview":
            return "qdrant_learning_preview", {"include_metadata": args.include_metadata}
        if learning_subcommand == "store":
            lesson = _non_empty(args.lesson, "lesson")
            return "qdrant_learning_store", {
                "lesson": lesson,
                "learning_type": args.learning_type,
                "trigger": args.trigger,
                "mistake": args.mistake,
                "correction": args.correction,
                "evidence": args.evidence,
                "tool_name": args.tool_name,
                "command": args.command,
                "importance": args.importance,
                "confidence": args.confidence,
                "tags": _split_tags(getattr(args, "tag", [])),
                "promote_to_skill_candidate": args.promote_to_skill_candidate,
            }
        if learning_subcommand == "approve":
            _require_live_approval(args)
            candidate_id = _non_empty(args.candidate_id, "candidate_id")
            return "qdrant_learning_approve", {"candidate_id": candidate_id, "dry_run": args.dry_run}

    if subcommand == "watcher":
        watcher_subcommand = getattr(args, "watcher_subcommand", None)
        if watcher_subcommand == "run":
            return "qdrant_memory_consolidate", {
                "dry_run": True,
                "scope": args.scope,
                "persist": True,
                "include_reconsolidation": args.include_reconsolidation,
                "include_examples": False,
                "max_points": args.max_points,
                "max_groups": args.max_groups,
                "reconsolidation_max_candidates": args.reconsolidation_max_candidates,
            }
        raise CliUsageError("unsupported qdrant watcher command")

    raise CliUsageError(f"unsupported qdrant command: {subcommand or '<missing>'}")


def _load_provider_class():
    provider_module_path = Path(__file__).resolve().parents[1] / "__init__.py"
    module_name = "_hermes_qdrant_memory_provider_cli"
    existing = sys.modules.get(module_name)
    if existing is not None and hasattr(existing, "QdrantMemoryProvider"):
        return existing.QdrantMemoryProvider

    spec = importlib.util.spec_from_file_location(module_name, provider_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load Qdrant provider from {provider_module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.QdrantMemoryProvider


def default_provider_factory():
    """Lazily construct and initialize the provider only when a command runs."""

    provider = _load_provider_class()()
    provider.initialize("cli", platform="cli", agent_context="cli")
    return provider


def execute_command(
    args: Namespace,
    *,
    provider_factory: Callable[[], Any] | None = None,
    stdout=None,
    stderr=None,
) -> int:
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    provider_factory = provider_factory or default_provider_factory

    try:
        local_exit = _execute_local_command(args, stdout)
    except CliUsageError as exc:
        _print_cli_error(str(exc), args, stderr)
        return 2
    except RuntimeError as exc:
        if getattr(args, "qdrant_subcommand", None) in {"show", "export", "backup", "restore"}:
            _print_cli_error(_local_service_error_message(exc), args, stderr)
            return 1
        raise
    except ValueError as exc:
        if exc.__class__.__name__ == "BackupError":
            _print_cli_error(str(exc), args, stderr)
            return 2
        raise
    if local_exit is not None:
        return local_exit

    if getattr(args, "qdrant_subcommand", None) == "doctor":
        report = build_doctor_report(provider_factory)
        if _json_flag(args):
            print(json.dumps(report, sort_keys=True), file=stdout)
        else:
            print(_format_doctor_summary(report), file=stdout)
        return 0 if report.get("ok") else 1

    try:
        tool_name, tool_args = build_tool_call(args)
    except CliUsageError as exc:
        _print_cli_error(str(exc), args, stderr)
        return 2

    provider = provider_factory()
    result = provider.handle_tool_call(tool_name, tool_args)
    try:
        parsed = json.loads(result)
    except Exception:
        if _json_flag(args):
            _print_cli_error("Provider returned invalid JSON; expected a JSON object", args, stderr)
            return 1
        print(_format_generic_summary(None, args, tool_name), file=stdout)
        return 0
    if not isinstance(parsed, dict):
        if _json_flag(args):
            _print_cli_error("Provider returned non-object JSON; expected a JSON object", args, stderr)
            return 1
        print(_format_generic_summary(parsed, args, tool_name), file=stdout)
        return 0
    if parsed.get("error"):
        if _json_flag(args):
            print(result, file=stderr)
        else:
            print(f"Error: {_provider_error_message(parsed)}", file=stderr)
        return 1
    if getattr(args, "qdrant_subcommand", None) == "watcher" and getattr(args, "watcher_subcommand", None) == "run":
        guarded_auto = apply_guarded_auto(provider, parsed, _guarded_auto_policy_from_args(args))
        parsed["guarded_auto"] = guarded_auto
        result = json.dumps(parsed, sort_keys=True)
    _record_watcher_run(args, parsed)
    if _json_flag(args):
        print(result, file=stdout)
    else:
        print(_format_human_payload(parsed, args, tool_name), file=stdout)
    return 0
