from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
import tempfile
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


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


def _non_empty(value: Any, name: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise CliUsageError(f"{name} is required")
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


def _watcher_status_payload() -> dict[str, Any]:
    state_path = _hermes_home() / "qdrant_memory" / "consolidation" / "watcher_state.json"
    state: dict[str, Any] = {}
    exists = state_path.exists()
    if exists:
        try:
            parsed = json.loads(state_path.read_text(encoding="utf-8"))
            state = parsed if isinstance(parsed, dict) else {"raw": parsed}
        except Exception as exc:
            state = {"error": f"failed to read watcher state: {exc}"}
    return {
        "configured": True,
        "state_path": str(state_path),
        "state_exists": exists,
        "state": state,
    }


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


def _execute_local_command(args: Namespace, stdout) -> int | None:
    subcommand = getattr(args, "qdrant_subcommand", None)
    if subcommand == "config" and getattr(args, "config_subcommand", None) == "show":
        print(json.dumps(_redacted_config(), sort_keys=True), file=stdout)
        return 0
    if subcommand == "watcher" and getattr(args, "watcher_subcommand", None) == "status":
        print(json.dumps(_watcher_status_payload(), sort_keys=True), file=stdout)
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
        text = _non_empty(args.text, "text")
        return "qdrant_memory_store", {
            "text": text,
            "source_type": args.source_type or "manual",
            "importance": args.importance,
            "tags": _split_tags(getattr(args, "tag", [])),
        }

    if subcommand == "search":
        return "qdrant_memory_search", {
            "query": args.query,
            "top_k": args.top_k,
            "source_type": args.source_type,
            "include_metadata": args.include_metadata,
        }

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
        return "qdrant_memory_consolidation_apply", {
            "report_id": args.report_id,
            "proposal_id": args.proposal_id,
            "action": args.action,
            "dry_run": args.dry_run,
            "approve": args.approve,
        }

    if subcommand == "learning":
        learning_subcommand = getattr(args, "learning_subcommand", None)
        if learning_subcommand == "search":
            return "qdrant_learning_search", {
                "query": args.query,
                "top_k": args.top_k,
                "learning_type": args.learning_type,
                "include_metadata": args.include_metadata,
            }
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

    local_exit = _execute_local_command(args, stdout)
    if local_exit is not None:
        return local_exit

    if getattr(args, "qdrant_subcommand", None) == "doctor":
        report = build_doctor_report(provider_factory)
        print(json.dumps(report, sort_keys=True), file=stdout)
        return 0 if report.get("ok") else 1

    try:
        tool_name, tool_args = build_tool_call(args)
    except CliUsageError as exc:
        print(json.dumps({"error": True, "message": str(exc)}), file=stderr)
        return 2

    provider = provider_factory()
    result = provider.handle_tool_call(tool_name, tool_args)
    print(result, file=stdout)
    try:
        parsed = json.loads(result)
    except Exception:
        return 0
    return 1 if isinstance(parsed, dict) and parsed.get("error") else 0
