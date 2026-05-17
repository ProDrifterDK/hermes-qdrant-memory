from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import Namespace
from collections.abc import Callable
from pathlib import Path
from typing import Any


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


def _redacted_config() -> dict[str, Any]:
    from qdrant_memory.config import load_config

    config = load_config(hermes_home=str(_hermes_home()))
    sanitized = dict(config)
    for key in ("qdrant_api_key", "embedding_api_key"):
        sanitized[key] = "<redacted>" if sanitized.get(key) else ""
    return sanitized


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

    if subcommand in {"status", "doctor"}:
        return "qdrant_memory_status", {}

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
