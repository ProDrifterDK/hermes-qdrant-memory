from __future__ import annotations

import importlib.util
import json
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


def build_tool_call(args: Namespace) -> tuple[str, dict[str, Any]]:
    """Convert parsed qdrant CLI args into the existing Hermes tool surface."""

    subcommand = getattr(args, "qdrant_subcommand", None)

    if subcommand in {"status", "doctor"}:
        return "qdrant_memory_status", {}

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
