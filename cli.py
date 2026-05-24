from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from qdrant_memory.cli_core import execute_command


def _top_k(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 20:
        raise argparse.ArgumentTypeError("must be between 1 and 20")
    return parsed


def _importance(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 10:
        raise argparse.ArgumentTypeError("must be between 1 and 10")
    return parsed


def _confidence(value: str) -> float:
    parsed = float(value)
    if parsed < 0 or parsed > 1:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _add_tag_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--tag", action="append", default=[], help="Optional tag. Repeat or pass comma-separated values.")


def _add_dry_run_flags(parser: argparse.ArgumentParser) -> None:
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Preview without live mutation. Default.")
    dry_run.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Allow live mutation when paired with --approve.")
    parser.add_argument("--approve", action="store_true", help="Required with --no-dry-run for live mutation.")


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Register the qdrant memory-provider CLI under `hermes qdrant`."""

    parser.description = "Manage the Qdrant Hermes memory provider."
    subcommands = parser.add_subparsers(dest="qdrant_subcommand", required=True)

    status = subcommands.add_parser("status", help="Show Qdrant provider status.")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    doctor = subcommands.add_parser("doctor", help="Run structured Qdrant memory diagnostics.")
    doctor.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    config = subcommands.add_parser("config", help="Inspect local qdrant_memory configuration without contacting services.")
    config_subcommands = config.add_subparsers(dest="config_subcommand", required=True)
    config_show = config_subcommands.add_parser("show", help="Show effective qdrant_memory config with secrets redacted.")
    config_show.set_defaults(qdrant_subcommand="config")
    config_show.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    store = subcommands.add_parser("store", help="Store an explicit manual memory.")
    store.add_argument("text", help="Memory text to store.")
    store.add_argument("--source-type", default="manual", help="Memory source type. Default: manual.")
    store.add_argument("--importance", type=_importance, default=5, help="Importance, 1 to 10. Default: 5.")
    _add_tag_flags(store)
    store.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    search = subcommands.add_parser("search", help="Search semantic memories.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    search.add_argument("--source-type", default=None, help="Optional source_type filter.")
    search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    index = subcommands.add_parser("index", help="Index files or directories into memory. Dry-run by default.")
    index.add_argument("paths", nargs="+", help="Files or directories to index.")
    _add_dry_run_flags(index)
    index.add_argument("--max-files", type=int, default=None, help="Maximum files to scan/index.")
    index.add_argument("--force", action="store_true", help="Re-index chunks even if unchanged.")
    index.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    forget = subcommands.add_parser("forget", help="Forget explicit point IDs. Dry-run by default.")
    forget.add_argument("ids", nargs="+", help="Explicit Qdrant point IDs to delete.")
    _add_dry_run_flags(forget)
    forget.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    export = subcommands.add_parser("export", help="Export a Qdrant collection to a private JSONL artifact.")
    export_subcommands = export.add_subparsers(dest="export_scope", required=True)
    for scope_name in ("memory", "learning"):
        export_scope = export_subcommands.add_parser(scope_name, help=f"Export the {scope_name} collection.")
        export_scope.set_defaults(qdrant_subcommand="export")
        export_scope.add_argument("--out", required=True, help="Output JSONL artifact path.")
        export_scope.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
        export_scope.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    backup = subcommands.add_parser("backup", help="Create, list, or inspect private Qdrant backup artifacts.")
    backup_subcommands = backup.add_subparsers(dest="backup_subcommand", required=True)
    backup_create = backup_subcommands.add_parser("create", help="Create a private backup artifact. No Qdrant mutation.")
    backup_create.set_defaults(qdrant_subcommand="backup")
    backup_create.add_argument("--scope", choices=["memory", "learning", "both"], default="both", help="Backup scope. Default: both.")
    backup_create.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    backup_list = backup_subcommands.add_parser("list", help="List local backup artifacts without contacting Qdrant.")
    backup_list.set_defaults(qdrant_subcommand="backup")
    backup_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    backup_inspect = backup_subcommands.add_parser("inspect", help="Inspect and verify one backup artifact without contacting Qdrant.")
    backup_inspect.set_defaults(qdrant_subcommand="backup")
    backup_inspect.add_argument("backup_id", help="Backup ID to inspect.")
    backup_inspect.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    restore = subcommands.add_parser("restore", help="Plan or explicitly restore a private Qdrant backup. Dry-run by default.")
    restore.add_argument("--backup", dest="backup_id", required=True, help="Backup ID to restore.")
    _add_dry_run_flags(restore)
    restore.add_argument("--backup-first", action="store_true", help="Create a fresh backup before live restore.")
    restore.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    learning = subcommands.add_parser("learning", help="Search or preview procedural learnings.")
    learning_subcommands = learning.add_subparsers(dest="learning_subcommand", required=True)

    learning_search = learning_subcommands.add_parser("search", help="Search procedural learnings.")
    learning_search.set_defaults(qdrant_subcommand="learning")
    learning_search.add_argument("query", help="Search query.")
    learning_search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    learning_search.add_argument("--learning-type", default=None, help="Optional learning_type filter.")
    learning_search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    learning_search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    learning_preview = learning_subcommands.add_parser("preview", help="Preview pending gated learning candidates.")
    learning_preview.set_defaults(qdrant_subcommand="learning")
    learning_preview.add_argument("--include-metadata", action="store_true", help="Include hook/source metadata.")
    learning_preview.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    learning_store = learning_subcommands.add_parser("store", help="Store an explicit procedural learning.")
    learning_store.set_defaults(qdrant_subcommand="learning")
    learning_store.add_argument("lesson", help="Durable lesson/procedure learned.")
    learning_store.add_argument("--learning-type", default=None, help="Optional learning type.")
    learning_store.add_argument("--trigger", default=None, help="Situation that should trigger recall.")
    learning_store.add_argument("--mistake", default=None, help="What went wrong or should be avoided.")
    learning_store.add_argument("--correction", default=None, help="Corrected action/procedure.")
    learning_store.add_argument("--evidence", default=None, help="Evidence that supports the lesson.")
    learning_store.add_argument("--tool-name", default=None, help="Tool involved, if any.")
    learning_store.add_argument("--command", default=None, help="Command involved, if any.")
    learning_store.add_argument("--importance", type=_importance, default=7, help="Importance, 1 to 10. Default: 7.")
    learning_store.add_argument("--confidence", type=_confidence, default=0.8, help="Confidence, 0 to 1. Default: 0.8.")
    _add_tag_flags(learning_store)
    learning_store.add_argument("--promote-to-skill-candidate", action="store_true", help="Mark as candidate for future skill promotion.")
    learning_store.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    learning_approve = learning_subcommands.add_parser("approve", help="Approve one pending gated learning candidate. Dry-run by default.")
    learning_approve.set_defaults(qdrant_subcommand="learning")
    learning_approve.add_argument("candidate_id", help="Candidate ID from learning preview.")
    _add_dry_run_flags(learning_approve)
    learning_approve.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    consolidate = subcommands.add_parser("consolidate", help="Generate a report-only consolidation proposal set.")
    consolidate.add_argument("--scope", choices=["memory", "learning", "both"], default="both", help="Report scope. Default: both.")
    consolidate.add_argument("--persist", action="store_true", help="Persist local report artifact.")
    consolidate.add_argument("--include-reconsolidation", action="store_true", help="Include reconsolidation candidates in the report.")
    consolidate.add_argument("--dry-run", action="store_true", default=True, help="Report generation is always dry-run. Default.")
    consolidate.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    apply_parser = subcommands.add_parser("apply", help="Preview or apply one explicit persisted consolidation proposal.")
    apply_parser.add_argument("--report-id", required=True, help="Persisted consolidation report ID.")
    apply_parser.add_argument("--proposal-id", required=True, help="Proposal ID to preview/apply.")
    apply_parser.add_argument("--action", required=True, choices=["merge", "delete", "promote_to_skill", "draft_review"], help="Expected proposal action.")
    _add_dry_run_flags(apply_parser)
    apply_parser.add_argument("--backup-first", action="store_true", help="Create a fresh backup before live apply.")
    apply_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher = subcommands.add_parser("watcher", help="Inspect or run report-only watcher consolidation helpers.")
    watcher_subcommands = watcher.add_subparsers(dest="watcher_subcommand", required=True)
    watcher_status = watcher_subcommands.add_parser("status", help="Show local watcher state without contacting services.")
    watcher_status.set_defaults(qdrant_subcommand="watcher")
    watcher_status.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    watcher_run = watcher_subcommands.add_parser("run", help="Run report-only watcher consolidation. No Qdrant mutation.")
    watcher_run.set_defaults(qdrant_subcommand="watcher")
    watcher_run.add_argument("--scope", choices=["memory", "learning", "both"], default="both", help="Report scope. Default: both.")
    watcher_run.add_argument("--max-points", type=_positive_int, default=300, help="Maximum points to inspect per collection. Default: 300.")
    watcher_run.add_argument("--max-groups", type=_positive_int, default=20, help="Maximum proposals to return. Default: 20.")
    watcher_run.add_argument("--reconsolidation-max-candidates", type=_positive_int, default=10, help="Maximum reconsolidation candidates. Default: 10.")
    try:
        watcher_run.add_argument("--include-reconsolidation", action=argparse.BooleanOptionalAction, default=True, help="Include reconsolidation candidates. Default: enabled.")
    except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
        watcher_run.add_argument("--include-reconsolidation", action="store_true", default=True, help="Include reconsolidation candidates. Default: enabled.")
        watcher_run.add_argument("--no-include-reconsolidation", dest="include_reconsolidation", action="store_false", help="Disable reconsolidation candidates.")
    watcher_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")


def qdrant_command(args) -> None:
    raise SystemExit(execute_command(args))
