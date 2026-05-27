from __future__ import annotations

# ruff: noqa: E402 - standalone CLI ensures plugin root is importable before loading command core.

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


def _add_search_filter_flags(parser: argparse.ArgumentParser, *, include_collection: bool = False) -> None:
    _add_tag_flags(parser)
    parser.add_argument("--source", default=None, help="Optional exact payload source filter.")
    parser.add_argument("--file-path", "--path", dest="file_path", default=None, help="Optional exact payload file_path filter.")
    parser.add_argument("--project-path", default=None, help="Optional exact payload project_path filter.")
    parser.add_argument("--since", default=None, help="Optional inclusive created_at lower bound (ISO timestamp).")
    parser.add_argument("--until", default=None, help="Optional inclusive created_at upper bound (ISO timestamp).")
    if include_collection:
        parser.add_argument("--collection", choices=["memory", "learning"], default=None, help="Collection to search. Default: memory.")


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
    _add_dry_run_flags(store)
    store.add_argument("--preview-duplicates", dest="preview_duplicates", action="store_true", help="Search for semantic duplicates before storing.")
    store.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    search = subcommands.add_parser("search", help="Search semantic memories.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    search.add_argument("--source-type", default=None, help="Optional source_type filter.")
    _add_search_filter_flags(search, include_collection=True)
    search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    search.add_argument("--include-fact-history", action="store_true", help="Include deprecated or superseded fact assertions in search results.")
    search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    show = subcommands.add_parser("show", help="Inspect one explicit Qdrant point by ID. Read-only.")
    show.add_argument("point_id", help="Explicit Qdrant point ID to inspect.")
    show.add_argument("--collection", required=True, choices=["memory", "learning"], help="Configured collection scope to inspect.")
    show.add_argument("--include-payload", action="store_true", help="Include redacted payload fields in output. Raw memory text may be sensitive.")
    show.add_argument("--include-vector", action="store_true", help="Include vector data. Omitted by default because vectors are large.")
    show.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    inspect = subcommands.add_parser("inspect", help="Inspect one explicit point payload and source metadata. Read-only.")
    inspect.add_argument("point_id", help="Explicit Qdrant point ID to inspect.")
    inspect.add_argument("--collection", choices=["memory", "learning"], default="memory", help="Configured collection scope. Default: memory.")
    inspect.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    trace = subcommands.add_parser("trace", help="Trace one point's derivation links. Read-only.")
    trace.add_argument("point_id", help="Explicit Qdrant point ID to trace.")
    trace.add_argument("--direction", choices=["upstream", "downstream", "both"], default="upstream", help="Trace direction. Default: upstream.")
    trace.add_argument("--collection", choices=["memory", "learning"], default="memory", help="Configured collection scope. Default: memory.")
    trace.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    expand = subcommands.add_parser("expand", help="Expand one point's source context with bounded output. Read-only.")
    expand.add_argument("point_id", help="Explicit Qdrant point ID to expand.")
    expand.add_argument("--mode", choices=["excerpt", "source", "neighbors"], default="excerpt", help="Expansion mode. Default: excerpt.")
    expand.add_argument("--max-chars", type=_positive_int, default=8000, help="Maximum characters to return. Default: 8000.")
    expand.add_argument("--collection", choices=["memory", "learning"], default="memory", help="Configured collection scope. Default: memory.")
    expand.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    source_status = subcommands.add_parser("source-status", help="Report one point's source existence/staleness. Read-only.")
    source_status.add_argument("point_id", help="Explicit Qdrant point ID whose source should be checked.")
    source_status.add_argument("--collection", choices=["memory", "learning"], default="memory", help="Configured collection scope. Default: memory.")
    source_status.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

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

    reports = subcommands.add_parser("reports", help="List or inspect persisted consolidation report artifacts. Local read-only.")
    reports_subcommands = reports.add_subparsers(dest="reports_subcommand", required=True)
    reports_list = reports_subcommands.add_parser("list", help="List persisted consolidation reports without contacting Qdrant.")
    reports_list.set_defaults(qdrant_subcommand="reports")
    reports_list.add_argument("--limit", type=_positive_int, default=20, help="Maximum reports to list. Default: 20.")
    reports_list.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    reports_show = reports_subcommands.add_parser("show", help="Show one persisted consolidation report by exact report ID.")
    reports_show.set_defaults(qdrant_subcommand="reports")
    reports_show.add_argument("report_id", help="Persisted report ID to inspect.")
    reports_show.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    proposals = subcommands.add_parser("proposals", help="Inspect proposals inside persisted consolidation reports. Local read-only.")
    proposals_subcommands = proposals.add_subparsers(dest="proposals_subcommand", required=True)
    proposals_show = proposals_subcommands.add_parser("show", help="Show one proposal by exact report ID and proposal ID.")
    proposals_show.set_defaults(qdrant_subcommand="proposals")
    proposals_show.add_argument("report_id", help="Persisted report ID containing the proposal.")
    proposals_show.add_argument("proposal_id", help="Proposal ID to inspect.")
    proposals_show.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    learning = subcommands.add_parser("learning", help="Search or preview procedural learnings.")
    learning_subcommands = learning.add_subparsers(dest="learning_subcommand", required=True)

    learning_search = learning_subcommands.add_parser("search", help="Search procedural learnings.")
    learning_search.set_defaults(qdrant_subcommand="learning")
    learning_search.add_argument("query", help="Search query.")
    learning_search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    learning_search.add_argument("--learning-type", default=None, help="Optional learning_type filter.")
    _add_search_filter_flags(learning_search)
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
    apply_parser.add_argument("--action", required=True, choices=["merge", "delete", "quarantine", "promote_to_skill", "draft_review"], help="Expected proposal action.")
    _add_dry_run_flags(apply_parser)
    apply_parser.add_argument("--backup-first", action="store_true", help="Create a fresh backup before live apply.")
    apply_parser.add_argument("--quarantine-days", type=_positive_int, default=30, help="Days to quarantine stale low-value memories when action=quarantine. Default: 30.")
    apply_parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher = subcommands.add_parser("watcher", help="Inspect watcher helpers; run defaults to report-only, guarded-auto is opt-in.")
    watcher_subcommands = watcher.add_subparsers(dest="watcher_subcommand", required=True)
    watcher_status = watcher_subcommands.add_parser("status", help="Show local watcher state without contacting services.")
    watcher_status.set_defaults(qdrant_subcommand="watcher")
    watcher_status.add_argument("--verbose", action="store_true", help="Include schedule, log, and recent event details.")
    watcher_status.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    watcher_run = watcher_subcommands.add_parser("run", help="Run watcher consolidation. Defaults to report-only; guarded-auto may apply preauthorized exact-ID proposals.")
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
    watcher_run.add_argument("--force-alert", action="store_true", help="Record an alert event even when the proposal signature is unchanged.")
    watcher_run.add_argument("--autonomy-mode", choices=["report-only", "guarded-auto"], default="report-only", help="Watcher autonomy policy. Default: report-only.")
    watcher_run.add_argument("--max-auto-actions", type=_positive_int, default=10, help="Maximum guarded-auto actions per run. Default: 10.")
    watcher_run.add_argument("--quarantine-days", type=_positive_int, default=30, help="Days to quarantine stale low-value memories before later hard deletion. Default: 30.")
    watcher_run.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher_install = watcher_subcommands.add_parser("install", help="Install or update the managed crontab watcher entry.")
    watcher_install.set_defaults(qdrant_subcommand="watcher")
    watcher_install.add_argument("--schedule", default="0 3 * * *", help="Five-field cron schedule. Default: 0 3 * * *.")
    watcher_install.add_argument("--scope", choices=["memory", "learning", "both"], default="both", help="Report scope for scheduled runs. Default: both.")
    watcher_install.add_argument("--max-points", type=_positive_int, default=300, help="Maximum points to inspect per collection. Default: 300.")
    watcher_install.add_argument("--max-groups", type=_positive_int, default=20, help="Maximum proposals to return. Default: 20.")
    watcher_install.add_argument("--reconsolidation-max-candidates", type=_positive_int, default=10, help="Maximum reconsolidation candidates. Default: 10.")
    try:
        watcher_install.add_argument("--include-reconsolidation", action=argparse.BooleanOptionalAction, default=True, help="Include reconsolidation candidates. Default: enabled.")
    except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
        watcher_install.add_argument("--include-reconsolidation", action="store_true", default=True, help="Include reconsolidation candidates. Default: enabled.")
        watcher_install.add_argument("--no-include-reconsolidation", dest="include_reconsolidation", action="store_false", help="Disable reconsolidation candidates.")
    watcher_install.add_argument("--approve", action="store_true", help="Required to replace an existing different managed watcher entry.")
    watcher_install.add_argument("--autonomy-mode", choices=["report-only", "guarded-auto"], default="report-only", help="Scheduled watcher autonomy policy. Default: report-only.")
    watcher_install.add_argument("--max-auto-actions", type=_positive_int, default=10, help="Maximum guarded-auto actions per run. Default: 10.")
    watcher_install.add_argument("--quarantine-days", type=_positive_int, default=30, help="Days to quarantine stale low-value memories before later hard deletion. Default: 30.")
    watcher_install.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher_uninstall = watcher_subcommands.add_parser("uninstall", help="Remove the managed crontab watcher entry. Requires --approve.")
    watcher_uninstall.set_defaults(qdrant_subcommand="watcher")
    watcher_uninstall.add_argument("--approve", action="store_true", help="Required to remove the managed watcher entry.")
    watcher_uninstall.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher_logs = watcher_subcommands.add_parser("logs", help="Read local watcher JSONL log events.")
    watcher_logs.set_defaults(qdrant_subcommand="watcher")
    watcher_logs.add_argument("--tail", type=_positive_int, default=20, help="Number of events to show. Default: 20.")
    watcher_logs.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher_inspect = watcher_subcommands.add_parser("inspect-state", help="Inspect local watcher state with sensitive fields redacted.")
    watcher_inspect.set_defaults(qdrant_subcommand="watcher")
    watcher_inspect.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    watcher_reset = watcher_subcommands.add_parser("reset-signature", help="Clear stored watcher proposal signatures. Requires --approve.")
    watcher_reset.set_defaults(qdrant_subcommand="watcher")
    watcher_reset.add_argument("--approve", action="store_true", help="Required to reset local watcher signatures.")
    watcher_reset.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")


def qdrant_command(args) -> None:
    raise SystemExit(execute_command(args))
