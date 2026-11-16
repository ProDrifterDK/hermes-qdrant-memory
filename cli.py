from __future__ import annotations

# ruff: noqa: E402 - standalone CLI ensures plugin root is importable before loading command core.

import argparse
import sys
from pathlib import Path

_PLUGIN_ROOT = Path(__file__).resolve().parent
if str(_PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(_PLUGIN_ROOT))

from qdrant_memory.cli_core import execute_command
from qdrant_memory.recipes import list_recipe_names
from qdrant_memory.schema import RELATION_TYPES


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
    store.add_argument("--cite", default=None, metavar="FIRST:LAST", help="Cite messages FIRST..LAST (state.db message ids) of --session-id as the fact's origin. Needs session_origin_mode: recorded; preview (dry-run) first, then approve with the same arguments.")
    store.add_argument("--session-id", dest="session_id", default=None, help="Session of the --cite span (required with --cite).")
    store.add_argument("--cite-quote", dest="cite_quote", action="store_true", help="Also store a redacted quote of the cited messages (with --cite).")
    store.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    search = subcommands.add_parser("search", help="Search semantic memories.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    search.add_argument("--source-type", default=None, help="Optional source_type filter.")
    _add_search_filter_flags(search, include_collection=True)
    search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    search.add_argument("--include-fact-history", action="store_true", help="Include deprecated or superseded fact assertions in search results.")
    search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    graph_search = subcommands.add_parser("graph-search", help="Read-only graph-aware search over entity/edge payloads with hybrid reranking.")
    graph_search.add_argument("query", help="Search query.")
    graph_search.add_argument("--top-k", type=_top_k, default=5, help="Maximum final results to return, 1 to 20. Default: 5.")
    graph_search.add_argument("--candidate-seed-top-k", type=_positive_int, default=20, help="Semantic seeds for graph expansion. Default: 20.")
    graph_search.add_argument("--max-graph-results", type=_positive_int, default=20, help="Hard cap on expansion pool before reranking. Default: 20.")
    graph_search.add_argument("--max-depth", type=int, choices=[1, 2, 3], default=2, help="BFS depth for graph expansion. Default: 2.")
    graph_search.add_argument("--entity-type", action="append", default=[], help="Optional entity_type filter. Repeat or pass comma-separated values.")
    graph_search.add_argument("--relation-type", action="append", default=[], choices=list(RELATION_TYPES), help="Optional relation_type filter. Repeat or pass comma-separated values.")
    graph_search.add_argument("--include-fact-history", action="store_true", help="Include deprecated/superseded graph edges in ranking.")
    try:
        graph_search.add_argument("--debug", action=argparse.BooleanOptionalAction, default=True, help="Include debug breakdown. Default: enabled.")
    except AttributeError:  # pragma: no cover - Python < 3.9 compatibility
        graph_search.add_argument("--debug", action="store_true", default=True, help="Include debug breakdown. Default: enabled.")
        graph_search.add_argument("--no-debug", dest="debug", action="store_false", help="Disable debug breakdown.")
    graph_search.add_argument("--collection", choices=["memory", "learning"], default="memory", help="Collection to search. Default: memory.")
    graph_search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    retrieve = subcommands.add_parser("retrieve", help="Read-only hybrid retrieve across dense+sparse, graph, and RAPTOR. Never writes.")
    retrieve.add_argument("query", help="Search query.")
    retrieve.add_argument("--top-k", type=_top_k, default=5, help="Maximum results per bucket, 1 to 20. Default: 5.")
    retrieve.add_argument("--mode", choices=["hybrid", "evidence"], default="hybrid", help="hybrid returns all lanes; evidence demotes parent-only summaries.")
    retrieve.add_argument("--include-fact-history", action="store_true", help="Include deprecated/superseded history.")
    retrieve.add_argument("--include-metadata", action="store_true", help="Include raw RAPTOR payload metadata.")
    retrieve.add_argument("--source-type", default=None, help="Optional source_type filter.")
    _add_search_filter_flags(retrieve, include_collection=True)
    retrieve.add_argument("--max-depth", type=int, choices=[1, 2, 3], default=2, help="RAPTOR BFS depth budget. Default: 2.")
    retrieve.add_argument("--max-children", type=_positive_int, default=8, help="RAPTOR children fanout budget. Default: 8.")
    retrieve.add_argument("--max-source-chars", type=_positive_int, default=1200, help="Per-result text cap. Default: 1200.")
    retrieve.add_argument("--candidate-seed-top-k", type=_positive_int, default=20, help="Semantic seeds for graph expansion. Default: 20.")
    retrieve.add_argument("--max-graph-results", type=_positive_int, default=20, help="Hard cap on expansion pool before reranking. Default: 20.")
    retrieve.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    context = subcommands.add_parser("context", help="Build a read-only provenance-explicit context packet from a recipe template.")
    context.add_argument("--template", choices=list_recipe_names(), default="source_backed_answer", help="Context template/recipe. Default: source_backed_answer.")
    context.add_argument("--topic", required=True, help="Topic/query for the context packet.")
    context.add_argument("--top-k", type=_top_k, default=None, help="Maximum compact recall results, 1 to 20. Default: recipe budget.")
    context.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

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

    origin_check = subcommands.add_parser(
        "origin-check",
        help="Re-verify session_origin citations against state.db and count them per status. Read-only.",
    )
    origin_check.add_argument("--collection", choices=["memory", "learning"], default="memory", help="Configured collection scope. Default: memory.")
    origin_check.add_argument("--limit", type=_positive_int, default=100, help="Points with session_origin per page (at most 500). Default: 100.")
    origin_check.add_argument("--offset", default=None, help="next_offset printed by the previous page.")
    origin_check.add_argument("--max-ids", type=_positive_int, default=100, help="Point ids listed per status (at most 500). Default: 100.")
    origin_check.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

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

    eval_parser = subcommands.add_parser(
        "eval",
        help="Offline Phase 6A evaluation of captured retrieval packets. No Qdrant calls.",
    )
    eval_parser.set_defaults(qdrant_subcommand="eval")
    eval_parser.add_argument(
        "--cases",
        required=True,
        help="Path to eval-cases JSONL artifact (case_id, query, expected_*, forbidden_*).",
    )
    eval_parser.add_argument(
        "--runs",
        required=True,
        help="Path to eval-runs JSONL artifact (case_id, variant, packet, latency_ms, error).",
    )
    eval_parser.add_argument(
        "--top-k",
        type=_positive_int,
        default=5,
        help="Per-lane top-k cutoff for hit metrics. Default: 5.",
    )
    eval_parser.add_argument(
        "--latency-budget-ms",
        type=int,
        default=750,
        help="Latency budget in ms; rows above this fail the latency_budget check. Default: 750.",
    )
    eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (default: human summary).",
    )

    eval_capture = subcommands.add_parser(
        "eval-capture",
        help="Phase 6B read-only capture of retrieval variants for offline eval. Constructs provider; never mutates Qdrant.",
    )
    eval_capture.set_defaults(qdrant_subcommand="eval-capture")
    eval_capture.add_argument(
        "--cases",
        required=True,
        help="Path to eval-cases JSONL artifact (case_id, query, expected_*, forbidden_*).",
    )
    eval_capture.add_argument(
        "--runs-out",
        required=True,
        help="Output path for the captured eval-runs JSONL artifact.",
    )
    eval_capture.add_argument(
        "--variants",
        default="all",
        help="Comma-separated variant list or 'all'. Default: all.",
    )
    eval_capture.add_argument(
        "--top-k",
        type=_positive_int,
        default=5,
        help="Per-lane top-k for retrieval. Default: 5.",
    )
    eval_capture.add_argument(
        "--mode",
        choices=["hybrid", "evidence"],
        default="hybrid",
        help="Hybrid router mode. Default: hybrid.",
    )
    eval_capture.add_argument(
        "--max-depth",
        type=int,
        choices=[1, 2, 3],
        default=2,
        help="RAPTOR/graph BFS depth budget. Default: 2.",
    )
    eval_capture.add_argument(
        "--max-children",
        type=_positive_int,
        default=8,
        help="RAPTOR children fanout budget. Default: 8.",
    )
    eval_capture.add_argument(
        "--max-source-chars",
        type=_positive_int,
        default=1200,
        help="Per-result text cap. Default: 1200.",
    )
    eval_capture.add_argument(
        "--candidate-seed-top-k",
        type=_positive_int,
        default=20,
        help="Semantic seeds for graph expansion. Default: 20.",
    )
    eval_capture.add_argument(
        "--max-graph-results",
        type=_positive_int,
        default=20,
        help="Hard cap on graph expansion pool before reranking. Default: 20.",
    )
    eval_capture.add_argument(
        "--include-fact-history",
        action="store_true",
        help="Include deprecated/superseded history in retrieval.",
    )
    eval_capture.add_argument(
        "--include-metadata",
        action="store_true",
        help="Include raw RAPTOR payload metadata in summaries/leaves.",
    )
    eval_capture.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (default: human summary).",
    )

    eval_gate = subcommands.add_parser(
        "eval-gate",
        help="Phase 6F shadow gate: evaluate explicit thresholds over an eval report. No Qdrant calls.",
    )
    eval_gate.set_defaults(qdrant_subcommand="eval-gate")
    eval_gate.add_argument(
        "--report",
        required=True,
        help="Path to a JSON report produced by 'hermes qdrant eval --json'.",
    )
    eval_gate.add_argument(
        "--candidate-variant",
        default="hybrid",
        help="Candidate variant to evaluate. Default: hybrid.",
    )
    eval_gate.add_argument(
        "--baseline-variant",
        action="append",
        default=None,
        dest="baseline_variants",
        help="Baseline variant to compare against. Repeatable. Default: dense-only,dense+sparse.",
    )
    eval_gate.add_argument(
        "--thresholds-file",
        default=None,
        help="Path to a JSON file of threshold overrides (auto-recall-default preset applies if omitted).",
    )
    eval_gate.add_argument(
        "--min-case-count", type=int, default=None, help="Override: minimum case_count.",
    )
    eval_gate.add_argument(
        "--min-scored-count", type=int, default=None, help="Override: minimum scored_count.",
    )
    eval_gate.add_argument(
        "--max-errored-count", type=int, default=None, help="Override: maximum errored_count.",
    )
    eval_gate.add_argument(
        "--min-candidate-hit-at-k", type=float, default=None, help="Override: minimum candidate hit_at_k_rate.",
    )
    eval_gate.add_argument(
        "--min-candidate-source-hit-at-k", type=float, default=None, help="Override: minimum candidate source_hit_at_k_rate.",
    )
    eval_gate.add_argument(
        "--min-hit-at-k-lift", type=float, default=None, help="Override: minimum hit_at_k lift over best baseline.",
    )
    eval_gate.add_argument(
        "--min-source-hit-at-k-lift", type=float, default=None, help="Override: minimum source_hit_at_k lift over best baseline.",
    )
    eval_gate.add_argument(
        "--max-exact-id-drop", type=float, default=None, help="Override: maximum exact_identifier_hit_rate drop vs best baseline.",
    )
    eval_gate.add_argument(
        "--max-wrong-memory-rate", type=float, default=None, help="Override: maximum wrong_memory_rate (absolute).",
    )
    eval_gate.add_argument(
        "--max-wrong-memory-regression", type=float, default=None, help="Override: maximum wrong_memory_rate regression vs best baseline.",
    )
    eval_gate.add_argument(
        "--max-latency-p95-ms", type=float, default=None, help="Override: maximum latency_ms_p95 (ms).",
    )
    eval_gate.add_argument(
        "--min-latency-budget-pass-rate", type=float, default=None, help="Override: minimum latency_budget_pass_rate.",
    )
    eval_gate.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON output (default: human summary).",
    )

    learning = subcommands.add_parser("learning", help="Search or store procedural learnings.")
    learning_subcommands = learning.add_subparsers(dest="learning_subcommand", required=True)

    learning_search = learning_subcommands.add_parser("search", help="Search procedural learnings.")
    learning_search.set_defaults(qdrant_subcommand="learning")
    learning_search.add_argument("query", help="Search query.")
    learning_search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    learning_search.add_argument("--learning-type", default=None, help="Optional learning_type filter.")
    _add_search_filter_flags(learning_search)
    learning_search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    learning_search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

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
    watcher_run.add_argument("--scan-window", choices=["fixed", "rotating"], default="fixed", help="fixed (default) inspects the first --max-points IDs every run; rotating resumes from a persisted cursor so successive runs cover the whole collection, then wrap.")
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
    watcher_install.add_argument("--scan-window", choices=["fixed", "rotating"], default="fixed", help="Scan window for scheduled runs. Default: fixed.")
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

    improve = subcommands.add_parser("improve", help="Post-session graph improve preview/apply. Preview is dry-run only.")
    improve_subcommands = improve.add_subparsers(dest="improve_subcommand", required=True)

    improve_preview = improve_subcommands.add_parser("preview", help="Preview improve candidates as a dry-run report. No Qdrant writes.")
    improve_preview.set_defaults(qdrant_subcommand="improve")
    improve_preview.add_argument("--source-scope", choices=["source_text", "point_ids"], default=None, help="Source scope for extraction.")
    improve_preview.add_argument("--source-text", default=None, help="Explicit source text to extract from.")
    improve_preview.add_argument("--source-uri", default=None, help="Source URI for explicit source text.")
    improve_preview.add_argument("--point-ids", nargs="*", default=None, help="Qdrant point IDs to read as source (when scope=point_ids).")
    improve_preview.add_argument("--session-id", default=None, help="Session ID. Defaults to current.")
    improve_preview.add_argument("--max-candidates", type=_positive_int, default=20, help="Maximum candidates. Default: 20.")
    improve_preview.add_argument("--no-persist", action="store_true", help="Do not persist report artifact.")
    improve_preview.add_argument("--include-metadata", action="store_true", help="Include verbose metadata.")
    improve_preview.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    improve_apply = improve_subcommands.add_parser("apply", help="Preview or apply one candidate from a report. Dry-run by default.")
    improve_apply.set_defaults(qdrant_subcommand="improve")
    improve_apply.add_argument("--report-id", required=True, help="Report ID from improve preview.")
    improve_apply.add_argument("--candidate-id", required=True, help="Candidate ID from the report.")
    _add_dry_run_flags(improve_apply)
    improve_apply.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    _register_session_recall(subcommands)
    _register_capture(subcommands)


def _recall_top_k(value: str) -> int:
    parsed = int(value)
    if parsed < 1 or parsed > 30:
        raise argparse.ArgumentTypeError("must be between 1 and 30")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or more")
    return parsed


def _register_capture(subcommands) -> None:
    capture = subcommands.add_parser(
        "capture",
        help="Automatic capture: read state.db since the capture cursor and classify who wrote each row.",
    )
    capture_subcommands = capture.add_subparsers(dest="capture_subcommand", required=True)
    scan = capture_subcommands.add_parser(
        "scan",
        help="Count new state.db rows per author class (host, machine, human) since the cursor. Read-only.",
    )
    scan.set_defaults(qdrant_subcommand="capture")
    scan.add_argument("--since-days", type=_non_negative_float, default=None,
                      help="Only rows from the last N days (bounds a first scan).")
    scan.add_argument("--advance", action="store_true",
                      help="Move the cursor to the end of this scan. Without it the cursor is left unchanged.")
    scan.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    run = capture_subcommands.add_parser(
        "run", help="Capture batch: new owner rows (heuristic detector) and, when enabled, the built-in memory "
                    "mirror. Dry-run by default.")
    run.set_defaults(qdrant_subcommand="capture")
    run_mode = run.add_mutually_exclusive_group()
    run_mode.add_argument("--dry-run", dest="apply", action="store_false", default=False,
                          help="Report what would happen; write only the run report. Default.")
    run_mode.add_argument("--apply", dest="apply", action="store_true",
                          help="Queue, apply the auto/mirror routes and advance the cursor (capture_mode: on).")
    run.add_argument("--since-days", type=_non_negative_float, default=None,
                     help="Only rows from the last N days (bounds a first run).")
    run.add_argument("--json", action="store_true", help="Emit machine-readable JSON output (ids only).")

    digest = capture_subcommands.add_parser(
        "digest", help="Plain-text digest of the capture queue since the last digest, for a cron delivery. "
                       "Prints nothing when nothing is new; writes only its own marker.")
    digest.set_defaults(qdrant_subcommand="capture")
    digest.add_argument("--max-items", type=_positive_int, default=10, help="Pending items to list (max 10).")
    digest.add_argument("--language", choices=["en", "es"], default=None,
                        help="Digest language. Default: qdrant_memory.capture_digest_language (en).")

    review = capture_subcommands.add_parser(
        "review", help="List capture items waiting for review (pending, oldest first). Read-only.")
    review.set_defaults(qdrant_subcommand="capture")
    review.add_argument("--limit", type=_positive_int, default=10, help="Items per page (max 50). Default: 10.")
    review.add_argument("--offset", type=int, default=0, help="Items to skip. Default: 0.")
    review.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    for name, text in (("approve", "Approve one review item: re-verify its cited message, then store it. "
                                   "Dry-run by default."),
                       ("reject", "Reject one review item (an event; nothing is written to memory). "
                                  "Dry-run by default.")):
        decide = capture_subcommands.add_parser(name, help=text)
        decide.set_defaults(qdrant_subcommand="capture")
        decide.add_argument("id", help="Capture item id: the full id or its 12-character short id.")
        if name == "approve":
            decide.add_argument("--fact", default=None, metavar="TEXT",
                                help="Store this one-sentence statement instead of the queued text. The citation "
                                     "still points at the queued message; the audit records the edit.")
        _add_dry_run_flags(decide)
        decide.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")


def _register_session_recall(subcommands) -> None:
    recall = subcommands.add_parser(
        "session-recall",
        help="Raw session-history recall: index state.db conversation windows, search them, report lag.",
    )
    recall_subcommands = recall.add_subparsers(dest="session_recall_subcommand", required=True)

    index = recall_subcommands.add_parser(
        "index",
        help="Incrementally index state.db into the session-window collection (its only write target).",
    )
    index.set_defaults(qdrant_subcommand="session-recall")
    index.add_argument("--full", action="store_true", help="Re-window every session, ignoring the cursors.")
    index.add_argument("--max-seconds", type=_non_negative_float, default=None,
                       help="Stop at the next batch boundary after this many seconds; the cursor keeps what was written.")
    index.add_argument("--embedding-url", default="",
                       help="Embedding server base URL (e.g. http://127.0.0.1:18100/v1) for a faster backfill. "
                            "Must serve the same model as the collection.")
    index.add_argument("--batch", type=_positive_int, default=16, help="Windows per /embeddings request. Default: 16.")
    index.add_argument("--lock-timeout", type=_non_negative_float, default=60.0,
                       help="Seconds to wait for the single-writer lock. Default: 60.")
    index.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    search = recall_subcommands.add_parser("search", help="Search session windows. Read-only: no catch-up.")
    search.set_defaults(qdrant_subcommand="session-recall")
    search.add_argument("query", help="What to recall, in natural language.")
    search.add_argument("--scope", choices=["current", "past", "all"], default="all", help="Default: all.")
    search.add_argument("--session-id", default="", help="Caller session for --scope current/past.")
    search.add_argument("--top-k", type=_recall_top_k, default=None, help="1 to 30. Default: configured top_k.")
    search.add_argument("--after", default=None, help="ISO date/datetime lower bound (local time).")
    search.add_argument("--before", default=None, help="ISO date/datetime upper bound (local time).")
    search.add_argument("--source", action="append", default=[], help="Session source filter. Repeatable.")
    search.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")

    status = recall_subcommands.add_parser(
        "status", help="Window count, cursor, index_lag, rerank configuration and collection health. Read-only."
    )
    status.set_defaults(qdrant_subcommand="session-recall")
    status.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")


def qdrant_command(args) -> None:
    raise SystemExit(execute_command(args))
