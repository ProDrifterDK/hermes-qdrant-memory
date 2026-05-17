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


def _add_dry_run_flags(parser: argparse.ArgumentParser) -> None:
    dry_run = parser.add_mutually_exclusive_group()
    dry_run.add_argument("--dry-run", dest="dry_run", action="store_true", default=True, help="Preview without live mutation. Default.")
    dry_run.add_argument("--no-dry-run", dest="dry_run", action="store_false", help="Allow live mutation when paired with --approve.")
    parser.add_argument("--approve", action="store_true", help="Required with --no-dry-run for live mutation.")


def register_cli(parser: argparse.ArgumentParser) -> None:
    """Register the qdrant memory-provider CLI under `hermes qdrant`."""

    parser.description = "Manage the Qdrant Hermes memory provider."
    subcommands = parser.add_subparsers(dest="qdrant_subcommand", required=True)

    subcommands.add_parser("status", help="Show Qdrant provider status.")
    subcommands.add_parser("doctor", help="Show status-backed lightweight diagnostics.")

    search = subcommands.add_parser("search", help="Search semantic memories.")
    search.add_argument("query", help="Search query.")
    search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    search.add_argument("--source-type", default=None, help="Optional source_type filter.")
    search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    search.add_argument("--json", action="store_true", help="Emit raw JSON output.")

    index = subcommands.add_parser("index", help="Index files or directories into memory. Dry-run by default.")
    index.add_argument("paths", nargs="+", help="Files or directories to index.")
    _add_dry_run_flags(index)
    index.add_argument("--max-files", type=int, default=None, help="Maximum files to scan/index.")
    index.add_argument("--force", action="store_true", help="Re-index chunks even if unchanged.")
    index.add_argument("--json", action="store_true", help="Emit raw JSON output.")

    forget = subcommands.add_parser("forget", help="Forget explicit point IDs. Dry-run by default.")
    forget.add_argument("ids", nargs="+", help="Explicit Qdrant point IDs to delete.")
    _add_dry_run_flags(forget)
    forget.add_argument("--json", action="store_true", help="Emit raw JSON output.")

    learning = subcommands.add_parser("learning", help="Search or preview procedural learnings.")
    learning_subcommands = learning.add_subparsers(dest="learning_subcommand", required=True)

    learning_search = learning_subcommands.add_parser("search", help="Search procedural learnings.")
    learning_search.set_defaults(qdrant_subcommand="learning")
    learning_search.add_argument("query", help="Search query.")
    learning_search.add_argument("--top-k", type=_top_k, default=5, help="Maximum results to return, 1 to 20. Default: 5.")
    learning_search.add_argument("--learning-type", default=None, help="Optional learning_type filter.")
    learning_search.add_argument("--include-metadata", action="store_true", help="Include full payload metadata.")
    learning_search.add_argument("--json", action="store_true", help="Emit raw JSON output.")

    learning_preview = learning_subcommands.add_parser("preview", help="Preview pending gated learning candidates.")
    learning_preview.set_defaults(qdrant_subcommand="learning")
    learning_preview.add_argument("--include-metadata", action="store_true", help="Include hook/source metadata.")
    learning_preview.add_argument("--json", action="store_true", help="Emit raw JSON output.")

    consolidate = subcommands.add_parser("consolidate", help="Generate a report-only consolidation proposal set.")
    consolidate.add_argument("--scope", choices=["memory", "learning", "both"], default="both", help="Report scope. Default: both.")
    consolidate.add_argument("--persist", action="store_true", help="Persist local report artifact.")
    consolidate.add_argument("--include-reconsolidation", action="store_true", help="Include reconsolidation candidates in the report.")
    consolidate.add_argument("--dry-run", action="store_true", default=True, help="Report generation is always dry-run. Default.")
    consolidate.add_argument("--json", action="store_true", help="Emit raw JSON output.")

    apply_parser = subcommands.add_parser("apply", help="Preview or apply one explicit persisted consolidation proposal.")
    apply_parser.add_argument("--report-id", required=True, help="Persisted consolidation report ID.")
    apply_parser.add_argument("--proposal-id", required=True, help="Proposal ID to preview/apply.")
    apply_parser.add_argument("--action", required=True, choices=["merge", "delete", "promote_to_skill", "draft_review"], help="Expected proposal action.")
    _add_dry_run_flags(apply_parser)
    apply_parser.add_argument("--json", action="store_true", help="Emit raw JSON output.")


def qdrant_command(args) -> None:
    raise SystemExit(execute_command(args))
