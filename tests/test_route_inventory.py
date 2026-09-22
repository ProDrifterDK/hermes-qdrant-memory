"""Executable inventory of the routes that can delete or replace a payload.

Four independent reviews of this closure found the same defect in four different
places: `qdrant_memory_store`, `qdrant_memory_extraction_approve`, restore and the
off-mode reindex each decided "is this payload protected?" from a local, narrower list
of fields. Every one of them was found by reading code, none by a failing test, and the
route table in ``docs/SAFETY.md`` claimed coverage that did not exist.

The fix for the four sites is the shared predicate. The fix for the *pattern* is this
file: the mutating surface is derived from the code, every derived entry has to be
classified, and a route declared protected has to name a test that actually exists.
Adding a tool or a CLI command now fails until someone decides what it does to a
protected payload, and ``docs/SAFETY.md`` cannot claim a pin that has been renamed or
deleted.

Classification is by hand because "does this command mutate Qdrant?" is not derivable
from the parser. ``uncovered`` exists so that the honest answer has a place to go: a
route whose refusal was reasoned from code but never executed stays visible as debt
instead of being silently counted as covered.
"""
from __future__ import annotations

import argparse
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# --- the surface, derived from the code ------------------------------------


def _tool_names() -> set[str]:
    """Every ``qdrant_memory_*`` / ``qdrant_learning_*`` name the provider dispatches."""
    tree = ast.parse((REPO / "__init__.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "handle_tool_call":
            for sub in ast.walk(node):
                if isinstance(sub, ast.If) and isinstance(sub.test, ast.Compare):
                    right = sub.test.comparators[0]
                    if isinstance(right, ast.Constant) and isinstance(right.value, str):
                        names.add(right.value)
    assert names, "handle_tool_call dispatch could not be parsed"
    return names


def _cli_commands() -> set[str]:
    """Every registered CLI command path, e.g. ``backup restore``.

    Loaded by path: ``import cli`` resolves to the host application's module, not this
    plugin's.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location("_route_inventory_cli", REPO / "cli.py")
    assert spec and spec.loader
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)

    parser = argparse.ArgumentParser()
    cli.register_cli(parser)

    def walk(current: argparse.ArgumentParser, prefix: str = "") -> set[str]:
        found: set[str] = set()
        for action in current._actions:
            if isinstance(action, argparse._SubParsersAction):
                for name, sub in action.choices.items():
                    path = f"{prefix} {name}".strip()
                    found.add(path)
                    found |= walk(sub, path)
        return found

    commands = walk(parser)
    assert commands, "register_cli produced no subcommands"
    return commands


# --- the inventory ---------------------------------------------------------
#
# kind:
#   protection  a route that can delete or replace an existing payload; it must refuse
#               the protected shapes and its refusal must be pinned by a test that
#               exists in this tree
#   transition  the reviewed lineage path, which is allowed to write
#   read_only   does not touch Qdrant payloads (may write local artifacts)
#   uncovered   mutates, and its refusal is reasoned from code but not executed

FORGET_PINS = [
    "tests/test_consolidation_apply.py::test_structural_lineage_refusal_wording_is_caller_agnostic",
    "tests/test_consolidation_apply.py::test_a_demoted_dependent_stays_retirable",
    "tests/test_consolidation_apply.py::test_forget_refuses_any_dependent_and_deletes_all_clear_targets",
    "tests/test_consolidation_apply.py::test_forget_lock_contention_refuses_without_any_write_and_respects_timeout",
]
STORE_PINS = [
    "tests/test_consolidation_apply.py::test_store_refuses_to_overwrite_a_protected_point_in_every_mode",
    "tests/test_consolidation_apply.py::test_store_refuses_a_history_only_dependent_in_every_mode",
    "tests/test_consolidation_apply.py::test_store_fails_closed_when_the_target_cannot_be_read",
    "tests/test_consolidation_apply.py::test_store_answers_the_retirement_text_for_a_point_that_carries_both",
]
EXTRACTION_PINS = [
    "tests/test_consolidation_apply.py::test_extraction_approval_refuses_to_overwrite_a_protected_point",
    "tests/test_consolidation_apply.py::test_extraction_approval_refuses_a_history_only_dependent",
    "tests/test_consolidation_apply.py::test_extraction_approval_fails_closed_when_the_target_cannot_be_read",
]
RESTORE_PINS = [
    "tests/test_backup_cli.py::test_restore_refuses_to_overwrite_every_protected_payload",
    "tests/test_backup_cli.py::test_restore_refuses_a_protected_target_in_the_learnings_scope",
    "tests/test_backup_cli.py::test_restore_refuses_the_review_only_shape",
    "tests/test_backup_cli.py::test_restore_still_overwrites_an_ordinary_payload",
    "tests/test_backup_cli.py::test_restore_refuses_missing_chunk_retired_by_committed_event",
]
INDEX_OFF_PINS = [
    "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state",
    "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field",
    "tests/test_lineage.py::test_off_mode_refuses_to_destroy_or_duplicate_captured_file",
    "tests/test_lineage.py::test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks",
    "tests/test_lineage.py::test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs",
]
CONSOLIDATION_PINS = [
    "tests/test_consolidation_apply.py::test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile",
]
# The learning store has no payload predicate: its safety is the collection-separation
# invariant, because an upsert there would otherwise be able to replace a memory point.
LEARNING_STORE_PINS = [
    "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
]

ROUTE_INVENTORY: dict[str, dict[str, object]] = {}


def _declare(names, kind, *, pins=None, note="", reason=""):
    for name in names:
        ROUTE_INVENTORY[name] = {"kind": kind, "pins": list(pins or []), "note": note, "reason": reason}


_declare(
    [
        "qdrant_memory_store", "store",
    ], "protection", pins=STORE_PINS,
    note="replaces the payload at a deterministic content-derived id",
)
_declare(["qdrant_memory_forget", "forget"], "protection", pins=FORGET_PINS)
_declare(
    ["qdrant_memory_extraction_approve", "learning approve"], "protection", pins=EXTRACTION_PINS,
    note="upserts a regenerated candidate id",
)
_declare(["restore"], "protection", pins=RESTORE_PINS, note="the third in-place overwrite route")
_declare(["qdrant_memory_index [off]", "index [off]"], "protection", pins=INDEX_OFF_PINS,
         note="rewrites chunk payloads at content-derived ids and deletes stale ids")
_declare(["qdrant_memory_consolidation_apply", "apply"], "protection", pins=CONSOLIDATION_PINS,
         note="merge/delete/quarantine; the managed check runs before the lock (N5)")
_declare(
    ["qdrant_learning_store", "learning store", "qdrant_learning_approve"], "protection",
    pins=LEARNING_STORE_PINS,
    note="no payload predicate: protected by the collection-separation invariant only",
)
_declare(["qdrant_memory_index [capture]", "index [capture]"], "transition")
_declare(["qdrant_memory_index [reconcile]", "index [reconcile]"], "transition")
_declare(["watcher run"], "protection", pins=[
    "tests/test_consolidation_apply.py::test_apply_guarded_auto_preauthorized_maintenance_refuses_sensitive_points",
], note="report-only by default; --guarded-auto reaches the apply path")

_declare(["qdrant_memory_improve_apply", "improve apply"], "uncovered",
         reason="predicate read on 567eeac; stricter than the shared one on the traced shapes, never executed")
_declare(["qdrant_memory_raptor_apply"], "uncovered",
         reason="same as improve apply: read, not executed")

_declare([
    # provider reads
    "qdrant_memory_status", "qdrant_memory_search", "qdrant_memory_graph_search",
    "qdrant_memory_context", "qdrant_memory_retrieve", "qdrant_memory_inspect",
    "qdrant_memory_trace", "qdrant_memory_expand", "qdrant_memory_source_status",
    "qdrant_memory_consolidate", "qdrant_memory_memory_pr", "qdrant_memory_raptor_status",
    "qdrant_memory_improve_preview", "qdrant_memory_extraction_preview",
    "qdrant_learning_preview", "qdrant_learning_search",
    # cli reads
    "status", "doctor", "config", "config show", "search", "graph-search", "retrieve",
    "context", "show", "inspect", "trace", "expand", "source-status",
    "export", "export memory", "export learning", "reports", "reports list",
    "reports show", "proposals", "proposals show", "eval", "eval-capture", "eval-gate",
    "learning", "learning search", "learning preview", "consolidate",
    "improve", "improve preview", "backup", "backup list", "backup inspect",
    "watcher", "watcher status", "watcher logs", "watcher inspect-state",
    "watcher install", "watcher uninstall", "watcher reset-signature",
], "read_only", note="no Qdrant payload mutation; some write local artifacts")

# Writes outside Qdrant (files, reports, proposal state) with no payload to protect.
_declare(["backup create", "eval-capture"], "read_only",
         note="writes files only; the restore route that consumes them is the protection row")


# --- assertions ------------------------------------------------------------


def _base(entry: str) -> str:
    return entry.split(" [", 1)[0]


def test_every_surface_entry_is_classified():
    """A new tool or command must be classified before it can ship."""
    derived = _tool_names() | _cli_commands()
    missing = sorted(
        name for name in derived
        if name not in ROUTE_INVENTORY and f"{name} [off]" not in ROUTE_INVENTORY
    )
    assert missing == [], (
        "these entry points are not classified in ROUTE_INVENTORY: "
        f"{missing}. Decide what each one does to a protected payload: protection "
        "(with pins), transition, read_only, or uncovered with a reason."
    )


def test_no_inventory_row_is_stale():
    derived = _tool_names() | _cli_commands()
    stale = sorted(key for key in ROUTE_INVENTORY if _base(key) not in derived)
    assert stale == [], f"ROUTE_INVENTORY rows that no longer name a real entry point: {stale}"


def test_every_protection_route_names_a_pin_that_exists():
    """The route table may not claim a pin that is not there.

    ``docs/SAFETY.md`` said "pinned" three times for rows no test covered. A claim is
    only as good as its resolvable node id, so this resolves them.
    """
    unresolved: list[str] = []
    for entry, row in sorted(ROUTE_INVENTORY.items()):
        if row["kind"] != "protection":
            continue
        pins = row["pins"]
        assert pins, f"{entry} is a protection route with no pins"
        for node in pins:  # type: ignore[union-attr]
            path, _, name = node.partition("::")
            name = name.split("[", 1)[0]
            target = REPO / path
            if not target.is_file() or not re.search(rf"^def {re.escape(name)}\(", target.read_text(encoding="utf-8"), re.M):
                unresolved.append(f"{entry} -> {node}")
    assert unresolved == [], f"protection routes naming a pin that does not exist: {unresolved}"


def test_every_uncovered_route_declares_a_reason():
    missing = sorted(
        entry for entry, row in ROUTE_INVENTORY.items()
        if row["kind"] == "uncovered" and not str(row["reason"]).strip()
    )
    assert missing == [], f"uncovered routes without a reason: {missing}"


def test_a_route_is_classified_once_with_one_kind():
    kinds = {"protection", "transition", "read_only", "uncovered"}
    bad = {entry: row["kind"] for entry, row in ROUTE_INVENTORY.items() if row["kind"] not in kinds}
    assert bad == {}, f"unknown kinds: {bad}"


@pytest.mark.parametrize("name", ["qdrant_memory_store", "restore", "qdrant_memory_index [off]"])
def test_the_inventory_sees_the_four_routes_that_failed_reviews(name):
    """The four historical blocker routes stay classified as protection routes."""
    assert name in ROUTE_INVENTORY
    assert ROUTE_INVENTORY[name]["kind"] == "protection"
    assert ROUTE_INVENTORY[name]["pins"]
