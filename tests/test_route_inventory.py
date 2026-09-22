"""Executable inventory of the routes that can delete or replace a payload.

Four independent reviews of this closure found the same defect in four different
places: `qdrant_memory_store`, `qdrant_memory_extraction_approve`, restore and the
off-mode reindex each decided "is this payload protected?" from a local, narrower list
of fields. Every one of them was found by reading code, none by a failing test, and the
route table in ``docs/SAFETY.md`` claimed coverage that did not exist.

The fix for the four sites is the shared predicate. The fix for the *pattern* is this
file: the mutating surface is derived from the code — tool dispatch, CLI commands and
provider hook overrides — every derived entry has to be classified, and a route declared
protected has to name a test that actually exists. Adding a tool, a command or a hook
now fails until someone decides what it does to a protected payload, and the SAFETY
route table cannot claim a pin that has been renamed or deleted.

Classification is by hand because "does this command mutate Qdrant?" is not derivable
from the parser. ``uncovered`` exists so that the honest answer has a place to go: a
route whose refusal was reasoned from code but never executed stays visible as debt
instead of being silently counted as covered.

The classification is itself pinned: the exact protection set and the exact pin list of
every protection row are frozen below, and the detectors that consume them are tested
against deliberately unclassified and deliberately broken inputs. A hand-written
classification that no test defends is the same claim surface that failed three times.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
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


def _provider_hooks() -> set[str]:
    """Provider hook overrides — the write routes the runtime calls directly.

    ``sync_turn`` reaches the same deterministic id as the store and is enabled by a
    config flag, so it is a write route; it is invisible to both the dispatch parser and
    the CLI parser. Derived by comparing the provider class against its bases, so a new
    hook override appears here without anyone remembering to add it.
    """
    spec = importlib.util.spec_from_file_location("_route_inventory_plugin", REPO / "__init__.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    provider_cls = module.QdrantMemoryProvider
    own = {name for name, value in vars(provider_cls).items() if callable(value) and not name.startswith("_")}
    inherited: set[str] = set()
    for klass in provider_cls.__mro__[1:]:
        inherited |= {name for name, value in vars(klass).items() if callable(value) and not name.startswith("_")}
    hooks = {name for name in own if name in inherited}
    assert hooks, "no provider hook overrides could be derived"
    return hooks


def _surface() -> set[str]:
    return _tool_names() | _cli_commands() | _provider_hooks()


# --- the inventory ---------------------------------------------------------
#
# kind:
#   protection  a route that can delete or replace an existing payload; it must refuse
#               the protected shapes and its refusal must be pinned by a test that
#               exists in this tree
#   transition  the reviewed lineage path, which is allowed to write
#   read_only   never deletes or replaces a payload; may read, may set access metadata
#               on the points it returns, and may write local artifacts
#   dispatch    a router; every route it can reach is classified separately
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
    "tests/test_lineage.py::test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file",
    "tests/test_lineage.py::test_a_removed_file_with_one_protected_sibling_survives_force_too",
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
# `watcher run` only reaches a mutation through `--autonomy-mode guarded-auto`, which
# calls `apply_guarded_auto` → the apply path, so its refusal is pinned through that
# caller rather than through the tool it eventually invokes.
GUARDED_AUTO_PINS = [
    "tests/test_consolidation_apply.py::test_watcher_guarded_auto_refuses_a_lineage_managed_target",
]
SYNC_TURN_PINS = [
    "tests/test_consolidation_apply.py::test_the_sync_turn_hook_cannot_overwrite_a_protected_target",
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
         note="rewrites chunk payloads at content-derived ids and deletes stale ids, per file")
_declare(["qdrant_memory_consolidation_apply", "apply"], "protection", pins=CONSOLIDATION_PINS,
         note="merge/delete/quarantine; the managed check runs before the lock (N5)")
_declare(
    ["qdrant_learning_store", "learning store", "qdrant_learning_approve"], "protection",
    pins=LEARNING_STORE_PINS,
    note="no payload predicate: protected by the collection-separation invariant only",
)
_declare(["watcher run"], "protection", pins=GUARDED_AUTO_PINS,
         note="report-only by default; --guarded-auto reaches the apply path through apply_guarded_auto")
_declare(["qdrant_memory_index [capture]", "index [capture]"], "transition")
_declare(["qdrant_memory_index [reconcile]", "index [reconcile]"], "transition")

# --- provider hooks --------------------------------------------------------
# The runtime calls these directly, so none of them is reachable from the parsers above.

_declare(["sync_turn"], "protection", pins=SYNC_TURN_PINS,
         note="writes the turn through ConversationWriter at the store's deterministic id")
_declare(["handle_tool_call", "get_tool_schemas"], "dispatch",
         note="the router itself; every route it can reach is classified in this file")
_declare(
    [
        "initialize", "shutdown", "is_available", "system_prompt_block",
        "prefetch", "queue_prefetch", "on_session_switch",
    ], "read_only",
    note="config, client construction, in-process cache and session bookkeeping; no payload write",
)
_declare(["on_pre_compress", "on_session_end"], "read_only",
         note="collects extraction candidates into local artifacts; the store modes that would write are inert")

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
], "read_only",
    note="searches set access metadata on the points they return (set/merge, never delete or replace); "
         "watcher status/install and the report commands write local artifacts only")

# Writes outside Qdrant (files, reports, proposal state) with no payload to protect.
_declare(["backup create", "eval-capture"], "read_only",
         note="writes files only; the restore route that consumes them is the protection row")


# --- frozen classification -------------------------------------------------
# A hand-written classification that nothing defends is a claim surface. These maps are
# the contract: a row reclassified away from `protection`, or a pin list trimmed, fails
# here even though every individual route still looks fine.
#
# The pin lists in EXPECTED_PROTECTION_ROUTES are written out as literals on purpose.
# Referencing the constants above would compare each list with itself, so trimming a
# constant (or a row's `pins=`) would move both sides together and the frozen map would
# survive its own mutation. That is the defect this file exists to catch, one level up:
# a check that reads the same source twice is not a check. Because these are duplicates,
# adding a pin means updating both the constant and this literal.

# The derived surface — 25 tools, 50 CLI commands, 12 provider hooks. A route added or
# removed changes this number, which forces someone to touch this file and classify it,
# even if the completeness check in the test body is later weakened.
FROZEN_SURFACE_SIZE = 87

EXPECTED_PROTECTION_ROUTES = {
    "qdrant_memory_store": (
        "tests/test_consolidation_apply.py::test_store_refuses_to_overwrite_a_protected_point_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_refuses_a_history_only_dependent_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_fails_closed_when_the_target_cannot_be_read",
        "tests/test_consolidation_apply.py::test_store_answers_the_retirement_text_for_a_point_that_carries_both",
    ),
    "store": (
        "tests/test_consolidation_apply.py::test_store_refuses_to_overwrite_a_protected_point_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_refuses_a_history_only_dependent_in_every_mode",
        "tests/test_consolidation_apply.py::test_store_fails_closed_when_the_target_cannot_be_read",
        "tests/test_consolidation_apply.py::test_store_answers_the_retirement_text_for_a_point_that_carries_both",
    ),
    "qdrant_memory_forget": (
        "tests/test_consolidation_apply.py::test_structural_lineage_refusal_wording_is_caller_agnostic",
        "tests/test_consolidation_apply.py::test_a_demoted_dependent_stays_retirable",
        "tests/test_consolidation_apply.py::test_forget_refuses_any_dependent_and_deletes_all_clear_targets",
        "tests/test_consolidation_apply.py::test_forget_lock_contention_refuses_without_any_write_and_respects_timeout",
    ),
    "forget": (
        "tests/test_consolidation_apply.py::test_structural_lineage_refusal_wording_is_caller_agnostic",
        "tests/test_consolidation_apply.py::test_a_demoted_dependent_stays_retirable",
        "tests/test_consolidation_apply.py::test_forget_refuses_any_dependent_and_deletes_all_clear_targets",
        "tests/test_consolidation_apply.py::test_forget_lock_contention_refuses_without_any_write_and_respects_timeout",
    ),
    "qdrant_memory_extraction_approve": (
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_to_overwrite_a_protected_point",
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_a_history_only_dependent",
        "tests/test_consolidation_apply.py::test_extraction_approval_fails_closed_when_the_target_cannot_be_read",
    ),
    "learning approve": (
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_to_overwrite_a_protected_point",
        "tests/test_consolidation_apply.py::test_extraction_approval_refuses_a_history_only_dependent",
        "tests/test_consolidation_apply.py::test_extraction_approval_fails_closed_when_the_target_cannot_be_read",
    ),
    "restore": (
        "tests/test_backup_cli.py::test_restore_refuses_to_overwrite_every_protected_payload",
        "tests/test_backup_cli.py::test_restore_refuses_a_protected_target_in_the_learnings_scope",
        "tests/test_backup_cli.py::test_restore_refuses_the_review_only_shape",
        "tests/test_backup_cli.py::test_restore_still_overwrites_an_ordinary_payload",
        "tests/test_backup_cli.py::test_restore_refuses_missing_chunk_retired_by_committed_event",
    ),
    "qdrant_memory_index [off]": (
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state",
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field",
        "tests/test_lineage.py::test_off_mode_refuses_to_destroy_or_duplicate_captured_file",
        "tests/test_lineage.py::test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks",
        "tests/test_lineage.py::test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file",
        "tests/test_lineage.py::test_a_removed_file_with_one_protected_sibling_survives_force_too",
        "tests/test_lineage.py::test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs",
    ),
    "index [off]": (
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state",
        "tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_one_identity_field",
        "tests/test_lineage.py::test_off_mode_refuses_to_destroy_or_duplicate_captured_file",
        "tests/test_lineage.py::test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks",
        "tests/test_lineage.py::test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file",
        "tests/test_lineage.py::test_a_removed_file_with_one_protected_sibling_survives_force_too",
        "tests/test_lineage.py::test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs",
    ),
    "qdrant_memory_consolidation_apply": (
        "tests/test_consolidation_apply.py::test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile",
    ),
    "apply": (
        "tests/test_consolidation_apply.py::test_destructive_consolidation_refuses_lineage_dependents_outside_reconcile",
    ),
    "qdrant_learning_store": (
        "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
    ),
    "learning store": (
        "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
    ),
    "qdrant_learning_approve": (
        "tests/test_config_schema_scoring.py::test_config_rejects_one_collection_for_memories_and_learnings",
    ),
    "watcher run": (
        "tests/test_consolidation_apply.py::test_watcher_guarded_auto_refuses_a_lineage_managed_target",
    ),
    "sync_turn": (
        "tests/test_consolidation_apply.py::test_the_sync_turn_hook_cannot_overwrite_a_protected_target",
    ),
}


# --- assertions ------------------------------------------------------------


# The `[mode]` suffix exists for the routes whose behaviour depends on the lineage mode,
# and for those only. A `forget [off]` row would otherwise classify the base `forget`
# route while describing a mode it does not have, which is the same claim-surface hole
# one level up: the row exists, the label is wrong, and completeness says fine.
_MODE_SCOPED = ("qdrant_memory_index", "index")
_MODES = ("off", "capture", "reconcile")


def _base(entry: str) -> str:
    return entry.split(" [", 1)[0]


def _classified_by(name: str, inventory) -> bool:
    if name in inventory:
        return True
    if name not in _MODE_SCOPED:
        return False
    return any(f"{name} [{mode}]" in inventory for mode in _MODES)


def _unclassified(derived: set[str], inventory=None) -> list[str]:
    inventory = ROUTE_INVENTORY if inventory is None else inventory
    return sorted(name for name in derived if not _classified_by(name, inventory))


def _unresolved_pins(inventory=None, repo: Path | None = None) -> list[str]:
    inventory = ROUTE_INVENTORY if inventory is None else inventory
    repo = REPO if repo is None else repo
    unresolved: list[str] = []
    for entry, row in sorted(inventory.items()):
        if row["kind"] != "protection":
            continue
        for node in row["pins"]:  # type: ignore[union-attr]
            path, _, name = node.partition("::")
            name = name.split("[", 1)[0]
            target = repo / path
            if not target.is_file() or not re.search(
                rf"^def {re.escape(name)}\(", target.read_text(encoding="utf-8"), re.M
            ):
                unresolved.append(f"{entry} -> {node}")
    return unresolved


def test_every_surface_entry_is_classified():
    """A new tool, command or hook must be classified before it can ship.

    The size pin below is deliberate: it makes a new route change a number that has to
    be updated by hand, so the gate keeps working even if someone later guts the
    ``missing`` computation in this body. The detector functions themselves are tested
    separately, against deliberately unclassified input.
    """
    derived = _surface()
    assert len(derived) == FROZEN_SURFACE_SIZE, (
        f"the derived surface changed from {FROZEN_SURFACE_SIZE} to {len(derived)} entries. "
        "If a route was added or removed on purpose, classify it in ROUTE_INVENTORY and "
        "update this pin; if not, find out why the derivation sees something new."
    )
    missing = _unclassified(derived)
    assert missing == [], (
        "these entry points are not classified in ROUTE_INVENTORY: "
        f"{missing}. Decide what each one does to a protected payload: protection "
        "(with pins), transition, read_only, dispatch, or uncovered with a reason."
    )


def test_the_completeness_detector_rejects_an_unclassified_entry():
    """The detector must fail when it should: this is what makes the row above a gate."""
    assert _unclassified({"qdrant_memory_newroute"}, ROUTE_INVENTORY) == ["qdrant_memory_newroute"]
    assert _unclassified({"qdrant_memory_store"}, ROUTE_INVENTORY) == []
    assert _unclassified({"index"}, ROUTE_INVENTORY) == []


def test_a_mode_suffix_only_classifies_a_mode_scoped_route():
    """`forget [off]` must not stand in for `forget`."""
    suffix_only = {"forget [off]": {"kind": "read_only", "pins": [], "note": "", "reason": ""}}
    assert _unclassified({"forget"}, suffix_only) == ["forget"]
    index_suffix = {
        key: {"kind": "protection" if "[" in key else "read_only", "pins": [], "note": "", "reason": ""}
        for key in ("index [off]", "index [capture]", "index [reconcile]", "forget")
    }
    assert _unclassified({"index", "forget"}, index_suffix) == []


def test_no_inventory_row_is_stale():
    derived = _surface()
    stale = sorted(
        key for key in ROUTE_INVENTORY
        if _base(key) not in derived
        or (" [" in key and _base(key) not in _MODE_SCOPED)
    )
    assert stale == [], f"ROUTE_INVENTORY rows that no longer name a real entry point: {stale}"


def test_every_protection_route_names_a_pin_that_exists():
    """The route table may not claim a pin that is not there.

    ``docs/SAFETY.md`` said "pinned" three times for rows no test covered. A claim is
    only as good as its resolvable node id, so this resolves them.
    """
    unresolved = _unresolved_pins()
    assert unresolved == [], f"protection routes naming a pin that does not exist: {unresolved}"


def test_the_pin_resolver_rejects_a_pin_that_does_not_exist():
    """The resolver must fail when it should, including on a renamed node."""
    broken = {"route": {"kind": "protection", "pins": ["tests/test_lineage.py::test_not_a_real_node"], "note": "", "reason": ""}}
    assert _unresolved_pins(broken) == ["route -> tests/test_lineage.py::test_not_a_real_node"]
    renamed = {"route": {"kind": "protection", "pins": ["tests/test_lineage.py::test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state"], "note": "", "reason": ""}}
    assert _unresolved_pins(renamed) == []


def test_every_uncovered_route_declares_a_reason():
    missing = sorted(
        entry for entry, row in ROUTE_INVENTORY.items()
        if row["kind"] == "uncovered" and not str(row["reason"]).strip()
    )
    assert missing == [], f"uncovered routes without a reason: {missing}"


def test_a_route_is_classified_once_with_one_kind():
    kinds = {"protection", "transition", "read_only", "dispatch", "uncovered"}
    bad = {entry: row["kind"] for entry, row in ROUTE_INVENTORY.items() if row["kind"] not in kinds}
    assert bad == {}, f"unknown kinds: {bad}"


def test_the_protection_set_is_exactly_this():
    """Frozen: reclassifying any of these away from `protection` fails here.

    Covers the four routes four reviews opened (store, extraction approval, restore, the
    off-mode reindex), the two the plan opened (forget, destructive consolidation), the
    learning store's collection invariant, the guarded-auto caller, and the hook writer.
    """
    protected = {entry: tuple(row["pins"]) for entry, row in ROUTE_INVENTORY.items() if row["kind"] == "protection"}
    assert protected == EXPECTED_PROTECTION_ROUTES


def test_every_protection_row_keeps_its_exact_pin_list():
    """Frozen: trimming a pin list to one entry fails here.

    A list that is only checked for "some pins exist and they resolve" survives
    mutation; the point of the list is that every claim in it is enumerated.
    """
    trimmed = {
        entry: row["pins"] for entry, row in ROUTE_INVENTORY.items()
        if row["kind"] == "protection" and list(row["pins"]) != list(EXPECTED_PROTECTION_ROUTES.get(entry, []))  # type: ignore[arg-type]
    }
    assert trimmed == {}, f"protection rows whose pin list drifted: {sorted(trimmed)}"
