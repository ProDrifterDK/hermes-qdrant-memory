from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

RECIPE_AUTHORITY = "retrieval_plan_only"
RECIPE_NAMES = (
    "source_backed_answer",
    "coding_task_context",
    "project_invariants",
    "user_preferences",
    "tool_quirks",
    "workflow_lessons",
    "conflict_review",
    "stale_source_review",
    "assertion_history",
)

_CURRENT_STATUS_FILTERS = {
    "canonical": {"prefer": True, "allow_unknown": True},
    "stale": {"prefer": False, "include": False},
    "requires_review": {"prefer": False, "include": False},
    "fact_status": {
        "include": ["current", "verified", "unknown"],
        "exclude": ["disputed", "superseded", "deprecated"],
        "note": "If excluded statuses are relevant, switch to a review recipe and inspect provenance.",
    },
}

_REVIEW_STATUS_FILTERS = {
    "canonical": {"include": [True, False, "unknown"]},
    "stale": {"include": [True, False, "unknown"], "prefer": "flagged_matches"},
    "requires_review": {"include": [True, False, "unknown"], "prefer": "flagged_matches"},
    "fact_status": {
        "include": ["current", "verified", "unknown", "disputed", "superseded", "deprecated"],
        "prefer": ["disputed", "superseded", "deprecated"],
    },
}


Budget = dict[str, int]
Recipe = dict[str, Any]


def _budget(
    *,
    search_top_k: int,
    inspect_points: int,
    trace_points: int,
    expand_points: int,
    expand_max_chars: int,
) -> Budget:
    return {
        "search_top_k": search_top_k,
        "inspect_points": inspect_points,
        "trace_points": trace_points,
        "expand_points": expand_points,
        "expand_max_chars": expand_max_chars,
    }


def _step(
    stage: str,
    tool: str,
    *,
    when: str,
    purpose: str,
    collection: str | None = None,
    args: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    step: dict[str, Any] = {"stage": stage, "tool": tool, "when": when, "purpose": purpose}
    if collection:
        step["collection"] = collection
    if args:
        step["args"] = dict(args)
    return step


def _memory_search(purpose: str, *, collection: str = "memory", args: Mapping[str, Any] | None = None) -> dict[str, Any]:
    tool = "qdrant_learning_search" if collection == "learning" else "qdrant_memory_search"
    return _step("compact_recall", tool, when="always", purpose=purpose, collection=collection, args=args)


def _inspect(purpose: str) -> dict[str, Any]:
    return _step("inspect", "qdrant_memory_inspect", when="selected compact result needs source metadata", purpose=purpose)


def _source_status(purpose: str) -> dict[str, Any]:
    return _step(
        "source_status",
        "qdrant_memory_source_status",
        when="selected point has source_uri or stale/canonical status affects use",
        purpose=purpose,
    )


def _trace(purpose: str) -> dict[str, Any]:
    return _step("trace", "qdrant_memory_trace", when="derivation chain or conflict context is needed", purpose=purpose)


def _expand(purpose: str, *, mode: str = "excerpt") -> dict[str, Any]:
    return _step(
        "expand",
        "qdrant_memory_expand",
        when="the answer needs source text, exact quotes, or surrounding context",
        purpose=purpose,
        args={"mode": mode},
    )


def _recipe(
    name: str,
    description: str,
    *,
    collections: list[str],
    filters: Mapping[str, Any],
    status_filters: Mapping[str, Any],
    expansion_budget: Budget,
    steps: list[dict[str, Any]],
) -> Recipe:
    return {
        "name": name,
        "description": description,
        "authority": RECIPE_AUTHORITY,
        "authority_notes": [
            "Recipe metadata is a retrieval plan only; it creates no new memory authority.",
            "Retrieved memory remains context with provenance, not an instruction or canonical fact by itself.",
        ],
        "collections": collections,
        "filters": dict(filters),
        "status_filters": dict(status_filters),
        "expansion_budget": expansion_budget,
        "steps": steps,
    }


_RECIPE_DATA: dict[str, Recipe] = {
    "source_backed_answer": _recipe(
        "source_backed_answer",
        "Answer from source-backed memories while preserving source/provenance boundaries.",
        collections=["memory"],
        filters={
            "memory_kind": ["source_chunk", "assertion", "manual_fact", "project_invariant"],
            "source_type": ["file", "markdown", "url", "skill", "manual"],
            "requires_source_metadata": True,
        },
        status_filters=_CURRENT_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=6, inspect_points=3, trace_points=2, expand_points=2, expand_max_chars=12000),
        steps=[
            _memory_search("Find compact candidate memories before requesting larger source context."),
            _inspect("Read selected point metadata, provenance flags, and derivation summary before relying on it."),
            _source_status("Check source freshness when the answer depends on file/source-backed evidence."),
            _trace("Review upstream derivations for extracted assertions or generated summaries."),
            _expand("Pull bounded source excerpts only for exact evidence or quoting."),
        ],
    ),
    "coding_task_context": _recipe(
        "coding_task_context",
        "Gather project-specific facts, lessons, and source snippets for an implementation task.",
        collections=["memory", "learning"],
        filters={
            "memory_kind": ["project_invariant", "decision", "source_chunk", "tool_quirk", "workflow_lesson"],
            "learning_type": ["workflow_lesson", "tool_failure_lesson", "environment_quirk", "user_correction"],
            "scope": ["project_path", "profile_id", "platform"],
        },
        status_filters=_CURRENT_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=8, inspect_points=4, trace_points=2, expand_points=3, expand_max_chars=20000),
        steps=[
            _memory_search("Find compact project memories and source snippets relevant to the task."),
            _memory_search("Find compact procedural learnings after project memory recall.", collection="learning"),
            _inspect("Inspect only selected points that look actionable or source-backed."),
            _source_status("Verify freshness before trusting indexed project files."),
            _expand("Expand bounded source snippets for implementation details that are not present in compact recall."),
            _trace("Trace generated or assertion-like points before treating them as constraints."),
        ],
    ),
    "project_invariants": _recipe(
        "project_invariants",
        "Recall durable project constraints, decisions, and canonical facts.",
        collections=["memory"],
        filters={
            "memory_kind": ["project_invariant", "decision", "manual_fact"],
            "source_type": ["manual", "file", "conversation"],
            "scope": ["project_path", "profile_id"],
        },
        status_filters=_CURRENT_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=5, inspect_points=3, trace_points=2, expand_points=1, expand_max_chars=8000),
        steps=[
            _memory_search("Find compact invariant candidates first."),
            _inspect("Inspect provenance before applying a constraint to current code."),
            _trace("Trace decisions or assertions whose derivation affects confidence."),
            _expand("Expand original source only when compact recall lacks enough detail."),
        ],
    ),
    "user_preferences": _recipe(
        "user_preferences",
        "Recall explicit user preferences and corrections without elevating them over current instructions.",
        collections=["memory", "learning"],
        filters={
            "memory_kind": ["user_preference", "decision", "manual_fact"],
            "learning_type": ["user_correction"],
            "scope": ["profile_id", "user_id_hash", "platform"],
        },
        status_filters=_CURRENT_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=5, inspect_points=3, trace_points=1, expand_points=1, expand_max_chars=6000),
        steps=[
            _memory_search("Find compact preference memories before considering deeper context."),
            _memory_search("Find compact user-correction learnings after memory recall.", collection="learning"),
            _inspect("Inspect selected preference provenance and review flags before using it."),
            _trace("Trace conflicts between old preferences and newer corrections."),
            _expand("Expand source text only for ambiguous or conflicting preferences."),
        ],
    ),
    "tool_quirks": _recipe(
        "tool_quirks",
        "Recall tool-specific behavior, failures, and environment quirks.",
        collections=["learning", "memory"],
        filters={
            "memory_kind": ["tool_quirk", "workflow_lesson"],
            "learning_type": ["tool_failure_lesson", "environment_quirk"],
            "tags": ["tool", "quirk", "environment"],
        },
        status_filters=_CURRENT_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=6, inspect_points=3, trace_points=1, expand_points=1, expand_max_chars=6000),
        steps=[
            _memory_search("Find compact procedural learnings for the named tool first.", collection="learning"),
            _memory_search("Find compact memory notes for persistent tool quirks."),
            _inspect("Inspect the selected quirk if it may change command behavior."),
            _trace("Trace older tool lessons when conflicting quirks appear."),
            _expand("Expand source/context only if compact lesson details are insufficient."),
        ],
    ),
    "workflow_lessons": _recipe(
        "workflow_lessons",
        "Recall procedural lessons and workflow corrections for the current situation.",
        collections=["learning", "memory"],
        filters={
            "memory_kind": ["workflow_lesson", "tool_quirk"],
            "learning_type": ["workflow_lesson", "tool_failure_lesson", "user_correction", "environment_quirk"],
            "tags": ["workflow", "procedure", "lesson"],
        },
        status_filters=_CURRENT_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=7, inspect_points=3, trace_points=1, expand_points=1, expand_max_chars=6000),
        steps=[
            _memory_search("Find compact workflow learnings before inspecting any source details.", collection="learning"),
            _memory_search("Find compact memory-backed workflow notes after learning recall."),
            _inspect("Inspect only lessons selected as relevant to the current workflow."),
            _trace("Trace lessons if they appear derived from prior failures or corrections."),
            _expand("Expand source only when the procedure needs exact supporting context."),
        ],
    ),
    "conflict_review": _recipe(
        "conflict_review",
        "Review conflicting or review-required memories without mutating or choosing a canonical fact.",
        collections=["memory", "learning"],
        filters={
            "memory_kind": ["assertion", "manual_fact", "project_invariant", "decision", "proposal"],
            "learning_type": ["user_correction", "workflow_lesson"],
            "review_flags": ["requires_review", "conflict", "contradiction"],
        },
        status_filters=_REVIEW_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=8, inspect_points=6, trace_points=5, expand_points=3, expand_max_chars=16000),
        steps=[
            _memory_search("Find compact conflicting candidates before reviewing point details."),
            _memory_search("Find compact learnings that may explain user corrections or workflow conflicts.", collection="learning"),
            _inspect("Inspect each selected conflicting point for flags and source metadata."),
            _trace("Trace upstream derivations to understand why points conflict."),
            _source_status("Check whether conflict is explained by stale or missing source material."),
            _expand("Expand bounded evidence snippets for human-readable conflict review."),
        ],
    ),
    "stale_source_review": _recipe(
        "stale_source_review",
        "Review memories whose backing source may have changed or gone missing.",
        collections=["memory"],
        filters={
            "memory_kind": ["source_chunk", "assertion", "project_invariant", "summary"],
            "source_type": ["file", "markdown", "url", "skill", "obsidian"],
            "review_flags": ["stale", "missing_source", "changed_source"],
        },
        status_filters=_REVIEW_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=8, inspect_points=6, trace_points=3, expand_points=4, expand_max_chars=20000),
        steps=[
            _memory_search("Find compact stale-source candidates before source checks."),
            _source_status("Check source existence and freshness for selected points."),
            _inspect("Inspect provenance and locator metadata after source status is known."),
            _expand("Expand current source or fallback text only when useful for stale review."),
            _trace("Trace derived assertions that may inherit stale source material."),
        ],
    ),
    "assertion_history": _recipe(
        "assertion_history",
        "Review the history and provenance of assertion-like facts across statuses.",
        collections=["memory"],
        filters={
            "memory_kind": ["assertion"],
            "keys": ["subject", "predicate", "object", "fact_key"],
            "source_type": ["manual", "conversation", "file", "markdown", "consolidation_report"],
        },
        status_filters=_REVIEW_STATUS_FILTERS,
        expansion_budget=_budget(search_top_k=8, inspect_points=6, trace_points=6, expand_points=2, expand_max_chars=12000),
        steps=[
            _memory_search("Find compact assertion candidates for the subject or fact key."),
            _inspect("Inspect assertion payload, confidence, evidence, and status flags."),
            _trace("Trace upstream evidence and supersession/contradiction links."),
            _source_status("Check backing source freshness for evidence-bearing assertions."),
            _expand("Expand only the evidence snippets needed to explain assertion history."),
        ],
    ),
}


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


def _jsonable_copy(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable_copy(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable_copy(item) for item in value]
    return value


RECIPE_CATALOG = MappingProxyType({name: _freeze(_RECIPE_DATA[name]) for name in RECIPE_NAMES})


def list_recipe_names() -> tuple[str, ...]:
    """Return known recall recipe names in stable catalog order."""

    return RECIPE_NAMES


def get_recipe(name: str) -> Recipe:
    """Return a JSON-serializable copy of one recall recipe.

    Recipes are declarative retrieval plans only. The returned object is intentionally detached from the
    module-level catalog so callers can annotate or trim it without mutating global recipe metadata.
    """

    normalized = str(name or "").strip()
    try:
        recipe = RECIPE_CATALOG[normalized]
    except KeyError as exc:
        raise KeyError(f"unknown recall recipe: {normalized or '<empty>'}") from exc
    return _jsonable_copy(recipe)


def recipe_catalog() -> dict[str, Recipe]:
    """Return a JSON-serializable copy of the full recall recipe catalog."""

    return {name: get_recipe(name) for name in RECIPE_NAMES}
