from __future__ import annotations

import json

import pytest

from qdrant_memory.recipes import get_recipe, list_recipe_names, recipe_catalog


REQUIRED_RECIPES = (
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

COMPACT_SEARCH_TOOLS = {"qdrant_memory_search", "qdrant_learning_search"}
DEEP_RECALL_TOOLS = {
    "qdrant_memory_inspect",
    "qdrant_memory_trace",
    "qdrant_memory_expand",
    "qdrant_memory_source_status",
}
REQUIRED_STATUS_FILTER_KEYS = {"canonical", "stale", "requires_review", "fact_status"}
REQUIRED_BUDGET_KEYS = {"search_top_k", "inspect_points", "trace_points", "expand_points", "expand_max_chars"}


def test_all_required_recipe_names_exist():
    names = list_recipe_names()

    assert isinstance(names, tuple)
    assert names == REQUIRED_RECIPES
    assert set(recipe_catalog()) == set(REQUIRED_RECIPES)
    for name in REQUIRED_RECIPES:
        assert get_recipe(name)["name"] == name


def test_each_recipe_declares_auditable_retrieval_plan_fields():
    for name in REQUIRED_RECIPES:
        recipe = get_recipe(name)

        assert recipe["authority"] == "retrieval_plan_only"
        assert recipe["collections"]
        assert set(recipe["collections"]) <= {"memory", "learning"}
        assert isinstance(recipe["filters"], dict)
        assert recipe["filters"]
        assert isinstance(recipe["status_filters"], dict)
        assert REQUIRED_STATUS_FILTER_KEYS <= set(recipe["status_filters"])
        assert isinstance(recipe["expansion_budget"], dict)
        assert REQUIRED_BUDGET_KEYS <= set(recipe["expansion_budget"])
        for budget_key in REQUIRED_BUDGET_KEYS:
            assert isinstance(recipe["expansion_budget"][budget_key], int)
            assert recipe["expansion_budget"][budget_key] > 0

        steps = recipe["steps"]
        assert isinstance(steps, list)
        assert steps
        assert steps[0]["stage"] == "compact_recall"
        assert steps[0]["tool"] in COMPACT_SEARCH_TOOLS
        assert steps[0]["when"] == "always"
        for index, step in enumerate(steps):
            assert step["tool"]
            assert step["purpose"]
            if step["tool"] in DEEP_RECALL_TOOLS:
                assert index > 0
                assert step["when"] != "always"


def test_recipe_results_are_copy_safe_and_serializable():
    recipe = get_recipe("source_backed_answer")
    catalog = recipe_catalog()

    json.dumps(recipe, sort_keys=True)
    json.dumps(catalog, sort_keys=True)

    recipe["collections"].append("mutated")
    recipe["filters"].setdefault("memory_kind", []).append("mutated")
    recipe["steps"][0]["tool"] = "mutated"
    catalog["source_backed_answer"]["status_filters"]["stale"] = "mutated"

    fresh = get_recipe("source_backed_answer")
    assert "mutated" not in fresh["collections"]
    assert "mutated" not in fresh["filters"].get("memory_kind", [])
    assert fresh["steps"][0]["tool"] in COMPACT_SEARCH_TOOLS
    assert fresh["status_filters"]["stale"] != "mutated"


def test_unknown_recipe_names_fail_clearly():
    with pytest.raises(KeyError, match="unknown recall recipe: missing_recipe"):
        get_recipe("missing_recipe")
