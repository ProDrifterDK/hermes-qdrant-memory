"""Tests for the Phase 6A offline retrieval evaluator.

The evaluator is stdlib-only and must never contact Qdrant. The
tests in this file build local in-memory dicts/JSONL strings instead
of relying on captured packets, so a regression in scoring is
detectable without any live harness.
"""

from __future__ import annotations

import json
import math

import pytest

from qdrant_memory import evaluation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _hybrid_packet(
    *,
    with_exact_hit: bool = True,
    with_summary: bool = True,
    with_leaf: bool = True,
    with_graph: bool = True,
) -> dict:
    """Build a Phase 5 ``qdrant_memory_retrieve`` packet.

    Layout matches the documented ``results.exact_hits`` /
    ``summaries`` / ``cited_leaves`` / ``graph_relations`` shape.
    """

    results: dict = {
        "exact_hits": [],
        "summaries": [],
        "cited_leaves": [],
        "graph_relations": [],
    }
    if with_exact_hit:
        results["exact_hits"].append(
            {
                "point_id": "pt-exact-1",
                "text": "phase 5 hybrid dense+sparse exact hit body",
                "file_path": "docs/RAPTOR.md",
                "source_uri": "file://docs/RAPTOR.md",
                "score": 0.91,
            }
        )
        results["exact_hits"].append(
            {
                "point_id": "pt-exact-2",
                "text": "second dense hit without expected labels",
                "file_path": "docs/UNRELATED.md",
                "source_uri": "file://docs/UNRELATED.md",
                "score": 0.5,
            }
        )
    if with_summary:
        results["summaries"].append(
            {
                "point_id": "pt-summary-1",
                "raptor_node_id": "rnode-1",
                "raptor_root_id": "rroot-1",
                "raptor_tree_id": "rtree-1",
                "raptor_build_id": "rbuild-1",
                "raptor_cluster_id": "rcluster-1",
                "raptor_level": 1,
                "text": "summary covering phase 5 contract",
                "file_path": "docs/RAPTOR.md",
            }
        )
    if with_leaf:
        results["cited_leaves"].append(
            {
                "point_id": "pt-leaf-1",
                "parent_raptor_node_id": "rnode-1",
                "parent_point_id": "pt-summary-1",
                "text": "leaf body that quotes the phase 5 spec",
                "file_path": "docs/RAPTOR.md",
                "source_uri": "file://docs/RAPTOR.md",
            }
        )
    if with_graph:
        results["graph_relations"].append(
            {
                "point_id": "pt-graph-1",
                "text": "graph relation describing the phase 5 lane",
                "file_path": "docs/RAPTOR.md",
            }
        )
    return {"results": results}


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def test_parse_jsonl_text_skips_blank_and_comment_lines():
    text = (
        "# header comment\n"
        "\n"
        "  \n"
        '{"case_id": "c1", "query": "q1"}\n'
        "# inline comment\n"
        '{"case_id": "c2", "query": "q2"}\n'
    )
    rows = evaluation.parse_jsonl_text(text)
    assert [row["case_id"] for row in rows] == ["c1", "c2"]


def test_parse_jsonl_text_reports_line_numbers_for_invalid_json():
    text = (
        '{"case_id": "c1", "query": "q1"}\n'
        "not-json\n"
        '{"case_id": "c3", "query": "q3"}\n'
    )
    with pytest.raises(evaluation.EvaluationError) as excinfo:
        evaluation.parse_jsonl_text(text)
    assert "line 2" in str(excinfo.value)


def test_load_jsonl_reads_local_file(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"case_id": "c1", "query": "q"}\n'
        '{"case_id": "c2", "query": "q"}\n',
        encoding="utf-8",
    )
    rows = evaluation.load_jsonl(str(path))
    assert [row["case_id"] for row in rows] == ["c1", "c2"]


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------


def test_normalize_packet_flattens_phase5_grouped_packet_in_lane_order():
    packet = _hybrid_packet()
    candidates = evaluation.normalize_packet(packet)
    lanes = [cand.lane for cand in candidates]
    assert lanes == [
        "exact_hits",
        "exact_hits",
        "summaries",
        "cited_leaves",
        "graph_relations",
    ]


def test_normalize_packet_does_not_mutate_input():
    packet = _hybrid_packet()
    snapshot = json.dumps(packet, sort_keys=True)
    evaluation.normalize_packet(packet)
    evaluation.normalize_packet(packet)
    assert json.dumps(packet, sort_keys=True) == snapshot


def test_normalize_packet_handles_legacy_list_shape():
    packet = [
        {"id": "a", "text": "alpha body", "source_uri": "file://a.md"},
        {"id": "b", "text": "bravo body", "file_path": "/x/b.md"},
    ]
    candidates = evaluation.normalize_packet(packet)
    assert [cand.lane for cand in candidates] == ["legacy", "legacy"]
    assert [cand.point_id for cand in candidates] == ["a", "b"]


def test_normalize_packet_handles_dict_of_lists_shape():
    packet = {
        "exact_hits": [{"id": "x1", "text": "x"}],
        "summaries": [{"id": "s1", "text": "s"}],
    }
    candidates = evaluation.normalize_packet(packet)
    assert {cand.lane for cand in candidates} == {"exact_hits", "summaries"}


def test_normalize_packet_mixed_known_and_unknown_top_level_lanes():
    """A top-level packet that mixes known ``_LANE_ORDER`` lanes with
    additional unknown list buckets must keep the known lanes' lane
    label AND surface the unknown buckets as ``legacy`` candidates so
    that poison content embedded in the unknown lanes is not dropped.

    Regression for Phase 6A fix3 P2-1.
    """

    packet = {
        "exact_hits": [{"point_id": "ok", "text": "clean body"}],
        "extra_lane": [
            {"point_id": "poison", "text": "contains forbidden-token"}
        ],
    }
    candidates = evaluation.normalize_packet(packet)
    # Two candidates: one from the known ``exact_hits`` lane and one
    # from the unknown ``extra_lane`` bucket projected as ``legacy``.
    assert len(candidates) == 2
    by_lane = {cand.lane: cand for cand in candidates}
    assert "exact_hits" in by_lane
    assert by_lane["exact_hits"].point_id == "ok"
    assert "legacy" in by_lane
    assert by_lane["legacy"].point_id == "poison"
    # And the unknown bucket must contribute to scoring: forbidden
    # term embedded in it must trip ``wrong_memory`` and inflate the
    # ``emitted_count``.
    case = {
        "case_id": "c1",
        "query": "any",
        "forbidden_terms": ["forbidden-token"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": packet,
    }
    row = evaluation.score_case_run(case, run)
    assert row["wrong_memory"] is True, row
    assert "forbidden-token" in row["wrong_reasons"].get("forbidden_terms", [])
    assert row["emitted_count"] == 2


def test_normalize_packet_unknown_top_level_lane_not_duplicated_as_known():
    """A top-level key that is NOT in ``_LANE_ORDER`` must surface its
    candidates exactly once (legacy), even when the same packet also
    contains well-known lane buckets.
    """

    packet = {
        "exact_hits": [{"point_id": "e1", "text": "exact body"}],
        "summaries": [{"point_id": "s1", "text": "summary body"}],
        "extra_lane": [
            {"point_id": "x1", "text": "alpha"},
            {"point_id": "x2", "text": "bravo"},
        ],
    }
    candidates = evaluation.normalize_packet(packet)
    # Two known + two legacy = four total, no duplicates.
    assert len(candidates) == 4
    lanes = [cand.lane for cand in candidates]
    assert lanes.count("exact_hits") == 1
    assert lanes.count("summaries") == 1
    assert lanes.count("legacy") == 2
    # And the legacy bucket's candidates keep the unknown key out of
    # the well-known lane counts: no candidate should be in
    # ``exact_hits`` AND have point_id from ``extra_lane``.
    for cand in candidates:
        if cand.lane == "legacy":
            assert cand.point_id in {"x1", "x2"}
        else:
            assert cand.point_id in {"e1", "s1"}


# ---------------------------------------------------------------------------
# Scoring: hits and source/term matches
# ---------------------------------------------------------------------------


def test_score_case_run_hybrid_packet_hits_point_source_and_term():
    case = {
        "case_id": "c1",
        "query": "where is the raptor retrieve contract",
        "expected_point_ids": ["pt-exact-1"],
        "expected_source_uris": ["file://docs/RAPTOR.md"],
        "expected_terms": ["phase 5"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "latency_ms": 312,
        "packet": _hybrid_packet(),
    }
    row = evaluation.score_case_run(case, run)
    assert row["errored"] is False
    assert row["hit_at_k"] is True
    assert row["source_hit_at_k"] is True
    assert row["exact_identifier_hit"] is True
    assert row["wrong_memory"] is False
    assert "pt-exact-1" in row["matched_expected"]["point_ids"]
    assert "file://docs/RAPTOR.md" in row["matched_expected"]["sources"]
    assert "phase 5" in row["matched_expected"]["terms"]
    # Three distinct useful handles: the expected point id, the
    # expected source URI on the same point, and the expected term
    # ``phase 5`` (which the exact hit text also contains). The
    # summary and leaf emit the same expected term but the term
    # entry is deduped in the useful_topk set.
    assert row["useful_topk_count"] == 3
    assert row["latency_budget_met"] is True
    assert row["zoom_efficiency"] > 0


def test_score_case_run_exact_identifier_is_null_when_no_expected_ids():
    case = {
        "case_id": "c1",
        "query": "any",
        "expected_terms": ["phase 5"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": _hybrid_packet(),
    }
    row = evaluation.score_case_run(case, run)
    assert row["exact_identifier_hit"] is None
    assert row["hit_at_k"] is True


# ---------------------------------------------------------------------------
# Scoring: wrong-memory (poison) detection
# ---------------------------------------------------------------------------


def test_score_case_run_wrong_memory_for_forbidden_terms_anywhere_in_packet():
    case = {
        "case_id": "c1",
        "query": "any",
        "expected_file_paths": ["docs/RAPTOR.md"],
        "forbidden_terms": ["never-emit-this"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "p1",
                        "text": "first body",
                        "file_path": "docs/RAPTOR.md",
                    },
                    {
                        "point_id": "p2",
                        "text": "second body that contains never-emit-this",
                        "file_path": "docs/OTHER.md",
                    },
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["hit_at_k"] is True
    assert row["wrong_memory"] is True
    assert "forbidden_terms" in row["wrong_reasons"]
    assert "never-emit-this" in row["wrong_reasons"]["forbidden_terms"]


def test_score_case_run_wrong_memory_for_forbidden_point_id():
    case = {
        "case_id": "c1",
        "query": "any",
        "forbidden_point_ids": ["poison-1"],
    }
    run = {
        "case_id": "c1",
        "variant": "raptor-only",
        "packet": {
            "results": {
                "exact_hits": [],
                "summaries": [{"point_id": "ok-1", "text": "clean"}],
                "cited_leaves": [
                    {
                        "point_id": "poison-1",
                        "text": "poison leaf",
                        "file_path": "x",
                    }
                ],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["wrong_memory"] is True
    assert "poison-1" in row["wrong_reasons"]["forbidden_point_ids"]


def test_score_case_run_wrong_memory_for_forbidden_source_uri():
    case = {
        "case_id": "c1",
        "query": "any",
        "forbidden_source_uris": ["file://secrets/keys.md"],
    }
    run = {
        "case_id": "c1",
        "variant": "dense-only",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "x1",
                        "text": "clean body",
                        "source_uri": "file://secrets/keys.md",
                    }
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["wrong_memory"] is True
    assert "file://secrets/keys.md" in row["wrong_reasons"]["forbidden_source_uris"]


def test_score_case_run_wrong_memory_inspects_full_packet_not_only_topk():
    """A poison candidate outside the top-k must still trip the check."""

    case = {
        "case_id": "c1",
        "query": "any",
        "forbidden_terms": ["forbidden-poison"],
    }
    # top_k=2 will only see the first two exact hits. The poison
    # candidate is in the third slot. We rely on the eval rule that
    # wrong_memory inspects all normalized items, not just top-k.
    case_top_k = 2
    run = {
        "case_id": "c1",
        "variant": "dense-only",
        "packet": {
            "results": {
                "exact_hits": [
                    {"point_id": "p1", "text": "first"},
                    {"point_id": "p2", "text": "second"},
                    {"point_id": "p3", "text": "third contains forbidden-poison"},
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run, top_k=case_top_k)
    assert row["wrong_memory"] is True


# ---------------------------------------------------------------------------
# Scoring: errored run rows
# ---------------------------------------------------------------------------


def test_score_case_run_errored_row_has_null_metrics_and_counts_in_error():
    case = {"case_id": "c1", "query": "q"}
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "error": "retrieval timed out",
        "packet": {"results": {}},
    }
    row = evaluation.score_case_run(case, run)
    assert row["errored"] is True
    assert row["hit_at_k"] is None
    assert row["source_hit_at_k"] is None
    assert row["exact_identifier_hit"] is None
    assert row["wrong_memory"] is None


def test_score_case_run_errored_row_redacts_raw_error_text():
    """Phase 6A fix3 P2-2: errored rows must NOT echo the raw captured
    error string. The default row payload uses a constant sentinel
    plus operational flags so a downstream report (which is persisted
    to disk via ``--json``) cannot leak raw queries, Qdrant
    request/response details, or packet snippets.
    """

    case = {"case_id": "c1", "query": "q"}
    raw_error = (
        "query=what is alan private packet=SECRET_SNIPPET "
        "endpoint=http://qdrant.internal:6333"
    )
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "error": raw_error,
        "packet": {"results": {}},
    }
    row = evaluation.score_case_run(case, run)
    # Operational signal is preserved.
    assert row["errored"] is True
    assert row.get("error_present") is True
    assert row.get("error_redacted") is True
    # The error slot is a constant sentinel, not the raw text.
    assert row["error"] == "<redacted>"
    # Numeric metrics stay null, same as before fix3.
    assert row["hit_at_k"] is None
    assert row["wrong_memory"] is None
    # And the JSON dump of the row carries none of the raw substrings
    # from the captured error.
    dumped = json.dumps(row)
    for needle in (
        "what is alan private",
        "SECRET_SNIPPET",
        "http://qdrant.internal:6333",
        raw_error,
    ):
        assert needle not in dumped, (
            f"raw error fragment leaked into row JSON: {needle!r}"
        )


def test_score_case_run_non_errored_row_has_no_error_redaction_flags():
    """Sanity: only errored rows carry the redaction flags. A clean row
    must not pretend its ``error`` slot was redacted.
    """

    case = {"case_id": "c1", "query": "q"}
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [{"point_id": "p1", "text": "body"}],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["errored"] is False
    assert "error_present" not in row
    assert "error_redacted" not in row
    assert "error" not in row


# ---------------------------------------------------------------------------
# Lane-aware top-k
# ---------------------------------------------------------------------------


def test_lane_aware_top_k_takes_per_lane_first_k_and_unions():
    candidates = evaluation.normalize_packet(
        {
            "results": {
                "exact_hits": [
                    {"point_id": "e1", "text": "e1"},
                    {"point_id": "e2", "text": "e2"},
                ],
                "summaries": [
                    {"point_id": "s1", "text": "s1"},
                    {"point_id": "s2", "text": "s2"},
                ],
                "cited_leaves": [
                    {"point_id": "l1", "text": "l1"},
                ],
                "graph_relations": [],
            }
        }
    )
    topk = evaluation._lane_aware_top_k(candidates, top_k=1)
    lanes = [cand.lane for cand in topk]
    # first k of each lane, unioned
    assert lanes == ["exact_hits", "summaries", "cited_leaves"]
    assert [cand.point_id for cand in topk] == ["e1", "s1", "l1"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def test_aggregate_scores_computes_rates_median_and_p95_per_variant():
    rows = [
        {
            "case_id": "c1", "variant": "hybrid", "errored": False,
            "hit_at_k": True, "source_hit_at_k": True,
            "exact_identifier_hit": True, "wrong_memory": False,
            "context_chars": 100, "zoom_efficiency": 0.5,
            "latency_ms": 200.0, "latency_budget_met": True,
        },
        {
            "case_id": "c2", "variant": "hybrid", "errored": False,
            "hit_at_k": False, "source_hit_at_k": False,
            "exact_identifier_hit": False, "wrong_memory": True,
            "context_chars": 200, "zoom_efficiency": 0.1,
            "latency_ms": 400.0, "latency_budget_met": True,
        },
        {
            "case_id": "c3", "variant": "hybrid", "errored": True,
            "hit_at_k": None, "source_hit_at_k": None,
            "exact_identifier_hit": None, "wrong_memory": None,
            "context_chars": 0, "zoom_efficiency": 0.0,
            "latency_ms": None, "latency_budget_met": None,
        },
        {
            # Term-only case: no expected source/file labels, so
            # source_hit_at_k is None under the Phase 6A labeled
            # contract. exact_identifier_hit is also None.
            "case_id": "c4", "variant": "raptor-only", "errored": False,
            "hit_at_k": True, "source_hit_at_k": None,
            "exact_identifier_hit": None, "wrong_memory": False,
            "context_chars": 50, "zoom_efficiency": 0.2,
            "latency_ms": 100.0, "latency_budget_met": True,
        },
    ]
    aggregates = evaluation.aggregate_scores(rows)
    assert set(aggregates) == {"hybrid", "raptor-only"}
    hybrid = aggregates["hybrid"]
    assert hybrid["case_count"] == 3
    assert hybrid["errored_count"] == 1
    assert hybrid["scored_count"] == 2
    assert hybrid["hit_at_k_rate"] == 50.0  # 1 of 2 non-errored
    assert hybrid["source_hit_at_k_rate"] == 50.0  # 1 of 2 source-labeled
    assert hybrid["source_hit_labeled_count"] == 2
    assert hybrid["exact_identifier_hit_rate"] == 50.0
    assert hybrid["exact_identifier_labeled_count"] == 2
    assert hybrid["wrong_memory_rate"] == 50.0
    assert hybrid["avg_context_chars"] == 150.0
    assert math.isclose(hybrid["avg_zoom_efficiency"], 0.3, rel_tol=1e-6)
    # Latency: 200, 400 -> median 300, p95 between 200 and 400.
    assert hybrid["latency_ms_median"] == 300.0
    assert hybrid["latency_ms_p95"] is not None
    assert 200.0 <= hybrid["latency_ms_p95"] <= 400.0
    assert hybrid["latency_budget_pass_rate"] == 100.0

    raptor = aggregates["raptor-only"]
    assert raptor["case_count"] == 1
    assert raptor["hit_at_k_rate"] == 100.0
    assert raptor["exact_identifier_hit_rate"] is None  # never had expected ids
    assert raptor["exact_identifier_labeled_count"] == 0
    # Term-only case contributes zero to the source-labeled rate;
    # the rate is None when no row carried source/file labels.
    assert raptor["source_hit_at_k_rate"] is None
    assert raptor["source_hit_labeled_count"] == 0


def test_aggregate_scores_returns_empty_when_no_rows():
    assert evaluation.aggregate_scores([]) == {}


# ---------------------------------------------------------------------------
# End-to-end evaluate()
# ---------------------------------------------------------------------------


def test_evaluate_loads_jsonl_and_emits_report(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    cases_path.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "query": "q",
                "expected_terms": ["phase 5"],
                "forbidden_terms": ["never-emit"],
            }
        )
        + "\n"
        + json.dumps({"case_id": "c2", "query": "q2", "expected_terms": ["other"]})
        + "\n",
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps(
            {
                "case_id": "c1",
                "variant": "hybrid",
                "latency_ms": 200,
                "packet": {
                    "results": {
                        "exact_hits": [{"point_id": "p1", "text": "phase 5 body"}],
                        "summaries": [],
                        "cited_leaves": [],
                        "graph_relations": [],
                    }
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "c1",
                "variant": "raptor-only",
                "latency_ms": 800,
                "packet": {
                    "results": {
                        "summaries": [
                            {
                                "point_id": "s1",
                                "text": "phase 5 summary with never-emit",
                            }
                        ],
                        "exact_hits": [],
                        "cited_leaves": [],
                        "graph_relations": [],
                    }
                },
            }
        )
        + "\n"
        + json.dumps(
            {
                "case_id": "missing-case",
                "variant": "hybrid",
                "packet": {"results": {"exact_hits": []}},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    report = evaluation.evaluate(str(cases_path), str(runs_path))
    assert report["summary"]["totals"]["case_count"] == 2
    assert report["summary"]["totals"]["run_count"] == 3
    assert report["summary"]["totals"]["runs_without_case_count"] == 1
    assert report["summary"]["totals"]["scored_count"] == 2
    hybrid_agg = report["summary"]["variants"]["hybrid"]
    raptor_agg = report["summary"]["variants"]["raptor-only"]
    assert hybrid_agg["hit_at_k_rate"] == 100.0
    assert hybrid_agg["wrong_memory_rate"] == 0.0
    assert raptor_agg["wrong_memory_rate"] == 100.0
    # The report must NOT include the raw packet/query text.
    for row in report["rows"]:
        assert "packet" not in row
        assert "query" not in row


def test_evaluate_raises_on_invalid_run_jsonl_with_line_number(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    cases_path.write_text(
        json.dumps({"case_id": "c1", "query": "q"}) + "\n",
        encoding="utf-8",
    )
    runs_path.write_text(
        json.dumps({"case_id": "c1", "variant": "hybrid", "packet": {}})
        + "\n"
        + "{not-json"
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(evaluation.EvaluationError) as excinfo:
        evaluation.evaluate(str(cases_path), str(runs_path))
    assert "line 2" in str(excinfo.value)


def test_evaluate_raises_on_duplicate_case_id(tmp_path):
    cases_path = tmp_path / "cases.jsonl"
    runs_path = tmp_path / "runs.jsonl"
    cases_path.write_text(
        json.dumps({"case_id": "c1", "query": "q1"})
        + "\n"
        + json.dumps({"case_id": "c1", "query": "q1b"})
        + "\n",
        encoding="utf-8",
    )
    runs_path.write_text("", encoding="utf-8")
    with pytest.raises(evaluation.EvaluationError) as excinfo:
        evaluation.evaluate(str(cases_path), str(runs_path))
    assert "duplicate case_id" in str(excinfo.value)


def test_evaluate_rejects_top_k_zero():
    with pytest.raises(evaluation.EvaluationError):
        evaluation.evaluate("/dev/null", "/dev/null", top_k=0)


# ---------------------------------------------------------------------------
# Imports / provider guard
# ---------------------------------------------------------------------------


def test_evaluation_module_does_not_import_qdrant_client():
    """The eval module must be stdlib-only at import time."""

    import importlib
    import sys

    saved = {
        name: mod
        for name, mod in sys.modules.items()
        if name.startswith("qdrant_client")
    }
    try:
        for name in list(sys.modules):
            if name == "qdrant_memory.evaluation":
                sys.modules.pop(name, None)
        importlib.import_module("qdrant_memory.evaluation")
        leaked = [
            name
            for name in sys.modules
            if name == "qdrant_client" or name.startswith("qdrant_client.")
        ]
        assert not leaked, f"qdrant_client leaked into eval import: {leaked}"
    finally:
        for name, mod in saved.items():
            sys.modules[name] = mod


# ---------------------------------------------------------------------------
# Phase 6A contract regressions: source-hit labeled denominator, lane dedupe,
# strict point_id for exact identifier
# ---------------------------------------------------------------------------


def test_source_hit_at_k_is_none_when_no_source_or_file_labels():
    """Per-row source_hit_at_k mirrors exact_identifier_hit: null on term-only cases."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_terms": ["phase 5"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "p1",
                        "text": "phase 5 body",
                        "file_path": "docs/RAPTOR.md",
                    }
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    # hit_at_k still trips on the term; the source/file label is absent,
    # so source_hit_at_k must be None rather than False.
    assert row["hit_at_k"] is True
    assert row["source_hit_at_k"] is None
    assert row["exact_identifier_hit"] is None


def test_source_hit_at_k_is_false_when_labels_present_but_no_source_match():
    """When the case carries source/file labels but the packet has no match,
    the per-row source_hit_at_k must be False (not None) so the aggregate
    can compute the rate over a labeled denominator."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_file_paths": ["docs/RAPTOR.md"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "p1",
                        "text": "clean body",
                        "file_path": "docs/OTHER.md",
                    }
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["source_hit_at_k"] is False
    # exact_identifier_hit also stays None: no expected point ids.
    assert row["exact_identifier_hit"] is None


def test_aggregate_source_hit_at_k_rate_uses_only_source_labeled_rows():
    """Mixed variant: one source-labeled row that misses + one term-only
    row. The source-hit rate must be computed only over the
    source-labeled row (0.0 here) and the labeled count must reflect
    the source-labeled rows, NOT all non-errored rows."""

    rows = [
        # Source-labeled row, no match -> source_hit_at_k False.
        {
            "case_id": "c1", "variant": "hybrid", "errored": False,
            "hit_at_k": False, "source_hit_at_k": False,
            "exact_identifier_hit": False, "wrong_memory": False,
            "context_chars": 100, "zoom_efficiency": 0.5,
            "latency_ms": 200.0, "latency_budget_met": True,
        },
        # Term-only row: no source/file labels -> source_hit_at_k None.
        {
            "case_id": "c2", "variant": "hybrid", "errored": False,
            "hit_at_k": True, "source_hit_at_k": None,
            "exact_identifier_hit": None, "wrong_memory": False,
            "context_chars": 80, "zoom_efficiency": 0.4,
            "latency_ms": 150.0, "latency_budget_met": True,
        },
        # Pure exact-id-only row: no source/file labels -> source_hit_at_k None.
        {
            "case_id": "c3", "variant": "hybrid", "errored": False,
            "hit_at_k": True, "source_hit_at_k": None,
            "exact_identifier_hit": True, "wrong_memory": False,
            "context_chars": 60, "zoom_efficiency": 0.3,
            "latency_ms": 120.0, "latency_budget_met": True,
        },
    ]
    aggregates = evaluation.aggregate_scores(rows)
    hybrid = aggregates["hybrid"]
    # 3 non-errored rows total but only 1 carried source/file labels.
    assert hybrid["scored_count"] == 3
    assert hybrid["source_hit_labeled_count"] == 1
    assert hybrid["source_hit_at_k_rate"] == 0.0  # 0 of 1 labeled


def test_lane_aware_top_k_dedupes_same_point_id_across_lanes():
    """A point emitted in both summaries and cited_leaves must dedupe to one
    entry in the lane-aware top-k, regardless of lane."""

    candidates = evaluation.normalize_packet(
        {
            "results": {
                "exact_hits": [],
                "summaries": [
                    {
                        "point_id": "shared-1",
                        "parent_point_id": "parent-1",
                        "text": "summary body",
                    }
                ],
                "cited_leaves": [
                    {
                        "point_id": "shared-1",
                        "parent_point_id": "parent-1",
                        "text": "leaf body for shared point",
                    }
                ],
                "graph_relations": [
                    {
                        "point_id": "shared-1",
                        "parent_point_id": "parent-1",
                        "text": "graph relation for shared point",
                    }
                ],
            }
        }
    )
    topk = evaluation._lane_aware_top_k(candidates, top_k=5)
    # Same point id appears in three lanes but must dedupe to one.
    assert len(topk) == 1
    assert topk[0].point_id == "shared-1"
    # First occurrence wins (summaries comes before cited_leaves,
    # which comes before graph_relations in _LANE_ORDER).
    assert topk[0].lane == "summaries"


def test_lane_aware_top_k_topk_count_dedupes_same_point_id_across_lanes():
    """The topk_count surfaced per-row must reflect lane-aware dedupe, so
    emitting the same point across lanes does not inflate the count."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_point_ids": ["shared-1"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [],
                "summaries": [
                    {
                        "point_id": "shared-1",
                        "parent_point_id": "parent-1",
                        "text": "summary body",
                    }
                ],
                "cited_leaves": [
                    {
                        "point_id": "shared-1",
                        "parent_point_id": "parent-1",
                        "text": "leaf body",
                    }
                ],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run, top_k=5)
    # Three lanes emit the same point id; topk_count must be 1.
    assert row["topk_count"] == 1
    # And exact_identifier_hit still trips on the single surviving entry.
    assert row["exact_identifier_hit"] is True


def test_lane_aware_top_k_falls_back_to_lane_rank_when_point_id_missing():
    """Candidates without a point_id must keep their lane::rank fallback
    so the legacy stream does not collapse two distinct anonymous
    candidates into one."""

    candidates = evaluation.normalize_packet(
        [
            {"text": "legacy alpha"},
            {"text": "legacy bravo"},
        ]
    )
    topk = evaluation._lane_aware_top_k(candidates, top_k=5)
    # Two anonymous candidates in the same legacy lane dedupe on rank.
    assert len(topk) == 2
    assert [cand.text for cand in topk] == ["legacy alpha", "legacy bravo"]


def test_exact_identifier_hit_strict_on_emitted_point_id_not_parent():
    """The contract is "exact identifier appears as an emitted point id".
    A leaf that only references the expected id via parent_point_id
    must NOT satisfy exact_identifier_hit."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_point_ids": ["summary-1"],
    }
    run = {
        "case_id": "c1",
        "variant": "raptor-only",
        "packet": {
            "results": {
                "exact_hits": [],
                "summaries": [],
                "cited_leaves": [
                    {
                        # The leaf's own id is different; only its parent
                        # references the expected identifier.
                        "point_id": "leaf-1",
                        "parent_point_id": "summary-1",
                        "text": "leaf body that cites summary-1",
                    }
                ],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run, top_k=5)
    # No candidate's emitted point_id equals "summary-1", so the strict
    # exact_identifier_hit must be False even though the leaf is parented
    # to it. Poison detection, however, still uses parent_point_id, so a
    # forbidden id declared via parent_point_id remains poison.
    assert row["exact_identifier_hit"] is False
    assert "summary-1" not in row["matched_expected"]["point_ids"]


def test_forbidden_point_id_still_fires_via_parent_point_id():
    """Companion to the strictness test: forbidden_point_ids is a poison
    rule, so a forbidden id declared via parent_point_id must still
    trip wrong_memory (kept conservative on purpose)."""

    case = {
        "case_id": "c1",
        "query": "any",
        "forbidden_point_ids": ["poison-parent-1"],
    }
    run = {
        "case_id": "c1",
        "variant": "raptor-only",
        "packet": {
            "results": {
                "summaries": [],
                "cited_leaves": [
                    {
                        "point_id": "leaf-1",
                        "parent_point_id": "poison-parent-1",
                        "text": "leaf body",
                    }
                ],
                "exact_hits": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["wrong_memory"] is True
    assert "poison-parent-1" in row["wrong_reasons"]["forbidden_point_ids"]


def test_dedupe_key_uses_point_id_globally_when_present():
    """_dedupe_key must return a lane-independent key for any non-empty
    point_id so the same point in two lanes collides."""

    exact = evaluation.NormalizedCandidate(
        lane="exact_hits", rank=1, point_id="p1",
    )
    summary = evaluation.NormalizedCandidate(
        lane="summaries", rank=2, point_id="p1",
    )
    assert evaluation._dedupe_key(exact) == evaluation._dedupe_key(summary)


def test_dedupe_key_falls_back_to_lane_rank_when_point_id_empty():
    """Candidates without a point_id must keep a lane-scoped fallback so
    anonymous entries in different lanes do not collide."""

    legacy_a = evaluation.NormalizedCandidate(
        lane="legacy", rank=1, point_id="",
    )
    legacy_b = evaluation.NormalizedCandidate(
        lane="legacy", rank=2, point_id="",
    )
    other_lane = evaluation.NormalizedCandidate(
        lane="exact_hits", rank=1, point_id="",
    )
    # Same lane, different rank: distinct.
    assert evaluation._dedupe_key(legacy_a) != evaluation._dedupe_key(legacy_b)
    # Same rank, different lane: distinct (lane is part of the fallback).
    assert evaluation._dedupe_key(legacy_a) != evaluation._dedupe_key(other_lane)


# ---------------------------------------------------------------------------
# Phase 6A contract regressions: matched_expected.sources carries expected
# labels that matched (not arbitrary emitted candidate fields), so a
# file_path-only match never surfaces a nonmatching source_uri, and so two
# candidates sharing the same expected file path cannot inflate
# useful_topk_count / zoom_efficiency.
# ---------------------------------------------------------------------------


def test_matched_expected_sources_reports_file_path_when_source_uri_mismatches():
    """When the case declares only expected_file_paths and a candidate
    matches the file_path but carries an unrelated source_uri, the row
    must surface the matched file_path, NOT the candidate's source_uri."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_file_paths": ["docs/RAPTOR.md"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "p1",
                        "text": "body",
                        # file_path matches the expected label, but the
                        # emitted source_uri is unrelated.
                        "file_path": "docs/RAPTOR.md",
                        "source_uri": "file://wrong.md",
                    }
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    assert row["source_hit_at_k"] is True
    sources = row["matched_expected"]["sources"]
    # Must report the matched expected file_path, never the unrelated
    # candidate source_uri.
    assert sources == ["docs/RAPTOR.md"]
    assert "file://wrong.md" not in sources


def test_duplicate_expected_file_path_does_not_inflate_useful_topk_count():
    """Two candidates sharing the same expected file_path but with
    distinct nonmatching source_uri values must dedupe to a single
    useful_topk source handle, not inflate useful_topk_count to 2."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_file_paths": ["docs/RAPTOR.md"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "p1",
                        "text": "first body",
                        "file_path": "docs/RAPTOR.md",
                        "source_uri": "file://wrong-a.md",
                    },
                    {
                        "point_id": "p2",
                        "text": "second body",
                        "file_path": "docs/RAPTOR.md",
                        "source_uri": "file://wrong-b.md",
                    },
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    # Both candidates match the same expected file_path. The expected
    # label is one ("docs/RAPTOR.md"); useful_topk_count must be 1 for
    # the source dimension (no other expected labels declared).
    assert row["useful_topk_count"] == 1
    assert row["matched_expected"]["sources"] == ["docs/RAPTOR.md"]
    # And the non-matching emitted source_uri values must not leak into
    # the matched_expected payload.
    assert "file://wrong-a.md" not in row["matched_expected"]["sources"]
    assert "file://wrong-b.md" not in row["matched_expected"]["sources"]
    # zoom_efficiency must be computed against the deduped count, not
    # the inflated per-candidate count.
    assert row["zoom_efficiency"] >= 0


def test_candidate_with_both_source_uri_and_file_path_match_returns_both_labels():
    """When a candidate matches both an expected source URI and an
    expected file path, the row must surface BOTH expected labels
    (deduped against the case-level expected list)."""

    case = {
        "case_id": "c1",
        "query": "any",
        "expected_source_uris": ["file://docs/RAPTOR.md"],
        "expected_file_paths": ["docs/RAPTOR.md"],
    }
    run = {
        "case_id": "c1",
        "variant": "hybrid",
        "packet": {
            "results": {
                "exact_hits": [
                    {
                        "point_id": "p1",
                        "text": "body",
                        "file_path": "docs/RAPTOR.md",
                        "source_uri": "file://docs/RAPTOR.md",
                    }
                ],
                "summaries": [],
                "cited_leaves": [],
                "graph_relations": [],
            }
        },
    }
    row = evaluation.score_case_run(case, run)
    sources = row["matched_expected"]["sources"]
    # Both expected labels must appear, source URI first.
    assert "file://docs/RAPTOR.md" in sources
    assert "docs/RAPTOR.md" in sources
    assert sources.index("file://docs/RAPTOR.md") < sources.index("docs/RAPTOR.md")
    # And the useful_topk set carries one source handle per expected
    # label, not per candidate field.
    assert row["useful_topk_count"] == 2  # one per expected label
    assert row["source_hit_at_k"] is True


def test_matched_expected_source_labels_helper_returns_only_expected_labels():
    """The match helper itself must return only expected labels that
    actually matched the candidate's emitted source_uri / file_path,
    in source-URI-then-file-path order, deduped per candidate."""

    cand = evaluation.NormalizedCandidate(
        lane="exact_hits",
        rank=1,
        point_id="p1",
        source_uri="file://docs/RAPTOR.md",
        file_path="docs/RAPTOR.md",
    )
    matched = evaluation._candidate_matched_expected_source_labels(
        cand,
        expected_source_uris=["file://docs/RAPTOR.md"],
        expected_file_paths=["docs/RAPTOR.md"],
    )
    assert matched == ["file://docs/RAPTOR.md", "docs/RAPTOR.md"]


def test_matched_expected_source_labels_helper_skips_nonmatching_source_uri():
    """A candidate whose emitted source_uri does not equal any expected
    source URI must not contribute a nonmatching URI to the matched
    list, even when the file_path matches an expected label."""

    cand = evaluation.NormalizedCandidate(
        lane="exact_hits",
        rank=1,
        point_id="p1",
        source_uri="file://wrong.md",
        file_path="docs/RAPTOR.md",
    )
    matched = evaluation._candidate_matched_expected_source_labels(
        cand,
        expected_source_uris=["file://docs/RAPTOR.md"],
        expected_file_paths=["docs/RAPTOR.md"],
    )
    # Only the expected file_path matched; the unrelated emitted
    # source_uri must not appear.
    assert matched == ["docs/RAPTOR.md"]
    assert "file://wrong.md" not in matched
