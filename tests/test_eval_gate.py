"""Phase 6F shadow gate / explicit thresholds tests.

These tests cover:

* ``evaluate_gate`` pass and fail paths (wrong_memory, missing variant,
  baseline lift, exact-id drop, latency p95).
* Default ``auto-recall-default`` preset is conservative: a Phase 6E-like
  ``hybrid`` variant with ``wrong_memory_rate = 4.0`` fails the gate.
* Threshold override merging (CLI flags) and JSON threshold file loading.
* CLI parser registration, local command dispatch (exit codes 0/1/2),
  and ``build_tool_call`` fail-closed block.
* The gate module never imports ``qdrant_client`` or the provider.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from qdrant_memory import eval_gate


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _variant_summary(
    *,
    case_count: int = 25,
    scored_count: int = 25,
    errored_count: int = 0,
    hit_at_k_rate: float = 88.0,
    source_hit_at_k_rate: float = 92.8571,
    exact_identifier_hit_rate: float = 94.4444,
    wrong_memory_rate: float = 4.0,
    latency_ms_p95: float = 265.7,
    latency_budget_pass_rate: float = 100.0,
) -> dict:
    return {
        "case_count": case_count,
        "scored_count": scored_count,
        "errored_count": errored_count,
        "hit_at_k_rate": hit_at_k_rate,
        "source_hit_at_k_rate": source_hit_at_k_rate,
        "exact_identifier_hit_rate": exact_identifier_hit_rate,
        "wrong_memory_rate": wrong_memory_rate,
        "latency_ms_p95": latency_ms_p95,
        "latency_budget_pass_rate": latency_budget_pass_rate,
    }


def _make_report(
    *,
    candidate: dict | None = None,
    baselines: dict[str, dict] | None = None,
) -> dict:
    """Build a minimal evaluator-report-shaped dict for gate tests."""
    variants: dict[str, dict] = {}
    if candidate is not None:
        variants["hybrid"] = candidate
    for name, summary in (baselines or {}).items():
        variants[name] = summary
    return {
        "config": {},
        "rows": [],
        "summary": {
            "totals": {
                "case_count": 25,
                "run_count": 175,
                "scored_count": 175,
            },
            "variants": variants,
        },
    }


def _phase6e_like_report() -> dict:
    """Report shaped like the Phase 6E final metrics so the default
    preset gate fails on wrong_memory_rate = 4.0."""
    return _make_report(
        candidate=_variant_summary(wrong_memory_rate=4.0),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.7143,
                exact_identifier_hit_rate=88.8889,
                wrong_memory_rate=4.0,
                latency_ms_p95=122.96,
            ),
            "dense+sparse": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.7143,
                exact_identifier_hit_rate=83.3333,
                wrong_memory_rate=4.0,
                latency_ms_p95=153.44,
            ),
        },
    )


# ---------------------------------------------------------------------------
# Module isolation
# ---------------------------------------------------------------------------


def test_eval_gate_module_does_not_import_qdrant_client():
    """The gate module must be stdlib-only and never import qdrant_client
    or the provider."""
    import importlib
    import sys

    # Ensure a fresh import of the module.
    modules_to_remove = [
        name
        for name in sys.modules
        if name.startswith("qdrant_memory.eval_gate")
    ]
    for name in modules_to_remove:
        del sys.modules[name]

    importlib.import_module("qdrant_memory.eval_gate")

    assert "qdrant_client" not in sys.modules
    # The provider __init__ should not have been imported just by
    # importing eval_gate. (It may already be in sys.modules from other
    # test imports, so we check the eval_gate module's own imports.)
    import qdrant_memory.eval_gate as eg

    source = Path(eg.__file__).read_text(encoding="utf-8")
    assert "import qdrant_client" not in source
    assert "from qdrant_client" not in source
    assert "from qdrant_memory.__init__" not in source
    assert "from qdrant_memory.client" not in source
    assert "QdrantMemoryProvider" not in source


# ---------------------------------------------------------------------------
# Default preset / thresholds
# ---------------------------------------------------------------------------


def test_auto_recall_default_thresholds_are_conservative():
    thresholds = eval_gate.GateThresholds.auto_recall_default()
    # The strict wrong_memory cap at 3.0 is the key conservative knob.
    assert thresholds.max_wrong_memory_rate == 3.0
    assert thresholds.min_candidate_hit_at_k == 80.0
    assert thresholds.min_candidate_source_hit_at_k == 80.0
    assert thresholds.min_case_count >= 10
    assert thresholds.max_latency_p95_ms > 0


def test_thresholds_from_dict_ignores_unknown_keys():
    thresholds = eval_gate.thresholds_from_dict(
        {"max_wrong_memory_rate": 5.0, "unknown_future_field": True},
    )
    assert thresholds.max_wrong_memory_rate == 5.0


def test_thresholds_from_dict_rejects_non_numeric():
    with pytest.raises(eval_gate.EvalGateError, match="must be a number"):
        eval_gate.thresholds_from_dict({"max_wrong_memory_rate": "not-a-number"})


def test_merge_threshold_overrides_applies_only_non_none():
    base = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.merge_threshold_overrides(
        base,
        {"max_wrong_memory_rate": 10.0, "min_case_count": None},
    )
    assert result.max_wrong_memory_rate == 10.0
    assert result.min_case_count == base.min_case_count


def test_load_thresholds_file(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"max_wrong_memory_rate": 7.0}),
        encoding="utf-8",
    )
    thresholds = eval_gate.load_thresholds_file(str(path))
    assert thresholds.max_wrong_memory_rate == 7.0


def test_load_thresholds_file_invalid_json(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(eval_gate.EvalGateError, match="invalid JSON"):
        eval_gate.load_thresholds_file(str(path))


# ---------------------------------------------------------------------------
# Threshold type preservation (P3 regression)
# ---------------------------------------------------------------------------
#
# Reviewer P3 (Phase 6F): under ``from __future__ import annotations``,
# ``GateThresholds.__dataclass_fields__[key].type`` is the raw annotation
# string (e.g. ``"int"``), so a ``field_type is int`` check silently fails
# and every threshold gets coerced via ``float(value)``. The fix routes
# int/float dispatch through an explicit set of integer field names. These
# tests guard the regression at both entry points (JSON/dict and CLI
# override merge) and assert the exact runtime type, not just equality.


def test_thresholds_from_dict_preserves_int_type_for_int_field():
    """JSON/dict override for an integer threshold field must yield a
    real ``int``, even when the source value is a numeric string."""
    thresholds = eval_gate.thresholds_from_dict({"min_case_count": "12"})
    as_dict = thresholds.to_dict()
    assert type(as_dict["min_case_count"]) is int
    assert as_dict["min_case_count"] == 12

    # All three int-typed fields must round-trip as ``int``.
    for key in ("min_case_count", "min_scored_count", "max_errored_count"):
        out = eval_gate.thresholds_from_dict({key: "5"}).to_dict()
        assert type(out[key]) is int, f"{key} should be int, got {type(out[key]).__name__}"


def test_thresholds_from_dict_preserves_float_type_for_float_field():
    """Float threshold fields must remain ``float`` so rate/latency math
    stays in float precision."""
    thresholds = eval_gate.thresholds_from_dict(
        {"max_wrong_memory_rate": "3.5", "max_latency_p95_ms": "500"},
    )
    as_dict = thresholds.to_dict()
    assert type(as_dict["max_wrong_memory_rate"]) is float
    assert type(as_dict["max_latency_p95_ms"]) is float


def test_merge_threshold_overrides_preserves_int_type_for_int_field():
    """CLI/merge override for an integer threshold field must yield a
    real ``int`` when the override is a numeric string (as argparse
    produces from CLI flags)."""
    base = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.merge_threshold_overrides(
        base,
        {"min_case_count": "15", "max_errored_count": "2"},
    )
    as_dict = result.to_dict()
    assert type(as_dict["min_case_count"]) is int
    assert as_dict["min_case_count"] == 15
    assert type(as_dict["max_errored_count"]) is int
    assert as_dict["max_errored_count"] == 2


def test_merge_threshold_overrides_preserves_float_type_for_float_field():
    base = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.merge_threshold_overrides(
        base,
        {"max_wrong_memory_rate": "7.5"},
    )
    as_dict = result.to_dict()
    assert type(as_dict["max_wrong_memory_rate"]) is float
    assert as_dict["max_wrong_memory_rate"] == 7.5


def test_load_thresholds_file_round_trip_preserves_int_type(tmp_path):
    """The end-to-end JSON threshold file path must also preserve the
    int type, since the CLI loads thresholds from disk via this entry."""
    path = tmp_path / "thresholds.json"
    path.write_text(
        json.dumps({"min_case_count": 20, "max_errored_count": 1}),
        encoding="utf-8",
    )
    thresholds = eval_gate.load_thresholds_file(str(path))
    as_dict = thresholds.to_dict()
    assert type(as_dict["min_case_count"]) is int
    assert type(as_dict["max_errored_count"]) is int


def test_thresholds_int_field_default_stays_int():
    """Sanity: even without any override, the dataclass defaults for the
    integer fields must already be plain ``int`` (not ``float``), so the
    bug is genuinely about override coercion and not about defaults."""
    defaults = eval_gate.GateThresholds.auto_recall_default().to_dict()
    for key in ("min_case_count", "min_scored_count", "max_errored_count"):
        assert type(defaults[key]) is int, (
            f"{key} default should be int, got {type(defaults[key]).__name__}"
        )


# ---------------------------------------------------------------------------
# Gate evaluation: pass
# ---------------------------------------------------------------------------


def test_evaluate_gate_passes_with_good_metrics():
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            exact_identifier_hit_rate=95.0,
            wrong_memory_rate=0.0,
            latency_ms_p95=200.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=80.0,
                source_hit_at_k_rate=85.0,
                exact_identifier_hit_rate=90.0,
                wrong_memory_rate=0.0,
            ),
            "dense+sparse": _variant_summary(
                hit_at_k_rate=80.0,
                source_hit_at_k_rate=85.0,
                exact_identifier_hit_rate=85.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["schema"] == "qdrant_eval_gate.v1"
    assert result["status"] == "pass"
    assert result["auto_recall_eligible"] is True
    assert result["candidate_variant"] == "hybrid"
    assert result["candidate_present"] is True
    assert result["summary"]["failed"] == 0
    assert set(result["baselines_found"]) == {"dense-only", "dense+sparse"}


# ---------------------------------------------------------------------------
# Gate evaluation: fail due wrong_memory_rate
# ---------------------------------------------------------------------------


def test_evaluate_gate_fails_on_wrong_memory_rate():
    """Phase 6E feature: wrong_memory_rate = 4.0 fails the default
    preset which caps at 3.0, even though hit/source/latency pass."""
    report = _phase6e_like_report()
    thresholds = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["status"] == "fail"
    assert result["auto_recall_eligible"] is False
    failed_names = result["summary"]["failed_checks"]
    assert "wrong_memory_rate" in failed_names

    # The check actual should be 4.0 and threshold 3.0.
    wm_check = [c for c in result["checks"] if c["name"] == "wrong_memory_rate"][0]
    assert wm_check["actual"] == 4.0
    assert wm_check["threshold"] == 3.0


def test_evaluate_gate_wrong_memory_regression_passes_when_candidate_improves():
    """When candidate has LOWER wrong_memory_rate than the best baseline,
    the regression drop is 0 (no regression), so the check passes."""
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            wrong_memory_rate=1.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=80.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=3.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        max_wrong_memory_rate=10.0,
        max_wrong_memory_regression=2.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)
    # Candidate improves: regression drop = max(0, 1.0 - 3.0) = 0 <= 2.0 -> PASS.
    assert result["status"] == "pass"
    assert "wrong_memory_regression" not in result["summary"]["failed_checks"]


def test_wrong_memory_regression_fails_when_candidate_is_worse():
    """The wrong_memory_regression check fails when candidate's wrong
    memory rate exceeds the best baseline by more than the threshold."""
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            wrong_memory_rate=5.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=80.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=1.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        max_wrong_memory_rate=10.0,
        max_wrong_memory_regression=1.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    # Regression drop = max(0, 5.0 - 1.0) = 4.0 > 1.0 threshold -> FAIL.
    assert result["status"] == "fail"
    assert "wrong_memory_regression" in result["summary"]["failed_checks"]


# ---------------------------------------------------------------------------
# Gate evaluation: fail-closed missing variant
# ---------------------------------------------------------------------------


def test_evaluate_gate_fails_closed_when_candidate_missing():
    report = _make_report(
        candidate=None,
        baselines={
            "dense-only": _variant_summary(),
        },
    )
    thresholds = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["status"] == "fail"
    assert result["auto_recall_eligible"] is False
    assert result["candidate_present"] is False
    assert len(result["checks"]) == 1
    assert result["checks"][0]["name"] == "candidate_present"
    assert result["checks"][0]["status"] == "fail"


# ---------------------------------------------------------------------------
# Gate evaluation: baseline lift
# ---------------------------------------------------------------------------


def test_evaluate_gate_fails_on_insufficient_hit_lift():
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=85.0,
            source_hit_at_k_rate=95.0,
            wrong_memory_rate=0.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        min_hit_at_k_lift=5.0,
        max_wrong_memory_rate=10.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["status"] == "fail"
    failed_names = result["summary"]["failed_checks"]
    assert "hit_at_k_lift" in failed_names
    lift_check = [c for c in result["checks"] if c["name"] == "hit_at_k_lift"][0]
    assert lift_check["actual"] == 1.0  # 85.0 - 84.0
    assert lift_check["threshold"] == 5.0


def test_evaluate_gate_passes_baseline_lift_when_meets_threshold():
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            wrong_memory_rate=0.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        min_hit_at_k_lift=5.0,
        min_source_hit_at_k_lift=5.0,
        max_wrong_memory_rate=10.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)
    assert result["status"] == "pass"


# ---------------------------------------------------------------------------
# Gate evaluation: exact-id drop
# ---------------------------------------------------------------------------


def test_evaluate_gate_fails_on_exact_id_drop():
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            exact_identifier_hit_rate=80.0,
            wrong_memory_rate=0.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.0,
                exact_identifier_hit_rate=90.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        max_exact_id_drop=5.0,
        max_wrong_memory_rate=10.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["status"] == "fail"
    failed_names = result["summary"]["failed_checks"]
    assert "exact_id_drop" in failed_names
    drop_check = [c for c in result["checks"] if c["name"] == "exact_id_drop"][0]
    assert drop_check["actual"] == 10.0  # 90.0 - 80.0
    assert drop_check["threshold"] == 5.0


# ---------------------------------------------------------------------------
# Gate evaluation: latency p95
# ---------------------------------------------------------------------------


def test_evaluate_gate_fails_on_latency_p95():
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            wrong_memory_rate=0.0,
            latency_ms_p95=600.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        max_latency_p95_ms=500.0,
        max_wrong_memory_rate=10.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["status"] == "fail"
    failed_names = result["summary"]["failed_checks"]
    assert "latency_p95" in failed_names


def test_evaluate_gate_fails_on_latency_budget_pass_rate():
    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            wrong_memory_rate=0.0,
            latency_budget_pass_rate=80.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=84.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    thresholds = eval_gate.GateThresholds(
        min_latency_budget_pass_rate=95.0,
        max_wrong_memory_rate=10.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    assert result["status"] == "fail"
    failed_names = result["summary"]["failed_checks"]
    assert "latency_budget_pass_rate" in failed_names


# ---------------------------------------------------------------------------
# Gate evaluation: sample-size checks
# ---------------------------------------------------------------------------


def test_evaluate_gate_fails_on_errored_count():
    report = _make_report(
        candidate=_variant_summary(errored_count=3),
        baselines={
            "dense-only": _variant_summary(),
        },
    )
    thresholds = eval_gate.GateThresholds(max_errored_count=0, max_wrong_memory_rate=10.0)
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)
    assert result["status"] == "fail"
    assert "errored_count" in result["summary"]["failed_checks"]


def test_evaluate_gate_fails_on_case_count():
    report = _make_report(
        candidate=_variant_summary(case_count=5, scored_count=5),
        baselines={
            "dense-only": _variant_summary(),
        },
    )
    thresholds = eval_gate.GateThresholds(
        min_case_count=10, max_wrong_memory_rate=10.0,
    )
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)
    assert result["status"] == "fail"
    assert "case_count" in result["summary"]["failed_checks"]


# ---------------------------------------------------------------------------
# Gate output: no raw query / packet leakage
# ---------------------------------------------------------------------------


def test_gate_output_has_no_raw_query_or_packet_keys():
    """The gate output must never carry 'query', 'packet', 'rows', or
    per-row payloads. It only reads aggregate metrics."""
    report = _phase6e_like_report()
    thresholds = eval_gate.GateThresholds.auto_recall_default()
    result = eval_gate.evaluate_gate(report, thresholds=thresholds)

    serialized = json.dumps(result, sort_keys=True)
    for forbidden_key in ("query", "packet", "rows", "matched_expected", "wrong_reasons"):
        assert forbidden_key not in serialized, (
            f"gate output leaked sensitive key: {forbidden_key!r}"
        )


def test_gate_output_schema_stable():
    report = _phase6e_like_report()
    result = eval_gate.evaluate_gate(report)
    assert result["schema"] == "qdrant_eval_gate.v1"
    # Top-level keys must be stable and sorted.
    expected_keys = {
        "schema", "status", "auto_recall_eligible", "candidate_variant",
        "candidate_present", "baseline_variants", "baselines_found",
        "thresholds", "checks", "candidate_metrics", "baseline_metrics",
        "summary",
    }
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# CLI tests
# ---------------------------------------------------------------------------


def _load_plugin_cli_module():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("qdrant_plugin_cli_test_gate", root / "cli.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parser():
    import argparse

    cli = _load_plugin_cli_module()
    parser = argparse.ArgumentParser(prog="hermes")
    subparsers = parser.add_subparsers(dest="command", required=True)
    qdrant_parser = subparsers.add_parser("qdrant")
    cli.register_cli(qdrant_parser)
    qdrant_parser.set_defaults(func=cli.qdrant_command)
    return parser


def test_cli_parser_eval_gate_defaults():
    parser = _parser()
    args = parser.parse_args(
        ["qdrant", "eval-gate", "--report", "report.json"],
    )
    assert args.qdrant_subcommand == "eval-gate"
    assert args.report == "report.json"
    assert args.candidate_variant == "hybrid"
    assert args.baseline_variants is None  # falls back to default
    assert args.thresholds_file is None
    assert args.json is False


def test_cli_parser_eval_gate_custom_baselines_and_overrides():
    parser = _parser()
    args = parser.parse_args(
        [
            "qdrant", "eval-gate", "--report", "r.json",
            "--candidate-variant", "hybrid-no-graph",
            "--baseline-variant", "dense-only",
            "--baseline-variant", "graph",
            "--max-wrong-memory-rate", "5.0",
            "--min-hit-at-k-lift", "3.0",
            "--json",
        ],
    )
    assert args.candidate_variant == "hybrid-no-graph"
    assert args.baseline_variants == ["dense-only", "graph"]
    assert args.max_wrong_memory_rate == 5.0
    assert args.min_hit_at_k_lift == 3.0
    assert args.json is True


def test_cli_eval_gate_local_command_pass(tmp_path, capsys):
    """A report with all-good metrics produces exit code 0."""
    from qdrant_memory.cli_core import execute_command

    report = _make_report(
        candidate=_variant_summary(
            hit_at_k_rate=90.0,
            source_hit_at_k_rate=95.0,
            exact_identifier_hit_rate=95.0,
            wrong_memory_rate=0.0,
            latency_ms_p95=200.0,
        ),
        baselines={
            "dense-only": _variant_summary(
                hit_at_k_rate=80.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=0.0,
            ),
            "dense+sparse": _variant_summary(
                hit_at_k_rate=80.0,
                source_hit_at_k_rate=85.0,
                wrong_memory_rate=0.0,
            ),
        },
    )
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    parser = _parser()
    args = parser.parse_args(
        ["qdrant", "eval-gate", "--report", str(report_path), "--json"],
    )

    exit_code = execute_command(
        args,
        provider_factory=lambda: pytest.fail("provider should not be constructed"),
    )
    assert exit_code == 0
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] == "pass"
    assert result["auto_recall_eligible"] is True


def test_cli_eval_gate_local_command_fail(tmp_path, capsys):
    """A report that fails the gate (wrong_memory_rate=4.0) produces
    exit code 1."""
    from qdrant_memory.cli_core import execute_command

    report = _phase6e_like_report()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    parser = _parser()
    args = parser.parse_args(
        ["qdrant", "eval-gate", "--report", str(report_path), "--json"],
    )

    exit_code = execute_command(
        args,
        provider_factory=lambda: pytest.fail("provider should not be constructed"),
    )
    assert exit_code == 1
    out = capsys.readouterr().out
    result = json.loads(out)
    assert result["status"] == "fail"
    assert result["auto_recall_eligible"] is False


def test_cli_eval_gate_human_summary_not_json(tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    report = _phase6e_like_report()
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    parser = _parser()
    args = parser.parse_args(
        ["qdrant", "eval-gate", "--report", str(report_path)],
    )

    exit_code = execute_command(
        args,
        provider_factory=lambda: pytest.fail("provider should not be constructed"),
    )
    assert exit_code == 1  # fails gate
    out = capsys.readouterr().out
    assert "Phase 6F eval-gate report" in out
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_cli_eval_gate_usage_error_on_missing_report(tmp_path, capsys):
    from qdrant_memory.cli_core import execute_command

    parser = _parser()
    args = parser.parse_args(
        ["qdrant", "eval-gate", "--report", str(tmp_path / "nonexistent.json"), "--json"],
    )

    exit_code = execute_command(
        args,
        provider_factory=lambda: pytest.fail("provider should not be constructed"),
    )
    assert exit_code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    error = json.loads(captured.err)
    assert error["error"] is True
    assert "failed to read report file" in error["message"]


def test_cli_eval_gate_missing_required_args_is_usage_error():
    parser = _parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["qdrant", "eval-gate"])


def test_cli_eval_gate_blocked_in_provider_dispatch():
    from qdrant_memory.cli_core import CliUsageError, build_tool_call

    parser = _parser()
    args = parser.parse_args(
        ["qdrant", "eval-gate", "--report", "report.json"],
    )
    with pytest.raises(CliUsageError, match="handled by"):
        build_tool_call(args)


# ---------------------------------------------------------------------------
# Format human summary
# ---------------------------------------------------------------------------


def test_format_human_summary_contains_status_and_checks():
    report = _phase6e_like_report()
    result = eval_gate.evaluate_gate(report)
    summary = eval_gate.format_human_summary(result)
    assert "Phase 6F eval-gate report" in summary
    assert "status: fail" in summary
    assert "auto_recall_eligible: False" in summary
    assert "[FAIL] wrong_memory_rate" in summary
