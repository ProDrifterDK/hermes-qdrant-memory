"""Phase 6F shadow gate / explicit thresholds for auto-recall eligibility.

This module is **stdlib-only** and **offline** by design. It reads a JSON
report produced by ``hermes qdrant eval --json`` and evaluates explicit
thresholds before auto-recall can be considered. It is the "shadow gate":
a named, deterministic decision surface that an operator can run locally
to decide whether a candidate retrieval variant is good enough to promote.

Hard rules enforced here:

* No live Qdrant client. No Qdrant mutation. No provider factory.
* No imports that initialize Hermes provider state. This module never
  imports :mod:`qdrant_memory.client`, :mod:`qdrant_memory.__init__`,
  or any provider module.
* Pure functions over dicts/JSON. The only I/O is reading the input
  report file.
* Reports never echo raw query text, packets, or per-row payloads. The
  gate reads aggregate metrics only and its output carries only
  aggregate-level numbers and pass/fail reasons.
* The gate is **advisory**: ``auto_recall_eligible`` is a boolean the
  operator reads. This module does NOT change runtime auto-recall
  behavior. Auto-recall runtime remains legacy/provider prefetch unless
  a future phase explicitly re-wires the runtime path.

Default thresholds (``auto-recall-default`` preset) are conservative.
Current Phase 6E metrics should NOT pass the gate because the default
``max_wrong_memory_rate`` is capped at ``3.0`` while Phase 6E's
``hybrid`` variant has a ``wrong_memory_rate`` of ``4.0``. This is a
feature, not a bug: the honest gate says "not yet".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping


# --------------------------------------------------------------------------- #
# Schema / constants
# --------------------------------------------------------------------------- #

GATE_SCHEMA: str = "qdrant_eval_gate.v1"

DEFAULT_CANDIDATE_VARIANT: str = "hybrid"
DEFAULT_BASELINE_VARIANTS: tuple[str, ...] = ("dense-only", "dense+sparse")

# Metric keys within a per-variant summary dict (Phase 6A evaluator output).
_METRIC_CASE_COUNT = "case_count"
_METRIC_SCORED_COUNT = "scored_count"
_METRIC_ERRORED_COUNT = "errored_count"
_METRIC_HIT_AT_K_RATE = "hit_at_k_rate"
_METRIC_SOURCE_HIT_AT_K_RATE = "source_hit_at_k_rate"
_METRIC_EXACT_ID_RATE = "exact_identifier_hit_rate"
_METRIC_WRONG_MEMORY_RATE = "wrong_memory_rate"
_METRIC_LATENCY_P95 = "latency_ms_p95"
_METRIC_LATENCY_BUDGET_PASS = "latency_budget_pass_rate"

# Names of :class:`GateThresholds` fields whose semantic type is ``int``.
#
# We can't read this from ``__dataclass_fields__[name].type`` because the
# module uses ``from __future__ import annotations``, which postpones
# evaluation and leaves ``.type`` as the raw annotation string (e.g.
# ``"int"``) at runtime. Comparing that string to the ``int`` type object
# silently fails and previously coerced all thresholds to ``float``.
# Keeping an explicit frozenset here makes the int/float split a single
# source of truth, stdlib-only, and trivially auditable.
_INTEGER_THRESHOLD_FIELDS: frozenset[str] = frozenset(
    {
        "min_case_count",
        "min_scored_count",
        "max_errored_count",
    }
)


def _coerce_threshold_value(name: str, value: Any) -> int | float:
    """Coerce a raw threshold ``value`` to the type declared by ``name``.

    Integer fields (see :data:`_INTEGER_THRESHOLD_FIELDS`) become
    :class:`int`; all other recognized fields become :class:`float``.
    Non-numeric inputs raise :class:`EvalGateError` via the callers.
    """

    if name in _INTEGER_THRESHOLD_FIELDS:
        return int(value)
    return float(value)


class EvalGateError(ValueError):
    """Raised when the gate input is structurally invalid.

    The CLI translates this into exit code ``2`` (usage/input error) with
    a single-line user-facing message.
    """


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GateThresholds:
    """Explicit, named thresholds for the auto-recall shadow gate.

    All rate fields are in the same scale as the Phase 6A evaluator
    output (percent, ``0.0``-``100.0``). Latency is in milliseconds.
    Counts are integers.
    """

    min_case_count: int = 10
    min_scored_count: int = 10
    max_errored_count: int = 0
    min_candidate_hit_at_k: float = 80.0
    min_candidate_source_hit_at_k: float = 80.0
    min_hit_at_k_lift: float = 0.0
    min_source_hit_at_k_lift: float = 0.0
    max_exact_id_drop: float = 5.0
    max_wrong_memory_rate: float = 3.0
    max_wrong_memory_regression: float = 1.0
    max_latency_p95_ms: float = 500.0
    min_latency_budget_pass_rate: float = 95.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def auto_recall_default(cls) -> GateThresholds:
        """Conservative default preset for auto-recall eligibility.

        These thresholds require a candidate to have a meaningful sample
        size, zero errored rows, strong absolute hit/source rates, a
        non-negative lift over the best baseline, and critically a
        ``wrong_memory_rate`` capped at ``3.0`` percent. Phase 6E's
        ``hybrid`` variant (``wrong_memory_rate = 4.0``) intentionally
        does NOT pass this preset.
        """

        return cls()


def thresholds_from_dict(data: Mapping[str, Any]) -> GateThresholds:
    """Build :class:`GateThresholds` from a dict, ignoring unknown keys.

    Only recognized field names are extracted; unrecognized keys are
    silently dropped so a JSON threshold file can carry comments or
    future-preset metadata without breaking the parser.
    """

    if not isinstance(data, Mapping):
        raise EvalGateError("thresholds must be a JSON object")
    valid = {f for f in GateThresholds.__dataclass_fields__}  # type: ignore[attr-defined]
    kwargs: dict[str, Any] = {}
    for key in valid:
        if key in data:
            value = data[key]
            try:
                kwargs[key] = _coerce_threshold_value(key, value)
            except (TypeError, ValueError) as exc:
                raise EvalGateError(
                    f"threshold {key!r} must be a number, got {value!r}",
                ) from exc
    return GateThresholds(**kwargs)


def load_thresholds_file(path: str) -> GateThresholds:
    """Read a JSON threshold file and return :class:`GateThresholds`."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalGateError(f"failed to read thresholds file: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvalGateError(f"invalid JSON in thresholds file: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise EvalGateError("thresholds file root must be a JSON object")
    return thresholds_from_dict(parsed)


def merge_threshold_overrides(
    base: GateThresholds,
    overrides: Mapping[str, Any] | None,
) -> GateThresholds:
    """Return a new :class:`GateThresholds` with explicit overrides applied.

    Only non-``None`` values in ``overrides`` are applied. This lets the
    CLI pass parsed flags that default to ``None`` without clobbering
    the base preset.
    """

    if not overrides:
        return base
    valid = {f for f in GateThresholds.__dataclass_fields__}  # type: ignore[attr-defined]
    current = base.to_dict()
    for key in valid:
        value = overrides.get(key)
        if value is None:
            continue
        try:
            current[key] = _coerce_threshold_value(key, value)
        except (TypeError, ValueError) as exc:
            raise EvalGateError(
                f"threshold {key!r} must be a number, got {value!r}",
            ) from exc
    return GateThresholds(**current)


# --------------------------------------------------------------------------- #
# Report loading
# --------------------------------------------------------------------------- #


def load_report(path: str) -> dict[str, Any]:
    """Read and structurally validate a Phase 6A evaluator JSON report.

    The report must be a JSON object with a ``summary`` dict containing
    a ``variants`` dict. We do not validate individual variant fields
    here; the gate evaluation handles missing/null metrics per-check.
    """

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise EvalGateError(f"failed to read report file: {exc}") from exc
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EvalGateError(f"invalid JSON in report file: {exc.msg}") from exc
    if not isinstance(parsed, Mapping):
        raise EvalGateError("report root must be a JSON object")
    summary = parsed.get("summary")
    if not isinstance(summary, Mapping):
        raise EvalGateError("report missing 'summary' object")
    variants = summary.get("variants")
    if not isinstance(variants, Mapping):
        raise EvalGateError("report 'summary.variants' must be a JSON object")
    return dict(parsed)


# --------------------------------------------------------------------------- #
# Variant metric extraction
# --------------------------------------------------------------------------- #


def _variant_summary(report: Mapping[str, Any], variant: str) -> dict[str, Any] | None:
    """Return the per-variant aggregate dict, or ``None`` if absent."""

    summary = report.get("summary")
    if not isinstance(summary, Mapping):
        return None
    variants = summary.get("variants")
    if not isinstance(variants, Mapping):
        return None
    raw = variants.get(variant)
    if not isinstance(raw, Mapping):
        return None
    return dict(raw)


def _metric(
    variant_summary: Mapping[str, Any] | None,
    key: str,
) -> float | int | None:
    """Extract a numeric metric, returning ``None`` for null/absent."""

    if variant_summary is None:
        return None
    value = variant_summary.get(key)
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _best_baseline_rate(
    report: Mapping[str, Any],
    baseline_variants: list[str],
    key: str,
) -> float | None:
    """Return the maximum rate across available baseline variants.

    For "higher is better" metrics (hit, source hit, exact-id), the best
    baseline is the maximum. For "lower is better" metrics (wrong
    memory), call :func:`_best_baseline_min` instead.

    Returns ``None`` when no baseline variant is present or none carry a
    non-null value for the metric.
    """

    best: float | None = None
    for variant in baseline_variants:
        summary = _variant_summary(report, variant)
        value = _metric(summary, key)
        if value is None:
            continue
        if best is None or value > best:
            best = float(value)
    return best


def _best_baseline_min(
    report: Mapping[str, Any],
    baseline_variants: list[str],
    key: str,
) -> float | None:
    """Return the minimum rate across available baseline variants.

    For "lower is better" metrics (wrong memory), the best baseline is
    the minimum.
    """

    best: float | None = None
    for variant in baseline_variants:
        summary = _variant_summary(report, variant)
        value = _metric(summary, key)
        if value is None:
            continue
        if best is None or value < best:
            best = float(value)
    return best


# --------------------------------------------------------------------------- #
# Check primitives
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class CheckResult:
    """One named gate check with its pass/fail status and context."""

    name: str
    status: str  # "pass" | "fail"
    actual: Any
    threshold: Any
    details: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_ge(
    name: str,
    actual: float | int | None,
    threshold: float | int,
    *,
    comparison_label: str,
) -> CheckResult:
    """Pass if ``actual >= threshold``. Fail closed when actual is None."""

    if actual is None:
        return CheckResult(
            name=name,
            status="fail",
            actual=None,
            threshold=threshold,
            details=f"{comparison_label} metric is null or absent; cannot verify minimum",
        )
    passed = float(actual) >= float(threshold)
    return CheckResult(
        name=name,
        status="pass" if passed else "fail",
        actual=actual,
        threshold=threshold,
        details=f"{comparison_label} {actual} >= {threshold}"
        if passed
        else f"{comparison_label} {actual} < {threshold}",
    )


def _check_le(
    name: str,
    actual: float | int | None,
    threshold: float | int,
    *,
    comparison_label: str,
) -> CheckResult:
    """Pass if ``actual <= threshold``. Fail closed when actual is None."""

    if actual is None:
        return CheckResult(
            name=name,
            status="fail",
            actual=None,
            threshold=threshold,
            details=f"{comparison_label} metric is null or absent; cannot verify maximum",
        )
    passed = float(actual) <= float(threshold)
    return CheckResult(
        name=name,
        status="pass" if passed else "fail",
        actual=actual,
        threshold=threshold,
        details=f"{comparison_label} {actual} <= {threshold}"
        if passed
        else f"{comparison_label} {actual} > {threshold}",
    )


def _check_lift_ge(
    name: str,
    candidate: float | None,
    best_baseline: float | None,
    threshold: float,
    *,
    metric_label: str,
) -> CheckResult:
    """Pass if ``(candidate - best_baseline) >= threshold``.

    Fails closed when either value is None.
    """

    if candidate is None:
        return CheckResult(
            name=name,
            status="fail",
            actual=None,
            threshold=threshold,
            details=f"candidate {metric_label} is null or absent; cannot verify lift",
        )
    if best_baseline is None:
        return CheckResult(
            name=name,
            status="fail",
            actual=None,
            threshold=threshold,
            details=f"no baseline variant provides a {metric_label} value; cannot verify lift",
        )
    lift = float(candidate) - float(best_baseline)
    passed = lift >= float(threshold)
    return CheckResult(
        name=name,
        status="pass" if passed else "fail",
        actual=round(lift, 4),
        threshold=threshold,
        details=f"{metric_label} lift {round(lift, 4)} (= {candidate} - {best_baseline}) >= {threshold}"
        if passed
        else f"{metric_label} lift {round(lift, 4)} (= {candidate} - {best_baseline}) < {threshold}",
    )


def _check_drop_le(
    name: str,
    candidate: float | None,
    best_baseline: float | None,
    threshold: float,
    *,
    metric_label: str,
) -> CheckResult:
    """Pass if ``max(0, best_baseline - candidate) <= threshold``.

    Measures how far the candidate drops below the best baseline.
    Fails closed when either value is None.
    """

    if candidate is None:
        return CheckResult(
            name=name,
            status="fail",
            actual=None,
            threshold=threshold,
            details=f"candidate {metric_label} is null or absent; cannot verify drop",
        )
    if best_baseline is None:
        return CheckResult(
            name=name,
            status="fail",
            actual=None,
            threshold=threshold,
            details=f"no baseline variant provides a {metric_label} value; cannot verify drop",
        )
    drop = max(0.0, float(best_baseline) - float(candidate))
    passed = drop <= float(threshold)
    return CheckResult(
        name=name,
        status="pass" if passed else "fail",
        actual=round(drop, 4),
        threshold=threshold,
        details=f"{metric_label} drop {round(drop, 4)} (= {best_baseline} - {candidate}, floored at 0) <= {threshold}"
        if passed
        else f"{metric_label} drop {round(drop, 4)} (= {best_baseline} - {candidate}) > {threshold}",
    )


# --------------------------------------------------------------------------- #
# Gate evaluation
# --------------------------------------------------------------------------- #


@dataclass
class GateResult:
    """Full gate evaluation result, serializable via :meth:`to_dict`."""

    status: str
    auto_recall_eligible: bool
    candidate_variant: str
    baseline_variants: list[str]
    baselines_found: list[str]
    candidate_present: bool
    thresholds: dict[str, Any]
    checks: list[CheckResult] = field(default_factory=list)
    candidate_metrics: dict[str, Any] = field(default_factory=dict)
    baseline_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GATE_SCHEMA,
            "status": self.status,
            "auto_recall_eligible": self.auto_recall_eligible,
            "candidate_variant": self.candidate_variant,
            "candidate_present": self.candidate_present,
            "baseline_variants": self.baseline_variants,
            "baselines_found": self.baselines_found,
            "thresholds": self.thresholds,
            "checks": [c.to_dict() for c in self.checks],
            "candidate_metrics": self.candidate_metrics,
            "baseline_metrics": self.baseline_metrics,
            "summary": {
                "total_checks": len(self.checks),
                "passed": sum(1 for c in self.checks if c.status == "pass"),
                "failed": sum(1 for c in self.checks if c.status == "fail"),
                "failed_checks": [c.name for c in self.checks if c.status == "fail"],
            },
        }


def evaluate_gate(
    report: Mapping[str, Any],
    *,
    thresholds: GateThresholds | None = None,
    candidate_variant: str = DEFAULT_CANDIDATE_VARIANT,
    baseline_variants: list[str] | None = None,
) -> dict[str, Any]:
    """Evaluate the shadow gate against a Phase 6A evaluator report.

    Parameters
    ----------
    report
        The parsed JSON report from ``hermes qdrant eval --json``.
    thresholds
        Explicit :class:`GateThresholds`. Defaults to
        :meth:`GateThresholds.auto_recall_default`.
    candidate_variant
        The variant to promote. Defaults to ``hybrid``.
    baseline_variants
        Variants to compare against. Defaults to
        :data:`DEFAULT_BASELINE_VARIANTS`.

    Returns
    -------
    dict
        A ``qdrant_eval_gate.v1`` JSON-serializable result. See the
        module docstring for the schema.
    """

    if thresholds is None:
        thresholds = GateThresholds.auto_recall_default()
    if baseline_variants is None:
        baseline_variants = list(DEFAULT_BASELINE_VARIANTS)

    # --- Candidate presence -------------------------------------------------
    candidate_summary = _variant_summary(report, candidate_variant)
    candidate_present = candidate_summary is not None

    # Collect candidate metrics (may contain None values).
    candidate_metrics: dict[str, Any] = {}
    if candidate_summary is not None:
        for key in (
            _METRIC_CASE_COUNT,
            _METRIC_SCORED_COUNT,
            _METRIC_ERRORED_COUNT,
            _METRIC_HIT_AT_K_RATE,
            _METRIC_SOURCE_HIT_AT_K_RATE,
            _METRIC_EXACT_ID_RATE,
            _METRIC_WRONG_MEMORY_RATE,
            _METRIC_LATENCY_P95,
            _METRIC_LATENCY_BUDGET_PASS,
        ):
            candidate_metrics[key] = _metric(candidate_summary, key)

    # --- Baseline presence --------------------------------------------------
    baselines_found: list[str] = []
    baseline_metrics: dict[str, dict[str, Any]] = {}
    for variant in baseline_variants:
        summary = _variant_summary(report, variant)
        if summary is None:
            continue
        baselines_found.append(variant)
        metrics: dict[str, Any] = {}
        for key in (
            _METRIC_HIT_AT_K_RATE,
            _METRIC_SOURCE_HIT_AT_K_RATE,
            _METRIC_EXACT_ID_RATE,
            _METRIC_WRONG_MEMORY_RATE,
        ):
            metrics[key] = _metric(summary, key)
        baseline_metrics[variant] = metrics

    # --- Checks -------------------------------------------------------------
    checks: list[CheckResult] = []

    if not candidate_present:
        # Fail closed: the candidate variant is missing entirely.
        checks.append(
            CheckResult(
                name="candidate_present",
                status="fail",
                actual=False,
                threshold=True,
                details=f"candidate variant {candidate_variant!r} not found in report",
            ),
        )
        result = GateResult(
            status="fail",
            auto_recall_eligible=False,
            candidate_variant=candidate_variant,
            baseline_variants=baseline_variants,
            baselines_found=baselines_found,
            candidate_present=False,
            thresholds=thresholds.to_dict(),
            checks=checks,
            candidate_metrics={},
            baseline_metrics=baseline_metrics,
        )
        return result.to_dict()

    # Count / sample-size checks.
    checks.append(
        _check_ge(
            "case_count",
            candidate_metrics.get(_METRIC_CASE_COUNT),
            thresholds.min_case_count,
            comparison_label="candidate case_count",
        ),
    )
    checks.append(
        _check_ge(
            "scored_count",
            candidate_metrics.get(_METRIC_SCORED_COUNT),
            thresholds.min_scored_count,
            comparison_label="candidate scored_count",
        ),
    )
    checks.append(
        _check_le(
            "errored_count",
            candidate_metrics.get(_METRIC_ERRORED_COUNT),
            thresholds.max_errored_count,
            comparison_label="candidate errored_count",
        ),
    )

    # Absolute hit / source-hit floors.
    checks.append(
        _check_ge(
            "candidate_hit_at_k",
            candidate_metrics.get(_METRIC_HIT_AT_K_RATE),
            thresholds.min_candidate_hit_at_k,
            comparison_label="candidate hit_at_k_rate",
        ),
    )
    checks.append(
        _check_ge(
            "candidate_source_hit_at_k",
            candidate_metrics.get(_METRIC_SOURCE_HIT_AT_K_RATE),
            thresholds.min_candidate_source_hit_at_k,
            comparison_label="candidate source_hit_at_k_rate",
        ),
    )

    # Lift checks (candidate vs best baseline).
    best_baseline_hit = _best_baseline_rate(report, baseline_variants, _METRIC_HIT_AT_K_RATE)
    best_baseline_source_hit = _best_baseline_rate(
        report, baseline_variants, _METRIC_SOURCE_HIT_AT_K_RATE,
    )
    best_baseline_exact_id = _best_baseline_rate(report, baseline_variants, _METRIC_EXACT_ID_RATE)
    best_baseline_wrong = _best_baseline_min(report, baseline_variants, _METRIC_WRONG_MEMORY_RATE)

    checks.append(
        _check_lift_ge(
            "hit_at_k_lift",
            candidate_metrics.get(_METRIC_HIT_AT_K_RATE),
            best_baseline_hit,
            thresholds.min_hit_at_k_lift,
            metric_label="hit_at_k_rate",
        ),
    )
    checks.append(
        _check_lift_ge(
            "source_hit_at_k_lift",
            candidate_metrics.get(_METRIC_SOURCE_HIT_AT_K_RATE),
            best_baseline_source_hit,
            thresholds.min_source_hit_at_k_lift,
            metric_label="source_hit_at_k_rate",
        ),
    )
    checks.append(
        _check_drop_le(
            "exact_id_drop",
            candidate_metrics.get(_METRIC_EXACT_ID_RATE),
            best_baseline_exact_id,
            thresholds.max_exact_id_drop,
            metric_label="exact_identifier_hit_rate",
        ),
    )

    # Wrong-memory checks (absolute + regression).
    checks.append(
        _check_le(
            "wrong_memory_rate",
            candidate_metrics.get(_METRIC_WRONG_MEMORY_RATE),
            thresholds.max_wrong_memory_rate,
            comparison_label="candidate wrong_memory_rate",
        ),
    )
    checks.append(
        _check_drop_le(
            "wrong_memory_regression",
            best_baseline_wrong,
            candidate_metrics.get(_METRIC_WRONG_MEMORY_RATE),
            thresholds.max_wrong_memory_regression,
            metric_label="wrong_memory_rate",
        ),
    )

    # Latency checks.
    checks.append(
        _check_le(
            "latency_p95",
            candidate_metrics.get(_METRIC_LATENCY_P95),
            thresholds.max_latency_p95_ms,
            comparison_label="candidate latency_ms_p95",
        ),
    )
    checks.append(
        _check_ge(
            "latency_budget_pass_rate",
            candidate_metrics.get(_METRIC_LATENCY_BUDGET_PASS),
            thresholds.min_latency_budget_pass_rate,
            comparison_label="candidate latency_budget_pass_rate",
        ),
    )

    # --- Aggregate ----------------------------------------------------------
    all_passed = all(c.status == "pass" for c in checks)

    result = GateResult(
        status="pass" if all_passed else "fail",
        auto_recall_eligible=all_passed,
        candidate_variant=candidate_variant,
        baseline_variants=baseline_variants,
        baselines_found=baselines_found,
        candidate_present=True,
        thresholds=thresholds.to_dict(),
        checks=checks,
        candidate_metrics=candidate_metrics,
        baseline_metrics=baseline_metrics,
    )
    return result.to_dict()


# --------------------------------------------------------------------------- #
# Human summary
# --------------------------------------------------------------------------- #


def format_human_summary(result: Mapping[str, Any]) -> str:
    """Compact human summary for non-JSON CLI output.

    Never echoes raw query text or packets; the gate only reads aggregate
    metrics so there is nothing sensitive to redact here.
    """

    lines: list[str] = [
        "Phase 6F eval-gate report",
        f"status: {result.get('status', 'fail')}",
        f"auto_recall_eligible: {result.get('auto_recall_eligible', False)}",
        f"candidate_variant: {result.get('candidate_variant', '?')}",
        f"baselines_found: {', '.join(result.get('baselines_found', [])) or '(none)'}",
    ]
    summary = result.get("summary", {})
    if isinstance(summary, Mapping):
        lines.append(
            f"checks: {summary.get('passed', 0)}/{summary.get('total_checks', 0)} passed, "
            f"{summary.get('failed', 0)} failed",
        )
        failed_names = summary.get("failed_checks", [])
        if failed_names:
            lines.append(f"failed: {', '.join(failed_names)}")
    lines.append("")
    lines.append("checks:")
    for check in result.get("checks", []):
        if not isinstance(check, Mapping):
            continue
        status_marker = "[OK]" if check.get("status") == "pass" else "[FAIL]"
        lines.append(f"  {status_marker} {check.get('name', '?')}: {check.get('details', '')}")
    return "\n".join(lines) + "\n"
