"""Phase 6A offline evaluation core for Qdrant memory retrieval.

This module is **stdlib-only** and **offline** by design. It scores
already-captured retrieval packets (JSONL run rows) against operator-
authored eval cases (JSONL case rows) and produces deterministic per-
``(case, variant)`` metrics plus per-variant aggregates.

Hard rules enforced here:

* No live Qdrant client. No Qdrant mutation. No provider factory.
* No imports that initialize Hermes provider state.
* Pure functions over dicts/lists/JSONL rows.
* Reports never echo raw query text from packets; eval case ``query``
  strings are operator-authored local artifacts and stay local.
* ``wrong_memory`` (poison) detection inspects all normalized emitted
  items, not just top-k, because a forbidden memory anywhere in the
  emitted packet is operationally important.
* Hit-style metrics are lane-aware: for Phase 5 grouped retrieve
  packets we take the first ``k`` from each lane and union them, so
  RAPTOR summaries, cited leaves, exact hits, and graph relations are
  not unfairly penalized just because the JSON envelope lists some
  lanes first.
* Errors (e.g. invalid JSONL, missing required fields) surface as
  :class:`EvaluationError` with a line number so the CLI can emit a
  nonzero exit with a user-facing message.

This module is Phase 6A scope: report-only, no live shadow
collection, no auto-recall switching, no Qdrant v2 migration, no
background jobs. Live shadow collectors and auto-recall defaults are
explicitly deferred to Phase 6B per ``docs/RAPTOR_EVAL.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default top_k when caller does not override. Matches the Phase 5
# hybrid retrieve default and the CLI default so reports are
# comparable to live usage out of the box.
DEFAULT_TOP_K: int = 5

# Default latency budget (ms). Phase 6A recommends a conservative
# local offline budget because runs here are captured from existing
# tooling, not generated live by the evaluator. The CLI exposes this
# as a flag so operators can tune per their environment.
DEFAULT_LATENCY_BUDGET_MS: int = 750

# Lane order used to flatten a Phase 5 grouped retrieve packet into a
# stable ordered stream. Top-k within each lane is preserved.
_LANE_ORDER: tuple[str, ...] = (
    "exact_hits",
    "summaries",
    "cited_leaves",
    "graph_relations",
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class EvaluationError(ValueError):
    """Raised when an eval input is structurally invalid.

    The CLI translates this into a nonzero exit with a user-facing
    line-numbered message. The class is intentionally small: just a
    message string with optional line number.
    """

    def __init__(self, message: str, *, line: int | None = None) -> None:
        prefix = f"line {line}: " if line is not None else ""
        super().__init__(f"{prefix}{message}")


# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------


_CASE_REQUIRED_FIELDS: tuple[str, ...] = ("case_id", "query")
_RUN_REQUIRED_STRING_FIELDS: tuple[str, ...] = ("case_id", "variant")
_RUN_REQUIRED_FIELDS: tuple[str, ...] = ("case_id", "variant", "packet")

_CASE_OPTIONAL_LIST_FIELDS: tuple[str, ...] = (
    "expected_point_ids",
    "expected_source_uris",
    "expected_file_paths",
    "expected_terms",
    "forbidden_point_ids",
    "forbidden_source_uris",
    "forbidden_file_paths",
    "forbidden_terms",
    "tags",
)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    out: list[str] = []
    for item in value:
        if isinstance(item, str):
            text = item.strip()
            if text:
                out.append(text)
    return out


def _validate_case_dict(raw: Any, *, line: int | None) -> dict[str, Any]:
    """Validate and project a raw eval-case dict.

    The output is a fresh dict so downstream code never mutates the
    caller's input. Unknown keys are preserved on the returned dict
    because some operators may want to attach ``notes`` or ``tags``
    for later human review.
    """

    if not isinstance(raw, Mapping):
        raise EvaluationError("eval case must be a JSON object", line=line)
    out: dict[str, Any] = dict(raw)
    for field_name in _CASE_REQUIRED_FIELDS:
        value = out.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationError(
                f"eval case missing required string field {field_name!r}",
                line=line,
            )
    for field_name in _CASE_OPTIONAL_LIST_FIELDS:
        value = out.get(field_name, [])
        if not _is_str_list(value):
            raise EvaluationError(
                f"eval case field {field_name!r} must be a list of strings",
                line=line,
            )
        out[field_name] = [str(item) for item in value]
    if "notes" in out and out["notes"] is not None and not isinstance(out["notes"], str):
        raise EvaluationError(
            "eval case field 'notes' must be a string when present",
            line=line,
        )
    if "domain" in out and out["domain"] is not None and not isinstance(out["domain"], str):
        raise EvaluationError(
            "eval case field 'domain' must be a string when present",
            line=line,
        )
    return out


def _validate_run_dict(raw: Any, *, line: int | None) -> dict[str, Any]:
    """Validate and project a raw eval-run dict.

    Run rows carry a captured ``packet`` (dict or list) which is
    never deep-cloned; normalization must not mutate it.
    """

    if not isinstance(raw, Mapping):
        raise EvaluationError("eval run must be a JSON object", line=line)
    out: dict[str, Any] = dict(raw)
    for field_name in _RUN_REQUIRED_STRING_FIELDS:
        value = out.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise EvaluationError(
                f"eval run missing required string field {field_name!r}",
                line=line,
            )
    if "packet" not in out:
        raise EvaluationError(
            "eval run missing required field 'packet'",
            line=line,
        )
    packet = out["packet"]
    if not isinstance(packet, (Mapping, list)):
        raise EvaluationError(
            "eval run 'packet' must be a JSON object or list",
            line=line,
        )
    latency = out.get("latency_ms", None)
    if latency is not None:
        try:
            out["latency_ms"] = float(latency)
        except (TypeError, ValueError):
            raise EvaluationError(
                "eval run 'latency_ms' must be a number when present",
                line=line,
            )
    err = out.get("error", None)
    if err is not None and not isinstance(err, str):
        raise EvaluationError(
            "eval run 'error' must be a string when present",
            line=line,
        )
    return out


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def load_jsonl(path: str) -> list[dict[str, Any]]:
    """Read a JSONL file and return a list of parsed row dicts.

    Blank lines and lines starting with ``#`` are ignored so eval
    artifacts can carry human comments. The first JSON parse error is
    re-raised as :class:`EvaluationError` annotated with the
    1-indexed line number so the CLI can report a useful message.
    """

    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as exc:
        raise EvaluationError(f"failed to read JSONL file: {exc}") from exc
    return parse_jsonl_text(text)


def parse_jsonl_text(text: str) -> list[dict[str, Any]]:
    """Parse a JSONL string into a list of row dicts.

    Splits the input on newlines, skipping blank lines and comment
    lines (those whose first non-whitespace character is ``#``). The
    first JSON parse error is re-raised as :class:`EvaluationError`
    with a 1-indexed line number.
    """

    rows: list[dict[str, Any]] = []
    for index, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvaluationError(
                f"invalid JSONL row: {exc.msg}", line=index,
            ) from exc
        if not isinstance(parsed, Mapping):
            raise EvaluationError(
                "JSONL row must be a JSON object",
                line=index,
            )
        rows.append(dict(parsed))
    return rows


# ---------------------------------------------------------------------------
# Candidate normalization
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NormalizedCandidate:
    """Stable, normalized view of a single emitted retrieval item.

    The evaluator never mutates the original packet; it builds a list
    of these dataclasses and discards them when the function returns.
    """

    lane: str
    rank: int
    point_id: str = ""
    parent_point_id: str = ""
    source_uri: str = ""
    file_path: str = ""
    heading: str = ""
    text: str = ""
    score: float | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return str(value)


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _candidate_from_mapping(
    item: Mapping[str, Any],
    *,
    lane: str,
    rank: int,
) -> NormalizedCandidate:
    """Project a dict item into a :class:`NormalizedCandidate`.

    Field selection rules:

    * ``point_id`` accepts ``point_id`` or ``id``.
    * ``source_uri`` accepts ``source_uri`` or ``source``.
    * ``file_path`` accepts ``file_path`` or ``path``.
    * ``text`` is the first non-empty value among
      ``text``/``snippet``/``excerpt``/``preview``/``summary_text``/
      ``content``.
    * ``score`` accepts ``score``/``final_score``/``_rrf_score``/
      ``qdrant_score``/``graph_score`` in that priority.
    """

    point_id = _coerce_text(item.get("point_id") or item.get("id"))
    parent_point_id = _coerce_text(item.get("parent_point_id") or item.get("parent_id"))
    source_uri = _coerce_text(item.get("source_uri") or item.get("source"))
    file_path = _coerce_text(item.get("file_path") or item.get("path"))
    heading = _coerce_text(item.get("heading"))

    text_keys = (
        "text",
        "snippet",
        "excerpt",
        "preview",
        "summary_text",
        "content",
    )
    text = ""
    for key in text_keys:
        candidate = item.get(key)
        if isinstance(candidate, str) and candidate:
            text = candidate
            break
    if not text:
        text = _coerce_text(item.get("text"))

    score = None
    for key in ("score", "final_score", "_rrf_score", "qdrant_score", "graph_score"):
        if key in item:
            score = _coerce_float(item.get(key))
            if score is not None:
                break

    return NormalizedCandidate(
        lane=lane,
        rank=rank,
        point_id=point_id,
        parent_point_id=parent_point_id,
        source_uri=source_uri,
        file_path=file_path,
        heading=heading,
        text=text,
        score=score,
        extra=dict(item),
    )


def normalize_packet(packet: Any) -> list[NormalizedCandidate]:
    """Normalize a captured retrieval packet into a flat, ordered list.

    Supports three packet shapes:

    1. Phase 5 ``qdrant_memory_retrieve`` dict with a ``results``
       sub-object whose ``exact_hits``/``summaries``/``cited_leaves``/
       ``graph_relations`` keys are lists.
    2. Generic search-like dict whose values are all lists (legacy
       shape).
    3. A bare list of items (legacy list packet). All items go to the
       ``legacy`` lane.

    The function never mutates the input. Order is stable: lanes
    flatten in ``_LANE_ORDER`` for shapes 1/2, and lists preserve the
    caller's order.
    """

    if isinstance(packet, list):
        return [
            _candidate_from_mapping(
                item,
                lane="legacy",
                rank=index,
            )
            if isinstance(item, Mapping)
            else _candidate_from_mapping(
                {"text": _coerce_text(item)},
                lane="legacy",
                rank=index,
            )
            for index, item in enumerate(packet, start=1)
        ]
    if not isinstance(packet, Mapping):
        return []

    # Phase 5 grouped retrieve shape.
    if "results" in packet and isinstance(packet.get("results"), Mapping):
        results = packet["results"]
        out: list[NormalizedCandidate] = []
        for lane in _LANE_ORDER:
            bucket = results.get(lane)
            if isinstance(bucket, list):
                for index, item in enumerate(bucket, start=1):
                    if not isinstance(item, Mapping):
                        continue
                    out.append(_candidate_from_mapping(item, lane=lane, rank=index))
        # Also surface a generic ``legacy`` lane if the results dict
        # already contains flat candidate lists outside the four
        # well-known buckets.
        for key, bucket in results.items():
            if key in _LANE_ORDER:
                continue
            if isinstance(bucket, list):
                for index, item in enumerate(bucket, start=1):
                    if not isinstance(item, Mapping):
                        continue
                    out.append(_candidate_from_mapping(item, lane="legacy", rank=index))
        return out

    # Generic dict-of-lists shape (e.g. capture harness).
    #
    # Phase 6A fix3: a top-level packet may carry *both* well-known
    # lane buckets (e.g. ``exact_hits``) and additional unknown list
    # buckets (``extra_lane``, ``plugin_output``, ...). The original
    # implementation returned early as soon as it saw any known lane,
    # silently dropping the unknown buckets. That hid poison candidates
    # from ``wrong_memory`` scoring. We now process known lanes first
    # and then continue scanning the remaining top-level list-valued
    # keys, projecting each as a ``legacy`` candidate. Known lane
    # keys are never re-processed as legacy, so the well-known lanes
    # keep their lane label and the unknown buckets are not lost.
    out_lists: list[NormalizedCandidate] = []
    for lane in _LANE_ORDER:
        bucket = packet.get(lane)
        if isinstance(bucket, list):
            for index, item in enumerate(bucket, start=1):
                if not isinstance(item, Mapping):
                    continue
                out_lists.append(_candidate_from_mapping(item, lane=lane, rank=index))
    # Any remaining top-level list-valued key (not in ``_LANE_ORDER``)
    # becomes ``legacy`` so unknown lanes are still scored and any
    # forbidden term embedded there can trip ``wrong_memory``.
    for key, bucket in packet.items():
        if key in _LANE_ORDER:
            continue
        if isinstance(bucket, list):
            for index, item in enumerate(bucket, start=1):
                if isinstance(item, Mapping):
                    out_lists.append(
                        _candidate_from_mapping(item, lane="legacy", rank=index)
                    )
                else:
                    out_lists.append(
                        _candidate_from_mapping(
                            {"text": _coerce_text(item)},
                            lane="legacy",
                            rank=index,
                        )
                    )
    return out_lists


# ---------------------------------------------------------------------------
# Match predicates
# ---------------------------------------------------------------------------


def _terms_present_in_candidate(
    candidate: NormalizedCandidate,
    terms: Iterable[str],
) -> list[str]:
    """Return the subset of ``terms`` that appear in the candidate's text fields.

    Match is case-insensitive substring over the candidate's visible
    text/heading/source/file fields. Empty candidate text or empty
    terms short-circuit cleanly.
    """

    if not terms:
        return []
    haystacks = [
        candidate.text or "",
        candidate.heading or "",
        candidate.source_uri or "",
        candidate.file_path or "",
    ]
    matched: list[str] = []
    for term in terms:
        if not term:
            continue
        lowered = term.lower()
        for hay in haystacks:
            if lowered in hay.lower():
                matched.append(term)
                break
    return matched


def _candidate_matches_expected_point(
    candidate: NormalizedCandidate,
    expected_point_ids: list[str],
) -> bool:
    """Return True iff an expected point id equals the candidate's emitted id.

    The metric name/contract is "exact identifier appears as an emitted
    point id". Matching against ``parent_point_id`` would let a leaf
    citation satisfy a top-level point label, which is too loose for
    this metric and would also double-count under the source/term
    handles. Parent refs are still meaningful for poison detection
    (``_candidate_forbidden``) and for the per-row ``matched_expected``
    debug payload.
    """
    if not expected_point_ids:
        return False
    for pid in expected_point_ids:
        if pid and candidate.point_id == pid:
            return True
    return False


def _candidate_matched_expected_source_labels(
    candidate: NormalizedCandidate,
    expected_source_uris: list[str],
    expected_file_paths: list[str],
) -> list[str]:
    """Return the expected source labels that actually matched this candidate.

    The matched labels are the *expected* URI/path strings from the case,
    not arbitrary emitted candidate fields. This prevents an unrelated
    candidate ``source_uri`` from being recorded in
    ``matched_expected.sources`` when only ``expected_file_paths`` matched,
    and prevents two candidates that share the same expected file path
    from inflating ``useful_topk_count`` with their distinct emitted
    ``source_uri`` values.

    Both label kinds are checked and deduped per candidate so a single
    candidate that satisfies both an expected source URI and an expected
    file path returns both labels (in source-URI-then-file-path order).
    Empty expected entries are skipped so a degenerate ``[""]`` does
    not accidentally match a candidate with empty ``source_uri`` or
    ``file_path``.
    """

    matched: list[str] = []
    seen: set[str] = set()
    for source in expected_source_uris:
        if source and candidate.source_uri == source and source not in seen:
            matched.append(source)
            seen.add(source)
    for path in expected_file_paths:
        if path and candidate.file_path == path and path not in seen:
            matched.append(path)
            seen.add(path)
    return matched


def _candidate_matches_expected_terms(
    candidate: NormalizedCandidate,
    expected_terms: list[str],
) -> list[str]:
    return _terms_present_in_candidate(candidate, expected_terms)


def _candidate_forbidden(
    candidate: NormalizedCandidate,
    *,
    forbidden_point_ids: list[str],
    forbidden_source_uris: list[str],
    forbidden_file_paths: list[str],
    forbidden_terms: list[str],
) -> dict[str, list[str]]:
    """Return any forbidden matches the candidate triggers, by reason.

    The returned dict is empty when the candidate is clean.
    """

    matches: dict[str, list[str]] = {}
    for pid in forbidden_point_ids:
        if pid and (candidate.point_id == pid or candidate.parent_point_id == pid):
            matches.setdefault("forbidden_point_ids", []).append(pid)
    for source in forbidden_source_uris:
        if source and candidate.source_uri == source:
            matches.setdefault("forbidden_source_uris", []).append(source)
    for path in forbidden_file_paths:
        if path and candidate.file_path == path:
            matches.setdefault("forbidden_file_paths", []).append(path)
    terms = _terms_present_in_candidate(candidate, forbidden_terms)
    if terms:
        matches["forbidden_terms"] = terms
    return matches


# ---------------------------------------------------------------------------
# Lane-aware top-k helpers
# ---------------------------------------------------------------------------


def _lane_top_k(
    candidates: list[NormalizedCandidate],
    top_k: int,
) -> list[NormalizedCandidate]:
    """Return the first ``top_k`` candidates of a single lane, in order."""

    if top_k <= 0:
        return []
    return list(candidates[: max(0, top_k)])


def _lane_aware_top_k(
    candidates: list[NormalizedCandidate],
    top_k: int,
) -> list[NormalizedCandidate]:
    """Take the first ``top_k`` from each lane and union by point_id.

    Phase 5 retrieve returns ``top_k`` per bucket, not a single
    globally-fused ranking. To avoid unfairly penalizing exact hits
    that fall into a non-first bucket we take the first ``top_k`` per
    lane and dedupe by ``point_id`` (first occurrence wins, so
    ordering stays stable across variants). Candidates without a
    non-empty ``point_id`` keep their ``lane::rank`` fallback so the
    legacy / unkeyed stream does not collapse into a single
    pseudo-id.
    """

    buckets: dict[str, list[NormalizedCandidate]] = {lane: [] for lane in _LANE_ORDER}
    legacy: list[NormalizedCandidate] = []
    for cand in candidates:
        if cand.lane in buckets:
            buckets[cand.lane].append(cand)
        else:
            legacy.append(cand)
    union: list[NormalizedCandidate] = []
    seen: set[str] = set()
    for lane in _LANE_ORDER:
        for cand in _lane_top_k(buckets[lane], top_k):
            key = _dedupe_key(cand)
            if key in seen:
                continue
            seen.add(key)
            union.append(cand)
    for cand in _lane_top_k(legacy, top_k):
        key = _dedupe_key(cand)
        if key in seen:
            continue
        seen.add(key)
        union.append(cand)
    return union


def _dedupe_key(cand: NormalizedCandidate) -> str:
    """Return the dedupe key used by :func:`_lane_aware_top_k`.

    A non-empty ``point_id`` dedupes globally across lanes (so the
    same point emitted in ``summaries`` and ``cited_leaves`` is
    counted once). Candidates without a ``point_id`` fall back to
    ``lane::rank`` so anonymous legacy items still dedupe on their
    own lane position and don't collide with each other across
    lanes.
    """
    if cand.point_id:
        return cand.point_id
    return f"{cand.lane}::r{cand.rank}"


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(100.0 * numerator / denominator, 4)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    if n % 2 == 1:
        return float(ordered[n // 2])
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _percentile(values: list[float], pct: float) -> float | None:
    """Return the linear-interpolation percentile of ``values``.

    Mirrors numpy's default (``linear``) interpolation so an operator
    can cross-check a report against a notebook. ``pct`` is expressed
    in [0, 100]. Returns ``None`` for empty input.
    """

    if not values:
        return None
    if not 0 <= pct <= 100:
        raise EvaluationError(f"percentile must be in [0, 100], got {pct}")
    ordered = sorted(values)
    n = len(ordered)
    if n == 1:
        return float(ordered[0])
    rank = (pct / 100.0) * (n - 1)
    lower = int(rank)
    upper = min(lower + 1, n - 1)
    weight = rank - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _score_zoom_efficiency(
    *,
    useful_topk: int,
    context_chars: int,
) -> float:
    """Phase 6A deterministic zoom-efficiency proxy.

    Formula::

        zoom_efficiency = useful_topk / max(1, context_chars / 1000)

    This rewards runs whose top-k contains a high fraction of
    expected evidence relative to the character budget they emitted.
    A run that uses little context for many useful hits scores
    higher than a run that emits a large context for the same
    hits. The denominator floor of 1 prevents division by zero for
    empty packets and keeps the metric finite for tiny packets.
    """

    if context_chars <= 0:
        return float(useful_topk)
    return round(useful_topk / max(1.0, context_chars / 1000.0), 6)


def score_case_run(
    case: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    top_k: int = DEFAULT_TOP_K,
    latency_budget_ms: float | None = DEFAULT_LATENCY_BUDGET_MS,
) -> dict[str, Any]:
    """Score a single ``(case, run)`` pair.

    Returns a JSON-serializable dict with per-run metric booleans,
    numerators/denominators, latency, and compact match handles. The
    output is suitable for both direct inspection and the per-variant
    aggregator. If the run row carries an ``error`` string, the row
    is marked errored and numeric metrics are emitted as ``null`` so
    the aggregator can drop it from rate denominators.
    """

    if top_k <= 0:
        raise EvaluationError("top_k must be positive")

    packet = run.get("packet")
    candidates = normalize_packet(packet)
    lane_topk = _lane_aware_top_k(candidates, top_k)
    context_chars = sum(len(cand.text) for cand in candidates)

    expected_point_ids = _as_str_list(case.get("expected_point_ids", []))
    expected_source_uris = _as_str_list(case.get("expected_source_uris", []))
    expected_file_paths = _as_str_list(case.get("expected_file_paths", []))
    expected_terms = _as_str_list(case.get("expected_terms", []))
    forbidden_point_ids = _as_str_list(case.get("forbidden_point_ids", []))
    forbidden_source_uris = _as_str_list(case.get("forbidden_source_uris", []))
    forbidden_file_paths = _as_str_list(case.get("forbidden_file_paths", []))
    forbidden_terms = _as_str_list(case.get("forbidden_terms", []))

    error = run.get("error")
    if isinstance(error, str) and error.strip():
        # Phase 6A privacy contract: errored rows must not echo the raw
        # captured error string. Captured errors from the retrieval
        # harness can include the raw query, Qdrant request/response
        # details, or packet/body snippets, and reports (which are
        # persisted to disk via ``--json``) are not the right place to
        # surface that. We keep the operational signal
        # (``errored: True``, ``error_present: True``) and emit a
        # constant sentinel for the ``error`` slot so any downstream
        # code that pattern-matches on the key still works, but the
        # raw text never reaches a report. ``error_redacted`` is the
        # explicit flag downstream tooling should check.
        return {
            "case_id": str(case.get("case_id", "")),
            "variant": str(run.get("variant", "")),
            "errored": True,
            "error": "<redacted>",
            "error_present": True,
            "error_redacted": True,
            "hit_at_k": None,
            "source_hit_at_k": None,
            "exact_identifier_hit": None,
            "wrong_memory": None,
            "useful_topk_count": 0,
            "context_chars": context_chars,
            "zoom_efficiency": 0.0,
            "latency_ms": run.get("latency_ms"),
            "latency_budget_met": None,
            "matched_expected": {
                "point_ids": [],
                "sources": [],
                "terms": [],
            },
            "wrong_reasons": {},
            "emitted_count": len(candidates),
            "topk_count": len(lane_topk),
        }

    matched_point_ids: list[str] = []
    matched_sources: list[str] = []
    matched_terms: list[str] = []
    useful_topk: set[tuple[str, str]] = set()
    for cand in lane_topk:
        if _candidate_matches_expected_point(cand, expected_point_ids):
            pid = cand.point_id or cand.parent_point_id
            matched_point_ids.append(pid)
            useful_topk.add(("point_id", pid))
        # Record only the expected labels that actually matched this
        # candidate (not arbitrary emitted candidate fields), so a
        # file_path-only match never surfaces a nonmatching source_uri,
        # and so two candidates that share the same expected file
        # path but emit distinct unrelated source_uri values cannot
        # inflate ``useful_topk_count`` or ``zoom_efficiency``.
        matched_source_labels = _candidate_matched_expected_source_labels(
            cand, expected_source_uris, expected_file_paths,
        )
        for source_label in matched_source_labels:
            if source_label not in matched_sources:
                matched_sources.append(source_label)
            useful_topk.add(("source", source_label))
        terms = _candidate_matches_expected_terms(cand, expected_terms)
        for term in terms:
            if term not in matched_terms:
                matched_terms.append(term)
            useful_topk.add(("term", term))

    hit_at_k = bool(matched_point_ids or matched_sources or matched_terms)
    if expected_source_uris or expected_file_paths:
        source_hit_at_k = bool(matched_sources)
    else:
        source_hit_at_k = None
    if expected_point_ids:
        exact_identifier_hit = bool(matched_point_ids)
    else:
        exact_identifier_hit = None

    wrong_reasons: dict[str, list[str]] = {}
    for cand in candidates:
        matches = _candidate_forbidden(
            cand,
            forbidden_point_ids=forbidden_point_ids,
            forbidden_source_uris=forbidden_source_uris,
            forbidden_file_paths=forbidden_file_paths,
            forbidden_terms=forbidden_terms,
        )
        for reason, values in matches.items():
            existing = wrong_reasons.setdefault(reason, [])
            for value in values:
                if value not in existing:
                    existing.append(value)
    wrong_memory = bool(wrong_reasons)

    latency_ms = run.get("latency_ms")
    if isinstance(latency_ms, (int, float)) and latency_budget_ms is not None:
        latency_budget_met = float(latency_ms) <= float(latency_budget_ms)
    else:
        latency_budget_met = None

    zoom_efficiency = _score_zoom_efficiency(
        useful_topk=len(useful_topk),
        context_chars=context_chars,
    )

    return {
        "case_id": str(case.get("case_id", "")),
        "variant": str(run.get("variant", "")),
        "errored": False,
        "hit_at_k": hit_at_k,
        "source_hit_at_k": source_hit_at_k,
        "exact_identifier_hit": exact_identifier_hit,
        "wrong_memory": wrong_memory,
        "useful_topk_count": len(useful_topk),
        "context_chars": context_chars,
        "zoom_efficiency": zoom_efficiency,
        "latency_ms": latency_ms if isinstance(latency_ms, (int, float)) else None,
        "latency_budget_met": latency_budget_met,
        "matched_expected": {
            "point_ids": matched_point_ids,
            "sources": matched_sources,
            "terms": matched_terms,
        },
        "wrong_reasons": wrong_reasons,
        "emitted_count": len(candidates),
        "topk_count": len(lane_topk),
    }


def aggregate_scores(scored_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-row scores by ``variant``.

    Aggregation rules:

    * Errored rows are counted in ``errored_count`` and excluded from
      all rate denominators so a flaky harness cannot deflate a
      variant's hit rate.
    * ``hit_at_k_rate`` is the percent of non-errored rows where
      ``hit_at_k`` is True.
    * ``source_hit_at_k_rate`` is computed over rows that actually
      carry source/file labels (``source_hit_at_k`` is non-null).
      Rows whose case has no ``expected_source_uris`` and no
      ``expected_file_paths`` are excluded so a term-only case does
      not silently drag the rate down. The labeled denominator is
      surfaced as ``source_hit_labeled_count``; ``source_hit_at_k_rate``
      is ``None`` when no non-errored row carried source/file
      labels.
    * ``exact_identifier_hit_rate`` is ``None`` when no non-errored
      row carries expected point IDs; otherwise it is the percent of
      labeled rows where ``exact_identifier_hit`` is True. The
      labeled denominator is surfaced as
      ``exact_identifier_labeled_count``.
    * ``wrong_memory_rate`` is the percent of non-errored rows with
      ``wrong_memory`` True.
    * ``context_chars``/``zoom_efficiency`` are averaged across non-
      errored rows.
    * Latency is summarized as median/p95 over rows that carry a
      numeric ``latency_ms``; ``latency_budget_pass_rate`` is the
      percent of latency-bearing rows that met the configured budget.
    """

    variants: dict[str, list[dict[str, Any]]] = {}
    for row in scored_rows:
        variant = str(row.get("variant", "") or "")
        variants.setdefault(variant, []).append(row)

    aggregates: dict[str, Any] = {}
    for variant, rows in variants.items():
        non_errored = [r for r in rows if not r.get("errored")]
        errored = [r for r in rows if r.get("errored")]
        hit_count = sum(1 for r in non_errored if r.get("hit_at_k"))
        source_rows = [
            r for r in non_errored if r.get("source_hit_at_k") is not None
        ]
        source_hit_count = sum(
            1 for r in source_rows if r.get("source_hit_at_k")
        )
        exact_rows = [r for r in non_errored if r.get("exact_identifier_hit") is not None]
        exact_count = sum(1 for r in exact_rows if r.get("exact_identifier_hit"))
        wrong_count = sum(1 for r in non_errored if r.get("wrong_memory"))
        context_chars_values = [
            int(r.get("context_chars") or 0) for r in non_errored
        ]
        zoom_values = [
            float(r.get("zoom_efficiency") or 0.0) for r in non_errored
        ]
        latency_values = [
            float(r["latency_ms"]) for r in non_errored
            if isinstance(r.get("latency_ms"), (int, float))
        ]
        latency_budget_rows = [
            r for r in non_errored
            if r.get("latency_budget_met") is not None
        ]
        latency_budget_pass = sum(
            1 for r in latency_budget_rows if r.get("latency_budget_met")
        )

        non_errored_count = len(non_errored)
        aggregates[variant] = {
            "case_count": len(rows),
            "scored_count": non_errored_count,
            "errored_count": len(errored),
            "hit_at_k_rate": _pct(hit_count, non_errored_count),
            "source_hit_at_k_rate": (
                _pct(source_hit_count, len(source_rows)) if source_rows else None
            ),
            "source_hit_labeled_count": len(source_rows),
            "exact_identifier_hit_rate": (
                _pct(exact_count, len(exact_rows)) if exact_rows else None
            ),
            "exact_identifier_labeled_count": len(exact_rows),
            "wrong_memory_rate": _pct(wrong_count, non_errored_count),
            "avg_context_chars": round(
                sum(context_chars_values) / non_errored_count, 2,
            ) if non_errored_count else 0.0,
            "avg_zoom_efficiency": round(
                sum(zoom_values) / non_errored_count, 6,
            ) if non_errored_count else 0.0,
            "latency_ms_median": _median(latency_values),
            "latency_ms_p95": _percentile(latency_values, 95),
            "latency_budget_pass_rate": _pct(
                latency_budget_pass, len(latency_budget_rows),
            ) if latency_budget_rows else None,
            "latency_budget_labeled_count": len(latency_budget_rows),
        }
    return aggregates


def evaluate(
    cases_path: str,
    runs_path: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    latency_budget_ms: float | None = DEFAULT_LATENCY_BUDGET_MS,
) -> dict[str, Any]:
    """High-level entry point: load JSONL inputs and produce a report.

    The returned dict is a JSON-serializable report with three top-
    level keys: ``summary`` (variant aggregates + totals), ``rows``
    (per-row scores), and ``config`` (effective inputs). It does NOT
    include raw packets or raw queries; only case_id/variant/counts.
    """

    if top_k <= 0:
        raise EvaluationError("top_k must be positive")

    raw_cases = load_jsonl(cases_path)
    raw_runs = load_jsonl(runs_path)

    cases: list[dict[str, Any]] = []
    case_ids_seen: set[str] = set()
    for index, raw in enumerate(raw_cases, start=1):
        case = _validate_case_dict(raw, line=index)
        case_id = case["case_id"]
        if case_id in case_ids_seen:
            raise EvaluationError(
                f"duplicate case_id {case_id!r}", line=index,
            )
        case_ids_seen.add(case_id)
        cases.append(case)

    case_by_id: dict[str, dict[str, Any]] = {c["case_id"]: c for c in cases}
    runs: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_runs, start=1):
        run = _validate_run_dict(raw, line=index)
        runs.append(run)

    scored_rows: list[dict[str, Any]] = []
    runs_without_case: list[str] = []
    for run in runs:
        case = case_by_id.get(str(run.get("case_id", "")))
        if case is None:
            runs_without_case.append(str(run.get("case_id", "")))
            continue
        scored_rows.append(
            score_case_run(
                case,
                run,
                top_k=top_k,
                latency_budget_ms=latency_budget_ms,
            )
        )

    aggregates = aggregate_scores(scored_rows)
    summary: dict[str, Any] = {
        "variants": aggregates,
        "totals": {
            "case_count": len(cases),
            "run_count": len(runs),
            "scored_count": len(scored_rows),
            "runs_without_case_count": len(runs_without_case),
        },
    }
    if runs_without_case:
        # Cap the surfaced missing-case ids so a single misnamed run
        # file cannot blow up the report size.
        summary["totals"]["runs_without_case_sample"] = sorted(
            {cid for cid in runs_without_case if cid}
        )[:25]
    return {
        "config": {
            "cases_path": cases_path,
            "runs_path": runs_path,
            "top_k": int(top_k),
            "latency_budget_ms": (
                float(latency_budget_ms) if latency_budget_ms is not None else None
            ),
        },
        "summary": summary,
        "rows": scored_rows,
    }


# ---------------------------------------------------------------------------
# CLI surface helpers
# ---------------------------------------------------------------------------


def _format_human_summary(report: Mapping[str, Any]) -> str:
    """Return a small human-readable summary, used when --json is off.

    The summary intentionally avoids dumping full row metrics; an
    operator can pipe the JSON to a tool for that. This keeps the
    human output free of long numeric tables and is consistent with
    the privacy rule "no raw query text, prefer case_id and counts".
    """

    if not isinstance(report, Mapping):
        return ""
    summary = report.get("summary", {}) if isinstance(report.get("summary"), Mapping) else {}
    totals = summary.get("totals", {}) if isinstance(summary.get("totals"), Mapping) else {}
    variants = summary.get("variants", {}) if isinstance(summary.get("variants"), Mapping) else {}
    config = report.get("config", {}) if isinstance(report.get("config"), Mapping) else {}
    top_k = config.get("top_k", DEFAULT_TOP_K)
    total_errored = sum(
        int(v.get("errored_count", 0) or 0) for v in variants.values() if isinstance(v, Mapping)
    )
    lines: list[str] = [
        "Phase 6A offline eval report",
        f"top_k: {top_k}",
        f"cases: {totals.get('case_count', 0)}",
        f"runs: {totals.get('run_count', 0)}",
        f"scored: {totals.get('scored_count', 0)}",
        f"errored runs: {total_errored}",
        f"runs without case: {totals.get('runs_without_case_count', 0)}",
    ]
    if variants:
        lines.append("variants:")
        for variant_name, agg in sorted(variants.items()):
            if not isinstance(agg, Mapping):
                continue
            lines.append(
                "  - "
                f"{variant_name}: "
                f"hit_at_k={agg.get('hit_at_k_rate', 0.0)}%, "
                f"wrong_memory={agg.get('wrong_memory_rate', 0.0)}%, "
                f"avg_zoom={agg.get('avg_zoom_efficiency', 0.0)}, "
                f"avg_chars={agg.get('avg_context_chars', 0)}"
            )
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_TOP_K",
    "DEFAULT_LATENCY_BUDGET_MS",
    "EvaluationError",
    "NormalizedCandidate",
    "load_jsonl",
    "parse_jsonl_text",
    "normalize_packet",
    "score_case_run",
    "aggregate_scores",
    "evaluate",
    "_format_human_summary",
]
