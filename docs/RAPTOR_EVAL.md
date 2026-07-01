# Phase 6A — offline RAPTOR/hybrid retrieval evaluation

Phase 6A adds a **local, deterministic evaluator** for the Phase 5
hybrid retrieve surface. It is intentionally narrow:

- It scores **already-captured** retrieval packets. It does NOT
  contact Qdrant. It does NOT initialize the provider. It does NOT
  re-run retrieval.
- It is the smallest useful slice of a future shadow-collection /
  auto-recall-tuning loop. The bigger pieces (live shadow
  collection, automatic recall path selection, threshold tuning that
  changes runtime behavior, Qdrant v2 / named sparse vector
  migration, cron jobs) are **deferred to Phase 6B or later** and
  are not implemented in this slice.
- All inputs and outputs are local JSONL artifacts. The evaluator is
  stdlib-only and never reaches the network, the filesystem outside
  the inputs the operator passes, or the Qdrant client.

## What is in scope

- `qdrant_memory.evaluation` — a stdlib-only module that:
  - loads JSONL eval cases and run rows;
  - normalizes Phase 5 grouped retrieve packets
    (`results.exact_hits` / `summaries` / `cited_leaves` /
    `graph_relations`) and legacy list/dict shapes into a stable
    internal candidate list;
  - scores per `(case, variant)` row at a configurable `top_k`
    (default `5`);
  - aggregates per variant (case count, errored count, hit rate,
    source-hit rate, exact-identifier-hit rate, wrong-memory rate,
    average context chars, average zoom-efficiency, latency
    median/p95, latency-budget pass rate).
- `hermes qdrant eval --cases CASES.jsonl --runs RUNS.jsonl
  [--top-k 5] [--latency-budget-ms 750] [--json]` — a local CLI
  subcommand wired through `_execute_local_command`. It runs
  without ever instantiating the provider.
- `docs/RAPTOR_EVAL.md` — this file.
- `tests/test_retrieve_evaluation.py` — focused unit tests for the
  evaluator.
- Additional CLI mapping and local-command tests in
  `tests/test_cli.py`.

## What is deferred (Phase 6B or later)

- Live shadow collection from real `qdrant_memory_retrieve` calls.
- Automatic recall path selection or auto-recall defaults.
- Threshold tuning that changes runtime recall behavior.
- Qdrant v2 migration, named sparse vectors, or collection schema
  changes.
- Provider-level latency instrumentation and live p95 budgets beyond
  pass-through `latency_ms` in captured run rows.
- Human/LLM qualitative `context_packet_usefulness` grading or
  downstream answer-quality scoring.
- Dashboards, cards, publication artifacts, or social-ready
  benchmark visuals.
- Cron / watcher jobs for eval collection.
- Statistical significance testing beyond simple deterministic
  aggregates.

## Inputs

Both inputs are local JSONL. Blank lines and lines whose first
non-whitespace character is `#` are ignored. The first JSON parse
error fails the whole eval with a user-facing line-numbered
message.

### Case row schema

Required:

| field      | type   | notes |
|------------|--------|-------|
| `case_id`  | string | stable unique id; reports surface this only |
| `query`    | string | operator-authored query used to capture the run packets; never echoed in reports |

Optional label fields (all `list[str]`):

| field                    | meaning |
|--------------------------|---------|
| `expected_point_ids`     | exact Qdrant point ids expected in top-k |
| `expected_source_uris`   | exact `source_uri` values expected in top-k |
| `expected_file_paths`    | exact `file_path` values expected in top-k |
| `expected_terms`         | case-insensitive substrings expected in any top-k candidate's text/heading/source/file |
| `forbidden_point_ids`    | exact point ids that must NOT appear anywhere in the emitted packet (poison) |
| `forbidden_source_uris`  | exact `source_uri` values that must NOT appear anywhere |
| `forbidden_file_paths`   | exact `file_path` values that must NOT appear anywhere |
| `forbidden_terms`        | substrings that must NOT appear anywhere in any candidate's text/heading/source/file |
| `tags`                   | free-form, never used for scoring |
| `notes`                  | free-form, never used for scoring |
| `domain`                 | free-form grouping label |

Validation rules (Phase 6A):

- Missing required string fields or non-string-list label fields
  fail with `EvaluationError` carrying the offending 1-indexed
  line number.
- Duplicate `case_id` in the cases file fails with
  `EvaluationError`.

### Run row schema

Required:

| field     | type              | notes |
|-----------|-------------------|-------|
| `case_id` | string            | must match a case row |
| `variant` | string            | e.g. `dense-only`, `dense+sparse`, `raptor-only`, `graph`, `hybrid`, `hybrid-no-graph`, `hybrid-no-raptor` |
| `packet`  | dict or list      | the captured retrieval packet from `qdrant_memory_retrieve` (or a legacy search-like list) |

Optional:

| field        | type   | notes |
|--------------|--------|-------|
| `latency_ms` | number | pass-through; aggregated as median/p95 |
| `error`      | string | if present/non-empty, row is counted as `errored`, all numeric metrics become `null`, and the **raw text is redacted from the report** (see [Error redaction](#error-redaction) below). The operator-facing report never echoes the raw captured string |
| `capture`    | dict   | small config labels (mode, top_k, max_depth, ...); not used for scoring in 6A |

The evaluator does not deep-clone or mutate `packet`. The
normalizer returns a list of `NormalizedCandidate` objects; the
caller's dict is left intact.

## Packet normalization

The evaluator recognizes three packet shapes:

1. **Phase 5 grouped retrieve** —
   `{"results": {"exact_hits": [...], "summaries": [...],
   "cited_leaves": [...], "graph_relations": [...]}}`. Each list is
   flattened into `NormalizedCandidate` rows in lane order:
   `exact_hits → summaries → cited_leaves → graph_relations`. Any
   additional list-valued key under `results` becomes the
   `legacy` lane.
2. **Generic dict-of-lists** — top-level keys whose values are
   lists. The evaluator first flattens the well-known lane keys
   (`exact_hits`, `summaries`, `cited_leaves`, `graph_relations`)
   into their named lanes, then projects any remaining list-valued
   keys into the `legacy` lane. This guarantees unknown lanes
   (e.g. `extra_lane`, plugin output, third-party capture
   extensions) are still scored so a forbidden term embedded there
   can still trip `wrong_memory`.
3. **Bare list** — every item is projected as a `legacy` lane
   candidate.

Field projection rules (applied in this order):

- `point_id` ← `point_id` or `id`
- `parent_point_id` ← `parent_point_id` or `parent_id`
- `source_uri` ← `source_uri` or `source`
- `file_path` ← `file_path` or `path`
- `heading` ← `heading`
- `text` ← first non-empty value among
  `text` / `snippet` / `excerpt` / `preview` / `summary_text` /
  `content`
- `score` ← first numeric value among
  `score` / `final_score` / `_rrf_score` / `qdrant_score` /
  `graph_score`

## Metrics

All metrics are computed per `(case_id, variant)` first, then
aggregated per variant.

### Per-row metrics

| field                     | meaning |
|---------------------------|---------|
| `errored`                 | True iff the run row carried a non-empty `error` string |
| `hit_at_k`                | True iff any expected point id / source URI / file path / expected term appears in the lane-aware top-k |
| `source_hit_at_k`         | True iff any expected source URI or file path appears in the lane-aware top-k; `null` if the case has no expected source URIs or file paths |
| `exact_identifier_hit`    | True iff any expected point id equals an emitted candidate's `point_id` in the lane-aware top-k; `null` if the case has no expected point ids. Parent refs are intentionally NOT considered here: the contract is "exact identifier appears as an emitted point id", not a parent-citation match |
| `wrong_memory`            | True iff any forbidden point id / source URI / file path / forbidden term appears **anywhere** in the normalized emitted packet (not just top-k). Forbidden point id detection is intentionally conservative and matches against both `point_id` and `parent_point_id`, so a poison id declared in a leaf's parent still fires |
| `useful_topk_count`       | distinct expected handle classes (point id, source, term) hit in the top-k |
| `context_chars`           | total emitted text length across all normalized candidates |
| `zoom_efficiency`         | see formula below |
| `latency_ms`              | pass-through (None when absent) |
| `latency_budget_met`      | True iff `latency_ms <= latency_budget_ms`; `null` when latency is absent or budget is disabled |
| `matched_expected`        | compact `{point_ids, sources, terms}` list of *expected* handles that hit. `sources` carries the expected source URI / file path strings from the case schema, NOT arbitrary emitted candidate fields: a candidate whose `file_path` matches but whose `source_uri` does not will surface only the matched `file_path`, never its non-matching `source_uri`. Both expected label kinds may appear in `sources` when both match on a single candidate (source URI first, then file path) |
| `wrong_reasons`           | compact dict of forbidden handles that fired (point ids, source URIs, file paths, terms) |
| `emitted_count` / `topk_count` | packet size and lane-aware top-k size for the row |

### Lane-aware top-k

Phase 5 retrieve returns `top_k` per bucket rather than one
globally-fused ranking. To avoid unfairly penalizing exact hits
that fall into a non-first bucket, hit metrics take the first
`top_k` items from each lane and union by `point_id` (first
occurrence wins). This keeps ordering stable across variants.

A non-empty `point_id` dedupes **globally across lanes** — a point
emitted in `summaries` and again in `cited_leaves` counts as one
entry in the lane-aware top-k. Candidates without a `point_id`
fall back to a `lane::rank` key so the anonymous / legacy stream
does not collapse two distinct entries into one.

### `matched_expected.sources` contract

The `sources` list under `matched_expected` is a list of *expected*
source labels from the case schema that matched an emitted
candidate — never arbitrary emitted candidate fields. Concretely:

- If a candidate's emitted `source_uri` equals one of the case's
  `expected_source_uris`, that expected URI is added.
- If a candidate's emitted `file_path` equals one of the case's
  `expected_file_paths`, that expected file path is added.
- If both match on a single candidate, both expected labels appear
  (source URI first, then file path).
- A candidate that matches an expected `file_path` but carries a
  non-matching `source_uri` will only surface the matched file path.
  Its emitted `source_uri` is **not** recorded.
- Two candidates that share the same expected `file_path` but emit
  distinct unrelated `source_uri` values dedupe to one entry: the
  expected file path. The non-matching emitted `source_uri` values
  are not recorded, and `useful_topk_count` is not inflated.

This keeps `useful_topk_count` and `zoom_efficiency` honest: the
source handle dimension is bounded by the case's expected labels,
not by the number of emitted candidates that happen to share a
file path.

### `zoom_efficiency` formula

```
zoom_efficiency = useful_topk_count / max(1, context_chars / 1000)
```

This rewards runs whose top-k contains a high fraction of
expected evidence relative to the character budget they emitted.
A run that uses little context for many useful hits scores
higher than a run that emits a large context for the same hits.
The denominator floor of `1` keeps the metric finite for empty or
tiny packets.

### Per-variant aggregates

| field                                | meaning |
|--------------------------------------|---------|
| `case_count`                         | total run rows seen for the variant |
| `scored_count`                       | non-errored rows |
| `errored_count`                      | rows that carried a non-empty `error` string |
| `hit_at_k_rate`                      | percent of non-errored rows with `hit_at_k == True` |
| `source_hit_at_k_rate`               | percent of **source-labeled** non-errored rows with `source_hit_at_k == True`. Rows whose case has no expected source URIs or file paths contribute `source_hit_at_k = None` and are excluded so a term-only case does not silently drag the rate down. The rate is `null` when no non-errored row carried source/file labels |
| `source_hit_labeled_count`           | rows that contributed to `source_hit_at_k_rate` |
| `exact_identifier_hit_rate`          | percent of labeled (non-null) rows with `exact_identifier_hit == True`; `null` when no rows had expected point ids |
| `exact_identifier_labeled_count`     | rows that contributed to the labeled exact-id rate |
| `wrong_memory_rate`                  | percent of non-errored rows with `wrong_memory == True` |
| `avg_context_chars`                  | mean of `context_chars` over non-errored rows |
| `avg_zoom_efficiency`                | mean of `zoom_efficiency` over non-errored rows |
| `latency_ms_median`                  | linear median over rows that carry a numeric `latency_ms`; `null` when absent |
| `latency_ms_p95`                     | linear-interpolation p95 (numpy-style) over latency-bearing rows; `null` when absent |
| `latency_budget_pass_rate`           | percent of latency-bearing rows that met the configured budget; `null` when no rows carried latency |
| `latency_budget_labeled_count`       | rows that contributed to the latency-budget pass rate |

Errored rows are always excluded from rate denominators so a flaky
capture harness cannot deflate a variant's hit rate. Label-aware
rates (`source_hit_at_k_rate`, `exact_identifier_hit_rate`) further
exclude rows whose case lacks the relevant label set so a label
gap does not silently deflate a variant's rate either.

## CLI

`hermes qdrant eval --cases CASES.jsonl --runs RUNS.jsonl [--top-k
N] [--latency-budget-ms N] [--json]`

- `--cases` / `--runs` are required.
- `--top-k` defaults to `5` (matches the Phase 5 retrieve default).
- `--latency-budget-ms` defaults to `750`; pass `0` or a negative
  value to disable the budget check.
- `--json` emits the full JSON report on stdout. Without `--json`,
  the CLI prints a small human summary that prefers
  `case_id`/variant/counts and never dumps raw packets or raw
  query text.

The command is wired through `_execute_local_command`. It never
imports `QdrantMemoryProvider` and never instantiates the
provider; a missing provider factory call is a regression in this
contract. Eval-side validation errors surface as a single-line
user-facing message with the offending 1-indexed line number, and
the CLI exits with code `2`. The `eval` subcommand is **blocked
from the provider-dispatch path** by `build_tool_call` raising
`CliUsageError`, so a code path that bypasses `_execute_local_command`
also fails closed.

## Privacy and safety rules

- The evaluator does NOT echo raw query text from captured packets
  in any report. Reports prefer `case_id`, variant, and counts.
- Forbidden labels (the strings the operator intentionally
  declared) may appear inside `wrong_reasons` because that is the
  diagnostic that explains the `wrong_memory` flag. The
  authoritative privacy concern is the *retrieval* query text, not
  the operator-authored case schema.
- Errored rows NEVER echo the raw captured `error` text — see
  [Error redaction](#error-redaction). The retrieval harness can
  embed raw queries, packet body snippets, or Qdrant endpoint
  details in error strings, and the default report must not leak
  any of that.
- The evaluator never calls Qdrant, never imports
  `qdrant_client`, and never instantiates the provider. This is
  enforced at the import level
  (`tests/test_retrieve_evaluation.py::test_evaluation_module_does_not_import_qdrant_client`).
- The evaluator never mutates memory. No `upsert`, no `delete_ids`,
  no `update_payload`, no `scroll_by_filter`.
- The CLI is local-only. There is no HTTP, no remote URL, no
  auto-upload.

## Error redaction

When a run row carries a non-empty `error` string, the default
Phase 6A report row is sanitized:

| field             | value on an errored row | meaning |
|-------------------|-------------------------|---------|
| `errored`         | `true`                  | row was an errored run |
| `error`           | `"<redacted>"`          | constant sentinel; the raw captured text is never emitted |
| `error_present`   | `true`                  | a non-empty `error` string was supplied on the run row |
| `error_redacted`  | `true`                  | explicit flag downstream tooling should check |
| `hit_at_k`        | `null`                  | numeric metrics stay null so they drop out of rate denominators |
| `source_hit_at_k` | `null`                  | same |
| `exact_identifier_hit` | `null`              | same |
| `wrong_memory`    | `null`                  | same |

A non-errored row does NOT carry `error`, `error_present`, or
`error_redacted` keys. The `evaluate()` JSON dump and the
`hermes qdrant eval --json` CLI report both honor this contract:
the raw captured string never reaches stdout or disk.

Operator workflow: when an errored row appears, re-open the
captured run JSONL on the machine that produced it (the evaluator
never had access to a human-readable view of the error). The
report only tells the operator that the row errored; it does not
surface what the error said.

## Worked example

`cases.jsonl`:

```json
{"case_id":"smoke-001","query":"where is the raptor retrieve contract documented?","expected_file_paths":["docs/RAPTOR.md"],"expected_terms":["Phase 5"],"forbidden_terms":["unrelated decoy"]}
{"case_id":"smoke-002","query":"how does the apply gate work?","expected_terms":["apply"]}
```

`runs.jsonl` (one row per case × variant):

```json
{"case_id":"smoke-001","variant":"hybrid","latency_ms":312,"packet":{"results":{"exact_hits":[{"point_id":"p1","text":"Phase 5 hybrid details","file_path":"docs/RAPTOR.md","score":0.91}],"summaries":[],"cited_leaves":[],"graph_relations":[]}}}
{"case_id":"smoke-001","variant":"raptor-only","latency_ms":410,"packet":{"results":{"summaries":[{"point_id":"s1","text":"raptor summary with unrelated decoy"}],"exact_hits":[],"cited_leaves":[],"graph_relations":[]}}}
{"case_id":"smoke-002","variant":"hybrid","latency_ms":260,"packet":{"results":{"exact_hits":[{"point_id":"p2","text":"apply gate contract body"}],"summaries":[],"cited_leaves":[],"graph_relations":[]}}}
```

```bash
hermes qdrant eval --cases cases.jsonl --runs runs.jsonl --json
```

Expected outcome (illustrative):

- `hybrid` variant: `hit_at_k_rate = 100.0`, `wrong_memory_rate = 0.0`.
- `raptor-only` variant: `hit_at_k_rate = 0.0`,
  `wrong_memory_rate = 100.0` (forbidden term fired).
- `latency_ms_p95` computed over both rows that carry latency.

## Deferred to Phase 6B

- Live shadow collection from real `qdrant_memory_retrieve` calls.
- Auto-recall defaulting or runtime threshold tuning.
- Qdrant v2 migration / named sparse vectors.
- Provider-level latency instrumentation beyond pass-through.
- Human/LLM qualitative usefulness grading or answer-quality
  scoring.
- Dashboards, cards, social-ready benchmark visuals.
- Cron / watcher jobs for eval collection.
- Statistical significance testing beyond deterministic
  aggregates.

## Files

- `qdrant_memory/evaluation.py` — the evaluator (stdlib-only).
- `qdrant_memory/cli_core.py` — `_execute_eval_command` is wired
  into `_execute_local_command`. `build_tool_call` raises
  `CliUsageError` for `eval` to block provider dispatch.
- `cli.py` — `hermes qdrant eval` subcommand registration.
- `tests/test_retrieve_evaluation.py` — focused unit tests.
- `tests/test_cli.py` — `eval` parser mapping, local-command
  tests, and the `build_tool_call` block.
- `docs/RAPTOR_EVAL.md` — this document.
