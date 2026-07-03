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

---

# Phase 6B — read-only variant capture for offline eval

Phase 6B adds a **CLI/operator-initiated capture** command that runs
read-only retrieval across seven comparison variants and writes JSONL
run rows directly consumable by the Phase 6A evaluator. This closes
the loop: Phase 6A scores already-captured packets; Phase 6B produces
those packets from live local Qdrant.

## What is in scope

- `qdrant_memory/eval_capture.py` — a capture core that accepts an
  initialized provider and validated Phase 6A cases, then runs
  read-only retrieval for each `(case, variant)` pair and emits JSONL
  run rows.
- `hermes qdrant eval-capture --cases CASES.jsonl --runs-out
  RUNS.jsonl [--variants all|comma-list] [--top-k N] [--mode
  hybrid|evidence] [--max-depth ...] [--max-children ...]
  [--max-source-chars ...] [--candidate-seed-top-k ...]
  [--max-graph-results ...] [--include-fact-history]
  [--include-metadata] [--json]` — a CLI subcommand that constructs
  the provider (unlike offline `eval`), runs the capture core, and
  writes the runs file.
- `build_tool_call` raises `CliUsageError` for `eval-capture` so
  provider dispatch fails closed (capture is CLI-only, not a Hermes
  tool).
- `tests/test_eval_capture.py` — focused unit tests.
- Additional CLI mapping tests in `tests/test_cli.py`.

## Variants

| variant           | what it runs                                                        |
|-------------------|---------------------------------------------------------------------|
| `dense-only`      | `MemoryRetriever.search(..., sparse suppressed, update_access=False)` |
| `dense+sparse`    | `MemoryRetriever.search(..., sparse allowed, update_access=False)`    |
| `graph`           | `GraphMemoryRetriever.search(...)` read-only with graph expansion     |
| `raptor-only`     | `RaptorSearcher.search(...)` → summaries + cited_leaves only          |
| `hybrid`          | `HybridRouter` dense seed (sparse suppressed by design) + graph + RAPTOR |
| `hybrid-no-graph` | `HybridRouter` with `graph_retriever=None`                            |
| `hybrid-no-raptor`| `HybridRouter` with `raptor_searcher=None`                           |

All variants use `update_access=False` on every retrieval call. The
`dense-only` variant suppresses sparse scroll
(`allow_sparse_scroll=False`); `dense+sparse` allows it
(`allow_sparse_scroll=True`) so the literal sparse baseline can run.
The sparse lane uses Qdrant `scroll`, which is a read operation.

> **Variant wording note:** `hybrid` does **not** mean "dense+sparse +
> graph + RAPTOR". The read-only `HybridRouter` always passes
> `allow_sparse_scroll=False` to the dense seed search (the sparse
> scroll lane is intentionally suppressed in the router to keep the
> read-only invariant tight). If you want a literal dense+sparse
> baseline, run the `dense+sparse` variant — that path is the one
> that actually invokes the Qdrant sparse `scroll`. The `hybrid`
> variant captures whatever the live router returns, which is dense
> seed (sparse suppressed) + graph + RAPTOR.

## Privacy and safety rules

- Run rows identify a capture by `case_id`, `variant`, `packet`,
  `latency_ms`, and a sanitized `capture` metadata dict only.
- **Raw query text is NEVER serialized** into run rows. The cases file
  already contains query text by operator design; the runs file must
  not duplicate it.
- Captured errors are sanitized to `<redacted>` plus a stable error
  kind (exception class name, with `timeout`/`connection` collapsed),
  never raw exception strings.
- The capture command is CLI/operator-initiated only. It is not
  auto-recall, not cron, and not a Hermes tool.
- It MAY read live local Qdrant/embeddings when the operator
  explicitly runs the command.
- It MUST NOT mutate Qdrant: no `upsert`, `delete`, `update_payload`,
  access metadata updates, or write-side tools.

## Creating local private eval cases

Create your eval cases JSONL **outside the repo** so private queries
and labels are never committed. Recommended path:

```
~/.hermes/qdrant_memory/eval/phase6b/cases.jsonl
```

Each line is one case (same schema as Phase 6A):

```json
{"case_id":"smoke-001","query":"your private query here","expected_file_paths":["docs/RAPTOR.md"],"expected_terms":["Phase 5"]}
```

> **Never commit private cases or runs.** Add the directory to your
> global gitignore or keep it entirely outside the repo working tree.
> The `.gitignore` in this repo does NOT cover `~/.hermes/` paths.

## Running eval-capture

```bash
# Capture all variants for all cases:
hermes qdrant eval-capture \
  --cases ~/.hermes/qdrant_memory/eval/phase6b/cases.jsonl \
  --runs-out ~/.hermes/qdrant_memory/eval/phase6b/runs.jsonl \
  --json

# Capture a subset of variants:
hermes qdrant eval-capture \
  --cases ~/.hermes/qdrant_memory/eval/phase6b/cases.jsonl \
  --runs-out ~/.hermes/qdrant_memory/eval/phase6b/runs.jsonl \
  --variants dense-only,hybrid,hybrid-no-graph
```

## Running eval to compare

After capture, run the Phase 6A evaluator over the same cases and the
captured runs:

```bash
hermes qdrant eval \
  --cases ~/.hermes/qdrant_memory/eval/phase6b/cases.jsonl \
  --runs ~/.hermes/qdrant_memory/eval/phase6b/runs.jsonl \
  --json
```

The runs file produced by `eval-capture` is directly compatible with
the Phase 6A `eval` command — same `case_id`/`variant`/`packet`/
`latency_ms` schema.

## Files (Phase 6B additions)

- `qdrant_memory/eval_capture.py` — the capture core.
- `qdrant_memory/cli_core.py` — `_execute_eval_capture_command` wired
  into `_execute_local_command`. `build_tool_call` raises
  `CliUsageError` for `eval-capture` to block provider dispatch.
- `cli.py` — `hermes qdrant eval-capture` subcommand registration.
- `tests/test_eval_capture.py` — focused unit tests for the capture
  core and CLI integration.
- `docs/RAPTOR_EVAL.md` — Phase 6B section (this section).

---

# Phase 6F — shadow gate / explicit thresholds for auto-recall eligibility

Phase 6F adds a **local, offline shadow gate** that evaluates explicit
thresholds over a Phase 6A evaluator JSON report before auto-recall
can be considered. It is the decision surface between "we have eval
data" and "the eval data is good enough to promote."

**This phase does NOT enable auto-recall.** Auto-recall runtime remains
legacy/provider prefetch. The gate is advisory: it produces a boolean
`auto_recall_eligible` that an operator reads. No runtime path is
changed by this phase.

## What is in scope

- `qdrant_memory/eval_gate.py` — a stdlib-only module that:
  - reads a JSON report produced by `hermes qdrant eval --json`;
  - builds explicit `GateThresholds` from a preset, a JSON file, and/or
    CLI flags;
  - evaluates named checks (case count, scored count, errored count,
    candidate hit/source floors, baseline lift, exact-id drop,
    wrong-memory absolute cap, wrong-memory regression, latency p95,
    latency-budget pass rate);
  - returns a compact `qdrant_eval_gate.v1` JSON result.
- `hermes qdrant eval-gate --report REPORT.json [knobs...] [--json]` —
  a CLI subcommand wired through `_execute_local_command`. It runs
  without ever instantiating the provider.
- `build_tool_call` raises `CliUsageError` for `eval-gate` so provider
  dispatch fails closed (gate is CLI/local-only, not a Hermes tool).
- `tests/test_eval_gate.py` — focused unit tests.
- Additional CLI mapping tests in `tests/test_cli.py`.
- `docs/RAPTOR_EVAL.md` — Phase 6F section (this section).

## What is deferred (Phase 6G or later)

- Enabling auto-recall at runtime based on gate output.
- Automatic threshold tuning / sweep loops.
- Cron / background gate runs.
- Statistical significance testing beyond deterministic aggregates.
- Gate-triggered Qdrant configuration changes.

## Workflow: eval-capture → eval → eval-gate

```
hermes qdrant eval-capture --cases ... --runs-out ...    # Phase 6B: live read-only capture
hermes qdrant eval --cases ... --runs ... --json         # Phase 6A: offline scoring → report
hermes qdrant eval-gate --report REPORT.json [--json]     # Phase 6F: threshold gate
```

The gate reads the Phase 6A report and evaluates whether a candidate
variant (`hybrid` by default) meets explicit thresholds before it can
be considered for auto-recall promotion.

## CLI

```
hermes qdrant eval-gate --report REPORT.json \
  [--candidate-variant hybrid] \
  [--baseline-variant dense-only] [--baseline-variant dense+sparse] \
  [--thresholds-file thresholds.json] \
  [--min-case-count N] [--min-scored-count N] [--max-errored-count N] \
  [--min-candidate-hit-at-k F] [--min-candidate-source-hit-at-k F] \
  [--min-hit-at-k-lift F] [--min-source-hit-at-k-lift F] \
  [--max-exact-id-drop F] \
  [--max-wrong-memory-rate F] [--max-wrong-memory-regression F] \
  [--max-latency-p95-ms F] [--min-latency-budget-pass-rate F] \
  [--json]
```

- `--report` is required (a JSON report from `hermes qdrant eval --json`).
- `--candidate-variant` defaults to `hybrid`.
- `--baseline-variant` is repeatable; defaults to `dense-only` and
  `dense+sparse` when omitted.
- `--thresholds-file` loads a JSON file of threshold overrides. The
  `auto-recall-default` preset applies if omitted. CLI flags override
  file values, which override the preset.
- `--json` emits the full JSON gate result on stdout. Without `--json`,
  the CLI prints a small human summary.

### Exit codes

| code | meaning |
|------|---------|
| `0`  | gate status = pass (all checks passed) |
| `1`  | gate status = fail (one or more checks failed) |
| `2`  | usage / input error (missing file, invalid JSON, missing variant) |

Tests call `evaluate_gate()` directly to avoid subprocess exit handling.

## Gate output JSON schema

```json
{
  "schema": "qdrant_eval_gate.v1",
  "status": "pass|fail",
  "auto_recall_eligible": false,
  "candidate_variant": "hybrid",
  "candidate_present": true,
  "baseline_variants": ["dense-only", "dense+sparse"],
  "baselines_found": ["dense-only", "dense+sparse"],
  "thresholds": { ... },
  "checks": [
    {"name": "...", "status": "pass|fail", "actual": ..., "threshold": ..., "details": "..."}
  ],
  "candidate_metrics": { ... },
  "baseline_metrics": { "dense-only": { ... }, ... },
  "summary": {"total_checks": 12, "passed": 11, "failed": 1, "failed_checks": ["wrong_memory_rate"]}
}
```

The gate output never includes raw query text, packets, per-row
payloads, `matched_expected`, or `wrong_reasons`. It only reads
per-variant aggregate metrics from the report's `summary.variants`.

## Default preset: `auto-recall-default`

The `GateThresholds.auto_recall_default()` preset is conservative:

| threshold | value | meaning |
|-----------|-------|---------|
| `min_case_count` | `10` | minimum candidate `case_count` |
| `min_scored_count` | `10` | minimum candidate `scored_count` |
| `max_errored_count` | `0` | zero errored rows tolerated |
| `min_candidate_hit_at_k` | `80.0` | absolute floor on candidate `hit_at_k_rate` |
| `min_candidate_source_hit_at_k` | `80.0` | absolute floor on candidate `source_hit_at_k_rate` |
| `min_hit_at_k_lift` | `0.0` | candidate must not regress vs best baseline hit |
| `min_source_hit_at_k_lift` | `0.0` | candidate must not regress vs best baseline source hit |
| `max_exact_id_drop` | `5.0` | max drop in `exact_identifier_hit_rate` vs best baseline |
| `max_wrong_memory_rate` | `3.0` | **strict**: absolute wrong-memory cap |
| `max_wrong_memory_regression` | `1.0` | max wrong-memory increase vs best baseline |
| `max_latency_p95_ms` | `500.0` | latency ceiling |
| `min_latency_budget_pass_rate` | `95.0` | latency budget compliance floor |

### Phase 6E does NOT pass the default gate

Current Phase 6E `hybrid` metrics: `hit_at_k_rate = 88.0`,
`source_hit_at_k_rate = 92.8571`, `wrong_memory_rate = 4.0`,
`latency_ms_p95 ≈ 265.7`.

The gate produces **11/12 checks pass, 1 fail** (`wrong_memory_rate`:
4.0 > 3.0 cap). `auto_recall_eligible = false`. This is a feature,
not a bug: the honest gate says "not yet." Until `wrong_memory_rate`
drops to `≤ 3.0` (either by tightening poison detection or expanding
the eval case set), the hybrid variant is not eligible for auto-recall
promotion under the default preset.

## Checks

The gate evaluates these named checks in order:

1. `candidate_present` — fail closed if the candidate variant is
   missing from the report entirely.
2. `case_count` — candidate `case_count >= min_case_count`.
3. `scored_count` — candidate `scored_count >= min_scored_count`.
4. `errored_count` — candidate `errored_count <= max_errored_count`.
5. `candidate_hit_at_k` — candidate `hit_at_k_rate >= min_candidate_hit_at_k`.
6. `candidate_source_hit_at_k` — candidate `source_hit_at_k_rate >= min_candidate_source_hit_at_k`.
7. `hit_at_k_lift` — `(candidate - best_baseline) hit_at_k_rate >= min_hit_at_k_lift`.
8. `source_hit_at_k_lift` — `(candidate - best_baseline) source_hit_at_k_rate >= min_source_hit_at_k_lift`.
9. `exact_id_drop` — `max(0, best_baseline - candidate) exact_identifier_hit_rate <= max_exact_id_drop`.
10. `wrong_memory_rate` — candidate `wrong_memory_rate <= max_wrong_memory_rate`.
11. `wrong_memory_regression` — `max(0, candidate - best_baseline_min) wrong_memory_rate <= max_wrong_memory_regression`.
12. `latency_p95` — candidate `latency_ms_p95 <= max_latency_p95_ms`.
13. `latency_budget_pass_rate` — candidate `latency_budget_pass_rate >= min_latency_budget_pass_rate`.

Every check fails closed when a required metric is `null` or absent
(e.g. `source_hit_at_k_rate` is `null` when no case carried source
labels). Missing baseline variants cause lift/drop checks to fail
closed rather than silently pass.

## Privacy and safety rules

- The gate never echoes raw query text, packets, or per-row payloads.
  It only reads per-variant aggregate metrics.
- The gate never calls Qdrant, never imports `qdrant_client`, and
  never instantiates the provider.
- The gate never mutates memory.
- The CLI is local-only. There is no HTTP, no remote URL, no
  auto-upload.
- `auto_recall_eligible` is advisory. This phase does NOT change
  runtime auto-recall behavior.

## Files (Phase 6F additions)

- `qdrant_memory/eval_gate.py` — the gate core (stdlib-only).
- `qdrant_memory/cli_core.py` — `_execute_eval_gate_command` wired
  into `_execute_local_command`. `build_tool_call` raises
  `CliUsageError` for `eval-gate` to block provider dispatch.
- `cli.py` — `hermes qdrant eval-gate` subcommand registration.
- `tests/test_eval_gate.py` — focused unit tests for the gate core
  and CLI integration.
- `docs/RAPTOR_EVAL.md` — Phase 6F section (this section).

---

# Phase 6H — controlled runtime shadow mode for hybrid auto-recall

Phase 6H adds a **real but controlled runtime shadow mode** that runs
the hybrid/RAPTOR retrieve path alongside the legacy dense auto-recall
path, captures aggregate-only telemetry, and **never alters prompt
context**.

**This phase does NOT enable hybrid auto-recall by default.** The
legacy dense prefetch/queue_prefetch path remains the sole source of
prompt-injected recall. Shadow mode is disabled by default and only
runs when an operator explicitly enables it.

## What is in scope

- `qdrant_memory/shadow_runtime.py` — a stdlib-only
  `ShadowRecorder` that persists append-only JSONL events and a
  compact state JSON.
- Config flags (safe defaults):
  - `auto_recall_shadow_enabled` (bool, default `False`)
  - `auto_recall_shadow_max_per_session` (int, default `20`)
  - `auto_recall_shadow_artifact_dir` (str, default `""` →
    `$HERMES_HOME/qdrant_memory/auto_recall_shadow`)
  - `auto_recall_shadow_mode` (str, default `"hybrid"`)
- `QdrantMemoryProvider.prefetch()` augmented to schedule a background
  shadow retrieve when shadow is enabled. `queue_prefetch()` is legacy
  cache priming only and is observed only when consumed by `prefetch`.
- `qdrant_memory_status` augmented with safe aggregate shadow fields.
- `tests/test_shadow_runtime.py` — focused unit tests.
- `docs/RAPTOR_EVAL.md` — Phase 6H section (this section).

## What is NOT changed

- Legacy auto-recall prompt injection is unchanged. The formatted
  result returned by `prefetch()` / `queue_prefetch()` is exactly the
  legacy dense `MemoryRetriever.search` + `format_for_prompt` output.
- No eval thresholds or `eval_gate.py` changes.
- No auto-promotion, no cron, no config mutation at runtime.
- No new model-visible tool for shadow events (operator visibility is
  through `qdrant_memory_status` only).

## How it works

When `auto_recall_shadow_enabled` is `True`:

1. `prefetch(query)` computes the legacy dense result and returns it
   to the caller unchanged. It then submits a background task
   (`_run_shadow_retrieve`) to the existing `ThreadPoolExecutor` that:
   - Builds the `HybridRouter` via `_ensure_hybrid_router(collection_name)`.
   - Calls `router.retrieve(query, top_k=auto_recall_top_k, mode="hybrid")`.
   - Extracts aggregate counts from the result via
     `_safe_hybrid_counts` (counts only — never accesses text, point
     IDs, paths, or warnings text).
   - Records a single sanitized event to the JSONL artifact.

2. `queue_prefetch(query)` runs the legacy queued search in the
   background as before. It is purely a cache-priming step and does
   NOT call `_run_shadow_retrieve`. Shadow telemetry is emitted only
   when the caller later invokes `prefetch()`.

3. The shadow path never calls any Qdrant mutation methods. The
   `HybridRouter.retrieve` always passes `update_access=False`.

## Privacy contract

The shadow event schema is **aggregate-only**. The following fields are
persisted per event:

| field                         | type   | source |
|-------------------------------|--------|--------|
| `schema`                      | str    | constant `"qdrant_shadow_event.v1"` |
| `timestamp`                   | str    | UTC ISO-8601 |
| `trigger`                     | str    | `"prefetch"` |
| `session_hash`                | str    | sha256[:16] of session_id |
| `query_length`                | int    | len(query) |
| `query_digest`                | str    | sha256[:16] of query |
| `latency_ms`                  | float  | hybrid retrieve wall-clock |
| `legacy_chars`                | int    | len of legacy formatted result |
| `legacy_empty`                | bool   | whether legacy result was empty |
| `hybrid_summaries_count`      | int    | len(result.summaries) |
| `hybrid_cited_leaves_count`   | int    | len(result.cited_leaves) |
| `hybrid_exact_hits_count`     | int    | len(result.exact_hits) |
| `hybrid_graph_relations_count`| int    | len(result.graph_relations) |
| `hybrid_warning_count`        | int    | len(result.warnings) |
| `hybrid_context_used_chars`   | int    | result.debug["context_used_chars"] |
| `status`                      | str    | `"ok"` or `"error"` |
| `error_code`                  | str    | sanitized code or `""` |

**Field allowlists.** `status`, `trigger`, and `error_code` are
sanitized through **per-field allowlists** before reaching JSONL.
Callers that supply any other value — including safe-alphabet
strings such as `abcdef0123456789_xyz` that would have been silently
accepted by an older generic `[a-z0-9_-]` token filter — never see
their raw input persisted verbatim:

| field         | allowed values                                    | collapse for unknown / non-string |
|---------------|---------------------------------------------------|------------------------------------|
| `status`      | `"ok"`, `"error"`, `"skipped"`                     | `"error"` |
| `trigger`     | `"prefetch"` (Phase 6H only emits from `prefetch`) | `"invalid"` |
| `error_code`  | `""`, `"router_unavailable"`, `"exception"`        | `"exception"` |

The empty string `""` for `error_code` is the caller's explicit
"no error code" sentinel and is preserved as-is so OK-path
statistics remain meaningful. `None` and non-string values always
collapse to the generic fallback.

**Never persisted**: raw query, raw packet, result text, point IDs,
source_uri, file_path, headings, warnings text, exception strings,
matched tokens, or payload excerpts.

## Prefetch shadow emission invariant

The fix2 invariant — `prefetch` is the single shadow emission point;
`queue_prefetch` is cache priming only and writes zero shadow events
itself — is preserved. Furthermore, `prefetch` honors a **truthiness**
check on the queued cache value:

- If `_prefetch_cache[sid]` is **non-empty** (the normal cache-hit
  case), `prefetch` returns the cached formatted string without
  re-running `MemoryRetriever.search + format_for_prompt`. The shadow
  event is still emitted exactly once from this path.
- If `_prefetch_cache[sid]` is **empty** or **missing** (the normal
  cache-miss case, including a `queue_prefetch` that legitimately
  produced no hits), `prefetch` runs the legacy dense search and
  `format_for_prompt` itself and returns the freshly formatted
  result. The shadow event is still emitted exactly once from this
  path.

This matches the pre-fix2 legacy prompt-context contract: the empty
string is treated identically to a cache miss, never as a cache hit.

## Per-session cap

`auto_recall_shadow_max_per_session` bounds the number of shadow
events per session (identified by sha256[:16] of the session id). Once
the cap is reached, subsequent `record_event` calls return `False`
without writing. The cap is persisted in the compact state JSON so a
fresh process respects counts from a prior process.

## Operator visibility

`qdrant_memory_status` now includes:

```json
{
  "shadow_enabled": false,
  "shadow_max_per_session": 20,
  "shadow_recorded_count": 0,
  "shadow_session_count": 0,
  "shadow_last_event": null
}
```

When events have been recorded, `shadow_last_event` is an object with
the same aggregate fields as the event schema (counts, latency,
status, timestamp) — **never** raw query or payload data.

## Enabling shadow mode

Shadow mode is opt-in. There are two equivalent ways to enable it.

### A) `$HERMES_HOME/qdrant_memory.json` (flat keys)

`load_config` reads this file as a **flat mapping of `auto_recall_shadow_*`
keys at the top level** — it does NOT nest under `qdrant_memory`:

```json
{
  "auto_recall_shadow_enabled": true,
  "auto_recall_shadow_max_per_session": 20,
  "auto_recall_shadow_artifact_dir": "",
  "auto_recall_shadow_mode": "hybrid"
}
```

### B) `config.yaml` under the `qdrant_memory` section (nested)

When configured via the global Hermes config file, the keys live
nested under the `qdrant_memory` section because the YAML root maps
the plugin's namespace:

```yaml
qdrant_memory:
  auto_recall_shadow_enabled: true
  auto_recall_shadow_max_per_session: 20
  auto_recall_shadow_artifact_dir: ""
  auto_recall_shadow_mode: "hybrid"
```

### Environment variable

Both forms can be overridden via environment variables:

```bash
HERMES_QDRANT_MEMORY_AUTO_RECALL_SHADOW_ENABLED=1
```

## Files (Phase 6H additions)

- `qdrant_memory/shadow_runtime.py` — the `ShadowRecorder` and
  `_safe_hybrid_counts` helper (stdlib-only).
- `qdrant_memory/config.py` — four new config keys with safe defaults.
- `__init__.py` — `_run_shadow_retrieve` helper invoked from `prefetch()`
  only, `queue_prefetch()` is legacy cache priming, and shadow fields in
  `_tool_status`.
- `tests/test_shadow_runtime.py` — 48 focused unit tests (41 from Phase 6H + 7 regression tests for fix3 P2 #1 cache-miss semantics and P2 #2 per-field allowlist sanitization).
- `docs/RAPTOR_EVAL.md` — Phase 6H section (this section).

## Phase 6I: Hybrid Auto-Recall Prompt Injection (Opt-In)

### Overview

Phase 6I adds opt-in hybrid/RAPTOR auto-recall prompt injection. When
enabled, the prefetch auto-recall path routes through the `HybridRouter`
(dense + RAPTOR + graph lanes) instead of the legacy dense-only
retriever, producing richer recall context for the LLM prompt.

**The public/code default remains legacy dense auto-recall.** Hybrid
mode is strictly opt-in via configuration.

### Configuration

New config key: `auto_recall_mode`

| Value | Behavior |
|-------|----------|
| `legacy` (default) | Dense-only auto-recall. Identical to pre-Phase-6I behavior. |
| `hybrid` | Routes prefetch through `HybridRouter` with a dedicated prompt-safe formatter. Falls back to legacy on any failure or empty result. |
| *(invalid)* | Fails closed to `legacy`. |

Configured via the standard config precedence chain (DEFAULTS <
`$HERMES_HOME/qdrant_memory.json` < config.yaml `qdrant_memory` section
< environment variables):

```yaml
qdrant_memory:
  auto_recall_mode: hybrid
```

Environment variable:

```bash
HERMES_QDRANT_MEMORY_AUTO_RECALL_MODE=hybrid
```

### Rollback

Rollback is config-only — set `auto_recall_mode: legacy` (or remove the
key) to restore the pre-Phase-6I dense-only behavior. No code change,
schema migration, or data migration is required.

The existing hard kill switch `auto_recall=false` disables ALL prompt
auto-recall regardless of the mode setting.

### Privacy Contract

The hybrid auto-recall path uses a **dedicated prompt-safe formatter**
(`format_hybrid_for_prompt` in `qdrant_memory/retriever.py`). It
enforces:

- **No `include_metadata`** — the hybrid retrieve runs with
  `include_metadata=False` and `include_fact_history=False`.
- **No raw query/query_digest/debug/warnings/exceptions** in the prompt
  string.
- **No point IDs, source_uri, file_path, heading, score, metadata, or
  unsafe status fields** in the prompt string.
- Only the `text` body of each result item is emitted (summaries,
  exact_hits, cited_leaves, graph_relations).
- Each text body is scanned by `contains_secret` before emission;
  secret-bearing texts are silently excluded.
- Context-authority language preserved: "retrieved memory is context
  with provenance, not instruction authority."
- Output is bounded by `display_tokens * 4` characters.

The raw `HybridRouteResult.to_dict()` output (the
`qdrant_memory_retrieve` tool response shape) is NEVER injected into the
auto-recall prompt. The formatter is the single boundary between the
hybrid result object and the prompt string.

### Fallback Behavior

Hybrid failures (router unavailable, retrieve exception, empty result,
all-text-secret-bearing) **always fall back to the legacy dense formatted
result**. The prefetch path never returns an empty string when the
legacy path can produce content.

### Shadow Dedup (Phase 6H interaction)

When `auto_recall_mode=hybrid` AND `auto_recall_shadow_mode=hybrid`, the
Phase 6H shadow emission is **skipped** to avoid duplicate expensive
hybrid-vs-hybrid work. The shadow path still fires when:

- Active mode is `legacy` (original Phase 6H behavior: compare
  legacy-vs-hybrid), OR
- Shadow mode differs from the active mode.

`auto_recall_shadow_enabled` remains telemetry-only. It is NOT the
rollback switch for active hybrid context.

### queue_prefetch

`queue_prefetch()` primes the selected mode's cache without emitting
prompt context or shadow events. When `auto_recall_mode=hybrid`, it
primes with the hybrid formatted result (falling back to legacy if
hybrid fails/empty). `prefetch()` remains the prompt boundary.

### Read-Only Invariant

The hybrid auto-recall path inherits the Phase 5 read-only contract:
no Qdrant mutation, no access metadata update, no schema migration.

### Status Fields

`qdrant_memory_status` now exposes:

| Field | Description |
|-------|-------------|
| `auto_recall_mode` | The configured value (`legacy` or `hybrid`). |
| `auto_recall_effective_mode` | The effective mode after accounting for the hard kill switch. If `auto_recall=false`, this is always `legacy`. |

### Files (Phase 6I additions)

- `qdrant_memory/config.py` — new `auto_recall_mode` config key with
  `_as_auto_recall_mode` coercion (fail-closed to `legacy`).
- `qdrant_memory/retriever.py` — `format_hybrid_for_prompt` function
  and helpers (`_extract_safe_text`, `_scan_text_for_secret`).
- `__init__.py` — `_run_hybrid_prefetch` helper, updated `prefetch()`
  and `queue_prefetch()` for mode selection + fallback, shadow dedup,
  and status fields.
- `tests/test_phase6i_hybrid_auto_recall.py` — focused Phase 6I tests
  covering config, formatter, prefetch fallback, privacy, shadow dedup,
  status, and kill switch.
- `tests/test_shadow_runtime.py` — updated `_build_provider` stub to
  include the new `auto_recall_mode` config key.
- `docs/RAPTOR_EVAL.md` — Phase 6I section (this section).

### Tests

- Config: default/coercion/env/invalid-fail-closed.
- Formatter: output shape, privacy contract, char budget, exception
  handling, context-authority language.
- Privacy: comprehensive secret-shaped field exclusion (text,
  source_uri, file_path, point_id, debug, warnings, graph path).
- Prefetch: legacy mode unchanged, hybrid mode returns hybrid
  formatted, fallback to legacy on empty/exception/router-none.
- Queue prefetch: hybrid priming, legacy fallback, no shadow emission.
- Shadow dedup: hybrid+hybrid skips shadow, legacy+hybrid emits shadow.
- Status: `auto_recall_mode` and `auto_recall_effective_mode` exposed.
- Kill switch: `auto_recall=false` disables all prompt auto-recall.
