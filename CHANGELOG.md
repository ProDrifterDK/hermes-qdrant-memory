# Changelog

All notable changes to this project are documented here.

This project is currently public beta / experimental. The format follows Keep a Changelog style, but the repository is still pre-1.0 and the public API may change.

## [Unreleased]

### Added

- Session origin is safe to turn on (C'6). No route drops a recorded `session_origin` any more, in either mode: the learning store, learning approval and extraction approval carry an existing mapping forward when they bring no capture (mode `off`, or switched back to it); backup restore merges the live point's mapping into the snapshot payload (`session_origin_kept_ids` in the plan); a consolidation merge moves the duplicates' citations onto the canonical point (`session_origin_merge`), and guarded-auto maintenance leaves cited facts to a human. Merges the plugin performs never refuse on the 8-citation / 4 KiB limits: they keep the earliest capture, the oldest two quotes and the most recent citations (deterministic, order-independent). With `recorded`, the legacy payload `session_id` follows `captured_session_id`; improve captures and `cite` previews persist privately under `$HERMES_HOME`, so an apply or approve after a restart (or from a second CLI process) still cites the line, and the approve still re-verifies against `state.db`; a momentary `state.db` lock is retried once; `origin-check` adds `point_counts`. New CLI `hermes qdrant store TEXT --cite FIRST:LAST --session-id SID [--cite-quote]`. New `scripts/e2e_session_origin.py` exercises every route in `recorded` mode against disposable `tmp_c6_*` collections and a read-only `state.db`, and prints a JSON receipt. See README "Turning session origin on".
- Session origin on improve, a read-only origin check, and origin in trace (opt-in, same `session_origin_mode`). With `recorded`, `qdrant_memory_improve_apply` records producer `improve` with the session the preview captured and the exact source line, resolved across that session's chain; the approver appears only as `approved_session_id`. A `point_ids` candidate records no citation, also over a legacy conversation-turn point whose session is still in `state.db`. The improve insert re-reads its target under the collection lock and refuses any existing point, so it never drops a `session_origin`. New read-only `qdrant_memory_origin_check` tool and `hermes qdrant origin-check` command: per page of the points that carry `session_origin` in this provider's memory or learning collection, the counts and point ids per current citation status, with no write of any kind (scroll only, a write-refusing client, `state.db` read-only); not on the `recall` surface. `qdrant_memory_trace` discloses `session_origin` and each citation's current status. With `off`, outputs, payloads and the prompt block are unchanged and the tool list gains only the new entry. Fixed in every mode: an improve preview over `point_ids` called a method the Qdrant client does not have and silently returned no candidates; it now reads the points. See README "Session origin on improve, origin-check and trace".
- Session origin on extraction approval, learning approval and the learning store (opt-in, same `session_origin_mode`; with `off` schemas, payloads and outputs are unchanged). Hook candidates (`on_pre_compress`, `on_session_end`) now keep the session they were captured in and the exact text the extractor read, so `qdrant_memory_extraction_approve` and `qdrant_learning_approve` record `captured_session_id` as the hook's session instead of the approver's, also after a compression switch, and add `approved_session_id`/`approved_at` when approval happened elsewhere. The citation resolves by exact line match across the captured session's `parent_session_id` chain, copies sharing a `message_uid` count once, and a line found in more than one message is recorded as `unresolved:ambiguous_match` instead of picking one. `qdrant_learning_store` records a `tool_position` citation, and a learning re-store merges citations into the existing mapping instead of replacing it. See README "Session origin on extraction and learning approval".
- Session origin on manual stores (opt-in, `session_origin_mode: off` by default; with `off` the store's schema, payloads and outputs are unchanged). With `recorded`, `qdrant_memory_store` writes a validated `session-origin-v1` mapping (`session_origin`) on the fact's payload and accepts an optional `cite` (`session_id` defaulting to the current session, `first_msg_id`, `last_msg_id`, optional `quote`). The preview resolves and shows the citation; the approve re-verifies the previewed digests and refuses if the span changed, an unpreviewed or wrong span is refused, and an unresolvable `state.db` is recorded as `unresolved:<reason>` without blocking the store. A re-store of the same text from another session merges citations and keeps the first capture time, and writers that bring no capture carry an existing mapping forward. The mapping is capped at 8 citations, 2 quoted citations (each at most 1,000 characters, redacted with Hermes' egress rules and `redact_secret_spans`) and 4,096 bytes, and oversize is refused. `qdrant_memory_inspect` discloses the origin and each citation's current status. `forget` still deletes such points. See README "Session origin on manual stores".
- Session recall in automatic recall (opt-in, off by default): `session_prefetch_enabled` appends verbatim session windows that compaction removed from the active prompt after the legacy prefetch block (shared budget, fail-open with a timeout, read-only), and `session_expand_enabled` registers the read-only `qdrant_memory_session_expand` tool (messages around a message id; indexed windows when the session was pruned). `scripts/c5_gate_harness.py` measures the lane for gate C'5. See docs/session-recall-prefetch.md.
- Session origin citations, read-only foundation: `qdrant_memory/session_origin.py` adds `resolve_span()`, which pins a `state.db` span as a `session-origin-v1` citation (`span_sha256`, `uids_sha256`, `content_sha256`, `msg_count`, `basis`, UTC `verified_at`) or returns `unresolved:<reason>`. It also adds `verify_citation()`, which returns exactly one of `verified`, `verified_via_uid`, `drifted`, `partially_missing`, `expired` and `unavailable`, and `make_quote()`, which builds a redacted quote of at most 1,000 characters. Digests hash the raw stored bytes, so they do not depend on the host timezone or the redactor. Both functions open `state.db` read-only, report a locked or busy database as unavailable, never create sidecar files, and never raise. No producer, tool or write path uses them yet. The README section "Session origin citations" documents the statuses and the recommended `sessions.retention_days` and pinning practice. The plugin never changes host configuration.
- Session recall: `qdrant_memory_session_recall` and `hermes qdrant session-recall index|search|status`. The lane indexes `state.db` conversation history verbatim into its own collection (`hermes_session_windows`) as overlapping fixed-size windows and answers with a dense top-K search plus an optional cross-encoder rerank. `scope=current` covers the caller's session and its compressed ancestors and catches them up inline. The incremental per-session cursor reproduces a full rebuild exactly. A debounced background catch-up runs behind a shared single-writer lock, and a `session_recall_lag` doctor warning flags a stalled index. Secret spans are redacted before embedding and storage.
- Local/sovereign profile lane (opt-in; defaults unchanged): `tool_surface: recall` exposes only the six recall tools (context, search, expand, session recall, learning search, status), names only those in the system prompt block, and refuses every other plugin tool with an error naming the surface. `read_sources` adds read-only collections (`{collection, profile_id, kind}`) that automatic recall (legacy and hybrid dense lane), `qdrant_memory_context`, `qdrant_memory_search`, `qdrant_learning_search` and `qdrant_memory_expand` read. Recall scores the own and read-source hits as one pool (one min-max normalization, then the usual importance/recency/provenance scoring and `min_final_score`), so scores from different collections are comparable. Tool output labels each hit with `source_collection`. See README "Local or sovereign profile".
- Configurable embedding input chunking in `EmbeddingClient` via `qdrant_memory.embedding_max_input_chars` and `qdrant_memory.embedding_max_chunks`, covering conversation writes, learning writes, indexing, search queries, and CLI/script clients through one centralized guard.

### Fixed

- Automatic recall no longer searches with host-generated incoming messages. A gateway notification (background-process completion, watch match or heartbeat, delegation report, with the `[INTERNAL NOTIFICATION …]` footer), a cron run prompt, a `/goal` continuation or an auto-resume note reached `prefetch` as the query and injected whatever shared its words; `queue_prefetch` could also cache such a block for the next real turn. `host_markers.recall_query` now decides before any search: a wholly host-generated message recalls nothing (legacy, hybrid, the lane, no embedding, nothing cached), and a real message behind a host wrapper (sender tag, Discord routing note, attachment note, steer frame) recalls on the user's words only. Always on, like the host-scaffolding filter. See docs/session-recall-prefetch.md "Host-generated incoming messages".
- Recall never returns host scaffolding, and automatic recall can be set to inject nothing rather than noise. Hermes stores its own text as conversation rows (context-compaction handoffs, gateway notifications and their `[INTERNAL NOTIFICATION …]` footer, background-process and delegation reports, steer frames, `<memory-context>` fences, skill/cron prompts, model and surface switch notices); `qdrant_memory/host_markers.py` recognizes them from the host's own constants. Session-window search (`qdrant_memory_session_recall` and the prefetch lane) cuts those messages out of every window at query time, attributing each window segment to its `state.db` row so a window that starts inside a summary is caught too (fail-closed when the session was pruned); the index is not rewritten. Conversation-turn points of the legacy collection whose user or assistant side is scaffolding are no longer returned, and `qdrant_memory_session_expand` skips such rows as neighbours and withholds their text. A row is judged whole, so a mid-turn steer row or an attachment note is left out with the user words it carries. This filter always applies. New optional relevance floors, all off by default so existing installs recall as before: `auto_recall_min_final_score` and `auto_recall_min_raw_score` for the legacy block, `session_prefetch_min_rerank_score` and `session_prefetch_min_dense_score` for the lane (judged on the scale the lane was ordered by), and `auto_recall_min_word_chars`, which skips automatic recall (legacy, hybrid and lane) for messages with fewer letters/digits (confirmations and pings). `off`, `""` or `null` disables each of them (`0` too on the legacy floors and the word count). Below a floor a block is empty, never a bare header. Recommended for bge-m3 with bge-reranker-v2-m3, to be re-derived for other models: final score 0.05 (0.1 was the measured pick, but recency decay then keeps an importance-5 memory older than about 67 days out of automatic recall), 12 word characters, rerank -1.5, dense 0.55. Measured with `scripts/c5_gate_harness.py` (now with a noise set, a floor grid and host-scaffolding counts): on one install, with final 0.1, injection on 41 real short confirmations fell from 41 turns (about 104,000 characters with the lane) to 9 turns (about 9,600), and injected scaffolding in the live arm (legacy plus reranked lane) fell from 26 entries (10 legacy memories, 16 lane excerpts) over the 92 measured messages to 0, with the gate sets within one hit of the previous baseline per arm. See docs/session-recall-prefetch.md "Host scaffolding" and "Relevance floor".
- Session lane in automatic recall: a slow or sleeping reranker now degrades to dense order instead of dropping the lane. The rerank call could use the whole `session_prefetch_timeout_s`, so the dense fallback came back after the turn had stopped waiting, and the still-running worker made the next turn skip the lane as well (a reranker with `--sleep-idle-seconds` cost the first turn after every idle pause). The worker now shares the turn's deadline, and the rerank call ends 0.3 s before it (at most a quarter of the timeout). When the reranker answers in time the output is unchanged. See docs/session-recall-prefetch.md "Rerank deadline".
- Automatic recall no longer injects a bare header. When the top-ranked memory was larger than the `display_tokens` budget, `format_for_prompt` stopped and the prompt received the header with no memory (42 of 51 measured questions at `display_tokens: 300`). The legacy and hybrid formatters now cut that memory to the remaining budget with a visible `… [truncated]` marker, skip an entry when too little of it would fit, never exceed `max(800, display_tokens * 4)` characters (separators included), and return an empty string when no memory can be shown, so the hybrid path falls back to legacy. The session lane's shared budget uses the same cap. docs/session-recall-prefetch.md now also documents two ratified behaviours: when only the reranker fails the lane keeps dense order, while a lane exception or timeout returns the legacy prefetch unchanged; and `qdrant_memory_session_expand` resolves any message id across sessions, so message ids act as capability handles (output redacted).

### Changed

- Session recall search now collapses duplicate windows at search time before the reranker: it over-fetches the dense pool (3x `session_recall_candidates`, capped at 512), groups exact copies by `text_sha256` (or the sha of the whitespace-collapsed text) and offset-shifted near-duplicates by a 75% contiguous token-run rule measured on real candidate pairs, keeps up to `session_recall_candidates` distinct texts, and returns `duplicate_count`/`also_in`/`also_in_truncated` provenance plus `candidates_raw` and `duplicates_collapsed` counts (`candidates` is now the distinct-text count). Scope, date and source filters still run inside the dense search, so provenance never names an out-of-scope session. `session_recall_dedup: false` restores the raw pool; the index itself is never deduplicated.
- Index refusals and partial failures now keep exit status 1 but write their human or JSON payload to stderr, matching the documented CLI output contract. Successful index output remains on stdout.

### Safety

- `QdrantClient` refuses every request against a `read_sources` collection except collection info, search, scroll, retrieve and count (an allow-list in `_request`, which every client method passes through), so no upsert, payload or access-counter update, delete, index or collection creation can reach a read source from any call site. Config load rejects a read source that names the provider's own memory, learning or session collection.
- Embedding requests that exceed the local llama.cpp physical batch/context budget are split into bounded chunks and combined into one normalized vector instead of failing the memory write/search path. Oversize-server-error fallback remains in place for tokenizer/Unicode cases where char estimates are imperfect.

## [0.9.0]

Ninth public beta release of the Hermes Qdrant Memory Provider. This release completes the Phase 3/4 source-derivation write-gate blocker and publishes the Phase 6 provenance layer: temporal assertion metadata, generic source-backed extraction candidates, read-only recall recipes/context templates, provenance-aware ranking, and ontology suggestion proposals. It also carries forward guarded-auto watcher policy work while preserving the plugin's no-Graphiti, no-graph-database, dry-run-first mutation boundary.

### Added

- Plugin metadata version bumped to `0.9.0`.
- Source derivation and progressive disclosure payloads for source-backed memory, including compact source metadata, resolver-backed inspect/trace/expand surfaces, memory grammar validation, and assertion-lite fields.
- Temporal validity metadata for assertions and source-backed facts: `observed_at`, `valid_from`, `valid_until`, `fact_status`, `supersedes`, `superseded_by`, and `invalidated_by`.
- Fact conflict, supersession, and status-update proposal types as local review/draft artifacts.
- Generic extraction candidate schema for memory, assertion, preference, invariant, risk, status-update, and ontology-suggestion candidates.
- Source-first extraction preview/approval flow with pending exact candidate IDs and shared write-gate validation.
- Recall recipe catalog and the read-only `qdrant_memory_context` tool / `hermes qdrant context` CLI for provenance-explicit context packets.
- Provenance-aware ranking policy that keeps raw vector scores auditable while applying transparent boosts and penalties for provenance, fact status, source health, derivation depth, and review/history queries.
- Ontology suggestion proposals for grammar/tag/fact-key improvements without schema self-modification.
- Watcher `--autonomy-mode guarded-auto` for preauthorized low-risk exact-ID maintenance: known heading-noise cleanup, exact-normalized duplicate merge, stale-low-value quarantine, and learning-to-skill draft artifacts.
- Guarded-auto policy controls for max actions and quarantine days, plus state/log fields for applied action and error counts.

### Safety

- `qdrant_learning_approve` and source-extraction approval gates now validate the full persisted payload, not only the display/lesson text, before any live write.
- Source extraction live approval requires prior dry-run preview, exact `candidate_id`, `dry_run=false`, and `approve=true`; unsafe candidates route to proposal drafts or fail closed.
- Missing provenance and secret-bearing persisted fields are refused before embedding/upsert.
- Deprecated and superseded facts are hidden from normal search by default, while inspect/trace/context review paths can include history explicitly.
- Identity-bearing fact conflicts redact snippets across response/report/draft surfaces and remain high-risk manual review.
- Guarded-auto still routes every live mutation through `qdrant_memory_consolidation_apply` with exact `report_id`, exact `proposal_id`, `dry_run=false`, and `approve=true`.
- Generic short markdown headings, near-duplicate clusters, secret-bearing inputs, `quality_warning`, reconsolidation candidates, fact-conflict proposals, and ontology suggestions remain manual-review/draft-only.
- No Graphiti runtime dependency, graph database, query-based mutation, automatic canonical assertion promotion, broad status rewrite, or self-modifying ontology was added.

### Documentation

- Documented the Hermes core discovery result for category-path installs: current Hermes discovers `~/.hermes/plugins/memory/qdrant` in the general plugin list as `memory/qdrant`, but memory-provider activation and native `hermes qdrant ...` CLI discovery still require the flat `~/.hermes/plugins/qdrant` compatibility path or symlink.
- Clarified the public distribution strategy: install the plugin under `~/.hermes/plugins/memory/qdrant` and expose it through the `~/.hermes/plugins/qdrant` compatibility symlink; defer Hermes core discovery changes until real adoption justifies a core PR.
- Added `docs/SOURCE_DERIVATIONS_BACKLOG.md` as the provenance/assertion backlog and design guardrail for future temporal/assertion work.

## [0.8.0]

Eighth public beta release of the Hermes Qdrant Memory Provider. This release focuses on operator safety and release confidence: watcher lifecycle CLI commands, live integration coverage, and dry-run-first manual memory store previews with optional non-destructive duplicate detection.

### Added

- Plugin metadata version bumped to `0.8.0`.
- Watcher lifecycle CLI commands: `hermes qdrant watcher install`, `uninstall`, `status --verbose`, `logs`, `inspect-state`, `reset-signature --approve`, and `run --force-alert`.
- Env-gated live integration tests for read-only search filters, provider store writes, file indexing upsert/stale-delete behavior, and gated consolidation report/apply paths against real Qdrant plus an OpenAI-compatible embedding endpoint.
- Documentation for `RUN_QDRANT_INTEGRATION` and `QDRANT_TEST_*` live test configuration in README and the operations runbook.
- Manual memory store previews: `qdrant_memory_store` now defaults to `dry_run=true`, and native `hermes qdrant store` exposes `--dry-run`, `--no-dry-run`, `--approve`, and `--preview-duplicates`.
- Optional semantic duplicate preview for manual stores, scoped to the configured memory collection/profile/source type, with configurable threshold and candidate count.

### Safety

- Watcher lifecycle commands are local scheduler/state/log operations only, with approval gates for uninstalling, replacing an existing managed cron block, and resetting proposal signatures.
- `watcher run` defaults to report-only consolidation with `dry_run=true`, `persist=true`, and no apply/proposal mutation; `--autonomy-mode guarded-auto` is opt-in and may apply only preauthorized low-risk exact-ID proposals through the gated apply path.
- Manual memory store live writes now require explicit `dry_run=false` plus `approve=true`; duplicate preview can skip an upsert but never deletes, merges, or rewrites existing memories.
- Live integration tests skip by default, use uniquely named temporary collections, delete only the exact collections created by the test fixture when names match the configured test prefix, and place consolidation artifacts under pytest temporary directories.

## [0.7.0]

Seventh public beta release of the Hermes Qdrant Memory Provider. This release adds read-only search filters for memory and learning search, including native CLI parity, while preserving the existing dry-run/review-gated mutation boundary.

### Added

- Plugin metadata version bumped to `0.7.0`.
- `qdrant_memory_search` now accepts `tags`, `source`, `file_path`, `project_path`, `since`, `until`, and `collection` filters.
- `qdrant_learning_search` now accepts `tags`, `source`, `file_path`, `project_path`, `since`, and `until` filters.
- Native CLI search commands now expose the same filters through `--tag`, `--source`, `--file-path` / `--path`, `--project-path`, `--since`, `--until`, and `--collection memory|learning` where applicable.
- `qdrant_memory_search(collection="learning")` routes through the learning collection search path for cross-collection operator workflows.

### Safety

- Search filters only narrow read results. They do not add query-based deletion, broad mutation, automatic reconsolidation, or any new write authority.
- Existing profile/platform scope filters are preserved and combined with the new Qdrant `must` conditions.
- Tool schemas keep strict `additionalProperties: false` behavior while adding the new filter parameters.

## [0.6.0]

Sixth public beta release of the Hermes Qdrant Memory Provider. This release adds read-only inspection commands for exact point lookup, persisted consolidation report review, and proposal review without expanding mutation authority.

### Added

- Plugin metadata version bumped to `0.6.0`.
- Read-only CLI inspection helpers:
  - `hermes qdrant show POINT_ID --collection memory|learning` for exact point metadata/payload/vector inspection.
  - `hermes qdrant reports list` and `hermes qdrant reports show REPORT_ID` for persisted consolidation artifacts.
  - `hermes qdrant proposals show REPORT_ID PROPOSAL_ID` for exact proposal review and expected-action display.

### Safety

- Inspection commands preserve the CLI output contract: human-readable defaults, stable `--json`, explicit payload/vector flags, exact IDs, local artifact validation, and no new mutation authority.
- `show` performs exact-ID retrieval and does not call embeddings, upsert, delete, or collection creation paths.
- Report/proposal IDs reject path traversal inputs; report/proposal inspection never contacts Qdrant or applies proposals.

## [0.5.0]

Fifth public beta release of the Hermes Qdrant Memory Provider. This release makes the native `hermes qdrant ...` CLI a dependable operator interface with human-readable defaults, stable `--json` mode, and documented stdout/stderr/error semantics.

### Added

- Plugin metadata version bumped to `0.5.0`.
- Canonical CLI output contract documentation for `hermes qdrant ...`.
- Human-readable default CLI output for status, doctor, config, provider-backed commands, watcher helpers, and recovery commands.
- `--json` support for `hermes qdrant status`.
- Fail-closed handling when provider-backed `--json` mode receives invalid JSON or non-object JSON.

### Changed

- `--json` is now the machine-readable mode; default success output is deterministic human text.
- Provider/service/safety errors print human text to stderr by default and JSON error objects to stderr when `--json` is set.
- CLI help now describes `--json` as machine-readable output rather than raw output.

### Safety

- Backup/export/restore human summaries remain sanitized and do not print raw memory payloads, vectors, or credentials.
- Existing dry-run and explicit-approval gates are unchanged.

## [0.4.0]

Fourth public beta release of the Hermes Qdrant Memory Provider. This release adds operator recovery primitives for export, backup, and restore after the v0.3.0 CLI parity and doctor diagnostics work.

### Added

- Plugin metadata version bumped to `0.4.0`.
- CLI recovery primitives for operators:
  - `hermes qdrant export memory|learning --out FILE [--overwrite]` writes one collection to a private JSONL artifact with raw payloads and vectors.
  - `hermes qdrant backup create [--scope memory|learning|both]` writes a private manifest plus collection JSONL files.
  - `hermes qdrant backup list` and `hermes qdrant backup inspect BACKUP_ID` inspect local backup artifacts without contacting Qdrant.
  - `hermes qdrant restore --backup BACKUP_ID` previews restore plans by default.

### Safety

- Live restore requires `--no-dry-run --approve`, validates backup checksums and target vector sizes before mutation, automatically creates a pre-restore backup, and performs additive/update-only upserts.
- Backup/export/restore stdout returns summaries only; raw payloads and vectors are confined to private local artifacts.
- Backup metadata URL redaction now fails closed and re-redacts stored manifest URLs during list/inspect.
- Local backup/export/restore service failures are reported as sanitized JSON CLI errors instead of raw tracebacks.

## [0.3.0]

Third public beta release of the Hermes Qdrant Memory Provider. This release broadens the native `hermes qdrant ...` CLI surface, keeps maintenance operations dry-run/review-gated, and tightens release hygiene for public users.

### Added

- Plugin metadata version bumped to `0.3.0`.
- M18 CLI parity commands:
  - `hermes qdrant config show` prints effective config JSON without constructing the provider and redacts API-key fields.
  - `hermes qdrant store TEXT` maps to `qdrant_memory_store` for explicit manual memory writes.
  - `hermes qdrant learning store LESSON` maps to `qdrant_learning_store`.
  - `hermes qdrant learning approve CANDIDATE_ID` maps to `qdrant_learning_approve` with dry-run default and live approval gate.
  - `hermes qdrant watcher status` reads local watcher state without contacting services.
  - `hermes qdrant watcher run` runs persisted consolidation in report-only mode by default; `--autonomy-mode guarded-auto` is opt-in and limited to preauthorized low-risk exact-ID apply paths.

### Fixed

- OpenAI-style `sk-...` secret detection now requires a left boundary, preventing false-positive quality warnings on TeamForge-style `task-...` IDs while still detecting real `sk-...` shaped keys.
- Release/test fixtures no longer contain a user-specific absolute Hermes venv path.

### Safety

- Preserved v0.2.1 CLI exit-code propagation and mutation gates.
- Watcher CLI remains report-only; no query deletion or autonomous apply was added.
- Backup/export helpers remain deferred until after v0.3.0 so broader mutation workflows are not widened without rollback primitives.

## [0.2.1]

Hotfix release discovered by the post-release consumer install/runtime smoke test.

### Fixed

- Native `hermes qdrant ...` commands now propagate `execute_command()` exit codes through `SystemExit`, so Hermes v0.13 CLI invocations return non-zero process status for usage/safety errors and provider JSON errors.
- Added regression coverage for `qdrant_command()` process-exit behavior.

### Verification

- Consumer smoke against the installed plugin found that v0.2.0 correctly blocked unapproved live mutation but exited with process status `0`; v0.2.1 fixes the process status while preserving the safety block.

## [0.2.0]

Second public beta release of the Hermes Qdrant Memory Provider.

### Added

- Plugin metadata version bumped to `0.2.0` for the second public beta.
- Qdrant-backed Hermes `MemoryProvider` for cross-session semantic recall.
- OpenAI-compatible embedding client support.
- Automatic completed-turn indexing through Hermes memory hooks.
- Manual memory tools:
  - `qdrant_memory_status`
  - `qdrant_memory_store`
  - `qdrant_memory_search`
  - `qdrant_memory_index`
  - `qdrant_memory_forget`
- Markdown/text file indexing with dry-run first behavior.
- File manifest sync, stale chunk detection, and conservative directory-level deleted-file sync.
- Separate procedural learning collection with explicit store/search/preview/approve tools.
- Gated automatic learning candidate extraction for narrow user-correction/tool-failure patterns.
- Semantic dedupe for automatic learning candidates.
- Conservative origin-time fact metadata for explicit tags, clear fact statements, headings, and structured learning context.
- Report-only sleep consolidation with persisted local report artifacts.
- Gated consolidation apply flow by exact `report_id` and `proposal_id`.
- Manual-review reconsolidation candidate reports and local draft-review artifacts.
- Native Hermes memory-provider CLI MVP via top-level `cli.py`:
  - `hermes qdrant status`
  - `hermes qdrant doctor`
  - `hermes qdrant search`
  - `hermes qdrant index`
  - `hermes qdrant forget`
  - `hermes qdrant learning search`
  - `hermes qdrant learning preview`
  - `hermes qdrant consolidate`
  - `hermes qdrant apply`
- CLI safety gates:
  - mutating commands default to dry-run;
  - live mutation requires `--no-dry-run --approve`;
  - deletion requires explicit point IDs;
  - consolidation apply requires exact report/proposal/action handles.
- Plugin metadata tests for `plugin.yaml` and `register(ctx)`.
- Scanner-safe fixture guard for docs/tests and CI.
- GitHub Actions CI for tests, compile checks, and scanner guard.

### Documentation

- Added formal plugin roadmap: `docs/PLUGIN_ROADMAP.md`.
- Added canonical safety contract: `docs/SAFETY.md`.
- Added operations runbook: `docs/OPERATIONS.md`.
- Added active-session LCM vs Qdrant memory boundary: `docs/LCM_BOUNDARY.md`.
- Added CLI feasibility spike: `docs/CLI_SPIKE.md`.
- Added setup/install playbook, Qdrant and embedding service guidance, CLI MVP examples, limitations, and examples.

### Safety

- Retrieved memories are context, not instructions.
- Current user instructions and live tool evidence override retrieved memory.
- Qdrant memory is not an LCM/current-session context engine replacement.
- Broad indexing and destructive maintenance operations must start with dry-run.
- Cron/watchers and consolidation are report/review oriented, not autonomous mutation surfaces.
- Reconsolidation does not rewrite Qdrant facts automatically.
- Quality-warning proposals remain manual-review only.

### Known limitations

- External Qdrant and embedding services are required.
- Only Markdown/text indexing is supported by default.
- No automatic secret detection for indexed user files.
- CLI discovery depends on the plugin being installed as the active Hermes memory provider.
- `hermes qdrant doctor` is status-backed in this release.
- The CLI is an MVP wrapper over the existing tool surface, not a separate long-running service.

## [0.1.0] - 2026-05-16

Initial public beta tag. The v0.1.0 tag predates the learning, consolidation, reconsolidation, scanner guard, and native CLI MVP work documented in later releases.

[0.9.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.9.0
[0.8.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.8.0
[0.7.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.7.0
[0.6.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.6.0
[0.5.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.5.0
[0.4.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.4.0
[0.3.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.3.0
[0.2.1]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.2.1
[0.2.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.2.0
[0.1.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.1.0
