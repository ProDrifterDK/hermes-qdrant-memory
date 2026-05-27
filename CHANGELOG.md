# Changelog

All notable changes to this project are documented here.

This project is currently public beta / experimental. The format follows Keep a Changelog style, but the repository is still pre-1.0 and the public API may change.

## [Unreleased]

_No changes yet._

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
