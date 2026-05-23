# Changelog

All notable changes to this project are documented here.

This project is currently public beta / experimental. The format follows Keep a Changelog style, but the repository is still pre-1.0 and the public API may change.

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
  - `hermes qdrant watcher run` runs report-only persisted consolidation with no apply/proposal mutation.

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

[0.3.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.3.0
[0.2.1]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.2.1
[0.2.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.2.0
[0.1.0]: https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.1.0
