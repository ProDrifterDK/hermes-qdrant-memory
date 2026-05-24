# Release Notes: v0.8.0 Public Beta

Hermes Qdrant Memory Provider v0.8.0 is a public beta release focused on safer operator workflows and stronger release confidence. It adds watcher lifecycle CLI commands, env-gated live integration coverage, and dry-run-first manual memory store previews with optional non-destructive semantic duplicate detection.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- Manual memory store is now dry-run-first:
  - `qdrant_memory_store` defaults to `dry_run=true`;
  - live writes require explicit `dry_run=false` plus `approve=true`;
  - native CLI exposes `hermes qdrant store --dry-run|--no-dry-run --approve`.
- Optional manual-store duplicate preview:
  - `--preview-duplicates` / `duplicate_preview=true` searches for semantic duplicates before live store;
  - duplicate hits return candidate details and skip the upsert;
  - duplicate preview never deletes, merges, or rewrites existing memories.
- Watcher lifecycle CLI commands:
  - `hermes qdrant watcher install`;
  - `hermes qdrant watcher uninstall --approve`;
  - `hermes qdrant watcher status --verbose`;
  - `hermes qdrant watcher logs`;
  - `hermes qdrant watcher inspect-state`;
  - `hermes qdrant watcher reset-signature --approve`;
  - `hermes qdrant watcher run --force-alert`.
- Env-gated live integration tests now cover real Qdrant plus an OpenAI-compatible embedding endpoint for:
  - read-only search filters;
  - provider store writes;
  - file indexing upsert/stale-delete behavior;
  - gated consolidation report/apply paths.
- README and operations docs now document `RUN_QDRANT_INTEGRATION` and `QDRANT_TEST_*` live test configuration.

## Manual store commands

Preview a memory without embedding or upserting:

```bash
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual
```

Preview and check for semantic duplicates:

```bash
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual --preview-duplicates
```

Perform an approved live store:

```bash
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual --preview-duplicates --no-dry-run --approve
```

Automation should use `--json` and check exit status:

```bash
hermes qdrant store "Remember this explicit memory" --preview-duplicates --json
hermes qdrant store "Remember this explicit memory" --preview-duplicates --no-dry-run --approve --json
```

## Watcher lifecycle commands

Inspect watcher state without contacting Qdrant:

```bash
hermes qdrant watcher status --verbose
hermes qdrant watcher inspect-state --json
hermes qdrant watcher logs --tail 20
```

Install or run the report-only watcher:

```bash
hermes qdrant watcher install --schedule "0 3 * * *" --json
hermes qdrant watcher run --scope both --force-alert --json
```

Approval-gated local scheduler/state mutations:

```bash
hermes qdrant watcher reset-signature --approve --json
hermes qdrant watcher uninstall --approve --json
```

## Safety behavior

The safety contract remains conservative:

- Default manual store calls do not mutate Qdrant.
- Live manual store requires explicit `dry_run=false` plus `approve=true`.
- Duplicate preview can skip an upsert but never deletes, merges, or rewrites existing memories.
- `watcher run` remains report-only consolidation with `dry_run=true`, `persist=true`, and no apply/proposal mutation.
- Watcher lifecycle commands mutate only local scheduler/state/log artifacts, never Qdrant collections.
- Live integration tests skip by default and use uniquely named temporary collections.
- Mutating maintenance commands still default to dry-run.
- Live maintenance mutation still requires explicit approval gates.
- `quality_warning` proposals remain manual-review only.
- Reconsolidation remains draft/review-only; no automatic fact rewrites.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.8.0
```

Start a fresh Hermes CLI process after upgrading. Restart the gateway only if gateway sessions should load the updated plugin code.

## Verification

Recommended local verification for maintainers:

```bash
python -m pytest tests -q
python scripts/check_no_literal_fake_secrets.py
python -m compileall -q qdrant_memory __init__.py cli.py scripts/check_no_literal_fake_secrets.py
git diff --check
```

Recommended targeted verification:

```bash
python -m pytest tests/test_tools_retriever_writer.py tests/test_cli.py tests/test_watcher_cli.py tests/integration/test_live_search_filters.py -q
```

Recommended consumer smoke after checking out the release tag:

```bash
hermes qdrant --help
hermes qdrant store --help
hermes qdrant store "release smoke memory" --preview-duplicates --json
hermes qdrant store "release smoke memory" --preview-duplicates --no-dry-run
hermes qdrant watcher status
hermes qdrant watcher status --json
hermes qdrant doctor
hermes qdrant doctor --json
```

For the unapproved live store check, expected behavior is:

- output contains `--approve is required when using --no-dry-run` or an equivalent approval-gate error;
- process exit status is non-zero;
- no Qdrant upsert occurs.

## Not included yet

The following remain future work:

- formal Python packaging beyond the Hermes plugin clone workflow;
- Hermes core support for user memory-provider discovery from `~/.hermes/plugins/memory/<name>` without the compatibility symlink. Post-release validation confirmed current core's general plugin scanner sees `~/.hermes/plugins/memory/qdrant` as `memory/qdrant`, but memory-provider activation and native `hermes qdrant ...` CLI discovery still require the flat `~/.hermes/plugins/qdrant` path or symlink;
- optional broader live-service stress tests beyond the current env-gated integration suite.
