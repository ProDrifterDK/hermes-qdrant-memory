# Release Notes: v0.7.0 Public Beta

Hermes Qdrant Memory Provider v0.7.0 is a public beta release focused on read-only search ergonomics. It adds structured filters for memory and learning search across both Hermes tools and the native `hermes qdrant ...` CLI while preserving the conservative mutation boundary established in earlier releases.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- New read-only filters for `qdrant_memory_search`:
  - `tags`
  - `source`
  - `file_path`
  - `project_path`
  - `since`
  - `until`
  - `collection` (`memory` or `learning`)
- New read-only filters for `qdrant_learning_search`:
  - `tags`
  - `source`
  - `file_path`
  - `project_path`
  - `since`
  - `until`
- Native CLI parity for search filters:
  - `--tag`
  - `--source`
  - `--file-path` / `--path`
  - `--project-path`
  - `--since`
  - `--until`
  - `--collection memory|learning` on base search
- `qdrant_memory_search(collection="learning")` routes through the learning collection search path for cross-collection operator workflows.
- Documentation now covers filter behavior in README, changelog, roadmap, and release notes.

## Filter commands

Default mode is for humans:

```bash
hermes qdrant search "agent memory" --tag docs --source-type project_doc
hermes qdrant search "agent memory" --source docs/api.md --file-path /repo/docs/api.md
hermes qdrant search "agent memory" --project-path /repo --since 2026-01-01T00:00:00Z --until 2026-01-31T23:59:59Z
hermes qdrant search "tool failure" --collection learning --tag pytest
hermes qdrant learning search "tool failure" --learning-type workflow_lesson --tag pytest
```

Automation should use `--json` and check exit status:

```bash
hermes qdrant search "agent memory" --tag docs --source-type project_doc --json
hermes qdrant search "tool failure" --collection learning --tag pytest --json
hermes qdrant learning search "tool failure" --learning-type workflow_lesson --tag pytest --json
```

Behavior summary:

- Repeated `--tag` values and comma-separated tags are combined into `tags` filter conditions.
- `--source`, `--file-path` / `--path`, and `--project-path` are exact payload-field filters.
- `--since` and `--until` apply inclusive `created_at` range filters.
- `--collection learning` on base search routes through the learning collection. Use `hermes qdrant learning search` when `--learning-type` is also needed.
- Human output remains bounded and intended for operators, not parsers.
- `--json` remains the stable machine-readable mode.

## Safety behavior

The safety contract remains conservative:

- Search filters only narrow read results.
- Filters do not add query-based deletion, broad mutation, automatic reconsolidation, or any new write authority.
- Existing profile/platform scope filters are preserved and combined with the new Qdrant `must` conditions.
- Tool schemas keep strict `additionalProperties: false` behavior while adding the new filter parameters.
- Mutating maintenance commands still default to dry-run.
- Live maintenance mutation still requires explicit approval gates.
- `quality_warning` proposals remain manual-review only.
- Reconsolidation remains draft/review-only; no automatic fact rewrites.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.7.0
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

Recommended targeted search-filter verification:

```bash
python -m pytest tests/test_retriever_search_filters.py tests/test_learning.py tests/test_cli.py -q
```

Recommended consumer smoke after checking out the release tag:

```bash
hermes qdrant --help
hermes qdrant search --help
hermes qdrant learning search --help
hermes qdrant search "agent memory" --tag docs --json
hermes qdrant search "tool failure" --collection learning --tag pytest --json
hermes qdrant learning search "tool failure" --learning-type workflow_lesson --tag pytest --json
hermes qdrant doctor
hermes qdrant doctor --json
hermes qdrant watcher status
hermes qdrant watcher status --json
```

Verify the CLI process-exit gate for live mutation without approval:

```bash
hermes qdrant forget 00000000-0000-0000-0000-000000000000 --no-dry-run
```

Expected:

- output contains `--approve is required when using --no-dry-run`;
- process exit status is non-zero;
- no Qdrant deletion occurs.

## Not included yet

The following remain future work:

- optional `--dry-run`/duplicate-preview flows for explicit manual stores;
- env-gated live-service integration tests;
- watcher install/uninstall/log management commands;
- formal Python packaging beyond the Hermes plugin clone workflow.
