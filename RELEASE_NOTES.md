# Release Notes: v0.5.0 Public Beta

Hermes Qdrant Memory Provider v0.5.0 is a public beta release focused on the native `hermes qdrant ...` operator experience: deterministic human-readable default output, stable machine-readable `--json`, and a documented CLI output contract.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- Default CLI output is now concise human-readable text for status, diagnostics, config, search/list-style commands, watcher helpers, and recovery commands.
- `--json` is the machine-readable mode for scripts and automation.
- `hermes qdrant status` now supports `--json`.
- JSON mode emits exactly one JSON object on success; JSON errors are written as one JSON object to stderr.
- Provider-backed `--json` mode fails closed when a provider returns invalid JSON or non-object JSON.
- Backup/export/restore human summaries remain sanitized: no raw memory payload text, vectors, or credentials are printed.
- The canonical output rules are documented in `docs/CLI_OUTPUT_CONTRACT.md`.

## CLI output contract

Default mode is for humans:

```bash
hermes qdrant status
hermes qdrant doctor
hermes qdrant config show
hermes qdrant search "agent memory" --top-k 5
hermes qdrant learning preview
hermes qdrant watcher status
hermes qdrant backup list
```

Automation should use `--json` and check exit status:

```bash
hermes qdrant status --json
hermes qdrant doctor --json
hermes qdrant config show --json
hermes qdrant search "agent memory" --top-k 5 --json
hermes qdrant learning preview --json
hermes qdrant watcher status --json
hermes qdrant backup list --json
```

Contract summary:

- Success in default mode prints bounded, deterministic human text on stdout.
- Success in `--json` mode prints one parseable JSON object on stdout.
- Usage/safety errors exit `2`.
- Provider/service/runtime errors exit `1`.
- Human errors print text to stderr.
- JSON errors print one parseable JSON object to stderr.
- Default human summaries are not a stable API and should not be parsed by scripts.

## Safety behavior

The safety contract remains conservative:

- Mutating maintenance commands default to dry-run.
- Live maintenance mutation requires both `--no-dry-run` and `--approve`.
- `forget` requires explicit point IDs.
- `apply` requires exact `report_id`, `proposal_id`, and expected action.
- `learning approve` is dry-run by default and live-gated.
- `watcher run` is report-only consolidation: it persists a local report artifact but does not apply proposals or mutate Qdrant.
- `quality_warning` proposals remain manual-review only.
- Reconsolidation remains draft/review-only; no automatic fact rewrites.

Recovery-specific safety:

- `export memory|learning` and `backup create` write local recovery artifacts only; they perform no Qdrant mutation.
- Export and backup artifacts intentionally contain raw memory payload text and vectors. Treat them as private recovery material.
- Artifact directories are private (`0700`) and artifact files are private (`0600`) where POSIX modes are available.
- CLI stdout prints summaries, counts, paths, and checksums only — not raw payloads or vectors.
- `backup list` and `backup inspect` read local artifacts only and re-redact stored Qdrant URLs before printing.
- `restore` validates artifact checksums and target collection vector sizes before any live mutation.
- Live restore requires `--no-dry-run --approve`, automatically creates a pre-restore backup, and performs additive/update-only upserts.
- Restore does not delete by query or filter.

Explicit manual write commands are intentionally live writes:

- `hermes qdrant store TEXT`
- `hermes qdrant learning store LESSON`

Use them only for content you deliberately want to store.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.5.0
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

Recommended consumer smoke after checking out the release tag:

```bash
hermes qdrant config show
hermes qdrant config show --json
hermes qdrant status
hermes qdrant status --json
hermes qdrant doctor
hermes qdrant doctor --json
hermes qdrant search "Hermes Qdrant memory" --top-k 3
hermes qdrant search "Hermes Qdrant memory" --top-k 3 --json
hermes qdrant learning preview
hermes qdrant learning preview --json
hermes qdrant consolidate --scope both --persist --include-reconsolidation
hermes qdrant watcher status
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both
hermes qdrant export memory --out /tmp/qdrant-memory-export.jsonl --json
hermes qdrant backup create --scope both --json
hermes qdrant backup list --json
hermes qdrant backup inspect BACKUP_ID --json
hermes qdrant restore --backup BACKUP_ID --dry-run --json
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

- point/report inspection ergonomics such as `show`, `reports list`, `reports show`, and `proposals show`;
- richer search filters by tag, source/path, date range, and collection;
- optional `--dry-run`/duplicate-preview flows for explicit manual stores;
- env-gated live-service integration tests;
- watcher install/uninstall/log management commands;
- formal Python packaging beyond the Hermes plugin clone workflow.
