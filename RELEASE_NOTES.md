# Release Notes: v0.4.0 Public Beta

Hermes Qdrant Memory Provider v0.4.0 is a public beta release focused on operator recovery: export, backup, inspection, and restore for the Qdrant-backed Hermes `MemoryProvider`.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- New CLI recovery primitives for Qdrant memory and learning collections.
- Private local backup/export artifacts with checksums and manifests.
- Restore planning is dry-run by default and validates artifacts before mutation.
- Live restore requires explicit approval and automatically creates a pre-restore backup.
- Backup/export/restore errors are sanitized JSON CLI errors rather than raw tracebacks.
- v0.3.0 CLI parity and doctor diagnostics remain available.

## New CLI commands

v0.4.0 adds the recovery layer:

```bash
hermes qdrant export memory --out memory.jsonl --json
hermes qdrant export learning --out learning.jsonl --json
hermes qdrant backup create --scope both --json
hermes qdrant backup list --json
hermes qdrant backup inspect BACKUP_ID --json
hermes qdrant restore --backup BACKUP_ID --dry-run --json
```

Existing CLI commands remain available:

```bash
hermes qdrant config show --json
hermes qdrant status
hermes qdrant doctor --json
hermes qdrant search "agent memory" --top-k 5 --json
hermes qdrant index docs README.md --dry-run
hermes qdrant forget POINT_ID --dry-run
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual
hermes qdrant learning search "tool failure" --top-k 5 --json
hermes qdrant learning preview --json
hermes qdrant learning store "Prefer dry-run before broad indexing" --learning-type workflow_lesson --confidence 0.8 --tag cli
hermes qdrant learning approve CANDIDATE_ID --dry-run --json
hermes qdrant consolidate --scope both --persist --dry-run --json
hermes qdrant apply --report-id REPORT_ID --proposal-id PROPOSAL_ID --action merge --dry-run --json
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both --json
```

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

## Fixed and hardened

- Restore correctly matches existing points with numeric IDs as well as string IDs.
- Backup metadata URL redaction fails closed and re-redacts stored manifest URLs during list/inspect.
- Malformed backup artifacts are reported as controlled backup errors.
- Qdrant service failures during local CLI operations are reported as sanitized JSON errors.
- Restore live preflights all affected collections before the first upsert.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.4.0
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
hermes qdrant config show --json
hermes qdrant status
hermes qdrant doctor --json
hermes qdrant search "Hermes Qdrant memory" --top-k 3 --json
hermes qdrant learning preview --json
hermes qdrant consolidate --scope both --persist --include-reconsolidation --json
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both --json
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

- human-friendly non-JSON formatting as the default CLI output;
- stable documented JSON schemas for automation;
- env-gated live-service integration tests;
- watcher install/uninstall/log management commands;
- point/report inspection ergonomics such as `show`, `reports list`, and `reports show`;
- formal Python packaging beyond the Hermes plugin clone workflow.
