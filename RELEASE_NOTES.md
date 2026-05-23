# Release Notes: v0.3.0 Public Beta

Hermes Qdrant Memory Provider v0.3.0 is a public beta release focused on operational CLI parity and release hygiene for the Qdrant-backed Hermes `MemoryProvider`.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- Native `hermes qdrant ...` CLI surface expanded beyond the v0.2.x MVP.
- Explicit manual write commands for memories and procedural learnings.
- Local config inspection and watcher-state/report commands.
- Preserved process-exit correctness from v0.2.1.
- Hardened secret/noise detection to avoid false-positive warnings on `task-...` IDs.
- Release fixtures cleaned of user-specific absolute paths.

## New CLI commands

v0.3.0 adds the M18 CLI parity layer:

```bash
hermes qdrant config show --json
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual
hermes qdrant learning store "Prefer dry-run before broad indexing" --learning-type workflow_lesson --confidence 0.8 --tag cli
hermes qdrant learning approve CANDIDATE_ID --dry-run --json
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both --json
```

Existing v0.2.x commands remain available:

```bash
hermes qdrant status
hermes qdrant doctor
hermes qdrant search "agent memory" --top-k 5 --json
hermes qdrant index docs README.md --dry-run
hermes qdrant forget POINT_ID --dry-run
hermes qdrant learning search "tool failure" --top-k 5 --json
hermes qdrant learning preview --json
hermes qdrant consolidate --scope both --persist --dry-run --json
hermes qdrant apply --report-id REPORT_ID --proposal-id PROPOSAL_ID --action merge --dry-run --json
```

## Safety behavior

Unchanged safety contract:

- Mutating maintenance commands default to dry-run.
- Live maintenance mutation requires both `--no-dry-run` and `--approve`.
- `forget` requires explicit point IDs.
- `apply` requires exact `report_id`, `proposal_id`, and expected action.
- `learning approve` is dry-run by default and live-gated.
- `watcher run` is report-only consolidation: it persists a local report artifact but does not apply proposals or mutate Qdrant.
- `quality_warning` proposals remain manual-review only.
- Reconsolidation remains draft/review-only; no automatic fact rewrites.

Explicit manual write commands are intentionally live writes:

- `hermes qdrant store TEXT`
- `hermes qdrant learning store LESSON`

Use them only for content you deliberately want to store.

## Fixed

- OpenAI-style `sk-...` secret detection now requires a left boundary, so TeamForge-style IDs such as `task-...` no longer trigger false-positive secret/quality warnings through their `sk-...` substring.
- Scanner/test fixtures no longer include a user-specific absolute Hermes venv path.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.3.0
```

Start a fresh Hermes CLI process after upgrading. Restart the gateway only if gateway sessions should load the updated plugin code.

## Verification

Recommended local verification for maintainers:

```bash
python -m pytest tests -q
python scripts/check_no_literal_fake_secrets.py
python -m compileall -q qdrant_memory __init__.py cli.py scripts/check_no_literal_fake_secrets.py
```

Recommended consumer smoke after checking out the release tag:

```bash
hermes qdrant config show --json
hermes qdrant status
hermes qdrant doctor
hermes qdrant search "Hermes Qdrant memory" --top-k 3 --json
hermes qdrant learning preview --json
hermes qdrant consolidate --scope both --persist --include-reconsolidation --json
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both --json
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

The following remain post-v0.3.0 work:

- backup/export/rollback CLI helpers;
- human-friendly non-JSON formatting as the default CLI output;
- deep `doctor` diagnostics beyond status-backed health checks;
- env-gated live-service integration tests;
- watcher install/uninstall/log management commands;
- formal Python packaging beyond the Hermes plugin clone workflow.
