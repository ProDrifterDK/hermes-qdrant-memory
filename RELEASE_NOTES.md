# Release Notes: v0.6.0 Public Beta

Hermes Qdrant Memory Provider v0.6.0 is a public beta release focused on read-only inspection ergonomics for the native `hermes qdrant ...` CLI. It adds exact point lookup, persisted consolidation report inspection, and proposal inspection while preserving the conservative mutation boundary established in earlier releases.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- New exact-ID point inspection: `hermes qdrant show POINT_ID --collection memory|learning`.
- Optional payload inspection with `--include-payload`; payloads remain omitted by default.
- Optional vector inspection with `--include-vector`; vectors remain omitted by default.
- New local report artifact inspection:
  - `hermes qdrant reports list`
  - `hermes qdrant reports show REPORT_ID`
- New local proposal inspection:
  - `hermes qdrant proposals show REPORT_ID PROPOSAL_ID`
- Report/proposal IDs are validated and path traversal inputs are rejected.
- Documentation now covers the inspection workflow in README, operations, roadmap, and CLI output-contract docs.

## Inspection commands

Default mode is for humans:

```bash
hermes qdrant show POINT_ID --collection memory
hermes qdrant show POINT_ID --collection learning --include-payload
hermes qdrant reports list
hermes qdrant reports show REPORT_ID
hermes qdrant proposals show REPORT_ID PROPOSAL_ID
```

Automation should use `--json` and check exit status:

```bash
hermes qdrant show POINT_ID --collection memory --json
hermes qdrant show POINT_ID --collection learning --include-payload --json
hermes qdrant show POINT_ID --collection memory --include-vector --json
hermes qdrant reports list --json
hermes qdrant reports show REPORT_ID --json
hermes qdrant proposals show REPORT_ID PROPOSAL_ID --json
```

Behavior summary:

- `show` retrieves exactly one explicit point ID from the selected configured collection.
- Missing points are valid read results and return `found: false` with process status `0`.
- `reports list/show` and `proposals show` read persisted local consolidation artifacts only.
- Human output remains bounded and intended for operators, not parsers.
- `--json` remains the stable machine-readable mode.

## Safety behavior

The safety contract remains conservative:

- Inspection commands do not introduce new mutation authority.
- `show` does not call embeddings, upsert, delete, or collection-creation paths.
- `reports` and `proposals` do not contact Qdrant and cannot apply proposals.
- Payloads are omitted unless `--include-payload` is explicit.
- Vectors are omitted unless `--include-vector` is explicit.
- Report/proposal IDs are exact local artifact handles and reject path traversal input.
- Mutating maintenance commands still default to dry-run.
- Live maintenance mutation still requires explicit approval gates.
- `quality_warning` proposals remain manual-review only.
- Reconsolidation remains draft/review-only; no automatic fact rewrites.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.6.0
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
hermes qdrant --help
hermes qdrant show --help
hermes qdrant reports list
hermes qdrant reports list --json
hermes qdrant doctor
hermes qdrant doctor --json
hermes qdrant watcher status
hermes qdrant watcher status --json
```

If you have known real point IDs and persisted report artifacts, also verify:

```bash
hermes qdrant show POINT_ID --collection memory
hermes qdrant show POINT_ID --collection memory --include-payload --json
hermes qdrant show POINT_ID --collection memory --include-vector --json
hermes qdrant reports show REPORT_ID
hermes qdrant proposals show REPORT_ID PROPOSAL_ID
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

- richer search filters by tag, source/path, date range, and collection;
- optional `--dry-run`/duplicate-preview flows for explicit manual stores;
- env-gated live-service integration tests;
- watcher install/uninstall/log management commands;
- formal Python packaging beyond the Hermes plugin clone workflow.
