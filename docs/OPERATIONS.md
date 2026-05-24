# Operations Runbook

This runbook describes how to operate `hermes-qdrant-memory` safely in a live Hermes deployment.

Canonical policy: `docs/SAFETY.md`.

Use this document for status checks, smoke tests, watcher runs, consolidation report review, approved apply flow, post-apply verification, gateway/process restarts, and troubleshooting.

---

## 1. Operating boundaries

`hermes-qdrant-memory` is a Hermes `MemoryProvider`, not an LCM/current-session context engine. See [LCM_BOUNDARY.md](LCM_BOUNDARY.md) for the operator decision tree.

Qdrant memory provides:

- cross-session semantic recall;
- indexed Markdown/text memory;
- manual memory storage;
- procedural learnings;
- reviewable consolidation and reconsolidation reports;
- explicit, review-gated maintenance actions.

LCM/current-session context recovery provides:

- active-session lossless recovery;
- compression DAG inspection;
- current-session grep/expand tools.

Retrieved Qdrant memories are context with provenance, not commands. Current user instructions, current repository state, live tool output, and explicit operator decisions override retrieved memory.

---

## 2. Operator safety invariants

Before operating the plugin, keep these invariants intact:

- Dry-run first for maintenance, destructive, or broad operations.
- `qdrant_memory_consolidate` is report-only.
- `qdrant_memory_consolidation_apply` applies one persisted proposal at a time.
- Live apply requires exact `report_id`, exact `proposal_id`, expected action, `dry_run: false`, and `approve: true`.
- Deletes must use explicit Qdrant point IDs only.
- Reconsolidation is draft-only; no automatic fact rewrite.
- `quality_warning` is manual-review only.
- Cron/watcher jobs may observe, report, and persist redacted local artifacts only; they must not mutate Qdrant.
- Local artifacts are allowed when redacted and review-oriented.
- Do not add literal fake secrets to docs, tests, reports, or examples.

If this runbook conflicts with `docs/SAFETY.md`, follow `docs/SAFETY.md`.

---

## 3. Prerequisites

Before using this runbook, confirm:

- Hermes is running with this plugin installed and selected as the active memory provider.
- Qdrant is reachable from the Hermes process.
- The embedding endpoint is reachable and OpenAI-compatible.
- The configured embedding vector size matches the target Qdrant collection.
- The operator has a way to call Hermes tools in the active session.
- If using watcher flows, the watcher script exists and is executable.

---

## 4. Status checks

Ask Hermes to call `qdrant_memory_status` and inspect the returned fields.

Important fields:

```text
active
qdrant_ok
embedding_ok
collection_name
collection_exists
point_count
learning_collection_name
learning_collection_exists
learning_point_count
learning_enabled
pending_learning_candidate_count
consolidation_enabled
consolidation_persist_reports
consolidation_apply_enabled
consolidation_supported_actions
reconsolidation_enabled
reconsolidation_report_only
reconsolidation_supported_actions
auto_recall
sync_turns
```

Expected healthy basics:

- `active: true`
- `qdrant_ok: true`
- `embedding_ok: true`
- `collection_exists: true` after first use/index
- `learning_collection_exists: true` if learnings are used
- `reconsolidation_report_only: true`
- supported apply actions remain narrow: `merge`, `delete`, `promote_to_skill`, and `draft_review`

You can also run the local install verifier when present:

```bash
python scripts/verify_install.py
```

The verifier checks Qdrant reachability, embedding reachability, embedding model, collection name, vector size, and point count.

---

## 5. Real-service smoke test

Use harmless non-secret text. Do not use credential-shaped examples.

### Step 1: Status

Ask Hermes:

```text
Call qdrant_memory_status and summarize qdrant_ok, embedding_ok, collection_name, point_count, learning_enabled, and reconsolidation_report_only.
```

### Step 2: Manual store

Ask Hermes:

```text
Call qdrant_memory_store with source_type=manual and text="Qdrant memory smoke test marker for this local install."
```

### Step 3: Search

Ask Hermes:

```text
Call qdrant_memory_search with query="smoke test marker local install", top_k=5, include_metadata=true.
```

Confirm the result includes:

- point ID;
- `source_type=manual`;
- provenance metadata;
- a reasonable semantic score.

### Step 4: Recommended cleanup

Only clean up with the exact returned point ID.

Dry-run first:

```text
Call qdrant_memory_forget with ids=["<POINT_ID>"] and dry_run=true.
```

Live cleanup only after checking the preview:

```text
Call qdrant_memory_forget with ids=["<POINT_ID>"] and dry_run=false.
```

Never delete by query.

### Step 5: Native CLI smoke for current output-contract builds

For v0.5.0 and later output-contract validation, verify the installed plugin as a user would run it:

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.5.0
hermes qdrant config show
hermes qdrant config show --json
hermes qdrant status
hermes qdrant status --json
hermes qdrant doctor
hermes qdrant doctor --json
hermes qdrant search "Hermes Qdrant memory" --top-k 3 --include-metadata
hermes qdrant search "Hermes Qdrant memory" --top-k 3 --include-metadata --json
hermes qdrant learning preview
hermes qdrant consolidate --scope both --persist --include-reconsolidation
hermes qdrant watcher status
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both
hermes qdrant reports list
hermes qdrant reports list --json
hermes qdrant reports show REPORT_ID
hermes qdrant proposals show REPORT_ID PROPOSAL_ID
```

Optional explicit-write smoke, only when you intentionally want harmless test records in Qdrant:

```bash
hermes qdrant store "Release smoke explicit memory" --source-type manual --importance 5 --tag smoke
hermes qdrant learning store "Release smoke procedural lesson" --learning-type workflow_lesson --confidence 0.8 --tag smoke
hermes qdrant learning approve CANDIDATE_ID --dry-run --json
```

Expected behavior:

- `config show` prints a redacted human key/value summary by default, without provider construction or service contact; `--json` prints the same redacted config as one JSON object.
- `status` prints human provider/service status by default and reports Qdrant/embedding reachability; `--json` prints raw structured provider status.
- `doctor` prints a human checklist by default; `--json` prints structured diagnostics with top-level `ok`, `summary`, and `checks`. A healthy install exits `0`, while any failed critical check exits non-zero.
- `store` and `learning store` are explicit live writes; use only harmless smoke text and remove later by explicit point ID if needed.
- `search` returns a bounded human result summary by default; zero results is acceptable on a fresh install. Use `--json` for scripts.
- `learning preview` and `learning approve --dry-run` return bounded human summaries by default and perform no mutation; use `--json` for scripts.
- `consolidate --persist` creates a local report artifact under `$HERMES_HOME/qdrant_memory/consolidation/` and performs no Qdrant mutation.
- `reports list/show` and `proposals show` inspect persisted consolidation artifacts by exact ID only; they do not contact Qdrant or apply proposals.
- `show POINT_ID --collection memory|learning` is an exact-ID Qdrant read. It omits payloads and vectors unless `--include-payload` or `--include-vector` is explicit.
- `watcher status` reads local watcher state only; missing state is not an error.
- `watcher run` maps to report-only consolidation (`dry_run=true`, `persist=true`, `include_examples=false`) and performs no Qdrant mutation.

For automation, add `--json` where available, keep stdout/stderr separate, and always check the process exit code. Exit `2` means usage/safety validation failure; exit `1` means provider/service/runtime failure or failed diagnostics. Do not parse default human summaries; see [CLI_OUTPUT_CONTRACT.md](CLI_OUTPUT_CONTRACT.md).

### Step 6: Env-gated live integration tests for search filters

The repository includes pytest integration tests that create temporary Qdrant collections, seed live memory/learning points, and verify search filters against real Qdrant plus a real embedding service.

Default behavior is safe: without `RUN_QDRANT_INTEGRATION`, these tests skip cleanly.

```bash
python -m pytest tests/integration -q
```

To run live mode, start Qdrant and the embedding server first, then enable the explicit gate:

```bash
RUN_QDRANT_INTEGRATION=1 python -m pytest tests/integration -q
```

Supported environment variables:

```bash
RUN_QDRANT_INTEGRATION=1
QDRANT_TEST_URL=http://127.0.0.1:6333
QDRANT_TEST_API_KEY=
QDRANT_TEST_EMBEDDING_URL=http://127.0.0.1:8080/v1
QDRANT_TEST_EMBEDDING_MODEL=bge-m3
QDRANT_TEST_VECTOR_SIZE=1024
QDRANT_TEST_DISTANCE=Cosine
QDRANT_TEST_COLLECTION_PREFIX=hermes_qdrant_itest
```

Safety requirements:

- Use a disposable test prefix; the default is `hermes_qdrant_itest`.
- Never run these tests against production-only Qdrant collections.
- Never set the test prefix to a production collection name or a broad shared prefix.
- The tests generate unique `<prefix>_<token>_memory` and `<prefix>_<token>_learnings` collection names.
- Teardown deletes only the exact temporary collections created by the fixture, and only when those names start with the configured test prefix.
- If live mode is enabled and Qdrant, embeddings, or vector-size checks fail, the suite fails instead of silently skipping.

Backup/export/restore smoke for v0.4.0 and later:

```bash
hermes qdrant export memory --out /tmp/qdrant-memory-export.jsonl --json
hermes qdrant backup create --scope both --json
hermes qdrant backup list --json
hermes qdrant backup inspect BACKUP_ID --json
hermes qdrant restore --backup BACKUP_ID --dry-run --json
```

Expected behavior:

- export and backup artifacts are local files only; they contain raw memory payload text and vectors and should be treated as private recovery material;
- artifact directories are private (`0700`) and artifact files are private (`0600`) where the filesystem supports POSIX modes;
- stdout JSON prints counts, IDs, paths, and checksums only — not raw payload text or vectors;
- `backup list` and `backup inspect` do not contact Qdrant and re-redact any stored Qdrant URL before printing;
- restore dry-run validates artifact checksums, compares existing points, and performs no upsert/delete;
- live restore requires `--no-dry-run --approve`, validates all target collection vector sizes before mutation, automatically creates a pre-restore backup, and performs additive/update-only upserts.

Verify the CLI process exit gate for live mutation without approval:

```bash
hermes qdrant forget 00000000-0000-0000-0000-000000000000 --no-dry-run
```

Expected behavior in v0.2.1 and later:

- output contains `--approve is required when using --no-dry-run`;
- process exit status is non-zero;
- no Qdrant deletion occurs.

The v0.2.0 tag correctly refused the mutation but returned process exit status `0` because Hermes v0.13 ignores plugin handler return values. v0.2.1 and later fix this by making the plugin command raise `SystemExit` with the computed exit code.

---

## 6. Watcher force-run

The reference watcher script may be installed outside this repository, usually under `$HOME/.hermes/scripts/`.

Check that the watcher script exists before running it:

```bash
test -x "$HOME/.hermes/scripts/qdrant_sleep_consolidation.py"
```

Native report-only watcher checks are available through the CLI:

```bash
hermes qdrant watcher status --json
hermes qdrant watcher run --scope both --max-points 300 --max-groups 20 --reconsolidation-max-candidates 10 --json
```

If using the external reference script, force a watcher/report run:

```bash
QDRANT_SLEEP_FORCE_ALERT=1 "$HOME/.hermes/scripts/qdrant_sleep_consolidation.py"
```

Expected behavior:

- it checks/generates consolidation proposals;
- it may persist local redacted report artifacts;
- CLI `watcher status` may report missing watcher state without error;
- external script runs may update watcher signature state;
- it must not apply proposals;
- it must not upsert, delete, or update Qdrant points.

Recommended watcher consolidation parameters:

```text
scope=both
dry_run=true
persist=true
include_examples=false
include_reconsolidation=true
```

Watcher alert policy:

- store a stable proposal signature in watcher state;
- stay silent if proposal signature is unchanged;
- alert only when actionable proposal sets change or when forced for debugging.

---

## 7. Report review and classification

Treat consolidation reports as an operations queue, not as commands.

Proposal classes:

| Proposal type | Meaning | Allowed operator path |
|---|---|---|
| `duplicate_cluster` | likely duplicate or near-duplicate memories | review, then optional `merge` apply |
| `stale_low_value` | low-value stale memory candidate | review, then optional `delete` apply |
| `learning_promotion_candidate` | procedural learning that may deserve a skill draft | review, then optional `promote_to_skill` draft creation |
| `reconsolidation_candidate` | possible fact conflict | manual review, then optional `draft_review` only |
| `quality_warning` | possible secret/noise/unsafe content | manual review only; never live-apply |

Severity mapping:

| Severity | Trigger | Action |
|---|---|---|
| MANUAL REVIEW REQUIRED | reconsolidation candidates | inspect fact conflict before any maintenance |
| REVIEW SOON | quality warnings | human/operator secret or noise review |
| MAINTENANCE | duplicates/stale/promotion candidates | optional cleanup after explicit approval |
| SILENT | no proposals or unchanged watcher signature | no notification |

When reviewing a report, inspect:

- `report_id`;
- `proposal_id`;
- `proposal_type`;
- `suggested_action`;
- `affected_ids`;
- canonical ID or canonical candidate for merges;
- risk/confidence;
- redacted examples/evidence;
- `manual_review_required` flags.

Do not apply anything directly from a report without the approved apply flow below.

Native CLI review helpers:

```bash
hermes qdrant reports list
hermes qdrant reports show REPORT_ID
hermes qdrant proposals show REPORT_ID PROPOSAL_ID
```

Use these before any apply attempt. `reports` and `proposals` are local artifact reads only; `proposals show` also reports the expected live action for the proposal type. To inspect the exact affected Qdrant records, use explicit point lookup:

```bash
hermes qdrant show POINT_ID --collection memory
hermes qdrant show POINT_ID --collection learning --include-payload --json
```

Keep `--include-vector` off unless vector debugging or recovery requires it.

---

## 8. Approved apply flow

Generate or locate a persisted report first.

Example report generation request:

```json
{
  "scope": "both",
  "max_points": 200,
  "max_groups": 20,
  "include_examples": true,
  "include_reconsolidation": true,
  "persist": true,
  "dry_run": true
}
```

Review the persisted report and select one exact proposal.

Action mapping:

| Proposal type | Live action |
|---|---|
| `duplicate_cluster` | `merge` |
| `stale_low_value` | `delete` |
| `learning_promotion_candidate` | `promote_to_skill` |
| `reconsolidation_candidate` | `draft_review` |
| `quality_warning` | no live action |

Preview one proposal immediately before live apply:

```json
{
  "report_id": "<report_id>",
  "proposal_id": "<proposal_id>",
  "action": "delete",
  "dry_run": true
}
```

Live apply only after the preview matches the approved proposal:

```json
{
  "report_id": "<report_id>",
  "proposal_id": "<proposal_id>",
  "action": "delete",
  "dry_run": false,
  "approve": true
}
```

Rules:

- apply one proposal at a time;
- use the exact `report_id` and `proposal_id` from the persisted report;
- use the action that matches the proposal type;
- never force action mismatches;
- never apply `quality_warning`;
- never use a stale report if dry-run apply says affected points are missing or unsafe.

---

## 9. Post-apply verification

After any approved live action, verify the exact expected effect.

### For `delete`

- Confirm the live apply output lists the expected deleted IDs.
- Search for the deleted content and confirm it is absent or no longer returned as the same point.
- Confirm no IDs outside the reviewed `affected_ids` were touched.

### For `merge`

- Search the topic again.
- Confirm the canonical point remains.
- Confirm duplicate IDs are gone.
- Confirm metadata/audit trail records the report/proposal and consolidated source IDs when available.

### For `promote_to_skill`

- Confirm a local draft artifact exists under:

```text
$HERMES_HOME/qdrant_memory/consolidation/skill_drafts/
```

- Confirm the skill was not installed automatically.
- Review/edit/install the skill through the normal Hermes skill workflow only after human/operator approval.

### For `draft_review`

- Confirm a local markdown draft exists under:

```text
$HERMES_HOME/qdrant_memory/consolidation/reconsolidation_drafts/
```

- Confirm no Qdrant facts were rewritten.
- Treat the draft as review material only.

### For all live actions

Check application audit records under:

```text
$HERMES_HOME/qdrant_memory/consolidation/applications/
```

Then optionally re-run consolidation to confirm the proposal set changed.

---

## 10. Gateway/process restart

Restart the Hermes gateway/process after:

- plugin install or update;
- config changes that affect provider discovery or memory provider selection;
- provider code changes;
- switching plugin path/symlink layout.

Use the deployment's normal supervisor. Examples may include a Hermes gateway command, a systemd user service restart, or restarting the active CLI session. This repository does not assume one universal restart command.

After restart, run `qdrant_memory_status` before any indexing or apply operation.

Post-restart minimum checks:

1. `qdrant_ok: true`
2. `embedding_ok: true`
3. active provider is Qdrant memory
4. known harmless marker search works, if a marker exists
5. watcher force-run only if validating report/watcher behavior

---

## 11. Troubleshooting

| Symptom | Checks / response |
|---|---|
| `qdrant_ok=false` | Check Qdrant service/container, `qdrant_memory.qdrant_url`, network/firewall, and API key config if using a hosted endpoint. |
| `embedding_ok=false` | Check embedding server, `embedding_url`, `embedding_model`, and whether `/v1/embeddings` is OpenAI-compatible. |
| vector size errors | Confirm embedding model dimension matches the Qdrant collection vector size; use a new collection or reindex after model changes. |
| `collection_exists=false` | Collection may not be initialized yet; run status/store/index flow after confirming config. |
| unexpected `point_count` | Check collection name, scope/profile, dry-run indexing output, stale IDs, deleted file IDs, and whether tests used another collection. |
| irrelevant/stale search results | Treat semantic retrieval as non-authoritative; use LCM for exact active-session detail; verify against files/tools; tune thresholds or dry-run reindex a narrow source path. See [LCM_BOUNDARY.md](LCM_BOUNDARY.md). |
| dry-run apply shows stale/missing points | Regenerate the report; do not live-apply stale proposals. |
| action mismatch | Use only the action matching the proposal type; do not force mismatched actions. |
| `quality_warning` proposals | Manual review only; never automatic apply. |
| `reconsolidation_candidate` proposals | Use `draft_review` only; no automatic rewrite. |
| watcher too noisy | Check watcher signature state, `include_examples=false`, `max_groups`, `max_points`, and severity filtering. |
| local artifact contains unexpected sensitive material | Stop live operations, inspect/redact/remove local artifact according to local policy, and do not apply the related proposal. |

---

## 12. Never automate these

Do not automate:

- `qdrant_memory_consolidation_apply`;
- reconsolidation fact rewrite;
- quality-warning resolution;
- broad live indexing;
- query-based deletion;
- learning approval;
- active skill installation from promotion candidates.

Automation may report. It must not decide.
