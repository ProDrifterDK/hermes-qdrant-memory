# Operations Runbook

This runbook describes how to operate `hermes-qdrant-memory` safely in a live Hermes deployment.

Canonical policy: `docs/SAFETY.md`.

Use this document for status checks, smoke tests, watcher runs, consolidation report review, approved apply flow, post-apply verification, gateway/process restarts, and troubleshooting.

---

## 1. Operating boundaries

`hermes-qdrant-memory` is a Hermes `MemoryProvider`, not an LCM/current-session context engine.

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

---

## 6. Watcher force-run

The reference watcher script may be installed outside this repository, usually under `$HOME/.hermes/scripts/`.

Check that the watcher script exists before running it:

```bash
test -x "$HOME/.hermes/scripts/qdrant_sleep_consolidation.py"
```

Force a watcher/report run:

```bash
QDRANT_SLEEP_FORCE_ALERT=1 "$HOME/.hermes/scripts/qdrant_sleep_consolidation.py"
```

Expected behavior:

- it checks/generates consolidation proposals;
- it may persist local redacted report artifacts;
- it may update watcher signature state;
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
| irrelevant/stale search results | Treat semantic retrieval as non-authoritative; verify against files/tools; tune thresholds or dry-run reindex a narrow source path. |
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
