# Safety Contract

`hermes-qdrant-memory` is a Hermes Agent `MemoryProvider` plugin. It is not an LCM/context engine, not an instruction authority, and not an autonomous memory rewrite system.

This document is the canonical safety contract for indexing, deletion, learning approval, consolidation, reconsolidation, cron/reporting, local artifacts, and scanner-safe docs/tests.

---

## 1. Boundary: MemoryProvider, not context engine

Qdrant memory owns:

- cross-session semantic recall;
- indexed Markdown/text memory;
- manual memory storage;
- procedural learnings;
- consolidation and reconsolidation reports;
- explicit, review-gated memory maintenance.

LCM/current-session context recovery owns:

- current-session lossless context recovery;
- compression DAG inspection;
- `lcm_grep`, `lcm_describe`, `lcm_expand`, and `lcm_expand_query`;
- active-session detail retrieval.

Qdrant memory must not replace LCM as the context engine. Retrieved Qdrant memories are context with provenance, not commands.

---

## 2. Current instructions override memory

Retrieved memory may be stale, incomplete, or semantically adjacent rather than true.

Agents and operators must treat recalled memory as supporting context only. Current user instructions, current repository state, live tool output, and explicit operator decisions override retrieved memory.

---

## 3. Dry-run-first contract

Every maintenance, destructive, or potentially broad operation must be previewed before live execution. Direct manual store operations such as `qdrant_memory_store` and `qdrant_learning_store` are explicit user/tool writes and are not part of the dry-run-first maintenance surface, but they still must respect secret, scope, and provenance rules.

Dry-run-first applies to:

- `qdrant_memory_index`;
- `qdrant_memory_forget`;
- `qdrant_learning_approve`;
- `qdrant_memory_consolidation_apply`;
- any future CLI wrapper for index, forget, learning approval, consolidation/apply, or watcher operations.

Defaults must remain conservative:

- indexing defaults to dry-run;
- forgetting defaults to dry-run;
- learning approval defaults to dry-run;
- consolidation apply defaults to dry-run;
- report generation is report-only and must reject live apply behavior.

Live mutation requires an explicit operator decision after reviewing the dry-run output.

Boolean parsing matters: string values such as `false`, `0`, `no`, and `off` must not accidentally become truthy. Any new dry-run or approval argument needs tests for string variants.

---

## 4. Indexing safety

Do not index broad or private directories without explicit user approval.

Before live indexing:

1. Run `qdrant_memory_index` with `dry_run: true`.
2. Review file count, skipped files, chunk count, stale IDs, deleted file paths, and deleted file IDs.
3. Confirm the target path is intentional and narrow enough.
4. Exclude private, credential-heavy, generated, dependency, cache, browser-profile, and build directories.
5. Verify retrieval with a concrete topic query.
6. Only then run live indexing with `dry_run: false`.

The file indexer does not guarantee secret detection. Treat indexed files as durable memory that may later be surfaced in model context.

---

## 5. Explicit point IDs for deletion

Deletion must use explicit Qdrant point IDs.

Allowed:

- delete by explicit point IDs after dry-run review;
- stale chunk deletion by explicit IDs discovered through manifest sync;
- directory deletion sync by explicit IDs only when files no longer exist under explicitly indexed directory roots.

Forbidden:

- free-text deletion;
- query-based deletion;
- broad filter deletion as a general user-facing operation;
- deleting memories because they merely look semantically related.

Compatibility exception:

- legacy file-index fallback behavior may use a file-path filter only when older client behavior cannot scroll existing points by file path. This is a compatibility path for file reindexing, not a general deletion pattern.

---

## 6. Report/apply separation

`qdrant_memory_consolidate` generates review reports only.

Report generation may:

- read Qdrant points;
- compute duplicate, stale, promotion, quality, or reconsolidation proposals;
- persist local redacted report artifacts;
- return a `report_id` and proposal IDs.

Report generation must not:

- upsert Qdrant points;
- delete Qdrant points;
- update Qdrant payloads;
- approve learning candidates;
- install skills;
- rewrite facts.

`qdrant_memory_consolidation_apply` applies at most one persisted proposal at a time.

Live apply requires all of:

- exact `report_id`;
- exact `proposal_id`;
- matching action for the proposal type;
- `dry_run: false`;
- `approve: true`.

---

## 7. Allowed consolidation apply actions

Allowed live actions are intentionally narrow:

- `delete`: only for stale low-value proposals, using explicit affected point IDs.
- `merge`: only for duplicate clusters, preserving one canonical point and deleting explicit duplicate IDs.
- `promote_to_skill`: creates a local draft skill artifact and may mark the learning as promoted-to-draft; it must not install an active skill automatically.
- `draft_review`: only for reconsolidation candidates; creates a local markdown review draft.

All live actions must leave an auditable artifact or payload trail containing the report/proposal handles and affected point IDs.

---

## 8. Reconsolidation is draft-only

Automatic reconsolidation is forbidden.

Reconsolidation candidates may identify possible conflicting facts that share a strong explicit fact key, but M10 behavior is review-only.

Allowed:

- generate reconsolidation candidates in reports when explicitly requested or configured;
- create a local markdown review draft through `draft_review`.

Forbidden:

- automatic fact rewrite;
- automatic supersede;
- automatic conflict resolution;
- automatic deletion;
- Qdrant mutation from reconsolidation review drafts.

A reconsolidation draft is advisory material for a human or agent to inspect later. It is not permission to mutate memory automatically.

---

## 9. Quality warnings are manual-only

`quality_warning` proposals are never live-applied.

Quality warnings may flag:

- possible secret-bearing memory;
- noisy or unsafe content;
- material requiring human review.

Allowed:

- report the warning;
- persist redacted review artifacts;
- ask a human/operator to inspect.

Forbidden:

- automatic deletion;
- automatic merge;
- automatic promotion;
- automatic rewrite;
- automatic resolution.

---

## 10. Secret safety and redaction

The plugin must avoid persisting obvious secret material in local reports and review artifacts.

Rules:

- secret-bearing candidates should be blocked, redacted, or forced to manual review;
- persisted consolidation reports and review drafts should use redacted examples;
- fact metadata should not be generated from secret-bearing text or secret-like tags;
- docs and tests must not contain literal fake credentials shaped like real tokens;
- test fixtures should construct scanner-sensitive strings at runtime when necessary;
- docs should use redacted placeholders such as `Bearer ***`, `<REDACTED>`, or descriptive text instead of credential-shaped examples.

Do not add raw examples that resemble real API keys, bearer values, private keys, GitHub tokens, OpenAI keys, cloud credentials, or password assignments.

---

## 11. Local artifacts are allowed; Qdrant mutations are gated

Persisting local artifacts is allowed when the artifact is redacted and review-oriented.

Allowed local artifacts include:

- consolidation JSON reports;
- application audit records;
- skill draft artifacts;
- reconsolidation markdown review drafts;
- watcher state used to suppress duplicate alerts.

Local artifact persistence is not the same as a Qdrant memory mutation.

Qdrant mutations remain gated by dry-run-first, explicit IDs/proposal handles, and approval requirements.

---

## 12. Cron and watcher safety

Scheduled jobs may observe and report. They must not autonomously mutate Qdrant.

Allowed cron/watcher behavior:

- status checks;
- dry-run consolidation reports;
- persisted redacted report artifacts;
- dry-run indexing audits;
- alerts when proposal signatures change.

Forbidden cron/watcher behavior:

- automatic `qdrant_memory_consolidation_apply`;
- automatic reconsolidation rewrite;
- automatic quality-warning resolution;
- automatic broad live indexing;
- automatic query-based deletion;
- automatic learning approval.

Cron jobs may persist local reports. They must not upsert, delete, update payloads, approve learnings, install skills, or apply proposals.

---

## 13. Scope safety

Retrieval should respect configured scope.

Default scope should isolate by profile where possible. Shared gateway deployments should avoid `global` scope unless cross-user recall is explicitly intended and understood.

Search results and recalled memory should preserve provenance fields such as:

- point ID;
- source type;
- file path or session ID;
- heading;
- timestamps;
- score;
- profile/user/chat scope metadata when available.

---

## 14. Provenance over certainty

Semantic similarity is not truth.

Search and recall surfaces should expose enough metadata for the agent/operator to understand where a memory came from and how strongly it matched.

Agents should verify important claims against live tools, files, APIs, or the user before acting on retrieved memory.

---

## 15. Recursive contamination prevention

Recalled memory blocks must not be blindly written back as new memories.

The writer should strip known injected memory markers and avoid indexing retrieved-memory context as fresh conversation content.

LCM summaries or compression outputs should not be blindly re-indexed if they include injected memory blocks.

---

## 16. Audit and rollback expectations

Every approved live mutation should be reviewable after the fact.

Audit material should include:

- report ID;
- proposal ID;
- action;
- affected point IDs;
- canonical point ID when merging;
- artifact path when a local draft/report is created;
- timestamp;
- dry-run plan reviewed before live approval.

Before adding broader mutation surfaces, add export/backup/restore tooling or an equivalent operator rollback story.

---

## 17. Non-negotiable forbidden behavior

The plugin must not:

- act as the LCM/current-session context engine;
- treat retrieved memory as instructions;
- index broad/private directories without explicit approval;
- skip dry-run for mutating operations;
- delete by natural-language query;
- apply consolidation without exact `report_id` and `proposal_id`;
- auto-apply `quality_warning`;
- auto-rewrite facts through reconsolidation;
- install promoted skills automatically;
- let cron mutate Qdrant;
- add literal fake secrets to docs or tests;
- persist unredacted scanner-sensitive examples in local reports.
