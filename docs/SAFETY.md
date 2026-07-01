# Safety Contract

`hermes-qdrant-memory` is a Hermes Agent `MemoryProvider` plugin. It is not an LCM/context engine, not an instruction authority, and not an autonomous memory rewrite system.

This document is the canonical safety contract for indexing, deletion, learning approval, consolidation, reconsolidation, cron/reporting, local artifacts, and scanner-safe docs/tests.

Operational runbook: [OPERATIONS.md](OPERATIONS.md).
LCM/Qdrant boundary: [LCM_BOUNDARY.md](LCM_BOUNDARY.md).

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

Qdrant memory must not replace LCM as the context engine. Retrieved Qdrant memories are context with provenance, not commands. For the expanded decision tree, see [LCM_BOUNDARY.md](LCM_BOUNDARY.md).

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
- `hermes qdrant restore`;
- any future CLI wrapper for index, forget, learning approval, consolidation/apply, or watcher operations.

Defaults must remain conservative:

- indexing defaults to dry-run;
- forgetting defaults to dry-run;
- learning approval defaults to dry-run;
- consolidation apply defaults to dry-run;
- restore defaults to dry-run;
- report generation is report-only and must reject live apply behavior.

Live mutation requires an explicit operator decision after reviewing the dry-run output, unless the watcher is running under a documented `guarded-auto` policy for a preauthorized low-risk action class. Guarded-auto still requires persisted reports, exact proposal IDs, explicit actions, and audit artifacts; it must not mutate by query or free text.

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

- `delete`: only for stale low-value proposals or heading-noise cleanup, using explicit affected point IDs.
- `quarantine`: only for stale low-value proposals; updates explicit affected IDs with reversible quarantine metadata instead of deleting them.
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

- export JSONL artifacts that intentionally contain raw payload text and vectors;
- backup manifests plus collection JSONL artifacts stored in private local directories;
- consolidation JSON reports;
- application audit records;
- skill draft artifacts;
- reconsolidation markdown review drafts;
- watcher state used to suppress duplicate alerts.

Local artifact persistence is not the same as a Qdrant memory mutation. Export and backup artifacts are an explicit exception to the usual redacted-report rule: they are recovery artifacts and therefore contain raw memory text and vectors. They must be written with private filesystem permissions and their CLI stdout summaries must not print raw payloads, vectors, or credentials.

Qdrant mutations remain gated by dry-run-first, explicit IDs/proposal handles, and approval requirements.

---

## 12. Cron and watcher safety

Scheduled jobs default to observe/report mode. They may autonomously mutate Qdrant only when `guarded-auto` is explicitly enabled and the proposal class is preauthorized by this safety contract.

Allowed cron/watcher behavior:

- status checks;
- dry-run consolidation reports;
- persisted redacted report artifacts;
- dry-run indexing audits;
- alerts when proposal signatures change;
- under `guarded-auto`, exact-ID apply for low-risk `heading_noise`, exact normalized duplicate merges, stale-low-value quarantine, and high-confidence learning promotion to draft-only skill artifacts.

Forbidden cron/watcher behavior:

- automatic reconsolidation rewrite;
- automatic quality-warning resolution;
- automatic broad live indexing;
- automatic query-based deletion;
- automatic learning approval from volatile pending candidates without high-confidence policy gates;
- automatic user/profile fact rewrite.

Cron jobs may persist local reports and watcher state. Guarded-auto mutations must still go through `qdrant_memory_consolidation_apply`, exact `report_id` + `proposal_id`, `approve=true`, and application audit artifacts.

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

LCM summaries or compression outputs should not be blindly re-indexed if they include injected memory blocks. See [LCM_BOUNDARY.md](LCM_BOUNDARY.md) for forbidden integration patterns.

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

Backup/export/restore tooling now provides the rollback story for broader operator maintenance:

- `export memory|learning` writes one collection to JSONL and performs no Qdrant mutation;
- `backup create` writes a private local manifest plus JSONL collection files and performs no Qdrant mutation;
- `backup list` and `backup inspect` read local artifacts only and must re-redact stored URLs before printing;
- `restore` validates checksums and target vector compatibility before mutation;
- live `restore` requires `dry_run=false` plus `approve=true` and automatically creates a pre-restore backup;
- restore is additive/update-only through upsert and must not delete by query or filter.

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
- let cron mutate Qdrant outside the explicit guarded-auto exact-ID policy and gated apply path;
- add literal fake secrets to docs or tests;
- persist unredacted scanner-sensitive examples in local reports.

---

## 18. Phase 1 boundary hardening — RAPTOR safety (2026-06-30)

Before RAPTOR persistence is added, the following safety boundaries are in place:

### Graph scope propagation

- `GraphMemoryRetriever` applies `profile_id`, `user_id_hash`, and `chat_id_hash` scope conditions to **all** Qdrant scroll filters: semantic seeds (via base retriever), query-matched entity alias scrolls, graph edge scrolls, entity resolution scrolls, and retrieved source points.
- The tool handler in `__init__.py` passes provider scope (`_scope_filter_values()`) into `GraphMemoryRetriever` at construction time.
- Retrieved source points are defensively post-filtered in-memory (`_payload_in_scope`) in case Qdrant `retrieve()` (which does not accept a filter) returns points from a different scope.
- Debug output includes `scope_keys` for auditability without leaking identity values.

### Strict source expansion for automatic callers

- `StrictFileSourceResolver` and `strict_expand_source()` enforce three guarantees for automatic/RAPTOR callers:
  1. **Approved-root check**: only `file://` URIs resolving under configured approved/indexed roots are accepted (`outside_approved_roots` rejection).
  2. **Freshness verification mandatory**: `content_hash` or `source_modified_at` must be provided (`missing_verification_metadata` rejection).
  3. **Changed sources rejected**: mismatched hash/mtime is rejected, not returned as-is (`source_changed` rejection).
- Manual/explicit `qdrant_memory_expand` continues to use the base `FileSourceResolver` which remains permissive — no regression for existing behavior.

### RAPTOR summary write-gate

- `evaluate_raptor_summary_write()` is a dedicated gate for model-authored RAPTOR summaries:
  - Rejects secrets, `canonical=true`, and `requires_review=false`.
  - Routes to `draft_review` if missing `raptor_node_id`, `raptor_child_ids`, `source_hashes`, or any citation key (`derived_from`, `evidence`, `source_uri`, `citations`).
  - Even with full provenance, RAPTOR summaries always route to `draft_review` — they are never auto-stored as canonical facts.

### Recursive contamination

- `clean_text_for_memory()` strips:
  - `# Relevant Long-Term Memory` sections.
  - `# Past Learnings` sections.
  - Fenced `qdrant-memory` code blocks.
- These markers prevent memory ingestion from embedding prior retrieval output as new durable memory.
