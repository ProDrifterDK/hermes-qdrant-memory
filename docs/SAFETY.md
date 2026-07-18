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
- Memory PR JSON and self-contained HTML review artifacts written only to an explicit caller-selected directory;
- watcher state used to suppress duplicate alerts.

Local artifact persistence is not the same as a Qdrant memory mutation. Export and backup artifacts are an explicit exception to the usual redacted-report rule: they are recovery artifacts and therefore contain raw memory text and vectors. They must be written with private filesystem permissions and their CLI stdout summaries must not print raw payloads, vectors, or credentials.

Qdrant mutations remain gated by dry-run-first, explicit IDs/proposal handles, and approval requirements.

Memory PR is an additional read-only boundary, not an apply mechanism. It accepts exact report/proposal IDs, retrieves only exact affected point IDs, fails closed on any ID-set mismatch, recursively redacts secret-shaped values, and suppresses snippets and proposal narrative for identity-bearing points. Persisted evidence must be attributable to exact affected IDs; identity-bearing evidence is replaced recursively as a whole, and missing/unknown evidence IDs are rejected. Its versioned drift projection excludes access/ranking bookkeeping that ordinary retrieval mutates. Its dry-run next step is descriptive data only and is never executed by packet generation. Omitting `output_dir` performs no file write or directory creation/permission change. A pre-existing output directory is never chmodded and must already be current-user-owned mode `0700`.

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

---

## 19. Phase 2 boundary hardening — Sparse / exact retrieval lane (2026-07-01)

A stdlib-only sparse retrieval lane (`qdrant_memory/sparse_search.py`) was added alongside the existing dense lane. It only improves recall for **literal identifiers** (UUIDs, point IDs, `/api/...` routes, dotted/colon symbols, snake_case identifiers, `Error`-class names, issue IDs like `SMDFS-455`, HTTP status codes).

Safety guarantees:

- **Gated on exact-signal patterns**: `has_strong_signal()` only fires the sparse lane for queries that look like literal lookups. Broad natural-language queries stay dense-only and never touch `scroll_by_filter`.
- **Same filter as dense search**: the sparse scroll reuses `_scope_filter()` so `profile_id` / `user_id_hash` / `chat_id_hash` / `source_type` / `tags` / `source` / `file_path` / `project_path` / `since` / `until` / `memory_kind` / `fact_status_exclude` / `stale` / `requires_review` / `canonical` / `include_fact_history` / quarantine are applied uniformly to both lanes.
- **Hard candidate cap**: `sparse_candidate_cap` defaults to `min(256, max(32, search_candidates * 4))` so a manual search cannot scroll unbounded collections.
- **Sparse secrets are rejected at scoring time**: `contains_secret()` is run on every indexed payload field; secret-bearing points receive `score=0.0` with `secret_blocked=True` and are never surfaced.
- **Quarantine marker respected**: payloads with `consolidation_quarantined=True` are dropped before scoring with `quarantined=True` so a reversible quarantine cannot be bypassed by an exact identifier lookup.
- **Degrades when scroll is absent**: if `QdrantClient.scroll_by_filter` is unavailable or raises, the sparse lane returns an empty list and the retriever falls back to dense-only without crashing.
- **Access metadata is updated only for selected chunks**: `update_access_metadata()` continues to be called on the final selected top-k, never on sparse candidates the scorer inspected and rejected.
- **No public API churn**: `qdrant_memory_search` arguments are unchanged; the sparse lane is internal and toggled per retriever via `sparse_enabled=True` (default).

---

## 20. Phase 4 boundary hardening — RAPTOR apply/status (2026-07-01)

Phase 4 closes the build → apply loop for RAPTOR candidate summary trees.
It owns the digest-gated apply path, the audit persistence, and the
read-only status helper. See [RAPTOR.md](RAPTOR.md) for the full public
surface.

### Digest-gated apply path

- `plan_apply(manifest, report_id, build_id, manifest_digest)` recomputes
  the manifest digest, rejects altered manifests, fails closed on
  shape/scope mismatches, and refuses any non-canonical
  `report_id` / `build_id` / `node_id`. It runs the existing
  write-gate before returning its write decisions.
- Live apply requires an exact `report_id` / `build_id` / `manifest_digest`
  triple, prior dry-run review, and explicit `approve=true`.
- The persist step writes a JSON audit record to
  `~/.hermes/qdrant_memory/raptor_applied/<report_id>.json` with exact
  `applied_node_ids`, `applied_at`, profile scope, and an
  `application_id` derived from `(applied_at, report_id, build_id, digest)`.
- Idempotent repeat apply: a persisted apply record matching the
  manifest's exact `(report_id, build_id, manifest_digest,
  expected_node_ids)` short-circuits live apply without re-upserting.
  Records whose `applied_node_ids` do not match the manifest fail closed
  via `RaptorApplyError` so a tampered or stale record cannot silently
  re-mark a manifest as applied.

### Read-only status helper

- `assess_leaf_safety(payload)` and `assess_parent_status(child_payloads)`
  classify leaves and parents conservatively. They never call Qdrant.
- `qdrant_memory_raptor_status` reads the persisted manifest, retrieves
  parent node existence and the actual child leaf payloads, runs
  `assess_parent_status` against the real children, and overlays a
  conservative parent-status override when leaves are missing or
  retrieval fails.
- Live apply refuses to run when the persisted apply record is missing
  required fields or has a record_type other than `raptor_apply`.

### Exact-ID only

- No `delete-by-filter`, no `delete_ids`, no broad `update_payload`, no
  `upsert` of anything outside the candidate node-id set.
- The retrieve-by-id path used by status post-filters by
  `profile_id` / `user_id_hash` / `chat_id_hash` (Qdrant `retrieve()`
  does not accept filters).

### No public churn for `qdrant_memory_search`

- Phase 4 only adds `qdrant_memory_raptor_apply`,
  `qdrant_memory_raptor_status`, and the underlying RAPTOR audit
  directories. Existing tool handlers, CLI commands, and schemas remain
  backward compatible.

---

## 21. Phase 3 boundary hardening — RAPTOR schema + dry-run builder (2026-07-01)

A new `qdrant_memory.raptor` package introduces the RAPTOR schema and a deterministic dry-run builder. Phase 3 **proposes** RAPTOR trees/manifests; it does **not** mutate Qdrant. See [RAPTOR.md](RAPTOR.md) for the full public surface.

### Pure dry-run contract

- The builder accepts plain Python point dicts (`{"id": ..., "payload": {...}}`) only — no Qdrant client, no HTTP, no I/O. An AST check in `tests/test_raptor_builder.py` enforces the absence of `qdrant_client`, `upsert`, `delete_payload`, `delete_filter`, `delete_ids`, `update_payload`, `scroll`, `search`, `retrieve`, `query_points` reachable from the builder module.
- Every manifest pins `dry_run=True` and `mutations_performed=False`. There is no apply/status tooling in Phase 3; Phase 4 will own that surface.

### MVP summaries are extractive

- Each cluster summary text is built from the child leaf snippets — one `- <point_id>: <snippet>` line per leaf, deterministic order, bounded by `summary_max_chars`. No LLM call, no abstractive freeform claims. Root summary enumerates cluster ids + first line of each cluster's extractive summary.

### Skip rules (conservative)

The builder drops, never re-emits, leaves that:

- have a missing or malformed point `id`
- have missing or empty `payload.text` / `payload.lesson`
- have text or payload fields that match `contains_secret()` from `lesson_extractor`
- carry `consolidation_quarantined=True`
- carry `stale=True` or `requires_review=True`
- carry `fact_status` in `{stale, deprecated, superseded, disputed, review_required}`

Skipped leaves are recorded in `manifest.skipped_leaves` with their reason code; they never appear in cluster summaries, source hashes, or any candidate payload.

### Cross-scope isolation

- Different `profile_id` / `user_id_hash` / `chat_id_hash` tuples split into separate RAPTOR trees (separate `tree_id` and `root_id`).
- Within a cluster, scope fields are propagated only when all leaves agree; disagreement yields a `scope_disagreement_across_clusters` warning and the manifest's top-level `scope` stays empty.

### Manifest digest is deterministic

- `compute_manifest_digest()` deliberately excludes volatile timestamps and only hashes structural inputs (`build_id`, `prompt_version`, `tree_id`, `root_id`, `config`, `leaf_count`, `node_count`, `skipped_leaves`, `warnings`, `candidate_node_payloads`). Repeating the build over the same inputs yields a byte-identical JSON manifest.

### Caller-supplied extras are filtered

- Any caller-supplied `extra` payload is filtered through `_safe_extra`, which drops reserved keys (all RAPTOR-owned structural fields, `fact_status`, `requires_review`, `canonical`, `profile_id` / `user_id_hash` / `chat_id_hash`, `schema` / `schema_version` / `version`, `source_uri` / `source_type` / `locator` / `content_hash` / `source_modified_at`, `derived_from`, `evidence`, plus obvious secret-shape names like `authorization`, `api_key`, `bearer`, `password`, `token`, `credential`, `private_key`) and any string value that matches `contains_secret()`. The denylist covers keys that the base payload *omits* on a given call as well as keys it owns, so callers cannot inject status, scope, provenance, schema, or trust fields through `extra`. Secret-shaped values cannot re-enter the candidate payload via the metadata path.

### Secret-shaped point IDs are rejected

- `_is_safe_leaf()` runs `contains_secret()` against the point id itself before accepting a leaf. Token-like ids (e.g. `sk-…`, `ghp_…`, `AKIA…`, `eyJ…`, `-----BEGIN … PRIVATE KEY-----`, basic-auth URLs) are skipped with reason `secret_id_bearing`.
- Skipped leaves whose reason is `secret_id_bearing` carry a stable redacted handle (`redacted:<sha256[:16]>`) in `manifest.skipped_leaves` instead of the raw id. The original secret-shaped id never appears in `raptor_child_ids`, `derived_from.child_node_id`, the extractive summary text, or any other field of the manifest.

### Manifest digest is stable under skipped-leaf reordering

- After the leaf-acceptance pass, the builder sorts `skipped_leaves` by `(point_id, reason)` before computing `manifest_digest`. Reordering the same safe/unsafe input set (including secret-shaped ids) produces identical manifests, identical `manifest_digest`, and identical serialized `skipped_leaves`.

### Required RAPTOR payload fields

Every candidate payload emitted by the builder contains at least:

`raptor_tree_id`, `raptor_node_id`, `raptor_level`, `raptor_parent_ids`,
`raptor_child_ids`, `raptor_cluster_id`, `raptor_summary_of`,
`raptor_root_id`, `raptor_build_id`, `raptor_prompt_version`,
`source_hashes`, `derived_from`, `derivation_type`, `canonical=False`,
`requires_review=True`.

### Review-required status

- Every RAPTOR candidate is `canonical=False` and `requires_review=True` (with `raptor_review_status="review_required"`). The Phase 1 RAPTOR summary write gate in `qdrant_memory.write_gate.evaluate_raptor_summary_write()` continues to enforce this on the apply path (Phase 4).

### No public tool/handler churn

- Phase 3 only adds `qdrant_memory.raptor`. No existing tool handler, CLI command, or schema field is modified. Phase 4 will own any new apply/status tools.

---

## 22. Phase 5 boundary hardening — RAPTOR search/zoom + hybrid retrieve (2026-07-01)

Phase 5 adds the read-only search/zoom + hybrid retrieve path. It owns
`qdrant_memory.raptor.search`, `qdrant_memory.hybrid`, and the
`qdrant_memory_retrieve` Hermes tool. See [RAPTOR.md](RAPTOR.md) for the
full public surface.

### Read-only invariant

- `RaptorSearcher` and `HybridRouter` only call
  `MemoryRetriever.search(..., update_access=False, allow_sparse_scroll=False)`
  and `QdrantClient.retrieve(...)`. They never call `upsert`,
  `delete_ids`, `delete_filter`, `update_payload`, or
  `scroll_by_filter`.
- An AST-based test in `tests/test_raptor_search.py` walks the
  `qdrant_memory.raptor.search` module and fails on any forbidden call.
- `HybridRouter.retrieve` always forwards `update_access=False` and
  `allow_sparse_scroll=False` (phase 5 fix5) to the base retriever, so
  a hybrid call never bumps `last_accessed` / `access_count` access
  metadata AND never invokes `scroll_by_filter` even when the query is
  a strong-signal pattern (UUID, issue id, route path).
- `RaptorSearcher.search` always forwards `update_access=False` AND
  `allow_sparse_scroll=False` (phase 5 fix5) to its underlying
  `MemoryRetriever` so the RAPTOR seed search cannot re-enable the
  scroll-by-filter lane. If a custom retriever lacks the kwarg the
  RAPTOR seed search fails closed (empty seeds + warning) instead of
  silently retrying without the flag.
- The RAPTOR seed-search warning (phase 5 fix7) is **sanitized**:
  neither the `TypeError` (missing-kwarg) arm nor the generic
  `Exception` arm interpolates `str(exc)`. Backend exceptions can
  echo the requested query (which may carry a secret-shaped token)
  or other raw backend strings into `warnings`. A stable
  `debug.stages.seed_search.error` (`type_error` / `exception`) is
  recorded server-side so operators correlate via debug logs
  without leaking the raw exception into the JSON envelope.
- **Graph lane `scroll_by_filter` suppression (phase 5 fix8, final6
  finding #1).** `HybridRouter.retrieve` propagates two read-only
  contract flags into the real `GraphMemoryRetriever.search`:
  `allow_sparse_scroll=False` AND `allow_graph_scroll=False`. The
  graph lane must never invoke `scroll_by_filter` from inside the
  Phase 5 retrieve path — neither via the dense+sparse seed lane
  nor via the BFS entity/edge expansion. When `allow_graph_scroll=False`
  the graph lane short-circuits BEFORE the BFS expansion with an
  empty result + sanitized warning + a `scroll_suppressed=True`
  debug flag. When the wrapped `MemoryRetriever` predates the
  `allow_sparse_scroll` kwarg the graph lane fails closed (empty
  seeds + sanitized warning). Standalone `qdrant_memory_graph_search`
  keeps the default `True/True` so its behaviour is unchanged. An
  end-to-end regression in
  `tests/test_raptor_search.py::TestHybridRouterNoScrollByFilterUnderStrongSignal`
  wires a real `HybridRouter` + real `GraphMemoryRetriever` + real
  `MemoryRetriever` + a strict fake Qdrant sentinel under a
  UUID-shaped query and asserts **zero** `scroll_by_filter` calls
  AND zero `update_payload` calls anywhere in the pipeline.

### Scope isolation (retrieve-by-id has no filter)

- Explicit `retrieve()` calls used by `RaptorSearcher` defensively
  post-filter each returned payload against the configured
  `profile_id` / `user_id_hash` / `chat_id_hash` scope. A payload
  from a different scope is silently dropped before reaching the
  output.
- `HybridRouter.scope` is passed into every lazy-built lane
  (`MemoryRetriever.search`, `GraphMemoryRetriever.search`,
  `RaptorSearcher`).

### Unsafe payloads stay hidden or warning-only

- Every cited leaf is run through `assess_leaf_safety`. Unsafe markers
  (`fact_status in {stale, deprecated, superseded, disputed,
  review_required}`, `consolidation_quarantined`, `stale`,
  `requires_review`, `raptor_excluded`, `raptor_forgotten`, secret-shaped
  text/payload) demote the leaf from `cited_leaves` into a warning
  entry.
- Parent summaries whose own `text` triggers `contains_secret()` are
  skipped entirely and surfaced only through warnings.
- **Dense-lane secret scan (phase 5 fix5).** `_dense_chunk_payload_secret`
  in `qdrant_memory.hybrid.router` scans `chunk.text`, the projected
  payload fields, `chunk.id`, AND the full `chunk.ranking_debug` object
  (including nested dicts and lists). If any field inside
  `ranking_debug` is secret-shaped — for example a non-projected
  payload field like `source_hash_current` that gets reflected into the
  audit envelope via `rank_memory_candidate` — the entire dense hit is
  dropped fail-closed and a redacted warning is emitted. Clean
  `ranking_debug` objects are preserved verbatim on the emitted hit
  so the audit envelope stays useful.

### Redacted warning handles

- Warning strings use the builder's stable redacted handle
  (`redacted:<sha256[:16]>`) for secret-shaped point IDs so neither raw
  IDs nor scanner-shaped literals can appear in JSON output.

### Bounded budgets

- Hard caps live in `qdrant_memory.raptor.search`:
  `HARD_MAX_DEPTH=3`, `HARD_MAX_CHILDREN=16`,
  `HARD_MAX_SOURCE_CHARS=2400`, `HARD_CONTEXT_CHAR_BUDGET=16000`,
  `HARD_SEED_TOP_K=32`. The router mirrors these caps.
- `top_k` is clamped 1..20. Caller-supplied `--max-depth`,
  `--max-children`, `--max-source-chars` are clamped at the tool handler.

### Evidence-mode demotion

- When `mode="evidence"`, RAPTOR parent summaries that have no
  cited leaf (`parent_point_id` referenced by zero leaves) are
  demoted from `summaries` to a warning entry. Parents cannot stand
  alone as authoritative evidence.

### Missing-children parent demotion (phase 5 fix7)

- `RaptorSearcher.search` tracks the per-parent **referenced**
  child count (deduped across `raptor_child_ids` and
  `raptor_summary_of`, capped by `safe_max_children`). When a
  parent references a child that the backend never returns
  (deleted, missing, scope-filtered by `_payload_matches_scope`,
  or dropped by the `retrieve` exception path), the
  `assess_parent_status` recomputation treats the missing child
  as unsafe so the parent never remains `active` while its
  evidence was silently dropped. Warnings cite the redacted
  parent handle only — the raw missing child id is never echoed
  through the JSON envelope.

### Shared-child per-parent accounting (phase 5 fix8, final6 finding #2)

- Pre-fix8, retrieval-pass dedupe used a single global
  `seen_leaf_ids` set combined with `setdefault`-wins attribution
  (`parent_point_for_leaf[child_id] = first_parent_seen`). When a
  child was shared across multiple parents, only the first parent
  counted the shared child in its per-parent referenced set, and
  only the first parent absorbed unsafe/missing accounting. A
  parent whose only child was shared with another parent could
  remain `active` while its evidence was demoted.
- Phase 5 fix8 separates **retrieval dedupe** from **per-parent
  safety accounting**:
  - `parents_for_leaf: dict[child_id, list[RaptorSummaryHit]]`
    tracks every parent that referenced each child.
  - `per_parent_referenced_children: dict[id(parent), set[child_id]]`
    and `per_parent_retrieved_children: dict[id(parent), set[child_id]]`
    track per-parent referenced / retrieved child sets.
  - Each unique child is still retrieved exactly once
    (`seen_leaf_ids`), but unsafe / safe / missing accounting is
    applied to **every** parent in `parents_for_leaf[child_id]`,
    not just the first parent.
  - Missing-count is computed as
    `max(0, len(referenced - retrieved))` against the parent's
    own set, never against the global dedupe set.
- Net effect: a shared unsafe child demotes every parent that
  referenced it (both parents clear text + drop out of `summaries`
  + appear in `unsafe_summary_ids`). A shared missing child does
  the same. A shared safe child keeps every parent active (when
  no other unsafe/missing children exist for those parents).
  Warnings cite the redacted parent handle only — the raw shared
  child id is never echoed through the JSON envelope, even when
  the child id itself is secret-shaped. Regressions in
  `tests/test_raptor_search.py::TestSharedUnsafeChildDemotesEveryParent`
  cover the stale-share, missing-share, safe-share, and the
  three-parent secret-shaped-share paths.

### Fanout cap is not missing evidence (phase 5 fix9, final7 finding #1)

- Pre-fix9, the per-parent referenced set recorded every child in
  `raptor_child_ids` / `raptor_summary_of` BEFORE the
  `safe_max_children` cap check. Children beyond the cap were
  intentionally not retrieved because of the fanout budget, but
  the subsequent `referenced_set - retrieved_set` treated them as
  missing evidence. The parent-status recomputation would inject a
  synthetic unsafe child, clear the parent text, and add the
  parent to `unsafe_summary_ids` — a false demotion of a
  perfectly safe parent.
- Phase 5 fix9 moves the cap check ahead of the referenced-set
  insertion: a child id is added to `referenced_for_parent` only
  when the searcher has fanout budget for it (i.e. `enqueued <
  safe_max_children`) or it was already enqueued by a previous
  parent sharing the cap (`already_enqueued` branch). Children
  that are also shared still count for the parent so the fix8
  shared-child safety path stays intact.
- A genuinely missing child within the cap (`referenced - retrieved`
  is non-empty) still demotes the parent. Children beyond the cap
  are explicitly excluded from `referenced` so they cannot
  contribute to the missing-count.
- Regressions in
  `tests/test_raptor_search.py::TestFanoutCapNotCountedAsMissing`
  cover: parent with >max_children all-safe children stays active
  and cites at most cap leaves; parent with a missing child inside
  the cap demotes; the fix8 shared-child safety path is preserved
  under the new cap-aware accounting.

### Dense exact_hits budgets (phase 5 fix9, final7 finding #2)

- Pre-fix9, `_dense_to_exact_hits` emitted `chunk.text` verbatim
  and the `debug.context_used_chars` counter only summed
  summaries + cited_leaves, so a 5000-char dense hit with
  `max_source_chars=10` would surface verbatim and not count
  against the budget at all.
- Phase 5 fix9 applies per-result truncation to dense exact_hits
  text using the caller-clamped `safe_max_source_chars`. The
  function also accepts a `hard_context_char_budget` kwarg; when
  adding a new hit would push the running total past the budget,
  the overflow hit is dropped and a sanitized warning is emitted
  that carries only the redacted handle. The hybrid router passes
  `_HARD_CONTEXT_CHAR_BUDGET` (16000) and the already-clamped
  `safe_max_source_chars` into the dense lane, and the
  `context_used_chars` debug counter now sums exact_hits text
  length on top of summaries + leaves so the dense lane cannot
  blow the RAPTOR-lane hard cap.
- Regressions in
  `tests/test_hybrid_retrieve.py::TestDenseExactHitsBudgetEnforcement`
  cover: long dense hit (5000 chars) with `max_source_chars=10`
  emits truncated text and `context_used_chars<=10`; many dense
  hits are dropped at the hard context budget; the empty dense
  lane produces `context_used_chars=0` (no regression on the
  no-hit case); per-hit truncation takes effect before the
  context counter runs.

### Learning retrieve sanitized error (phase 5 fix7)

- `_tool_retrieve_learning` no longer interpolates `str(exc)` into
  the JSON error envelope on `LearningStore.search` failure.
  Backend exceptions can echo the requested query (which may carry
  a secret-shaped token) or other raw backend strings into
  `__str__`; the JSON error is replaced with a sanitized message
  (`"Learning retrieve failed (no raw exception leaked; see server
  logs)"`). The raw exception remains available server-side via
  Python logging for operator correlation.

### Global hard context budget across lanes (phase 5 fix10, final8 finding #1)

- Pre-fix10, the dense+sparse lane and the RAPTOR lane each
  clamped their own content to `HARD_CONTEXT_CHAR_BUDGET` (16000)
  independently and the hybrid router only reported
  `debug.context_used_chars`. The union of `summaries` +
  `cited_leaves` + `exact_hits` could therefore exceed 16000
  chars (e.g. 15600 dense exact_hits + 1200 RAPTOR summary =
  16800) and the debug counter was at most additive, not a hard
  cap.
- Phase 5 fix10 introduces `_enforce_global_context_budget` at
  the final packing stage of `HybridRouter.retrieve`. The helper
  enforces ONE hard budget across all three lanes. The
  deterministic policy is **preserve RAPTOR summaries +
  cited_leaves first** (tree evidence is more provenance-
  anchored and harder to reconstruct than the dense lane's
  exact_hits), **then fit dense exact_hits into the remaining
  budget**. Overflow dense exact_hits are dropped first-seen-wins
  with a sanitized warning per drop (redacted handle, no raw
  text/ids). `context_used_chars` is recomputed from the actual
  emitted text so the debug envelope cannot disagree with the
  wire. The hard cap is non-negotiable: the caller's LLM context
  window cannot grow past 16000 chars no matter how many lanes
  fire.
- Regressions in
  `tests/test_hybrid_retrieve.py::TestHybridGlobalContextBudget`
  cover: dense + RAPTOR combined exceeds 16000 → union is capped
  to <=16000 with RAPTOR preserved; the RAPTOR-first policy
  preserves summaries/leaves and drops dense hits to fit; under-
  budget unions are unchanged (no spurious drop warning).

### Learning active-context safety + per-hit cap + hard budget (phase 5 fix10, final8 finding #2)

- Pre-fix10, `_tool_retrieve_learning` (collection=learning)
  secret-scanned hits but did NOT apply the active-context
  status vocabulary the dense memory lane enforces. A learning
  hit with `requires_review=True`, `fact_status=review_required`,
  `stale=True`, `consolidation_quarantined=True`,
  `raptor_excluded=True` / `raptor_forgotten=True`, or unsafe
  `fact_status` values (`stale`, `review_required`, `disputed`,
  `deprecated`, `superseded`) would surface as a normal active
  `results.exact_hit` despite the safety gate. The learning
  path also did NOT enforce `max_source_chars` on a per-hit
  basis, and did NOT enforce a cumulative hard context budget
  across emitted hits. The probe emitted a learning hit with
  `requires_review=true`, `fact_status=review_required`, and a
  5000-char text as a normal active exact_hit despite
  `max_source_chars=10`.
- Phase 5 fix10:
  - Reuses `_dense_payload_unsafe_for_active_context` from the
    hybrid router so the learning path uses the SAME status
    vocabulary as the dense memory lane. Unsafe-status hits are
    demoted to warning-only (no active `results.exact_hits`).
  - The caller-clamped `max_source_chars` (default 1200, hard
    cap 2400) is applied to each learning exact_hit text via
    `_truncate_dense_text` so a 5000-char learning hit is
    truncated to a safe per-hit size.
  - `_enforce_learning_context_budget` enforces a single
    `HARD_CONTEXT_CHAR_BUDGET` (16000) across the union of
    emitted learning exact_hits. Overflow hits are dropped
    first-seen-wins with a sanitized per-hit warning (redacted
    handle, no raw text/ids).
  - `debug.context_used_chars` and `debug.max_source_chars` are
    populated so an operator can correlate via debug without the
    warning channel alone.
  - `include_fact_history` is supported but the learning path
    does not surface a fact history lane; the default active-
    context gate holds. The hook is wired for symmetry with the
    memory lane.
- Regressions in
  `tests/test_learning_retrieve.py::TestLearningActiveContextStatusSafety`,
  `TestLearningMaxSourceCharsEnforcement`,
  `TestLearningHardContextBudgetEnforcement`, and
  `TestLearningWarningNoRawSecretLeak` cover: review-required,
  stale, quarantined, and unsafe `fact_status` learning hits
  are not active exact_hits by default; long learning hits are
  capped to `max_source_chars`; many learning hits cannot
  exceed the hard budget; per-hit drop warnings use the redacted
  handle (no raw ids or secret-shaped text).

### No public churn for `qdrant_memory_search`

- Phase 5 only adds the `qdrant_memory_retrieve` tool, the
  `HybridRouter` / `RaptorSearcher` modules, and the `hermes qdrant
  retrieve` CLI subcommand. The existing `qdrant_memory_search` tool
  schema and behavior are unchanged.
