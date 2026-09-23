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

Memory PR is an additional read-only boundary, not an apply mechanism. It accepts exact report/proposal IDs, permits only the configured memory or learning collection, retrieves only exact affected point IDs, and fails closed on any ID-set mismatch. A strict bounded schema—not an open-ended alias list—defines review-safe point, evidence, summary, status, locator, derivation, exact-ID link, and ranking shapes. Any unknown key, unsupported type, unmodeled mapping/list, or traversal-bound breach suppresses the entire record before snippets, provenance, status, rendering, or digest construction and forces `unknown` drift. Normalized identity aliases, including SSN, DOB, driver-license, passport, and taxpayer families, are additional defense in depth. Sensitive records retain only exact booleans/fact-status enums and validated timezone-bearing timestamps/derivation types; invalid or free-form state/provenance is omitted. Persisted evidence must be attributable to exact affected IDs, and missing/unknown evidence IDs are rejected. Secret-shaped values remain recursively redacted. Its versioned drift projection excludes access/ranking bookkeeping that ordinary retrieval mutates and requires an actual integer version—JSON booleans are invalid. Its dry-run next step is descriptive data only and is never executed by packet generation. Omitting `output_dir` performs no file write or directory creation/permission change. A pre-existing output directory is never chmodded and must already be current-user-owned mode `0700`.

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

---

## 23. Lineage W0 prerequisites — identity, mechanical gate, containment

Structural lineage groundwork is valid-point-safe. Dense and sparse search,
the semantic-graph entity-scroll path, and consolidation candidate selection
exclude `lineage_record=True` / `lineage_pending=True` server-side and also
post-filter defensively. The semantic graph's direct-ID source-reference path
is narrower: it caps each entity's `source_point_ids` at the first eight
before fetching payloads, then applies structural exclusion defensively. A
structural or pending ID within those eight therefore consumes a reference
slot, and an active ID beyond the cap is not fetched. Two search-path gaps
remain open and are NOT covered by the qualification:

- N1: the RAPTOR direct-ID search and builder paths can still admit
  `lineage_pending` chunks into their own pools.
- N2: `source_extraction.extract_source_candidates_from_point` and
  `improve.extract_improve_candidates_from_point` rebuild candidates from a
  point without preserving pending-state markers. Pending state can therefore
  be lost before the semantic gate sees the candidate; two retained N2 probes
  still fail.

The W0 ordinary-indexer deletion gap is closed in W1 where the backend exposes
`scroll_by_filter`: structural records stay outside stale-chunk deletion, and
off mode blocks lineage-managed chunks instead of deleting or duplicating them.
A no-scroll compatibility backend cannot discover that state; force deletion is
then limited to the explicit legacy fallback and is refused without an exact
user-and-chat scope.

No lineage writer is enabled (`lineage_mode` defaults to `off`).

- Logical `entity-*` / `edge-*` handles are never used as Qdrant point IDs;
  storage uses deterministic UUIDs (`make_graph_point_id`). Improve apply
  refuses stale reports whose stored targets predate the mapping.
- The mechanical lineage gate (`evaluate_mechanical_lineage_write`) is a
  separate function fed by an internal evidence type. It fails closed on
  unsupported operations, claim-level relations, forged endpoint types,
  missing/altered provenance, ownership mismatch, and any validation error;
  import errors propagate instead of being swallowed. `edge_class` alone is
  never authorization. Structural token, hash, UUID-reference, logical-handle,
  locator, URI, and controlled-vocabulary fields are validated by raw type and
  domain whenever their payload key is present, before relation requirements
  and evidence comparison. For those fields, key absence is omission; a
  present empty, null, false, or container value is rejected unless that value
  belongs to the field's stated domain. Builder arguments use explicit
  omission sentinels and omit the corresponding payload key. Trust and state
  booleans are separate: `lineage_pending=False` is valid, `canonical` must be
  `False`, and optional ownership hashes may be empty strings. Structural
  builders preserve the safe raw `profile_id` spelling so they cannot change
  the owner the gate evaluates; non-lineage profile sanitization is unchanged.
  The evidence record is held to the same named token, hash, locator, URI, and
  vocabulary domains, with its documented omission sentinels. All W0 shape
  domains (raw digest, `sha256:`-prefixed content hash, UUID reference token,
  logical handle, controlled vocabulary, file-chunk locator, URI) are defined once in the
  shared validation surface (`graph_schema` `w0_*` functions). The gate and
  the structural builder paths call those same functions. The builders'
  contract is classification, not enforcement: at the serialization boundary
  they classify handles/URIs with the shared domains before any legacy
  normalizer — unsafe values (non-string, secret-bearing) raise, and
  malformed-but-serializable raw strings serialize with their exact spelling
  (never stripped into a storable token), leaving the mechanical gate as the
  enforcing write boundary that refuses them. A `content_hash` on a
  `file_source` record is refused
  outright: the source node designates the exact path, not a content
  snapshot, so no digest on it can be independently evidenced.
- Acceptance boundary: the mechanical gate validates the W0 identity,
  provenance, token, hash, UUID-reference, logical-handle, locator, URI, and
  controlled-vocabulary domains, plus relation requirements and evidence
  binding. It does not validate the generic presentation/scoring fields
  `confidence`, `truth_confidence`, `usefulness_weight`, `description`, or
  `file_size`. Those remain the domain of the public serializers
  (`build_entity_payload`, `GraphEntity.to_payload`, `build_edge_payload`, and
  `GraphEdge.to_payload`). A caller that assembles raw payload dicts
  and skips those serializers must not rely on the gate for those fields. A
  `store` decision must not be read as a claim that every payload argument is
  well formed.
- Generic-metadata enforcement at the final gate is explicitly deferred, not
  fixed. The reviewed sites are `qdrant_memory/graph_schema.py:1000-1011`,
  `:1169-1173`, `:1404-1408`, `qdrant_memory/lineage.py:546-846`, and
  `qdrant_memory/write_gate.py:555-563`, `:679-685`. Closing that boundary
  requires a separately authorized wave; this deferral does not authorize a
  live writer.
- The gate binds the exact write-target collection: callers pass the
  collection they will upsert into, and it must equal the evidence's
  collection. The payload's scope digest is recomputed from that collection
  plus the evidence ownership tuple (never taken from a claimed digest),
  and the source key is recomputed from the recomputed scope plus the
  evidence file path. The final payload's `lineage_operation` must equal the
  authorized operation.
- Structural identity is verified by full-digest recomputation from the
  evidence's own binding material, not by the 16-hex handle truncation
  alone: a same-prefix/different-suffix full identity is refused.
  Compare-before-overwrite against an already-stored record's digest stays
  a persistence-time duty for the W1 writer.
- `SUPERSEDES` runs only between two distinct content-bearing file versions
  (endpoint content hashes present, valid, and different). Its predecessor
  and current `lineage_event_id` are distinct exact UUID reference tokens, and
  the current token must equal the evidence token. Every `approved_citation`
  operation requires a non-empty approval reference supplied by the evidence,
  including `DERIVED_FROM`. W0 does not look up event or approval records and
  does not observe persistence existence, transition commit state, or approval
  state; a future caller must do that before invoking the gate.
- `file_version_id` and `lineage_event_id` are exact UUID reference tokens on
  mechanical edge records where they are valid fields. `file_version_id` is
  bound to the deterministic file-version endpoint UUID, while
  `lineage_event_id` is bound to the evidence token. Neither token is looked
  up by W0. Logical handles live only in the explicitly named `*entity_id`
  reference fields. Entity records (file_source / file_version) do not
  interpret `lineage_event_id` at all and refuse it whenever the key is
  present, whatever the value.
- `extra` cannot forge lineage fields or the ownership keys
  (`user_id_hash` / `chat_id_hash`): the reserved-key wall covers every
  structural field at the builder boundary. That wall is a reserved-extra-key
  rule, not a complete write boundary: builders construct payloads but are
  not writers, final payloads can be constructed directly, and the mechanical
  gate therefore validates the final payload itself (shared shape domains)
  instead of trusting where it came from. When `lineage_record` is false,
  the legacy generic sanitizers keep their historical strip/normalize
  behavior for non-lineage callers.
- The extraction gate also refuses semantic candidates carrying a bare
  `lineage_pending=True` payload flag, so the semantic path cannot create a
  point that containment would permanently hide.
- Structural records keep `canonical=False` and `requires_review=True`; a
  store decision authorizes persistence, not assertion use.
- Dense and sparse search, semantic-graph entity scroll, and consolidation
  candidate selection exclude `lineage_record=True` /
  `lineage_pending=True` before their candidate budgets and defensively after
  retrieval. The semantic graph's direct-ID source-reference path instead
  caps `source_point_ids` at eight before payload fetch and only then applies
  the defensive structural filter, so excluded IDs can consume those eight
  slots. Consolidation apply refuses proposals selecting structural records.

---

## 24. Lineage W1 additive capture limits

W1 remains disabled by default: `lineage_mode="off"`. Capture adds deterministic
file-source, file-version, `PART_OF`, and `DERIVED_FROM` records. It does not
retire or delete lineage state.

- Where the backend exposes `scroll_by_filter`, off mode reads every chunk for
  the selected path without a scope filter and fails closed when any is
  lineage-managed or when that inventory read fails. It performs no delete,
  embedding, or legacy-ID upsert for that file, including when the caller is
  unscoped or has a different profile or user/chat hashes. Read failures retain
  the transport error and use the distinct
  `lineage_inventory_unavailable_refused_fail_closed` refusal. Deletion
  inventory remains exact-owner-only. Foreign-scope reports expose only a count
  for paths within the requested roots, never point IDs. The report discloses
  foreign paths under those roots, including paths no longer on disk, not only
  counts; no tool consumes a reported path for deletion. The bundled client
  pages `next_page_offset` until completion, and these inventory reads do not
  set a `max_total` cap. On a no-scroll compatibility backend, this protection
  cannot inspect lineage state; broad force deletion is refused unless user and
  chat scope are both present.
- `lineage_repair_ids` names structural writes and chunk payload patches issued
  by the current apply. "Repair" covers the fixed read-back invariant key set,
  not full payload reconciliation; drift in fields outside that set can remain
  unchanged and unreported.
- `_owned_file_chunks` normalizes a stored null ownership hash to the empty
  string before exact-owner comparison. The later capture-plan scope check also
  rejects null using its raw comparison, so that second capture-side check is a
  redundant fail-closed guard rather than a separate acceptance path.
- Intentional lineage or capability refusal is reported as `refused` plus
  structured `refusals`; it also sets `partial_failure` because requested work
  was not applied. Transport and write errors set `partial_failure` without
  `refused`, so callers can distinguish the cause. Successful summaries keep
  `partial_failure=false`.
- An interrupted multi-chunk capture may resume only when the durable source head
  and version match, every surviving owned chunk belongs to the deterministic
  expected ID set, and no unexpected owned chunk exists. Other partial states
  stay blocked as `incomplete_capture`; nothing is retired.
- External deletion of a lineage-managed chunk can leave a legacy edge pointing
  at a missing target. W1 does not delete or retire that edge. Reconciliation and
  retirement remain W2 inputs. A non-lineage legacy chunk at the same path and
  deterministic legacy point ID is overwritten regardless of scope; this is
  unchanged W0 behavior.
- The W0 RAPTOR direct-ID/pending-state gap (N1) and extraction/improve
  pending-marker loss (N2) remain open. W1 capture does not change those paths.
- Boolean argument coercion outside the index tool remains out of scope. In
  particular, `_tool_forget` treats an empty-string `dry_run` value as false and
  can perform a live delete. Callers must omit the argument or send an actual
  boolean until a separate bounded fix closes that interface.

---

## 25. Lineage W2 — the dependency fence is profile-scoped

The destructive-operation dependency fence (`find_direct_dependents`, used by the
consolidation apply fence, `qdrant_memory_forget`, and the reconcile lineage-impact
snapshot) matches dependents inside a single scope: `profile_id`, `user_id_hash`
and `chat_id_hash` must equal the caller's. A dependency edge whose `profile_id`
belongs to a **different profile** is therefore invisible to the fence.

This boundary is a recorded decision, not an accident:

- A dependent that lives in another profile does **not** block a forget or a
  destructive consolidation. The named target is deleted and the out-of-profile
  edge is left pointing at a missing target.
- The boundary already applied to the consolidation fence, and
  `qdrant_memory_forget` inherits it because it reuses the same lookup.
- Widening the fence would mean reading and reasoning about records outside the
  caller's profile, so no broadening is done.

Operators sharing one collection across profiles must treat a cross-profile
dependency as unenforced: the fence is a per-profile safety check, not a
collection-wide referential-integrity guarantee.

---

## 26. Lineage W2 — one lock-refusal contract on every tool surface

Failing to take the lineage collection lock is one condition, so it reads as one
string: `lineage collection lock unavailable`. The constant, the refusal exception
(`LineageLockUnavailable`) and the redactor live next to the lock helper in
`qdrant_memory/lineage.py` as the single source.

- The helper normalizes its own OS failures: the lock *directory* check (including
  its symlink test, whose `lstat` re-raises `EACCES` on an unsearchable parent
  component and `ENAMETOOLONG` for an over-long one) and the lock *file*
  (`open`/`fdopen`/`fstat`/`flock`) both report a refusal marker instead of letting a
  bare `OSError` or `ValueError` out (`lineage lock directory unavailable`,
  `lineage lock file unavailable`). The `ValueError` half is load-bearing at the
  directory block, where `mkdir`/`stat` raise it for a NUL byte or an unencodable
  surrogate; at the lock-file sites it is carried for symmetry, since such a path
  fails at the directory check first. This matters in both directions: neither an
  `OSError` nor a `ValueError` is a `TimeoutError`/`RuntimeError`, so either would
  bypass the callers' normalization *and* the response redaction at the same time.
  Cleanup (unlock, close) is best-effort on purpose — the kernel releases the flock
  when the file description closes, so a failed unlock must not mask the body's
  exception nor turn a completed write into a refusal.
- The consolidation fence, `qdrant_memory_forget` and the locked-upsert writers
  (extraction approval, improve apply, RAPTOR apply) return that text, with the
  writers keeping only an operation prefix. Each normalizes the acquisition only,
  so a failure raised inside the guarded body is still reported as itself.
- The `qdrant_memory_index` response is redacted at the tool boundary:
  `redact_lock_refusals` truncates a refusal clause at its marker and appends the
  fixed text, preserving the operation prefix (`lineage capture failed: lineage
  collection lock unavailable`) while dropping the errno and the lock-directory
  path. A marker glued to a non-space character is data (a path or a directory name
  that happens to spell a marker), not a refusal clause, and is left untouched.
- The raw clauses are logged — one `logger.warning` per clause, before the response
  is redacted — so a misconfigured `lineage_lock_dir` stays diagnosable: the server
  log carries the errno and the path even though the tool response does not. The
  per-clause cap is derived from PATH_MAX plus the lock-file suffix plus the longest
  operation prefix, so any path the OS accepts reaches the log whole; an over-long
  *attempted* path (an `ENAMETOOLONG` message is not bounded by PATH_MAX) is cut,
  prefix first.
- `qdrant_memory_consolidation_apply` answers an unusable lock with its own generic
  `consolidation_apply_failed`. It carries no lock detail at all: the same promise
  with a different wording.
- Direct callers keep the raw texts (`lineage collection lock acquisition timed
  out`, `lineage lock directory unavailable: …`), because the frozen W1 tests pin
  those messages on the helper itself. Redaction is a response-boundary concern,
  not a change to the helper's contract.
- Out of scope and unchanged: the `backup` restore path is reached from the CLI,
  not from a tool response, and keeps its own raw text. Runtime activation,
  migration and backfill remain out of scope (see the closing notes for this
  delta).

---

## 27. Lineage W2 — refusal-only ordinary-root retirement (activation statement)

W2 ships as a **refusal-only activation mode** for ordinary roots. The plan
leaves two branches open: ratify the source-neutral ordinary-root transition
contract and prove demotion before retirement, or refuse every unsupported
destructive route permanently. This build takes the second branch. The refusal
is the safety property, not a defect: it stops a destructive route from
degrading into a legacy delete that would leave dependent memories citing
removed evidence, and it is the reason a retired root can always be explained.

### What is supported

- **File-root transitions** (`reconcile`): a changed, deleted or re-chunked file
  goes through the reviewed plan — typed file event, observed baseline,
  immutable replacement chunk IDs, bounded dependent invalidation, read-back
  before commit. Retirement of the replaced chunks happens inside that plan.
- **Dependent blocking** (`off` and `capture`): a destructive consolidation or a
  forget whose target has in-profile dependents is refused (`lineage dependents
  block destructive consolidation`, `lineage dependents block forget`), and so
  is an incomplete or failed dependency lookup.
- **Reindex retirement**: under `capture` a reindex never retires; a file whose
  content changed is reported with `retirement_requires_reconcile`. Under
  `off`, files that are already lineage-managed are refused with
  `lineage_managed_requires_capture_or_retirement` instead of being replaced. A
  reindex replaces every chunk payload in place at its content-derived id and, under
  `--force`, deletes the ids it is about to rewrite, so "managed" here is the shared
  **overwrite** predicate (identity **or** the review state a transition wrote) and
  not a subset of the identity fields: a demoted chunk is blocked in every mode, with
  or without `--force`, and so are the chunks of a file that was removed from disk.
- **Restore**: a backup restore that would replace content carrying lineage identity
  **or** the review state a transition wrote is refused (`restore would overwrite
  lineage-managed content; run explicit reconciliation`), and so is a restore that
  would resurrect content whose lineage records are already retired (`restore would
  resurrect retired lineage-managed content; run explicit reconciliation`). Restore
  reads the shared in-place predicate plus the shared structural test
  (`lineage_record` / `lineage_pending`), in both the memory and the learnings scope.
- **Store and extraction-approval upserts**: both routes target an id derived
  from the content, so re-storing the same text, or approving a regenerated
  candidate, can land on a point that already exists. When that point carries
  lineage identity or the W2 review state, the write is refused — the retirement
  text in the first case, the review-state text in the second — instead of
  replacing the payload. The turn-sync hook skips such a turn and logs at debug
  level. Both routes refuse before writing but **without** the collection lock (the
  store takes none; extraction reads, then locks for its upsert, the same shape as
  improve apply), so the window between read and write is declared rather than
  closed. The store's unlocked upsert is also the one that can abort a concurrent
  transition at that transition's read-back: the failure is fail-closed and surfaced,
  not silent, but it is a second consequence of the same window. Consolidation's
  managed check runs before it takes the lock and is not repeated on the guarded-auto
  re-retrieval, so no test pins the ordering of those two. Neither guard is consulted
  by its preview path: a dry-run store or a review-only approval can still report that
  it would write.
- **Ordinary points without dependents**: `qdrant_memory_forget` still retires a
  plain memory point that carries no lineage identity in every mode, so the
  refusal is not a blanket ban on deletion. Destructive consolidation does not:
  under `reconcile` it refuses an ordinary root by impact validation, as the
  route table records.

### What is not supported

Retirement of an **ordinary root** — a memory point that is not a file chunk —
has no ratified transition or cause representation yet. Two consequences:

- Under `reconcile`, every consolidation delete/merge/quarantine on an ordinary
  root returns `lineage impact blocks ordinary-root transition:
  ordinary_root_transition_cause_unratified`. There is no operator override, and
  the refusal never falls back to the legacy delete path.
- A point that belongs to a captured file carries lineage **identity** while
  staying an ordinary content point. Which fields protect it is decided **per
  effect class**, because retiring a point and replacing one in place destroy
  different things:
  - **Retirement** (`qdrant_memory_forget`, destructive consolidation) removes the
    point, so it reads the identity binding alone: `file_version_id`,
    `lineage_entity_id`, `lineage_schema_version`, `lineage_scope_key`,
    `lineage_source_key`, `lineage_role`. It refuses with `lineage-managed points
    require the reviewed retirement path` in `off`, `capture` and `reconcile`. The
    `off` refusal is deliberate: turning the mode off, or downgrading, must not turn
    an unsupported retirement into a silent delete. Sequence note: that refusal is
    decided from the payloads, so `forget` answers it after the dependency fence and
    consolidation before it; for a managed point that also has dependents the
    canonical text therefore depends on the route, and both are refusals.
  - **In-place overwrite** (the store upsert at the content-derived id, the
    extraction-approval upsert at the candidate id) keeps the id and replaces the
    payload, so it protects the demotion marker `lineage_review_event_ids` as well,
    refusing with `lineage review state requires the reviewed transition path`. The
    harm there is not bookkeeping: the replacement erases `requires_review`,
    `fact_status` and the recorded causes, and `requires_review` is the gate the
    retriever and auto-recall read — a fact a transition flagged as
    stale-pending-review would be served as an active one.
  So the marker's membership is effect-dependent by construction: excluding it from
  retirement is what keeps a demoted dependent retirable (it has no reviewed path
  otherwise), and including it in overwrite is what keeps its review state from
  being erased. Both texts are refusals; neither falls back to a legacy write.
- Retiring a demoted dependent is allowed and is not free, and the remedy is in the
  operator's hands. The dependency edge the graph layer wrote survives with its source
  gone, so a fence that reads it answers `lineage dependency fence is incomplete` in
  all three modes; `qdrant_memory_forget` retires that edge like any ordinary point, and
  doing so unblocks the root in all three modes. A file root's live chunks are
  unaffected. The failure is fail-closed and non-destructive, and pre-existing for
  dependents a transition never demoted; treating the marker as identity would have
  made it permanent for every demoted point too.

Recorded support limitation, in the terms the plan asks for: **complete planning
plus zero root retirement or overwrite on every unsupported route, with the
limitation reported rather than hidden.** No ordinary-root retirement proof is
claimed, because none exists yet. To retire such a point today, change the file it
belongs to and let the file transition run under `reconcile`, or wait for the
ordinary-root transition contract. Stripping the lineage fields from a payload to
make a delete succeed is not a supported path.

### Route table

Verdict every destructive route must give, pinned by
`tests/test_consolidation_apply.py` (retirement route oracle) and
`tests/test_lineage.py` (reindex rows):

- `forget`, ordinary root without dependents — `off` delete, `capture` delete,
  `reconcile` delete.
- `forget`, ordinary root with dependents — refuse in all three modes.
- `forget`, structural record (truthy `lineage_record` / `lineage_pending`) —
  refuse in all three modes.
- `forget`, lineage-managed point — refuse in all three modes.
- `forget`, demoted ordinary dependent (history marker only, no identity field) —
  delete in all three modes; the history marker is not identity. The consequence is
  recorded above: the dependency edge the transition wrote survives with its source
  gone, so the next fence on the cited root answers `lineage dependency fence is
  incomplete` in all three modes. That edge is a **structural** record (mechanical
  edge, `lineage_record=True`), so `forget` cannot retire it — asking it to answers
  `refusing to touch structural lineage records` — and the unblocking path is a
  reviewed transition that marks it `lineage_retired`, which the fence skips. A
  hand-built edge without the structural marker would be deletable, but no writer in
  this tree produces that shape for a point-cited dependency: the only writer of
  `target_point_id` edges is the mechanical builder, which always sets the marker. The
  live collection was measured while writing this section — 15,387 points and zero
  carrying `target_point_id`, `lineage_record`, `file_version_id`, `lineage_entity_id`,
  `lineage_role`, `lineage_scope_key`, `lineage_source_key` or
  `lineage_review_event_ids`. That is an observation about the deployed collection at
  that moment, not part of the contract above; the contract is pinned by test either way.
  A later read-only `count` on 2026-09-23 reported 15,395 points and zero on all eight
  fields, which is the whole point of the sentence: the number moves with ordinary use
  and none of the lineage fields move with it.
- `consolidation delete/merge/quarantine`, lineage-managed point — refuse in all
  three modes.
- `consolidation delete/merge/quarantine`, ordinary root with dependents —
  refuse under `off` and `capture` (fence); under `reconcile` the impact
  validation refuses first.
- `consolidation delete/merge/quarantine`, ordinary root under `reconcile` —
  refuse with the unratified-cause text.
- store upsert, existing point at the content-derived id is lineage-managed —
  refuse in all three modes with the retirement text. Ordinary point — overwrite,
  unchanged. Demoted dependent carrying the review state — refuse in all three
  modes with the review-state text.
- extraction-approval upsert, existing point at the candidate id is
  lineage-managed or carries the review state — refuse in all three modes, with the
  text matching the case.
- either overwrite route, target unreadable (Qdrant retrieval error) — refuse; the
  store reports the failure, the approval answers `Unable to verify target point
  identity`.
- restore, replacement of a live payload carrying lineage identity, the review state,
  or a structural marker (`lineage_record` / `lineage_pending`) — refuse in all three
  modes, in **both** scopes, pinned over the whole predicate.
- restore, resurrection of retired lineage-managed content — refuse in all three
  modes, pinned by `tests/test_backup_cli.py::test_restore_refuses_missing_chunk_retired_by_committed_event`.
- reindex under `off`, a file whose chunk carries lineage identity **or** the
  review state a transition wrote — the file is blocked
  (`lineage_managed_requires_capture_or_retirement`), so neither the payload rewrite
  nor the stale-id deletion runs, with or without `--force`. Decided **per file**, the
  same way the present-file site decides: a removed file whose chunks are gone from
  disk is blocked whole when any one of its chunks is protected, so a demoted sibling
  cannot be retired through the file it belongs to, and the path is not reported as
  deleted. Decided over **the same population** as the present-file site too: every
  scope plus the points whose ownership the filter cannot attribute
  (`profile_id=None`), not just the chunks this indexer owns. The two branches of this
  block asking the same question over different populations was the last way they
  disagreed: a removed path whose only protected chunk was foreign-scope was not
  blocked, and its owned chunks were deleted live. The deletion set itself stays
  owned-only, so the widening can only add refusals. Pinned by `tests/test_lineage.py` (`test_off_mode_refuses_to_rewrite_a_chunk_carrying_only_review_state`,
  `..._one_identity_field`, `..._destroy_or_duplicate_captured_file`,
  `test_directory_off_mode_does_not_delete_removed_lineage_managed_chunks`,
  `test_a_removed_file_whose_siblings_are_ordinary_is_blocked_as_a_file`,
  `test_a_removed_file_with_one_protected_sibling_survives_force_too`,
  `test_a_removed_file_whose_only_protected_chunk_is_foreign_scope_is_blocked`,
  `test_a_present_file_whose_only_protected_chunk_is_foreign_scope_is_blocked_too`,
  `test_off_mode_force_skips_lineage_managed_filter_delete_in_dry_and_live_runs`).
- provider hook `sync_turn` (`sync_turns` on), target at the store's deterministic id
  is protected — the write is suppressed by the same guard the store uses, and the
  failure is logged at debug rather than surfaced, since the runtime calls it with no
  caller to answer. The pin asserts the refusal, not only the silence:
  `tests/test_consolidation_apply.py::test_the_sync_turn_hook_cannot_overwrite_a_protected_target`
  requires `ManagedOverwriteRefused` from the writer and requires the hook to have logged
  that same exception, because "no write happened" is also what a hook that declined for
  an unrelated reason looks like.
- improve apply (not an exact replay) and RAPTOR apply (differing node metadata) —
  refuse by their own checks; read, not executed, in the W2 closure reviews, and
  declared `uncovered` in `tests/test_route_inventory.py`.
- reindex retirement — not reachable: `retirement_requires_reconcile` under
  `capture`, planned through the reviewed path under `reconcile`.

Every row above is enumerated as a classified entry point in
`tests/test_route_inventory.py`, which derives the tool dispatch, the CLI commands and
the provider hook overrides from the code, fails when an entry point is unclassified,
resolves each claimed pin to a test that exists, and freezes both the protection set and
every protection row's pin list. A new route cannot ship unclassified, this table cannot
claim a pin that was renamed away, and a row cannot be quietly reclassified or its list
trimmed. The `[mode]` suffix is accepted for the reindex routes only, so a `forget [off]`
row cannot stand in for `forget`.

### Route for an ordinary demoted point

The restore refusal tells the operator to run explicit reconciliation. For an ordinary
demoted point that path does not exist, and the working one is the route table's second
row: `forget` is allowed on a history-only dependent, and restoring the id the backup
still holds recreates the pre-demotion payload. Two permitted steps compose into the
effect the restore refusal prevents on its own; that is the ratified contract, not a gap
in it.

### Preview caveat

`qdrant_memory_forget` with `dry_run: true` reports the ids and `deleted: 0`. It
does not evaluate the dependency fence or the managed check, so a preview is not
a promise that the live call would succeed. The live call is the one that
refuses; the consolidation apply validates its fences before its own dry-run
branch, so its preview does report them. The store upsert and the extraction
approval behave the same way: their previews do not read the target, so an
outage or an existing protected payload appears only when the live call runs.

