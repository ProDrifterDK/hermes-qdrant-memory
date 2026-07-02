# RAPTOR (Phases 3, 4, and 5)

This document now describes the **complete** RAPTOR surface across Phases 3–5.
Earlier copies described only Phase 3; that file is kept for history under
`docs/RAPTOR.md.bak`, while this revision documents Phase 4 (apply/status) and
Phase 5 (search/zoom + hybrid retrieve) as well.

Phase 3 of the LCM + Qdrant RAPTOR backlog introduces a deterministic dry-run
builder for RAPTOR trees. The builder proposes summary nodes and per-build
manifests **without mutating Qdrant**.

Phase 4 adds the digest-gated apply/status surface:

- `plan_apply(...)` validates a manifest, fails closed on missing fields,
  digest mismatch, scope disagreement, or unsafe payloads, and produces a
  write-decision plan.
- `persist_apply_record(...)` writes a JSON audit record under
  `~/.hermes/qdrant_memory/raptor_applied/` after a successful live apply.
- `assess_leaf_safety(payload)` and `assess_parent_status(child_payloads)`
  classify leaves and parents conservatively.
- The `qdrant_memory_raptor_apply` and `qdrant_memory_raptor_status` Hermes
  tools expose both phases, with `--dry-run` and explicit `report_id` /
  `build_id` / `manifest_digest` arguments.

Phase 5 adds the **read-only** search/zoom + hybrid retrieve surface:

- `qdrant_memory.raptor.search.RaptorSearcher` walks RAPTOR trees via
  dense+sparse seeds and `retrieve()`; it never mutates Qdrant.
- `qdrant_memory.hybrid.router.HybridRouter` combines dense+sparse,
  graph, and RAPTOR lanes into one stable JSON output.
- The `qdrant_memory_retrieve` Hermes tool exposes this path; CLI users
  reach it via `hermes qdrant retrieve ...`.

## Public surface

### Phase 3 — schema + dry-run builder

The RAPTOR module lives at `qdrant_memory.raptor`. The package re-exports:

- `RaptorBuilder`, `build_raptor_dry_run`
- `RaptorNode`, `RaptorTree`, `RaptorCluster`, `RaptorScope`,
  `RaptorBuildManifest`
- `compute_tree_id`, `compute_root_id`, `compute_node_id`,
  `compute_build_id`, `compute_manifest_digest`
- `RAPTOR_DERIVATION_TYPE`, `RAPTOR_LEVEL_LEAF`, `RAPTOR_LEVEL_ROOT`,
  `DEFAULT_PROMPT_VERSION`, `RAPTOR_REQUIRED_NODE_FIELDS`,
  `DEFAULT_MAX_CLUSTER_SIZE`

### Phase 4 — apply/status

From `qdrant_memory.raptor.apply`:

- `plan_apply(manifest, ...)` — digest-gated dry-run plan
- `persist_apply_record(...)` and `load_apply_record(...)` — audit I/O
- `assess_leaf_safety(payload)` and `assess_parent_status(...)`
- `validate_manifest(...)`, `verify_manifest_digest(...)`
- Hermes tools: `qdrant_memory_raptor_apply`,
  `qdrant_memory_raptor_status`

### Phase 5 — search/zoom + hybrid retrieve

From `qdrant_memory.raptor.search`:

- `RaptorSearcher` — read-only search/zoom helper
- `RaptorSearchResult`, `RaptorSummaryHit`, `RaptorLeafHit`
- Hard caps: `HARD_MAX_DEPTH=3`, `HARD_MAX_CHILDREN=16`,
  `HARD_MAX_SOURCE_CHARS=2400`, `HARD_CONTEXT_CHAR_BUDGET=16000`,
  `HARD_SEED_TOP_K=32`

From `qdrant_memory.hybrid`:

- `HybridRouter` — read-only packer over dense+sparse, graph, and RAPTOR
- `HybridRouteResult` — stable output shape
- `rrf_fuse(...)` and `deduplicate_by_point_id(...)`

Hermes tool: `qdrant_memory_retrieve`. CLI: `hermes qdrant retrieve ...`.

## Hard guarantees

### Phase 3

1. **No Qdrant mutation.** The builder accepts plain Python point dicts only
   and never imports `qdrant_client`. There is no `upsert`, `delete_payload`,
   `delete_filter`, `delete_ids`, or `update_payload` call reachable from
   the builder. The manifest pins `dry_run=True` and
   `mutations_performed=False`.
2. **MVP summaries are extractive.** Each cluster summary is built from
   child leaf snippets (one anchor line per leaf, deterministic order).
   No LLM call, no abstractive freeform claims.
3. **Unsafe leaves are skipped.** Leaves are skipped (and recorded in
   `skipped_leaves`) when any of the following holds:
   - missing point `id`
   - missing or empty `payload.text` / `payload.lesson`
   - text or payload contains a known secret shape
   - `payload.consolidation_quarantined=True`
   - `payload.stale=True` or `payload.requires_review=True`
   - `payload.fact_status` in `{stale, deprecated, superseded, disputed,
     review_required}`
4. **Cross-scope leaves never cluster together.** Different
   `profile_id` / `user_id_hash` / `chat_id_hash` tuples split into
   separate RAPTOR trees. Scope fields are propagated to a payload only
   when all leaves in a cluster agree; disagreement emits a warning.
5. **Determinism.** Leaves are sorted by point id before clustering.
   Cluster ids, node ids, tree ids, build ids, and the manifest digest
   are all sha256-based and stable across runs given the same input
   and config. Repeating the build yields byte-identical JSON.
6. **Manifest digest is deterministic.** Volatile timestamps are excluded
   from the digest input set. Callers can inject a deterministic
   `timestamp` via `RaptorBuildManifest(timestamp=...)` if they want a
   fingerprint that includes wall-clock information.
7. **Caller-supplied extras are filtered.** Any `extra` payload field is
   run through the safety filter (`_safe_extra`) so reserved keys
   (`authorization`, `api_key`, `bearer`, `password`, `token`, …) and
   secret-shaped values can never re-enter the candidate payload via
   the metadata path.

### Phase 4 — apply/status

1. **Exact-ID only.** Every candidate node is addressed by its
   `raptor_node_id`. No `delete-by-filter`, no broad update, no
   `delete_ids`.
2. **Digest-gated.** `report_id`, `build_id`, and `manifest_digest` must
   match exactly before any live mutation.
3. **Write-gate validation.** Every candidate payload is validated
   through `evaluate_raptor_summary_write()` before *and* after provider
   enrichment.
4. **Audit trail.** A live application writes a JSON record under
   `~/.hermes/qdrant_memory/raptor_applied/` containing exact
   `report_id`, `manifest_digest`, `applied_node_ids`, and timestamps.
5. **Status is read-only.** `assess_leaf_safety` and
   `assess_parent_status` classify leaves and parents conservatively;
   `qdrant_memory_raptor_status` never mutates Qdrant.

### Phase 5 — search/zoom + hybrid retrieve

1. **Read-only.** `RaptorSearcher` and `HybridRouter` never call
   `upsert`, `delete_ids`, `delete_filter`, `update_payload`, or
   `scroll_by_filter`. They only call `search` (via `MemoryRetriever`,
   always with `update_access=False` AND `allow_sparse_scroll=False`,
   phase 5 fix5) and `retrieve`. `MemoryRetriever.search` defaults
   `allow_sparse_scroll` to `True`; the Phase 5 retrieve path always
   overrides it to `False` so strong-signal queries (UUIDs, issue
   IDs, route paths) cannot reach `scroll_by_filter` through the
   dense lane or the RAPTOR seed search lane. If a custom retriever
   does not accept the `allow_sparse_scroll` kwarg, the RAPTOR seed
   search fails closed (empty seeds + warning) rather than silently
   re-enabling the scroll lane.
2. **Graph lane `scroll_by_filter` is suppressed on the retrieve
   path (phase 5 fix8, final6 finding #1).** `HybridRouter.retrieve`
   also propagates `allow_sparse_scroll=False` AND
   `allow_graph_scroll=False` into the real
   `GraphMemoryRetriever.search`. The graph lane short-circuits
   BEFORE the BFS entity/edge expansion and returns an empty
   result with a sanitized warning + `scroll_suppressed=True`
   debug flag. Standalone `qdrant_memory_graph_search` keeps the
   default `True/True` so its behaviour is unchanged. An
   end-to-end regression with a strict fake Qdrant sentinel under
   a UUID-shaped query asserts **zero** `scroll_by_filter` calls
   AND zero `update_payload` calls anywhere in the pipeline.
3. **Scope isolation.** Explicit `retrieve()` calls (which Qdrant does
   not let filter) defensively post-filter payloads against
   `profile_id` / `user_id_hash` / `chat_id_hash`.
4. **Unsafe payloads stay hidden.** Leaves with secrets, fact-status in
   `{stale, deprecated, superseded, disputed, review_required}`,
   `consolidation_quarantined`, `stale`, `requires_review`,
   `raptor_excluded`, or `raptor_forgotten` are dropped from
   `cited_leaves` and surfaced only as warnings.
5. **Redacted warnings.** Warning strings never echo the raw
   secret-shaped point IDs; they use the builder's
   `redacted:<sha256[:16]>` handle.
6. **Bounded budgets.** `max_depth`, `max_children`, `max_source_chars`,
   and a single `context_char_budget` clamp the result. Top_k clamps
   to 1–20.
7. **Evidence-mode demotion.** When `mode="evidence"`, RAPTOR parents
   without cited leaves are demoted (the parent does not stand alone as
   authoritative evidence).
8. **Per-leaf parent attribution.** Each cited leaf is bound to the
   parent RAPTOR summary that declared it in `raptor_child_ids` /
   `raptor_summary_of` during the BFS walk. When a leaf is shared by
   multiple parents, the first parent encountered wins
   (`setdefault`-based deterministic attribution). Leaves with no
   parent attribution are dropped to `unsafe_leaf_ids` rather than
   mis-attributed to the first summary.
9. **Shared-child per-parent accounting (phase 5 fix8, final6
   finding #2).** Pre-fix8, retrieval-pass dedupe used a single
   global `seen_leaf_ids` set combined with `setdefault`-wins
   attribution, so a parent whose only child was shared with
   another parent could remain `active` while its evidence was
   demoted. fix8 separates **retrieval dedupe** from **per-parent
   safety accounting**: each unique child is still retrieved
   exactly once, but unsafe / safe / missing accounting is applied
   to **every** parent in `parents_for_leaf[child_id]`. A shared
   unsafe child demotes every parent that referenced it (text
   cleared + parent added to `unsafe_summary_ids`). A shared safe
   child keeps every parent active. Warnings cite the redacted
   parent handle only — the raw shared child id is never echoed.
9a. **Fanout cap is not missing evidence (phase 5 fix9, final7
    finding #1).** The per-parent referenced set is built from
    children the searcher actually intended to retrieve — i.e.
    children that fit inside `safe_max_children` (or were already
    enqueued by a previous parent sharing the cap). Children
    beyond the cap are budget-skipped, not missing / deleted /
    scope-filtered evidence. Pre-fix9 every child in
    `raptor_child_ids` / `raptor_summary_of` was added to the
    referenced set BEFORE the cap check, so the subsequent
    `referenced_set - retrieved_set` would falsely inflate the
    missing count and demote a perfectly safe parent. fix9 moves
    the cap check ahead of the referenced-set insertion; children
    that are also shared (already enqueued) still count for the
    parent. A genuinely missing child within the cap still demotes
    the parent.
10. **Collection separation.** `qdrant_memory_retrieve(collection="learning")`
    is wired through `LearningStore.search(..., update_access=False)`,
    not through the memory `HybridRouter` / `MemoryRetriever` /
    `RaptorSearcher` / `GraphMemoryRetriever`. The memory cache is
    never touched on a learning call, and RAPTOR / graph lanes are
    intentionally skipped for the learning collection.
11. **Metadata redaction under `include_metadata=true`.** When a parent
    summary's `derived_from`, `parent_assessment`, or `extra` payload
    carries a secret-shaped value (credential URI, bearer token, etc.),
    `RaptorSummaryHit.to_dict` fails closed: the raw values are
    replaced with empty containers and `parent_assessment.metadata_redacted=True`
   is set. The dense lane applies the same rule to `chunk.text` +
   the projected payload JSON before it surfaces in `exact_hits`.

10. **Pre-promotion core-field secret scan.** Before a `RaptorSummaryHit`
    is constructed, the searcher projects the exact default-emitted
    core fields (`point_id`, `raptor_node_id`, `raptor_root_id`,
    `raptor_tree_id`, `raptor_build_id`, `raptor_cluster_id`,
    `raptor_child_ids`, `raptor_parent_ids`, `raptor_summary_of`,
    `source_hashes`) and runs `contains_secret` over the stable JSON
    blob. If any of those carries a secret-shaped value, the parent
    is demoted to warning-only with a `redacted:<sha256[:16]>`
    handle and never reaches `summaries` or the unsafe-id envelope.
    The no-child downgrade warning uses the same redacted handle.

## Schema fields

Every RAPTOR candidate summary payload includes the fields required by
Phase 3 acceptance:

| Field                   | Type           | Notes                                   |
| ----------------------- | -------------- | --------------------------------------- |
| `raptor_tree_id`        | str            | Stable per (build, prompt, root).       |
| `raptor_node_id`        | str            | UUID-shaped, stable per (tree, level, cluster). Used as the Qdrant point ID on live apply. |
| `raptor_level`          | int            | 0 = leaf ref, 1 = cluster, 2 = root.    |
| `raptor_parent_ids`     | list[str]      | Empty for level-2 root in Phase 3.      |
| `raptor_child_ids`      | list[str]      | Sorted, deterministic, leaf or cluster. |
| `raptor_cluster_id`     | str            | Stable per cluster.                     |
| `raptor_summary_of`     | list[str]      | Sorted leaf IDs this node summarizes.   |
| `raptor_root_id`        | str            | Tree root id.                           |
| `raptor_build_id`       | str            | Stable per (prompt, config, leaves).    |
| `raptor_prompt_version` | str            | Currently `raptor-mvp-extractive-v1`.   |
| `source_hashes`         | list[str]      | Sorted, de-duplicated content hashes.   |
| `derived_from`          | list[dict]     | Per-child provenance edges.             |
| `derivation_type`       | str            | Always `raptor_summary`.                |
| `canonical`             | false          | Always False.                           |
| `requires_review`       | true           | Always True.                            |
| `raptor_review_status`  | str            | Always `review_required`.               |
| `raptor_node_role`      | str            | `leaf_ref` (level ≤ 0) or `summary`.    |

## Usage sketch

```python
from qdrant_memory.raptor import (
    RaptorBuilder,
    RaptorSearcher,
    HybridRouter,
)

# Phase 3 — dry-run build
def _point(pid, text, profile_id="default"):
    return {
        "id": pid,
        "payload": {
            "text": text,
            "source_type": "manual",
            "profile_id": profile_id,
        },
    }

points = [
    _point("leaf-1", "first note about RAPTOR schema"),
    _point("leaf-2", "second note about RAPTOR schema"),
    _point("leaf-3", "third note about RAPTOR schema", profile_id="profile-A"),
]
manifest = RaptorBuilder(max_cluster_size=2).build(points)
# manifest.dry_run == True, manifest.mutations_performed == False
# manifest.candidate_node_payloads is sorted by raptor_node_id
# manifest.manifest_digest is deterministic (no volatile timestamps)

# Phase 5 — read-only search/zoom + hybrid retrieve
raptor_searcher = RaptorSearcher(
    qdrant=client, retriever=memory_retriever,
    collection_name="memory", scope={"profile_id": "default"},
)
graph_retriever = GraphMemoryRetriever(
    qdrant=client, embeddings=emb, collection_name="memory",
)
router = HybridRouter(
    qdrant=client, embeddings=emb, collection_name="memory",
    base_retriever=memory_retriever,
    graph_retriever=graph_retriever,
    raptor_searcher=raptor_searcher,
    scope={"profile_id": "default"},
)
result = router.retrieve("how do we apply RAPTOR schemas?", mode="hybrid", top_k=5)
# result is JSON-serializable; contains summaries, cited_leaves,
# exact_hits, graph_relations, warnings, and debug.
```

## What is intentionally NOT in Phases 3–5

- No `qdrant.upsert`, `update_payload`, `delete`, `delete_filter`,
  `delete_ids`, or network/API calls. Phase 5 explicitly excludes these.
- No abstractive LLM summaries — the builder keeps summaries strictly
  extractive (child snippets only).
- No new dependencies — stdlib only.
- No `qdrant_memory_search` behavior change — Phase 5 adds a new tool
  (`qdrant_memory_retrieve`) and a new CLI subcommand
  (`hermes qdrant retrieve`); existing tools remain unchanged.
- Phase 5 does not promote parent summaries on their own when
  `mode="evidence"`; they must have at least one cited leaf.
