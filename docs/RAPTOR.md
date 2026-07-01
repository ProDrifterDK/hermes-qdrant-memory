# RAPTOR (Phase 3 — schema + dry-run builder)

Phase 3 of the LCM + Qdrant RAPTOR backlog introduces a deterministic dry-run
builder for RAPTOR trees. The builder proposes summary nodes and per-build
manifests **without mutating Qdrant**. Phase 4 (apply/status) is not yet
implemented.

## Public surface

The RAPTOR module lives at `qdrant_memory.raptor`. The package re-exports:

- `RaptorBuilder`, `build_raptor_dry_run`
- `RaptorNode`, `RaptorTree`, `RaptorCluster`, `RaptorScope`,
  `RaptorBuildManifest`
- `compute_tree_id`, `compute_root_id`, `compute_node_id`,
  `compute_build_id`, `compute_manifest_digest`
- `RAPTOR_DERIVATION_TYPE`, `RAPTOR_LEVEL_LEAF`, `RAPTOR_LEVEL_ROOT`,
  `DEFAULT_PROMPT_VERSION`, `RAPTOR_REQUIRED_NODE_FIELDS`,
  `DEFAULT_MAX_CLUSTER_SIZE`

## Hard guarantees

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

## Schema fields

Every RAPTOR candidate summary payload includes the fields required by
Phase 3 acceptance:

| Field                   | Type           | Notes                                   |
| ----------------------- | -------------- | --------------------------------------- |
| `raptor_tree_id`        | str            | Stable per (build, prompt, root).       |
| `raptor_node_id`        | str            | Stable per (tree, level, cluster).      |
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
from qdrant_memory.raptor import RaptorBuilder

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
```

## What is intentionally NOT in Phase 3

- No `qdrant.upsert`, `update_payload`, `delete`, `delete_filter`,
  `delete_ids`, or network/API calls.
- No apply/status tooling — Phase 4 will own that surface.
- No abstractive LLM summaries — Phase 3 keeps summaries strictly
  extractive (child snippets only).
- No new dependencies — stdlib only.
- No public tool/handler surface change — Phase 3 only adds the
  `qdrant_memory.raptor` package; existing tools remain unchanged.