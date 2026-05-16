# Examples

## Check status

Ask Hermes:

```text
Call qdrant_memory_status and summarize qdrant_ok, embedding_ok, collection_name, and point_count.
```

Expected fields:

```json
{
  "provider": "qdrant",
  "active": true,
  "qdrant_ok": true,
  "embedding_ok": true,
  "collection_name": "hermes_memory",
  "point_count": 123,
  "learning_collection_name": "hermes_learnings",
  "learning_point_count": 12
}
```

## Store a manual memory

Ask Hermes:

```text
Store this in Qdrant memory: "Project X uses pnpm and deploys on Railway." Use source_type manual and tags project-x,railway.
```

Internally this should call `qdrant_memory_store`.

## Search memory

Ask Hermes:

```text
Search Qdrant memory for "Project X Railway deployment pnpm" and include metadata.
```

`qdrant_memory_search` is for durable semantic memory, not exact active-session recovery. Use [LCM_BOUNDARY.md](LCM_BOUNDARY.md) to decide when LCM is the right tool.

Useful filters:

```json
{
  "query": "Project X Railway deployment pnpm",
  "top_k": 5,
  "source_type": "manual",
  "include_metadata": true
}
```

## Dry-run note indexing

Ask Hermes:

```text
Dry-run Qdrant indexing for ~/Documents/Notes with max_files=100. Do not index live yet.
```

Equivalent tool payload:

```json
{
  "paths": ["~/Documents/Notes"],
  "dry_run": true,
  "max_files": 100
}
```

## Live note indexing

After reviewing the dry run:

```text
Index ~/Documents/Notes into Qdrant memory with dry_run=false, force=true, max_files=100.
```

Equivalent tool payload:

```json
{
  "paths": ["~/Documents/Notes"],
  "dry_run": false,
  "force": true,
  "max_files": 100
}
```

## Verify auto-recall

Start a fresh Hermes session and ask about a topic that exists only in the indexed notes. If auto-recall is working, Hermes should be able to cite the relevant `file_path` or source metadata without manually calling search.

Auto-recall is Qdrant semantic recall, not LCM active-session expansion. For the boundary, see [LCM_BOUNDARY.md](LCM_BOUNDARY.md).

Example:

```bash
hermes chat -q 'Without using tools, if Qdrant memory contains context about Project X, answer with the file_path of the most relevant memory; otherwise answer NO_QDRANT_CONTEXT.' --quiet
```

## Store a procedural learning

Ask Hermes:

```text
Store this as a Qdrant learning: lesson="When llama.cpp embeddings fail with physical batch size 512, lower qdrant_memory.max_chunk_tokens to 128." learning_type=tool_failure_lesson trigger="embedding HTTP 500 input too large" correction="set max_chunk_tokens=128" evidence="vault-wide indexing passes after lowering chunk size"
```

Equivalent tool payload:

```json
{
  "lesson": "When llama.cpp embeddings fail with physical batch size 512, lower qdrant_memory.max_chunk_tokens to 128.",
  "learning_type": "tool_failure_lesson",
  "trigger": "embedding HTTP 500 input too large",
  "correction": "set max_chunk_tokens=128",
  "evidence": "vault-wide indexing passes after lowering chunk size",
  "tool_name": "qdrant_memory_index"
}
```

## Search procedural learnings

```json
{
  "query": "llama.cpp embedding input too large",
  "learning_type": "tool_failure_lesson",
  "top_k": 5,
  "include_metadata": true
}
```

## Preview and approve gated automatic candidates

Automatic extraction is off by default. To test it temporarily:

```yaml
qdrant_memory:
  learning_auto_extract_enabled: true
  learning_auto_extract_mode: preview
  learning_auto_extract_semantic_dedupe_enabled: true
  learning_auto_extract_semantic_dedupe_threshold: 0.9
```

When `on_pre_compress` or `on_session_end` sees a strong candidate, M7.2 first searches existing `hermes_learnings` with the same learning type. If no high-similarity duplicate is found, inspect the pending candidate:

```json
{
  "include_metadata": true
}
```

Approve only after review. Dry-run is the default:

```json
{
  "candidate_id": "<candidate_id>"
}
```

Store live only when the candidate is durable and non-secret:

```json
{
  "candidate_id": "<candidate_id>",
  "dry_run": false
}
```

## Safe delete

First search with metadata and copy the exact point IDs. Then ask Hermes to call `qdrant_memory_forget` with `dry_run: true`.

Only run with `dry_run: false` after verifying the IDs.

## M9/M10 consolidation report and gated apply

These examples are operational examples only. The canonical safety rules are in `docs/SAFETY.md`.

Generate and persist a reflection pass over memory and procedural learnings:

```json
{
  "scope": "both",
  "max_points": 200,
  "max_groups": 20,
  "include_examples": true,
  "include_reconsolidation": true,
  "persist": true
}
```

`qdrant_memory_consolidate` returns duplicate clusters, stale low-value candidates, learning promotion candidates, quality warnings, a `report_id`, and a local artifact path. It still cannot apply actions; `dry_run: false` is rejected for report generation.

Preview one proposal without mutation:

```json
{
  "report_id": "abc123...",
  "proposal_id": "stale_low_value-...",
  "action": "delete"
}
```

Live apply requires explicit approval:

```json
{
  "report_id": "abc123...",
  "proposal_id": "stale_low_value-...",
  "action": "delete",
  "dry_run": false,
  "approve": true
}
```

Supported live actions are explicit-ID `delete`, canonical-preserving duplicate `merge`, `promote_to_skill` draft creation, and M10 `draft_review` for reconsolidation candidates. `quality_warning` proposals are manual-only. `draft_review` writes only a local markdown review artifact and never mutates Qdrant memory.

Preview a reconsolidation review draft:

```json
{
  "report_id": "abc123...",
  "proposal_id": "reconsolidation_candidate-...",
  "action": "draft_review"
}
```

Create the review draft after inspecting the dry-run plan:

```json
{
  "report_id": "abc123...",
  "proposal_id": "reconsolidation_candidate-...",
  "action": "draft_review",
  "dry_run": false,
  "approve": true
}
```
