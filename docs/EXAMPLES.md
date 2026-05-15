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
  "point_count": 123
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

Example:

```bash
hermes chat -q 'Without using tools, if Qdrant memory contains context about Project X, answer with the file_path of the most relevant memory; otherwise answer NO_QDRANT_CONTEXT.' --quiet
```

## Safe delete

First search with metadata and copy the exact point IDs. Then ask Hermes to call `qdrant_memory_forget` with `dry_run: true`.

Only run with `dry_run: false` after verifying the IDs.
