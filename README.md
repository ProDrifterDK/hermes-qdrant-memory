# Hermes Qdrant Memory Provider

Qdrant-backed semantic memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), based on Alan Garate / Resyst Softwares' hippocampal associative memory system originally implemented in ResystBot, Alan's PicoClaw fork.

This plugin turns Hermes memory into an external associative substrate: conversations, manually stored memories, and selected Markdown/text files are embedded into Qdrant and recalled when semantically relevant to the current turn.

> Qdrant memory is not a bigger prompt. It is an associative substrate: retrieve what matters, when it matters, with provenance.

## Status

Public beta / experimental. The MVP is functional and tested, but it depends on external Qdrant and embedding services. Learning, sleep consolidation, and reconsolidation are roadmap items and are intentionally disabled by default.

## What it does

- Uses Qdrant as a vector database for long-term semantic memory.
- Uses any OpenAI-compatible `/v1/embeddings` endpoint.
- Implements Hermes `MemoryProvider` hooks for cross-session recall.
- Injects relevant memories into the current turn as ephemeral context.
- Indexes completed conversation turns asynchronously.
- Provides explicit tools for status, search, store, indexing, and safe deletion.
- Can index Markdown and text notes/directories with dry-run first.
- Preserves provenance fields such as `source_type`, `file_path`, `heading`, `session_id`, `profile_id`, and timestamps.

## What it does not do

- It does not replace LCM/current-session context recovery.
- It does not bundle Qdrant or an embedding model.
- It does not parse PDFs, DOCX, images, or audio by default.
- It does not automatically detect and remove secrets before indexing.
- It does not guarantee truth; it retrieves semantically similar chunks.
- It does not automatically mutate or rewrite memories through reconsolidation.
- It is not a complete ResystBot memory clone yet; it is the stable MemoryProvider foundation.

## Capabilities

| Capability | Status |
|---|---|
| Qdrant REST client | Implemented |
| OpenAI-compatible embedding client | Implemented |
| Hermes MemoryProvider auto-recall | Implemented |
| Completed-turn write-through indexing | Implemented |
| Manual memory store/search/status tools | Implemented |
| Markdown/text file indexing | Implemented |
| Dry-run index preview | Implemented |
| Force reindex by `file_path` | Implemented |
| Safe forget by explicit point IDs | Implemented |
| Learning collection | Configured, future behavior |
| Sleep consolidation | Future |
| Reconsolidation | Future, disabled by design |
| Dashboard/UI | Not included |

## Requirements

- Linux/macOS/WSL environment capable of running Hermes Agent.
- Hermes Agent with user memory-provider plugin support.
- Python 3.10+.
- Qdrant reachable over HTTP.
- An OpenAI-compatible embedding endpoint at `/v1/embeddings`.
- A known vector size that matches the embedding model.

Tested local stack:

- Qdrant: `http://127.0.0.1:6333`
- Embeddings: local llama.cpp OpenAI-compatible server at `http://127.0.0.1:8080/v1`
- Model: `bge-m3`
- Vector size: `1024`

Example Qdrant local service:

```bash
docker run -p 6333:6333 \
  -v "$HOME/.qdrant/hermes:/qdrant/storage" \
  qdrant/qdrant
```

Any embedding server is acceptable if it supports OpenAI-style requests:

```http
POST /v1/embeddings
{
  "model": "bge-m3",
  "input": "search_document: your text"
}
```

## Installation

Recommended install path for current Hermes user plugins:

```bash
git clone https://github.com/ProDrifterDK/hermes-qdrant-memory ~/.hermes/plugins/qdrant
```

Activate the provider:

```bash
hermes config set memory.provider qdrant
hermes config set qdrant_memory.enabled true
hermes config set qdrant_memory.qdrant_url http://127.0.0.1:6333
hermes config set qdrant_memory.embedding_url http://127.0.0.1:8080/v1
hermes config set qdrant_memory.embedding_model bge-m3
hermes config set qdrant_memory.vector_size 1024
```

Start a fresh Hermes session or restart the gateway after changing memory provider/plugin configuration.

Verify:

```bash
hermes chat -q 'Call the qdrant_memory_status tool and summarize whether qdrant_ok and embedding_ok are true.' --quiet
```

If both backends are reachable, the status tool should report `qdrant_ok: true` and `embedding_ok: true`.

## Configuration

The plugin reads config in this order:

1. Built-in defaults.
2. `$HERMES_HOME/qdrant_memory.json`.
3. `qdrant_memory:` section in Hermes `config.yaml`.
4. Environment variables named `HERMES_QDRANT_MEMORY_<KEY>`.

Example config:

```yaml
memory:
  provider: qdrant

qdrant_memory:
  enabled: true
  qdrant_url: http://127.0.0.1:6333
  embedding_url: http://127.0.0.1:8080/v1
  embedding_model: bge-m3
  collection_name: hermes_memory
  learning_collection_name: hermes_learnings
  vector_size: 1024
  distance: Cosine
  auto_recall: true
  auto_recall_top_k: 5
  search_candidates: 20
  decay_rate: 0.001
  max_chunk_tokens: 256
  display_tokens: 500
  sync_turns: true
  sync_subagents: false
  learning_enabled: false
  consolidation_enabled: false
  reconsolidation_enabled: false
  query_prefix: "search_query: "
  document_prefix: "search_document: "
  scope_mode: profile
  min_raw_score: 0.0
  min_final_score: 0.0
  qdrant_api_key: ""
  embedding_api_key: ""
  index_dirs: []
  index_extensions: [".md", ".txt"]
  index_exclude_dirs: [".git", "node_modules", ".venv", "venv", "__pycache__", "dist", "build", "target", ".next", ".cache"]
  index_max_files: 500
  index_dry_run_default: true
```

### Chunk-size note

Some local embedding servers have strict physical batch/context limits. If llama.cpp or another backend returns errors such as `input (...) is too large to process`, reduce:

```bash
hermes config set qdrant_memory.max_chunk_tokens 128
```

Changing embedding model or vector size generally requires a new collection or full reindex.

## Tools exposed to Hermes

### `qdrant_memory_status`

Checks provider, Qdrant, embedding endpoint, collection existence, and point counts.

### `qdrant_memory_search`

Semantic search over stored memories and indexed files.

Useful arguments:

- `query`
- `top_k`
- `source_type`
- `include_metadata`

### `qdrant_memory_store`

Manually stores a memory chunk with optional metadata.

### `qdrant_memory_index`

Indexes Markdown/text files from explicit paths or configured `index_dirs`.

Safe default: `dry_run` is true unless overridden.

Example dry run:

```text
Call qdrant_memory_index with:
{
  "paths": ["~/Documents/Notes"],
  "dry_run": true,
  "max_files": 100
}
```

Example live index:

```text
Call qdrant_memory_index with:
{
  "paths": ["~/Documents/Notes"],
  "dry_run": false,
  "force": true,
  "max_files": 100
}
```

`force: true` deletes existing chunks for each indexed `file_path` before upserting the fresh chunks.

### `qdrant_memory_forget`

Deletes explicit point IDs only. Dry-run defaults to true. There is intentionally no query-based deletion without preview.

## Indexing safety

Before indexing a broad directory:

1. Start with dry-run.
2. Inspect file count, skipped files, and chunk count.
3. Exclude private/secret-heavy directories.
4. Use `force: true` when reindexing changed files.
5. Verify retrieval with a concrete topic query.

This plugin does not know which files are safe for your threat model. Treat indexed files as memory that may later be surfaced in model context.

## Philosophy

The design follows a hippocampal pattern rather than a static prompt-stuffing pattern:

- The prompt stays small; recall is query-aware.
- Memories are retrieved by association, not blindly appended.
- Recalled context is ephemeral and should not recursively re-index itself.
- Every retrieved chunk should carry provenance.
- Forgetting is explicit and conservative.
- Reconsolidation is dangerous and should stay gated/manual until mature.

## Development

Run tests:

```bash
python -m pytest tests -q
python -m compileall -q qdrant_memory __init__.py
```

No third-party Python package is required by the plugin runtime; it uses the Python standard library for HTTP calls. Tests require `pytest`.

## Documentation

- `docs/ARCHITECTURE.md`
- `docs/REQUIREMENTS.md`
- `docs/LIMITATIONS.md`
- `docs/EXAMPLES.md`

## License

MIT
