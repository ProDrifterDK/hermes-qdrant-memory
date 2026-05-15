# Hermes Qdrant Memory Provider

Qdrant-backed semantic memory provider for [Hermes Agent](https://github.com/NousResearch/hermes-agent), based on Alan Gárate / Resyst Softwares' hippocampal associative memory system originally implemented in ResystBot, Alan's PicoClaw fork.

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

Tested local stack used by Resyst Softwares:

- Qdrant: `http://127.0.0.1:6333`
- Embeddings: local llama.cpp OpenAI-compatible server at `http://127.0.0.1:8080/v1`
- Model: `bge-m3-q6_k.gguf`, exposed to Hermes as `bge-m3`
- Vector size: `1024`

### Install and run Qdrant locally

The plugin needs Qdrant reachable over HTTP. The simplest local setup is Docker:

```bash
docker run -p 6333:6333 \
  -v "$HOME/.qdrant/hermes:/qdrant/storage" \
  qdrant/qdrant
```

Recommended persistent container:

```bash
docker run -d \
  --name hermes-qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$HOME/.qdrant/hermes:/qdrant/storage" \
  qdrant/qdrant:latest
```

Verify Qdrant:

```bash
curl http://127.0.0.1:6333/collections
```

If you already run Qdrant elsewhere, point Hermes at it:

```bash
hermes config set qdrant_memory.qdrant_url http://127.0.0.1:6333
```

For Qdrant Cloud or another authenticated endpoint, also set:

```bash
hermes config set qdrant_memory.qdrant_api_key YOUR_QDRANT_API_KEY
```

### Install and run the local embedding server with llama.cpp

This plugin does not require a specific embedding backend. It only needs an OpenAI-compatible `/v1/embeddings` API. The Resyst Softwares local setup uses llama.cpp serving `bge-m3-q6_k.gguf` on localhost.

Build llama.cpp with server support:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/src/llama.cpp
cmake -S ~/src/llama.cpp -B ~/src/llama.cpp/build -DLLAMA_CURL=ON
cmake --build ~/src/llama.cpp/build --config Release -j"$(nproc)"
```

Download a GGUF BGE-M3 embedding model. The Resyst Softwares deployment uses:

```text
~/.local/share/llama/models/bge-m3-q6_k.gguf
```

Start llama.cpp as an embedding server:

```bash
~/src/llama.cpp/build/bin/llama-server \
  -m ~/.local/share/llama/models/bge-m3-q6_k.gguf \
  --embeddings \
  --port 8080 \
  --host 127.0.0.1 \
  -ngl 999 \
  -c 4096
```

The currently tested Resyst Softwares service runs the same shape as a user systemd service:

```ini
[Unit]
Description=llama.cpp embedding server (bge-m3)
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart=/path/to/llama.cpp/build/bin/llama-server \
    -m /path/to/bge-m3-q6_k.gguf \
    --embeddings \
    --port 8080 \
    --host 127.0.0.1 \
    -ngl 999 \
    -c 4096
Restart=on-failure
RestartSec=3
StartLimitBurst=5
StartLimitIntervalSec=30

[Install]
WantedBy=default.target
```

Verify the embedding endpoint:

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"search_document: Hermes Qdrant memory test"}'
```

Configure Hermes for this local embedding setup:

```bash
hermes config set qdrant_memory.embedding_url http://127.0.0.1:8080/v1
hermes config set qdrant_memory.embedding_model bge-m3
hermes config set qdrant_memory.vector_size 1024
hermes config set qdrant_memory.query_prefix 'search_query: '
hermes config set qdrant_memory.document_prefix 'search_document: '
```

If llama.cpp returns an error like `input (...) is too large to process. increase the physical batch size`, reduce chunk size:

```bash
hermes config set qdrant_memory.max_chunk_tokens 128
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

## AI agent install playbook

This section is written for AI agents operating a user's machine. Follow it exactly unless the user gives different paths or already has Qdrant/embeddings running.

### Goal

Install this plugin as the active Hermes memory provider using:

- plugin path: `$HOME/.hermes/plugins/qdrant`
- Qdrant URL: `http://127.0.0.1:6333`
- embedding URL: `http://127.0.0.1:8080/v1`
- embedding model name in Hermes: `bge-m3`
- embedding model file example: `~/.local/share/llama/models/bge-m3-q6_k.gguf`
- vector size: `1024`
- query prefix: `search_query: `
- document prefix: `search_document: `

### Agent rules

1. Do not index broad directories until the user explicitly approves the target path.
2. Always run `qdrant_memory_index` with `dry_run: true` first.
3. Do not delete memories by query. Use `qdrant_memory_forget` only with explicit point IDs and `dry_run: true` first.
4. If the user already has Qdrant or an embedding server, reuse it instead of starting another one.
5. Restart Hermes/gateway only after warning the user if they are in an active gateway conversation.

### Step 1: Check prerequisites

```bash
command -v hermes
command -v git
command -v docker || true
curl -fsS http://127.0.0.1:6333/collections || true
curl -fsS http://127.0.0.1:8080/v1/models || true
```

If Qdrant is not reachable and Docker is available, start Qdrant:

```bash
docker run -d \
  --name hermes-qdrant \
  --restart unless-stopped \
  -p 6333:6333 \
  -p 6334:6334 \
  -v "$HOME/.qdrant/hermes:/qdrant/storage" \
  qdrant/qdrant:latest
```

Verify:

```bash
curl -fsS http://127.0.0.1:6333/collections
```

### Step 2: Ensure the embedding endpoint

If `http://127.0.0.1:8080/v1/embeddings` is already available, do not start a second server.

If the user wants the Resyst Softwares local setup, ensure llama.cpp is built and run:

```bash
mkdir -p "$HOME/src" "$HOME/.local/share/llama/models"
if [ ! -d "$HOME/src/llama.cpp/.git" ]; then
  git clone https://github.com/ggml-org/llama.cpp "$HOME/src/llama.cpp"
fi
cmake -S "$HOME/src/llama.cpp" -B "$HOME/src/llama.cpp/build" -DLLAMA_CURL=ON
cmake --build "$HOME/src/llama.cpp/build" --config Release -j"$(nproc)"
```

Place a BGE-M3 GGUF model at:

```text
$HOME/.local/share/llama/models/bge-m3-q6_k.gguf
```

Then start the embedding server:

```bash
"$HOME/src/llama.cpp/build/bin/llama-server" \
  -m "$HOME/.local/share/llama/models/bge-m3-q6_k.gguf" \
  --embeddings \
  --port 8080 \
  --host 127.0.0.1 \
  -ngl 999 \
  -c 4096
```

For long-running use, put that command in a user systemd service. After starting, verify:

```bash
curl -fsS http://127.0.0.1:8080/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","input":"search_document: Hermes Qdrant memory test"}' \
  | head -c 300
```

### Step 3: Install the plugin

```bash
mkdir -p "$HOME/.hermes/plugins"
if [ -d "$HOME/.hermes/plugins/qdrant/.git" ]; then
  git -C "$HOME/.hermes/plugins/qdrant" pull --ff-only
else
  rm -rf "$HOME/.hermes/plugins/qdrant"
  git clone https://github.com/ProDrifterDK/hermes-qdrant-memory "$HOME/.hermes/plugins/qdrant"
fi
```

### Step 4: Configure Hermes

```bash
hermes config set memory.provider qdrant
hermes config set qdrant_memory.enabled true
hermes config set qdrant_memory.qdrant_url http://127.0.0.1:6333
hermes config set qdrant_memory.embedding_url http://127.0.0.1:8080/v1
hermes config set qdrant_memory.embedding_model bge-m3
hermes config set qdrant_memory.vector_size 1024
hermes config set qdrant_memory.query_prefix 'search_query: '
hermes config set qdrant_memory.document_prefix 'search_document: '
hermes config set qdrant_memory.max_chunk_tokens 128
hermes config set qdrant_memory.index_dry_run_default true
```

Start a fresh Hermes CLI session or restart the gateway if the user wants gateway sessions to use the plugin immediately.

### Step 5: Verify inside Hermes

```bash
hermes chat -q 'Call the qdrant_memory_status tool. Answer OK only if qdrant_ok and embedding_ok are true; otherwise answer FAIL and summarize the failing fields.' --quiet
```

Expected result: `OK`.

### Step 6: Optional safe indexing

Only after the user gives a path, dry-run first:

```text
Call qdrant_memory_index with {"paths":["~/Documents/Notes"],"dry_run":true,"max_files":100}
```

If the dry-run looks correct and the user approves live indexing:

```text
Call qdrant_memory_index with {"paths":["~/Documents/Notes"],"dry_run":false,"force":true,"max_files":100}
```

Then verify recall with a concrete topic from the indexed notes.

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
