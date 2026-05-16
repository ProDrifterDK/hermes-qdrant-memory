# Requirements

## Hermes

- Hermes Agent with user plugin support and memory-provider CLI discovery for `hermes qdrant ...`.
- Configure `memory.provider: qdrant`.
- Install this repository at `~/.hermes/plugins/qdrant` for current compatibility.

## Python

- Python 3.10+ recommended.
- Runtime dependencies: Python standard library only.
- Test dependency: `pytest`.

## Qdrant

A Qdrant HTTP endpoint must be reachable.

Default:

```text
http://127.0.0.1:6333
```

Local Docker example:

```bash
docker run -p 6333:6333 \
  -v "$HOME/.qdrant/hermes:/qdrant/storage" \
  qdrant/qdrant
```

Remote Qdrant is supported if reachable from Hermes. Use `qdrant_memory.qdrant_api_key` for authenticated deployments.

## Embeddings

An OpenAI-compatible embedding endpoint must be reachable.

Default:

```text
http://127.0.0.1:8080/v1
```

The endpoint must accept:

```http
POST /v1/embeddings
```

with JSON shaped like:

```json
{
  "model": "bge-m3",
  "input": "search_document: text to embed"
}
```

The returned vector size must match `qdrant_memory.vector_size`.

## Model selection

The default config assumes:

- embedding model: `bge-m3`
- vector size: `1024`

Other models are fine, but update both `embedding_model` and `vector_size`.

Changing embedding models after indexing usually means you should either:

1. use a new Qdrant collection, or
2. delete/recreate the existing collection and reindex.

## Config checklist

Minimum useful config:

```bash
hermes config set memory.provider qdrant
hermes config set qdrant_memory.enabled true
hermes config set qdrant_memory.qdrant_url http://127.0.0.1:6333
hermes config set qdrant_memory.embedding_url http://127.0.0.1:8080/v1
hermes config set qdrant_memory.embedding_model bge-m3
hermes config set qdrant_memory.vector_size 1024
```

Optional indexing config:

```bash
hermes config set qdrant_memory.index_dirs '["~/Documents/Notes"]'
hermes config set qdrant_memory.index_dry_run_default true
hermes config set qdrant_memory.index_max_files 500
```

If your embedding backend rejects large chunks:

```bash
hermes config set qdrant_memory.max_chunk_tokens 128
```
