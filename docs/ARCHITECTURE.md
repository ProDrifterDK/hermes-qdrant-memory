# Architecture

`hermes-qdrant-memory` is a Hermes Agent `MemoryProvider` plugin.

It is intentionally not a context engine. Context engines such as LCM recover detail from the current session. This plugin provides cross-session associative recall and explicit memory tooling.

## Runtime flow

1. Hermes loads the provider from `~/.hermes/plugins/qdrant` when `memory.provider: qdrant` is configured.
2. `QdrantMemoryProvider.initialize()` loads config, creates clients, and ensures Qdrant collections exist.
3. On each turn, Hermes calls `prefetch()` or `queue_prefetch()` with the current query.
4. The plugin embeds the query with the configured query prefix.
5. Qdrant returns semantic candidates.
6. The retriever applies lightweight scoring using relevance, importance, and recency decay.
7. Retrieved chunks are formatted into a provenance-rich context block.
8. Completed turns are written asynchronously through `sync_turn()` when enabled.

## Main modules

- `__init__.py` — Hermes provider class and tool dispatch.
- `qdrant_memory/config.py` — defaults and config loading.
- `qdrant_memory/client.py` — minimal Qdrant REST client.
- `qdrant_memory/embeddings.py` — OpenAI-compatible embedding client.
- `qdrant_memory/schema.py` — memory chunk schema helpers.
- `qdrant_memory/retriever.py` — semantic search and prompt formatting.
- `qdrant_memory/writer.py` — completed-turn storage.
- `qdrant_memory/indexer.py` — Markdown/text file indexing.
- `qdrant_memory/tools.py` — Hermes tool schemas.
- `qdrant_memory/scoring.py` — retrieval scoring helpers.

## Collections

Default collections:

- `hermes_memory` — conversations, manual memories, indexed files.
- `hermes_learnings` — reserved for future procedural/outcome learning.

## Payload fields

Typical payload fields include:

- `text`
- `source`
- `source_type`
- `chunk_type`
- `importance`
- `confidence`
- `access_count`
- `created_at`
- `last_accessed`
- `decay_score`
- `tags`
- `profile_id`
- `platform`
- `user_id_hash`
- `chat_id_hash`
- `session_id`
- `project_path`
- `model`
- `provider`
- `file_path`
- `file_mtime`
- `chunk_index`
- `chunk_count`
- `heading`

## Source types

Common `source_type` values:

- `conversation`
- `manual`
- `indexed_file`
- `project_doc`
- `vault_note`
- `skill_doc`
- `learning`
- `reflection`
- `consolidated`

## Scope

`scope_mode` controls retrieval isolation:

- `profile` — default; isolate by Hermes profile/agent identity.
- `user` — include user hash when available.
- `chat` — include chat/thread hash when available.
- `global` — no scope filters.

For shared gateway deployments, avoid `global` unless you explicitly want cross-user recall.

## Safety design

- Auto-recall is context-only; retrieved memory is not itself a command.
- The writer strips known injected memory markers to reduce recursive contamination.
- File indexing defaults to dry-run.
- Deletion requires explicit point IDs and defaults to dry-run.
- Reconsolidation is not enabled in this MVP.
