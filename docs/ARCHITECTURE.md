# Architecture

`hermes-qdrant-memory` is a Hermes Agent `MemoryProvider` plugin.

It is intentionally not a context engine. Context engines such as LCM recover detail from the current session. This plugin provides cross-session associative recall and explicit memory tooling. See [LCM_BOUNDARY.md](LCM_BOUNDARY.md) for the detailed interoperability boundary.

## Runtime flow

1. Hermes loads the provider through the `~/.hermes/plugins/qdrant` compatibility symlink when `memory.provider: qdrant` is configured; the public checkout path is `~/.hermes/plugins/memory/qdrant`.
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
- `qdrant_memory/memory_pr.py` — pure Memory PR packet builder, static HTML renderer, private artifact writer, and offline fixture CLI.
- `qdrant_memory/scoring.py` — retrieval scoring helpers.

## Collections

Default collections:

- `hermes_memory` — conversations, manual memories, indexed files.
- `hermes_learnings` — procedural lessons, tool failure lessons, workflow lessons, environment quirks, and user corrections.

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
- `file_size`
- `file_sha256`
- `manifest_version`
- `chunk_id`
- `chunk_hash`
- `chunk_index`
- `chunk_count`
- `heading`
- optional fact metadata: `fact_key`, `reconsolidation_key`, `subject`, `topic`, `entity`

Fact metadata is generated conservatively at write origin by `qdrant_memory/fact_metadata.py`. It accepts explicit tags such as `fact:teamforge.mcp.binary`, `subject:TeamForge MCP binary`, `topic:TeamForge`, and `entity:terminal`; clear single-line fact statements such as `TeamForge MCP binary is teamforge-mcp`; file headings as weak `topic`; and structured learning context (`learning_type`, `tool_name`, command executable, project basename). Secret-bearing text or secret-like tags produce no fact metadata. Generic conversation turns do not get fact keys by default.

## Learning collection

M7 separates declarative memory from procedural learning:

- `hermes_memory` stores conversations, manual memories, and indexed files.
- `hermes_learnings` stores operational lessons that should change future behavior.

Learning payloads use `source_type=learning` plus `learning_type` / `chunk_type` values:

- `tool_failure_lesson`
- `user_correction`
- `workflow_lesson`
- `environment_quirk`

Learning payloads also carry `trigger`, `mistake`, `correction`, `evidence`, `tool_name`, `command`, and `promote_to_skill_candidate` fields.

M7 is manual/gated: use `qdrant_learning_store` and `qdrant_learning_search`. `qdrant_memory.learning_enabled` controls whether learning tools are available.

M7.1 adds gated automatic extraction candidates from `on_pre_compress` and `on_session_end`. Automatic extraction is disabled by default (`learning_auto_extract_enabled: false`). When enabled, heuristics detect explicit user corrections and tool failures with explicit follow-up corrections, then place candidates in an in-memory pending buffer. The candidates are reviewed with `qdrant_learning_preview` and stored only after explicit `qdrant_learning_approve` with `dry_run: false`. There is still no blind auto-store path.

M7.2 adds semantic dedupe before candidates enter the pending buffer. It queries existing `hermes_learnings` with the same scope and `learning_type`, uses raw Qdrant similarity, does not update access metadata, and fails open if Qdrant/embeddings are unavailable.

## M8/M9/M10 sleep consolidation

M8 introduced `qdrant_memory_consolidate` as a dry-run reflection pass. It reads points with Qdrant scroll filters, respects the active scope, and analyzes `hermes_memory`, `hermes_learnings`, or both. Report generation never calls memory-action upsert/delete/approval/payload update methods and does not change access metadata.

M9a persists each report as a local JSON artifact under `$HERMES_HOME/qdrant_memory/consolidation/` (or `consolidation_artifact_dir`). M9b/M9c add `qdrant_memory_consolidation_apply`, which loads a persisted `report_id`, finds one exact `proposal_id`, previews by default, and only performs live work when `dry_run=false` and `approve=true`.

M10 adds optional `reconsolidation_candidate` proposals. These detect multiple memory points that share an explicit strong fact key (`reconsolidation_key`, `fact_key`, or `subject`) but contain different normalized text. `topic` and `entity` are retained as filters/provenance only and do not drive grouping. Reconsolidation is opt-in via `include_reconsolidation` or config. M10 never rewrites Qdrant facts; its only supported live action is `draft_review`, which writes a local markdown review artifact.

M11b reduces consolidation noise. Markdown indexing skips heading-only sections (for example `## Tareas`, `## Notas`, `### Contribution`) so they are not embedded as standalone memories. Secret detection distinguishes conceptual token-budget/cache/counting language from credential-bearing token assignments and still flags bearer/API/password/private-key patterns.

The report may propose duplicate clusters, stale low-value memory candidates, learning promotion candidates, quality warnings, and optional reconsolidation candidates. Apply semantics are intentionally narrow: duplicate clusters can merge by preserving one canonical point and deleting explicit duplicates; stale low-value proposals can delete explicit IDs only; learning promotion candidates create local draft skill artifacts and mark the learning as promoted-to-draft; reconsolidation candidates create review drafts only. Quality warnings remain manual-review only.

Safety gates:

- report generation rejects `dry_run=false` and cannot apply proposals;
- live apply requires `report_id`, `proposal_id`, `dry_run=false`, and `approve=true`;
- actions are derived from proposal type and mismatches are rejected;
- all deletes use explicit IDs only; `delete_filter` is not used by consolidation apply;
- current affected points are re-fetched before live action, so missing points reject as stale;
- secret-bearing quality warnings and secret-bearing merge/promotion inputs require manual review;
- skill promotion creates a draft artifact only and never installs a skill automatically;
- reconsolidation candidates can only create local review drafts and must not mutate Qdrant facts.

## Memory PR review packets

The Build Week 2026 Memory PR extension sits strictly on the read side of the report/apply boundary. `qdrant_memory_memory_pr` loads one persisted report, selects one exact proposal, resolves its configured collection, retrieves only its explicit affected point IDs with payloads and without vectors, and requires the retrieved ID set to match exactly. It never delegates to `qdrant_memory_consolidation_apply`.

New consolidation proposals carry `review_point_snapshots`: sorted point ID, projection descriptor, and SHA-256 digest records computed from stable sanitized point content. Projection `memory-pr-review-point` version 1 is a whitelist of memory text, provenance, fact identity, canonical/stale/review state, validity, and supersession fields. It intentionally excludes operational access/ranking bookkeeping such as `access_count`, `last_accessed`, `decay_score`, and ranking debug data. The builder recomputes those digests from the current exact points and labels each comparison `unchanged`, `changed`, or `unknown`. `Unknown` is conservative: it covers missing/unversioned snapshots, unsupported projection versions, and identity- or secret-bearing content whose private text is replaced before hashing.

Persisted proposal evidence uses its own versioned schema. Every evidence record must carry an exact affected point ID. Missing, malformed, or unknown IDs fail closed. Evidence attributed to a currently identity-bearing point—or carrying identity-bearing metadata at any nesting level—is replaced as a whole rather than selectively redacting top-level keys.

The packet identity hashes canonical sorted JSON containing stable sanitized review content. It excludes generation time and artifact paths. HTML rendering is a pure escaped transformation over the packet. Report reads resolve paths without creating or chmodding directories. Artifact persistence occurs only when the caller supplies an output directory; new output directories are created private, while pre-existing directories must already be current-user-owned mode `0700`. The module fixture CLI is independent of the Hermes provider and all external services.

## File manifest sync

Indexed files are treated as a source of truth by `file_path`.

Each file chunk carries manifest metadata (`file_sha256`, `file_size`, `chunk_id`, `chunk_hash`, `chunk_index`, `chunk_count`). On reindex, the indexer uses Qdrant scroll by `file_path` to compare existing point IDs with the freshly prepared chunk IDs:

- `dry_run: true` reports `stale_ids` and `stale_count` without embedding, upserting, or deleting.
- live indexing deletes only stale point IDs with `delete_ids` when scroll is available.
- `force: true` falls back to broad `delete_filter(file_path)` only for older clients that cannot scroll existing points.
- empty files still participate in manifest sync, so shrinking a file to zero chunks removes all previous chunks for that file.

When an input path is a directory, M6.1 also performs directory-level deletion sync:

- scroll existing `chunk_type=file_chunk` points;
- client-side filter to points whose `file_path` is under the indexed directory roots;
- compare against the current file manifest;
- report `deleted_file_paths` and `deleted_file_ids` in dry-run;
- delete those IDs in live mode only when the path no longer exists on disk.

Directory deletion sync is disabled when file walking hits `max_files`, because a truncated walk is not a safe source of truth.

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

The canonical safety contract is [SAFETY.md](SAFETY.md); this section summarizes the architectural enforcement points. The active-session/current-session boundary is defined in [LCM_BOUNDARY.md](LCM_BOUNDARY.md).

- Auto-recall is context-only; retrieved memory is not itself a command.
- The writer strips known injected memory markers to reduce recursive contamination.
- File indexing defaults to dry-run.
- File reindexing uses manifest sync to prevent stale high-index chunks after a file shrinks and stale chunks after files are deleted from indexed directories.
- Deletion requires explicit point IDs and defaults to dry-run.
- Reconsolidation is manual-review only in M10: report generation is opt-in, and live `draft_review` creates only local markdown artifacts without Qdrant mutation.
