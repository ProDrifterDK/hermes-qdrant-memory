# Limitations

This plugin is intentionally conservative. It provides the memory-provider foundation, not a fully autonomous memory organism.

## External services required

The plugin does not run Qdrant or embeddings for you. You must operate those services separately.

## No secret filtering

The file indexer does not automatically detect secrets, tokens, credentials, or private data.

Do not index broad home directories blindly. Start with a dry-run and explicit paths.

See `docs/SAFETY.md` for the scanner-safe docs/tests policy and the rule against literal fake secrets.

## Text-only file indexing

Default file indexing supports:

- `.md`
- `.txt`

It does not parse:

- PDFs
- DOCX
- images
- audio/video
- databases
- browser profiles

You can preprocess those formats externally and index extracted text if appropriate.

## Embedding model compatibility

Vector size must match the Qdrant collection. If you switch from a 1024-dimensional model to a 768-dimensional model, the old collection cannot accept the new vectors.

Use a new collection or reindex from scratch.

## Retrieval is semantic, not authoritative

Qdrant returns similar chunks. Similarity is not truth. Hermes should still reason, verify, and prefer current user instructions over stale memory. Use [LCM_BOUNDARY.md](LCM_BOUNDARY.md) to decide when active-session LCM recovery is the correct tool instead.

## Auto-recall noise

Low thresholds can surface weakly related chunks. Tune:

- `auto_recall_top_k`
- `search_candidates`
- `min_raw_score`
- `min_final_score`
- `display_tokens`

## Learning auto-extraction is gated

M7 implements the `hermes_learnings` collection plus explicit `qdrant_learning_store` and `qdrant_learning_search` tools. These tools are enabled by default and can be disabled with `qdrant_memory.learning_enabled: false`.

M7.1 adds heuristic candidate extraction, not blind auto-learning. `learning_auto_extract_enabled` defaults to false. When turned on, candidates are kept pending for review through `qdrant_learning_preview`; `qdrant_learning_approve` defaults to dry-run and requires explicit `dry_run: false` to store.

M7.2 adds semantic dedupe against existing `hermes_learnings` before preview/pending. This reduces repeated candidates, but the threshold uses the raw Qdrant similarity score, so deployments using a non-default distance/embedding scale may need to tune `learning_auto_extract_semantic_dedupe_threshold`.

The extractor is deliberately narrow: explicit user corrections and tool failures only when followed by a correction/resolution. It does not use an LLM yet and should not be treated as a full learning pipeline. Compression/session-end behavior must preserve the [LCM boundary](LCM_BOUNDARY.md).

## Consolidation live actions are gated in M9

`qdrant_memory_consolidate` still cannot apply actions and still rejects `dry_run: false`; it only analyzes and persists report artifacts. `qdrant_memory_consolidation_apply` can apply one proposal at a time, but live mode requires `dry_run: false` plus `approve: true`, an exact `report_id`, and an exact `proposal_id`.

M9 actions are deliberately narrow: delete uses explicit IDs only, merge preserves a canonical point and deletes explicit duplicates, and promote-to-skill writes a local draft artifact rather than installing an active skill. `quality_warning` proposals and secret-bearing inputs remain manual-review only.

See `docs/SAFETY.md` for the canonical report/apply and quality-warning rules.

The duplicate detector is intentionally conservative and mostly text-fingerprint based; it is a safety report, not a semantic merge engine. Future versions may add similarity search or LLM-assisted abstraction, but those should remain gated and apply-by-proposal-id.

## Reconsolidation is manual-review only in M10

Automatically rewriting remembered facts is dangerous. M10 can optionally report `reconsolidation_candidate` proposals when explicitly requested with `include_reconsolidation: true` (or enabled by config), but it does not rewrite Qdrant memory.

Reconsolidation apply supports only `draft_review`: a local markdown artifact under `$HERMES_HOME/qdrant_memory/consolidation/reconsolidation_drafts/`. Live draft creation still requires exact `report_id`, exact `proposal_id`, `dry_run: false`, and `approve: true`. It must not call Qdrant `upsert`, `update_payload`, `delete_ids`, or `delete_filter`.

See `docs/SAFETY.md` for the canonical reconsolidation rules.

Future fact rewriting, if ever added, should remain gated by:

- explicit enablement,
- dry-run preview,
- similarity checks,
- source provenance,
- and user confirmation for important facts.

## Current install path and CLI discovery caveat

Current Hermes user memory-provider discovery is most compatible with:

```text
~/.hermes/plugins/qdrant
```

If you prefer category layout such as:

```text
~/.hermes/plugins/memory/qdrant
```

create a compatibility symlink:

```bash
ln -s ~/.hermes/plugins/memory/qdrant ~/.hermes/plugins/qdrant
```

The native CLI MVP uses Hermes memory-provider CLI discovery. `hermes qdrant ...` is available only when this plugin is installed as the active `qdrant` memory provider and a fresh Hermes process has loaded that configuration. Top-level `hermes --help` may not list plugin commands because Hermes avoids eager plugin imports for startup performance; use `hermes qdrant --help` as the direct check.

`hermes qdrant doctor` is status-backed in v0.2.1. It is a lightweight diagnostics alias, not a separate deep health checker yet.

## Performance

Large indexing runs are embedding-bound. Local CPU embedding servers may take several minutes for thousands of chunks.

Use `max_files`, dry-run, and small initial directories before broad indexing.

## Directory deletion sync

M6.1 supports conservative directory-level deletion sync. When the input path is a directory, the indexer can detect existing Qdrant file chunks whose `file_path` is under that directory but whose file no longer exists on disk. Dry-run reports `deleted_file_paths` and `deleted_file_ids`; live indexing deletes those explicit IDs.

Safety constraints:

- directory deletion sync is skipped if walking hit `max_files`, because a truncated walk is not a complete source of truth;
- existing files that are merely excluded by extension/config are not deleted;
- only points with `chunk_type=file_chunk` participate in directory deletion sync;
- paths outside the explicitly indexed directory roots are ignored.
