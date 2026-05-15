# Limitations

This plugin is intentionally conservative. It provides the memory-provider foundation, not a fully autonomous memory organism.

## External services required

The plugin does not run Qdrant or embeddings for you. You must operate those services separately.

## No secret filtering

The file indexer does not automatically detect secrets, tokens, credentials, or private data.

Do not index broad home directories blindly. Start with a dry-run and explicit paths.

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

Qdrant returns similar chunks. Similarity is not truth. Hermes should still reason, verify, and prefer current user instructions over stale memory.

## Auto-recall noise

Low thresholds can surface weakly related chunks. Tune:

- `auto_recall_top_k`
- `search_candidates`
- `min_raw_score`
- `min_final_score`
- `display_tokens`

## Learning is not complete

The `hermes_learnings` collection exists as a future target for procedural lessons, tool failures, and user corrections. The MVP does not yet implement a mature learning pipeline.

## Consolidation is not complete

Sleep-style abstraction, pruning, strengthening, and reflection are roadmap items.

## Reconsolidation is intentionally absent

Automatically rewriting remembered facts is dangerous. Future reconsolidation should be gated by:

- explicit enablement,
- dry-run preview,
- similarity checks,
- source provenance,
- and user confirmation for important facts.

## Current install path caveat

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

## Performance

Large indexing runs are embedding-bound. Local CPU embedding servers may take several minutes for thousands of chunks.

Use `max_files`, dry-run, and small initial directories before broad indexing.
