# Automatic capture

Automatic capture turns durable facts said in conversation into cited memories. It is being
built in slices; this page describes what exists today.

| Slice | What it adds | State |
|---|---|---|
| CP1 | Removes the session-hook capture paths that never worked | done |
| CP2 | Read-only `state.db` reader, authorship classifier, cursor | see below |
| later | Mirror of built-in memory writes, persisted review queue, watcher drain | not built |

Nothing in this page writes to Qdrant or to `state.db`.

## CP1: what was removed and what stays

Removed:

- The learning heuristic (`lesson_extractor.extract_learning_candidates_*`: the "actually / no /
  correction / I prefer" and tool-failure regexes) and the source-line extractor's hook producer
  (`source_extraction.extract_source_candidates_from_messages`), which `on_pre_compress` and
  `on_session_end` ran over every transcript they saw.
- The in-memory candidate queues they filled (`_pending_learning_candidates`,
  `_pending_extraction_candidates`).
- The tools that only read those queues: `qdrant_learning_preview`, `qdrant_learning_approve`,
  `qdrant_memory_extraction_preview`, `qdrant_memory_extraction_approve`, and the CLI commands
  `hermes qdrant learning preview` and `hermes qdrant learning approve`.
- The `pending_session` scope of `qdrant_memory_improve_preview` (same queue).

Why: on a labelled sample of real conversations the source extractor matched no message, the
learning heuristic fired on the Spanish word "no" and on host warnings, the hooks never run for
sessions the host reaps (more than half of the human messages), and the queue died with the
process. No candidate was ever approved.

`on_pre_compress` returns `""` and `on_session_end` does nothing.

Kept, as the backend of the persisted queue a later slice adds:

- the candidate types (`ExtractionCandidate`, `LearningCandidate`);
- the write gate (`evaluate_extraction_candidate_write`, `evaluate_source_extraction_candidate`,
  the learning write gate);
- the extraction approval write path, now `QdrantMemoryProvider.approve_extraction_candidate`
  (gate on the candidate and on the persisted payload, lineage overwrite refusal, locked upsert
  that merges `session_origin`), and the learning approval write path,
  `QdrantMemoryProvider.approve_learning_candidate`;
- `extract_source_candidates_from_text`, because `qdrant_memory_improve_preview` builds its
  candidates with it.

Deprecated settings: `learning_auto_extract_*` and `source_extraction_*` are still accepted
(config file, `qdrant_memory.json`, `HERMES_QDRANT_MEMORY_*`), have no effect, and the first load
that sees one logs a single warning for the process.
