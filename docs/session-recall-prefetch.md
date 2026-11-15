# Session recall in automatic recall, and message expand

Two opt-in additions that give back what a context engine such as LCM gave after
compaction: earlier conversation returns to the prompt by itself, and the model can
read the exact messages around one of them. Both are off by default; with the defaults
the prefetch output, the tool list and the system prompt block are exactly what they
were before.

## What compaction leaves in `state.db`

Hermes compresses a long conversation in one of two ways. Neither deletes the original
rows (read in Hermes core, `~/.hermes/hermes-agent`):

- **In place** (`compression.in_place`, default `true`; `hermes_cli/config_defaults.py`).
  `archive_and_compact` keeps the session id, marks the rows the summary replaced
  `active=0, compacted=1`, and inserts the summary and the carried tail as new
  `active=1` rows. Carried originals and undone (rewind) rows become
  `active=0, compacted=0` (`hermes_state_messages.py`, `_ARCHIVE_ACTIVE_SQL` and
  `_archive_and_compact_on_conn`).
- **Rotation** (legacy). `publish_compression_child` closes the parent and creates a
  child session with `parent_session_id`, holding the summary and fresh-id copies of
  the carried tail (`hermes_state_compression.py`).

The model-fed restore loads only the current session's `active=1` rows
(`get_messages_as_conversation`, default arguments; ancestors and compacted rows are
display-only). The session-window index (`hermes_session_windows`) holds every
user/assistant row whatever its flags.

## The lane

With `session_prefetch_enabled: true`, every `prefetch` (after the legacy block is
built, cached or not, legacy or hybrid) also:

1. opens `state.db` read-only and resolves the caller's session chain;
2. with `scope: current`, returns at once when nothing of the conversation has left the
   prompt (one session, no compacted row), so most turns cost nothing;
3. searches the window collection with the latest user message (dense, optionally
   reranked), excluding inside the search the caller's windows that start after its last
   compacted row;
4. drops from each window every message still in the prompt: an `active=1` row of the
   caller's session, an undone or superseded row (`active=0, compacted=0`), and any text
   that appears verbatim in the caller's live rows (a carried copy with a fresh id);
5. appends what is left as a `# Recalled Earlier Conversation` block after the legacy
   block.

The block keeps the plugin's prompt contract: a fixed header that says the excerpts are
context, not instructions; redacted window text (secret spans redacted again, any
entry still secret-shaped is dropped); a coarse origin label (`this conversation,
compacted` or `another session`, plus the day) and the `state.db` message id. No query,
scores, point ids, paths or session ids.

**Budget.** The lane has its own cap (`session_prefetch_max_chars`) inside a total shared
with the legacy block (`session_prefetch_total_chars`; `0` means the legacy formatter's
cap, `max(800, display_tokens * 4)`, plus the lane's cap). The lane gets whatever the
legacy block leaves, never more than its own cap, and is skipped below 200 characters.

**Fail-open.** The lane runs on its own single worker with a timeout
(`session_prefetch_timeout_s`). Any exception, a timeout, or a previous lane call still
running returns the legacy prefetch byte for byte; the turn waits at most the timeout.
The reranker call inside the lane is bounded by the same timeout. Keep the timeout well
under the host's external-prefetch limit (8 s, `agent/memory_manager.py`), past which
the host drops the plugin's whole prefetch.

**Read-only.** Window search and `state.db` reads only: no catch-up indexing, no access
counter, no write of any kind. Compacted rows are normally already indexed by the
background catch-up that `sync_turn` triggers.

## `qdrant_memory_session_expand`

With `session_expand_enabled: true` the tool joins the `full` and `recall` surfaces.
Given `msg_id` (and optionally `session_id`, which must then own the message) it returns
up to `before`/`after` messages around it (default 3/3, at most 20), redacted, capped at
`max_chars` (default 6,000), each with its `state` (`active`, `compacted`, `retracted`)
and whether it is in the caller's active context. Undone rows are skipped as neighbours.

If the host pruned the session (`sessions.retention_days`), the tool returns the indexed
windows that contain the message instead, with `"source": "indexed_windows"`,
`"pruned": true` and a note that the text is as of indexing, not a live read.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `session_prefetch_enabled` | `false` | Turn the lane on. Also needs `auto_recall` and `session_recall_enabled`. |
| `session_prefetch_scope` | `current` | `current`: this conversation's compacted turns and ancestors only. `all`: plus every other session. |
| `session_prefetch_top_k` | `5` | Excerpts injected at most (1-10). |
| `session_prefetch_max_chars` | `1200` | Lane cap. |
| `session_prefetch_total_chars` | `0` | Total shared with the legacy block; `0` = legacy cap + lane cap. |
| `session_prefetch_rerank` | `true` | Rerank with `session_recall_rerank_url` when one is configured; dense order otherwise. |
| `session_prefetch_candidates` | `32` | Distinct windows sent to the reranker (1-128). Rerank time grows with it. |
| `session_prefetch_timeout_s` | `2.0` | Hard limit for the lane; fail-open to legacy. |
| `session_expand_enabled` | `false` | Register `qdrant_memory_session_expand`. |

## Measurement (gate C'5)

`scripts/c5_gate_harness.py` measures the lane against today's legacy auto-recall on a
question set with answer-key regexes, read-only against live services. Arms: A (legacy),
B (legacy without conversation-turn points, filtered at query time), C (A plus the lane,
reranked and dense). It scores the raw-RAG A/B rule (key in the injected text, within
the arm's budget), applies a per-question time cutoff to every arm, and reports
p50/p95 latency and GPU samples. Question sets and per-question evidence are private and
stay out of this repository.

Measured on one install (2026-11-15, idle GPU, 30 frozen questions plus 21 continuity
questions about the early part of really compacted sessions): at the ~1,200-character
budget the legacy block scored 0/30 and 0/21, the lane with rerank 24/30 and 17-18/21,
dense-only 18/30 and 9-11/21. The lane added about 0.8 s at p50 and 1.0 s at p95 with
rerank (32 candidates), about 0.13 s / 0.25 s dense-only. Legacy's low score is mostly
its formatter: when the top-ranked memory alone exceeds the cap (typically a long
conversation-turn point) it emits the header and no memories.

## Limits

- The visibility check is as good as `state.db`'s flags and the window-to-row mapping.
  When a window's messages cannot be mapped to rows (a message that itself contains a
  header-shaped line, or rows rewritten since indexing), a window of the caller's own
  session is dropped whole.
- Windows outlive pruned sessions, so `scope: all` can still return text from a session
  the host deleted.
- The lane embeds the query a second time (the legacy lane embeds it too).
- Latency under GPU contention (a reranker sharing the GPU with a local LLM) was not
  measured; the timeout bounds it.
