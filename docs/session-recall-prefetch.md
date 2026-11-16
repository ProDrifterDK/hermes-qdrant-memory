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
   host scaffolding was already cut out by the window search (see "Host scaffolding is
   not conversation");
5. leaves out excerpts below the relevance floor, when one is set (see "Relevance
   floor");
6. appends what is left as a `# Recalled Earlier Conversation` block after the legacy
   block. With nothing left there is no block, not a header.

When `auto_recall_min_word_chars` is set (default off), messages with fewer letters/digits
skip automatic recall altogether: the legacy block, the hybrid mode and the lane. A
host-generated incoming message always does, and a host wrapper around real words is
taken off before any search (see "Host-generated incoming messages").

The block keeps the plugin's prompt contract: a fixed header that says the excerpts are
context, not instructions; redacted window text (secret spans redacted again, any
entry still secret-shaped is dropped); a coarse origin label (`this conversation,
compacted` or `another session`, plus the day) and the `state.db` message id. No query,
scores, point ids, paths or session ids.

**Budget.** The lane has its own cap (`session_prefetch_max_chars`) inside a total shared
with the legacy block (`session_prefetch_total_chars`; `0` means the legacy formatter's
cap, `max(800, display_tokens * 4)`, plus the lane's cap). The lane gets whatever the
legacy block leaves, never more than its own cap, and is skipped below 200 characters.
The legacy and hybrid blocks never exceed that cap and are never a bare header: when the
top-ranked memory is larger than what is left, it is cut to the remaining budget and
ends with `… [truncated]`; with nothing to show they return an empty string.

**Degrade and fail-open.** Two levels, ratified on 2026-11-15:

- *Only the reranker fails* (connection refused, an HTTP error, a malformed response,
  or no answer in time): the lane keeps the dense order of the window search and still
  injects its block.
- *The lane itself fails*: any exception, a timeout of the lane, or a previous lane call
  still running returns the legacy prefetch byte for byte. The lane runs on its own
  single worker with `session_prefetch_timeout_s`, so the turn waits at most that long.

**Rerank deadline.** The turn and the lane's worker share one deadline,
`session_prefetch_timeout_s` after the lane starts. The rerank call gets whatever is left
of it when it starts, minus a reserve of 0.3 s (at most a quarter of the timeout) for
the `state.db` reads, the formatting and the hand-off, and is not started at all with
less than 0.05 s left. `session_recall_rerank_timeout_s` stays an upper bound. So a
reranker that is slow or asleep ends in dense order that reaches the turn, and the worker
is done before the turn stops waiting, so the next turn does not skip the lane. A
reranker started with `--sleep-idle-seconds` takes a few seconds to answer the first
request after a sleep: that turn and any turn during the wake get dense order, later
turns are reranked again. When the reranker answers in time nothing changes.

The deadline is derived instead of being a separate setting because the time the
embedding and the dense search take varies (a cold embedder too): a fixed rerank timeout
would either waste budget or overrun it. The reranker's HTTP timeout is a socket timeout
(connect, then each read), which bounds a llama.cpp server that sends its answer in one
piece; a server that trickled bytes could keep the worker longer.

Keep the timeout well under the host's external-prefetch limit (8 s,
`agent/memory_manager.py`), past which the host drops the plugin's whole prefetch.

**Read-only.** Window search and `state.db` reads only: no catch-up indexing, no access
counter, no write of any kind. Compacted rows are normally already indexed by the
background catch-up that `sync_turn` triggers.

## Host scaffolding is not conversation

Hermes stores some of its own text as ordinary `user` (sometimes `assistant`) rows of
`messages.content`. None of it was said by a person or the assistant, and much of it is
instruction-shaped, so recall that replays it into another conversation invites the
model to resume foreign work (the post-compaction drift this guards against). Read in
Hermes core (`~/.hermes/hermes-agent` at `9431f2e448`, 2026-11-15):

| Marker | Where the host writes it |
|---|---|
| `[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted …` (`SUMMARY_PREFIX`), `## Historical Task Snapshot` | `agent/context_compressor.py:269`, `:259` |
| `[CONTEXT SUMMARY]:` (legacy prefix) | `agent/context_compressor.py:303` |
| `[PRIOR CONTEXT — for reference only; not a new message]`, `[END OF PRIOR CONTEXT — COMPACTION SUMMARY BELOW]`, `--- END OF CONTEXT SUMMARY …` (summary merged into a tail message) | `agent/context_compressor.py:516`, `:517`, `:512` |
| `[STILL IN PROGRESS — this is the active request, restated after the …` | `agent/context_compressor.py:528` |
| core's own list of synthetic user rows (`[System:`, `[CONTEXT`, `[PRIOR CONTEXT`, `[IMPORTANT: Background`, `[Your active task list`, `[Planning state preserved`, `[ASYNC DELEGATION`, `[OUT-OF-BAND`, `Cronjob Response:`) | `agent/context_compressor.py:938` |
| control-frame openers (`System note:`, `Runtime note:`, `SYSTEM]`, `IMPORTANT:`, …) and machine prefixes | `agent/prompt_builder.py:572`, `agent/title_generator.py:159` |
| `[INTERNAL NOTIFICATION — generated by the Hermes gateway, not written by a human user. …]`, appended to notification text | `gateway/run_notifications.py:93` |
| `[IMPORTANT: Background process … completed …]`, `[Background process … heartbeat …]`, `[ASYNC DELEGATION (BATCH) COMPLETE — …]` | `tools/process_registry_notifications.py:586`, `:561`, `:374`, `:296` |
| `[OUT-OF-BAND USER MESSAGE — a direct message from the user, …]` (mid-turn steer frame) | `agent/prompt_builder.py:562` |
| `<memory-context>` + `[System note: The following is recalled memory context, …]` | `agent/memory_context.py:193` |
| `[Your active task list was preserved across context compression]` | `tools/todo_tool.py:21` |
| `[IMPORTANT: The user has invoked the "…" skill …]` (skill body), `[IMPORTANT: You are running as a scheduled cron job. …]`, `[Cron delivery: …]`, `Cronjob Response: …` | `agent/skill_commands.py:50`, `cron/scheduler_prompt.py:271`, `cron/scheduler_delivery.py:209`, `:1772` |
| `[System: The active model for this chat has changed to …`, `[System: This conversation is now being answered on a different interface: …`, `[System: Your previous response was truncated …` | `tui_gateway/server.py:1145`, `agent/surface_switch.py:22`, `agent/conversation_loop.py:996` |
| `[The user sent an image~ Here's what I can see …]`, `[The user sent a text document: …]` (host descriptions of attachments) | `gateway/run_inbound.py:1947`, `gateway/run.py:2507` |
| `[The user reacted … to …]`, `[Recent channel messages]`, `[Continuing toward your standing goal]`, `[Note: …]` | `tui_gateway/methods_prompt.py:230`, `gateway/relay/ws_transport.py:135`, `gateway/run_busy.py:310`, `tools/tts_streaming.py:50` |
| `[Recent/Session Arc/Durable/Depth-N Summary (dN, node …)]`, `[Current user objective preserved from compacted history]` | hermes-lcm plugin `engine.py:6119`, `reconcile.py:52` |

Two host tags wrap real user text and are looked through, not dropped: the shared-session
sender tag `[Name] ` (`gateway/session.py:698`) and the Discord routing note
`[Triggering message id: …]` (`gateway/run_inbound.py:59`). A notification behind a sender
tag (`[Name] [IMPORTANT: Background process …`) is still scaffolding.

`qdrant_memory/host_markers.py` decides per message: an opener at the start of the row
(after those tags; every stripping stage is checked, because a bracketed label has the
shape of a sender tag), or one of the host's full frame sentences anywhere in it
(appended footers, merged summaries, steer and memory frames). A real message that only
quotes a bare label, as when discussing this code, is kept. Over every user and assistant
row of one install's `state.db` it flagged 5,968 of 39,448 rows; no remaining row starting
with `[` or `<` was host text.

**Query time, not index time.** `hermes_session_windows` already holds such windows (one
install: 10,227 of 65,994 windows have a scaffolding segment recognizable from text alone),
and the index is a faithful copy of `state.db` that the expand fallback relies on;
filtering at index time would also change the serialization and force a full re-index.
So the window search (`SessionRecall.search`, shared by `qdrant_memory_session_recall`
and the lane) cuts scaffolding out of every dense hit before dedup and rerank see it:

- each segment of a window (split at message headers) is attributed to its `state.db` row
  (the window's eligible rows in id order) and dropped when the row is scaffolding. This
  is what catches a window that starts in the middle of a summary: 64,709 of 65,994 windows
  start mid-message, with no header to recognize;
- the session's rows are gone (pruned): only segments that start with a header are kept,
  each judged by its own text (fail-closed for the unattributable leading part);
- the segments do not line up with the rows and one row is scaffolding: the window is
  dropped;
- no readable `state.db` at all: every segment is judged by its own text.

A window left with no conversation is dropped; `host_text_dropped` in the tool output
counts them. The lane then maps the cleaned text onto the non-scaffolding rows, so its
visibility check is unchanged.

The legacy collection has the same problem: the turn writer stored whatever the host
passed as the user message (one install: 3,983 of 17,727 points, all
`conversation`/`turn`). Recall (automatic and tool) leaves out a conversation-turn point
whose user or assistant side is scaffolding; documents and manual facts are not judged.
`qdrant_memory_session_expand` skips scaffolding rows as neighbours and withholds the text
of a scaffolding target (`host_scaffolding: true`).

A row is judged as a whole, so two kinds of row that carry some user text are dropped
with it: attachment notes (`[The user sent an image~ …]`, the user's caption included) and
mid-turn steer rows (`[OUT-OF-BAND USER MESSAGE …] … [/OUT-OF-BAND …]`), whose user words
are lost to recall. The rows stay in `state.db` and in the index; only recall leaves
them out.

The filter has no switch: it applies with every configuration, floors on or off.

## Host-generated incoming messages

The filter above judges what recall returns. The query matters too: automatic recall
searches with the incoming message, and when the host generated that message (a
background-process report, a cron run) the search finds whatever shares its words and
injects it. Read in Hermes core (`~/.hermes/hermes-agent` at `4162ecf11f`, 2026-11-15):

- **What the plugin receives.** `prefetch` gets the turn's persisted text, not the
  model-facing one: `original_user_message = persist_user_message or user_message`
  (`agent/turn_context.py:1294`), flattened (`:1016`), skipped by core only when trivial
  (`:1081`, `agent/memory_provider.py:75`); `MemoryManager.prefetch_all` takes off a skill
  body and redacts (`agent/memory_manager.py:361`). After the turn the same text goes to
  `queue_prefetch` (`run_agent.py:931`, `agent/memory_manager.py:439`), whose block the
  plugin hands to the next `prefetch`. The gateway passes the persisted text
  (`gateway/run_turn_runner.py:1086`); it has the timestamp and the Discord routing note
  taken off (`gateway/run_turn_prepare.py:747`), but keeps the shared-session sender tag
  (`gateway/session.py:698`) and the attachment notes.
- **Wholly host-generated turns.** A process completion, watch match or heartbeat
  (`tools/process_registry_notifications.py:586`, `:568`, `:561`) or a delegation report
  (`:374`, `:296`) arrives as a user turn with `INTERNAL_NOTIFICATION_FOOTER` appended
  (`gateway/run_notifications.py:93`, `:797`, `:937`), optionally behind the sender tag;
  completions that queue up are folded under `[IMPORTANT: N background process results …]`
  (`gateway/session_automation.py:150`). A cron run's prompt is `_CRON_HINT` (`[IMPORTANT:
  You are running as a scheduled cron job. …]`) plus `[CURRENT RUN TIME: …]` before the job
  prompt (`cron/scheduler_prompt.py:270`, `:307`, `:363`), with memory loaded
  (`cron/scheduler.py:2508`). A `/goal` continuation is `[Continuing toward your standing
  goal]\nGoal: …` (`hermes_cli/goals.py:61`). The auto-resume turn after a restart persists
  its `[System note: The previous turn was interrupted by …]` (`gateway/run.py:1036`, `:1079`).
- **Wrappers around real words.** The sender tag `[Name] ` (`gateway/session.py:698`); the
  Discord note `[Triggering message id: …]` (`gateway/run_inbound.py:56`; current core strips
  it from the persisted text, older rows carry it); attachment notes before the caption:
  `[The user sent an image~ …]` plus its `[If you need a closer look …]` line, `[The user sent
  an audio file attachment / a video attachment / a text document / a voice message …]`
  (`gateway/run_inbound.py:1628`, `:1947`, `gateway/run.py:2507`,
  `gateway/run_inbound_transcription.py:169`); the steer frame `[OUT-OF-BAND USER MESSAGE …]
  … [/OUT-OF-BAND USER MESSAGE]` (`agent/prompt_builder.py:562`). In current core a steer row
  is placed mid-turn (`agent/prompt_builder.py:586`) and a late one becomes the next turn as
  plain text (`gateway/run_busy.py:224`), so the steer frame is handled defensively.
- **Not seen by the plugin.** Notes the gateway puts only in the model-facing message:
  recovery and model-switch notes (`gateway/run_turn_runner.py:958`, `:1018`), the Telegram
  observed-context wrapper (`gateway/run.py:1343`, persisted pre-wrap). Compaction handoffs
  are history rows, not an incoming message.

**The rule** (`host_markers.recall_query`, applied by `prefetch` and `queue_prefetch` before
anything else, with every configuration):

1. Take off, one at a time from the start, the wrappers above: sender tag, Discord note,
   attachment notes (the whole bracket frame), steer frame (the frame and its closing marker).
2. If the message, at any of those stages, starts with any other host opener
   (`is_host_injected`'s list), it is host-generated: no recall. A frame that never closes
   counts as host-generated too (fail closed).
3. If what is left is empty or is host text by `is_host_injected` (for example a footer
   anywhere in it), no recall.
4. Otherwise recall (legacy or hybrid, the lane, and the shadow comparison) runs on what is
   left. `auto_recall_min_word_chars` is judged on it.

No recall means `prefetch` returns `""` and nothing is embedded or searched, and
`queue_prefetch` caches nothing, so the next turn's block is never one computed from host
text. A cached block from an earlier real turn stays for the next real turn, as with the
short-message skip.

Consequences: a cron run gets no automatic recall (the job prompt follows a host frame; the
memory tools still work), nor does a `/goal` continuation or an image sent without a
caption. A real message that starts with a short bracketed label (`[WIP] …`) is searched
without the label, which has the shape of a sender tag. A reply pointer (`[Replying to …]`)
is kept, since the quoted message is what the user is asking about.

## Relevance floor

With nothing relevant to recall, automatic recall injects its top five anyway: a
confirmation such as `ok` or `lanzalo` receives legacy fragments a few words long and lane
windows that share one word. Optional checks let it inject nothing instead. All are off
by default, so an install that sets none of them recalls as before, apart from the
host-scaffolding filter above, which always applies:

| Key | Default | Applies to | Signal |
|---|---|---|---|
| `auto_recall_min_word_chars` | `off` | legacy, hybrid and lane | Messages with fewer letters/digits (`\w`, any script) skip automatic recall. |
| `auto_recall_min_final_score` | `off` | legacy block | Ranked score (min-max normalized similarity x importance x recency x provenance). |
| `auto_recall_min_raw_score` | `off` | legacy block | Raw cosine. Not recommended: short messages score *higher* than real questions (below). |
| `session_prefetch_min_rerank_score` | `off` | lane, reranked | Cross-encoder score. |
| `session_prefetch_min_dense_score` | `off` | lane, dense order | Cosine, when there is no reranker or it failed. |

`off`, `""` and `null` disable each of these keys, as does an unreadable value. On
`auto_recall_min_word_chars` and the two legacy floors `0` disables too (those scores are
never negative); on the two lane floors `0` is a real threshold, because rerank logits and
cosine similarity can be negative.

Recommended values for bge-m3 with bge-reranker-v2-m3 (measured below). They must be
re-derived for any other embedding or reranker model:

    qdrant_memory:
      auto_recall_min_final_score: 0.05   # 0.1 was the measured pick; see the decay note
      auto_recall_min_word_chars: 12
      session_prefetch_min_rerank_score: -1.5
      session_prefetch_min_dense_score: 0.55

A reranked excerpt is judged only by the rerank floor and a dense-ordered one only by the
dense floor: they are on different scales. The floors cut a suffix of the rank order, and a
block with nothing left is `""`. Tool searches are not affected (`min_raw_score` and
`min_final_score` remain the tool-search filters).

The lane values belong to bge-m3 cosine and bge-reranker-v2-m3 logits. A reranker that
returns probabilities never scores below -1.5, so that floor would be inert there; another
embedding model can sit on a different cosine range, where 0.55 can empty the lane.
Re-derive them with the harness (below). The legacy final score includes recency
(`decay_rate` 0.001/h halves it in about 29 days), so with a floor of 0.1 an old memory of
ordinary importance (5) never reaches automatic recall after about 67 days, even as a
perfect match; at 0.05 that limit moves to about 96 days, and 0.05 kept one more gate hit
(table below). One CJK character carries about a word, so CJK-heavy profiles want a lower
`auto_recall_min_word_chars`.

**Measured** (one install, 2026-11-15, live budget `display_tokens: 700`, rerank on; the
gate sets of C'5 plus a frozen noise set of 41 real short confirmations and status pings
in Spanish and English, each searched with its own session and timestamp):

- Top-1 scores of the noise set against the gate sets: legacy raw cosine p50 0.686 vs
  0.58 (inverted, hence no raw floor), legacy final p50 0.007 vs 0.077, lane rerank p50 1.45
  vs 3.2-3.7, lane dense p50 0.54 vs 0.60-0.62. Only the legacy final score separates well.
- The floors alone barely touch the lane's noise before they cost gate hits (rerank 0.5
  already loses 3). What separates is the message: every real user turn of that install
  under 12 word characters (889 turns, 16.8 % of user turns, 179 distinct texts, all read)
  is a confirmation, a choice, a short command about the live conversation, a ping or
  transcription noise; at 16 the cut starts to reach content questions.

| Setting | Gate hits A / A+lane rr / A+lane dense (of 51) | Noise turns injected (of 41) | Noise characters (A+lane rr) |
|---|---|---|---|
| before (main) | 7 / 42 / 32 | 41 | 103,974 |
| host filter only | 8 / 43 / 33 | 41 | 103,773 |
| + final floor 0.1 | 6 / 43 / 32 | 41 | 49,370 |
| + skip under 12 word chars | 6 / 43 / 32 | 12 | 14,305 |
| + lane floors -1.5 / 0.55 | 6 / 43 / 31 | 9 | 9,619 |
| final floor 0.05, skip under 12, lane floors -1.0 / 0.55 | 7 / 42 / 31 | 9 | 12,228 |

With the skip on and the final floor at 0.1, every rerank floor from -1.5 to 0.0 gave the
same noise. Every arm stays within one hit of the previous baseline; injected host scaffolding goes
from 26 entries to 0. `scripts/c5_gate_harness.py --noise … --grid …` reproduces the
trade-off and recomputes it offline from a recorded run (`--analyze-only`), with the
plugin's own floor and formatting functions; on the noise set that simulation matched the
real `prefetch` output for 41 of 41 messages.

Not solved by a floor: a short status ping that repeats an earlier one word for word (the
cross-encoder rates an exact phrase match high) can still recall that earlier exchange.

## `qdrant_memory_session_expand`

With `session_expand_enabled: true` the tool joins the `full` and `recall` surfaces.
Given `msg_id` (and optionally `session_id`, which must then own the message) it returns
up to `before`/`after` messages around it (default 3/3, at most 20), redacted, capped at
`max_chars` (default 6,000), each with its `state` (`active`, `compacted`, `retracted`)
and whether it is in the caller's active context. Undone rows are skipped as neighbours.

If the host pruned the session (`sessions.retention_days`), the tool returns the indexed
windows that contain the message instead, with `"source": "indexed_windows"`,
`"pruned": true` and a note that the text is as of indexing, not a live read.

**Message ids are capability handles** (ratified on 2026-11-15). Without `session_id`
the tool resolves any `state.db` message id in any session, not only the caller's
conversation or the ids the lane injected. `state.db` ids are sequential integers, so
holding (or guessing) an id is enough to read the messages around it. Every message
text is redacted (`redact_secret_spans`) before it is returned, as in the lane. Enable
the tool only on profiles that may read the whole local conversation history.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `session_prefetch_enabled` | `false` | Turn the lane on. Also needs `auto_recall` and `session_recall_enabled`. |
| `session_prefetch_scope` | `current` | `current`: this conversation's compacted turns and ancestors only. `all`: plus every other session. |
| `session_prefetch_top_k` | `5` | Excerpts injected at most (1-10). |
| `session_prefetch_max_chars` | `1200` | Lane cap. |
| `session_prefetch_total_chars` | `0` | Total shared with the legacy block; `0` = legacy cap + lane cap. |
| `session_prefetch_rerank` | `true` | Rerank with `session_recall_rerank_url` when one is configured. Dense order when none is configured or when only the reranker call fails or does not answer before the rerank deadline; a lane timeout returns legacy unchanged (see "Degrade and fail-open"). |
| `session_prefetch_candidates` | `32` | Distinct windows sent to the reranker (1-128). Rerank time grows with it. |
| `session_prefetch_timeout_s` | `2.0` | Hard limit for the lane; fail-open to legacy. |
| `session_prefetch_min_rerank_score` | `off` | Lane floor for reranked excerpts (see "Relevance floor"; model-specific). |
| `session_prefetch_min_dense_score` | `off` | Lane floor for dense-ordered excerpts (model-specific). |
| `auto_recall_min_final_score` | `off` | Legacy block floor on the ranked score. |
| `auto_recall_min_raw_score` | `off` | Legacy block floor on raw cosine. |
| `auto_recall_min_word_chars` | `off` | Shorter messages skip automatic recall (legacy, hybrid and lane). |
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
rerank (32 candidates), about 0.13 s / 0.25 s dense-only. Legacy's low score was mostly
its formatter: when the top-ranked memory alone exceeded the cap (typically a long
conversation-turn point) it emitted the header and no memories (42 of 51 questions).

Re-measured after the formatter fix (same sets and budgets, idle GPU): no legacy block
is header-only any more (0 of 51 at both 1,200 and 2,800 characters). Legacy scored
4/51 at 1,200 characters and 7/51 at 2,800; legacy plus the lane scored 42/51 with
rerank and 30/51 dense-only at 1,200 characters, and lost no question that legacy
answered. Cutting the oversized top memory beat skipping it (4/51 against 1/51 at 1,200
characters, where skipping left 26 of 51 blocks empty).

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
