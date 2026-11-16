# Automatic capture

Automatic capture turns durable facts said in conversation into cited memories. It is being
built in slices; this page describes what exists today.

| Slice | What it adds | State |
|---|---|---|
| CP1 | Removes the session-hook capture paths that never worked | done |
| CP2 | Read-only `state.db` reader, authorship classifier, cursor, opt-in session-lane filter | done |
| CP2 follow-up | Opt-in legacy/hybrid recall filter for stored turns of harnesses and probes | done |
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

## CP2: read-only reader, authorship classifier, cursor

Module: `qdrant_memory/capture_reader.py`. Nothing here writes to Qdrant, and `state.db` is only
ever opened read-only.

### Reader

- Opens `state.db` through the C' read-only reader (`session_origin._read_retrying`): `mode=ro`,
  or `immutable=1` when no Hermes process has it open (so no `-wal`/`-shm` file is created), one
  retry when a writer holds a lock. A schema without the required columns, a lock that outlasts
  the retry, or an unreadable file is a `CaptureReadError`.
- Rows are the session-recall eligible rows (`session_recall._ELIGIBLE`: role `user` or
  `assistant`, non-blank content). Each becomes a `CaptureRecord`: session id, message id, role,
  session source, classification, kind, the body with envelopes stripped, and the gateway sender
  tag. Records hold message text; receipts and CLI output hold only ids and counts.
- A scan first takes the largest `messages.id` as its snapshot bound, then pages through
  `cursor < id <= bound` in id order, one read transaction per page. Ids are never reused and a
  row is committed before any row with a higher id exists, so paging under a fixed bound neither
  skips nor repeats a row while the host keeps writing. Rows edited after they were read are not
  re-read.
- Optional columns (`display_kind`, `title`, `origin_json`, `user_id`, `chat_type`,
  `parent_session_id`) may be missing on older hosts; they read as NULL.

### Cursor

`$HERMES_HOME/qdrant_memory/capture/cursor.json` (mode 0600): the last message id handed out and
the resolved `state.db` path. Written atomically (temporary file in the same directory, fsync,
rename). A cursor recorded for another `state.db` is refused, and so is a cursor ahead of the
database's largest id (a restored or replaced database), instead of silently skipping rows.

### Authorship classifier

`classify_row()` returns `host` (with the host shape as `kind`), `machine` (with the rule as
`kind`) or `human`, plus the body without envelopes. It is deterministic: no learned model, no
threshold. Decision order:

1. **Role.** `assistant` rows are `machine` (`assistant`) unless their text is host text (step 3)
   or their `display_kind` is `failed_turn` or `hidden` (host, `system_notice`).
2. **Host display kinds.** A `user` row whose `messages.display_kind` is `process_complete`,
   `internal_notification`, `async_delegation_complete` or `model_switch` is host
   (`background_process`, `notification`, `delegation_report`, `system_notice`). `steer` marks a
   message the user typed during a running turn; it is unwrapped and judged like any other row.
3. **Envelopes, then host text.** Stripped: the out-of-band steer wrapper; a leading
   `[Note: model was just switched ...]` line; `[Triggering message id: ...]`; the
   `Gateway message origin (JSON data ...)` block; a `[Recent channel messages]` transcript up to
   `[New message]`; a `[Replying to: "..."]` quote, with its `[Context around the replied-to message]` block up to
   `[New message]`; a host media description (`[The user sent ...~]` and the `vision_analyze`
   hint); the `--- Attached Context ---` / `--- Context Warnings ---` tail; the LCM
   `[Note: This conversation uses Lossless Context Management ...]` tail. A quoted host frame
   inside a reply quote therefore does not make the reply host text. The rest is host text when
   `host_markers.is_host_injected` says so or when it contains one of the extra shapes below; then
   the `[Name] ` sender tag is removed (a bracket label with a colon is never a sender). Nothing
   left: host `envelope_only` (`media_note` when the row was only a media description).
4. **Session source.** A `user` row whose session source is not one a person types into
   (`cli`, `tui`, `desktop`, `gui`, `local`, `acp` and the messaging gateways) is `machine`:
   `one_shot` when the source is the host's one-shot run (`oneshot`: a single scripted prompt,
   such as a test harness or a probe), otherwise `non_interactive_source` (cron, subagent and
   tool runs, API clients, test canaries, and rows whose session row is missing).
5. **Delegation brief.** A `user` row is `machine` `delegation_brief` when all of these hold:
   the session source is a messaging gateway (telegram, discord, slack, whatsapp, signal, matrix,
   mattermost, email, sms); the session has no `parent_session_id`; it carries no gateway identity
   (`user_id`, `chat_type` and `origin_json` all NULL, so no platform ever delivered a message
   into it); it has no title; and the row is the session's first eligible row. Older hosts stored
   a child agent's task that way. Each condition is needed: a person answering inside such a
   session is never its first row, and the titled identity-less sessions are context-management
   continuations whose first row is the person's or the host's.
6. Otherwise `human`.

Features evaluated and not used: the language of the brief (agent briefs are written in more than
one language, and so are people's messages), length (one-line briefs and long pasted messages both
exist), `display_metadata.gateway_input_owner` (only recent gateway rows carry it, so its absence
proves nothing), and `platform_message_id` (same).

Host shapes this module adds without editing `host_markers.py` (to reconcile there):

| kind | shape |
|---|---|
| `agent_loop` | `You've reached the maximum number of tool-calling iterations allowed.` |
| `skill_or_cron_prompt` | `You are running as a scheduled cron job`, `has invoked the "` |
| host by `display_kind` | `process_complete`, `internal_notification`, `async_delegation_complete`, `model_switch`; assistant `failed_turn`, `hidden` |
| envelopes (stripped, not host) | steer wrapper, model-switch note, trigger id, gateway-origin block, recent-channel transcript, reply quote and context block, media description, vision hint, attached-context and LCM tails |

### Session-lane filter (opt-in)

`session_prefetch_exclude_machine: true` (default `false`) makes the automatic-recall session lane
drop a recalled window prompted only by a one-shot run: a test harness's "Reply with exactly: OK",
a smoke probe. The prompting user side is the window's eligible `user` rows that are not host
text; host rows decide nothing, because the lane already cuts host text out of every window and a
person's conversation is full of host notices. When the window holds no such row (a window inside
one long answer, or one whose only user rows are notices), it is the last non-host `user` row
before the window: the prompt its assistant text answers. The window is dropped when that set is
non-empty and every row in it is a machine kind in `capture_reader.LANE_NOISE_KINDS`, which is
`one_shot` alone. A window with no such row anywhere before it is kept
(`capture_reader.window_is_lane_noise`).

Windows prompted by an agent are kept: delegation briefs, subagent, cron, tool and API-client
runs. Those sessions hold findings the user asks about later, so dropping every machine-prompted
window costs real recall. On one install's recall question sets, a rule that dropped every
machine-prompted window lost a question whose answer sat only in a delegated agent's session, and
a rule that also counted host rows as machine text lost more (four hits net). The one-shot rule
loses none there, and drops a few dozen windows of a 66,000-window index.

Dropped windows are listed in `SessionPrefetchLane.last_excluded` (ids only). With the option off
the lane is unchanged.

### Legacy-recall filter (opt-in)

`auto_recall_exclude_machine: true` (default `false`) does the same for the legacy block and the
hybrid mode's dense lane: they leave out a stored conversation-turn point (`memory_kind`
`conversation_turn`, or `source_type` `conversation` with `chunk_type` `turn`) that a test harness
or a probe produced. Module: `qdrant_memory/recall_noise.py`. Query time only: the index is not
rewritten, and nothing is written to Qdrant or to `state.db`.

The rule, in order (`recall_noise.turn_noise_reason`):

1. **Host text** (`host_user_side`): the turn's user side is host text under `classify_row`,
   judged with the session's fields. Today the always-on scaffolding filter already hides almost
   all of these; the classifier adds the shapes `host_markers` does not know (for example the
   tool-iteration limit notice).
2. **Known session.** When `state.db` has the turn's session (payload `session_id`), structure
   decides and the text does not: a `oneshot` session (`ONE_SHOT_SOURCES`, the same kind the
   session-lane filter drops, including a one-shot run's compression continuation) is left out
   (`one_shot_session`); every other session, interactive or agent (a delegated child is a
   `subagent` session; cron, tool, API clients), is kept. A probe typed into a person's
   interactive session is therefore kept.
3. **Session gone.** Sessions leave `state.db` (retention, pruning) while their turn points stay in
   Qdrant: on one
   install 7,260 of 10,355 turn points (70%) name a session its `state.db` no longer holds. Such a
   point (or any point while `state.db` cannot be read) is kept unless its own text proves a
   scripted check (`echo_probe`, `recall_noise.is_echo_probe`): the user side, envelopes
   stripped, is at most 500 characters and demands an exact reply (`exactly`, `exactamente`,
   `verbatim`, `literally`, `literalmente`), and the whole assistant side is one line of at most
   80 characters that occurs in the user side (case-insensitive, without trailing `.`/`!` and
   quotes). "Reply with exactly: OK" answered "OK" is one; a long prompt (a group-chat
   transcript, a brief) or an answer that adds anything is not.

The session's fields come from one read-only query per recall, through the C' reader without
its lock retry (a busy `state.db` means "no structural evidence", never a wait on the turn), for
the session ids not yet cached. Rows found are cached for the process (a session's source and
parent never change); ids with no row are re-checked after ten minutes.

The filter runs before the top-k cut: the provider asks the retriever for
`max(auto_recall_top_k, min(20, search_candidates))` results, which is the same Qdrant request
as without the filter, drops the noise and keeps the first `auto_recall_top_k`. A dropped turn
frees its slot for the next memory, and the kept memories keep their scores and order. RAPTOR
cited leaves and graph relations of the hybrid mode are not filtered (on one install two RAPTOR
leaves are turn points, neither a probe).

Measured on one install (main profile floors: final score 0.05, 12 word characters, read-only):

- 10,355 turn points: 71 newly left out (7 `one_shot_session`, 64 `echo_probe`); 3,920 more are
  `host_user_side`, all of them already hidden by the always-on scaffolding filter.
- The gate sets of C'5 (`scripts/c5_gate_harness.py --legacy-exclude-machine`), legacy arm: 6 of
  51 hits off and on, no question lost or gained, and the injected text was identical for all 51
  (no probe turn ranked among their candidates).
- A probe-shaped message ("Reply with exactly: OK, and tell me about the relay gateway") received
  three stored probe turns (749 characters) with the option off and nothing with it on.

### CLI

```bash
hermes qdrant capture scan [--since-days N] [--json] [--advance]
```

Counts the rows after the cursor (optionally only the last N days) per class, per role and per
kind, and prints the cursor state. Read-only on `state.db`; the cursor moves only with `--advance`
(to the scan's snapshot bound). It never contacts Qdrant or the embedding server.
