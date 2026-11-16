# Automatic capture

Automatic capture turns durable facts said in conversation into cited memories. It is being
built in slices; this page describes what exists today.

| Slice | What it adds | State |
|---|---|---|
| CP1 | Removes the session-hook capture paths that never worked | done |
| CP2 | Read-only `state.db` reader, authorship classifier, cursor, opt-in session-lane filter | done |
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

### CLI

```bash
hermes qdrant capture scan [--since-days N] [--json] [--advance]
```

Counts the rows after the cursor (optionally only the last N days) per class, per role and per
kind, and prints the cursor state. Read-only on `state.db`; the cursor moves only with `--advance`
(to the scan's snapshot bound). It never contacts Qdrant or the embedding server.
