# Automatic capture

Automatic capture turns durable facts said in conversation into cited memories. It is being
built in slices; this page describes what exists today.

| Slice | What it adds | State |
|---|---|---|
| CP1 | Removes the session-hook capture paths that never worked | done |
| CP2 | Read-only `state.db` reader, authorship classifier, cursor, opt-in session-lane filter | done |
| CP4 | Persisted review queue, approval routes, nightly `capture run` with the heuristic detector, digest | done (review only; the auto route needs a second signal that does not exist yet) |
| CP5 | Mirror of the agent's built-in `memory` tool writes | done |
| CP2 follow-up | Opt-in legacy/hybrid recall filter for stored turns of harnesses and probes | done |
| later | Local-model extractor (second signal), watcher drain | not built |

Capture is off by default (`capture_mode: off`): no capture tool is registered, `capture run`
refuses, and nothing below touches Qdrant or the queue. Capture never writes `state.db`.

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

## CP4 and CP5: the capture pipeline

### Settings

| Setting | Default | Meaning |
|---|---|---|
| `capture_mode` | `off` | `off`: no capture tools, `capture run` refuses. `dry_run`: runs only report. `on`: `capture run --apply` queues, writes the auto and mirror routes and advances the cursor. |
| `capture_owner_senders` | `[]` | Gateway sender tags (the `[Name] ` prefix the gateway puts on messages in shared channels) that are the owner. Empty: no row of a shared channel qualifies. A row without a sender tag in a one-to-one conversation (DM, CLI, desktop, TUI) is the owner's. |
| `capture_review_expiry_days` | `21` | A review item nobody decided expires after this many days: an `expired` event, no write. |
| `capture_mirror_builtin_memory` | `false` | Lane C: mirror the agent's built-in `memory` tool writes (part of `capture run`). |
| `capture_auto_apply` | `false` | Allow the auto route of lane A (see "Auto route"). |
| `capture_daily_auto_cap` | `5` | At most this many auto-applied facts per local day. |
| `capture_dedupe_threshold` | `0.92` | Cosine at or above which a candidate is a near duplicate of a stored capture. |
| `capture_digest_language` | `en` | `en` or `es`. |

A bare `on`/`off` in YAML is read as a boolean; both spellings work for `capture_mode`.

### The queue

`$HERMES_HOME/qdrant_memory/capture/queue.jsonl` (mode 0600) is an append-only file of events, one
JSON object per line, appended under an exclusive lock and fsynced. The state of an item is the fold
of its events, so the nightly run, the CLI, a gateway provider and a fresh process after a restart all
see the same queue.

| Event | Written when | Holds |
|---|---|---|
| `queued` | a candidate enters the queue, with its route (`review`, `auto`, `mirror`) | the fact text, the pinned citation, the write it would make, rule ids, run id |
| `approved` / `rejected` | the owner decides a review item | actor, approving session and platform |
| `applied` | a write happened | point id, route, rule id, citation and fact digests, run id |
| `expired` | a review item passed its expiry | rule id, run id |
| `merged` | a duplicate's citation was added to a queued item or a stored point | the target id and the citation (ids and digests) |

Only `queued` events carry text. Every other event, and every run report
(`capture/runs/<run_id>.json`), holds ids, digests, rule ids and run ids, never message text.
The candidate id is `sha256(session_id, msg_id, kind, normalized fact)`; its first 12 hex
characters are the short id the digest and the review list show, and either form is accepted.

### `hermes qdrant capture run [--dry-run | --apply] [--since-days N] [--json]`

One batch over the rows after the capture cursor (optionally only the last N days):

1. **Lane A, heuristic.** User rows the classifier marks `human` and that the owner wrote
   (`capture_owner_senders`). A detector of generic Spanish and English signal forms (asking to
   remember or record something, never/always, preferences, corrections, "from now on", scope
   decisions, must/must not) runs on the body after code, quoted lines and links are removed. It
   has no thresholds and no confidence: a hit is a hit, and it cannot rewrite a message into a
   standalone fact, so the fact is the message itself (one line, 3 to 1,000 characters). A message
   with a secret is dropped before anything else.
2. **Citation.** `resolve_span` on the exact row (basis `extractor_source_line`); anything but a
   resolved span drops the candidate with a reason.
3. **Gate.** The extraction write gate (`evaluate_source_extraction_candidate`); `reject` and
   `skip` drop.
4. **Dedupe**, in order: the same candidate id already queued (any state); the same normalized
   fact pending in the queue (the citation is merged into that item); the exact point already
   stored; a dense near match among stored captures (`capture_lane` set, not superseded) at or
   above `capture_dedupe_threshold` (the citation is merged into that point). A row the host
   re-copied during compaction (same `message_uid`) is neither a new item nor a new citation.
5. **Route**: `review`, or `auto` when the auto rule holds.
6. **Lane C**, when enabled (next section).

A dry run (the default) only reads (Qdrant search included) and writes its run report.
`--apply` needs `capture_mode: on`: it expires overdue review items, appends the queue events,
performs the auto and mirror writes and advances the cursor to the run's snapshot bound. Only one
`--apply` run at a time (a lock under `capture/`).

#### Auto route

Behind `capture_auto_apply` (default `false`), lane A uses the design's rule. All must hold: the
heuristic fired, a second signal says the statement is durable with confidence at least 0.9, the
kind is preference, invariant, correction or decision, the second signal does not scope it to the
task and saw no reversal in the next messages, the item is not identity, money or security, the
write gate says `store`, and the day has fewer than `capture_daily_auto_cap` auto-applied items.
The second signal is the local-model extractor of a later slice; **no second signal ships yet, so
with `capture_auto_apply: true` nothing auto-applies today** and every hit goes to review.

### Lane C: mirror of the built-in `memory` tool

`capture_mirror_builtin_memory: true` makes `capture run` read the assistant rows whose
`tool_calls` call the built-in `memory` tool, in every argument shape the host accepted: an
`operations` batch (with or without a top-level `target`; an op may carry its own), and the
single-op shapes with `action`, `content`, `old_text` and `new_text` (`new_text` stands in for an
empty `content`, as in the host). A call whose arguments do not parse still counts as one op, so
the op count equals what the agent sent. The run report's `mirror` block counts calls, ops,
shapes, actions and outcomes.

- Only calls the tool accepted are mirrored (`success: true`; a persisted oversized result is read
  from its preview). A batch is atomic: a refused batch mirrors nothing.
- The whole entry text comes from the tool result when the host returned it (the replaced,
  removed or patched entry, before and after). Otherwise from an entry the mirror already knows
  that contains `old_text` exactly once. Otherwise the op is dropped as `entry_text_unknown`; a
  patch is never stored as its bare span.
- `target=memory` add, replace and patch results are stored at once through the manual store's
  write step (producer `manual_store`, basis `agent_proposed`): the agent decided to remember them.
- The old text of a replace, patch or remove is stored as a capture with
  `fact_status: superseded`, cited to the turn that removed it. Nothing is deleted. A replace
  leaves two points: the new entry and the superseded old one.
- `target=user` (identity) ops, new and superseded, go to the review queue.
- The citation runs from the last human row before the call to the last eligible row at or before
  the calling row (a row that only carries tool calls has no text). In a compaction continuation
  the person's message stayed in the parent session; the span then starts as far back in the
  calling session as one span may reach (200 rows). A call the host copied into a later
  continuation (same call id and arguments) is mirrored once.
- The secret and manual write gates apply.

### Approving and rejecting

```bash
hermes qdrant capture review [--limit 10] [--offset 0] [--json]
hermes qdrant capture approve <id> [--fact "one sentence"] [--no-dry-run --approve] [--json]
hermes qdrant capture reject <id> [--no-dry-run --approve] [--json]
```

The tools `qdrant_capture_review` (pending items, paginated), `qdrant_capture_approve` and
`qdrant_capture_reject` (exact id, `dry_run` defaults to true, a live call needs `approve: true`
and a dry run of the same id in the same process) do the same in chat. They exist only on the
`full` tool surface and only while `capture_mode` is not `off`.

**Distilled fact.** A heuristic item queues the owner's whole message. Approval takes an optional
`fact` (tool argument, CLI `--fact TEXT`): a rewritten one-sentence statement of what holds,
written by the agent (the tool description asks it to propose one and show it) or by the owner.
Whitespace is collapsed to one line; at most 500 characters. The rest does not change:

- the citation is still the queued row and is re-verified as above;
- the write gate (extraction gate for heuristic items, manual gate for mirror items) runs on the
  new text, so a secret in it is refused;
- the dry run binds the text: a live call with another text, or with none, needs its own dry run;
- the stored point carries `capture_fact_edited: true` and `capture_original_fact_sha256`, and the
  `approved` and `applied` events carry `fact_edited: true` and `original_fact_sha256` (the
  digest of the queued text, never the text); `applied.fact_sha256` is the stored text's digest.

Mirror items are already sentences; `fact` is optional there too.

- Approval re-verifies the pinned citation against `state.db` and refuses when the row changed
  (`drifted`) or is gone (`expired`, `partially_missing`).
- It writes only through an existing gated path: `approve_extraction_candidate` for heuristic
  items, the manual store's write step for mirror items. The point's `session_origin` names the
  captured session (`captured_session_id`, and the legacy `session_id` follows it); the approving
  session appears only as `approved_session_id`.
- An item is decided once: approved, rejected, expired or applied items refuse further decisions.
- **Cron sessions never decide** (decision P7): a provider initialized with `platform="cron"` or
  `agent_context="cron"`, or running with `HERMES_CRON_SESSION` set, can list but every approve or
  reject is refused.
- To undo a stored capture: `hermes qdrant forget <point_id> --no-dry-run --approve`.

### `hermes qdrant capture digest [--max-items 10] [--language en|es]`

A short plain-text summary for a no-agent Hermes cron job that delivers stdout to a chat: counts
per route and decision since the last digest, up to 10 pending review items as one-line facts with
their short ids (oldest first), items expiring within three days, facts the auto route stored with
the point id to forget, and the commands and chat phrasing to approve or reject. When nothing
happened since the last digest it prints nothing, so the job delivers nothing. It writes only its
own marker, `capture/digest.json`.

### Turning it on

1. Set `capture_owner_senders` if the owner writes in shared channels (otherwise only one-to-one
   conversations count).
2. `capture_mode: dry_run`, then `hermes qdrant capture run --since-days 7` and read the report.
3. Decide where capture starts. With no cursor yet, the first `--apply` reads every row
   `state.db` still holds (and the mirror stores every accepted memory write in it). To start from
   now, with no backfill, run `hermes qdrant capture scan --advance` once: it moves the cursor to
   the newest row without capturing anything. Otherwise bound the first run with `--since-days`.
4. `capture_mode: on`, `capture_mirror_builtin_memory: true`, and schedule
   `hermes qdrant capture run --apply` nightly and `hermes qdrant capture digest` daily as
   no-agent cron jobs.
5. Keep `capture_auto_apply: false` for the review-only phase.

Superseded captures are hidden from default recall (like any `superseded` fact); they stay
reachable through searches that include fact history.
