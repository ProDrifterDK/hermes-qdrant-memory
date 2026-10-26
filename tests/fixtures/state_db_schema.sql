-- Hermes state.db schema, dumped read-only (mode=ro) from a live Hermes install on 2026-10-26.
-- Session-recall tests build their fixture database from this file so the indexer
-- runs against the production shape (triggers, FTS tables, extra columns) without
-- ever opening a real state.db. FTS shadow tables and sqlite_* tables are omitted:
-- SQLite creates them itself.

CREATE TABLE async_delegations (
            delegation_id TEXT PRIMARY KEY,
            origin_session TEXT NOT NULL,
            origin_ui_session_id TEXT NOT NULL DEFAULT '',
            parent_session_id TEXT,
            state TEXT NOT NULL,
            dispatched_at REAL NOT NULL,
            completed_at REAL,
            updated_at REAL NOT NULL,
            event_json TEXT,
            result_json TEXT,
            delivery_state TEXT NOT NULL DEFAULT 'pending',
            delivery_attempts INTEGER NOT NULL DEFAULT 0,
            delivered_at REAL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            task_json TEXT,
            delivery_claim TEXT,
            delivery_claimed_at REAL
        , origin_session_id TEXT, "owner_execution_id" TEXT);

CREATE TABLE compression_locks (
    session_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE conversation_generations (
    source TEXT NOT NULL,
    session_key TEXT NOT NULL,
    generation INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (source, session_key)
);

CREATE TABLE delivery_obligations (
            obligation_id TEXT PRIMARY KEY,
            session_key TEXT NOT NULL,
            platform TEXT NOT NULL,
            chat_id TEXT NOT NULL,
            thread_id TEXT,
            content TEXT NOT NULL,
            state TEXT NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            owner_pid INTEGER,
            owner_started_at INTEGER,
            last_error TEXT
        , adapter_profile TEXT);

CREATE TABLE gateway_heartbeats (
    backend_id TEXT PRIMARY KEY,
    pid INTEGER NOT NULL,
    started_at REAL NOT NULL,
    last_heartbeat REAL NOT NULL,
    profile TEXT NOT NULL DEFAULT '',
    host TEXT NOT NULL DEFAULT ''
);

CREATE TABLE gateway_hygiene_state (
    session_key TEXT PRIMARY KEY,
    failure_streak INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE gateway_routing (
    scope TEXT NOT NULL DEFAULT '',
    session_key TEXT NOT NULL,
    entry_json TEXT NOT NULL,
    updated_at REAL NOT NULL,
    PRIMARY KEY (scope, session_key)
);

CREATE TABLE hosted_room_events (
            room_id TEXT NOT NULL,
            seq INTEGER NOT NULL CHECK (seq >= 1),
            event_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            actor_json TEXT NOT NULL,
            authority_epoch INTEGER CHECK (authority_epoch IS NULL OR authority_epoch >= 1),
            payload_json TEXT NOT NULL,
            created_at REAL NOT NULL,
            PRIMARY KEY (room_id, seq),
            UNIQUE (room_id, event_id),
            FOREIGN KEY (room_id) REFERENCES hosted_rooms(room_id)
        );

CREATE TABLE hosted_room_links (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_url TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            grant TEXT NOT NULL,
            catalog_json TEXT NOT NULL,
            cancellation_scope_id TEXT NOT NULL,
            trace_id TEXT NOT NULL,
            transport_security TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'ready',
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id)
        );

CREATE TABLE hosted_room_peer_reservations (
            room_id TEXT NOT NULL,
            member_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            expires_at REAL NOT NULL,
            revoked_at REAL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (room_id, member_id, target_profile)
        );

CREATE TABLE hosted_room_policy_cursors (
                    room_id TEXT PRIMARY KEY,
                    through_seq INTEGER NOT NULL DEFAULT 0,
                    stopped_through_seq INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL DEFAULT 0
                );

CREATE TABLE hosted_room_policy_events (
                    room_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    discussion_event_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_json TEXT NOT NULL,
                    PRIMARY KEY(room_id, seq)
                );

CREATE TABLE hosted_room_policy_publications (
                    room_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    execution_generation INTEGER NOT NULL DEFAULT 0,
                    seq INTEGER NOT NULL,
                    PRIMARY KEY(room_id, task_id, kind, execution_generation)
                );

CREATE TABLE hosted_room_policy_threads (
                    room_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    discussion_event_id TEXT NOT NULL,
                    latest_user_seq INTEGER NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(room_id, thread_id)
                );

CREATE TABLE hosted_room_policy_transcript (
                    room_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    kind TEXT NOT NULL,
                    settled_seq INTEGER,
                    PRIMARY KEY(room_id, thread_id, seq)
                );

CREATE TABLE hosted_room_policy_transcript_state (
                    room_id TEXT PRIMARY KEY,
                    schema_version INTEGER NOT NULL
                );

CREATE TABLE hosted_room_policy_watermarks (
                    room_id TEXT NOT NULL,
                    thread_id TEXT NOT NULL,
                    member_id TEXT NOT NULL,
                    seen_through_seq INTEGER NOT NULL,
                    PRIMARY KEY(room_id, thread_id, member_id)
                );

CREATE TABLE hosted_room_remote_runs (
            room_id TEXT NOT NULL,
            home_install_id TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL CHECK (authority_epoch >= 1),
            member_id TEXT NOT NULL,
            task_id TEXT NOT NULL,
            execution_generation INTEGER NOT NULL CHECK (execution_generation >= 1),
            target_install_id TEXT NOT NULL,
            target_profile TEXT NOT NULL,
            run_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (
                room_id, home_install_id, authority_gateway_id, authority_epoch,
                member_id, target_install_id, target_profile, task_id,
                execution_generation
            )
        );

CREATE TABLE hosted_room_retired_ids (
            room_id TEXT PRIMARY KEY,
            retired_at REAL NOT NULL
        );

CREATE TABLE hosted_room_revoked_grants (
            scope_key TEXT PRIMARY KEY,
            expires_at REAL NOT NULL,
            revoked_before REAL NOT NULL
        );

CREATE TABLE hosted_rooms (
            room_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            members_json TEXT NOT NULL,
            authority_gateway_id TEXT NOT NULL,
            authority_epoch INTEGER NOT NULL DEFAULT 1 CHECK (authority_epoch >= 1),
            next_seq INTEGER NOT NULL DEFAULT 1 CHECK (next_seq >= 1),
            event_bytes INTEGER NOT NULL DEFAULT 0 CHECK (event_bytes >= 0),
            revision INTEGER NOT NULL DEFAULT 1 CHECK (revision >= 1),
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            disbanded_at REAL
        );

CREATE TABLE messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT,
    tool_call_id TEXT,
    tool_calls TEXT,
    tool_name TEXT,
    timestamp REAL NOT NULL,
    token_count INTEGER,
    finish_reason TEXT,
    reasoning TEXT,
    reasoning_details TEXT,
    codex_reasoning_items TEXT
, "reasoning_content" TEXT, "codex_message_items" TEXT, "platform_message_id" TEXT, "observed" INTEGER DEFAULT 0, "active" INTEGER NOT NULL DEFAULT 1, "compacted" INTEGER NOT NULL DEFAULT 0, "effect_disposition" TEXT, "api_content" TEXT, "display_kind" TEXT, "display_metadata" TEXT, "_compressed_summary" INTEGER NOT NULL DEFAULT 0, "display_identity" BLOB, "display_order" INTEGER, "message_uid" TEXT, "absorbed_message_uids" TEXT, "tool_call_uids" TEXT, "tool_call_uid" TEXT);

CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_src',
    content_rowid='id'
);

CREATE VIRTUAL TABLE messages_fts_trigram USING fts5(
    content,
    tool_name,
    tool_calls,
    content='messages_fts_trigram_src',
    content_rowid='id',
    tokenize='trigram'
);

CREATE TABLE runtime_epoch (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    epoch INTEGER NOT NULL CHECK (epoch > 0),
    instance_id TEXT NOT NULL
);

CREATE TABLE schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE session_admissions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    admission_id TEXT NOT NULL UNIQUE,
    request_id TEXT NOT NULL,
    principal_id TEXT NOT NULL,
    target_session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    lineage_json TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    payload_digest TEXT NOT NULL,
    intent TEXT NOT NULL CHECK (intent IN ('queue','steer','redirect')),
    status TEXT NOT NULL CHECK (status IN ('queued','started','unknown','terminal')),
    outcome TEXT,
    owner_epoch INTEGER,
    generation INTEGER,
    UNIQUE (principal_id, target_session_id, request_id),
    CHECK (status != 'started' OR (owner_epoch IS NOT NULL AND generation IS NOT NULL))
);

CREATE TABLE session_model_usage (
                                   session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                                   model TEXT NOT NULL,
                                   billing_provider TEXT NOT NULL DEFAULT '',
                                   billing_base_url TEXT NOT NULL DEFAULT '',
                                   billing_mode TEXT NOT NULL DEFAULT '',
                                   task TEXT NOT NULL DEFAULT '',
                                   api_call_count INTEGER NOT NULL DEFAULT 0,
                                   input_tokens INTEGER NOT NULL DEFAULT 0,
                                   output_tokens INTEGER NOT NULL DEFAULT 0,
                                   cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                                   cache_write_tokens INTEGER NOT NULL DEFAULT 0,
                                   reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                                   estimated_cost_usd REAL NOT NULL DEFAULT 0,
                                   actual_cost_usd REAL NOT NULL DEFAULT 0,
                                   cost_status TEXT,
                                   cost_source TEXT,
                                   first_seen REAL,
                                   last_seen REAL,
                                   PRIMARY KEY (session_id, model, billing_provider, billing_base_url, billing_mode, task)
                               );

CREATE TABLE session_turn_leases (
    conversation_id TEXT PRIMARY KEY,
    holder TEXT NOT NULL,
    acquired_at REAL NOT NULL,
    expires_at REAL NOT NULL
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    source TEXT NOT NULL,
    user_id TEXT,
    model TEXT,
    model_config TEXT,
    system_prompt TEXT,
    parent_session_id TEXT,
    started_at REAL NOT NULL,
    ended_at REAL,
    end_reason TEXT,
    message_count INTEGER DEFAULT 0,
    tool_call_count INTEGER DEFAULT 0,
    input_tokens INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    cache_read_tokens INTEGER DEFAULT 0,
    cache_write_tokens INTEGER DEFAULT 0,
    reasoning_tokens INTEGER DEFAULT 0,
    billing_provider TEXT,
    billing_base_url TEXT,
    billing_mode TEXT,
    estimated_cost_usd REAL,
    actual_cost_usd REAL,
    cost_status TEXT,
    cost_source TEXT,
    pricing_version TEXT,
    title TEXT, "api_call_count" INTEGER DEFAULT 0, "handoff_state" TEXT, "handoff_platform" TEXT, "handoff_error" TEXT, "cwd" TEXT, "rewind_count" INTEGER NOT NULL DEFAULT 0, "archived" INTEGER NOT NULL DEFAULT 0, "git_branch" TEXT, "git_repo_root" TEXT, "session_key" TEXT, "chat_id" TEXT, "chat_type" TEXT, "thread_id" TEXT, "compression_failure_cooldown_until" REAL, "compression_failure_error" TEXT, "display_name" TEXT, "origin_json" TEXT, "expiry_finalized" INTEGER DEFAULT 0, "compression_fallback_streak" INTEGER NOT NULL DEFAULT 0, "profile_name" TEXT, "compression_ineffective_count" INTEGER NOT NULL DEFAULT 0, "pinned" INTEGER NOT NULL DEFAULT 0, "system_prompt_hash" TEXT, "last_activity_at" REAL, "last_activity_description" TEXT, "last_activity_provenance" TEXT, "last_read_at" REAL, "title_source" TEXT, "git_metadata_generation" INTEGER NOT NULL DEFAULT 0, "hidden" INTEGER NOT NULL DEFAULT 0, "compression_recovery_deadline" REAL, "tool_names" TEXT, "transport_profile" TEXT, "created_source" TEXT, "compression_overload_streak" INTEGER NOT NULL DEFAULT 0, "auto_archived" INTEGER NOT NULL DEFAULT 0, "runtime_revision" INTEGER NOT NULL DEFAULT 0, "runtime_generation" INTEGER NOT NULL DEFAULT 0, "transcript_revision" INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (parent_session_id) REFERENCES sessions(id)
);

CREATE TABLE state_meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE system_prompts (
    hash TEXT PRIMARY KEY,
    prompt TEXT NOT NULL
);

CREATE TABLE worker_executions (
    execution_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    kind TEXT NOT NULL CHECK (kind IN ('cron','child','compute','kanban')),
    owner_epoch INTEGER NOT NULL,
    generation INTEGER NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('registered','running','unknown','terminal')),
    adoption_digest TEXT NOT NULL,
    last_sequence INTEGER NOT NULL DEFAULT 0 CHECK (last_sequence >= 0)
);

CREATE TABLE worker_receipts (
    execution_id TEXT NOT NULL REFERENCES worker_executions(execution_id) ON DELETE RESTRICT,
    sequence INTEGER NOT NULL CHECK (sequence > 0),
    payload_digest TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (execution_id, sequence)
);

CREATE VIEW messages_fts_src AS
    SELECT id,
           CASE WHEN role = 'tool'
                THEN substr(COALESCE(content, ''), 1, 8192)
                ELSE content END AS content,
           tool_name, tool_calls
    FROM messages;

CREATE VIEW messages_fts_trigram_src AS
    SELECT id, role, content, tool_name, tool_calls
    FROM messages
    WHERE role <> 'tool';

CREATE INDEX idx_async_delegations_delivery
    ON async_delegations(delivery_state, completed_at);

CREATE INDEX idx_compression_locks_expires ON compression_locks(expires_at);

CREATE INDEX idx_hosted_room_events_cursor
           ON hosted_room_events(room_id, seq);

CREATE INDEX idx_hosted_room_policy_events_active
                   ON hosted_room_policy_events(
                       room_id, discussion_event_id, seq
                   );

CREATE INDEX idx_hosted_room_policy_pending
                   ON hosted_room_policy_threads(
                       room_id, completed, latest_user_seq, thread_id
                   );

CREATE INDEX idx_hosted_room_user_thread
       ON hosted_room_events(room_id, json_extract(payload_json, '$.thread_id'), seq)
       WHERE kind='message.user';

CREATE INDEX idx_messages_active_null
    ON messages(active) WHERE active IS NULL;

CREATE INDEX idx_messages_assistant_calls_by_session
    ON messages(session_id)
    WHERE role = 'assistant' AND tool_calls IS NOT NULL;

CREATE INDEX idx_messages_display_backfill
    ON messages(session_id) WHERE (display_order IS NULL OR display_identity IS NULL)
    AND (active = 1 OR compacted = 1);

CREATE INDEX idx_messages_display_identity
    ON messages(session_id, display_identity, display_order)
    WHERE display_identity IS NOT NULL AND (active = 1 OR compacted = 1);

CREATE INDEX idx_messages_display_page
    ON messages(session_id, display_order, active DESC, id DESC)
    WHERE active = 1 OR compacted = 1;

CREATE INDEX idx_messages_platform_msg_id ON messages(session_id, platform_message_id) WHERE platform_message_id IS NOT NULL;

CREATE INDEX idx_messages_session ON messages(session_id, timestamp);

CREATE INDEX idx_messages_session_active
    ON messages(session_id, active, timestamp);

CREATE INDEX idx_messages_session_id ON messages(session_id, id);

CREATE INDEX idx_messages_session_uid
    ON messages(session_id, message_uid) WHERE active = 1;

CREATE INDEX idx_session_admissions_cron_request
    ON session_admissions(principal_id, request_id, target_session_id) WHERE principal_id = 'cron-owner';

CREATE INDEX idx_session_model_usage_model ON session_model_usage(model);

CREATE INDEX idx_session_model_usage_session ON session_model_usage(session_id);

CREATE INDEX idx_session_turn_leases_expires ON session_turn_leases(expires_at);

CREATE INDEX idx_sessions_effective_activity
    ON sessions(COALESCE(last_activity_at, started_at) DESC, started_at DESC);

CREATE INDEX idx_sessions_gateway_peer
    ON sessions(source, user_id, chat_id, chat_type, thread_id, started_at DESC);

CREATE INDEX idx_sessions_handoff_state
    ON sessions(handoff_state, started_at);

CREATE INDEX idx_sessions_local_created_started
    ON sessions(created_source, started_at DESC)
    WHERE source = 'local';

CREATE INDEX idx_sessions_parent ON sessions(parent_session_id);

CREATE INDEX idx_sessions_session_key
    ON sessions(session_key, started_at DESC);

CREATE INDEX idx_sessions_source ON sessions(source);

CREATE INDEX idx_sessions_source_id ON sessions(source, id);

CREATE INDEX idx_sessions_started ON sessions(started_at DESC);

CREATE INDEX idx_sessions_system_prompt_hash
    ON sessions(system_prompt_hash);

CREATE UNIQUE INDEX idx_sessions_title_unique ON sessions(title) WHERE title IS NOT NULL;

CREATE INDEX idx_sessions_tool_names
    ON sessions(tool_names);

CREATE INDEX idx_worker_executions_session_status
    ON worker_executions(session_id, status);

CREATE INDEX session_admissions_pending
    ON session_admissions(target_session_id, status, seq);

CREATE TRIGGER messages_display_identity_delete
AFTER DELETE ON messages WHEN old.active = 1 OR old.compacted = 1
BEGIN
    UPDATE messages SET display_order = (
        SELECT MIN(peer.id) FROM messages AS peer
        WHERE peer.session_id = old.session_id AND (peer.active = 1 OR peer.compacted = 1)
          AND peer.display_identity = old.display_identity
    ) WHERE session_id = old.session_id AND (active = 1 OR compacted = 1)
      AND display_identity = old.display_identity;
END;

CREATE TRIGGER messages_display_identity_update
AFTER UPDATE OF role, content, timestamp, tool_call_id, tool_calls, tool_name,
                display_kind ON messages
WHEN new.role IS NOT old.role
  OR new.content IS NOT old.content
  OR new.timestamp IS NOT old.timestamp
  OR new.tool_call_id IS NOT old.tool_call_id
  OR new.tool_calls IS NOT old.tool_calls
  OR new.tool_name IS NOT old.tool_name
  OR new.display_kind IS NOT old.display_kind
BEGIN
    UPDATE messages SET display_identity = NULL, display_order = NULL
    WHERE id = new.id OR (
        session_id = old.session_id AND display_identity = old.display_identity
        AND (active = 1 OR compacted = 1)
    );
END;

CREATE TRIGGER messages_display_order_insert
AFTER INSERT ON messages WHEN new.display_order IS NULL
BEGIN
    UPDATE messages SET display_order = COALESCE((
        SELECT display_order FROM messages
        WHERE session_id = new.session_id AND id <> new.id
          AND (active = 1 OR compacted = 1)
          AND display_identity = new.display_identity AND display_order IS NOT NULL
        ORDER BY display_order LIMIT 1
    ), new.id) WHERE id = new.id;
END;

CREATE TRIGGER messages_display_visibility_update
AFTER UPDATE OF active, compacted ON messages
WHEN (new.active = 1 OR new.compacted = 1) <> (old.active = 1 OR old.compacted = 1)
BEGIN
    UPDATE messages SET display_order = MIN(new.id, COALESCE((
        SELECT display_order FROM messages
        WHERE session_id = new.session_id AND id <> new.id
          AND (active = 1 OR compacted = 1)
          AND display_identity = new.display_identity AND display_order IS NOT NULL
        ORDER BY display_order LIMIT 1
    ), new.id)) WHERE id = new.id
      AND (new.active = 1 OR new.compacted = 1);
    UPDATE messages SET display_order = (SELECT display_order FROM messages WHERE id = new.id)
    WHERE session_id = new.session_id AND id <> new.id AND (active = 1 OR compacted = 1)
      AND display_identity = new.display_identity
      AND (new.active = 1 OR new.compacted = 1);
    UPDATE messages SET display_order = (
        SELECT MIN(peer.id) FROM messages AS peer
        WHERE peer.session_id = old.session_id AND (peer.active = 1 OR peer.compacted = 1)
          AND peer.display_identity = old.display_identity
    ) WHERE session_id = old.session_id AND (active = 1 OR compacted = 1)
      AND display_identity = old.display_identity
      AND NOT (new.active = 1 OR new.compacted = 1);
END;

CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages
WHEN (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES (
        'delete',
        old.id,
        CASE WHEN old.role = 'tool'
         THEN substr(COALESCE(old.content, ''), 1, 8192)
         ELSE old.content END,
        old.tool_name,
        old.tool_calls
    );
END;

CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages
WHEN (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                         WHERE key = 'fts_rebuild_high_water'), -1)
   OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                          WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (
        new.id,
        CASE WHEN new.role = 'tool'
         THEN substr(COALESCE(new.content, ''), 1, 8192)
         ELSE new.content END,
        new.tool_name,
        new.tool_calls
    );
END;

CREATE TRIGGER messages_fts_trigram_delete AFTER DELETE ON messages
WHEN old.role <> 'tool'
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    VALUES ('delete', old.id, old.content, old.tool_name, old.tool_calls);
END;

CREATE TRIGGER messages_fts_trigram_insert AFTER INSERT ON messages
WHEN new.role <> 'tool'
   AND (new.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR new.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    VALUES (new.id, new.content, new.tool_name, new.tool_calls);
END;

CREATE TRIGGER messages_fts_trigram_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts_trigram(messages_fts_trigram, rowid, content, tool_name, tool_calls)
    SELECT 'delete', old.id, old.content, old.tool_name, old.tool_calls
    WHERE old.role <> 'tool';
    INSERT INTO messages_fts_trigram(rowid, content, tool_name, tool_calls)
    SELECT new.id, new.content, new.tool_name, new.tool_calls
    WHERE new.role <> 'tool';
END;

CREATE TRIGGER messages_fts_update
AFTER UPDATE OF content, tool_name, tool_calls, role ON messages
WHEN (old.content IS NOT new.content
    OR old.tool_name IS NOT new.tool_name
    OR old.tool_calls IS NOT new.tool_calls
    OR old.role IS NOT new.role)
   AND (old.id > COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                           WHERE key = 'fts_rebuild_high_water'), -1)
     OR old.id <= COALESCE((SELECT CAST(value AS INTEGER) FROM state_meta
                            WHERE key = 'fts_rebuild_progress'), -1))
BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content, tool_name, tool_calls)
    VALUES (
        'delete',
        old.id,
        CASE WHEN old.role = 'tool'
         THEN substr(COALESCE(old.content, ''), 1, 8192)
         ELSE old.content END,
        old.tool_name,
        old.tool_calls
    );
    INSERT INTO messages_fts(rowid, content, tool_name, tool_calls)
    VALUES (
        new.id,
        CASE WHEN new.role = 'tool'
         THEN substr(COALESCE(new.content, ''), 1, 8192)
         ELSE new.content END,
        new.tool_name,
        new.tool_calls
    );
END;

CREATE TRIGGER messages_message_uid_insert
AFTER INSERT ON messages WHEN new.message_uid IS NULL
BEGIN
    -- Every row carries a message_uid, whoever wrote it: a build that predates the column binds NULL, so
    -- the store mints one on its behalf (the current build always binds a uid; this never fires for it).
    -- The note lives inside the body: a comment before DROP/CREATE would defeat the settled-trigger skip.
    UPDATE messages SET message_uid = lower(hex(randomblob(16))) WHERE id = new.id;
END;

CREATE TRIGGER messages_transcript_revision_update
AFTER UPDATE OF role, content, api_content, tool_calls, tool_call_id, tool_name, reasoning, reasoning_content,
    reasoning_details ON messages
BEGIN
    -- An in-place rewrite (prompt expansion, multimodal context, a redaction) moves neither the
    -- active row count nor the highest id; this write-side revision is what a transcript-derived
    -- cache keys on (conversation_revision). Whoever wrote the row, old builds included.
    UPDATE sessions SET transcript_revision = transcript_revision + 1 WHERE id = new.session_id;
END;

CREATE TRIGGER sessions_runtime_ledger_delete_guard
BEFORE DELETE ON sessions
WHEN EXISTS (SELECT 1 FROM session_admissions WHERE target_session_id = old.id)
  OR EXISTS (SELECT 1 FROM worker_executions WHERE session_id = old.id)
BEGIN
    SELECT RAISE(ABORT, 'Refused to delete a session that has gateway runtime history: state.db was upgraded by a newer Hermes, and this older Hermes cannot delete such sessions. Downgrading is unsupported. Run hermes update to go back to the newer release, or restore the pre-update snapshot: https://hermes-agent.nousresearch.com/docs/getting-started/updating#rolling-back-to-an-older-release');
END;
