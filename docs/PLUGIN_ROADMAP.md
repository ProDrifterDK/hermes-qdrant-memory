# Formal Hermes Plugin Roadmap

> **For Hermes:** Use `subagent-driven-development` if this roadmap is implemented task-by-task. Keep every mutating memory feature dry-run-first and review-gated.

**Goal:** Turn `hermes-qdrant-memory` from a working local/public-beta MemoryProvider into a formal, documented, installable, operable Hermes memory plugin.

**Architecture:** The plugin remains a Hermes `MemoryProvider`, not a `ContextEngine`. Qdrant handles cross-session semantic recall, indexed files, procedural learnings, and reviewable consolidation reports. LCM continues to own current-session lossless context recovery.

**Tech Stack:** Hermes Agent plugin API, Python 3.10+, Qdrant HTTP API, OpenAI-compatible embeddings, pytest, GitHub Actions, optional Hermes CLI plugin commands, optional Hermes cron/no-agent watcher.

---

## 1. Current state

The project is published as `v0.8.0 Public Beta` and is functional as a Hermes Qdrant-backed memory provider. The latest release URL is:

- <https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.8.0>

Implemented capabilities through v0.8.0:

- Hermes `MemoryProvider` subclass: `QdrantMemoryProvider`.
- Plugin registration through `register(ctx)`.
- Qdrant REST client and OpenAI-compatible embedding client.
- Semantic recall via `prefetch()` / `queue_prefetch()`.
- Completed-turn write-through through `sync_turn()`.
- Manual memory tools: status, search, store, index, forget.
- Markdown/text indexing with dry-run defaults.
- File manifest sync and explicit-ID stale chunk deletion.
- Separate procedural learning collection: `hermes_learnings`.
- Gated automatic learning candidates from `on_pre_compress()` and `on_session_end()`.
- Semantic dedupe for learning candidates.
- Conservative origin-time fact metadata.
- Report-only sleep consolidation.
- Persisted consolidation reports and gated apply-by-proposal-id.
- Reconsolidation candidates as review drafts only.
- Conservative no-agent watcher model, now exposed through native watcher lifecycle CLI commands.
- Native Hermes memory-provider CLI beta surface: `hermes qdrant ...`.
- Backup/export/restore recovery primitives with private local artifacts, dry-run restore default, live approval gate, and automatic pre-restore backup.
- Human-readable default CLI output with `--json` as the stable machine-readable mode, documented in `docs/CLI_OUTPUT_CONTRACT.md`.
- Read-only CLI inspection helpers for exact point lookup, persisted report listing/showing, and proposal inspection.
- Read-only search filters for memory and learning search by tag, source, file path, project path, creation date range, and collection routing.
- GitHub Actions test workflow with pytest, compileall, and scanner guard.
- Release documentation: `CHANGELOG.md`, `RELEASE_NOTES.md`, install/update/remove/rollback notes.

Known compatibility detail:

- Desired category layout: `~/.hermes/plugins/memory/qdrant/`.
- Current Hermes compatibility path: `~/.hermes/plugins/qdrant` or symlink from `plugins/qdrant` to `plugins/memory/qdrant`.
- Verified result: current Hermes core does **not** activate user memory providers from `~/.hermes/plugins/memory/<name>` alone. The general plugin scanner records `memory/qdrant` as an exclusive plugin, but the memory-provider discovery path still scans user providers at `$HERMES_HOME/plugins/<name>`.
- Keep the compatibility shim until Hermes core memory-provider discovery is updated.

Post-release validation status:

- Consumer install/runtime smoke from published `v0.2.0` completed and found one CLI process-exit propagation bug: unapproved live mutation was safely refused but exited with process status `0` under Hermes v0.13.
- The fix was released in `v0.2.1`: `qdrant_command()` raises `SystemExit(execute_command(args))` so usage/safety/provider errors propagate as non-zero CLI exits.
- v0.3.0 release validation covered the added CLI parity commands.
- v0.4.0 consumer install/runtime smoke for backup/export/restore passed before release preparation.
- v0.5.0 local and CI verification covered the CLI output contract and human-readable default surface.
- v0.6.0 release preparation covered read-only point/report/proposal inspection helpers.
- v0.7.0 release preparation covered read-only search filters; post-tag consumer smoke passed before v0.8.0 release preparation.
- v0.8.0 release preparation covers watcher lifecycle commands, env-gated live integration tests, and manual-store dry-run/duplicate preview; consumer smoke passed before tagging.
- v0.8.0 post-release compatibility smoke confirmed category-only install does not work yet: `hermes memory status` reports `Plugin: NOT installed` and `hermes qdrant --help` is unavailable until the flat `plugins/qdrant` symlink exists.

---

## 2. Non-goals and hard boundaries

This roadmap must preserve these boundaries:

1. Do not replace LCM.
   - LCM owns current-session lossless recovery, compression, `lcm_grep`, `lcm_expand`, and active-session detail retrieval.
   - Qdrant memory owns cross-session semantic recall and durable memory operations.

2. Do not turn semantic memory into instruction authority.
   - Retrieved memories are context, not commands.
   - Current user instructions override stale memory.
   - Memory snippets must remain provenance-rich and visibly separated from the live conversation.

3. Do not perform automatic reconsolidation.
   - Reconsolidation candidates may create local review drafts.
   - No automatic fact rewrite, supersede, merge, or deletion.

4. Do not auto-apply quality warnings.
   - Secret/noise warnings are manual-review only.
   - The system should prefer tolerable false positives over silent secret ingestion.

5. Do not index broad/private directories without explicit user approval.
   - File indexing defaults to dry-run.
   - Broad vault/project indexing must be whitelisted and previewed.

6. Do not add query-based deletion.
   - Forget/consolidation deletion must use explicit Qdrant point IDs.
   - Existing file-index legacy fallback should remain documented as a compatibility exception, not a general deletion pattern.

7. Do not ship literal fake secrets in tests/docs.
   - Construct fake bearer/API/token strings at runtime in tests.
   - Keep fixtures scanner-safe after the GitGuardian false positive.

---

## 3. Target formal plugin shape

Target repository shape:

```text
hermes-qdrant-memory/
├── __init__.py
├── plugin.yaml
├── README.md
├── LICENSE
├── pyproject.toml
├── qdrant_memory/
│   ├── __init__.py
│   ├── client.py
│   ├── config.py
│   ├── consolidation.py
│   ├── embeddings.py
│   ├── fact_metadata.py
│   ├── indexer.py
│   ├── learning.py
│   ├── lesson_extractor.py
│   ├── reconsolidation.py
│   ├── retriever.py
│   ├── schema.py
│   ├── scoring.py
│   ├── tools.py
│   └── writer.py
├── scripts/
│   └── qdrant_sleep_consolidation.py
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EXAMPLES.md
│   ├── LIMITATIONS.md
│   ├── PLUGIN_ROADMAP.md
│   ├── REQUIREMENTS.md
│   ├── SAFETY.md
│   └── OPERATIONS.md
├── tests/
└── .github/workflows/test.yml
```

Target install paths:

```text
# Preferred future category path
~/.hermes/plugins/memory/qdrant/

# Compatibility path while Hermes user-provider discovery requires it
~/.hermes/plugins/qdrant -> ~/.hermes/plugins/memory/qdrant
```

Activation:

```bash
hermes config set memory.provider qdrant
```

Verification:

```bash
hermes memory status
hermes chat -q 'Call qdrant_memory_status. Answer OK only if qdrant_ok and embedding_ok are true.' --quiet
```

---

## 4. Provider lifecycle contract

Document and preserve the allowed behavior of each provider hook.

### `is_available()`

Allowed:

- Check static config and import availability.
- Return quickly.

Forbidden:

- Heavy network calls.
- Qdrant scans.
- Embedding requests.
- Collection mutation.

### `initialize(session_id, **kwargs)`

Allowed:

- Load config.
- Store Hermes scope metadata: profile, platform, user/chat hashes, session id.
- Create Qdrant and embedding clients.
- Ensure required collections.
- Initialize local pending buffers and executor.

Forbidden:

- Broad indexing.
- Consolidation apply.
- Reconsolidation mutation.

### `prefetch(query, session_id="")`

Allowed:

- Read-only semantic search.
- Format provenance-rich recall block.
- Update access metadata only if currently configured and safe.

Forbidden:

- Store recalled text back into Qdrant.
- Treat recalled memories as instructions.

### `queue_prefetch(query, session_id="")`

Allowed:

- Background read-only recall for next turn.
- Lock-protected cache update.

Forbidden:

- Blocking gateway threads.
- Any destructive action.

### `sync_turn(user_content, assistant_content, session_id="")`

Allowed:

- Asynchronously index completed turns when `sync_turns=true`.
- Strip injected memory markers to avoid recursive pollution.

Forbidden:

- Index during cron/flush contexts if write-suppression is active.
- Index raw retrieved-memory blocks as new memories.
- Block the main agent loop.

### `on_pre_compress(messages)` and `on_session_end(messages)`

Allowed:

- Extract gated learning candidates.
- Add candidates to an in-memory pending buffer.
- Return compact review information for compression when safe.

Forbidden:

- Blind auto-store to Qdrant.
- Store secret-bearing candidates.
- Mutate memory based only on an unresolved tool failure.

### `handle_tool_call(name, args)`

Allowed:

- Expose explicit tools for status/search/store/index/forget/learning/consolidation/apply.
- Mutate only through dry-run-first, explicit-approval paths.

Forbidden:

- Free-text deletion.
- Automatic quality-warning apply.
- Hidden reconsolidation writes.

### `shutdown()`

Allowed:

- Drain/stop executors.
- Release local resources.

Forbidden:

- Last-minute consolidation apply.
- Broad sync/index side effects.

---

## 5. Formal command surface

The current stable surface includes Hermes tool calls and the native `hermes qdrant ...` provider CLI. The CLI remains a thin operator wrapper over the tool surface and must not weaken tool safety.

### M13 implemented v0.3.0 commands

Implemented command namespace:

```bash
hermes qdrant config show --json
hermes qdrant status
hermes qdrant doctor
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual --preview-duplicates
hermes qdrant store "Remember this explicit memory" --source-type manual --importance 5 --tag manual --preview-duplicates --no-dry-run --approve
hermes qdrant search "query text" --top-k 5 --json
hermes qdrant index docs README.md --dry-run
hermes qdrant forget POINT_ID --dry-run
hermes qdrant learning search "pytest hermes venv" --top-k 5 --json
hermes qdrant learning preview --json
hermes qdrant learning store "Procedural lesson" --learning-type workflow_lesson --confidence 0.8 --tag manual
hermes qdrant learning approve CANDIDATE_ID --dry-run --json
hermes qdrant consolidate --scope both --persist --include-reconsolidation --dry-run --json
hermes qdrant apply --report-id REPORT --proposal-id PROPOSAL --action merge --dry-run --json
hermes qdrant watcher status --verbose --json
hermes qdrant watcher install --schedule "0 3 * * *" --json
hermes qdrant watcher run --scope both --force-alert --json
hermes qdrant watcher logs --tail 20 --json
hermes qdrant watcher inspect-state --json
hermes qdrant watcher reset-signature --approve --json
hermes qdrant watcher uninstall --approve --json
```

CLI rules:

- `config show`, `watcher status`, `watcher logs`, `watcher inspect-state`, `watcher install`, `watcher uninstall`, and `watcher reset-signature` are local provider-free operations.
- `store` previews by default and requires `--no-dry-run --approve` for live manual writes.
- `store --preview-duplicates` can refuse an upsert when a semantic duplicate is found, but it never deletes, merges, or rewrites existing points.
- `learning store` remains an explicit live write by design.
- Maintenance mutations default to `--dry-run`.
- Live maintenance mutation requires both `--no-dry-run` and `--approve`.
- Apply commands require exact `report_id` and `proposal_id`.
- Delete commands require explicit point IDs.
- `quality_warning` proposals must refuse all live apply attempts.
- `draft_review` is the only allowed reconsolidation action and writes local markdown only.

Implementation:

- Native Hermes memory-provider CLI discovery is implemented through top-level `cli.py`.
- Reusable dispatch logic lives in `qdrant_memory/cli_core.py`.
- If future Hermes versions change provider CLI discovery, ship a standalone wrapper that reuses `cli_core`.

---

## 6. Cron and watcher integration

The formal plugin should define a cron-safe operating model.

### Cron-safe rule

Scheduled jobs may observe and report. They must not autonomously mutate Qdrant.

Local artifact persistence is allowed: `persist=true` means writing a redacted JSON/markdown review artifact under `$HERMES_HOME/qdrant_memory/consolidation`. That is not a Qdrant mutation. Cron jobs must still avoid Qdrant upserts, deletes, payload updates, learning approvals, and consolidation apply.

Allowed cron jobs:

```text
weekly status report
weekly consolidation report with persist=true
monthly dry-run indexing audit for configured index_dirs
manual forced watcher run for debugging
```

Forbidden cron jobs:

```text
automatic qdrant_memory_consolidation_apply
automatic reconsolidation fact rewrite
automatic quality_warning resolution
automatic broad live indexing
automatic query-based deletion
```

Recommended watcher behavior:

- Run `qdrant_memory_consolidate` with:
  - `scope=both`
  - `dry_run=true`
  - `persist=true`
  - `include_examples=false`
  - `include_reconsolidation=true`
- Store a stable proposal signature in watcher state.
- Stay silent if proposal signature is unchanged.
- Emit compact alerts only when actionable proposal sets change.

Severity mapping:

| Severity | Trigger | Action |
|---|---|---|
| MANUAL REVIEW REQUIRED | reconsolidation candidates | Human review of fact conflicts |
| REVIEW SOON | quality warnings | Human secret/noise review |
| MAINTENANCE | duplicates/stale/promotion candidates | Optional cleanup |
| SILENT | no proposals or unchanged signature | No user notification |

Deliverable:

- Move the watcher contract into `docs/OPERATIONS.md`.
- Treat native `hermes qdrant watcher ...` lifecycle commands as the reference implementation for new installs.
- Keep any external watcher script as legacy/manual compatibility only.

---

## 7. LCM interoperability boundary

Formal boundary:

| Area | LCM | Qdrant Memory |
|---|---|---|
| Scope | Current session | Cross-session and indexed external material |
| Retrieval | Lossless expansion of compacted context | Semantic nearest-neighbor recall |
| Tools | `lcm_grep`, `lcm_describe`, `lcm_expand`, `lcm_expand_query` | `qdrant_memory_search`, `qdrant_learning_search`, consolidation tools |
| Mutability | Context store / compression DAG | Memory collections and local review artifacts |
| Truth model | Original messages/summaries | Similar chunks with provenance and scores |

Possible future integration:

- Qdrant may index LCM-generated session summaries after session end.
- Qdrant may store durable learnings extracted from compression boundaries.
- LCM may remain the first choice for active-session recall; Qdrant is used when the question spans sessions or project notes.

Forbidden integration:

- Qdrant must not mutate LCM internals.
- Qdrant must not replace LCM as the context engine.
- LCM summaries should not be blindly re-indexed if they contain injected memory blocks.

- [LCM_BOUNDARY.md](LCM_BOUNDARY.md) documents this boundary in full.

Deliverable:

- Add [LCM_BOUNDARY.md](LCM_BOUNDARY.md) or a section in `docs/ARCHITECTURE.md`.

Status: completed in [LCM_BOUNDARY.md](LCM_BOUNDARY.md).

---

## 8. Safety policy

Create `docs/SAFETY.md` and make it the canonical contract.

Canonical policy: `docs/SAFETY.md`.

Required policy points:

1. Dry-run first.
   - Indexing, forget, learning approval, and consolidation apply default to dry-run.

2. Explicit proposal handles.
   - Live consolidation apply requires exact `report_id` and `proposal_id`.

3. Explicit point IDs.
   - Delete actions use explicit point IDs only.

4. Manual quality warnings.
   - `quality_warning` is never live-applied.

5. Reconsolidation draft-only.
   - Fact conflict proposals can only create local review drafts.

6. Skill promotion draft-only.
   - Learning promotion creates draft skill artifacts; it never installs active skills automatically.

7. Secret safety.
   - Secret-bearing candidates are blocked or redacted.
   - Persisted reports and review artifacts are recursively redacted.
   - Tests and docs avoid literal fake secrets.

8. Scope safety.
   - Retrieval filters by profile/user/chat according to `scope_mode`.
   - Avoid `global` in shared gateway deployments.

9. Provenance over certainty.
   - Search results must expose source, IDs, timestamps, and scores.
   - Semantic similarity is not truth.

10. Current instruction priority.
   - Live user instruction beats retrieved memory.

11. Rollback and audit.
   - Live approved mutations must leave application artifacts.
   - Artifacts must include report/proposal IDs and explicit affected point IDs.
   - Operators should be able to audit what was changed and why.
   - Future export/backup tooling should be added before any broader mutation surface.

---

## 9. Release milestones

### M12: Formal plugin packaging alignment

Objective: make the project visibly conform to Hermes plugin conventions.

Status: completed before v0.2.0.

Completed tasks:

1. Document preferred install path and compatibility symlink.
2. Verify `plugin.yaml` metadata: name, category, provider type, version, description.
3. Add plugin registration tests for `register(ctx)`.
4. Add compatibility note to README install section.
5. Add `docs/PLUGIN_ROADMAP.md`.

Verification:

```bash
python -m pytest tests -q
python -m compileall -q qdrant_memory __init__.py
hermes memory status
```

### M13: CLI parity

Objective: provide a stable human CLI for the same operations exposed as tools.

Status: completed for v0.3.0 as a native Hermes memory-provider CLI beta surface.

Completed tasks:

1. Confirmed Hermes memory-provider CLI discovery path.
2. Added top-level `cli.py` and `qdrant_memory/cli_core.py`.
3. Implemented `status`, `doctor`, `search`, `index`, `forget`, `learning search`, `learning preview`, `consolidate`, and `apply`.
4. Added M18 parity commands in v0.3.0: `config show`, `store`, `learning store`, `learning approve`, `watcher status`, and report-only `watcher run`.
5. Preserved dry-run defaults and live-mutation approval gates.
6. Added CLI tests for parsing, command-to-tool mapping, import isolation, exit codes, config redaction/no-provider behavior, watcher state reads, and safety gates.
7. Added manual store dry-run defaults, live approval gating, and duplicate-preview mapping for `hermes qdrant store`.

Deferred to post-v0.3.0 CLI roadmap:

- optional backup/export commands

Verification:

```bash
hermes qdrant config show --json
hermes qdrant status
hermes qdrant store "Manual memory" --tag manual --preview-duplicates
hermes qdrant store "Manual memory" --tag manual --preview-duplicates --no-dry-run --approve
hermes qdrant search "Hermes Qdrant memory"
hermes qdrant learning store "Procedural lesson" --tag manual
hermes qdrant learning approve CANDIDATE_ID --dry-run
hermes qdrant watcher status --verbose --json
hermes qdrant watcher install --schedule "0 3 * * *" --json
hermes qdrant watcher run --scope both --force-alert --json
hermes qdrant watcher logs --tail 20 --json
hermes qdrant watcher inspect-state --json
hermes qdrant watcher reset-signature --approve --json
hermes qdrant watcher uninstall --approve --json
hermes qdrant consolidate --scope both --persist --dry-run
# For apply, first generate a real persisted report and select one real proposal_id.
# If no proposal exists, verify argument validation and dry-run refusal paths instead.
hermes qdrant apply --report-id REPORT_ID --proposal-id PROPOSAL_ID --dry-run
python -m pytest tests -q
```

### M14: Cron-safe automation

Objective: make watcher/reporting operationally durable and non-spammy.

Tasks:

1. Document watcher install/update/remove commands.
2. Add watcher state inspection.
3. Add no-agent cron recipe.
4. Add regression tests for unchanged-signature silence.
5. Add failure-mode docs for Qdrant/embedding outages.

Verification:

```bash
QDRANT_SLEEP_FORCE_ALERT=1 ~/.hermes/scripts/qdrant_sleep_consolidation.py
hermes cron list
```

Expected:

- Alert only when proposal signature changes or force flag is set.
- No Qdrant mutations.

### M15: Safety hardening

Objective: consolidate safety rules and make scanners/test gates enforce them.

Status: completed before v0.2.0.

Completed tasks:

1. Create `docs/SAFETY.md`.
2. Add tests for scanner-safe fake secret construction.
3. Add tests that persisted artifacts never contain raw bearer/API/private-key markers.
4. Add docs warning against broad directory indexing.
5. Add optional secret-scan CI step if it can be configured without fixture false positives.

Verification:

```bash
python -m pytest tests -q
python -m compileall -q qdrant_memory __init__.py
python scripts/check_no_literal_fake_secrets.py
```

### M16: LCM boundary and interoperability

Objective: avoid conceptual drift between context recovery and semantic memory.

Status: completed before v0.2.0.

Completed tasks:

1. Add [LCM_BOUNDARY.md](LCM_BOUNDARY.md) or expand `docs/ARCHITECTURE.md`.
2. Document active-session recall decision tree: LCM first, Qdrant for cross-session/project/vault recall.
3. Document allowed future integration points.
4. Add test/fixture to ensure injected memory markers are stripped before write-through.

Verification:

```bash
python -m pytest tests/test_tools_retriever_writer.py tests/test_learning_auto_extract.py -q
```

### M17: Release readiness

Objective: produce a stable public release with clear upgrade path.

Status: completed for `v0.2.0 Public Beta`.

Completed tasks:

1. Added `CHANGELOG.md`.
2. Bumped `plugin.yaml` to `0.2.0`.
3. Added install/update/remove/rollback docs.
4. Added real-service smoke test checklist.
5. Created `RELEASE_NOTES.md` with honest beta limitations.
6. Created annotated tag `v0.2.0`.
7. Published GitHub prerelease: <https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.2.0>.
8. Verified release CI: <https://github.com/ProDrifterDK/hermes-qdrant-memory/actions/runs/25977733752>.

Still useful for future releases:

- Keep the compatibility matrix current as Hermes core plugin discovery evolves.
- Run a consumer install/runtime smoke test after each published tag.

Verification used for v0.2.0:

```bash
python -m pytest tests -q
python scripts/check_no_literal_fake_secrets.py
python -m compileall -q qdrant_memory __init__.py cli.py scripts/check_no_literal_fake_secrets.py
gh release view v0.2.0 --repo ProDrifterDK/hermes-qdrant-memory
gh run list --repo ProDrifterDK/hermes-qdrant-memory --limit 5
```

---

## 10. Implementation task breakdown

Recommended execution order:

| Order | Task | Status | Blocks | Reason |
|---|---|---|---|---|
| 1 | Plugin roadmap | Completed | All later roadmap work | Establishes shared plan. |
| 2 | Safety policy | Completed | CLI, cron, release | Safety contract must exist before adding easier mutation surfaces. |
| 3 | Operations document | Completed | Cron, release | Operators need report/review/apply runbooks. |
| 4 | LCM boundary document | Completed | Release | Prevents wrong mental model and support burden. |
| 5 | Plugin metadata tests | Completed | Release | Locks formal plugin shape. |
| 6 | Scanner-safe fixture guard | Completed | More tests/docs | Prevents repeat GitGuardian incidents. |
| 7 | CLI feasibility spike | Completed | CLI MVP | Determines whether commands are Hermes-native or standalone wrapper. |
| 8 | CLI MVP | Completed | Public beta release | Adds human operation surface after safety gates. |
| 9 | Release documentation update | Completed | Release | Packages everything for public users. |

Verification categories:

- Unit tests: `python -m pytest tests -q`.
- Static checks: `python -m compileall -q qdrant_memory __init__.py` and scanner guard.
- Local-service smoke: requires live Qdrant plus embedding endpoint.
- Hermes runtime checks: require local Hermes config and plugin installation.
- GitHub/release checks: require `gh` auth and repository permissions.

### Task 1: Add plugin roadmap document

**Objective:** Create this roadmap as the project coordination document.

**Files:**

- Create: `docs/PLUGIN_ROADMAP.md`

**Verification:**

```bash
test -f docs/PLUGIN_ROADMAP.md
git diff -- docs/PLUGIN_ROADMAP.md
```

### Task 2: Add safety policy document

**Objective:** Make safety behavior canonical instead of scattered across README/architecture/tests.

**Files:**

- Create: `docs/SAFETY.md`
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`

**Minimum sections:**

- Dry-run-first contract.
- Explicit-ID deletion.
- Report/apply separation.
- Reconsolidation draft-only.
- Quality warning manual-only.
- Secret redaction and scanner-safe tests.
- Scope and provenance.

**Verification:**

```bash
python -m pytest tests/test_consolidation.py tests/test_consolidation_apply.py tests/test_reconsolidation.py -q
```

### Task 3: Add operations document

**Objective:** Provide an operator runbook for watcher reports and cron usage.

**Files:**

- Create: `docs/OPERATIONS.md`
- Modify: `README.md`

Status: completed in `docs/OPERATIONS.md`.

**Minimum sections:**

- Status check.
- Real-service smoke test.
- Watcher force-run.
- Report classification.
- Applying approved duplicate/stale proposals.
- What never to apply automatically.
- Gateway restart note.

**Verification:**

```bash
QDRANT_SLEEP_FORCE_ALERT=1 "$HOME/.hermes/scripts/qdrant_sleep_consolidation.py"
```

### Task 4: Add LCM boundary document

**Objective:** Prevent conceptual confusion between Qdrant memory and LCM.

**Files:**

- Create: [LCM_BOUNDARY.md](LCM_BOUNDARY.md)
- Modify: `docs/ARCHITECTURE.md`

Status: completed in [LCM_BOUNDARY.md](LCM_BOUNDARY.md).

**Minimum sections:**

- Decision table: use LCM vs use Qdrant.
- Current-session vs cross-session retrieval.
- Allowed future integration.
- Forbidden integration.

**Verification:**

```bash
python -m pytest tests/test_tools_retriever_writer.py -q
```

### Task 5: Add plugin metadata tests

**Objective:** Lock down formal plugin registration and metadata.

**Files:**

- Modify/Create: `tests/test_plugin_metadata.py`
- Possibly modify: `plugin.yaml`

**Test cases:**

- `plugin.yaml` exists.
- Plugin name/category/provider metadata are correct.
- `register(ctx)` calls memory-provider registration.
- Registered provider is `QdrantMemoryProvider`.

**Verification:**

```bash
python -m pytest tests/test_plugin_metadata.py -q
```

Status: completed in `tests/test_plugin_metadata.py`.

### Task 6: Add scanner-safe fixture guard

**Objective:** Avoid future GitGuardian-style false positives.

**Files:**

- Create: `scripts/check_no_literal_fake_secrets.py`
- Create/Modify: `tests/test_secret_fixture_scan.py`
- Modify: `.github/workflows/test.yml`

**Policy:**

Flag literal test/doc strings matching:

- literal authorization headers with an unredacted bearer value;
- literal fake strings built from words like raw + secret + token;
- literal OpenAI-style key prefixes followed by plausible key material;
- literal GitHub token prefixes followed by plausible key material;
- unredacted private-key block markers in executable fixtures.


Allow:

- runtime concatenation in tests;
- redacted placeholders like `Bearer ***`;
- docs that discuss patterns without literal credential-shaped values.

**Verification:**

```bash
python scripts/check_no_literal_fake_secrets.py
python -m pytest tests/test_secret_fixture_scan.py -q
```

Status: completed with `scripts/check_no_literal_fake_secrets.py`, `tests/test_secret_fixture_scan.py`, and CI workflow integration.

### Task 7: CLI feasibility spike

**Objective:** Determine the cleanest CLI integration path for Hermes user plugins.

**Files:**

- Create: [CLI_SPIKE.md](CLI_SPIKE.md)
- Maybe create: top-level `cli.py` if feasible.

Status: completed in [CLI_SPIKE.md](CLI_SPIKE.md). Native Hermes memory-provider CLI integration is feasible via a top-level plugin `cli.py` using `register_cli(subparser)`; standalone wrapper remains a fallback.

**Questions:**

- Does Hermes currently load plugin CLI hooks?
- If yes, what registration API is stable?
- If no, should this plugin ship a temporary standalone command?
- How do commands call provider logic without booting a full agent session?

**Verification:**

```bash
hermes plugins list
hermes --help
python -m pytest tests -q
```

### Task 8: Implement CLI MVP

**Objective:** Add minimal CLI wrapper after Task 7 answers the integration path.

**Files:**

- Create/Modify: top-level `cli.py`
- Create/Modify: `qdrant_memory/cli_core.py`
- Add tests under `tests/`.

Status: completed with top-level `cli.py`, `qdrant_memory/cli_core.py`, and `tests/test_cli.py`. The CLI is a thin native Hermes memory-provider wrapper over existing tool calls, with dry-run/approval gates and installed-plugin import coverage.

**Commands MVP:**

```bash
hermes qdrant status
hermes qdrant search "query"
hermes qdrant index <path...> --dry-run
hermes qdrant consolidate --scope both --persist --dry-run
```

**Verification:**

```bash
python -m pytest tests/test_cli.py -q
python -m pytest tests -q
python -m compileall -q qdrant_memory __init__.py cli.py
python scripts/check_no_literal_fake_secrets.py
```

### Task 9: Release documentation update

**Objective:** Prepare the next public release.

Status: completed for `v0.2.0 Public Beta`.

**Files:**

- Create: `CHANGELOG.md`
- Create: `RELEASE_NOTES.md`
- Modify: `README.md`
- Modify: `docs/LIMITATIONS.md`

**Verification:**

```bash
python -m pytest tests -q
gh run list --repo ProDrifterDK/hermes-qdrant-memory --limit 5
```

---

## 11. Acceptance criteria

The plugin is ready to call “formal Hermes plugin beta” when:

Status after v0.8.0: release smoke and category-path discovery verification are complete. The remaining compatibility work is a Hermes core change or wrapper fallback if the flat compatibility path ever needs to be removed.

- README documents preferred install path and compatibility symlink.
- Install, update, remove, and rollback docs exist for both preferred category path and current compatibility path.
- Category-path discovery verification is documented: current Hermes core cannot load user memory providers from `plugins/memory/<name>` without the flat `plugins/<name>` compatibility symlink.
- `docs/SAFETY.md` exists and matches implemented behavior.
- `docs/OPERATIONS.md` exists and covers watcher/report handling.
- [LCM_BOUNDARY.md](LCM_BOUNDARY.md) or equivalent architecture section exists.
- Plugin metadata/registration tests pass.
- Literal fake-secret scanner guard exists and passes.
- CLI feasibility is resolved.
- If CLI MVP is implemented, all mutating commands default to dry-run.
- GitHub Actions runs tests, compileall, and scanner guard.
- Real local smoke test verifies Qdrant and embedding endpoints.
- No automatic memory mutation is possible from cron.
- No automatic reconsolidation is possible.

---

## 12. Decision log

| Decision | Status | Rationale |
|---|---|---|
| Implement as `MemoryProvider`, not `ContextEngine` | Accepted | Cross-session memory and tools belong in memory provider layer; LCM owns current-session context. |
| Keep Qdrant external | Accepted | Plugin should not bundle storage; users can run local or remote Qdrant. |
| Keep embeddings external/OpenAI-compatible | Accepted | Avoid coupling to one model/runtime; bge-m3 local is tested default. |
| Keep reconsolidation draft-only | Accepted | Fact rewriting is high risk. |
| Keep quality warnings manual-only | Accepted | False positives are safer than secret ingestion. |
| Use compatibility symlink until Hermes scans user memory-provider category paths | Accepted / verified required | Category-only smoke showed general plugin discovery sees `memory/qdrant`, but memory-provider activation and `hermes qdrant ...` require `plugins/qdrant`. |
| Add CLI wrapper | Accepted | Implemented as native memory-provider CLI MVP in v0.2.0. |
| Add safety/operations/LCM docs | Accepted | Implemented before widening the CLI/release surface. |
| Publish v0.2.0 as prerelease | Accepted | Honest beta label while the plugin remains experimental and externally service-dependent. |
| Publish v0.3.0 as prerelease | Accepted | CLI parity was useful enough for public beta while backup/export and doctor diagnostics remained follow-up work. |
| Publish v0.4.0 as prerelease | Accepted | Backup/export/restore recovery primitives passed real install smoke and are substantial enough for a new beta release. |
| Publish v0.5.0 as prerelease | Accepted | CLI output contract makes the operator surface human-friendly by default while preserving stable `--json` automation output. |

---

## 13. Post-v0.5.0 CLI and operations roadmap

Recommended next phase: post-v0.5.0 should focus on inspection/search ergonomics, watcher operations, and integration confidence rather than expanding memory mutation authority. The plugin already has the MemoryProvider core, safe CLI beta surface, doctor diagnostics, backup/restore recovery primitives, and a documented CLI output contract.

### Priority 1: Consumer install/runtime smoke

Status: completed for v0.5.0 output-contract smoke before release preparation. Future releases should repeat this from the published tag.

Verify from the installed plugin outside the development checkout:

- `git checkout <release-tag>`
- `hermes qdrant config show`
- `hermes qdrant config show --json`
- `hermes qdrant status`
- `hermes qdrant status --json`
- `hermes qdrant doctor`
- `hermes qdrant doctor --json`
- `hermes qdrant search ...`
- `hermes qdrant search ... --json`
- `hermes qdrant learning preview`
- `hermes qdrant learning preview --json`
- `hermes qdrant consolidate --scope both --persist --include-reconsolidation`
- `hermes qdrant watcher status`
- `hermes qdrant watcher status --json`
- `hermes qdrant watcher run --scope both`
- `hermes qdrant watcher run --scope both --json`
- safety-gated `forget`, `apply`, and `learning approve` without `--approve`, all returning non-zero exit status and performing no mutation.

### Priority 2: CLI UX and output contract (implemented in v0.5.0)

The v0.5.0 CLI defines the operator output contract:

- concise human-readable default output;
- `--json` as stable machine-readable output;
- one JSON object on successful `--json` stdout;
- JSON error objects on `--json` stderr;
- consistent exit-code semantics for usage/safety, provider, and service errors;
- sanitized human summaries for backup/export/restore.

Optional `--pretty` / `--quiet` modes remain future ergonomic enhancements, not release blockers.

### Priority 3: Real `doctor` diagnostics (implemented)

`hermes qdrant doctor` now emits structured diagnostics instead of forwarding the raw status payload. The checks cover:

- active provider is `qdrant`;
- plugin path/symlink/category discovery is valid;
- metadata version matches release docs;
- Qdrant is reachable;
- embedding endpoint is reachable;
- collection vector size matches configured embedding size;
- memory and learning collections exist;
- watcher state/artifact directory are readable/writable;
- config redaction catches API-key fields and credentialed URLs.

### Priority 4: Backup/export/rollback before broader mutation (implemented)

Operator recovery primitives now exist before any broader cleanup/rewrite workflow is added:

- `hermes qdrant export memory|learning --out FILE`;
- `hermes qdrant backup create`;
- `hermes qdrant backup list`;
- `hermes qdrant backup inspect ID`;
- `hermes qdrant restore --backup ID --dry-run`;
- live `restore` requires `--no-dry-run --approve`, preflights target vector compatibility, and automatically creates a pre-restore backup;
- optional `--backup-first` exists for live `apply`.

### Priority 5: Inspection/search ergonomics

Point/report inspection commands and search filters are implemented for the first ergonomics pass:

- `hermes qdrant show POINT_ID --collection memory|learning`;
- `hermes qdrant reports list`;
- `hermes qdrant reports show REPORT_ID`;
- `hermes qdrant proposals show REPORT_ID PROPOSAL_ID`;
- `hermes qdrant search "query" --tag TAG --source-type TYPE --source SOURCE --file-path PATH --project-path PATH --since ISO --until ISO`;
- `hermes qdrant search "query" --collection learning --tag TAG`;
- `hermes qdrant learning search "query" --learning-type TYPE --tag TAG --source SOURCE --file-path PATH --project-path PATH --since ISO --until ISO`.

Safety shape:

- `show` is exact-ID only and omits payloads/vectors unless explicitly requested;
- `reports` and `proposals` inspect local persisted artifacts only;
- report/proposal IDs are validated and path traversal inputs are rejected;
- search filters only add read-side Qdrant `must` constraints and collection routing;
- no provider construction, embeddings, upserts, deletes, collection creation, or apply shortcut is introduced.

### Priority 6: Safer explicit write workflows

Keep manual writes ergonomic, but consider:

- optional `--dry-run` preview for `store` and `learning store`;
- duplicate/similarity preview before live store;
- `--stdin` and `--file` input modes;
- scanner/noise warnings before live store;
- explicit `--approve` option for non-interactive scripts.

### Priority 7: Watcher/cron management CLI — implemented in v0.8.0

Watcher remains report-only while lifecycle commands now cover:

- `watcher install`;
- `watcher uninstall --approve`;
- `watcher status --verbose`;
- `watcher run --force-alert`;
- `watcher logs`;
- `watcher inspect-state`;
- `watcher reset-signature --approve`.

Safety boundary: install/uninstall edits only the sentinel-managed crontab block, local state/log commands do not construct the provider, and `watcher run` still maps only to report-only consolidation (`dry_run=true`, `persist=true`, `include_examples=false`).

### Priority 8: Integration and compatibility tests

Add optional live-service tests gated by explicit environment variables, e.g. `RUN_QDRANT_INTEGRATION=1`, so CI remains hermetic by default but maintainers can validate real Qdrant + embedding behavior before releases. Current live coverage includes search filters, provider store writes, file indexing upsert/stale-delete behavior, and gated consolidation report/apply paths over disposable collections.

Also verify:

- installed-plugin `hermes qdrant --help`;
- category path vs compatibility symlink;
- Hermes version matrix;
- CLI process exit behavior from real subprocesses.

### Priority 9: Hermes core compatibility or wrapper fallback

Verified result: current Hermes core cannot activate user memory providers from `~/.hermes/plugins/memory/<name>` without a flat compatibility path. The general `PluginManager` can discover the category path as `memory/qdrant`, but `plugins/memory/__init__.py` still resolves user providers through `$HERMES_HOME/plugins/<name>`.

Keep the shim documented. Future options:

- patch Hermes core memory-provider discovery to also scan `$HERMES_HOME/plugins/memory/<name>`;
- open a Hermes core issue/PR with the category-only smoke evidence;
- ship a standalone wrapper that reuses `qdrant_memory.cli_core` if native provider CLI discovery changes.

---

## 14. Immediate next action

1. Decide whether to patch Hermes core memory-provider discovery for `$HERMES_HOME/plugins/memory/<name>` or keep the flat symlink as the documented public install path for now.
2. Continue expanding live-service coverage for installed-plugin and real-subprocess compatibility paths.
3. Repeat consumer install/runtime smoke from the published tag before each future release.

This order keeps the project grounded: watcher/cron lifecycle is now operable without expanding Qdrant mutation authority; explicit manual store ergonomics now remain dry-run-first, and the next work improves runtime confidence before adding stronger memory mutation capabilities.
