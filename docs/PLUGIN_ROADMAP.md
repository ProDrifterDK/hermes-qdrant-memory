# Formal Hermes Plugin Roadmap

> **For Hermes:** Use `subagent-driven-development` if this roadmap is implemented task-by-task. Keep every mutating memory feature dry-run-first and review-gated.

**Goal:** Turn `hermes-qdrant-memory` from a working local/public-beta MemoryProvider into a formal, documented, installable, operable Hermes memory plugin.

**Architecture:** The plugin remains a Hermes `MemoryProvider`, not a `ContextEngine`. Qdrant handles cross-session semantic recall, indexed files, procedural learnings, and reviewable consolidation reports. LCM continues to own current-session lossless context recovery.

**Tech Stack:** Hermes Agent plugin API, Python 3.10+, Qdrant HTTP API, OpenAI-compatible embeddings, pytest, GitHub Actions, optional Hermes CLI plugin commands, optional Hermes cron/no-agent watcher.

---

## 1. Current state

The project is published as `v0.2.1 Public Beta Hotfix` and is functional as a Hermes Qdrant-backed memory provider. The latest release URL is:

- <https://github.com/ProDrifterDK/hermes-qdrant-memory/releases/tag/v0.2.1>

Implemented capabilities through v0.2.1:

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
- Conservative no-agent watcher script.
- Native Hermes memory-provider CLI MVP: `hermes qdrant ...`.
- GitHub Actions test workflow with pytest, compileall, and scanner guard.
- Release documentation: `CHANGELOG.md`, `RELEASE_NOTES.md`, install/update/remove/rollback notes.

Known compatibility detail:

- Desired category layout: `~/.hermes/plugins/memory/qdrant/`.
- Current Hermes compatibility path: `~/.hermes/plugins/qdrant` or symlink from `plugins/qdrant` to `plugins/memory/qdrant`.
- This is documented as a compatibility shim until Hermes core scans category paths for user memory providers.

Post-release validation status:

- Consumer install/runtime smoke from published `v0.2.0` completed and found one CLI process-exit propagation bug: unapproved live mutation was safely refused but exited with process status `0` under Hermes v0.13.
- The fix is released in `v0.2.1`: `qdrant_command()` raises `SystemExit(execute_command(args))` so usage/safety/provider errors propagate as non-zero CLI exits.
- Still verify whether current Hermes core can discover providers from `~/.hermes/plugins/memory/<name>` without the compatibility symlink.

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

The current stable surface is Hermes tool calls. The formal plugin should add a human-friendly CLI wrapper without weakening tool safety.

### M13 provisional target commands

These commands are target UX, not a confirmed Hermes plugin API contract. Task 7 must first verify whether user plugins can register CLI subcommands. If Hermes does not expose stable plugin CLI hooks yet, ship the same command surface as a temporary standalone wrapper and document it honestly.

Preferred command namespace:

```bash
hermes qdrant status
hermes qdrant doctor
hermes qdrant config show
hermes qdrant search "query text"
hermes qdrant store --text "..." --source-type manual --importance 5 --dry-run
hermes qdrant index --path ~/Documentos/Resyst\ Vault --dry-run
hermes qdrant forget --id POINT_ID --dry-run
hermes qdrant learning search "pytest hermes venv"
hermes qdrant learning store --lesson "..." --dry-run
hermes qdrant learning preview
hermes qdrant learning approve --candidate-id ID --dry-run
hermes qdrant consolidate --scope both --persist --include-reconsolidation --dry-run
hermes qdrant apply --report-id REPORT --proposal-id PROPOSAL --action merge --dry-run
hermes qdrant watcher status
hermes qdrant watcher run --dry-run
```

CLI rules:

- Mutating commands default to `--dry-run`.
- Live mutation requires both `--no-dry-run` and `--approve`.
- Apply commands require exact `report_id` and `proposal_id`.
- Delete commands require explicit point IDs.
- `quality_warning` proposals must refuse all live apply attempts.
- `draft_review` is the only allowed reconsolidation action and writes local markdown only.

Implementation options:

1. If Hermes plugin CLI hooks are available:
   - Add a `cli.py` or plugin registration hook that attaches subcommands.
2. If plugin CLI hooks are not available yet:
   - Provide a standalone script `scripts/hermes-qdrant-memory` or Python module entrypoint.
   - Keep docs explicit that this is a temporary wrapper.

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
- Keep `scripts/qdrant_sleep_consolidation.py` as the reference implementation.
- Add `hermes qdrant watcher status/run` if CLI support lands.

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

Status: partially completed in v0.2.0 as a native Hermes memory-provider CLI MVP. The implemented command surface is enough for beta operation, but not full parity with the original target UX.

Completed tasks:

1. Confirmed Hermes memory-provider CLI discovery path.
2. Added top-level `cli.py` and `qdrant_memory/cli_core.py`.
3. Implemented `status`, `doctor`, `search`, `index`, `forget`, `learning search`, `learning preview`, `consolidate`, and `apply`.
4. Preserved dry-run defaults and live-mutation approval gates.
5. Added CLI tests for parsing, command-to-tool mapping, import isolation, exit codes, and safety gates.

Deferred to v0.3.0+ CLI parity:

- `config show`
- `store`
- `learning store`
- `learning approve`
- `watcher status`
- `watcher run`
- optional backup/export commands

Verification:

```bash
hermes qdrant status
hermes qdrant search "Hermes Qdrant memory"
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
| 8 | CLI MVP | Completed | Release candidate | Adds human operation surface after safety gates. |
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

Status after v0.2.0: most criteria are satisfied by the published beta. The remaining operational checks are the post-release consumer install/runtime smoke and the category-path discovery verification.

- README documents preferred install path and compatibility symlink.
- Install, update, remove, and rollback docs exist for both preferred category path and current compatibility path.
- A verification step exists for detecting whether Hermes core can load providers from `plugins/memory/<name>` without a symlink.
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
| Use compatibility symlink until Hermes scans category paths | Accepted | Avoid Hermes core patch while preserving desired layout. |
| Add CLI wrapper | Accepted | Implemented as native memory-provider CLI MVP in v0.2.0. |
| Add safety/operations/LCM docs | Accepted | Implemented before widening the CLI/release surface. |
| Publish v0.2.0 as prerelease | Accepted | Honest beta label while the plugin remains experimental and externally service-dependent. |

---

## 13. Post-v0.2.0 roadmap

Recommended next phase: `v0.3.0` should focus on operational confidence, not conceptual expansion. The plugin already has the philosophical/safety core; the next work should make it easier to install, inspect, and keep healthy.

### Priority 1: Consumer install/runtime smoke

- Install or pin `~/.hermes/plugins/qdrant` to the published `v0.2.0` tag.
- Start a fresh Hermes CLI/gateway process.
- Verify `hermes qdrant status`, `doctor`, `search`, and `consolidate --persist`.
- Verify the tag works outside the development checkout.
- Document the exact commands and any gotchas in `docs/OPERATIONS.md` or `RELEASE_NOTES.md` for the next release.

### Priority 2: CLI parity

Add the deferred command groups without weakening safety gates:

- `hermes qdrant config show`
- `hermes qdrant store`
- `hermes qdrant learning store`
- `hermes qdrant learning approve`
- `hermes qdrant watcher status`
- `hermes qdrant watcher run`
- optional `backup` / `export` helpers before broader mutation surfaces

### Priority 3: Cron/watcher durability

- Add watcher state inspection.
- Add install/update/remove docs for the watcher.
- Add unchanged-signature silence regression tests.
- Add Qdrant/embedding outage failure-mode docs.
- Keep cron no-agent and report-only; no autonomous Qdrant mutation.

### Priority 4: Integration tests

Add optional live-service tests gated by explicit environment variables, e.g. `QDRANT_MEMORY_INTEGRATION=1`, so CI remains hermetic by default but maintainers can validate real Qdrant + embedding behavior before releases.

### Priority 5: Hermes core compatibility

Verify whether current Hermes core can discover user memory providers from `~/.hermes/plugins/memory/<name>` without a symlink. If not, keep the shim documented or open a Hermes core issue/PR.

---

## 14. Immediate next action

1. Run the post-release consumer install/runtime smoke test from the published `v0.2.0` tag.
2. Update `docs/OPERATIONS.md` with any runtime gotchas found.
3. Start v0.3.0 work with CLI parity and watcher durability.

This order keeps the project grounded: verify the published artifact first, then extend ergonomics. Do not add stronger memory mutation capabilities until backup/export and runtime smoke coverage exist.
