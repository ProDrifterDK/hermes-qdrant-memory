# Source Derivations and Progressive Disclosure Backlog

> **For implementers:** Treat this as a design backlog for the Hermes Qdrant memory plugin. Keep the core implementation application-agnostic: no note-taking app, vault format, or desktop application is required for the default path.

**Goal:** Add source-aware memory retrieval so Qdrant recall can move from compact semantic hints to inspectable provenance and expandable original sources without assuming Obsidian or any other external knowledge tool.

**Architecture:** Qdrant remains the associative memory and retrieval substrate. Source derivations describe where a memory came from, how it was derived, whether it is canonical or stale, and how to expand it through a generic resolver. Optional adapters can provide richer behavior for specific environments such as Obsidian, but the base plugin must work with generic files, sessions, URLs, skills, and manual memories.

**Primary inspirations:** xino-mem's progressive disclosure, source derivation chain-of-custody, corpus/source-first philosophy, write gates, and generated knowledge drafts; plus Graphiti's episode-first temporal knowledge graph discipline, typed entity/edge extraction, and fact invalidation patterns. These are patterns to adapt, not dependencies to adopt.

---

## 1. Design stance

The plugin should not be designed around Obsidian as a required layer.

Instead, the stable abstraction is:

```text
Qdrant memory point
  -> provenance metadata
  -> generic source resolver
  -> optional adapter-specific expansion
```

Obsidian can be a high-fidelity adapter over this model, but the default behavior must work for users with no Obsidian installation and no personal vault.

### Core principle

> The vector store is not the source of truth. It is an associative index over sources, derivations, and explicit memories.

### Required default capabilities

- Store and recall memory without any external corpus.
- Index generic Markdown/text files and directories.
- Preserve source metadata for conversations, files, manual memories, procedural learnings, and generated consolidation artifacts.
- Expand a recalled point into its source context when possible.
- Detect stale file-backed chunks when source hashes or timestamps change.
- Keep mutating operations dry-run-first and explicit-ID based.

### Optional adapter capabilities

Optional integrations may enrich source resolution for specific tools:

- Obsidian: vault root, wikilinks, frontmatter, tags, backlinks, note titles, MOC/proposal locations.
- Git repositories: commit, branch, file path, line range, blame/source hash.
- Web/cache: URL, retrieval timestamp, content hash, archived copy.
- Skills: skill name, file path, version, linked reference file.

These adapters must not be required for the base plugin to function.

---

## 2. Backlog overview

### Epics

1. **Source derivation schema** — stable metadata model for provenance and derivation chains.
2. **Progressive disclosure retrieval** — compact recall, inspect, trace, and expand flows.
3. **Generic source resolver** — adapter interface for sessions, files, URLs, skills, memory points, and optional apps.
4. **Source-aware file indexing** — robust line/heading/hash metadata for generic files.
5. **Write gate and derivation safety** — prevent low-quality, secret-bearing, or unsupported derived memories.
6. **Knowledge draft proposals** — neutral Markdown proposals that can optionally be routed into Obsidian or any corpus directory.
7. **Adapter layer** — optional integrations, starting with Obsidian only after the generic path is stable.
8. **CLI/tool ergonomics** — inspect, expand, trace, stale-check, and proposal commands/tools.
9. **Tests and docs** — regression coverage, examples, and migration notes.
10. **Temporal assertion layer** — optional assertion-lite payloads, memory grammar, and validity windows.
11. **General extraction candidates** — source-backed memory/assertion candidates that reuse write gates and proposal approval.
12. **Recall recipes and context templates** — reusable retrieval plans that compose search, inspect, trace, and expand.
13. **Provenance-aware ranking and ontology suggestions** — ranking that respects source health plus review-only grammar improvements.

---

## 3. Epic 1 — Source derivation schema

### Objective

Extend Qdrant payloads so every stored memory can identify its source, source location, derivation type, and expansion path.

### Proposed payload fields

```json
{
  "source_uri": "file:///home/user/notes/project.md",
  "source_type": "markdown",
  "locator": {
    "heading": "Architecture",
    "line_start": 42,
    "line_end": 88
  },
  "content_hash": "sha256:...",
  "source_modified_at": "2026-05-26T00:00:00Z",
  "derivation_type": "indexed_chunk",
  "derived_from": [
    {
      "source_uri": "session://2026-05-25/example-session",
      "locator": {"message_id": 12345},
      "derivation_type": "summary"
    }
  ],
  "canonical": true,
  "stale": false,
  "requires_review": false
}
```

### Field semantics

- `source_uri`: Stable URI for the original or canonical source.
- `source_type`: Normalized source category such as `conversation`, `manual`, `markdown`, `file`, `url`, `skill`, `learning`, `consolidation_report`, or `obsidian`.
- `locator`: Source-specific location object. Must remain JSON-serializable.
- `content_hash`: Hash of the source content or source excerpt used for the point.
- `source_modified_at`: Best-known source modification timestamp.
- `derivation_type`: How the memory point was created: `completed_turn`, `manual_fact`, `indexed_chunk`, `summary`, `consolidation_summary`, `learning`, `proposal`, etc.
- `derived_from`: Zero or more provenance edges to upstream sources or memory points.
- `canonical`: Whether this point directly reflects a canonical source rather than a generated summary or draft.
- `stale`: Whether the source appears changed, missing, or no longer matching the indexed hash.
- `requires_review`: Whether the point or proposal should not be used as canonical without human/agent review.

### Tasks

#### Task 1.1: Define typed provenance structures

**Files:**

- Modify: `qdrant_memory/schema.py`
- Test: `tests/test_schema.py` or new focused schema tests

**Acceptance criteria:**

- Dataclasses or typed helpers exist for source references, locators, and derivation edges.
- Existing payloads remain backward-compatible.
- Unknown/legacy points without `source_uri` still search and format correctly.

#### Task 1.2: Add payload builder support

**Files:**

- Modify: `qdrant_memory/schema.py`
- Modify: writer/indexer code paths that call `build_payload()`
- Test: writer and indexer tests

**Acceptance criteria:**

- `build_payload()` accepts source derivation metadata.
- Secret scanning applies to user-controlled string fields where appropriate.
- Metadata is omitted or normalized when empty instead of storing noisy null fields.

#### Task 1.3: Add migration-safe formatting

**Files:**

- Modify: `qdrant_memory/retriever.py`
- Modify: CLI/tool output formatting code
- Test: search/formatting tests

**Acceptance criteria:**

- Recall output includes compact provenance when available.
- Legacy points do not produce confusing `unknown` spam.
- The prompt-facing format distinguishes memory text from metadata.

---

## 4. Epic 2 — Progressive disclosure retrieval

### Objective

Separate fast semantic recall from deeper source inspection so the model can pull more context only when needed.

### Disclosure levels

```text
Level 1: recall/search
  Compact semantic result, short text, score, source summary, flags.

Level 2: inspect
  Full point payload, provenance metadata, derivation edges, source status.

Level 3: trace
  Upstream/downstream derivation chain, related points, canonical source hints.

Level 4: expand
  Source excerpt or full source context through the relevant resolver.
```

### Tool/CLI candidates

```text
qdrant_memory_inspect(point_id)
qdrant_memory_trace(point_id, direction="upstream|downstream|both")
qdrant_memory_expand(point_id, mode="excerpt|source|neighbors")
qdrant_memory_source_status(point_id)
```

CLI equivalents:

```bash
hermes qdrant inspect <point-id>
hermes qdrant trace <point-id>
hermes qdrant expand <point-id> --mode excerpt
hermes qdrant source-status <point-id>
```

### Tasks

#### Task 2.1: Implement exact point inspection

**Files:**

- Modify or extend: `qdrant_memory/client.py`
- Modify: `qdrant_memory/cli_core.py`
- Modify: `qdrant_memory/tools.py`
- Test: CLI and tool tests

**Acceptance criteria:**

- Inspecting a point by explicit ID returns full payload metadata.
- Missing points produce a clear non-zero CLI/tool error.
- No query-based mutation or deletion is introduced.

#### Task 2.2: Add derivation trace output

**Files:**

- Modify: retrieval/client layer
- Modify: CLI/tool schemas
- Test: trace tests with synthetic derived chains

**Acceptance criteria:**

- Trace shows direct upstream `derived_from` links.
- Trace handles missing upstream points/sources gracefully.
- Output is compact by default and has JSON mode for automation.

#### Task 2.3: Add source expansion by point ID

**Files:**

- Create or modify: `qdrant_memory/sources.py`
- Modify: CLI/tool schemas
- Test: file/session/manual source expansion tests

**Acceptance criteria:**

- `expand` works for at least manual memory points and file-backed indexed chunks.
- Expansion has size limits and never dumps unbounded files into context.
- Expansion reports when a source cannot be resolved.

---

## 5. Epic 3 — Generic source resolver

### Objective

Introduce an application-agnostic resolver interface that can expand and stat different source URI schemes.

### Proposed interface

```python
class SourceResolver(Protocol):
    schemes: set[str]

    def stat(self, source_uri: str, locator: dict[str, Any] | None = None) -> SourceStatus:
        ...

    def expand(
        self,
        source_uri: str,
        locator: dict[str, Any] | None = None,
        *,
        mode: str = "excerpt",
        max_chars: int = 8000,
    ) -> SourceExpansion:
        ...
```

### Initial resolver schemes

- `memory://point/<id>` — exact Qdrant point lookup.
- `file://...` — local file excerpts by line range, heading, or hash.
- `session://...` — session/message lookup when Hermes session DB support is available.
- `skill://...` — installed skill and linked-file lookup when skill metadata is available.
- `url://...` or `https://...` — optional cached/fetched URL expansion, read-only and bounded.
- `obsidian://...` — optional adapter, not part of required default path.

### Tasks

#### Task 3.1: Implement resolver registry

**Files:**

- Create: `qdrant_memory/sources.py`
- Test: `tests/test_sources.py`

**Acceptance criteria:**

- Resolvers are registered by URI scheme.
- Unknown schemes return structured unsupported-source errors.
- Resolver calls enforce max character budgets.

#### Task 3.2: Implement file resolver

**Files:**

- Modify: `qdrant_memory/sources.py`
- Test: file expansion tests with line ranges and headings

**Acceptance criteria:**

- Supports `file://` URIs.
- Can expand by line range.
- Can report missing/changed/stale files.
- Refuses to read binary or very large files unbounded.

#### Task 3.3: Implement memory resolver

**Files:**

- Modify: `qdrant_memory/sources.py`
- Test: exact point source expansion tests

**Acceptance criteria:**

- Supports `memory://point/<id>`.
- Returns point payload and text without semantic search.
- Does not recursively expand indefinitely.

---

## 6. Epic 4 — Source-aware file indexing

### Objective

Improve generic file indexing so each chunk can later be expanded, checked for staleness, and traced back to an exact source region.

### Tasks

#### Task 4.1: Add chunk locator metadata

**Files:**

- Modify: `qdrant_memory/indexer.py`
- Modify: tests for file indexing

**Acceptance criteria:**

- Indexed chunks include `source_uri`, `locator.line_start`, `locator.line_end`, and optional `locator.heading`.
- Markdown heading chunking preserves heading hierarchy where possible.
- Existing file manifest sync remains compatible.

#### Task 4.2: Add source hash strategy

**Files:**

- Modify: `qdrant_memory/indexer.py`
- Modify: `qdrant_memory/schema.py`
- Test: stale-detection tests

**Acceptance criteria:**

- Each indexed chunk stores a stable content hash.
- File-level hashes remain available for manifest sync.
- Staleness can be detected without re-embedding every unchanged chunk.

#### Task 4.3: Add source-status command

**Files:**

- Modify: CLI/tool surfaces
- Test: source-status CLI tests

**Acceptance criteria:**

- Reports source exists/missing/changed/unknown.
- Distinguishes changed source from missing source.
- Does not mutate Qdrant by default.

---

## 7. Epic 5 — Write gate and derivation safety

### Objective

Before storing derived memories, classify whether the content should become a memory, learning, skill candidate, proposal draft, or be rejected/no-op.

### Candidate gates

- Secret or credential detection.
- Duplicate semantic similarity.
- Low information/trivial utterance detection.
- Unsupported source or missing provenance.
- Contradiction/conflict with existing canonical facts.
- Confidence below configured threshold.
- Derivation chain too long or too lossy.

### Tasks

#### Task 5.1: Define write decision object

**Files:**

- Create or modify: `qdrant_memory/write_gate.py`
- Test: write-gate classification tests

**Acceptance criteria:**

- Gate returns structured decisions: `store`, `skip`, `draft_review`, `learning_candidate`, `skill_candidate`, `reject`.
- Decisions include reasons and confidence.
- Existing manual store dry-run/duplicate-preview behavior remains compatible.

#### Task 5.2: Apply gate to derived/consolidated writes

**Files:**

- Modify: consolidation and learning candidate code paths
- Test: consolidation apply tests

**Acceptance criteria:**

- Derived summaries without provenance are review-only by default.
- Secret-bearing content is rejected or manual-review only.
- Low-risk exact duplicate/noise cleanup remains explicit-ID based.

---

## 8. Epic 6 — Knowledge draft proposals

### Objective

Allow consolidation to produce neutral Markdown drafts without requiring Obsidian.

### Default proposal path

```text
~/.hermes/qdrant-memory/proposals/
```

Profile-aware implementations should use the active Hermes home, not a hardcoded `~/.hermes` path.

### Proposal types

- Entity/topic summary draft.
- Memory conflict review draft.
- Duplicate cluster review draft.
- Learning-to-skill candidate draft.
- Source cleanup/staleness review draft.

### Tasks

#### Task 6.1: Add neutral proposal writer

**Files:**

- Create or modify: `qdrant_memory/proposals.py`
- Modify: consolidation apply/report code where appropriate
- Test: proposal writer tests

**Acceptance criteria:**

- Writes Markdown drafts under a profile-safe proposal directory.
- Includes source point IDs and derivation metadata.
- Does not require Obsidian.

#### Task 6.2: Add proposal listing/inspection

**Files:**

- Modify: CLI output helpers
- Test: proposal CLI tests

**Acceptance criteria:**

- Users can list and inspect proposal drafts.
- JSON mode returns stable fields for automation.
- Human-readable mode remains concise.

---

## 9. Epic 7 — Optional adapters

### Objective

Support richer integrations without coupling the core plugin to a specific external app.

### Adapter contract

Adapters may provide:

- Source URI normalization.
- Locator enrichment.
- Expansion behavior.
- Proposal path routing.
- Optional graph/backlink metadata.

Adapters must not:

- Be required for default install.
- Change core payload semantics.
- Mutate user files without explicit approval.
- Replace the generic resolver path.

### Initial optional adapter candidates

#### Obsidian adapter

Capabilities:

- Detect configured vault root.
- Normalize note paths to `obsidian://` or enriched `file://` source URIs.
- Parse frontmatter/tags/wikilinks as optional metadata.
- Route proposal drafts into a configured vault-relative folder.

Default behavior:

- Disabled unless explicitly configured.
- Falls back to generic `file://` Markdown behavior when disabled.

#### Git adapter

Capabilities:

- Add repo root, commit SHA, branch, and file path metadata for indexed repo docs.
- Expand file excerpts at current worktree state.
- Optionally flag when indexed commit differs from current checkout.

#### URL/cache adapter

Capabilities:

- Track fetched URL, retrieval timestamp, hash, and cached artifact path.
- Expand from cache first.
- Avoid unbounded live network fetches during normal recall.

---

## 10. Epic 8 — CLI and tool ergonomics

### Objective

Expose source-aware behavior through both Hermes tools and native CLI commands.

### Proposed commands

```bash
hermes qdrant inspect <point-id> [--json]
hermes qdrant trace <point-id> [--direction upstream|downstream|both] [--json]
hermes qdrant expand <point-id> [--mode excerpt|source|neighbors] [--max-chars 8000]
hermes qdrant source-status <point-id> [--json]
hermes qdrant proposals list [--json]
hermes qdrant proposals show <proposal-id> [--json]
```

### Proposed tools

```text
qdrant_memory_inspect
qdrant_memory_trace
qdrant_memory_expand
qdrant_memory_source_status
```

### Acceptance criteria

- Every read-only command has stable JSON output.
- Human-readable output is concise and does not dump large sources by default.
- Mutation commands remain dry-run-first and require explicit approval.
- Tool descriptions clearly state whether they are read-only or mutating.

---

## 11. Epic 9 — Tests and documentation

### Objective

Make source derivations safe, public-plugin friendly, and maintainable.

### Required tests

- Backward compatibility with legacy payloads.
- Payload builder source metadata tests.
- File resolver line range and stale-source tests.
- Progressive disclosure CLI tests.
- Tool schema tests.
- Proposal writer tests.
- Adapter-disabled default-path tests.
- Obsidian adapter tests must be optional and use temp directories only.
- Secret/noise/write-gate tests.

### Required docs

- Update `README.md` capabilities after implementation.
- Update `docs/ARCHITECTURE.md` with source resolver architecture.
- Update `docs/SAFETY.md` with source expansion and proposal-writing boundaries.
- Update `docs/OPERATIONS.md` with inspect/trace/expand workflows.
- Add examples to `docs/EXAMPLES.md`.

### Verification commands

Use the repository's existing test style. Candidate commands:

```bash
python -m pytest tests/test_cli.py tests/test_consolidation.py -q
python -m pytest tests/test_sources.py tests/test_schema.py -q
python -m compileall qdrant_memory
```

Broader release validation should continue using the existing project workflow and CI.

---

## 12. Graphiti-inspired extension backlog — Temporal Assertion Layer

This section captures the Graphiti-inspired patterns that are not yet fully represented by Epics 1-9. The intent is to add typed, temporal, source-backed assertions on top of Qdrant's existing associative memory model without introducing Graphiti, a graph database, or blind automatic fact rewriting as runtime dependencies.

### Relationship to existing backlog

The current backlog already covers the core substrate that these features depend on:

- Epics 1-4 cover provenance, source derivation chains, resolvers, source hashes, stale-source detection, and expandable original source excerpts.
- Epics 5-6 cover write gates and neutral proposal drafts, which are the correct place for derived assertions and extraction candidates to pass through review.
- Epic 8 covers the CLI/tool surfaces that should expose inspect/trace/expand/status workflows before any higher-level context template is added.

The missing layer is not "more semantic search." The missing layer is a small, typed assertion model that can represent what a remembered fact means, when it is valid, what source supports it, whether another fact supersedes it, and how safely it may be used.

### Design stance

Adopt a **temporal assertion layer**, not a full knowledge graph:

```text
source episode / file / memory point
  -> provenance edge
  -> optional assertion-lite point
  -> review-gated status/supersession metadata
  -> recall recipe that chooses compact, source-backed context
```

Rules:

- Qdrant remains the storage and nearest-neighbor substrate.
- Source derivations remain the first dependency; assertion features should not ship before inspect/trace/expand are stable.
- Assertions are payload conventions and tools, not a mandatory graph runtime.
- Automatic extraction may propose candidates, but must not store, rewrite, supersede, or delete memories without explicit approval.
- Ontology changes are review-only proposals, not self-modifying schema changes.

### Epic 10 — Temporal assertion layer

**Objective:** Add a minimal, review-safe grammar for durable facts so the plugin can distinguish raw/source chunks from typed assertions, decisions, preferences, risks, and invalidated facts.

#### Task 10.1: Define memory grammar enums

**Files:**

- Modify: `qdrant_memory/schema.py`
- Modify: `docs/SOURCE_DERIVATIONS_BACKLOG.md`
- Test: `tests/test_schema.py`

**Fields:**

```text
memory_kind:
  conversation_turn
  manual_fact
  source_chunk
  learning
  assertion
  decision
  user_preference
  project_invariant
  tool_quirk
  workflow_lesson
  risk
  proposal
  summary

relation_type:
  DERIVED_FROM
  EXTRACTED_FROM
  SUMMARIZES
  SUPPORTS
  CONTRADICTS
  SUPERSEDES
  REFERENCES
  APPLIES_TO
  USES_TOOL
  PREFERS
  BLOCKS
```

**Rules:**

- `memory_kind` is optional for legacy payloads.
- New derived writes should set `memory_kind` when known.
- `relation_type` applies to derivation/relationship edges, not to vector search itself.
- Unknown kinds should fail validation for new generated writes, but legacy points without the field remain readable.

**Acceptance criteria:**

- Existing memories without `memory_kind` still inspect/search normally.
- New payload builders can validate known `memory_kind` and `relation_type` values.
- Formatting surfaces these fields only when present, without flooding recall output.

#### Task 10.2: Add assertion-lite payload convention

**Objective:** Represent source-backed factual claims as atomic optional points rather than only as free-text chunks.

**Proposed fields:**

```json
{
  "memory_kind": "assertion",
  "claim_text": "Qwen3-TTS ROCm is unsafe on Alan's display GPU unless explicitly isolated.",
  "subject": "Qwen3-TTS ROCm",
  "predicate": "is_unsafe_on",
  "object": "Alan display GPU",
  "confidence": 0.86,
  "evidence": [
    {
      "source_uri": "session://...",
      "locator": {"message_id": 12345},
      "relation_type": "SUPPORTS"
    }
  ],
  "derived_from": ["memory://point/<source-point-id>"],
  "canonical": false,
  "requires_review": true
}
```

**Rules:**

- Assertions must be evidence-backed through `source_uri`, `locator`, or `derived_from`.
- First implementation should create assertion candidates/proposals, not live assertion points by default.
- `claim_text` is the human-readable assertion; `subject/predicate/object` is the retrieval/filter grammar.
- `confidence` is an extraction confidence, not truth.

**Acceptance criteria:**

- A source-backed assertion can be inspected to show claim, evidence, and source chain.
- Assertion points can be filtered by `memory_kind=assertion`, `subject`, and `requires_review`.
- No assertion can become canonical solely because it was extracted by an LLM or heuristic.

#### Task 10.3: Add temporal validity metadata

**Objective:** Represent when a fact was observed, when it applies, and whether it is currently usable.

**Proposed fields:**

```text
observed_at
valid_from
valid_until
fact_status: active | stale | deprecated | disputed | superseded | review_required
supersedes: [point_id]
superseded_by: [point_id]
invalidated_by: [point_id]
```

**Rules:**

- `created_at` remains the write timestamp; it is not a validity window.
- `stale=true` describes source freshness; `fact_status` describes assertion usability.
- `valid_until` may be unknown/null.
- Supersession links must use explicit point IDs.
- No automatic fact rewrite is allowed; changing status requires a proposal/apply flow or explicit manual tool.

**Acceptance criteria:**

- Search can optionally hide `fact_status=superseded|deprecated` by default.
- Inspect/trace can show supersession history.
- Conflicting active assertions are surfaced as review candidates rather than silently resolved.

#### Task 10.4: Add conflict and supersession proposals

**Objective:** Turn detected contradictions or newer facts into reviewable proposals instead of mutating memory directly.

**Proposal types:**

```text
fact_conflict_candidate
fact_supersession_candidate
fact_status_update_candidate
```

**Rules:**

- Proposals must include affected point IDs, source snippets, proposed status changes, and risk/confidence.
- Live apply, if added later, must require exact `report_id`, exact `proposal_id`, compatible action, `dry_run=false`, and `approve=true`.
- Initial implementation may stop at draft artifacts only.

**Acceptance criteria:**

- Conflicts are visible without increasing memory mutation authority.
- Operators can see why one assertion may supersede another.
- Secret-bearing or identity-bearing conflicts remain manual-review only.

### Epic 11 — General extraction candidates

**Objective:** Generalize the current gated learning candidate pattern to source-backed memory and assertion candidates.

#### Task 11.1: Define generic extraction candidate schema

**Candidate types:**

```text
memory_candidate
assertion_candidate
preference_candidate
invariant_candidate
risk_candidate
status_update_candidate
ontology_suggestion
```

**Required fields:**

```text
candidate_id
candidate_type
source_uri
locator
derived_from
proposed_payload
reason
confidence
risk
requires_review
created_at
```

**Acceptance criteria:**

- Candidates can be previewed without constructing embeddings or mutating Qdrant.
- Candidate IDs are stable enough for exact approval during a single pending-buffer lifecycle.
- Candidate payloads validate against the memory grammar before approval.

#### Task 11.2: Implement source-first extraction flow

**Flow:**

```text
source text / recalled point / completed turn
  -> extractor emits candidates
  -> write gate validates candidate kind, provenance, confidence, and secret risk
  -> preview/list shows candidates
  -> approve creates memory/assertion/proposal only through explicit approval
```

**Rules:**

- Default remains disabled or dry-run.
- Extraction must prefer explicit user corrections, decisions, tool quirks, project invariants, and resolved conflicts over generic summaries.
- Low-confidence candidates become draft proposals, not live memory points.
- The extractor must not infer secrets, credentials, private identity expansions, or unsupported conclusions.

#### Task 11.3: Reuse write gate and proposal infrastructure

**Objective:** Avoid a separate mutation path for assertions.

**Rules:**

- `write_gate.py` should validate candidate risk, provenance, and destination.
- `proposals.py` should handle draft-only output when live writes are not safe.
- Approval path must use exact candidate/proposal IDs.

**Acceptance criteria:**

- Existing learning candidate behavior remains unchanged.
- New memory/assertion candidates cannot bypass dry-run-first safety.
- Tests cover approval refusal for missing source provenance and secret-bearing payloads.

### Epic 12 — Recall recipes and context templates

**Objective:** Provide reusable retrieval plans that compose search, filters, inspect, trace, and expand into predictable context packets.

#### Task 12.1: Define recipe catalog

**Initial recipes:**

```text
source_backed_answer
coding_task_context
project_invariants
user_preferences
tool_quirks
workflow_lessons
conflict_review
stale_source_review
assertion_history
```

**Rules:**

- Recipes are retrieval plans, not new memory authority.
- Recipes must declare which collections, filters, expansion budget, and status filters they use.
- Recipes should prefer compact recall first, then inspect/trace/expand only when needed.

#### Task 12.2: Add context template command/tool only after disclosure tools stabilize

**Potential surface:**

```text
qdrant_memory_context(template="source_backed_answer", topic="...")
hermes qdrant context --template source_backed_answer --topic "..." --json
```

**Acceptance criteria:**

- Output cites point IDs and source URIs.
- Output flags stale/review-required/conflicting assertions.
- Output does not hide the distinction between source text, generated summary, and extracted assertion.

### Epic 13 — Provenance-aware ranking and ontology suggestions

**Objective:** Improve recall quality using source health and typed metadata while keeping ontology changes review-gated.

#### Task 13.1: Add provenance-aware ranking policy

**Ranking inputs:**

```text
vector_score
importance
recency_decay
memory_kind
fact_status
canonical
stale
requires_review
source_hash_current
derivation_depth
exact subject/fact_key match
```

**Rules:**

- Penalize `stale=true`, `requires_review=true`, `fact_status=disputed|superseded|deprecated` unless the query asks for review/history.
- Boost `canonical=true`, fresh source hash, exact source/project filters, and exact subject/fact-key matches.
- Prefer direct source-backed assertions over long derivation chains when confidence is equal.
- Keep raw vector score visible in JSON/debug output so ranking remains auditable.

#### Task 13.2: Add ontology suggestion proposals

**Objective:** Let the system suggest grammar improvements without self-modifying the schema.

**Proposal examples:**

```text
new memory_kind candidate
new relation_type candidate
merge/rename tags
normalize subject aliases
promote repeated fact_key pattern
```

**Rules:**

- Ontology suggestions are proposal/draft artifacts only.
- No auto-application in cron/watchers.
- Accepted ontology changes must be implemented through normal code/docs changes and tests.

### Safety constraints for the temporal assertion layer

- No graph database dependency for the base plugin.
- No Graphiti runtime dependency.
- No blind LLM extraction into live memory.
- No auto-rewrite of canonical facts.
- No query-based deletion or broad status mutation.
- No assertion without provenance.
- No self-modifying ontology.
- No current-session replacement: LCM/HDFP still owns active-session recovery.

---

## 13. Suggested implementation phases

### Phase 1 — Foundation

- Epic 1: source derivation schema.
- Epic 4: source-aware generic file indexing.
- Minimal formatting updates for recall output.

**Exit criteria:** New payload fields exist, legacy points work, file-backed chunks include expandable source locators.

### Phase 2 — Progressive disclosure

- Epic 2: inspect, trace, expand.
- Epic 3: resolver registry with `file://` and `memory://` support.
- CLI/tool read-only surfaces.

**Exit criteria:** A recalled point can be inspected and expanded without semantic search or external apps.

### Phase 3 — Safety and proposals

- Epic 5: write gate.
- Epic 6: neutral proposal writer.
- Consolidation reports can produce reviewable source-aware drafts.

**Exit criteria:** Derived memories and drafts preserve chain-of-custody and require review when confidence/provenance is weak.

### Phase 4 — Optional adapters

- Epic 7: optional Obsidian adapter.
- Optional Git and URL/cache adapters after the core resolver is stable.

**Exit criteria:** Obsidian improves behavior when configured, but disabling it returns to generic file/source behavior with no broken features.

### Phase 5 — Documentation and release hardening

- Epic 9 docs and examples.
- Consumer install smoke.
- CLI output contract updates if needed.

**Exit criteria:** Public users can understand and use source derivations without knowing anything about Obsidian.

### Phase 6 — Temporal assertions and retrieval grammar

- Epic 10: memory grammar, assertion-lite payloads, and temporal validity metadata.
- Epic 11: generic memory/assertion extraction candidates through the existing write gate.
- Epic 12: recall recipes/context templates after inspect/trace/expand are stable.
- Epic 13: provenance-aware ranking and review-only ontology suggestions.

**Exit criteria:** Source-backed assertions can be proposed, inspected, traced, ranked, and supersession-reviewed without adding a graph runtime, weakening dry-run defaults, or auto-rewriting canonical memories.

---

## 14. Non-goals

- Do not adopt xino-mem as a runtime dependency.
- Do not adopt Graphiti or any graph database as a runtime dependency for the base plugin.
- Do not require Obsidian or any note-taking app.
- Do not replace LCM/HDFP current-session recovery.
- Do not make Qdrant memories instruction-authoritative.
- Do not add query-based deletion.
- Do not auto-rewrite canonical facts through reconsolidation or temporal assertions.
- Do not allow generated assertions without provenance.
- Do not auto-apply ontology changes suggested by the system.
- Do not write into user files or vaults without explicit approval.
- Do not build a full knowledge graph before source derivations and progressive disclosure are stable.

---

## 15. Open questions

1. Should `source_uri` use custom schemes such as `session://` and `memory://`, or should these be represented as structured payload fields only?
2. Should exact point inspection be added to existing tools first, or only through native `hermes qdrant` CLI commands?
3. How much session expansion should the plugin own versus delegating to Hermes session search/LCM tools?
4. Should generated proposal drafts be stored in Qdrant, on disk, or both?
5. What is the minimum useful stale-source detection for v1: file missing/modified only, or excerpt hash mismatch too?
6. Should optional adapters live inside the plugin or as separate plugin modules later?
7. What should be the default max expansion budget for prompt-facing tools?
8. What is the smallest useful `memory_kind` vocabulary for v1 without overfitting to Alan's current workflows?
9. Should assertion-lite points live in the existing memory collection or a separate assertion collection?
10. Should supersession/status updates mutate payloads directly after approval, or create replacement assertion points plus links only?
11. Which recall recipes are worth exposing as tools versus documenting as CLI/operator patterns first?
12. How should provenance-aware ranking balance raw vector score against freshness, canonicality, and review flags?
13. Should ontology suggestions be persisted as local proposal artifacts only, or also indexed as reviewable Qdrant points?

---

## 16. Success criteria

The feature set is successful when:

- Search results provide useful compact recall without flooding context.
- Any important recalled point can be inspected for provenance.
- File-backed memories can be expanded to their original source excerpt.
- Derived memories show a clear chain-of-custody.
- Stale or missing sources are visible to the user/agent.
- Source-backed assertions can be represented without losing their source evidence.
- Temporal validity and supersession state are visible without automatic fact rewriting.
- Recall can prefer canonical, fresh, source-backed facts while flagging stale or review-required material.
- Extraction candidates remain preview/approval gated and cannot bypass write safety.
- Obsidian-specific behavior is optional and never required for public plugin users.
- The plugin remains a Hermes MemoryProvider, not a replacement for LCM/HDFP or a full note-taking system.
