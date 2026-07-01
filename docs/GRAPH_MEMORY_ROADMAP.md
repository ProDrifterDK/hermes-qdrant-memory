# Graph-Aware Memory Backlog and Roadmap

> **For Hermes:** This is the execution backlog for extending `hermes-qdrant-memory` with graph-aware retrieval primitives while preserving the plugin's safety contract: retrieved memory is context, not authority; broad mutation stays dry-run/review-gated; source/provenance beats model inference.

**Goal:** Add graph-aware retrieval, post-session improvement, feedback weighting, and schema/ontology-grounded extraction to the Hermes Qdrant memory plugin without replacing Qdrant, Vault, or LCM.

**Architecture stance:** Qdrant remains the primary local associative substrate. Graph memory starts as lightweight, source-backed payload/index primitives inside this plugin, not as a mandatory external graph database. The implementation must keep explicit provenance and review gates so graph edges never become unverified truth.

**Hard constraint:** Nothing in this workstream may depend on Cognee at runtime or as a required build/test dependency. Cognee was only prior art for comparison. These are new, self-contained features of `hermes-qdrant-memory`.

**Non-goals:**

- Do not replace LCM current-session recovery or context governance. HDFP is no longer an active/default architecture component; LCM is the sole active-session context engine.
- Do not introduce blind automatic ontology/schema mutation.
- Do not auto-promote extracted entities/edges/facts to canonical truth.
- Do not add broad query-based deletion or mutation.
- Do not require Obsidian, Cognee, Graphiti, Neo4j, or any external graph service for the default path.

---

## Backlog epics

### Epic G1 — Graph schema primitives

**Problem:** The plugin has relation enums and provenance edges, but no first-class entity/edge model suitable for graph-aware retrieval.

**Build:**

- Typed entity records with stable IDs, labels, aliases, entity type, source provenance, confidence, `canonical`, `requires_review`, and temporal fields.
- Typed edge records with subject/object entity IDs, relation type, source point IDs, confidence, feedback/usefulness weights, provenance, and fact status.
- Backward-compatible payload fields so legacy memory points still search normally.
- Explicit tests for secret/identity redaction and no automatic canonical promotion.

**Acceptance:** Entity/edge payload builders validate known relation/entity grammar, reject unsafe IDs, preserve provenance, and serialize deterministically for Qdrant upsert/search.

### Epic G2 — Graph-aware retrieval

**Problem:** Recall is currently semantic top-k plus provenance-aware reranking. It cannot expand through related entities/edges.

**Build:**

- Query planner that extracts candidate entities from query text and/or top semantic hits.
- Neighbor expansion over graph edge payloads using explicit Qdrant filters/scroll, bounded by depth and result caps.
- Hybrid reranker combining vector score, provenance rank, edge confidence, graph distance, feedback usefulness, fact status, and source health.
- Debug output showing vector candidates, graph expansions, and final score components.
- New read-only tool/CLI surface, e.g. `qdrant_memory_graph_search` / `hermes qdrant graph-search`.

**Acceptance:** A deterministic fixture can show a semantically weak but graph-connected point being promoted above unrelated high-vector-score candidates, with transparent ranking debug.

### Epic G3 — Post-session `improve()` / graph enrichment

**Problem:** Learning/source extraction candidates exist, but there is no explicit post-session enrichment pass that distills entities/edges and updates retrieval structures.

**Build:**

- Preview-first `qdrant_memory_improve_preview(session_id|source_scope)` that proposes entities, edges, feedback updates, and ontology/schema suggestions from completed session/source material.
- Exact-ID approval flow for applying safe graph updates: `qdrant_memory_improve_apply(candidate_id/report_id, dry_run=true, approve=false)`.
- Review/draft-only path for identity-bearing, secret-bearing, low-confidence, or ontology-changing proposals.
- Idempotency keys so repeated improve runs do not duplicate entities/edges.
- Session bridge semantics: useful session facts can become graph candidates, but not canonical without provenance and approval.

**Acceptance:** Improve preview produces a local report with proposed graph mutations and no Qdrant writes; live apply requires explicit approval and only upserts exact approved records.

### Epic G4 — Feedback weights over nodes/edges

**Problem:** Current ranking handles provenance/status but does not track which memories/edges were useful in answers.

**Build:**

- Separate feedback concepts:
  - `usefulness_weight`: answer/session utility signal.
  - `truth_confidence`: evidence/provenance confidence, never derived solely from usefulness.
  - `preference_weight`: user preference signal when applicable.
  - `staleness/source_health`: source validity signal.
- Feedback events linked to query/session/answer provenance without storing private raw prompts unnecessarily.
- Read-only feedback inspection and dry-run feedback application.
- Ranking integration that can boost useful edges without making them canonical truth.

**Acceptance:** Positive feedback changes future graph-aware ordering through `usefulness_weight`, while fact status/canonical/truth fields remain unchanged unless separately approved.

### Epic G5 — Ontologies / schemas for grounded extraction

**Problem:** Ontology suggestions are draft artifacts only. There is no runtime schema catalog that extraction can use to constrain entity/edge proposals.

**Build:**

- Versioned local schema catalog for entity types, relation types, allowed properties, aliases, and extraction prompts.
- Domain schema packs for initial internal use:
  - `memory-core`: MemoryPoint, Entity, Edge, Source, Session, FeedbackEvent.
  - `teamforge`: Task, Agent, Worktree, Review, QA, Blocker, Dependency.
  - `nucleogenesis`: Hypothesis, Mechanism, Experiment, Metric, Seed, Decision, FailureMode.
- Schema validation used during improve/extraction preview.
- Ontology changes remain suggestions until normal code/docs/tests update the catalog.

**Acceptance:** Improve/extraction can run with a named schema pack and reject/flag out-of-schema relation/entity proposals instead of silently storing arbitrary grammar.

---

## Roadmap

### Phase 0 — Architecture spike and safety baseline

**Objective:** Freeze contracts before code expansion.

Tasks:

1. Map current modules and tests touched by graph memory.
2. Define minimal graph data model and ranking contract.
3. Decide storage shape: same Qdrant collection with payload types vs separate graph collection(s).
4. Write fixture scenarios for graph-aware retrieval.
5. Verify no broad mutation or query deletion is needed.

Deliverables:

- `docs/GRAPH_MEMORY_ROADMAP.md` maintained as backlog source.
- Implementation design report from specialist swarm.
- Test fixture plan and acceptance criteria.

### Phase 1 — Graph schema foundation

**Objective:** Add safe entity/edge schema primitives and tests.

Likely files:

- `qdrant_memory/schema.py`
- `qdrant_memory/ranking.py`
- new `qdrant_memory/graph_schema.py` or similar
- `tests/test_graph_schema.py`

Acceptance:

- Entity/edge builders serialize to safe payloads.
- Relation/entity grammar is validated.
- Existing schema tests remain green.

### Phase 2 — Read-only graph expansion and hybrid reranking

**Objective:** Implement graph-aware retrieval with no mutation.

Likely files:

- `qdrant_memory/retriever.py`
- `qdrant_memory/ranking.py`
- `qdrant_memory/client.py`
- `qdrant_memory/tools.py`
- `qdrant_memory/cli_core.py`
- new `qdrant_memory/graph_retriever.py`
- `tests/test_graph_retriever.py`

Acceptance:

- Graph search is read-only.
- Bounded depth/candidate limits prevent runaway scroll.
- Debug output explains graph boosts and penalties.

### Phase 3 — Improve preview/apply pipeline

**Objective:** Add post-session enrichment as preview-first candidate workflow.

Likely files:

- `__init__.py`
- `qdrant_memory/extraction_candidates.py`
- `qdrant_memory/source_extraction.py`
- `qdrant_memory/write_gate.py`
- new `qdrant_memory/improve.py`
- `tests/test_improve_flow.py`

Acceptance:

- Preview creates report/candidates only.
- Apply requires exact candidate/report ID plus approval.
- Unsafe proposals route to draft review.

### Phase 4 — Feedback events and edge/node weighting

**Objective:** Store usefulness feedback separately from truth/canonical status and include it in ranking.

Likely files:

- new `qdrant_memory/feedback.py`
- `qdrant_memory/ranking.py`
- `qdrant_memory/tools.py`
- `qdrant_memory/cli_core.py`
- `tests/test_feedback_weights.py`

Acceptance:

- Feedback updates are auditable and reversible/reviewable.
- Usefulness affects ordering; truth/canonical fields do not change automatically.

### Phase 5 — Schema catalog and ontology-grounded extraction

**Objective:** Add versioned schemas and route extraction through schema validation.

Likely files:

- new `qdrant_memory/schema_catalog.py`
- new `qdrant_memory/schemas/*.json`
- `qdrant_memory/ontology_suggestions.py`
- `qdrant_memory/source_extraction.py`
- `tests/test_schema_catalog.py`

Acceptance:

- Named schema packs load deterministically.
- Out-of-schema proposals are rejected or flagged for review.
- Ontology mutation remains code/docs/tests, not runtime self-modification.

### Phase 6 — External interop later, not now

**Deferred:** COGX-style import/export, external memory-system interop, and visualization are intentionally deferred until graph primitives prove useful on local fixtures and Alan's real corpora. Any future interop must remain optional; no Cognee runtime/build/test dependency is allowed for this workstream.

---

## First execution wave

Use a collaborative fanout swarm rather than a single implementer:

- `architect`: refine graph-memory architecture and storage/ranking invariants.
- `backend-pro`: implement or draft Phase 1 graph schema primitives if safe within current repo state.
- `backend-medium` lane A: design/prepare graph-aware retrieval fixtures and minimal read-only expansion plan.
- `backend-medium` lane B: design/prepare improve/feedback/schema-catalog candidate flow.
- `reviewer` only after implementation evidence exists; do not run final approval in the same no-barrier wave.

Every lane must coordinate through `agent-swarm`, register artifacts, and write an auditable report/result JSON in the fanout workspace. Main Hermes verifies diffs/tests before merging or reporting completion.
