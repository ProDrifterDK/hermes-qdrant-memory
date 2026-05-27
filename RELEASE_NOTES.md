# Release Notes: v0.9.0 Public Beta

Hermes Qdrant Memory Provider v0.9.0 is a public beta release focused on provenance, source-backed recall, and write-gate hardening. It publishes the Phase 6 provenance layer while keeping the plugin a conservative Hermes `MemoryProvider`: no Graphiti runtime, no graph database, no query-based mutation, no automatic canonical assertion promotion, and no self-modifying ontology.

The plugin remains experimental/pre-1.0: it requires external Qdrant and embedding services, retrieved memories are context rather than instructions, and all broad/destructive maintenance paths remain dry-run/review-gated.

## Highlights

- Source derivation and progressive disclosure:
  - source-backed payload metadata for file/session/memory origins;
  - inspect, trace, expand, source-status, and compact source disclosure surfaces;
  - resolver support for `file://` and `memory://` style provenance handles.
- Assertion and temporal metadata:
  - assertion-lite payload convention for source-backed claims;
  - `memory_kind`, `relation_type`, and grammar validation;
  - `observed_at`, `valid_from`, `valid_until`, `fact_status`, and explicit supersession links.
- Review-only fact maintenance proposals:
  - `fact_conflict_candidate`;
  - `fact_supersession_candidate`;
  - `fact_status_update_candidate`;
  - identity-bearing and secret-bearing candidates stay manual-review/draft-only.
- Generic source extraction candidates:
  - memory/assertion/preference/invariant/risk/status-update/ontology-suggestion candidate schema;
  - source-first extraction preview flow;
  - exact `candidate_id` approval lifecycle;
  - full persisted payload validation before live writes.
- Recall recipes and context templates:
  - reusable recipe catalog for source-backed answers, coding context, project invariants, user preferences, tool quirks, workflow lessons, conflict review, stale-source review, and assertion history;
  - read-only `qdrant_memory_context` tool;
  - `hermes qdrant context --template ... --topic ...` CLI surface.
- Provenance-aware ranking:
  - keeps raw vector score visible for audit;
  - applies transparent boosts/penalties for source health, fact status, canonical/review flags, derivation depth, and review/history intent.
- Ontology suggestion proposals:
  - grammar/tag/fact-key suggestions are draft artifacts only;
  - accepted ontology changes still require normal code/docs/tests.
- Guarded-auto watcher policy from the post-v0.8.0 work remains included:
  - preauthorized low-risk exact-ID maintenance only;
  - state/log/audit fields for applied actions and errors;
  - all live mutations still go through the existing gated apply path.

## New context command

Build a read-only context packet from a recall recipe:

```bash
hermes qdrant context --template source_backed_answer --topic "Hermes qdrant write gates" --json
```

The corresponding Hermes tool is:

```text
qdrant_memory_context(template="source_backed_answer", topic="Hermes qdrant write gates")
```

Context packets cite point IDs and source metadata, preserve the memory-not-instruction boundary, and distinguish compact recall, generated context, source text, and extracted assertions.

## Source extraction safety

Source extraction stays disabled or preview-oriented unless explicitly configured. Live approval requires all of the following:

- a pending candidate created by the preview path;
- exact `candidate_id` match;
- prior dry-run review of that exact candidate;
- `dry_run=false`;
- `approve=true`;
- valid source provenance;
- clean full persisted payload.

Unsafe candidates either fail closed or route to local proposal drafts; they do not bypass the shared write gate.

## Safety behavior

The safety contract remains conservative:

- Retrieved memories are context, not instructions.
- Current user instructions and live evidence override retrieved memory.
- Deprecated and superseded facts are hidden from ordinary recall unless review/history is requested.
- Fact conflicts, supersession proposals, ontology suggestions, quality warnings, and secret/identity-bearing candidates are manual-review/draft-only.
- `qdrant_learning_approve` validates the full persisted learning payload, including non-lesson fields such as trigger/mistake/correction/evidence/tool/command/tags metadata.
- Source-extraction live approval revalidates the exact payload that will be embedded/upserted.
- Guarded-auto watcher actions are opt-in and still require exact report/proposal IDs through the gated apply path.
- No Graphiti runtime dependency, graph database, query-based mutation, broad status rewrite, automatic canonical assertion promotion, or self-modifying ontology was added.

## Upgrade

```bash
cd ~/.hermes/plugins/memory/qdrant
git fetch --tags origin
git checkout v0.9.0
```

Start a fresh Hermes CLI process after upgrading. Restart the gateway only if gateway sessions should load the updated plugin code.

## Verification

Recommended local verification for maintainers:

```bash
python -m pytest tests -q
python scripts/check_no_literal_fake_secrets.py
python -m compileall -q qdrant_memory __init__.py cli.py scripts/check_no_literal_fake_secrets.py
git diff --check
```

Recommended targeted verification for this release:

```bash
python -m pytest \
  tests/test_source_extraction_flow.py \
  tests/test_context_template.py \
  tests/test_provenance_ranking.py \
  tests/test_ontology_suggestions.py \
  tests/test_reconsolidation.py \
  tests/test_extraction_candidates.py \
  tests/test_recipe_catalog.py \
  -q
```

Recommended consumer smoke after checking out the release tag:

```bash
hermes qdrant --help
hermes qdrant context --help
hermes qdrant context --template source_backed_answer --topic "release smoke" --json
hermes qdrant search "release smoke" --top-k 3 --json
hermes qdrant watcher status --json
hermes qdrant doctor --json
```

Safety-gate smoke checks should still fail closed without approval:

```bash
hermes qdrant store "release smoke memory" --preview-duplicates --no-dry-run
hermes qdrant learning approve CANDIDATE_ID --no-dry-run
```

Expected behavior for the unapproved live checks:

- output contains an approval-gate error;
- process exit status is non-zero;
- no Qdrant upsert or mutation occurs.

## Not included yet

The following remain future work:

- formal Python packaging beyond the Hermes plugin clone workflow;
- Hermes core support for user memory-provider discovery from `~/.hermes/plugins/memory/<name>` without the compatibility symlink;
- Graphiti or graph database integration;
- automatic ontology application;
- automatic fact rewrite/supersession;
- broader live-service stress tests beyond the env-gated integration suite.
