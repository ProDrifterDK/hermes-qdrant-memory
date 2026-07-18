# OpenAI Build Week 2026: Memory PR

Product: **Memory PR**

Track: **Developer Tools**
Pitch: **Git made code changes reviewable. Memory PR makes persistent agent memory reviewable.**

## Frozen eligibility baseline

- Tag: `build-week-2026-baseline`
- SHA: `cc7ade5e47aaadb9c2f70eeb164696c08f42e4f2`
- Contest branch: `feature/openai-build-week-memory-pr`

All Memory PR implementation, tests, and documentation are commits after that frozen baseline.

## What existed before the baseline

The baseline already provided the durable contracts that Memory PR deliberately reuses:

- persisted consolidation reports with exact `report_id` and stable proposal IDs;
- exact proposal lookup and exact current-point retrieval;
- proposal types for duplicate, stale, learning-promotion, quality, conflict, supersession, and fact-status review;
- source/provenance and canonical/stale/review/fact-status payload fields;
- recursive secret redaction and identity-bearing memory classification;
- current-point revalidation in guarded apply paths;
- manual-review reconsolidation drafts;
- dry-run-first, exact-ID, action-match, approval, and audit gates for the separate apply surface.

The baseline did not provide a deterministic portable packet, report-versus-current drift presentation, polished static review experience, or dependency-free public demo for one exact proposal.

## What Memory PR adds

Memory PR adds:

- a versioned JSON packet with deterministic `memory_pr_id` and `content_digest`;
- digest-only point snapshots on newly generated consolidation proposals, bound to the versioned review projection used to compute them;
- current exact-point reload with strict affected-ID equality;
- per-point and overall `unchanged`, `changed`, or conservative `unknown` drift labels;
- bounded sanitized current evidence, provenance, and visible canonical/stale/review/fact status;
- one bounded fail-closed identity classifier shared by current payloads, snapshots/digests, snippets, proposal narratives, status reasons, and recursively nested persisted evidence;
- a self-contained escaped HTML review artifact with restrictive CSP, no JavaScript, and no external resources;
- private explicit artifact persistence (`0700` directory, `0600` files where supported);
- the public read-only `qdrant_memory_memory_pr` provider tool;
- an offline deterministic synthetic fact-supersession fixture and verifier.

It does not widen any consolidation or guarded-auto mutation authority.

## Judge quickstart: under five minutes

From the repository root with Python 3.10 or newer:

```bash
python3 -m qdrant_memory.memory_pr fixture --output-dir /tmp/hermes-memory-pr-demo --overwrite
python3 -m qdrant_memory.memory_pr verify-fixture
```

The first command writes one private JSON file and one private HTML file under `/tmp/hermes-memory-pr-demo`. Open the generated `memory-pr-*.html` file in a browser. The second command generates the fixture twice in disposable directories and verifies the Memory PR ID, content digest, JSON bytes, and HTML bytes all match.

This path imports only Python standard-library code and repository modules. It does not initialize Hermes, contact Qdrant, request embeddings, read Obsidian, use a network, or include private Alan data.

## Architecture and data flow

The reusable module is `qdrant_memory/memory_pr.py`:

1. Validate canonical exact report/proposal IDs.
2. Select exactly one proposal and validate its unique affected IDs.
3. Accept only currently reloaded exact points whose ID set equals the proposal set.
4. Classify nested mappings/lists once through the shared bounded identity policy, including existing profile/fact markers and normalized identity-sensitive keys; suppress complete sensitive records before hashing or output.
5. Require every persisted evidence item to name an exact affected point; reject unattributed/unknown IDs and replace identity-bearing records recursively as a whole.
6. Recursively redact and bound proposal/current evidence.
7. Recompute the versioned review-relevant point projection and compare it with compatible persisted report snapshots. Version 1 includes text/provenance/review state and excludes access/ranking bookkeeping; its version must be an integer, never a JSON boolean.
8. Hash canonical stable review content, excluding timestamps and paths.
9. Return the JSON packet and optionally render/write private JSON and HTML artifacts.

The provider handler performs only a non-creating report path resolution/read, exact proposal selection, strict configured memory/learning collection validation, and `QdrantClient.retrieve(..., with_payload=True, with_vector=False)`. The pure builder and fixture path have no Qdrant client dependency.

## Non-mutation boundary

Packet generation never calls upsert, payload update, delete, filter delete, learning approval, embedding, search, or consolidation apply. It never alters report artifacts, current memory points, sources, or user files. A missing report read creates no directory and changes no permissions. The only allowed write is the explicitly requested artifact pair in `output_dir`; no path is persisted inside the packet or HTML. A new output directory is mode `0700`; a pre-existing directory must already be current-user-owned mode `0700` and is never chmodded.

The packet includes a reviewer checklist and serialized arguments for the existing exact-proposal dry-run gate. Those arguments are evidence for a possible next review step and are not executed.

## HTML review experience

The renderer uses a Resyst-style dark/void visual system with restrained amber evidence accents. It provides semantic landmarks, a skip link, structured headings, textual drift labels, high-contrast badges, responsive evidence cards, horizontally safe tables, visible focus treatment, forced-colors support, reduced-motion behavior, and print styles. All untrusted values pass through HTML escaping. The CSP denies scripts, connections, frames, objects, media, fonts, images, forms, and base-URL changes.

## How Codex GPT-5.6 was used

Codex GPT-5.6 was used in this official project thread to inspect the frozen repository architecture, derive the smallest compatible extension, write test-first increments, implement the packet builder/provider/renderer/fixture, investigate regression behavior, run verification, and draft technical documentation.

The human/product decisions remained human: the Memory PR pitch and track, the frozen baseline, the non-mutation promise, exact-ID requirement, evidence fields, safety bar, visual direction, offline judge requirement, contest scope, and explicit cut of native Hermes CLI integration. Codex implemented and validated those decisions; it did not redefine the product authority or approve any memory mutation.

## Known limitations and explicit cuts

- Legacy reports without Memory PR point snapshots produce `unknown` drift, never a false `unchanged` claim.
- Reports with unversioned or unsupported review snapshot projections also produce `unknown` drift rather than comparing incompatible digests.
- Projection version 1 intentionally ignores operational access/ranking fields. Changes to those fields alone are not review drift and do not create a new Memory PR identity.
- Identity- or secret-bearing content is replaced before hashing; those sensitive points report `unknown` content drift while still exposing safe state/provenance changes. This avoids publishing a stable digest that could aid guessing private text.
- Memory PR detects persisted-point drift; it does not establish real-world truth or choose a winning fact.
- Secret detection is conservative pattern-based defense in depth, not a universal data-loss-prevention system.
- The HTML artifact is static and contains no approve/apply control.
- No native `hermes qdrant memory-pr` command was added. The supported live surface is the provider tool; the supported judge surface is `python3 -m qdrant_memory.memory_pr`.
- No dashboard, server, generic workflow engine, OKF subsystem, semantic proposal lookup, query mutation, or automatic apply path was added.
