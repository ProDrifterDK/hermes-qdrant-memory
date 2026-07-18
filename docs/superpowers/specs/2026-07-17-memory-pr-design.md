# Memory PR Design

## Product boundary

Memory PR turns one exact proposal from one persisted consolidation report into a portable, review-only evidence packet. It does not decide whether the proposal is correct and it never applies, approves, deletes, updates, or upserts memory. The only permitted writes are JSON and HTML files in a caller-selected output directory.

The extension is limited to this plugin repository. Native `hermes qdrant` CLI integration, dashboards, servers, workflow engines, and changes to Hermes core are explicitly out of scope.

## Architecture

The implementation adds a focused `qdrant_memory.memory_pr` module with four independent responsibilities:

1. Exact-ID validation and proposal selection.
2. Pure packet construction from a persisted report, one selected proposal, and currently reloaded exact points.
3. Self-contained HTML rendering from a completed packet.
4. Explicit private artifact persistence and a standard-library fixture CLI.

The existing provider exposes a read-only `qdrant_memory_memory_pr` tool. It loads the report with the established artifact loader, selects the exact proposal, reloads the proposal's exact affected point IDs from the proposal's configured collection, and passes those values to the pure builder. It does not use the consolidation apply path and does not call any Qdrant mutation method.

Newly persisted consolidation reports are enriched with stable review snapshot digests and an explicit projection name/version for affected points. Projection version 1 whitelists text, provenance, fact identity, validity, supersession, and canonical/stale/review state while excluding access/ranking bookkeeping such as `access_count`, `last_accessed`, and `decay_score`. Older or unversioned reports remain readable but label drift as `unknown` rather than comparing incompatible digests.

## Packet contract and determinism

The versioned JSON packet includes:

- schema name and version;
- deterministic `memory_pr_id` and `content_digest`;
- `generated_at` and `generation_mode`;
- exact report, proposal, collection, and affected point IDs;
- proposal type, expected and suggested actions, risk, and confidence;
- sanitized summary and proposed status changes;
- bounded current point evidence and provenance;
- current and report snapshot digests plus per-point drift labels;
- canonical, stale, review, and fact-status visibility;
- an explicit non-mutation boundary;
- a reviewer checklist and an exact dry-run next step.

The builder serializes stable sanitized review content with sorted keys and compact JSON separators, then hashes those bytes with SHA-256. `generated_at`, output paths, dictionary insertion order, and runtime-specific data are excluded. `memory_pr_id` is derived from the content digest. The synthetic fixture uses a fixed timestamp so its complete JSON and HTML outputs are byte-stable as well.

All snippets and free-text fields are bounded. Recursive redaction runs before digesting or rendering. Identity-bearing points retain status and provenance visibility but replace their memory snippet with the existing identity-redaction sentinel.

## Validation and drift

Report and proposal IDs accept only the established alphanumeric, hyphen, and underscore form. Empty, whitespace-padded, path-like, or traversal values fail closed.

The builder requires all of the following:

- the report's embedded ID matches the requested report ID;
- the selected proposal's embedded ID matches the requested proposal ID;
- affected IDs are non-empty, unique, and valid exact point IDs;
- the set of current point IDs equals the proposal's affected-ID set;
- the current points all belong to the proposal's single resolved collection.

Missing, additional, duplicate, or mismatched points cause an error; the builder never broadens retrieval scope.

For ordinary points, the same versioned stable sanitized snapshot projection is hashed during report persistence and Memory PR generation. Equal digests produce `unchanged`; unequal digests produce `changed`; absent, unversioned, or unsupported projections produce `unknown`. Identity- or secret-bearing point content is replaced before hashing and receives conservative `unknown` content drift so the artifact does not publish a guessable digest of private text. The packet also carries an overall drift status.

Persisted proposal evidence has a separate versioned schema. Each item must be an object with an exact ID in the proposal's affected set. Missing or unknown IDs fail closed. A single bounded recursive classifier is shared by current point payloads, persisted evidence, snapshot/digest construction, snippets, summaries, and status changes. It recognizes existing nested profile/fact markers plus normalized identity-sensitive keys for email, phone, address, usernames/handles, identity names, and national/passport/tax identifiers. Records with any sensitive nested part—or structures exceeding the classifier's review bounds—are replaced entirely with the identity-redaction sentinel before hashing or output, with drift forced to `unknown`.

## HTML artifact

The HTML renderer uses a restrained industrial review aesthetic: dark void surfaces, amber evidence markers, strong typographic hierarchy, and compact provenance rails. It has no JavaScript, CDN, analytics, fonts, images, or network calls.

The document uses semantic landmarks, a skip link, ordered heading levels, lists, tables or definition lists where appropriate, visible keyboard focus, high-contrast text, non-color status labels, responsive grids, forced-color compatibility, reduced-motion compatibility, and print styles. Every untrusted value is escaped. A restrictive Content Security Policy permits only the embedded style block and denies every network-capable resource type.

## Artifact and provider behavior

Without an output directory, the provider tool resolves and reads the report without creating or chmodding any directory, requires the proposal collection to exactly match the configured memory or learning collection, returns the packet as a read-only preview, and reports that no artifact was persisted. With an explicit output directory, it writes deterministic filenames based on `memory_pr_id`. New directories use mode `0700` and files mode `0600` where the platform supports them. A pre-existing directory is accepted only if it is already owned by the current user with mode `0700`; it is never chmodded. Existing files are not replaced unless the caller explicitly opts into overwrite.

The offline CLI command is:

```bash
python3 -m qdrant_memory.memory_pr fixture --output-dir /tmp/memory-pr-demo
```

It loads only a license-safe repository fixture and requires no Hermes installation, Qdrant, embedding service, network, Obsidian vault, or third-party package. A `verify-fixture` subcommand generates the fixture twice in temporary directories and verifies matching identities, digests, and artifact bytes.

## Error handling

Pure functions raise a dedicated validation error with safe, bounded messages. The provider converts these to the plugin's JSON error envelope without echoing raw memory. Files are prepared before replacement, and output paths never become part of the packet or HTML content. Partial write failures are surfaced and do not trigger any Qdrant operation.

## Testing

Focused tests cover deterministic identity and digest, access-bookkeeping stability, exact integer snapshot versions, exact/malformed IDs, changed and unchanged drift, legacy unknown drift, affected-ID mismatch, nested secret redaction, bearer-like values, bounded recursive identity suppression across current payloads/evidence/summaries/status changes, missing/unknown evidence IDs, live-provider JSON/HTML/pre-hash projection privacy, configured-collection rejection, HTML injection, absence of external resources, private permissions, shared-directory rejection without chmod, non-creating missing-report reads, fixture generation and verification, and zero mutation calls on provider and fixture paths.

Existing consolidation, reconsolidation, proposal, provider tool, and CLI tests provide regression evidence that report/apply and guarded-auto contracts remain intact. Verification also includes the full repository suite under a disposable `HERMES_HOME`, compilation, fake-secret scanning, whitespace checks, fixture generation from a clean output directory, and explicit inspection for private paths, secrets, and external assets.

## Explicit cuts and limitations

- No native Hermes CLI command; the provider tool and module CLI are the supported entrypoints.
- No live mutation or apply shortcut in the packet or HTML.
- No semantic search or query selection; only exact persisted IDs are accepted.
- Legacy reports without review snapshot digests show `unknown` drift.
- HTML is a static review artifact and has no interactive approval controls.
