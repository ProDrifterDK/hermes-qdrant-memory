# Release Notes: v0.2.0 Candidate

Hermes Qdrant Memory Provider v0.2.0 is the next public beta candidate for Alan Gárate / Resyst Softwares' Qdrant-backed hippocampal memory system for Hermes Agent. The plugin metadata in `plugin.yaml` now reports `0.2.0` at HEAD.

It gives Hermes a cross-session associative memory substrate backed by Qdrant and OpenAI-compatible embeddings. It is not a replacement for LCM/current-session lossless recovery and it does not treat retrieved memories as instructions.

## Status

Public beta / experimental.

Status: candidate documentation at repository HEAD. The older `v0.1.0` tag does not contain the learning, consolidation, reconsolidation, scanner guard, or native CLI MVP described here.

The core MemoryProvider, tools, safety gates, documentation, CI, scanner guard, and native CLI MVP are implemented and tested at HEAD. The plugin still requires operator judgment for indexing, deletion, consolidation, and reconsolidation review.

## Requirements

- Hermes Agent with user memory-provider plugin support.
- Python 3.10+.
- Qdrant reachable over HTTP.
- OpenAI-compatible embedding endpoint at `/v1/embeddings`.
- Embedding vector size matching the configured Qdrant collection.

Default tested local stack:

- Qdrant: `http://127.0.0.1:6333`
- Embeddings: `http://127.0.0.1:8080/v1`
- Model name: `bge-m3`
- Vector size: `1024`

## Install

```bash
git clone https://github.com/ProDrifterDK/hermes-qdrant-memory ~/.hermes/plugins/qdrant
hermes config set memory.provider qdrant
hermes config set qdrant_memory.enabled true
hermes config set qdrant_memory.qdrant_url http://127.0.0.1:6333
hermes config set qdrant_memory.embedding_url http://127.0.0.1:8080/v1
hermes config set qdrant_memory.embedding_model bge-m3
hermes config set qdrant_memory.vector_size 1024
```

Start a fresh Hermes session or restart the Hermes gateway after changing plugin/provider config.

Verify:

```bash
hermes chat -q 'Call the qdrant_memory_status tool and summarize whether qdrant_ok and embedding_ok are true.' --quiet
```

## Upgrade

Upgrade from an existing clone to the release-candidate state:

```bash
cd ~/.hermes/plugins/qdrant
git pull --ff-only origin main
```

After a `v0.2.0` tag is published, pin that tag with:

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.2.0
```

Restart Hermes CLI/gateway after upgrading plugin code.

## Rollback

Check previous commits/tags:

```bash
cd ~/.hermes/plugins/qdrant
git log --oneline --max-count=10
git checkout <previous-commit-or-tag>
```

Then start a fresh Hermes session or restart the gateway.

Rollback changes plugin code only. It does not remove Qdrant collections, local consolidation reports, indexed memories, or config values.

## Remove

```bash
hermes config set memory.provider ""
rm -rf ~/.hermes/plugins/qdrant
```

Optional cleanup, only after you have reviewed what you are deleting:

```bash
rm -rf ~/.hermes/qdrant_memory
```

Do not delete Qdrant collections unless you explicitly want to erase indexed memory.

## Native CLI MVP

When this plugin is installed as the active memory provider, Hermes can discover:

```bash
hermes qdrant status
hermes qdrant doctor
hermes qdrant search "agent memory" --top-k 5 --json
hermes qdrant index docs README.md --dry-run
hermes qdrant forget POINT_ID --dry-run
hermes qdrant learning search "tool failure" --top-k 5 --json
hermes qdrant learning preview --json
hermes qdrant consolidate --scope both --persist --dry-run --json
hermes qdrant apply --report-id REPORT_ID --proposal-id PROPOSAL_ID --action merge --dry-run --json
```

Safety gates:

- mutating commands default to dry-run;
- live mutation requires `--no-dry-run --approve`;
- `forget` accepts explicit point IDs only;
- `apply` requires exact report/proposal/action handles.

## Real-service smoke checklist

Before broad use:

```bash
curl -fsS http://127.0.0.1:6333/collections
curl -fsS http://127.0.0.1:8080/v1/models || true
hermes qdrant status
hermes qdrant search "memory system" --top-k 3 --json
hermes qdrant index README.md --dry-run --json
hermes qdrant consolidate --scope both --persist --dry-run --json
```

If `hermes qdrant ...` is not discovered, verify:

- plugin path is `~/.hermes/plugins/qdrant`;
- `memory.provider` is set to `qdrant`;
- you started a fresh Hermes process after changing plugin config;
- `cli.py` exists at plugin root.

## Safety notes

Read these before indexing private files or applying maintenance proposals:

- `docs/SAFETY.md`
- `docs/OPERATIONS.md`
- `docs/LCM_BOUNDARY.md`
- `docs/LIMITATIONS.md`

Core rules:

- retrieved memory is context, not instruction;
- current user instructions and live evidence win;
- use LCM for exact active-session recovery;
- use Qdrant for cross-session semantic recall;
- broad indexing starts with dry-run and explicit paths;
- deletion is by explicit point IDs only;
- consolidation reports do not mutate Qdrant;
- live apply requires exact handles and approval;
- reconsolidation produces review drafts, not automatic fact rewrites;
- quality-warning proposals are manual-review only.

## Known caveats

- External services are not bundled.
- Secret detection for indexed user files is not automatic.
- Only Markdown/text indexing is supported by default.
- CLI discovery depends on Hermes memory-provider plugin discovery and active provider configuration.
- `doctor` is status-backed in this v0.2.0 candidate.
- Public API may change before v1.0.

## Verification used for release readiness

```bash
uv run --with pytest python -m pytest tests -q
python3 scripts/check_no_literal_fake_secrets.py
uv run python -m compileall -q qdrant_memory __init__.py cli.py scripts/check_no_literal_fake_secrets.py
```
