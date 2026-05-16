# CLI Feasibility Spike

This spike verifies whether `hermes-qdrant-memory` can expose a native Hermes CLI surface such as:

```bash
hermes qdrant status
hermes qdrant search "project memory"
hermes qdrant index ~/Notes --dry-run
```

The result: **native Hermes CLI integration is feasible today**, with an important constraint: for this plugin, the best path is the Hermes memory-provider `cli.py` convention, not `ctx.register_cli_command()` inside `register(ctx)`.

Roadmap: [PLUGIN_ROADMAP.md](PLUGIN_ROADMAP.md).
Safety contract: [SAFETY.md](SAFETY.md).
Operations runbook: [OPERATIONS.md](OPERATIONS.md).

---

## 1. Summary decision

Recommended path for CLI MVP:

- Add a top-level plugin `cli.py`.
- Implement `register_cli(subparser)`.
- Implement optional `qdrant_command(args)` dispatcher.
- Keep parser setup lightweight: no Qdrant client creation and no embedding client creation during argparse setup.
- Implement command behavior through shared code, ideally `qdrant_memory/cli_core.py`, so a future standalone wrapper can reuse it.

Expected native command namespace:

```bash
hermes qdrant <subcommand>
```

This should work when:

- the plugin is installed under the Hermes plugin path as the `qdrant` provider; and
- `memory.provider` is set to `qdrant`.

Fallback path:

- If future Hermes versions change plugin CLI internals, ship a standalone wrapper such as `hermes-qdrant-memory` that reuses the same `cli_core` command implementation.

---

## 2. Hermes supports two CLI plugin paths

### 2.1 General plugin CLI commands

Hermes supports general plugin CLI commands through plugin context registration:

```python
ctx.register_cli_command(...)
```

Evidence in Hermes core:

- `hermes_cli/plugins.py` defines `PluginContext.register_cli_command(...)`.
- `PluginManager` stores registered commands in `_cli_commands`.
- `hermes_cli/main.py` dynamically adds registered plugin commands to argparse.
- Bundled plugins such as `plugins/teams_pipeline` use this path.

This path is real and tested in Hermes core, but it is not the best fit for this memory-provider plugin.

### 2.2 Memory-provider CLI convention

Hermes also supports memory-provider-specific CLI discovery.

For the active memory provider, Hermes checks the provider directory for:

```text
cli.py
```

The expected shape is:

```python
def register_cli(subparser):
    ...

def qdrant_command(args):
    ...
```

Hermes registers the command using the active provider name. For this plugin, that means:

```bash
hermes qdrant ...
```

Evidence in Hermes core:

- `plugins/memory/__init__.py` has `discover_plugin_cli_commands()`.
- It reads the active provider from config.
- It resolves the active provider directory.
- It looks for `cli.py`.
- It imports `register_cli(subparser)`.
- It optionally wires `<provider>_command(args)` as the command handler.
- Hermes core tests cover this behavior in `tests/hermes_cli/test_plugin_cli_registration.py`.

Important detail:

- The memory-provider loader's collector intentionally treats `register_cli_command` as a no-op.
- Therefore a memory provider should not rely on calling `ctx.register_cli_command(...)` from `register(ctx)` for its provider CLI.
- Use the `cli.py` convention instead.

---

## 3. Current plugin shape

Current `hermes-qdrant-memory` facts:

- `plugin.yaml` declares:
  - `name: qdrant`
  - `kind: exclusive`
  - `category: memory`
- `__init__.py` defines `QdrantMemoryProvider`.
- `register(ctx)` calls:

```python
ctx.register_memory_provider(QdrantMemoryProvider())
```

- The stable command surface today is Hermes tool calls, not CLI commands.
- Existing tools include:
  - `qdrant_memory_status`
  - `qdrant_memory_store`
  - `qdrant_memory_search`
  - `qdrant_memory_index`
  - `qdrant_memory_forget`
  - `qdrant_memory_consolidate`
  - `qdrant_memory_consolidation_apply`
  - `qdrant_learning_store`
  - `qdrant_learning_search`
  - `qdrant_learning_preview`
  - `qdrant_learning_approve`

The CLI should wrap or reuse this behavior without weakening safety gates.

---

## 4. Stability assessment

Native Hermes CLI integration is feasible, but the public contract should be described honestly.

Stable enough for MVP:

- Hermes core implements memory-provider CLI discovery.
- Hermes core has tests for memory-provider `cli.py` discovery.
- The command namespace naturally matches the provider name: `qdrant`.
- No packaging dependency is required for a top-level `cli.py` in the plugin directory.

Caveats:

- Memory-provider CLI commands appear only for the active memory provider.
- The command may not appear in top-level help if plugin discovery is skipped for startup performance.
- `hermes qdrant --help` should trigger discovery when `qdrant` is an unknown positional command and the active provider is configured.
- The memory-provider CLI convention is implemented in Hermes core, but it is not yet documented as a stable public extension API in this plugin's docs.
- Future Hermes internals could change; keep command implementation reusable outside Hermes argparse.

Conclusion:

- Proceed with native `cli.py` MVP.
- Keep standalone wrapper as fallback, not primary path.

---

## 5. Recommended CLI architecture

### Files

Create:

```text
cli.py
qdrant_memory/cli_core.py
tests/test_cli.py
```

Optional later:

```text
scripts/hermes-qdrant-memory
```

### Responsibilities

`cli.py`:

- Hermes integration shim.
- Defines `register_cli(subparser)`.
- Defines `qdrant_command(args)`.
- Does not instantiate Qdrant or embedding clients during parser setup.
- Imports heavy provider code only when executing a command, if possible.

`qdrant_memory/cli_core.py`:

- Shared command implementation.
- Converts parsed CLI arguments into provider/tool calls or direct service calls.
- Owns formatting for JSON/plain output.
- Preserves dry-run and approval gates.
- Can be reused by a standalone wrapper later.

`tests/test_cli.py`:

- Parser construction tests.
- Default dry-run tests.
- Live mutation gate tests.
- Explicit-ID requirement tests.
- Quality-warning refusal tests.
- No-client-construction-during-parser-setup test.

---

## 6. MVP command set

Start with read-only and dry-run-safe commands.

Recommended M13 MVP:

```bash
hermes qdrant status
hermes qdrant doctor
hermes qdrant search <query> [--top-k N] [--source-type TYPE] [--json]
hermes qdrant index <path...> [--dry-run] [--no-dry-run --approve] [--max-files N] [--force]
hermes qdrant forget <point-id...> [--dry-run] [--no-dry-run --approve]
hermes qdrant learning search <query> [--top-k N] [--json]
hermes qdrant learning preview [--json]
hermes qdrant consolidate [--scope memory|learning|both] [--persist] [--include-reconsolidation] [--json]
hermes qdrant apply --report-id ID --proposal-id ID --action ACTION [--dry-run] [--no-dry-run --approve]
```

Possible later commands:

```bash
hermes qdrant store <text>
hermes qdrant learning store ...
hermes qdrant learning approve ...
hermes qdrant watcher run
hermes qdrant config show
```

Do not start with broad live mutation commands.

---

## 7. Safety rules for CLI MVP

The CLI must preserve the same safety contract as the tools.

Rules:

- Mutating commands default to dry-run.
- Live mutation requires both:
  - `--no-dry-run`; and
  - `--approve`.
- `forget` must require explicit point IDs.
- `apply` must require exact `--report-id`, exact `--proposal-id`, and expected `--action`.
- `quality_warning` proposals must remain manual-review only and must not live-apply.
- `draft_review` may create local markdown artifacts but must not mutate Qdrant facts.
- `consolidate` remains report-only.
- `persist` means local redacted report artifacts, not Qdrant mutation.
- `index` live mode requires explicit path arguments and reviewed dry-run output.
- CLI output must not print raw scanner-sensitive values.

Reference policies:

- [SAFETY.md](SAFETY.md)
- [OPERATIONS.md](OPERATIONS.md)
- [LCM_BOUNDARY.md](LCM_BOUNDARY.md)

---

## 8. Activation constraints

For native Hermes CLI usage, document these constraints:

1. Plugin must be installed in the Hermes plugin path as `qdrant`.
2. Hermes config must select the provider:

```bash
hermes config set memory.provider qdrant
```

3. Start a fresh Hermes CLI process after plugin/config changes.
4. Run:

```bash
hermes qdrant --help
```

If the command is not discovered:

- verify plugin path;
- verify `memory.provider`;
- verify `plugin.yaml` name/category;
- verify `cli.py` exists at plugin root;
- run `hermes plugins list` or `hermes memory status` if available;
- fall back to the standalone wrapper if provided.

---

## 9. Testing plan

Before shipping CLI MVP:

```bash
python -m pytest tests/test_cli.py -q
python -m pytest tests -q
python scripts/check_no_literal_fake_secrets.py
python -m compileall -q qdrant_memory __init__.py cli.py
```

Recommended test cases:

- `register_cli(subparser)` adds expected subcommands.
- Parser setup does not create Qdrant or embedding clients.
- `index` defaults to dry-run.
- `forget` rejects no point IDs.
- `forget --no-dry-run` rejects without `--approve`.
- `apply` rejects missing report ID.
- `apply` rejects missing proposal ID.
- `apply --no-dry-run` rejects without `--approve`.
- `quality_warning` live apply remains impossible.
- JSON output is parseable.
- Scanner guard passes over CLI examples.

---

## 10. Recommendation

Proceed with native Hermes CLI MVP using the memory-provider `cli.py` convention.

Do not implement a standalone wrapper first. Keep the command implementation reusable so a standalone wrapper can be added later if Hermes internals change.

Do not mark the full CLI surface stable until:

- `cli.py` exists;
- `tests/test_cli.py` covers parser and safety gates;
- `hermes qdrant --help` works in a real installed plugin path;
- at least `status`, `search`, `index --dry-run`, `consolidate --dry-run`, and `apply --dry-run` work against the provider without bypassing tool safety rules.
