# Release Notes: v0.2.1 Public Beta Hotfix

Hermes Qdrant Memory Provider v0.2.1 is a small hotfix release for the native `hermes qdrant ...` CLI.

## Why this release exists

The post-release consumer install/runtime smoke test against the published `v0.2.0` tag found that the CLI safety gate behaved safely but did not propagate the intended process exit status through Hermes v0.13.

Observed in v0.2.0:

```bash
hermes qdrant forget 00000000-0000-0000-0000-000000000000 --no-dry-run
```

Output:

```json
{"error": true, "message": "--approve is required when using --no-dry-run"}
```

The command correctly refused live mutation, but the process exited with status `0` because Hermes calls plugin command handlers without using their return value.

## Fixed

- `qdrant_command(args)` now raises `SystemExit(execute_command(args))` instead of returning the integer code.
- Hermes CLI invocations now receive non-zero process status for:
  - usage/safety errors such as missing `--approve` with `--no-dry-run`;
  - provider JSON error responses.
- Added regression test coverage for process-exit propagation.

## Unchanged safety behavior

- Mutating commands still default to dry-run.
- Live mutation still requires both `--no-dry-run` and `--approve`.
- `forget` still requires explicit point IDs.
- `apply` still requires exact report/proposal/action handles.
- Cron/reporting behavior remains report-only and review-gated.

## Upgrade

```bash
cd ~/.hermes/plugins/qdrant
git fetch --tags origin
git checkout v0.2.1
```

Start a fresh Hermes CLI process after upgrading. Restart the gateway only if it should load the updated plugin code for gateway sessions.

## Verification

Local verification used for this hotfix:

```bash
uv run --with pytest python -m pytest tests/test_cli.py -q
uv run --with pytest python -m pytest tests -q
python3 scripts/check_no_literal_fake_secrets.py
uv run python -m compileall -q qdrant_memory __init__.py cli.py scripts/check_no_literal_fake_secrets.py
```

Runtime smoke expectation after upgrading:

```bash
hermes qdrant forget 00000000-0000-0000-0000-000000000000 --no-dry-run
```

Expected:

- output contains `"--approve is required when using --no-dry-run"`;
- process exit status is non-zero;
- no Qdrant deletion occurs.
