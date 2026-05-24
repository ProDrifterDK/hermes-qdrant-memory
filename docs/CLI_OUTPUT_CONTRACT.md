# CLI Output Contract

This document describes the `v0.5.0` and later `hermes qdrant ...` command-line output contract.

## Modes

- Default mode is human-readable text.
  - Success output goes to stdout.
  - Output is deterministic, line-oriented, bounded, and has no ANSI control codes.
  - Human summaries avoid dumping raw JSON payloads by default.
- `--json` mode is machine-readable.
  - Successful commands emit exactly one JSON object.
  - Provider-backed commands preserve the provider/tool JSON payload unchanged when possible.
  - Local/recovery commands emit the same structured summary object used by the recovery helpers.
  - If a provider-backed command returns invalid JSON or non-object JSON, `--json` mode fails closed with one JSON error object on stderr.
- Errors are non-zero.
  - Default errors are human text on stderr.
  - `--json` errors are one JSON object on stderr.

## Redaction and payload safety

- `config show` redacts secret-like keys and credentialed URLs in both human and `--json` modes.
- `config show` and `watcher status` are local reads; they do not construct the provider or contact Qdrant/embedding services.
- `show`, `reports list`, `reports show`, and `proposals show` are read-only inspection commands.
  - `show` contacts Qdrant only to retrieve one explicit point ID from a configured collection.
  - `reports` and `proposals` inspect local consolidation artifacts only and do not contact Qdrant.
  - Point payloads are omitted unless `--include-payload` is set; vectors are omitted unless `--include-vector` is set.
- `export`, `backup`, and `restore` human output includes scope, IDs, paths, counts, and checksums only.
- Export/backup artifacts may contain raw memory text and vectors. Treat them as private recovery material.
- Human recovery output must not print raw backup/export/restore point payload text, vectors, or credential URLs.

## Command families

- `status`
  - Default: concise provider/service/collection status, for example `Qdrant memory provider: active`.
  - `--json`: raw `qdrant_memory_status` JSON object.
- `doctor`
  - Default: checklist summary, for example `Diagnostics: 10/10 checks passed` plus `[OK]` / `[FAIL]` lines.
  - `--json`: structured diagnostics with `ok`, `summary`, and `checks`.
  - Exit code remains `0` only when all critical checks pass.
- `config show`
  - Default: redacted key/value configuration summary.
  - `--json`: redacted JSON object.
- `search`, `learning search`
  - Default: bounded numbered result summary with IDs, scores, and snippets.
  - `--json`: raw provider search JSON.
- `show`
  - Default: exact point metadata summary for `--collection memory|learning`; payload and vector data are omitted.
  - `--include-payload`: includes redacted payload fields and a bounded redacted snippet.
  - `--include-vector`: requests vector data from Qdrant and includes vector data in `--json`; human output summarizes vector size/names.
  - Missing points are valid read results and return `found: false` with exit code `0`.
- `reports list`, `reports show`, `proposals show`
  - Default: local artifact/proposal summaries with exact IDs, counts, suggested actions, and affected IDs.
  - `--json`: redacted structured artifact/proposal JSON.
  - Invalid or path-like report/proposal IDs fail with exit code `2`.
- `store`, `learning store`
  - Default: saved ID summary.
  - `--json`: raw provider store JSON.
- `index`, `forget`, `learning approve`, `consolidate`, `watcher run`, `apply`
  - Default: safety-oriented summary showing dry-run/live state, IDs, proposal counts, and affected counts.
  - `--json`: raw provider JSON.
- `export`, `backup create`, `backup list`, `backup inspect`, `restore`
  - Default: safe artifact/plan summary.
  - `--json`: parseable recovery summary JSON.

## Mutation gates preserved

The output contract does not widen mutation authority:

- Maintenance mutations default to `--dry-run`.
- Live maintenance mutation requires `--no-dry-run --approve`.
- `forget` accepts explicit point IDs only.
- `consolidate` and `watcher run` are report-only.
- `apply` requires exact report ID, proposal ID, and expected action.
- `restore` remains dry-run by default and validates checksums/vector sizes before live upsert.
