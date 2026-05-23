#!/usr/bin/env python3
"""Reject scanner-sensitive fake secrets in docs and tests.

The guard intentionally focuses on documentation and executable fixtures. Runtime
implementation files may contain detection regexes, config key names, or auth
header construction that should be reviewed by normal code review instead.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_TARGETS = ("README.md", "docs", "tests", "plugin.yaml")
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}
TEXT_SUFFIXES = {".md", ".py", ".yaml", ".yml", ".txt"}

REDACTED_MARKERS = (
    "***",
    "<REDACTED>",
    "[REDACTED]",
    "REDACTED_",
    "redacted",
    "example",
    "placeholder",
)


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]


# Build suspicious prefixes from fragments so the guard source itself does not
# contain contiguous fake credential examples that trip external scanners.
OPENAI_PREFIX = "".join(("s", "k", "-"))
GITHUB_PREFIXES = tuple("".join(parts) for parts in (("g", "h", "p", "_"), ("g", "h", "o", "_"), ("g", "h", "u", "_"), ("g", "h", "s", "_"), ("g", "h", "r", "_")))
AWS_PREFIX = "".join(("A", "K", "I", "A"))
JWT_PREFIX = "".join(("e", "y", "J"))

RULES = (
    Rule("authorization-bearer", re.compile(r"Authorization:\s*Bearer\s+[\"']?([^\s`'\"]{8,})[\"']?", re.IGNORECASE)),
    Rule("bare-bearer", re.compile(r"\bBearer\s+[\"']?([^\s`'\"]{12,})[\"']?", re.IGNORECASE)),
    Rule("openai-style-key", re.compile(r"(?<![A-Za-z0-9_])" + re.escape(OPENAI_PREFIX) + r"[A-Za-z0-9_\-]{8,}")),
    Rule("github-token", re.compile(r"(?:" + "|".join(re.escape(p) for p in GITHUB_PREFIXES) + r")[A-Za-z0-9_]{8,}")),
    Rule("aws-access-key", re.compile(re.escape(AWS_PREFIX) + r"[0-9A-Z]{12,}")),
    Rule("jwt-like-token", re.compile(re.escape(JWT_PREFIX) + r"[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")),
    Rule("private-key-marker", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    Rule("url-basic-auth", re.compile(r"https?://[^\s/:@]+:[^\s/@]+@[^\s/]+")),
    Rule("inline-secret-assignment", re.compile(r"\b(?:password|passwd|api[_-]?key|secret|token)\s*[=:]\s*[\"']?([^\s`'\"]{8,})[\"']?", re.IGNORECASE)),
)


def _is_probably_redacted(match_text: str) -> bool:
    lowered = match_text.lower()
    if any(marker.lower() in lowered for marker in REDACTED_MARKERS):
        return True
    if "<" in match_text and ">" in match_text:
        return True
    return False


def iter_files(targets: Iterable[Path]) -> Iterable[Path]:
    for target in targets:
        if not target.exists():
            continue
        if target.is_file():
            if target.suffix in TEXT_SUFFIXES:
                yield target
            continue
        for path in target.rglob("*"):
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            if path.is_file() and path.suffix in TEXT_SUFFIXES:
                yield path


def scan_file(path: Path) -> list[tuple[int, str]]:
    findings: list[tuple[int, str]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError:
        return findings

    for line_no, line in enumerate(lines, start=1):
        for rule in RULES:
            for match in rule.pattern.finditer(line):
                snippet = match.group(0)
                if _is_probably_redacted(snippet):
                    continue
                findings.append((line_no, rule.name))
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check docs/tests for literal scanner-sensitive fake secrets.")
    parser.add_argument("paths", nargs="*", help="Files or directories to scan. Defaults to README.md docs tests plugin.yaml.")
    args = parser.parse_args(argv)

    root = Path.cwd()
    targets = [Path(p) for p in (args.paths or DEFAULT_TARGETS)]
    findings: list[str] = []

    for path in iter_files(targets):
        for line_no, rule_name in scan_file(path):
            display_path = path.relative_to(root) if path.is_absolute() and path.is_relative_to(root) else path
            findings.append(f"{display_path}:{line_no}: {rule_name}: <redacted>")

    if findings:
        print("Scanner-sensitive literal examples found:", file=sys.stderr)
        for finding in findings:
            print(f"  {finding}", file=sys.stderr)
        print("Use redacted placeholders or construct scanner-shaped strings at runtime in tests.", file=sys.stderr)
        return 1

    print("No scanner-sensitive literal examples found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
