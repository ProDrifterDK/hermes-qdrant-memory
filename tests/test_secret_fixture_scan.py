import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "check_no_literal_fake_secrets.py"


def _run_checker(*paths: Path):
    args = [sys.executable, str(SCRIPT)]
    args.extend(str(path) for path in paths)
    return subprocess.run(args, cwd=ROOT, text=True, capture_output=True)


def test_repo_docs_and_tests_have_no_literal_fake_secrets():
    result = _run_checker()

    assert result.returncode == 0, result.stderr
    assert "No scanner-sensitive literal examples found" in result.stdout


def test_secret_scan_detects_constructed_bearer_fixture(tmp_path):
    token = "".join(["alpha", "beta", "gamma", "delta"])
    fixture = tmp_path / "fixture.md"
    fixture.write_text(" ".join(["Authorization:", "Bearer", token]) + "\n", encoding="utf-8")

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "authorization-bearer" in result.stderr
    assert "fixture.md" in result.stderr
    assert token not in result.stderr
    assert "<redacted>" in result.stderr


def test_secret_scan_detects_quoted_assignment_fixture(tmp_path):
    value = "".join(["alpha", "beta", "gamma", "delta"])
    fixture = tmp_path / "fixture.md"
    fixture.write_text(f'api_key="{value}"\n', encoding="utf-8")

    result = _run_checker(tmp_path)

    assert result.returncode == 1
    assert "inline-secret-assignment" in result.stderr
    assert value not in result.stderr
    assert "<redacted>" in result.stderr


def test_secret_scan_allows_redacted_placeholders(tmp_path):
    fixture = tmp_path / "fixture.md"
    redacted_bearer = " ".join(["Authorization:", "Bearer", "***"])
    fixture.write_text(
        f"{redacted_bearer}\n"
        "qdrant_api_key=<REDACTED_QDRANT_API_KEY>\n"
        "token=<REDACTED>\n",
        encoding="utf-8",
    )

    result = _run_checker(tmp_path)

    assert result.returncode == 0, result.stderr
