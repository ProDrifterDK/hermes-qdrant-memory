from __future__ import annotations

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from qdrant_memory.memory_pr import (
    IDENTITY_REDACTED_SNIPPET,
    MemoryPRValidationError,
    _is_identity_sensitive_key,
    _snapshot_projection,
    build_memory_pr,
    generate_fixture_artifacts,
    render_memory_pr_html,
    stable_point_snapshot_digest,
    validate_exact_id,
    write_memory_pr_artifacts,
)
from qdrant_memory.consolidation import persist_consolidation_report
from qdrant_memory.config import load_config
from qdrant_memory.tools import MEMORY_PR_SCHEMA, TOOL_SCHEMAS


def _point(point_id: str, text: str, **payload):
    return {
        "id": point_id,
        "payload": {
            "text": text,
            "source_type": "manual",
            "source_uri": f"memory://fixture/{point_id}",
            "canonical": False,
            "stale": False,
            "requires_review": True,
            "fact_status": "active",
            **payload,
        },
    }


def _report_and_points():
    points = [
        _point(
            "runtime-old",
            "The Atlas worker uses runtime v1.",
            fact_key="atlas.runtime.version",
            observed_at="2026-01-10T00:00:00Z",
        ),
        _point(
            "runtime-new",
            "The Atlas worker uses runtime v2.",
            canonical=True,
            fact_key="atlas.runtime.version",
            observed_at="2026-03-15T00:00:00Z",
        ),
    ]
    proposal = {
        "proposal_id": "fact-supersession-1234abcd",
        "proposal_type": "fact_supersession_candidate",
        "collection_name": "memory",
        "affected_ids": ["runtime-old", "runtime-new"],
        "suggested_action": "draft_review_only",
        "risk": "medium",
        "confidence": 0.93,
        "candidate_statement": "Runtime v2 supersedes runtime v1 for Atlas.",
        "proposed_status_changes": [
            {
                "id": "runtime-old",
                "from": "active",
                "to": "superseded",
                "reason": "newer observation",
                "superseded_by": ["runtime-new"],
            },
            {
                "id": "runtime-new",
                "from": "active",
                "to": "active",
                "reason": "newer observation remains current",
            },
        ],
        "review_point_snapshots": [
            {
                "id": point["id"],
                "projection": {"name": "memory-pr-review-point", "version": 1},
                "snapshot_digest": stable_point_snapshot_digest(point),
            }
            for point in points
        ],
    }
    report = {
        "schema_version": 1,
        "report_id": "report-1234abcd",
        "scope": "memory",
        "proposals": [proposal],
    }
    return report, proposal, points


def _build(*, report=None, points=None, generated_at="2026-07-17T12:00:00Z"):
    base_report, proposal, base_points = _report_and_points()
    return build_memory_pr(
        report=report or base_report,
        proposal_id=proposal["proposal_id"],
        current_points=points or base_points,
        report_id=base_report["report_id"],
        generated_at=generated_at,
        generation_mode="fixture",
    )


def test_memory_pr_identity_is_deterministic_and_excludes_timestamp_and_order():
    report, proposal, points = _report_and_points()
    reordered_report = {
        "proposals": [{key: proposal[key] for key in reversed(list(proposal))}],
        "scope": "memory",
        "report_id": report["report_id"],
        "schema_version": 1,
    }

    first = _build(report=report, points=points, generated_at="2026-07-17T12:00:00Z")
    second = _build(report=reordered_report, points=list(reversed(points)), generated_at="2026-07-18T12:00:00Z")

    assert first["memory_pr_id"] == second["memory_pr_id"]
    assert first["content_digest"] == second["content_digest"]
    assert first["generated_at"] != second["generated_at"]
    assert first["memory_pr_id"].startswith("mpr-")
    assert len(first["content_digest"]) == 64


def test_memory_pr_reports_no_drift_and_detects_changed_current_content():
    report, _, points = _report_and_points()
    unchanged = _build(report=report, points=points)
    changed_points = json.loads(json.dumps(points))
    changed_points[0]["payload"]["text"] = "The Atlas worker now uses a patched runtime v1."
    changed = _build(report=report, points=changed_points)

    assert unchanged["drift_status"] == "unchanged"
    assert {item["drift_status"] for item in unchanged["current_evidence"]} == {"unchanged"}
    assert changed["drift_status"] == "changed"
    assert {item["id"] for item in changed["current_evidence"] if item["drift_status"] == "changed"} == {"runtime-old"}


def test_review_snapshot_projection_is_versioned_and_ignores_access_bookkeeping():
    report, proposal, points = _report_and_points()
    before = _build(report=report, points=points)
    accessed_points = json.loads(json.dumps(points))
    for index, point in enumerate(accessed_points, start=1):
        point["payload"].update(
            {
                "access_count": 40 + index,
                "last_accessed": f"2026-07-18T00:00:0{index}Z",
                "decay_score": 0.01 * index,
                "ranking_debug": {"final_score": 0.99 - (index * 0.01)},
            }
        )

    after = _build(report=report, points=accessed_points)

    assert proposal["review_point_snapshots"][0]["projection"] == {
        "name": "memory-pr-review-point",
        "version": 1,
    }
    assert after["drift_status"] == "unchanged"
    assert {item["drift_status"] for item in after["current_evidence"]} == {"unchanged"}
    assert [item["snapshot_digest"] for item in before["current_evidence"]] == [
        item["snapshot_digest"] for item in after["current_evidence"]
    ]
    assert before["memory_pr_id"] == after["memory_pr_id"]
    assert before["content_digest"] == after["content_digest"]


@pytest.mark.parametrize(
    ("payload_change", "value"),
    [
        ("source_uri", "memory://fixture/revised-source"),
        ("stale", True),
        ("fact_status", "superseded"),
    ],
)
def test_review_snapshot_projection_preserves_provenance_and_review_state(payload_change, value):
    report, _, points = _report_and_points()
    changed_points = json.loads(json.dumps(points))
    changed_points[0]["payload"][payload_change] = value

    packet = _build(report=report, points=changed_points)

    changed = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")
    assert changed["drift_status"] == "changed"


def test_legacy_report_without_snapshots_labels_drift_unknown():
    report, proposal, points = _report_and_points()
    proposal.pop("review_point_snapshots")

    packet = _build(report=report, points=points)

    assert packet["drift_status"] == "unknown"
    assert {item["drift_status"] for item in packet["current_evidence"]} == {"unknown"}


def test_unversioned_legacy_snapshots_are_not_compared_as_current_projection():
    report, _, points = _report_and_points()
    for snapshot in report["proposals"][0]["review_point_snapshots"]:
        snapshot.pop("projection")

    packet = _build(report=report, points=points)

    assert packet["drift_status"] == "unknown"
    assert {item["report_snapshot_projection"] for item in packet["current_evidence"]} == {None}


def test_boolean_snapshot_projection_version_is_rejected_not_compared_as_v1():
    report, _, points = _report_and_points()
    for snapshot in report["proposals"][0]["review_point_snapshots"]:
        snapshot["projection"]["version"] = True

    with pytest.raises(MemoryPRValidationError, match="projection"):
        _build(report=report, points=points)


@pytest.mark.parametrize("value", ["", " report-1", "report-1 ", "../report", "a/b", "a.b", "a\\b"])
def test_exact_id_validation_rejects_malformed_or_path_like_values(value):
    with pytest.raises(MemoryPRValidationError):
        validate_exact_id(value, "report_id")


def test_exact_report_and_proposal_ids_must_match_selected_artifact():
    report, proposal, points = _report_and_points()

    with pytest.raises(MemoryPRValidationError, match="report_id"):
        build_memory_pr(
            report=report, report_id="report-other", proposal_id=proposal["proposal_id"], current_points=points
        )
    with pytest.raises(MemoryPRValidationError, match="proposal_id"):
        build_memory_pr(
            report=report, report_id=report["report_id"], proposal_id="proposal-other", current_points=points
        )


def test_current_affected_id_mismatch_fails_closed():
    report, proposal, points = _report_and_points()

    with pytest.raises(MemoryPRValidationError, match="affected point IDs"):
        build_memory_pr(
            report=report, report_id=report["report_id"], proposal_id=proposal["proposal_id"], current_points=points[:1]
        )
    with pytest.raises(MemoryPRValidationError, match="affected point IDs"):
        build_memory_pr(
            report=report,
            report_id=report["report_id"],
            proposal_id=proposal["proposal_id"],
            current_points=[*points, _point("unrequested", "Unrelated point")],
        )


def test_nested_secrets_and_bearer_like_values_are_redacted_everywhere():
    report, proposal, points = _report_and_points()
    secret = "".join(["unit", "Test", "Credential", "42", "Value"])
    bearer = "".join(["Bearer ", "fixture", "Token", "1234567890"])
    points[0]["payload"]["nested"] = {"api_key": secret, bearer: "nested-key", "items": [{"note": bearer}]}
    proposal["candidate_statement"] = {
        "safe": "review",
        "nested": {"password": secret, bearer: "safe nested key value"},
    }
    report["proposals"] = [proposal]

    packet = _build(report=report, points=points)
    rendered = json.dumps(packet, sort_keys=True)
    html = render_memory_pr_html(packet)
    secret_evidence = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")

    assert secret not in rendered
    assert secret not in html
    assert bearer not in rendered
    assert bearer not in html
    assert "redacted" in rendered.lower()
    assert secret_evidence["identity_bearing"] is True
    assert secret_evidence["secret_bearing"] is True
    assert secret_evidence["drift_status"] == "unknown"


def test_identity_bearing_point_never_exposes_its_snippet():
    report, proposal, points = _report_and_points()
    identity_text = "Casey Example prefers the private contact channel."
    points[0]["payload"].update(
        {"text": identity_text, "source_type": "user_profile", "fact_key": "user.preferred_name"}
    )
    proposal["review_point_snapshots"][0]["snapshot_digest"] = stable_point_snapshot_digest(points[0])
    proposal["candidate_statement"] = identity_text
    proposal["source_snippets"] = [{"id": "runtime-old", "snippet": identity_text, "source_type": "user_profile"}]
    proposal["proposed_status_changes"][0]["reason"] = identity_text

    packet = _build(report=report, points=points)
    evidence = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")
    html = render_memory_pr_html(packet)

    assert evidence["snippet"] == IDENTITY_REDACTED_SNIPPET
    assert identity_text not in json.dumps(packet)
    assert identity_text not in html


def test_identity_persisted_evidence_is_recursively_suppressed():
    report, proposal, points = _report_and_points()
    identity_text = "Casey Example uses the private recovery contact."
    points[0]["payload"].update({"source_type": "user_profile", "fact_key": "user.contact"})
    proposal["source_snippets"] = [
        {
            "id": "runtime-old",
            "source_type": "manual",
            "metadata": {"contact": {"note": identity_text, "labels": [identity_text]}},
        }
    ]

    packet = _build(report=report, points=points)
    rendered = json.dumps(packet, sort_keys=True)

    assert identity_text not in rendered
    assert packet["persisted_evidence"] == [
        {
            "id": "runtime-old",
            "identity_bearing": True,
            "evidence": IDENTITY_REDACTED_SNIPPET,
        }
    ]


@pytest.mark.parametrize(
    "source_snippets",
    [
        [{"metadata": {"note": "Casey Example private contact"}}],
        [{"id": "unknown-point", "snippet": "Casey Example private contact"}],
    ],
)
def test_identity_persisted_evidence_without_exact_affected_id_fails_closed(source_snippets):
    report, proposal, points = _report_and_points()
    points[0]["payload"].update({"source_type": "user_profile", "fact_key": "user.contact"})
    proposal["source_snippets"] = source_snippets

    with pytest.raises(MemoryPRValidationError, match="persisted evidence"):
        _build(report=report, points=points)


def test_persisted_identity_metadata_is_suppressed_when_current_point_is_not_identity_bearing():
    report, proposal, points = _report_and_points()
    identity_text = "Casey Example private recovery contact."
    proposal["source_snippets"] = [
        {
            "id": "runtime-old",
            "metadata": {
                "source_type": "user_profile",
                "contact_note": identity_text,
            },
        }
    ]

    packet = _build(report=report, points=points)

    assert identity_text not in json.dumps(packet, sort_keys=True)
    assert packet["persisted_evidence"][0]["identity_bearing"] is True


@pytest.mark.parametrize(
    "identity_metadata",
    [
        {"email": "casey@example.invalid"},
        {"phone": "+1 555 010 0199"},
        {"address": "19 Synthetic Avenue"},
        {"username": "casey-fixture"},
        {"handle": "casey-fixture"},
        {"legalName": "Casey Fixture"},
        {"preferred_name": "Casey Fixture"},
        {"full-name": "Casey Fixture"},
        {"nationalId": "fixture-national-id"},
        {"passport_number": "fixture-passport-id"},
        {"tax-id": "fixture-tax-id"},
        {"contactEmail": "casey@example.invalid"},
        {"mobile_phone": "+1 555 010 0199"},
        {"homeAddress": "19 Synthetic Avenue"},
        {"passportId": "fixture-passport-id"},
    ],
)
def test_nested_identity_sensitive_evidence_keys_suppress_the_entire_record(identity_metadata):
    report, proposal, points = _report_and_points()
    proposal["source_snippets"] = [
        {
            "id": "runtime-old",
            "metadata": {"records": [identity_metadata]},
        }
    ]

    packet = _build(report=report, points=points)

    assert packet["persisted_evidence"] == [
        {
            "id": "runtime-old",
            "identity_bearing": True,
            "evidence": IDENTITY_REDACTED_SNIPPET,
        }
    ]


def test_nested_identity_in_summary_and_status_change_is_suppressed_with_safe_current_points():
    report, proposal, points = _report_and_points()
    summary_identity = "casey@example.invalid"
    status_identity = "+1 555 010 0199"
    proposal["candidate_statement"] = {"metadata": {"contactEmail": summary_identity}}
    proposal["proposed_status_changes"][0].update(
        {
            "reason": "review requested",
            "metadata": {"contacts": [{"mobilePhone": status_identity}]},
        }
    )

    packet = _build(report=report, points=points)
    rendered = json.dumps(packet, sort_keys=True)

    assert "Identity-bearing proposal summary suppressed" in packet["proposal_summary"]
    assert packet["proposed_status_changes"][0]["reason"] == IDENTITY_REDACTED_SNIPPET
    assert summary_identity not in rendered
    assert status_identity not in rendered


def test_identity_classifier_bounds_fail_closed_before_persisted_evidence_output():
    report, proposal, points = _report_and_points()
    over_limit = [{"note": f"bounded-{index}"} for index in range(101)]
    proposal["source_snippets"] = [
        {
            "id": "runtime-old",
            "metadata": {"records": over_limit},
        }
    ]

    packet = _build(report=report, points=points)

    assert packet["persisted_evidence"] == [
        {
            "id": "runtime-old",
            "identity_bearing": True,
            "evidence": IDENTITY_REDACTED_SNIPPET,
        }
    ]


@pytest.mark.parametrize(
    "identity_key",
    [
        "socialSecurityNumber",
        "social_security_id",
        "dateOfBirth",
        "birthDate",
        "birthdate",
        "dob",
        "driverLicenseNumber",
        "passportNo",
        "taxpayerId",
    ],
)
def test_normalized_identity_alias_vocabulary_covers_reported_legacy_families(identity_key):
    assert _is_identity_sensitive_key(identity_key) is True


def test_html_escapes_hostile_values_and_has_no_external_resources():
    report, proposal, points = _report_and_points()
    hostile = "<script>alert('memory-pr')</script><img src=x onerror=alert(2)>"
    points[0]["payload"]["text"] = hostile
    proposal["review_point_snapshots"][0]["snapshot_digest"] = stable_point_snapshot_digest(points[0])

    html = render_memory_pr_html(_build(report=report, points=points))

    assert hostile not in html
    assert "&lt;script&gt;" in html
    assert "<script" not in html.lower()
    assert "<img" not in html.lower()
    assert "<link" not in html.lower()
    assert "<iframe" not in html.lower()
    assert "url(http" not in html.lower()
    assert 'href="http' not in html.lower()
    assert "default-src 'none'" in html


def test_fixture_generation_writes_stable_json_and_html_without_private_paths(tmp_path):
    first = generate_fixture_artifacts(tmp_path / "first")
    second = generate_fixture_artifacts(tmp_path / "second")
    first_json = Path(first["json_path"]).read_bytes()
    second_json = Path(second["json_path"]).read_bytes()
    first_html = Path(first["html_path"]).read_bytes()
    second_html = Path(second["html_path"]).read_bytes()

    assert first["memory_pr_id"] == second["memory_pr_id"]
    assert first["content_digest"] == second["content_digest"]
    assert first_json == second_json
    assert first_html == second_html
    assert str(tmp_path).encode() not in first_json
    assert str(tmp_path).encode() not in first_html
    assert b"Alan" not in first_json
    assert b"Alan" not in first_html


def test_artifact_directory_and_files_use_restrictive_permissions(tmp_path):
    packet = _build()
    output_dir = tmp_path / "private-memory-pr"
    result = write_memory_pr_artifacts(packet, output_dir)

    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(Path(result["json_path"]).stat().st_mode) == 0o600
    assert stat.S_IMODE(Path(result["html_path"]).stat().st_mode) == 0o600


def test_existing_shared_output_directory_is_rejected_without_chmod_or_artifacts(tmp_path):
    packet = _build()
    shared_dir = tmp_path / "shared-output"
    shared_dir.mkdir(mode=0o770)
    shared_dir.chmod(0o770)
    before = shared_dir.stat()

    with pytest.raises(MemoryPRValidationError, match="private"):
        write_memory_pr_artifacts(packet, shared_dir)

    after = shared_dir.stat()
    assert stat.S_IMODE(after.st_mode) == 0o770
    assert after.st_ctime_ns == before.st_ctime_ns
    assert list(shared_dir.iterdir()) == []


def test_existing_private_output_directory_is_accepted_without_permission_change(tmp_path):
    packet = _build()
    private_dir = tmp_path / "private-output"
    private_dir.mkdir(mode=0o700)
    private_dir.chmod(0o700)

    result = write_memory_pr_artifacts(packet, private_dir)

    assert stat.S_IMODE(private_dir.stat().st_mode) == 0o700
    assert Path(result["json_path"]).is_file()


def test_documented_fixture_module_commands_run_without_services(tmp_path):
    generated = subprocess.run(
        [sys.executable, "-m", "qdrant_memory.memory_pr", "fixture", "--output-dir", str(tmp_path / "cli")],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )
    verified = subprocess.run(
        [sys.executable, "-m", "qdrant_memory.memory_pr", "verify-fixture"],
        cwd=Path(__file__).resolve().parents[1],
        check=False,
        capture_output=True,
        text=True,
    )

    assert generated.returncode == 0, generated.stdout + generated.stderr
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert json.loads(verified.stdout)["valid"] is True
    assert list((tmp_path / "cli").glob("memory-pr-*.json"))
    assert list((tmp_path / "cli").glob("memory-pr-*.html"))


class _ReadOnlyQdrant:
    def __init__(self, points):
        self.points = {str(point["id"]): point for point in points}
        self.retrieve_calls = []
        self.scroll_calls = []
        self.upsert_calls = []
        self.payload_update_calls = []
        self.delete_ids_calls = []
        self.delete_filter_calls = []

    def retrieve(self, name, ids, *, with_payload=True, with_vector=False):
        self.retrieve_calls.append((name, list(ids), with_payload, with_vector))
        return [self.points[point_id] for point_id in ids if point_id in self.points]

    def scroll_by_filter(self, name, filter, *, limit=256, with_payload=True, with_vector=False, max_total=None):
        self.scroll_calls.append((name, filter, limit, with_payload, with_vector, max_total))
        return list(self.points.values())[:max_total]

    def upsert(self, *args, **kwargs):
        self.upsert_calls.append((args, kwargs))

    def update_payload(self, *args, **kwargs):
        self.payload_update_calls.append((args, kwargs))

    def delete_ids(self, *args, **kwargs):
        self.delete_ids_calls.append((args, kwargs))

    def delete_filter(self, *args, **kwargs):
        self.delete_filter_calls.append((args, kwargs))


def _provider(tmp_path, points):
    from __init__ import QdrantMemoryProvider

    provider = QdrantMemoryProvider()
    provider._active = True
    provider._hermes_home = str(tmp_path)
    provider._profile_id = "architect"
    provider._config = load_config(hermes_home=str(tmp_path), hermes_config={})
    provider._config.update({"collection_name": "memory", "learning_collection_name": "learnings"})
    provider._qdrant = _ReadOnlyQdrant(points)
    return provider


def _persist_provider_report(tmp_path, points):
    report, _, _ = _report_and_points()
    report["profile_id"] = "architect"
    return persist_consolidation_report(report, hermes_home=str(tmp_path))


def test_memory_pr_tool_is_registered_as_explicitly_read_only():
    schemas = {schema["name"]: schema for schema in TOOL_SCHEMAS}

    assert MEMORY_PR_SCHEMA["name"] == "qdrant_memory_memory_pr"
    assert "read-only" in MEMORY_PR_SCHEMA["description"].lower()
    assert schemas["qdrant_memory_memory_pr"] is MEMORY_PR_SCHEMA
    assert MEMORY_PR_SCHEMA["parameters"]["required"] == ["report_id", "proposal_id"]
    assert "output_dir" in MEMORY_PR_SCHEMA["parameters"]["properties"]


def test_provider_memory_pr_reloads_exact_points_without_mutation_or_default_artifact(tmp_path):
    _, proposal, points = _report_and_points()
    report = _persist_provider_report(tmp_path, points)
    provider = _provider(tmp_path, points)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_memory_pr",
            {"report_id": report["report_id"], "proposal_id": proposal["proposal_id"]},
        )
    )

    assert result["read_only"] is True
    assert result["persisted"] is False
    assert result["packet"]["drift_status"] == "unchanged"
    assert provider._qdrant.retrieve_calls == [("memory", ["runtime-new", "runtime-old"], True, False)]
    assert provider._qdrant.upsert_calls == []
    assert provider._qdrant.payload_update_calls == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []
    assert list(tmp_path.rglob("memory-pr-*.json")) == []
    assert list(tmp_path.rglob("memory-pr-*.html")) == []


def test_provider_memory_pr_persists_only_with_explicit_output_directory(tmp_path):
    _, proposal, points = _report_and_points()
    report = _persist_provider_report(tmp_path, points)
    provider = _provider(tmp_path, points)
    output_dir = tmp_path / "requested-output"

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_memory_pr",
            {
                "report_id": report["report_id"],
                "proposal_id": proposal["proposal_id"],
                "output_dir": str(output_dir),
            },
        )
    )

    assert result["read_only"] is True
    assert result["persisted"] is True
    assert Path(result["artifact"]["json_path"]).is_file()
    assert Path(result["artifact"]["html_path"]).is_file()
    assert provider._qdrant.upsert_calls == []
    assert provider._qdrant.payload_update_calls == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []


def test_live_provider_suppresses_nested_current_and_persisted_identity_before_hash_and_output(tmp_path):
    report, proposal, points = _report_and_points()
    identity_values = [
        "casey@example.invalid",
        "+1 555 010 0199",
        "19 Synthetic Avenue",
        "Casey Fixture",
    ]
    points[0]["payload"].update(
        {
            "text": f"Contact {identity_values[0]} for the private runtime account.",
            "metadata": {
                "classifiers": [
                    {"source_type": "user_profile"},
                    {"fact_key": "user.email"},
                ]
            },
        }
    )
    proposal["candidate_statement"] = f"Review the runtime preference for {identity_values[3]}."
    proposal["proposed_status_changes"][0]["reason"] = f"Requested by {identity_values[0]}."
    proposal["source_snippets"] = [
        {
            "id": "runtime-old",
            "metadata": {
                "contacts": [
                    {"email": identity_values[0]},
                    {"phone": identity_values[1]},
                    {"address": identity_values[2]},
                    {"preferredName": identity_values[3]},
                ]
            },
        }
    ]
    for snapshot in proposal["review_point_snapshots"]:
        point = next(item for item in points if item["id"] == snapshot["id"])
        snapshot["snapshot_digest"] = stable_point_snapshot_digest(point)
    report["profile_id"] = "architect"
    persisted_report = persist_consolidation_report(report, hermes_home=str(tmp_path))
    provider = _provider(tmp_path, points)
    output_dir = tmp_path / "nested-identity-output"

    projection = _snapshot_projection(points[0])
    response_text = provider.handle_tool_call(
        "qdrant_memory_memory_pr",
        {
            "report_id": persisted_report["report_id"],
            "proposal_id": proposal["proposal_id"],
            "output_dir": str(output_dir),
        },
    )
    result = json.loads(response_text)
    packet = result["packet"]
    evidence = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")
    written_json = Path(result["artifact"]["json_path"]).read_text(encoding="utf-8")
    written_html = Path(result["artifact"]["html_path"]).read_text(encoding="utf-8")
    all_outputs = [response_text, written_json, written_html, json.dumps(projection, sort_keys=True)]

    assert result["read_only"] is True
    assert result["persisted"] is True
    assert evidence["identity_bearing"] is True
    assert evidence["snippet"] == IDENTITY_REDACTED_SNIPPET
    assert evidence["snapshot_scope"] == "redacted_sensitive_state"
    assert evidence["drift_status"] == "unknown"
    assert packet["drift_status"] == "unknown"
    assert projection["text"] == IDENTITY_REDACTED_SNIPPET
    assert projection["payload"] == {
        "identity_bearing": True,
        "canonical": False,
        "stale": False,
        "requires_review": True,
        "fact_status": "active",
    }
    assert packet["persisted_evidence"] == [
        {
            "id": "runtime-old",
            "identity_bearing": True,
            "evidence": IDENTITY_REDACTED_SNIPPET,
        }
    ]
    assert "Identity-bearing proposal summary suppressed" in packet["proposal_summary"]
    assert packet["proposed_status_changes"][0]["reason"] == IDENTITY_REDACTED_SNIPPET
    for identity_value in identity_values:
        assert all(identity_value not in output for output in all_outputs)
    assert provider._qdrant.retrieve_calls == [("memory", ["runtime-new", "runtime-old"], True, False)]
    assert provider._qdrant.upsert_calls == []
    assert provider._qdrant.payload_update_calls == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []


@pytest.mark.parametrize(
    ("case_name", "current_extension", "evidence_extension", "identity_values"),
    [
        (
            "ssn-dob-aliases",
            {
                "metadata": {
                    "records": [
                        {"socialSecurityNumber": "SSN-IDENTITY-SENTINEL-9042"},
                        {"dateOfBirth": "DOB-IDENTITY-SENTINEL-1970"},
                    ]
                }
            },
            {
                "metadata": {
                    "records": [
                        {"birthDate": "DOB-IDENTITY-SENTINEL-1970"},
                        {"taxpayerId": "SSN-IDENTITY-SENTINEL-9042"},
                    ]
                }
            },
            ["SSN-IDENTITY-SENTINEL-9042", "DOB-IDENTITY-SENTINEL-1970"],
        ),
        (
            "unknown-nested-shape",
            {
                "reviewExtension": {
                    "records": [{"opaqueBiographicalClaim": "UNKNOWN-IDENTITY-SENTINEL-731"}]
                }
            },
            {
                "legacyEnvelope": {
                    "claims": [{"unlistedPersonalAttribute": "UNKNOWN-IDENTITY-SENTINEL-731"}]
                }
            },
            ["UNKNOWN-IDENTITY-SENTINEL-731"],
        ),
    ],
)
def test_live_provider_structurally_suppresses_alias_and_unknown_nested_shapes(
    tmp_path, case_name, current_extension, evidence_extension, identity_values
):
    report, proposal, points = _report_and_points()
    points[0]["payload"].update(current_extension)
    points[0]["payload"]["text"] = f"{case_name}: {' '.join(identity_values)}"
    proposal["candidate_statement"] = f"Review {' '.join(identity_values)}"
    proposal["proposed_status_changes"][0]["reason"] = f"Reason {' '.join(identity_values)}"
    proposal["source_snippets"] = [{"id": "runtime-old", **evidence_extension}]
    for snapshot in proposal["review_point_snapshots"]:
        point = next(item for item in points if item["id"] == snapshot["id"])
        snapshot["snapshot_digest"] = stable_point_snapshot_digest(point)
    report["profile_id"] = "architect"
    persisted_report = persist_consolidation_report(report, hermes_home=str(tmp_path))
    provider = _provider(tmp_path, points)
    output_dir = tmp_path / f"structural-{case_name}"

    projection = _snapshot_projection(points[0])
    response_text = provider.handle_tool_call(
        "qdrant_memory_memory_pr",
        {
            "report_id": persisted_report["report_id"],
            "proposal_id": proposal["proposal_id"],
            "output_dir": str(output_dir),
        },
    )
    result = json.loads(response_text)
    packet = result["packet"]
    evidence = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")
    written_json = Path(result["artifact"]["json_path"]).read_text(encoding="utf-8")
    written_html = Path(result["artifact"]["html_path"]).read_text(encoding="utf-8")
    outputs = [response_text, written_json, written_html, json.dumps(projection, sort_keys=True)]

    assert evidence["identity_bearing"] is True
    assert evidence["snippet"] == IDENTITY_REDACTED_SNIPPET
    assert evidence["snapshot_scope"] == "redacted_sensitive_state"
    assert evidence["drift_status"] == "unknown"
    assert packet["drift_status"] == "unknown"
    assert projection["text"] == IDENTITY_REDACTED_SNIPPET
    assert packet["persisted_evidence"] == [
        {
            "id": "runtime-old",
            "identity_bearing": True,
            "evidence": IDENTITY_REDACTED_SNIPPET,
        }
    ]
    assert "Identity-bearing proposal summary suppressed" in packet["proposal_summary"]
    assert packet["proposed_status_changes"][0]["reason"] == IDENTITY_REDACTED_SNIPPET
    for identity_value in identity_values:
        assert all(identity_value not in output for output in outputs)
    assert provider._qdrant.retrieve_calls == [("memory", ["runtime-new", "runtime-old"], True, False)]
    assert provider._qdrant.upsert_calls == []
    assert provider._qdrant.payload_update_calls == []
    assert provider._qdrant.delete_ids_calls == []
    assert provider._qdrant.delete_filter_calls == []


def test_live_provider_sensitive_state_and_provenance_preserve_only_validated_values(tmp_path):
    report, proposal, points = _report_and_points()
    invalid_values = [
        "FREEFORM-STATE-SENTINEL",
        "FREEFORM-TIMESTAMP-SENTINEL",
        "FREEFORM-DERIVATION-SENTINEL",
    ]
    points[0]["payload"].update(
        {
            "metadata": {"records": [{"dob": "DOB-STATE-SENTINEL"}]},
            "canonical": "false",
            "stale": 1,
            "requires_review": "yes",
            "fact_status": invalid_values[0],
            "observed_at": invalid_values[1],
            "derivation_type": invalid_values[2],
            "created_at": "2026-01-10T00:00:00Z",
        }
    )
    for snapshot in proposal["review_point_snapshots"]:
        point = next(item for item in points if item["id"] == snapshot["id"])
        snapshot["snapshot_digest"] = stable_point_snapshot_digest(point)
    report["profile_id"] = "architect"
    persisted_report = persist_consolidation_report(report, hermes_home=str(tmp_path))
    provider = _provider(tmp_path, points)

    projection = _snapshot_projection(points[0])
    response_text = provider.handle_tool_call(
        "qdrant_memory_memory_pr",
        {"report_id": persisted_report["report_id"], "proposal_id": proposal["proposal_id"]},
    )
    packet = json.loads(response_text)["packet"]
    evidence = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")
    rendered = json.dumps({"packet": packet, "projection": projection}, sort_keys=True)

    assert projection["payload"] == {"identity_bearing": True, "requires_review": True}
    assert projection["provenance"] == {"created_at": "2026-01-10T00:00:00Z"}
    assert evidence["state"] == {"requires_review": True}
    assert evidence["provenance"] == {"created_at": "2026-01-10T00:00:00Z"}
    assert evidence["drift_status"] == "unknown"
    assert "DOB-STATE-SENTINEL" not in rendered
    for invalid_value in invalid_values:
        assert invalid_value not in rendered


def test_live_provider_safe_control_schema_remains_informative_and_comparable(tmp_path):
    report, proposal, points = _report_and_points()
    points[0]["payload"].update(
        {
            "source": "fixture-release-notes",
            "chunk_type": "fact",
            "importance": 7,
            "confidence": 0.91,
            "access_count": 4,
            "last_accessed": "2026-07-18T00:00:00Z",
            "decay_score": 0.8,
            "tags": ["atlas", "runtime"],
            "locator": {"line_start": 4, "line_end": 8, "heading": "Runtime support"},
            "derived_from": [
                {
                    "source_uri": "fixture://atlas/releases/2026-01",
                    "source_type": "release_note",
                    "content_hash": "fixture-atlas-source",
                    "source_modified_at": "2026-01-10T00:00:00Z",
                    "derivation_type": "indexed_chunk",
                    "relation_type": "DERIVED_FROM",
                    "locator": {"line_start": 4},
                }
            ],
        }
    )
    proposal["source_snippets"] = [
        {
            "id": "runtime-old",
            "snippet": "The Atlas worker uses runtime v1.",
            "source_type": "release_note",
            "fact_status": "active",
            "observed_at": "2026-01-10T00:00:00Z",
        }
    ]
    for snapshot in proposal["review_point_snapshots"]:
        point = next(item for item in points if item["id"] == snapshot["id"])
        snapshot["snapshot_digest"] = stable_point_snapshot_digest(point)
    report["profile_id"] = "architect"
    persisted_report = persist_consolidation_report(report, hermes_home=str(tmp_path))
    provider = _provider(tmp_path, points)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_memory_pr",
            {"report_id": persisted_report["report_id"], "proposal_id": proposal["proposal_id"]},
        )
    )
    packet = result["packet"]
    evidence = next(item for item in packet["current_evidence"] if item["id"] == "runtime-old")
    projection = _snapshot_projection(points[0])

    assert evidence["identity_bearing"] is False
    assert evidence["snippet"] == "The Atlas worker uses runtime v1."
    assert evidence["drift_status"] == "unchanged"
    assert packet["drift_status"] == "unchanged"
    assert evidence["provenance"]["observed_at"] == "2026-01-10T00:00:00Z"
    assert evidence["provenance"]["derived_from"][0]["relation_type"] == "DERIVED_FROM"
    assert projection["text"] == "The Atlas worker uses runtime v1."
    assert projection["payload"]["locator"]["heading"] == "Runtime support"
    assert packet["persisted_evidence"][0]["snippet"] == "The Atlas worker uses runtime v1."


def test_live_provider_rejects_proposal_collection_outside_configured_allowlist(tmp_path):
    report, proposal, points = _report_and_points()
    report["profile_id"] = "architect"
    proposal["collection_name"] = "restored_archive"
    persisted_report = persist_consolidation_report(report, hermes_home=str(tmp_path))
    provider = _provider(tmp_path, points)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_memory_pr",
            {"report_id": persisted_report["report_id"], "proposal_id": proposal["proposal_id"]},
        )
    )

    assert "error" in result
    assert "configured memory or learning collection" in result["error"]
    assert provider._qdrant.retrieve_calls == []


def test_provider_missing_report_without_output_does_not_create_or_chmod_paths(tmp_path):
    fresh_home = tmp_path / "fresh-hermes-home"
    fresh_home.mkdir(mode=0o755)
    fresh_home.chmod(0o755)
    before = fresh_home.stat()
    provider = _provider(fresh_home, [])

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_memory_pr",
            {"report_id": "missing-report", "proposal_id": "missing-proposal"},
        )
    )

    after = fresh_home.stat()
    assert "error" in result
    assert "not found" in result["error"]
    assert not (fresh_home / "qdrant_memory").exists()
    assert stat.S_IMODE(after.st_mode) == stat.S_IMODE(before.st_mode) == 0o755
    assert after.st_mtime_ns == before.st_mtime_ns
    assert after.st_ctime_ns == before.st_ctime_ns


def test_consolidation_report_snapshots_cover_every_proposal_affected_id(tmp_path):
    _, _, points = _report_and_points()
    provider = _provider(tmp_path, points)

    result = json.loads(
        provider.handle_tool_call(
            "qdrant_memory_consolidate",
            {"scope": "memory", "include_reconsolidation": True, "persist": True},
        )
    )
    proposal = next(item for item in result["proposals"] if item["proposal_type"] == "fact_supersession_candidate")

    assert {item["id"] for item in proposal["review_point_snapshots"]} == set(proposal["affected_ids"])
    assert all(len(item["snapshot_digest"]) == 64 for item in proposal["review_point_snapshots"])
    assert {item["projection"]["version"] for item in proposal["review_point_snapshots"]} == {1}
