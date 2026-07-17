from __future__ import annotations

import json

from qdrant_memory.guarded_auto import GuardedAutoPolicy, apply_guarded_auto


def _eligible_duplicate(*, proposal_id: str = "duplicate-safe") -> dict:
    return {
        "proposal_id": proposal_id,
        "proposal_type": "duplicate_cluster",
        "affected_ids": ["a", "b"],
        "confidence": 0.99,
        "risk": "low",
        "match_kind": "exact_normalized",
        "guarded_auto_eligible": True,
        "preauthorized_policy": "guarded-auto:exact-duplicate-merge",
    }


def test_guarded_auto_redacts_exception_text_and_untrusted_proposal_handle():
    sentinel = "credential-" + "must-not-escape"

    class RaisingProvider:
        def handle_tool_call(self, _tool_name, _args):
            raise RuntimeError(f"provider failed with {sentinel}")

    proposal_id = f"unsafe/{sentinel}"
    summary = apply_guarded_auto(
        RaisingProvider(),
        {"report_id": "report-safe", "proposals": [_eligible_duplicate(proposal_id=proposal_id)]},
        GuardedAutoPolicy(mode="guarded-auto"),
    )
    encoded = json.dumps(summary, sort_keys=True)

    assert sentinel not in encoded
    assert proposal_id not in encoded
    assert summary["errors"] == [
        {
            "code": "provider_exception",
            "proposal_handle": summary["errors"][0]["proposal_handle"],
            "action": "merge",
        }
    ]
    assert summary["errors"][0]["proposal_handle"].startswith("sha256:")


def test_guarded_auto_redacts_provider_error_text():
    sentinel = "bearer-" + "must-not-escape"

    class RejectingProvider:
        def handle_tool_call(self, _tool_name, _args):
            return json.dumps({"error": f"upstream rejected {sentinel}"})

    summary = apply_guarded_auto(
        RejectingProvider(),
        {"report_id": "report-safe", "proposals": [_eligible_duplicate()]},
        GuardedAutoPolicy(mode="guarded-auto"),
    )
    encoded = json.dumps(summary, sort_keys=True)

    assert sentinel not in encoded
    assert summary["errors"] == [
        {
            "code": "provider_rejected",
            "proposal_handle": "duplicate-safe",
            "action": "merge",
        }
    ]


def test_guarded_auto_reports_invalid_json_without_raw_decoder_text():
    class InvalidJsonProvider:
        def handle_tool_call(self, _tool_name, _args):
            return "not-json"

    summary = apply_guarded_auto(
        InvalidJsonProvider(),
        {"report_id": "report-safe", "proposals": [_eligible_duplicate()]},
        GuardedAutoPolicy(mode="guarded-auto"),
    )

    assert summary["errors"] == [
        {
            "code": "invalid_provider_response",
            "proposal_handle": "duplicate-safe",
            "action": "merge",
        }
    ]


def test_guarded_auto_missing_report_id_uses_allowlisted_error_code():
    summary = apply_guarded_auto(
        object(),
        {"proposals": [_eligible_duplicate()]},
        GuardedAutoPolicy(mode="guarded-auto"),
    )

    assert summary["errors"] == [{"code": "missing_report_id"}]