from __future__ import annotations

import json

import pytest

from qdrant_memory.consolidation import expected_action_for_proposal
from qdrant_memory.guarded_auto import GuardedAutoPolicy, guarded_auto_action_for_proposal
from qdrant_memory.ontology_suggestions import (
    ONTOLOGY_SUGGESTION_TYPES,
    build_ontology_extraction_candidate,
    preview_ontology_suggestions,
    suggest_fact_key_pattern_promotion,
    suggest_new_memory_kind,
    suggest_new_relation_type,
    suggest_subject_alias_normalization,
    suggest_tag_merge_or_rename,
    write_ontology_suggestion_draft,
)
from qdrant_memory.schema import MEMORY_KINDS, RELATION_TYPES


class FakeEmbeddings:
    def __init__(self):
        self.documents = []

    def embed_document(self, text):  # pragma: no cover - must not be called
        self.documents.append(text)
        raise AssertionError("ontology suggestion preview must not construct embeddings")


class FakeQdrant:
    def __init__(self):
        self.upserts = []
        self.deletes = []
        self.payload_updates = []

    def upsert(self, collection_name, points):  # pragma: no cover - must not be called
        self.upserts.append((collection_name, points))
        raise AssertionError("ontology suggestion preview must not mutate Qdrant")

    def delete_ids(self, collection_name, ids):  # pragma: no cover - must not be called
        self.deletes.append((collection_name, ids))
        raise AssertionError("ontology suggestion preview must not mutate Qdrant")

    def update_payload(self, collection_name, point_id, payload):  # pragma: no cover - must not be called
        self.payload_updates.append((collection_name, point_id, payload))
        raise AssertionError("ontology suggestion preview must not mutate Qdrant")


def _all_suggestions():
    return [
        suggest_new_memory_kind(
            "experiment_note",
            evidence=["Three reviewed memories describe experiment notes without a precise kind."],
            source_uri="session://task-13-2/memory-kind",
            confidence=0.72,
        ),
        suggest_new_relation_type(
            "VALIDATES",
            evidence=["Several assertion edges say one check validates another claim."],
            source_uri="session://task-13-2/relation-type",
            confidence=0.68,
        ),
        suggest_tag_merge_or_rename(
            source_tags=["build", "ci"],
            canonical_tag="build-ci",
            evidence=["The tags build and ci appear together in repeated workflow memories."],
            source_uri="session://task-13-2/tags",
            confidence=0.81,
        ),
        suggest_subject_alias_normalization(
            canonical_subject="hermes-agent",
            aliases=["Hermes", "Hermes Agent", "hermes-agent"],
            evidence=["Subject aliases refer to the same agent/runtime."],
            source_uri="session://task-13-2/subjects",
            confidence=0.77,
        ),
        suggest_fact_key_pattern_promotion(
            pattern="project.build_tool",
            examples=["project.alpha.build_tool", "project.beta.build_tool", "project.gamma.build_tool"],
            evidence=["Repeated fact keys share the project.<name>.build_tool pattern."],
            source_uri="session://task-13-2/fact-key",
            confidence=0.84,
        ),
    ]


def test_ontology_suggestions_preview_all_types_as_draft_only_without_schema_mutation_or_auto_apply():
    before_memory_kinds = tuple(MEMORY_KINDS)
    before_relation_types = tuple(RELATION_TYPES)
    embeddings = FakeEmbeddings()
    qdrant = FakeQdrant()

    preview = preview_ontology_suggestions(_all_suggestions(), qdrant=qdrant, embeddings=embeddings)

    assert set(ONTOLOGY_SUGGESTION_TYPES) == {
        "new_memory_kind",
        "new_relation_type",
        "merge_rename_tags",
        "normalize_subject_aliases",
        "promote_fact_key_pattern",
    }
    assert preview["dry_run"] is True
    assert preview["auto_apply_allowed"] is False
    assert preview["schema_mutation_allowed"] is False
    assert preview["count"] == 5
    assert preview["proposal_ids"] == [
        "ontology-new-memory-kind-experiment-note",
        "ontology-new-relation-type-validates",
        "ontology-merge-rename-tags-build-ci",
        "ontology-normalize-subject-aliases-hermes-agent",
        "ontology-promote-fact-key-pattern-project-build-tool",
    ]
    assert [item["suggestion_type"] for item in preview["proposals"]] == list(ONTOLOGY_SUGGESTION_TYPES)

    for proposal in preview["proposals"]:
        assert proposal["proposal_type"] == "ontology_suggestion"
        assert proposal["suggested_action"] == "draft_review_only"
        assert proposal["affected_ids"] == []
        assert proposal["requires_review"] is True
        assert proposal["manual_review_required"] is True
        assert proposal["auto_apply_eligible"] is False
        assert proposal["schema_mutation_allowed"] is False
        assert proposal["accepted_change_path"] == "normal_code_docs_and_tests"
        assert expected_action_for_proposal(proposal["proposal_type"]) is None
        action, reason = guarded_auto_action_for_proposal(proposal, GuardedAutoPolicy(mode="guarded-auto"))
        assert action is None
        assert "manual review" in reason or "affected_ids" in reason

    assert tuple(MEMORY_KINDS) == before_memory_kinds
    assert tuple(RELATION_TYPES) == before_relation_types
    assert "experiment_note" not in MEMORY_KINDS
    assert "VALIDATES" not in RELATION_TYPES
    assert embeddings.documents == []
    assert qdrant.upserts == []
    assert qdrant.deletes == []
    assert qdrant.payload_updates == []


@pytest.mark.parametrize(
    ("suggestion", "expected_proposal_id"),
    [
        (_all_suggestions()[0], "ontology-new-memory-kind-experiment-note"),
        (_all_suggestions()[1], "ontology-new-relation-type-validates"),
        (_all_suggestions()[2], "ontology-merge-rename-tags-build-ci"),
        (_all_suggestions()[3], "ontology-normalize-subject-aliases-hermes-agent"),
        (_all_suggestions()[4], "ontology-promote-fact-key-pattern-project-build-tool"),
    ],
)
def test_ontology_suggestion_drafts_are_review_artifacts_for_each_suggestion_type(tmp_path, suggestion, expected_proposal_id):
    metadata = write_ontology_suggestion_draft(
        suggestion,
        report_id="ontology-13-2",
        hermes_home=str(tmp_path / "hermes"),
        config={},
    )

    assert metadata["report_id"] == "ontology-13-2"
    assert metadata["proposal_id"] == expected_proposal_id
    assert metadata["proposal_type"] == "ontology_suggestion"
    assert metadata["requires_review"] is True
    assert metadata["auto_apply_eligible"] is False
    assert metadata["schema_mutation_allowed"] is False
    assert metadata["kind"] == "ontology_suggestion_draft"

    markdown = (tmp_path / "hermes" / "qdrant_memory" / "proposals" / f"{metadata['draft_id']}.md").read_text(encoding="utf-8")
    metadata_payload = json.loads((tmp_path / "hermes" / "qdrant_memory" / "proposals" / f"{metadata['draft_id']}.json").read_text(encoding="utf-8"))

    assert expected_proposal_id in markdown
    assert "Proposal/draft artifact only" in markdown
    assert "must not mutate schema" in markdown
    assert "No cron/watcher auto-apply" in markdown
    assert "normal code/docs changes and tests" in markdown
    assert metadata_payload["proposal_id"] == expected_proposal_id
    assert metadata_payload["auto_apply_eligible"] is False
    assert metadata_payload["schema_mutation_allowed"] is False


def test_secret_and_identity_bearing_ontology_suggestions_are_redacted_and_forced_to_manual_review(tmp_path):
    sensitive_key = "api" + "_key"
    sensitive_value = "secret" + "-value"
    identity_value = "alice" + "@" + "example.test"
    suggestion = suggest_subject_alias_normalization(
        canonical_subject="project-contact",
        aliases=[identity_value, "release owner"],
        evidence=[{"note": f"alias observed for {identity_value}", sensitive_key: sensitive_value}],
        source_uri="session://task-13-2/sensitive",
        confidence=0.93,
    )

    payload = suggestion.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    assert payload["requires_review"] is True
    assert payload["manual_review_required"] is True
    assert payload["risk"] == "high"
    assert payload["redaction_applied"] is True
    assert payload["proposed_payload"]["safety"]["identity_bearing"] is True
    assert payload["proposed_payload"]["safety"]["secret_bearing"] is True
    assert identity_value not in serialized
    assert sensitive_value not in serialized
    assert "redacted" in serialized

    metadata = write_ontology_suggestion_draft(suggestion, report_id="ontology-13-2", hermes_home=str(tmp_path / "hermes"), config={})
    markdown = (tmp_path / "hermes" / "qdrant_memory" / "proposals" / f"{metadata['draft_id']}.md").read_text(encoding="utf-8")
    assert identity_value not in markdown
    assert sensitive_value not in markdown
    assert "manual review" in markdown
    assert "redacted" in markdown


def test_numeric_secret_and_identity_keyed_evidence_is_redacted_without_hiding_harmless_counters():
    chat_id_value = 8675309001
    user_id_value = 4242424242
    numeric_secret_value = 3141592653
    harmless_counter_value = 2718281828
    sensitive_key = "tok" + "en"
    suggestion = suggest_fact_key_pattern_promotion(
        pattern="session.identity.example_count",
        examples=["session.alpha.example_count", "session.beta.example_count"],
        evidence=[
            {
                "chat_id": chat_id_value,
                "user_id": user_id_value,
                sensitive_key: numeric_secret_value,
                "retry_count": harmless_counter_value,
            }
        ],
        source_uri="session://task-13-2/numeric-keyed-evidence",
        confidence=0.88,
    )

    payload = suggestion.to_dict()
    serialized = json.dumps(payload, sort_keys=True)
    redacted_evidence = payload["evidence"][0]
    assert payload["requires_review"] is True
    assert payload["manual_review_required"] is True
    assert payload["risk"] == "high"
    assert payload["redaction_applied"] is True
    assert payload["proposed_payload"]["safety"]["identity_bearing"] is True
    assert payload["proposed_payload"]["safety"]["secret_bearing"] is True
    assert redacted_evidence["chat_id"] == "[redacted: identity-bearing value]"
    assert redacted_evidence["user_id"] == "[redacted: identity-bearing value]"
    assert redacted_evidence[sensitive_key] == "[redacted: possible secret-bearing value]"
    assert redacted_evidence["retry_count"] == harmless_counter_value
    assert str(chat_id_value) not in serialized
    assert str(user_id_value) not in serialized
    assert str(numeric_secret_value) not in serialized
    assert str(harmless_counter_value) in serialized


def test_ontology_suggestions_can_be_serialized_as_existing_extraction_candidates_without_validating_as_live_schema():
    suggestion = suggest_new_memory_kind(
        "experiment_note",
        evidence=["Repeated reviewed memories use experiment-note semantics."],
        source_uri="session://task-13-2/candidate",
        confidence=0.72,
    )

    candidate = build_ontology_extraction_candidate(suggestion, created_at="2026-05-27T00:00:00+00:00")

    assert candidate.candidate_type == "ontology_suggestion"
    assert candidate.proposed_payload["proposal_id"] == "ontology-new-memory-kind-experiment-note"
    assert candidate.proposed_payload["ontology_field"] == "memory_kind"
    assert candidate.proposed_payload["candidate_value"] == "experiment_note"
    assert candidate.proposed_payload["schema_mutation_allowed"] is False
    assert candidate.requires_review is True
    json.dumps(candidate.to_dict(), sort_keys=True, allow_nan=False)
    assert "experiment_note" not in MEMORY_KINDS
