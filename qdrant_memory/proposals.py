from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from qdrant_memory.consolidation import redact_secrets
from qdrant_memory.write_gate import WriteDecision

_SOURCE_KEYS = (
    "source_uri",
    "source_type",
    "locator",
    "content_hash",
    "source_modified_at",
    "derivation_type",
    "derived_from",
    "canonical",
    "stale",
    "requires_review",
)


def _safe_id(value: str) -> str:
    safe = "".join(ch for ch in str(value or "") if ch.isalnum() or ch in {"-", "_"})
    if not safe or safe != str(value):
        raise ValueError("invalid proposal draft id")
    return safe


def _snippet(text: str, max_chars: int = 500) -> str:
    text = " ".join(str(text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _point_id(point: Any) -> str:
    if isinstance(point, dict):
        return str(point.get("id") or "")
    return str(getattr(point, "id", "") or "")


def _point_text(point: Any) -> str:
    if isinstance(point, dict):
        raw_payload = point.get("payload")
        payload = raw_payload if isinstance(raw_payload, dict) else {}
        return str(point.get("text") or payload.get("text") or payload.get("lesson") or "")
    return str(getattr(point, "text", "") or "")


def _point_payload(point: Any) -> dict[str, Any]:
    if isinstance(point, dict):
        payload = point.get("payload")
    else:
        payload = getattr(point, "payload", None)
    return payload if isinstance(payload, dict) else {}


def _source_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: payload.get(key) for key in _SOURCE_KEYS if payload.get(key) not in (None, "", [], {})}


def proposal_root(hermes_home: str, config: Mapping[str, Any] | None = None) -> Path:
    config = config or {}
    configured = str(config.get("proposal_artifact_dir") or "").strip()
    if configured:
        root = Path(configured).expanduser()
    elif config.get("obsidian_adapter_enabled") and str(config.get("obsidian_vault_root") or "").strip() and str(config.get("obsidian_proposal_dir") or "").strip():
        vault = Path(str(config.get("obsidian_vault_root"))).expanduser().resolve()
        root = (vault / str(config.get("obsidian_proposal_dir"))).resolve()
        try:
            root.relative_to(vault)
        except ValueError as exc:
            raise ValueError("obsidian_proposal_dir must stay inside obsidian_vault_root") from exc
    else:
        root = Path(hermes_home or str(Path.home() / ".hermes")) / "qdrant_memory" / "proposals"
    root.mkdir(parents=True, exist_ok=True)
    try:
        root.chmod(0o700)
    except Exception:
        pass
    return root


def render_proposal_markdown(
    *,
    report: dict[str, Any],
    proposal: dict[str, Any],
    points: list[Any],
    write_decision: WriteDecision | dict[str, Any] | None = None,
) -> str:
    safe_report = redact_secrets(report)
    safe_proposal = redact_secrets(proposal)
    decision = write_decision.to_dict() if isinstance(write_decision, WriteDecision) else write_decision
    title = "# Reconsolidation review draft" if safe_proposal.get("proposal_type") == "reconsolidation_candidate" else "# Qdrant memory proposal draft"
    lines = [
        title,
        "",
        "This is a neutral review artifact. It does not mutate Qdrant memory or user files by itself.",
        "",
        f"- report_id: {safe_report.get('report_id', 'N/A')}",
        f"- proposal_id: {safe_proposal.get('proposal_id', 'N/A')}",
        f"- proposal_type: {safe_proposal.get('proposal_type', 'N/A')}",
        f"- suggested_action: {safe_proposal.get('suggested_action', safe_proposal.get('action', 'N/A'))}",
        f"- affected_ids: {', '.join(str(item) for item in safe_proposal.get('affected_ids', []))}",
    ]
    if decision:
        lines.extend([f"- write_decision: {decision.get('decision')}", f"- requires_review: {decision.get('requires_review')}"])
    lines.extend(["", "## Source points"])
    for point in points:
        point_id = _point_id(point)
        payload = redact_secrets(_point_payload(point))
        source = _source_metadata(payload)
        lines.extend(["", f"### {point_id or 'unknown-point'}", "", "```json", json.dumps({"id": point_id, "source": source}, indent=2, sort_keys=True, default=str), "```", "", "#### Snippet", _snippet(str(redact_secrets(_point_text(point)))) or "N/A"])
    lines.extend(
        [
            "",
            "## Reviewer checklist",
            "",
            "- Verify the source point IDs and provenance before treating this draft as canonical.",
            "- Prefer exact point IDs and dry-run previews for any follow-up memory mutation.",
            "- If this should become a durable skill or memory, perform that write through the explicit approved tool path.",
            "",
        ]
    )
    return "\n".join(lines)


def proposal_draft_metadata(
    *,
    report: dict[str, Any],
    proposal: dict[str, Any],
    points: list[Any],
    hermes_home: str,
    config: Mapping[str, Any] | None = None,
    write_decision: WriteDecision | dict[str, Any] | None = None,
) -> dict[str, Any]:
    proposal_id = _safe_id(str(proposal.get("proposal_id") or "proposal"))
    report_id = _safe_id(str(report.get("report_id") or "report"))
    digest = hashlib.sha256(json.dumps({"report_id": report_id, "proposal_id": proposal_id, "ids": [_point_id(p) for p in points]}, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    draft_id = f"{report_id}-{proposal_id}-{digest}"
    root = proposal_root(hermes_home, config)
    markdown_path = root / f"{draft_id}.md"
    metadata_path = root / f"{draft_id}.json"
    decision_payload = write_decision.to_dict() if isinstance(write_decision, WriteDecision) else write_decision
    return {
        "draft_id": draft_id,
        "report_id": report_id,
        "proposal_id": proposal_id,
        "proposal_type": proposal.get("proposal_type"),
        "path": str(markdown_path),
        "metadata_path": str(metadata_path),
        "source_point_ids": [_point_id(point) for point in points if _point_id(point)],
        "requires_review": True,
        "write_decision": decision_payload,
    }


def write_proposal_draft(
    *,
    report: dict[str, Any],
    proposal: dict[str, Any],
    points: list[Any],
    hermes_home: str,
    config: Mapping[str, Any] | None = None,
    write_decision: WriteDecision | dict[str, Any] | None = None,
) -> dict[str, Any]:
    metadata = proposal_draft_metadata(report=report, proposal=proposal, points=points, hermes_home=hermes_home, config=config, write_decision=write_decision)
    markdown_path = Path(str(metadata["path"]))
    metadata_path = Path(str(metadata["metadata_path"]))
    markdown = render_proposal_markdown(report=report, proposal=proposal, points=points, write_decision=write_decision)
    markdown_path.write_text(markdown, encoding="utf-8")
    metadata_path.write_text(json.dumps(redact_secrets(metadata), indent=2, sort_keys=True, default=str), encoding="utf-8")
    return metadata


def list_proposal_drafts(*, hermes_home: str, config: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    root = proposal_root(hermes_home, config)
    drafts: list[dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and data.get("draft_id"):
            drafts.append(data)
    return drafts


def load_proposal_draft(draft_id: str, *, hermes_home: str, config: Mapping[str, Any] | None = None) -> dict[str, Any]:
    safe_id = _safe_id(draft_id)
    root = proposal_root(hermes_home, config)
    metadata_path = root / f"{safe_id}.json"
    markdown_path = root / f"{safe_id}.md"
    if not metadata_path.exists() or not markdown_path.exists():
        raise FileNotFoundError("proposal draft not found")
    data = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("invalid proposal draft metadata")
    data["markdown"] = markdown_path.read_text(encoding="utf-8")
    return data
