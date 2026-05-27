from __future__ import annotations

STATUS_SCHEMA = {
    "name": "qdrant_memory_status",
    "description": "Check Qdrant memory provider status, collection counts, embedding model, and queue health.",
    "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
}

STORE_SCHEMA = {
    "name": "qdrant_memory_store",
    "description": "Store an explicit memory in the local Qdrant memory collection.",
    "parameters": {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "The memory text to store."},
            "source_type": {"type": "string", "description": "Memory source type. Defaults to manual."},
            "importance": {"type": "integer", "description": "Importance from 1 to 10. Defaults to 5.", "minimum": 1, "maximum": 10},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
            "dry_run": {"type": "boolean", "description": "When true, preview without mutating Qdrant. Defaults to true."},
            "approve": {"type": "boolean", "description": "Required true when dry_run=false."},
            "duplicate_preview": {"type": "boolean", "description": "When true, search for semantic duplicates before storing and return duplicate details without upserting if one is found."},
            "duplicate_threshold": {"type": "number", "description": "Semantic duplicate threshold. Defaults to config/manual_store_duplicate_threshold.", "minimum": 0, "maximum": 1},
            "duplicate_top_k": {"type": "integer", "description": "Maximum duplicate candidates to inspect. Defaults to config/manual_store_duplicate_top_k.", "minimum": 1, "maximum": 20},
        },
        "required": ["text"],
        "additionalProperties": False,
    },
}

SEARCH_SCHEMA = {
    "name": "qdrant_memory_search",
    "description": "Search local Qdrant long-term memory by semantic similarity.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "top_k": {"type": "integer", "description": "Maximum results, 1 to 20. Defaults to 5.", "minimum": 1, "maximum": 20},
            "source_type": {"type": "string", "description": "Optional source_type filter."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags filter. All tags must match."},
            "source": {"type": "string", "description": "Optional exact payload source filter."},
            "file_path": {"type": "string", "description": "Optional exact payload file_path filter."},
            "project_path": {"type": "string", "description": "Optional exact payload project_path filter."},
            "since": {"type": "string", "description": "Optional inclusive created_at lower bound (ISO timestamp)."},
            "until": {"type": "string", "description": "Optional inclusive created_at upper bound (ISO timestamp)."},
            "collection": {"type": "string", "enum": ["memory", "learning"], "description": "Collection to search. Defaults to memory.", "default": "memory"},
            "include_metadata": {"type": "boolean", "description": "Include full payload metadata. Defaults to false."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

INDEX_SCHEMA = {
    "name": "qdrant_memory_index",
    "description": "Safely index markdown/text files or source folders into local Qdrant memory. Dry-run defaults to true.",
    "parameters": {
        "type": "object",
        "properties": {
            "paths": {"type": "array", "items": {"type": "string"}, "description": "Files or directories to index. Defaults to configured index_dirs."},
            "dry_run": {"type": "boolean", "description": "When true, prepare chunks but do not embed/upsert. Defaults to true."},
            "force": {"type": "boolean", "description": "Re-index chunks even if unchanged. Currently accepted for compatibility."},
            "max_files": {"type": "integer", "description": "Maximum files to scan/index for this run.", "minimum": 1},
        },
        "additionalProperties": False,
    },
}

FORGET_SCHEMA = {
    "name": "qdrant_memory_forget",
    "description": "Delete explicit memory points by id. Dry-run defaults to true; no query deletion is supported.",
    "parameters": {
        "type": "object",
        "properties": {
            "ids": {"type": "array", "items": {"type": "string"}, "description": "Point IDs to delete."},
            "dry_run": {"type": "boolean", "description": "When true, only report what would be deleted. Defaults to true."},
        },
        "required": ["ids"],
        "additionalProperties": False,
    },
}

INSPECT_SCHEMA = {
    "name": "qdrant_memory_inspect",
    "description": "Read-only exact inspection of one Qdrant memory point by explicit ID. No semantic search or mutation.",
    "parameters": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "Explicit Qdrant point ID to inspect."},
            "collection": {"type": "string", "enum": ["memory", "learning"], "description": "Collection to inspect. Defaults to memory.", "default": "memory"},
        },
        "required": ["point_id"],
        "additionalProperties": False,
    },
}

TRACE_SCHEMA = {
    "name": "qdrant_memory_trace",
    "description": "Read-only provenance trace for one point. Shows direct upstream derived_from links; downstream is reported unsupported unless enabled later.",
    "parameters": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "Explicit Qdrant point ID to trace."},
            "direction": {"type": "string", "enum": ["upstream", "downstream", "both"], "description": "Trace direction. Defaults to upstream.", "default": "upstream"},
            "collection": {"type": "string", "enum": ["memory", "learning"], "description": "Collection to trace. Defaults to memory.", "default": "memory"},
        },
        "required": ["point_id"],
        "additionalProperties": False,
    },
}

EXPAND_SCHEMA = {
    "name": "qdrant_memory_expand",
    "description": "Read-only bounded expansion of one point's source using source_uri and locator metadata when available.",
    "parameters": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "Explicit Qdrant point ID to expand."},
            "mode": {"type": "string", "enum": ["excerpt", "source", "neighbors"], "description": "Expansion mode. Defaults to excerpt.", "default": "excerpt"},
            "max_chars": {"type": "integer", "description": "Maximum characters to return. Defaults to 8000.", "minimum": 1, "maximum": 100000},
            "collection": {"type": "string", "enum": ["memory", "learning"], "description": "Collection to expand from. Defaults to memory.", "default": "memory"},
        },
        "required": ["point_id"],
        "additionalProperties": False,
    },
}

SOURCE_STATUS_SCHEMA = {
    "name": "qdrant_memory_source_status",
    "description": "Read-only source existence/staleness check for one point's source_uri and locator metadata.",
    "parameters": {
        "type": "object",
        "properties": {
            "point_id": {"type": "string", "description": "Explicit Qdrant point ID whose source should be checked."},
            "collection": {"type": "string", "enum": ["memory", "learning"], "description": "Collection to inspect. Defaults to memory.", "default": "memory"},
        },
        "required": ["point_id"],
        "additionalProperties": False,
    },
}

LEARNING_STORE_SCHEMA = {
    "name": "qdrant_learning_store",
    "description": "Store an explicit procedural learning in the separate Qdrant learning collection. Manual/gated only; not automatic learning.",
    "parameters": {
        "type": "object",
        "properties": {
            "lesson": {"type": "string", "description": "The durable lesson/procedure learned."},
            "learning_type": {
                "type": "string",
                "description": "tool_failure_lesson, user_correction, workflow_lesson, or environment_quirk. Auto-classified if omitted.",
            },
            "trigger": {"type": "string", "description": "Situation that should trigger recall of this lesson."},
            "mistake": {"type": "string", "description": "What went wrong or what should be avoided."},
            "correction": {"type": "string", "description": "The corrected action/procedure."},
            "evidence": {"type": "string", "description": "Evidence that supports the lesson."},
            "tool_name": {"type": "string", "description": "Tool involved, if any."},
            "command": {"type": "string", "description": "Command involved, if any."},
            "importance": {"type": "integer", "description": "Importance from 1 to 10. Defaults to 7.", "minimum": 1, "maximum": 10},
            "confidence": {"type": "number", "description": "Confidence 0 to 1. Defaults to 0.8.", "minimum": 0, "maximum": 1},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags."},
            "promote_to_skill_candidate": {"type": "boolean", "description": "Mark as candidate for future skill promotion. Defaults false."},
        },
        "required": ["lesson"],
        "additionalProperties": False,
    },
}

LEARNING_SEARCH_SCHEMA = {
    "name": "qdrant_learning_search",
    "description": "Search procedural learnings from the separate Qdrant learning collection.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What procedural lesson to search for."},
            "top_k": {"type": "integer", "description": "Maximum results, 1 to 20. Defaults to 5.", "minimum": 1, "maximum": 20},
            "learning_type": {"type": "string", "description": "Optional learning_type filter."},
            "tags": {"type": "array", "items": {"type": "string"}, "description": "Optional tags filter. All tags must match."},
            "source": {"type": "string", "description": "Optional exact payload source filter."},
            "file_path": {"type": "string", "description": "Optional exact payload file_path filter."},
            "project_path": {"type": "string", "description": "Optional exact payload project_path filter."},
            "since": {"type": "string", "description": "Optional inclusive created_at lower bound (ISO timestamp)."},
            "until": {"type": "string", "description": "Optional inclusive created_at upper bound (ISO timestamp)."},
            "include_metadata": {"type": "boolean", "description": "Include full payload metadata. Defaults to false."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

LEARNING_PREVIEW_SCHEMA = {
    "name": "qdrant_learning_preview",
    "description": "Preview gated automatic learning candidates detected from session/compression hooks. Dry-run only; does not store memories.",
    "parameters": {
        "type": "object",
        "properties": {
            "include_metadata": {"type": "boolean", "description": "Include hook/source metadata. Defaults to false."},
        },
        "additionalProperties": False,
    },
}

LEARNING_APPROVE_SCHEMA = {
    "name": "qdrant_learning_approve",
    "description": "Approve one pending gated learning candidate by candidate_id. Dry-run defaults to true; live mode stores to the learning collection only.",
    "parameters": {
        "type": "object",
        "properties": {
            "candidate_id": {"type": "string", "description": "Candidate ID from qdrant_learning_preview."},
            "dry_run": {"type": "boolean", "description": "When true, preview approval without storing. Defaults to true."},
        },
        "required": ["candidate_id"],
        "additionalProperties": False,
    },
}

CONSOLIDATE_SCHEMA = {
    "name": "qdrant_memory_consolidate",
    "description": "Generate a dry-run sleep consolidation report and optionally persist it as a local artifact. Live memory actions require qdrant_memory_consolidation_apply.",
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {"type": "boolean", "description": "Must remain true for report generation. Defaults to true; false is rejected."},
            "scope": {"type": "string", "enum": ["memory", "learning", "both"], "description": "Report scope: memory, learning, or both. Defaults to both."},
            "max_points": {"type": "integer", "description": "Maximum points to inspect per collection. Defaults to config.", "minimum": 1, "maximum": 1000},
            "max_groups": {"type": "integer", "description": "Maximum proposals to return. Defaults to config.", "minimum": 1, "maximum": 100},
            "include_examples": {"type": "boolean", "description": "Include redacted representative snippets. Defaults to false."},
            "persist": {"type": "boolean", "description": "Persist the report as a local JSON artifact. Defaults to config/default true."},
            "include_reconsolidation": {"type": "boolean", "description": "Include M10 reconsolidation candidates. Defaults to config/default false."},
            "reconsolidation_max_candidates": {"type": "integer", "description": "Maximum reconsolidation candidates to include.", "minimum": 1, "maximum": 100},
        },
        "additionalProperties": False,
    },
}

CONSOLIDATION_APPLY_SCHEMA = {
    "name": "qdrant_memory_consolidation_apply",
    "description": "Preview or apply one persisted consolidation proposal by report_id and proposal_id. Dry-run defaults to true; live actions require approve=true.",
    "parameters": {
        "type": "object",
        "properties": {
            "report_id": {"type": "string", "description": "Persisted report id returned by qdrant_memory_consolidate."},
            "proposal_id": {"type": "string", "description": "Proposal id to preview/apply."},
            "action": {"type": "string", "enum": ["merge", "delete", "quarantine", "promote_to_skill", "draft_review"], "description": "Expected action for the proposal type."},
            "dry_run": {"type": "boolean", "description": "When true, return the operation plan without mutating Qdrant or writing drafts. Defaults true."},
            "approve": {"type": "boolean", "description": "Required true when dry_run=false."},
            "quarantine_days": {"type": "integer", "description": "For action=quarantine, days to keep a reversible quarantine marker before later hard deletion.", "minimum": 1, "maximum": 365},
        },
        "required": ["report_id", "proposal_id"],
        "additionalProperties": False,
    },
}

TOOL_SCHEMAS = [
    STATUS_SCHEMA,
    STORE_SCHEMA,
    SEARCH_SCHEMA,
    INDEX_SCHEMA,
    FORGET_SCHEMA,
    INSPECT_SCHEMA,
    TRACE_SCHEMA,
    EXPAND_SCHEMA,
    SOURCE_STATUS_SCHEMA,
    LEARNING_STORE_SCHEMA,
    LEARNING_SEARCH_SCHEMA,
    LEARNING_PREVIEW_SCHEMA,
    LEARNING_APPROVE_SCHEMA,
    CONSOLIDATE_SCHEMA,
    CONSOLIDATION_APPLY_SCHEMA,
]
