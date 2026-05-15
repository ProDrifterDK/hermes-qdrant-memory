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
            "include_metadata": {"type": "boolean", "description": "Include full payload metadata. Defaults to false."},
        },
        "required": ["query"],
        "additionalProperties": False,
    },
}

INDEX_SCHEMA = {
    "name": "qdrant_memory_index",
    "description": "Safely index markdown/text files or vault folders into local Qdrant memory. Dry-run defaults to true.",
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

TOOL_SCHEMAS = [STATUS_SCHEMA, STORE_SCHEMA, SEARCH_SCHEMA, INDEX_SCHEMA, FORGET_SCHEMA]
