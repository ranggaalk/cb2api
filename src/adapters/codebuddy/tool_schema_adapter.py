"""Tool JSON-Schema sanitization for CodeBuddyAdapterV2 (spec §8).

Claude Code sends tool definitions whose ``function.parameters`` are JSON Schema
documents that may use ``$ref``/``$defs``/``definitions`` and a range of
keywords. This module produces a deterministic, upstream-safe copy of each tool
schema:

  * local ``$ref`` (``#/$defs/...`` / ``#/definitions/...``) are resolved inline
    so the upstream never has to follow references it may not support;
  * supported structural keywords are preserved exactly (``type``,
    ``properties``, ``required``, ``items``, ``enum``, ``additionalProperties``,
    ``anyOf``/``oneOf``/``allOf``);
  * tool names and parameter names are preserved verbatim — nothing is renamed,
    no required field is dropped, no nested object is flattened;
  * unknown/unsupported keywords are dropped from the copy rather than forwarded
    raw.

Everything here is pure: it takes a tools list and returns a new one, so it is
fully unit-testable and never mutates the caller's data.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# JSON Schema keywords preserved verbatim (structure-bearing).
_SUPPORTED_SCHEMA_KEYS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "const",
    "additionalProperties",
    "anyOf",
    "oneOf",
    "allOf",
    "not",
    "description",
    "title",
    "default",
    "format",
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "minLength",
    "maxLength",
    "pattern",
    "minItems",
    "maxItems",
    "uniqueItems",
    "minProperties",
    "maxProperties",
    "nullable",
}

# Maximum depth guard so a pathological or cyclic schema cannot recurse forever.
_MAX_DEPTH = 64


def _resolve_ref(ref: str, defs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Resolve a local ``#/$defs/Name`` or ``#/definitions/Name`` reference.

    Returns the referenced schema dict, or ``None`` for non-local / unknown refs
    (which are then simply dropped, leaving a permissive empty schema).
    """
    if not isinstance(ref, str) or not ref.startswith("#/"):
        return None
    parts = ref[2:].split("/")
    if len(parts) != 2 or parts[0] not in ("$defs", "definitions"):
        return None
    target = defs.get(parts[1])
    return target if isinstance(target, dict) else None


def _sanitize_schema(
    schema: Any,
    defs: Dict[str, Any],
    depth: int = 0,
    seen_refs: Optional[frozenset] = None,
) -> Any:
    """Return an upstream-safe deep copy of a JSON Schema node.

    ``defs`` is the pooled ``$defs``/``definitions`` map from the root schema so
    references can be resolved at any depth. ``seen_refs`` breaks reference
    cycles: a ``$ref`` already being expanded on the current path resolves to a
    permissive ``{}`` instead of recursing forever.
    """
    if seen_refs is None:
        seen_refs = frozenset()

    if depth > _MAX_DEPTH:
        return {}

    if isinstance(schema, list):
        return [_sanitize_schema(item, defs, depth + 1, seen_refs) for item in schema]

    if not isinstance(schema, dict):
        # Primitive (already-resolved default/enum value etc.) — copy verbatim.
        return schema

    # Resolve a local $ref by inlining the referenced schema.
    ref = schema.get("$ref")
    if isinstance(ref, str):
        if ref in seen_refs:
            # Cycle: stop expanding and emit a permissive schema.
            return {}
        resolved = _resolve_ref(ref, defs)
        if resolved is not None:
            return _sanitize_schema(resolved, defs, depth + 1, seen_refs | {ref})
        # Unknown/non-local ref: drop it, leaving a permissive schema.
        return {}

    out: Dict[str, Any] = {}
    for key, value in schema.items():
        if key in ("$defs", "definitions", "$ref", "$schema", "$id"):
            # Definitions are inlined at reference sites; meta-keywords dropped.
            continue
        if key not in _SUPPORTED_SCHEMA_KEYS:
            # Unsupported keyword: omit from the upstream copy.
            continue

        if key == "properties" and isinstance(value, dict):
            out["properties"] = {
                prop_name: _sanitize_schema(prop_schema, defs, depth + 1, seen_refs)
                for prop_name, prop_schema in value.items()
            }
        elif key in ("items", "not") and isinstance(value, (dict, list)):
            out[key] = _sanitize_schema(value, defs, depth + 1, seen_refs)
        elif key in ("anyOf", "oneOf", "allOf") and isinstance(value, list):
            out[key] = [
                _sanitize_schema(sub, defs, depth + 1, seen_refs) for sub in value
            ]
        elif key == "additionalProperties" and isinstance(value, dict):
            out[key] = _sanitize_schema(value, defs, depth + 1, seen_refs)
        elif key == "required" and isinstance(value, list):
            # Preserve required field names verbatim — never drop them.
            out["required"] = list(value)
        else:
            out[key] = value

    return out


def sanitize_tool(tool: Dict[str, Any]) -> Dict[str, Any]:
    """Return an upstream-safe copy of a single OpenAI tool definition.

    The tool ``name`` is preserved verbatim. ``function.parameters`` is
    sanitized (refs resolved, unsupported keywords dropped). A tool without a
    parameters schema is returned structurally unchanged.
    """
    if not isinstance(tool, dict):
        return tool

    func = tool.get("function")
    if not isinstance(func, dict):
        return tool

    params = func.get("parameters")
    if not isinstance(params, dict):
        return tool

    # Pool both $defs and definitions so references anywhere resolve.
    defs: Dict[str, Any] = {}
    for defs_key in ("$defs", "definitions"):
        block = params.get(defs_key)
        if isinstance(block, dict):
            defs.update(block)

    sanitized_params = _sanitize_schema(params, defs)

    new_func = dict(func)
    new_func["parameters"] = sanitized_params
    new_tool = dict(tool)
    new_tool["function"] = new_func
    return new_tool


def sanitize_tools(tools: Any) -> Any:
    """Return an upstream-safe copy of a tools list.

    Non-list input is returned unchanged so callers can forward it as-is. Each
    tool is sanitized independently; tool ordering and names are preserved.
    """
    if not isinstance(tools, list):
        return tools
    return [sanitize_tool(tool) for tool in tools]
