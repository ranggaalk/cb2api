"""OpenAI response shaping for CodeBuddyAdapterV2 (spec §10).

Two output shapes:

  * non-streaming — a single ``chat.completion`` object with a ``message`` that
    may carry ``content`` and/or ``tool_calls``. Never emit a
    ``chat.completion.chunk`` here.
  * streaming — ``chat.completion.chunk`` frames, ending with a single
    ``data: [DONE]``.

``reasoning_content`` is surfaced as a distinct field, never merged into the
final answer content (spec §10).

Every builder is pure: it takes decoded values and returns dict/str output, so
these are trivially unit-testable without any upstream call.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional


def _tool_call_id_openai(codebuddy_id: str) -> str:
    """Normalize a CodeBuddy tool-call id to the OpenAI ``call_`` convention.

    CodeBuddy sometimes emits ``tooluse_<x>``; OpenAI clients expect ``call_``.
    A verbatim id (already ``call_`` / ``toolu_``) is returned unchanged so the
    tool_call/tool_result pairing is preserved.
    """
    if isinstance(codebuddy_id, str) and codebuddy_id.startswith("tooluse_"):
        return f"call_{codebuddy_id[len('tooluse_'):]}"
    return codebuddy_id


def build_non_stream_response(
    *,
    response_id: str,
    model: str,
    created: int,
    content: str,
    tool_calls: List[Dict[str, Any]],
    finish_reason: str,
    usage: Optional[Dict[str, Any]] = None,
    system_fingerprint: Optional[str] = None,
    reasoning_content: str = "",
) -> Dict[str, Any]:
    """Build a complete OpenAI ``chat.completion`` object.

    ``model`` is the model CodeBuddy reported (or the mapped model when upstream
    omitted it); the adapter never rewrites it to hide the upstream model.
    """
    message: Dict[str, Any] = {"role": "assistant", "content": content or ""}
    if reasoning_content:
        # Surface reasoning separately; do NOT fold it into content.
        message["reasoning_content"] = reasoning_content
    if tool_calls:
        message["tool_calls"] = [
            {
                "id": _tool_call_id_openai(tc.get("id", "")),
                "type": "function",
                "function": {
                    "name": tc.get("function", {}).get("name", ""),
                    "arguments": tc.get("function", {}).get("arguments", "{}"),
                },
            }
            for tc in tool_calls
        ]

    response: Dict[str, Any] = {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }
    if usage:
        response["usage"] = usage
    if system_fingerprint:
        response["system_fingerprint"] = system_fingerprint
    return response


def sse_chunk(obj: Dict[str, Any]) -> str:
    """Serialize a chunk object as an SSE ``data:`` frame."""
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """The single terminating SSE frame."""
    return "data: [DONE]\n\n"


def content_chunk(
    *, response_id: str, model: str, created: int, content: str
) -> Dict[str, Any]:
    """Build a streaming ``chat.completion.chunk`` carrying a content delta."""
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [
            {"index": 0, "delta": {"content": content}, "finish_reason": None}
        ],
    }


def final_chunk(
    *, response_id: str, model: str, created: int, finish_reason: str
) -> Dict[str, Any]:
    """Build the terminal streaming chunk carrying only ``finish_reason``."""
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
    }


def moderation_chunk(
    *, response_id: str, created: int, message: str
) -> Dict[str, Any]:
    """Build a content_filter streaming chunk for a moderation refusal."""
    return {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": created,
        "choices": [
            {
                "index": 0,
                "delta": {"content": message},
                "finish_reason": "content_filter",
            }
        ],
    }
