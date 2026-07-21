"""Deterministic message normalizer for CodeBuddyAdapterV2 (spec §4, §5, §6).

Claude Code and 9Router send messages in a mix of two shapes:

  * OpenAI shape — ``content`` is a string (or a multimodal array), assistant
    tool calls live in ``tool_calls``, and tool results are ``role: "tool"``
    messages keyed by ``tool_call_id``.
  * Anthropic shape — ``content`` is an array of typed blocks: ``text``,
    ``image``, ``tool_use`` (an assistant issuing a call), and ``tool_result``
    (a user turn carrying the tool output).

CodeBuddy speaks the OpenAI schema. This module rewrites Anthropic tool blocks
into OpenAI messages while preserving every relationship, and guarantees that
every final message has both ``role`` and ``content``. The cardinal rule
(spec §5) is that array content is NEVER replaced with an empty string — doing
so is what silently dropped ``tool_result`` file contents in the legacy path.

The transformation is a pure function of its input: no I/O, no globals, fully
unit-testable.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Tuple

from .errors import MessageNormalizationError

logger = logging.getLogger(__name__)

# Anthropic block types that must be converted away before upstream.
_ANTHROPIC_TOOL_BLOCK_TYPES: Tuple[str, ...] = ("tool_use", "tool_result")


def _text_from_content(content: Any) -> str:
    """Flatten any content shape into plain text (for length metrics / matching).

    Never used to REPLACE content — only to measure or to concatenate assistant
    text blocks. Non-text blocks are serialized so their information is counted
    rather than silently dropped.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _assistant_text_from_blocks(blocks: List[Any]) -> str:
    """Concatenate only the ``text`` blocks of an assistant content array."""
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _flatten_tool_result_content(content: Any) -> str:
    """Flatten an Anthropic ``tool_result.content`` into a complete string.

    The content may be a plain string or a list of blocks (``text``/``image``/
    ...). Every part is preserved so a tool result — e.g. the full text of a
    file read — is never truncated or dropped. Non-text blocks are serialized to
    JSON so their information survives.
    """
    if isinstance(content, str):
        return content
    if content is None:
        return ""
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(json.dumps(item, ensure_ascii=False))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _has_block_type(content: Any, block_type: str) -> bool:
    if not isinstance(content, list):
        return False
    return any(isinstance(b, dict) and b.get("type") == block_type for b in content)


def convert_anthropic_messages_to_openai(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Rewrite Anthropic block-array messages into OpenAI-shaped messages.

    Rules (spec §6):

      * An assistant turn whose content array contains ``tool_use`` blocks
        becomes a ``role: assistant`` message with ``content`` = the text blocks
        (or ``""``) and a ``tool_calls`` list. Each tool call keeps the
        ``tool_use.id`` verbatim and encodes ``input`` as a JSON-string
        ``function.arguments``. Multiple ``tool_use`` blocks → multiple entries.
      * A user turn whose content array contains ``tool_result`` blocks is split
        so each ``tool_result`` becomes its own ``role: tool`` message whose
        ``tool_call_id`` is the verbatim ``tool_use_id`` and whose ``content`` is
        the complete flattened tool output. Multiple results → multiple
        messages, each standalone — never merged into unrelated user text.
      * When a user content array mixes ``text``/``image`` with ``tool_result``,
        the order is preserved by emitting each contiguous run of non-tool blocks
        as its own user message positioned exactly where it appeared.
      * Messages whose content is a plain string, or an array with no tool
        blocks (plain text or multimodal image arrays), are passed through
        unchanged — array content is never replaced with an empty string.

    The verbatim ID reuse guarantees ``tool_use.id`` == emitted
    ``tool_call.id`` == matching tool message's ``tool_call_id``. Each block is
    emitted exactly once, so no tool call or result is duplicated.
    """
    converted: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            converted.append(msg)
            continue

        role = msg.get("role")
        content = msg.get("content")

        # Non-array content (string / None) passes through untouched.
        if not isinstance(content, list):
            converted.append(msg)
            continue

        has_tool_use = _has_block_type(content, "tool_use")
        has_tool_result = _has_block_type(content, "tool_result")

        # Plain text or multimodal (image) array with no tool blocks: preserve.
        if not has_tool_use and not has_tool_result:
            converted.append(msg)
            continue

        if has_tool_use:
            # Assistant turn issuing one or more tool calls. Preserve any
            # accompanying assistant text as the message content.
            tool_calls: List[Dict[str, Any]] = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "tool_use":
                    continue
                tool_calls.append(
                    {
                        "id": block.get("id", ""),
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            # OpenAI requires arguments to be a JSON *string*.
                            "arguments": json.dumps(
                                block.get("input", {}) or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            new_msg: Dict[str, Any] = {
                "role": role or "assistant",
                "content": _assistant_text_from_blocks(content),
                "tool_calls": tool_calls,
            }
            converted.append(new_msg)
            continue

        # has_tool_result: walk the array in order. Each ``tool_result`` becomes
        # its own tool message; contiguous non-tool blocks (text/image) are
        # flushed as a separate user message in their original position so
        # ordering and multimodal content are preserved and nothing is merged
        # into an unrelated turn.
        pending_blocks: List[Any] = []

        def _flush_pending() -> None:
            if pending_blocks:
                converted.append({"role": role or "user", "content": list(pending_blocks)})
                pending_blocks.clear()

        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                _flush_pending()
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _flatten_tool_result_content(block.get("content")),
                    }
                )
            else:
                pending_blocks.append(block)
        _flush_pending()

    return converted


def normalize_messages_for_upstream(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Guarantee every message has a valid ``role`` and a ``content`` field.

    Run as the final shaping step before validation. Rules (spec §4, §5):

      * A message with an explicit non-empty ``role`` keeps it. A role that is
        present but nonstandard is accepted (never rejected on a technicality).
      * A missing role is inferred: ``tool_call_id`` → ``tool``; ``tool_calls``
        → ``assistant``. If still undeterminable, raise
        :class:`MessageNormalizationError` with the index (local HTTP 400).
      * ``content`` is added as ``""`` ONLY when missing or ``null``. Existing
        content — including empty strings and multimodal arrays — is preserved
        verbatim. Assistant tool-call turns therefore get ``content: ""`` only
        when they truly lack content.
    """
    normalized: List[Dict[str, Any]] = []
    repaired = 0

    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise MessageNormalizationError(index, "message is not an object")

        role = msg.get("role")
        if not (isinstance(role, str) and role.strip()):
            if msg.get("tool_call_id"):
                role = "tool"
            elif msg.get("tool_calls"):
                role = "assistant"
            else:
                raise MessageNormalizationError(
                    index,
                    "role is missing and cannot be inferred from message structure",
                )
        else:
            role = role.strip()

        new_msg = dict(msg)
        new_msg["role"] = role

        if "content" not in new_msg or new_msg["content"] is None:
            new_msg["content"] = ""
            repaired += 1

        normalized.append(new_msg)

    if repaired:
        logger.info(
            "Normalized upstream messages: total=%d content_defaulted=%d",
            len(messages),
            repaired,
        )

    return normalized


def normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Full normalization pipeline: Anthropic conversion then role/content fill.

    Returns OpenAI-shaped messages ready for validation and the upstream
    payload. Sanitization of system prompts is applied separately by the
    orchestrator so this stays a pure structural transform.
    """
    converted = convert_anthropic_messages_to_openai(messages)
    return normalize_messages_for_upstream(converted)
