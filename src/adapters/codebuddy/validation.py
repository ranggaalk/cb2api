"""Pre-upstream conversation validation for CodeBuddyAdapterV2 (spec §7).

After normalization, and immediately before building the upstream payload, the
entire conversation is validated. A malformed tool conversation is rejected
LOCALLY with HTTP 400 (:class:`InvalidToolConversationError` carrying the
offending message index) instead of being forwarded to CodeBuddy, which would
otherwise fail with an opaque "Message N must have 'role' and 'content'" error.

Rules enforced (spec §7):
  1. Every tool result references a preceding assistant tool call with the same
     ID.
  2. Every assistant tool call carries ``content`` (at least ``""``).
  3. Every tool result has ``role == "tool"``.
  4. Every tool result has a ``tool_call_id``.
  5. Tool call IDs are unchanged (validated structurally — the normalizer copies
     them verbatim).
  6. ``function.arguments`` is a valid JSON string.
  7. No unsupported Anthropic blocks remain in the final OpenAI payload.

This is a pure function: no I/O, no globals.
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from .errors import InvalidToolConversationError, MessageNormalizationError

logger = logging.getLogger(__name__)

_ANTHROPIC_TOOL_BLOCK_TYPES = ("tool_use", "tool_result")


def validate_conversation(messages: List[Dict[str, Any]]) -> None:
    """Validate the final OpenAI-shaped conversation just before upstream.

    Raises :class:`MessageNormalizationError` for basic role/content problems and
    :class:`InvalidToolConversationError` for tool-structure problems. Both carry
    the offending message index so the router can return a precise local 400.
    """
    seen_tool_call_ids: set = set()

    for index, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise MessageNormalizationError(index, "message is not an object")

        role = msg.get("role")
        if not (isinstance(role, str) and role.strip()):
            raise MessageNormalizationError(index, "message is missing a valid role")

        # Rule 2: content must be present (assistant tool-call turns included).
        if "content" not in msg:
            raise MessageNormalizationError(index, "message is missing content")

        # Rule 7: no unconverted Anthropic blocks may remain in content arrays.
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and block.get("type") in _ANTHROPIC_TOOL_BLOCK_TYPES
                ):
                    raise InvalidToolConversationError(
                        index,
                        f"unconverted Anthropic '{block.get('type')}' block remains "
                        f"in content",
                    )

        # Rules 5 & 6: register tool-call IDs and validate their JSON arguments.
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise InvalidToolConversationError(index, "tool_calls must be a list")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise InvalidToolConversationError(
                        index, "tool_call must be an object"
                    )
                tc_id = tc.get("id")
                if tc_id:
                    seen_tool_call_ids.add(str(tc_id))
                func = tc.get("function", {})
                arguments = func.get("arguments") if isinstance(func, dict) else None
                if not isinstance(arguments, str):
                    raise InvalidToolConversationError(
                        index, "tool_call function.arguments must be a JSON string"
                    )
                try:
                    json.loads(arguments)
                except (ValueError, TypeError):
                    raise InvalidToolConversationError(
                        index, "tool_call function.arguments is not valid JSON"
                    )

        # Rules 1, 3 & 4: a tool result must reference a preceding tool call.
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                raise InvalidToolConversationError(
                    index, "tool message is missing tool_call_id"
                )
            if str(tool_call_id) not in seen_tool_call_ids:
                raise InvalidToolConversationError(
                    index,
                    "tool result references tool_call_id with no preceding "
                    "matching tool call",
                )
