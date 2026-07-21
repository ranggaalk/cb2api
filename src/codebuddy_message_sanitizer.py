"""
Message sanitization and moderation detection for CodeBuddy2API.

Two concerns live here, both aimed at reducing CodeBuddy false-positive
moderation without weakening legitimate upstream moderation:

1. Agent system-prompt sanitization. Long coding-agent identity blocks
   (Claude Code, Cursor, Cline, aider, ...) trigger CodeBuddy's content
   filter far more often than a plain request. We replace ONLY the system
   message with a short, neutral instruction. User/assistant/tool messages
   are never touched.

2. Mandarin moderation detection. CodeBuddy returns a Mandarin refusal when
   it blocks a request. We detect specific markers so the router can surface
   an OpenAI-compatible ``content_filter`` error instead of leaking the
   refusal as an assistant reply.
"""
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Neutral replacement used when an agent system prompt is sanitized.
NEUTRAL_SYSTEM_PROMPT = (
    "You are a helpful coding assistant. Answer accurately, follow the "
    "user's request, and reply in the same language as the user."
)

# Lower-cased substrings that identify a coding-agent system prompt. These are
# matched case-insensitively against system-message text only.
_AGENT_PROMPT_MARKERS: Tuple[str, ...] = (
    "you are claude code",
    "claude code",
    "official cli",
    "cursor",
    "windsurf",
    "cline",
    "aider",
    "continue",
    "copilot",
    "coding agent",
    "code agent",
    "agentic coding assistant",
    "cc_entrypoint",
    "<agent-identity>",
    "<behavior_instructions>",
    "<tool_instructions>",
)

# Specific markers found in CodeBuddy's Mandarin moderation refusal. Matching
# any one of these flags a moderation response. We deliberately do NOT treat
# arbitrary Mandarin text as moderation.
_MODERATION_MARKERS: Tuple[str, ...] = (
    "敏感内容",
    "系统检测到",
    "无法响应您的请求",
    "请检查后重新输入",
)


def _extract_text(content: Any) -> str:
    """Flatten string or structured content into plain text for matching."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    # Include other structured values so markers embedded in
                    # nested dicts are still detected.
                    parts.append(str(item))
            elif isinstance(item, str):
                parts.append(item)
            else:
                parts.append(str(item))
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def _looks_like_agent_prompt(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _AGENT_PROMPT_MARKERS)


# Roles CodeBuddy accepts on an upstream message.
_VALID_UPSTREAM_ROLES: Tuple[str, ...] = ("system", "user", "assistant", "tool")


class MessageNormalizationError(Exception):
    """Raised when an upstream message cannot be given a valid role.

    Carries the offending message index so the router can return a clear local
    HTTP 400 identifying which message is malformed, instead of forwarding data
    that CodeBuddy rejects with a generic "Message N must have 'role' and
    'content' fields".
    """

    def __init__(self, index: int, reason: str):
        super().__init__(f"message {index}: {reason}")
        self.index = index
        self.reason = reason


def _content_block_types(content: Any) -> List[str]:
    """Return the ``type`` of each block in an array content, else empty list."""
    if not isinstance(content, list):
        return []
    types: List[str] = []
    for block in content:
        if isinstance(block, dict):
            types.append(str(block.get("type", "unknown")))
        else:
            types.append(type(block).__name__)
    return types


def _collect_tool_ids(msg: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return ``(tool_call_ids, tool_result_ids)`` for a message.

    Handles both OpenAI shape (``tool_calls[*].id`` / ``tool_call_id``) and
    Anthropic array shape (``tool_use`` / ``tool_result`` blocks). IDs are
    opaque routing tokens (e.g. ``toolu_...`` / ``call_...``), never secrets or
    content, so logging them is safe and is required for diagnosing lost tool
    results.
    """
    tool_call_ids: List[str] = []
    tool_result_ids: List[str] = []

    tool_calls = msg.get("tool_calls")
    if isinstance(tool_calls, list):
        for tc in tool_calls:
            if isinstance(tc, dict) and tc.get("id"):
                tool_call_ids.append(str(tc.get("id")))

    if msg.get("tool_call_id"):
        tool_result_ids.append(str(msg.get("tool_call_id")))

    content = msg.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use" and block.get("id"):
                tool_call_ids.append(str(block.get("id")))
            elif block.get("type") == "tool_result" and block.get("tool_use_id"):
                tool_result_ids.append(str(block.get("tool_use_id")))

    return tool_call_ids, tool_result_ids


def _log_structural(level: int, prefix: str, index: int, msg: Any) -> None:
    """Log safe structural metadata for a single message.

    Emits, per requirement, the fields needed to diagnose lost/malformed
    tool-use conversion: index, role, content_type, content_block_types,
    has_tool_calls, tool_call_ids, tool_result_ids, content_length. It never
    logs message content values, prompts, or credentials: block *types* and
    opaque tool IDs are structural routing metadata, not user data. content_length
    is a character count only.
    """
    if isinstance(msg, dict):
        role = msg.get("role")
        content = msg.get("content")
        content_type = type(content).__name__
        block_types = _content_block_types(content)
        has_tool_calls = bool(msg.get("tool_calls"))
        tool_call_ids, tool_result_ids = _collect_tool_ids(msg)
        content_length = len(_extract_text(content))
    else:
        role = None
        content_type = type(msg).__name__
        block_types = []
        has_tool_calls = False
        tool_call_ids, tool_result_ids = [], []
        content_length = 0

    logger.log(
        level,
        "%s index=%d role=%s content_type=%s content_block_types=[%s] "
        "has_tool_calls=%s tool_call_ids=[%s] tool_result_ids=[%s] content_length=%d",
        prefix,
        index,
        role,
        content_type,
        ",".join(block_types),
        has_tool_calls,
        ",".join(tool_call_ids),
        ",".join(tool_result_ids),
        content_length,
    )


def log_messages_structural(prefix: str, messages: List[Dict[str, Any]], level: int = logging.DEBUG) -> None:
    """Log structural diagnostics for an entire message list (content-free)."""
    for index, msg in enumerate(messages):
        _log_structural(level, prefix, index, msg)


def _infer_role(msg: Dict[str, Any]) -> Any:
    """Infer a message role from structure when it is missing or blank.

    Returns a valid role string, or ``None`` when it genuinely cannot be
    determined.
    """
    role = msg.get("role")
    if isinstance(role, str) and role.strip():
        return role.strip()
    # A message carrying a tool_call_id is a tool result.
    if msg.get("tool_call_id"):
        return "tool"
    # A message carrying tool_calls is an assistant turn.
    if msg.get("tool_calls"):
        return "assistant"
    return None


def normalize_messages_for_upstream(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return a copy of ``messages`` guaranteed valid for the CodeBuddy upstream.

    This is the final step before the upstream request, run after any
    Anthropic/OpenAI/tool conversion and system-prompt sanitization, so it
    inspects the exact messages CodeBuddy will receive rather than the original
    client messages. It guarantees every message has:

      * a valid, non-empty ``role``
      * a ``content`` field that is never missing or ``null``

    Rules:
      * A message with an explicit non-empty ``role`` keeps it verbatim; a role
        that is present is considered determined even if it is not one of the
        common OpenAI roles, so legitimate conversations are never rejected on a
        role-value technicality.
      * Assistant messages carrying ``tool_calls`` keep ``content: ""`` when
        content is missing or null, so an assistant tool-call turn is never sent
        without a content field.
      * Tool-result messages resolve to role ``tool`` (inferred from
        ``tool_call_id`` when the role is absent) and keep their
        ``tool_call_id`` and ``content``.
      * Existing non-null content is preserved verbatim, including multimodal
        content arrays CodeBuddy already accepts; an empty string is only added
        when content is missing or ``null``.
      * A message whose role genuinely cannot be determined (absent/blank role
        with no ``tool_call_id`` or ``tool_calls`` to infer from) raises
        :class:`MessageNormalizationError` with its index, so the caller can
        return a local HTTP 400 instead of forwarding malformed data.
    """
    normalized: List[Dict[str, Any]] = []
    repaired = 0

    for index, msg in enumerate(messages):
        # Structural, content-free trace of every message inspected upstream.
        _log_structural(logging.DEBUG, "upstream message", index, msg)

        if not isinstance(msg, dict):
            _log_structural(
                logging.WARNING, "upstream message not an object", index, msg
            )
            raise MessageNormalizationError(index, "message is not an object")

        role = _infer_role(msg)
        if not role:
            _log_structural(
                logging.WARNING, "upstream message role undeterminable", index, msg
            )
            raise MessageNormalizationError(
                index,
                "role is missing and cannot be inferred from message structure",
            )

        new_msg = dict(msg)
        new_msg["role"] = role

        # Add an empty content string only when the field is missing or null;
        # never overwrite existing content (strings, empty strings, or
        # multimodal arrays CodeBuddy already accepts).
        if "content" not in new_msg or new_msg["content"] is None:
            new_msg["content"] = ""
            repaired += 1
            _log_structural(
                logging.INFO, "upstream message content defaulted to empty", index, msg
            )

        normalized.append(new_msg)

    if repaired:
        logger.info(
            "Normalized upstream messages: total=%d content_defaulted=%d",
            len(messages),
            repaired,
        )

    return normalized


# Anthropic content-block types that must be converted away before upstream.
_ANTHROPIC_TOOL_BLOCK_TYPES: Tuple[str, ...] = ("tool_use", "tool_result")


def _flatten_tool_result_content(content: Any) -> str:
    """Flatten an Anthropic ``tool_result.content`` into a complete text string.

    ``tool_result.content`` may be a plain string or a list of blocks
    (``text``/``image``/...). Every part is preserved so a tool result (e.g. the
    full text of a file read) is never truncated or dropped. Non-text blocks are
    serialized to JSON so their information survives rather than being discarded.
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
    """Concatenate the text of the ``text`` blocks in an assistant content array."""
    parts: List[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


def _has_block_type(content: Any, block_type: str) -> bool:
    if not isinstance(content, list):
        return False
    return any(
        isinstance(b, dict) and b.get("type") == block_type for b in content
    )


def convert_anthropic_messages_to_openai(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convert Anthropic block-array messages into OpenAI-shaped messages.

    Claude Code sends tool interactions as Anthropic content blocks:

      * an assistant turn whose ``content`` array contains ``tool_use`` blocks
      * a following user turn whose ``content`` array contains ``tool_result``
        blocks carrying the tool output (e.g. the full text of a file read)

    CodeBuddy speaks the OpenAI schema, so without conversion these blocks are
    forwarded verbatim and the tool output never reaches the model — the
    reported "file contents are not in context" failure. This function rewrites
    them into the OpenAI shape while preserving every relationship:

      * assistant ``tool_use`` -> ``role: assistant`` message with ``content``
        (the text blocks, or ``""``) and a ``tool_calls`` list. Each tool call
        keeps the Anthropic ``tool_use.id`` verbatim and encodes ``input`` as a
        JSON-string ``function.arguments``.
      * each user ``tool_result`` -> its own ``role: tool`` message whose
        ``tool_call_id`` is the verbatim ``tool_use_id`` and whose ``content`` is
        the complete flattened tool output. Tool results become standalone
        messages and are never merged into an unrelated user message.
      * messages whose content is a plain string, or an array with no tool
        blocks (plain text or multimodal image arrays), are passed through
        unchanged — array content is never replaced with an empty string.

    The verbatim ID reuse guarantees ``tool_use.id`` == the emitted
    ``tool_call.id`` == the matching tool message's ``tool_call_id``, so the
    OpenAI tool-call/tool-result pairing mirrors the Anthropic one exactly. Each
    block is emitted exactly once, so no tool call or result is duplicated.
    """
    converted: List[Dict[str, Any]] = []

    for msg in messages:
        if not isinstance(msg, dict):
            converted.append(msg)
            continue

        role = msg.get("role")
        content = msg.get("content")

        # Non-array content (string / None) and arrays without tool blocks are
        # preserved verbatim. This keeps plain text, and multimodal image
        # arrays, exactly as the client sent them.
        if not isinstance(content, list):
            converted.append(msg)
            continue

        has_tool_use = _has_block_type(content, "tool_use")
        has_tool_result = _has_block_type(content, "tool_result")

        if not has_tool_use and not has_tool_result:
            # Plain text or multimodal (image) array: preserve as-is.
            converted.append(msg)
            continue

        if has_tool_use:
            # Assistant turn issuing one or more tool calls.
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
                            # OpenAI arguments must be a JSON *string*.
                            "arguments": json.dumps(
                                block.get("input", {}) or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            new_msg = dict(msg)
            new_msg["role"] = role or "assistant"
            # Assistant tool-call turns carry text content when present, else "".
            new_msg["content"] = _assistant_text_from_blocks(content)
            new_msg["tool_calls"] = tool_calls
            converted.append(new_msg)
            continue

        # has_tool_result: split the array so each tool_result becomes its own
        # tool message; any remaining non-tool blocks become a trailing user
        # message so tool output is never merged into unrelated user text.
        leftover_blocks: List[Any] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                converted.append(
                    {
                        "role": "tool",
                        "tool_call_id": block.get("tool_use_id", ""),
                        "content": _flatten_tool_result_content(block.get("content")),
                    }
                )
            else:
                leftover_blocks.append(block)

        if leftover_blocks:
            # Preserve any accompanying user content (text/image) as a separate
            # message, keeping array shape so multimodal content survives.
            converted.append({"role": role or "user", "content": leftover_blocks})

    return converted


def validate_upstream_messages(messages: List[Dict[str, Any]]) -> None:
    """Validate the final OpenAI-shaped messages just before the upstream call.

    Raises :class:`MessageNormalizationError` (carrying the offending message
    index and a structural description) when any of these hold, so the router
    can return a local HTTP 400 instead of forwarding a malformed payload:

      * a message is missing ``role`` or ``content``
      * a ``tool`` message's ``tool_call_id`` has no preceding matching
        ``tool_calls`` id (an orphaned tool result)
      * a ``tool_calls`` entry has ``function.arguments`` that is not a valid
        JSON string
      * any unconverted Anthropic ``tool_use`` / ``tool_result`` block remains in
        a message's content array
    """
    seen_tool_call_ids: set = set()

    for index, msg in enumerate(messages):
        _log_structural(logging.DEBUG, "validate upstream message", index, msg)

        if not isinstance(msg, dict):
            raise MessageNormalizationError(index, "message is not an object")

        role = msg.get("role")
        if not isinstance(role, str) or not role.strip():
            raise MessageNormalizationError(index, "message is missing a valid role")
        if "content" not in msg:
            raise MessageNormalizationError(index, "message is missing content")

        # No unconverted Anthropic tool blocks may remain in content arrays.
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in _ANTHROPIC_TOOL_BLOCK_TYPES:
                    raise MessageNormalizationError(
                        index,
                        f"unconverted Anthropic '{block.get('type')}' block remains "
                        f"in content",
                    )

        # Register tool_call ids and validate their arguments are JSON strings.
        tool_calls = msg.get("tool_calls")
        if tool_calls is not None:
            if not isinstance(tool_calls, list):
                raise MessageNormalizationError(index, "tool_calls must be a list")
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    raise MessageNormalizationError(index, "tool_call must be an object")
                tc_id = tc.get("id")
                if tc_id:
                    seen_tool_call_ids.add(str(tc_id))
                func = tc.get("function", {})
                arguments = func.get("arguments") if isinstance(func, dict) else None
                if not isinstance(arguments, str):
                    raise MessageNormalizationError(
                        index, "tool_call function.arguments must be a JSON string"
                    )
                try:
                    json.loads(arguments)
                except (ValueError, TypeError):
                    raise MessageNormalizationError(
                        index, "tool_call function.arguments is not valid JSON"
                    )

        # A tool result must reference a tool_call id already seen upstream.
        if role == "tool":
            tool_call_id = msg.get("tool_call_id")
            if not tool_call_id:
                raise MessageNormalizationError(
                    index, "tool message is missing tool_call_id"
                )
            if str(tool_call_id) not in seen_tool_call_ids:
                raise MessageNormalizationError(
                    index,
                    "tool result references tool_call_id with no preceding "
                    "matching tool call",
                )


def sanitize_messages(
    messages: List[Dict[str, Any]],
    *,
    enabled: bool = True,
    max_system_prompt_length: int = 2000,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Return a sanitized copy of ``messages`` and whether any system message was
    replaced.

    Only ``system`` messages are considered. A system message is replaced with
    :data:`NEUTRAL_SYSTEM_PROMPT` when either:

      * its text matches a coding-agent marker, or
      * its text length exceeds ``max_system_prompt_length``.

    User, assistant, and tool messages are copied through unchanged. This never
    attempts to disguise sensitive content; it only removes agent identity
    boilerplate that causes false-positive moderation.
    """
    if not enabled:
        return messages, False

    sanitized: List[Dict[str, Any]] = []
    changed = False

    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "system":
            sanitized.append(msg)
            continue

        text = _extract_text(msg.get("content"))
        too_long = len(text) > max_system_prompt_length
        is_agent = _looks_like_agent_prompt(text)

        if too_long or is_agent:
            new_msg = dict(msg)
            new_msg["content"] = NEUTRAL_SYSTEM_PROMPT
            sanitized.append(new_msg)
            changed = True
            # Log only metadata, never the prompt contents.
            logger.info(
                "System prompt sanitized: agent_marker=%s over_length=%s "
                "original_length=%d",
                is_agent,
                too_long,
                len(text),
            )
        else:
            sanitized.append(msg)

    return sanitized, changed


def is_codebuddy_moderation_response(text: Any) -> bool:
    """
    Return True when ``text`` contains a CodeBuddy Mandarin moderation marker.

    Accepts a string or any value coercible to text (e.g. nested JSON already
    serialized). Only the specific refusal markers count; ordinary Mandarin
    responses are not flagged.
    """
    if text is None:
        return False
    haystack = text if isinstance(text, str) else str(text)
    if not haystack:
        return False
    return any(marker in haystack for marker in _MODERATION_MARKERS)
