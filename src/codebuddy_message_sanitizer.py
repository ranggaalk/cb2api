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


def _log_structural(level: int, prefix: str, index: int, msg: Any) -> None:
    """Log safe structural metadata for a message.

    Logs the message index, role, its field-name set, the content field's type,
    and whether it carries tool_calls / tool_call_id. It never logs message
    content values or any credential: the field names logged (role, content,
    tool_calls, ...) are fixed OpenAI-schema identifiers, not user data.
    """
    if isinstance(msg, dict):
        role = msg.get("role")
        field_names = ",".join(sorted(str(k) for k in msg.keys()))
        content_type = type(msg.get("content")).__name__
        has_tool_calls = bool(msg.get("tool_calls"))
        has_tool_call_id = bool(msg.get("tool_call_id"))
    else:
        role = None
        field_names = ""
        content_type = type(msg).__name__
        has_tool_calls = False
        has_tool_call_id = False

    logger.log(
        level,
        "%s index=%d role=%s keys=[%s] content_type=%s has_tool_calls=%s "
        "has_tool_call_id=%s",
        prefix,
        index,
        role,
        field_names,
        content_type,
        has_tool_calls,
        has_tool_call_id,
    )


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
