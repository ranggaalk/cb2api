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
from typing import Any, Dict, List, Tuple

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
