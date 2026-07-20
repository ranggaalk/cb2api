"""
Keyword replacement utility with centralized replacement rules.
Prevents CodeBuddy from detecting competitor-related keywords.
"""
import logging

logger = logging.getLogger(__name__)


def apply_keyword_replacement(text: str) -> str:
    """
    Apply the standard keyword replacements.

    Args:
        text: Text content to process.

    Returns:
        str: The processed text.
    """
    if not isinstance(text, str):
        return text

    # Define replacement rules.
    replacements = {
        "Claude Code": "CodeBuddy Code",
        "Anthropic's official CLI for Claude": "Tencent's official CLI for CodeBuddy",
        "Claude": "CodeBuddy",
        "Anthropic": "Tencent",
        "https://github.com/anthropics/claude-code/issues": "https://cnb.cool/codebuddy/codebuddy-code/-/issues"
    }

    original_text = text

    # Apply every replacement rule.
    for old_keyword, new_keyword in replacements.items():
        text = text.replace(old_keyword, new_keyword)

    # Record replacements at debug level only.
    if text != original_text:
        logger.debug(f"[KEYWORD_REPLACE] Applied keyword replacements, original length: {len(original_text)}, new length: {len(text)}")

    return text


def apply_keyword_replacement_to_system_message(content) -> str:
    """
    Apply keyword replacements to system messages.
    Supports both strings and structured content.

    Args:
        content: Message content, either a string or a list structure.

    Returns:
        str: The processed content.
    """
    if isinstance(content, str):
        return apply_keyword_replacement(content)
    elif isinstance(content, list):
        # Process structured system-message content.
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                item["text"] = apply_keyword_replacement(item.get("text", ""))
        return content
    else:
        return content
