"""
CodeBuddy API Client - Calls the CodeBuddy API directly.
"""
import json
import time
import uuid
import secrets
import httpx
import logging
from typing import Dict, Any, Optional, AsyncGenerator, List

logger = logging.getLogger(__name__)


class CodeBuddyAPIClient:
    """CodeBuddy API client."""

    def __init__(self):
        from config import get_codebuddy_api_endpoint
        self.base_url = get_codebuddy_api_endpoint()
        self.api_endpoint = self.base_url  # Use base_url directly; no plugin prefix is required.

    def convert_openai_to_codebuddy_messages(self, openai_messages: List[Dict]) -> List[Dict]:
        """Convert OpenAI-format messages to CodeBuddy format."""
        codebuddy_messages = []

        # Filter messages containing errors to avoid triggering channel detection 11128.
        filtered_messages = []
        for msg in openai_messages:
            content = msg.get("content", "")
            # Skip assistant messages containing API error text.
            if (msg.get("role") == "assistant" and
                isinstance(content, str) and
                ("Error: API error" in content or "API error:" in content)):
                continue
            filtered_messages.append(msg)

        # CodeBuddy requires at least two messages. Add a system message when only one user message exists.
        if len(filtered_messages) == 1 and filtered_messages[0].get("role") == "user":
            system_msg = {
                "role": "system",
                "content": "You are a helpful assistant."
            }
            codebuddy_messages.append(system_msg)

        for msg in filtered_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            logger.debug(f"[DEBUG] Processing message - role: {role}, content type: {type(content)}")

            # Convert the special tool role to a user role.
            if role == "tool":
                role = "user"
                logger.info("[ROLE_CONVERSION] Converting 'tool' role to 'user'")

            # Detect tool-call content.
            has_tool_content = False

            # Parse stringified JSON content.
            if isinstance(content, str) and content.startswith('[{') and content.endswith('}]'):
                try:
                    parsed_content = json.loads(content)
                    if isinstance(parsed_content, list):
                        content = parsed_content
                        logger.info("[JSON_PARSE] Parsed stringified JSON content")
                except json.JSONDecodeError:
                    pass

            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") in ["tool_result", "tool_use"]:
                        has_tool_content = True
                        break

            if has_tool_content:
                # Preserve structured tool-call content.
                logger.info(f"[TOOL_CONTENT] Preserving structured content for role: {role}")

                # Ensure every tool result has a valid toolUseId.
                processed_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "tool_result":
                            tool_use_id = item.get("toolUseId") or item.get("tool_use_id") or item.get("id")
                            if not tool_use_id:
                                # Generate a valid toolUseId.
                                tool_use_id = f"tool_{uuid.uuid4().hex[:8]}"
                                logger.warning(f"[TOOL_RESULT] Missing toolUseId, generated: {tool_use_id}")

                            # Ensure toolUseId matches [a-zA-Z0-9_-]+.
                            if not tool_use_id or not all(c.isalnum() or c in '_-' for c in tool_use_id):
                                tool_use_id = f"tool_{uuid.uuid4().hex[:8]}"
                                logger.warning(f"[TOOL_RESULT] Invalid toolUseId format, regenerated: {tool_use_id}")

                            # Normalize the tool-result format.
                            tool_result = {
                                "type": "tool_result",
                                "toolUseId": tool_use_id,
                                "content": item.get("content", item.get("text", ""))
                            }
                            processed_content.append(tool_result)
                            logger.info(f"[TOOL_RESULT] Processed tool result with toolUseId: {tool_use_id}")
                        elif item.get("type") == "tool_use":
                            # Ensure every tool use has an ID.
                            tool_id = item.get("id") or f"tool_{uuid.uuid4().hex[:8]}"
                            tool_use = {
                                "type": "tool_use",
                                "id": tool_id,
                                "name": item.get("name", ""),
                                "input": item.get("input", {})
                            }
                            processed_content.append(tool_use)
                            logger.info(f"[TOOL_USE] Processed tool use with id: {tool_id}")
                        elif item.get("type") == "text":
                            # Preserve plain text content.
                            processed_content.append(item)
                        else:
                            # Convert simplified tool-result formats when possible.
                            if "text" in item and not item.get("type"):
                                tool_use_id = f"tool_{uuid.uuid4().hex[:8]}"
                                tool_result = {
                                    "type": "tool_result",
                                    "toolUseId": tool_use_id,
                                    "content": item.get("text", "")
                                }
                                processed_content.append(tool_result)
                                logger.info(f"[TOOL_RESULT] Converted text item to tool result with toolUseId: {tool_use_id}")
                            else:
                                processed_content.append(item)
                    else:
                        processed_content.append(item)

                codebuddy_msg = {
                    "role": role,
                    "content": processed_content
                }
            else:
                # Convert ordinary text content to a string.
                if isinstance(content, str):
                    text_content = content
                elif isinstance(content, list):
                    text_parts = []
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                text_parts.append(item.get("text", ""))
                            else:
                                text_parts.append(json.dumps(item, ensure_ascii=False))
                        elif isinstance(item, str):
                            text_parts.append(item)
                        else:
                            text_parts.append(str(item))
                    text_content = "".join(text_parts)
                else:
                    text_content = str(content) if content is not None else ""

                codebuddy_msg = {
                    "role": role,
                    "content": text_content
                }

            codebuddy_messages.append(codebuddy_msg)

        return codebuddy_messages

    def generate_codebuddy_headers(
        self,
        bearer_token: str,
        user_id: str = None,
        conversation_id: Optional[str] = None,
        conversation_request_id: Optional[str] = None,
        conversation_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
        api_key_header: str = "bearer"
    ) -> Dict[str, str]:
        """
        Generate the complete header set required by the CodeBuddy API.
        Prefer supplied conversation IDs and generate missing IDs automatically.
        """
        if api_key_header not in {"x-api-key", "bearer", "both"}:
            raise ValueError("api_key_header must be x-api-key, bearer, or both")
        headers = {
            'Host': 'www.codebuddy.ai',
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'X-Requested-With': 'XMLHttpRequest',
            'x-stainless-arch': 'x64',
            'x-stainless-lang': 'js',
            'x-stainless-os': 'Windows',
            'x-stainless-package-version': '5.10.1',
            'x-stainless-retry-count': '0',
            'x-stainless-runtime': 'node',
            'x-stainless-runtime-version': 'v22.13.1',
            'X-Conversation-ID': conversation_id or str(uuid.uuid4()),
            'X-Conversation-Request-ID': conversation_request_id or secrets.token_hex(16),
            'X-Conversation-Message-ID': conversation_message_id or str(uuid.uuid4()).replace('-', ''),
            'X-Request-ID': request_id or str(uuid.uuid4()).replace('-', ''),
            'X-Agent-Intent': 'craft',
            'X-IDE-Type': 'CLI',
            'X-IDE-Name': 'CLI',
            'X-IDE-Version': '1.0.7',
            'X-Domain': 'www.codebuddy.ai',
            'User-Agent': 'CLI/1.0.7 CodeBuddy/1.0.7',
            'X-Product': 'SaaS',
            'X-User-Id': user_id or 'b5be3a67-237e-4ee6-9b9a-0b9ecd7b454b'
        }
        if api_key_header in {"x-api-key", "both"}:
            headers["X-API-Key"] = bearer_token
        if api_key_header in {"bearer", "both"}:
            headers["Authorization"] = f"Bearer {bearer_token}"
        return headers


# Global client instance.
codebuddy_api_client = CodeBuddyAPIClient()
