"""
CodeBuddy API Router - compatible with the official CodeBuddy API format
Refactored version - improved code structure, error handling, and resource management
"""
import json
import time
import uuid
import hashlib
import logging
import asyncio
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, AsyncGenerator, Set

import httpx
from fastapi import APIRouter, HTTPException, Depends, Request, Header
from fastapi.responses import JSONResponse, StreamingResponse

from .auth import ClientAuthContext, authenticate_admin, authenticate_inference
from .codebuddy_api_client import codebuddy_api_client
from .codebuddy_api_key_manager import (
    ApiKeyConfigurationError,
    codebuddy_api_key_manager,
)
from .codebuddy_token_manager import codebuddy_token_manager
from .usage_stats_manager import usage_stats_manager
from .keyword_replacer import apply_keyword_replacement_to_system_message
from .codebuddy_message_sanitizer import (
    MessageNormalizationError,
    is_codebuddy_moderation_response,
    normalize_messages_for_upstream,
    sanitize_messages,
)
from config import (
    get_codebuddy_default_model,
    get_codebuddy_model_aliases,
    get_codebuddy_request_profile,
    get_codebuddy_unknown_model_policy,
    get_max_system_prompt_length,
    get_sanitize_agent_prompt,
    get_upstream_api_key_header,
)
logger = logging.getLogger(__name__)

router = APIRouter()

# --- Lazily loaded configuration constants - avoids circular imports ---
_codebuddy_api_url: Optional[str] = None
_available_models: Optional[List[str]] = None

def get_codebuddy_api_url() -> str:
    """Lazily load the CodeBuddy API URL"""
    global _codebuddy_api_url
    if _codebuddy_api_url is None:
        from config import get_codebuddy_api_endpoint
        _codebuddy_api_url = f"{get_codebuddy_api_endpoint()}/v2/chat/completions"
    return _codebuddy_api_url

def get_available_models_list() -> List[str]:
    """Lazily load the list of available models"""
    global _available_models
    if _available_models is None:
        from config import get_available_models
        _available_models = get_available_models()
    return _available_models

# --- Configuration management ---
class SecurityConfig:
    """Security configuration manager"""

    @staticmethod
    def get_ssl_verify() -> bool:
        """Get the SSL verification setting - disabled by default, can be enabled via environment variable"""
        import os
        # SSL verification is disabled by default; only enabled when explicitly set to true
        ssl_verify_env = os.getenv("CODEBUDDY_SSL_VERIFY", "false").lower()
        ssl_verify = ssl_verify_env == "true"

        if not ssl_verify:
            logger.warning("⚠️  SSL verification is disabled - for development use only! Set CODEBUDDY_SSL_VERIFY=true in production")
        
        return ssl_verify

# --- HTTP client configuration ---
HTTP_CLIENT_CONFIG = {
    "verify": SecurityConfig.get_ssl_verify(),
    "timeout": httpx.Timeout(300.0, connect=30.0, read=300.0),
    "limits": httpx.Limits(max_keepalive_connections=20, max_connections=100)
}

# --- Async-safe HTTP client pool ---
_http_client_pool: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()

async def get_http_client() -> httpx.AsyncClient:
    """Get the global HTTP client pool - async-safe"""
    global _http_client_pool
    if _http_client_pool is None:
        async with _client_lock:
            # Double-checked locking pattern - async version
            if _http_client_pool is None:
                _http_client_pool = httpx.AsyncClient(**HTTP_CLIENT_CONFIG)
    return _http_client_pool

async def close_http_client():
    """Close the global HTTP client pool - async-safe"""
    global _http_client_pool
    async with _client_lock:
        if _http_client_pool is not None:
            await _http_client_pool.aclose()
            _http_client_pool = None

# --- Application lifecycle management ---
class AppLifecycleManager:
    """Application lifecycle manager - handles resource cleanup"""

    @staticmethod
    async def startup():
        """Initialization at application startup"""
        logger.info("CodeBuddy Router starting up...")
        # Warm up the connection pool, and start API key file reloading only in the relevant auth modes
        from config import get_codebuddy_auth_mode

        await get_http_client()
        try:
            auth_mode = get_codebuddy_auth_mode()
        except ValueError:
            auth_mode = "auto"
            logger.error("Invalid CODEBUDDY_AUTH_MODE configuration")
        if auth_mode in {"auto", "api_key_file"}:
            await codebuddy_api_key_manager.start_periodic_reload()
        logger.info("HTTP connection pool and API key pool initialized")

    @staticmethod
    async def shutdown():
        """Cleanup at application shutdown"""
        logger.info("CodeBuddy Router shutting down...")
        await codebuddy_api_key_manager.stop_periodic_reload()
        await close_http_client()
        logger.info("Resource cleanup complete")

# Export the lifecycle manager for use by the main application
lifecycle_manager = AppLifecycleManager()

# --- Standard response headers ---
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "*"
}

# --- Helper functions ---

def format_sse_error(message: str, error_type: str = "stream_error") -> str:
    """Format an SSE error response"""
    error_data = {
        "error": {
            "message": message,
            "type": error_type
        }
    }
    return f'data: {json.dumps(error_data, ensure_ascii=False)}\n\n'

class OpenAICompatibilityConverter:
    """Convert the CodeBuddy format to the OpenAI-compatible format"""

    @staticmethod
    def convert_tool_call_id(codebuddy_id: str) -> str:
        """Convert the tool call ID format: tooluse_xxx -> call_xxx"""
        if codebuddy_id.startswith('tooluse_'):
            return f"call_{codebuddy_id[8:]}"
        return codebuddy_id
    
    @staticmethod
    def convert_sse_chunk_to_openai_format(chunk_data: Dict[str, Any], tool_call_index_map: Dict[str, int]) -> Dict[str, Any]:
        """Convert a CodeBuddy SSE chunk to OpenAI format"""
        if not chunk_data.get('choices'):
            return chunk_data
        
        choice = chunk_data['choices'][0]
        delta = choice.get('delta', {})
        tool_calls = delta.get('tool_calls', [])
        
        if not tool_calls:
            return chunk_data
        
        # Convert tool call format
        converted_tool_calls = []
        for tc in tool_calls:
            converted_tc = tc.copy()

            # Convert ID format
            if tc.get('id'):
                original_id = tc['id']
                converted_id = OpenAICompatibilityConverter.convert_tool_call_id(original_id)
                converted_tc['id'] = converted_id

                # Assign a new index
                if original_id not in tool_call_index_map:
                    tool_call_index_map[original_id] = len(tool_call_index_map)

                converted_tc['index'] = tool_call_index_map[original_id]

            # If there is no ID, use the current latest index
            elif tool_call_index_map:
                # Use the index of the last tool call
                converted_tc['index'] = max(tool_call_index_map.values())

            converted_tool_calls.append(converted_tc)

        # Update chunk data
        converted_chunk = chunk_data.copy()
        converted_chunk['choices'][0]['delta']['tool_calls'] = converted_tool_calls
        
        return converted_chunk

def parse_sse_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a single line of SSE data"""
    if not line.startswith('data: '):
        return None
    
    data = line[6:].strip()
    if not data or data == '[DONE]':
        return None
    
    try:
        return json.loads(data)
    except json.JSONDecodeError:
        return None

def validate_and_fix_tool_call_args(args: str) -> str:
    """Enhanced validation and repair of tool call arguments - specifically handles multi-tool-call issues"""
    if not args:
        return '{}'
    
    args = args.strip()
    
    # Check whether multiple JSON objects are concatenated - this is the main multi-tool-call problem
    if args.count('}{') > 0:
        # Try to separate the multiple JSON objects
        json_objects = []
        current_obj = ""
        brace_count = 0
        
        for i, char in enumerate(args):
            current_obj += char
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and current_obj.strip():
                    # Completed one JSON object
                    try:
                        parsed = json.loads(current_obj.strip())
                        json_objects.append(parsed)
                        current_obj = ""
                    except json.JSONDecodeError:
                        current_obj = ""
        
        if json_objects:
            return json.dumps(json_objects[0], ensure_ascii=False)
    
    # Original repair logic
    try:
        json.loads(args)
        return args
    except json.JSONDecodeError as e:


        # Try to fix common JSON problems
        original_args = args
        if not args.endswith('}') and args.count('{') > args.count('}'):
            args += '}'
            
        elif not args.endswith(']') and args.count('[') > args.count(']'):
            args += ']'
            
        
        try:
            json.loads(args)
            
            return args
        except json.JSONDecodeError:
            return '{}'

class SSEConnectionManager:
    """SSE connection manager, including reconnection logic"""
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        self.max_retries = max_retries
        self.retry_delay = retry_delay
    
    async def stream_with_retry(self, stream_func, *args, **kwargs):
        """Streaming processing with reconnection"""
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in stream_func(*args, **kwargs):
                    yield chunk
                break  # Completed successfully, exit the retry loop
            except (httpx.TimeoutException, httpx.NetworkError) as e:
                if attempt < self.max_retries:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                    logger.warning(f"Connection failed, retrying in {wait_time}s (attempt {attempt + 1}): {e}")
                    yield format_sse_error(f"Connection lost, retrying in {wait_time}s... (attempt {attempt + 1})", "connection_retry")
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Reconnection failed, maximum retry count reached: {e}")
                    yield format_sse_error(f"Connection failed after {self.max_retries} retries: {str(e)}", "connection_failed")
                    raise
            except Exception as e:
                # Other exceptions are not retried; re-raise directly
                logger.error(f"Streaming processing exception: {e}")
                yield format_sse_error(f"Stream error: {str(e)}", "stream_error")
                raise

class StreamResponseAggregator:
    """Streaming response aggregator - fixes multi-tool-call issues by using the tool call ID as the key"""
    
    def __init__(self):
        self.data = {
            "id": None,
            "model": None,
            "content": "",
            "tool_calls": [],
            "finish_reason": None,
            "usage": None,
            "system_fingerprint": None
        }
        # 🔑 Key point: use the tool call ID as the key, because the index is always 0 and would overwrite
        self.tool_call_map = {}  # key: tool_call_id, value: tool_call_data
        self.tool_call_order = []  # Preserve the order in which tool calls are received
        self.current_tool_id = None  # ID of the tool call currently being processed

    def process_chunk(self, obj: Dict[str, Any]):
        """Process a single response chunk"""
        # Aggregate basic information
        self.data["id"] = self.data["id"] or obj.get('id')
        self.data["model"] = self.data["model"] or obj.get('model')
        self.data["system_fingerprint"] = obj.get('system_fingerprint') or self.data["system_fingerprint"]
        
        if obj.get('usage'):
            self.data["usage"] = obj.get('usage')
        
        choices = obj.get('choices', [])
        if not choices:
            return
        
        choice = choices[0]
        if choice.get('finish_reason'):
            self.data["finish_reason"] = choice.get('finish_reason')
        
        delta = choice.get('delta', {})
        
        # Aggregate content
        if delta.get('content'):
            self.data["content"] += delta.get('content')

        # Handle tool calls
        if delta.get('tool_calls'):
            self._process_tool_calls(delta.get('tool_calls'))

    def _process_tool_calls(self, tool_calls: List[Dict[str, Any]]):
        """Handle tool calls - fixed version: use the tool call ID and correctly handle chunked transfer"""
        for tc in tool_calls:
            tool_id = tc.get('id')

            # If there is an ID, this is a new tool call
            if tool_id:
                # New tool call
                if tool_id not in self.tool_call_map:
                    self.tool_call_map[tool_id] = {
                        'id': tool_id,
                        'type': tc.get('type', 'function'),
                        'function': {
                            'name': '',
                            'arguments': ''
                        }
                    }
                    self.tool_call_order.append(tool_id)
                    self.current_tool_id = tool_id
                    logger.info(f"🔧 New tool call: {tool_id}")
                else:
                    # Update the current tool call ID
                    self.current_tool_id = tool_id

                # Update tool call information
                if tc.get('type'):
                    self.tool_call_map[tool_id]['type'] = tc.get('type')
                
                func = tc.get('function', {})
                if func.get('name'):
                    self.tool_call_map[tool_id]['function']['name'] = func.get('name')
                if func.get('arguments'):
                    self.tool_call_map[tool_id]['function']['arguments'] += func.get('arguments')
            
            # If there is no ID but there is a current tool call ID, this is incremental data
            elif self.current_tool_id and self.current_tool_id in self.tool_call_map:
                func = tc.get('function', {})
                if func.get('name'):
                    self.tool_call_map[self.current_tool_id]['function']['name'] = func.get('name')
                if func.get('arguments'):
                    self.tool_call_map[self.current_tool_id]['function']['arguments'] += func.get('arguments')
            
            else:
                # No ID and no current tool call, skip
                logger.warning("⚠️ Tool call is missing an ID and there is no current tool call context, skipping")

    def finalize(self) -> Dict[str, Any]:
        """Finish aggregation and return the final response"""
        # Build the tool call list in the order received
        if self.tool_call_map:
            self.data["tool_calls"] = []
            for tool_id in self.tool_call_order:
                if tool_id in self.tool_call_map:
                    tc = self.tool_call_map[tool_id]
                    # Validate and repair the tool call arguments
                    tc['function']['arguments'] = validate_and_fix_tool_call_args(
                        tc['function']['arguments']
                    )
                    self.data["tool_calls"].append(tc)
                    logger.info(f"📋 Tool call: {tool_id} - {tc['function']['name']}")

            logger.info(f"✅ Successfully aggregated {len(self.data['tool_calls'])} tool call(s)")

        # Build the final response
        final_message = {"role": "assistant", "content": self.data["content"]}
        if self.data["tool_calls"]:
            final_message["tool_calls"] = self.data["tool_calls"]
        
        finish_reason = "tool_calls" if self.data["tool_calls"] else (self.data["finish_reason"] or "stop")
        
        final_response = {
            "id": self.data["id"] or str(uuid.uuid4()),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": self.data["model"] or "unknown",
            "choices": [
                {
                    "index": 0,
                    "message": final_message,
                    "finish_reason": finish_reason,
                    "logprobs": None
                }
            ]
        }
        
        if self.data["usage"]:
            final_response["usage"] = self.data["usage"]
        if self.data["system_fingerprint"]:
            final_response["system_fingerprint"] = self.data["system_fingerprint"]
        
        return final_response

class UpstreamAttemptError(Exception):
    """Safe error that does not carry the upstream response body."""

    def __init__(self, kind: str, status_code: int, code: str):
        super().__init__(code)
        self.kind = kind
        self.status_code = status_code
        self.code = code


class CodeBuddyModerationError(Exception):
    """Raised when CodeBuddy rejects a request via its content moderation.

    This is distinct from an upstream failure: the key is valid and the
    request reached upstream. It must not trigger key failover or retries.
    """


class UnknownModelError(Exception):
    """Raised when a requested model is unknown and the policy is 'reject'.

    Signals that the request must be rejected with HTTP 400 (code
    ``unknown_model``) before any upstream call, instead of silently rewriting
    the requested model to a fallback.
    """

    def __init__(self, requested_model: str):
        super().__init__(requested_model)
        self.requested_model = requested_model


# Human-readable message returned to the client on a moderation rejection.
MODERATION_MESSAGE = (
    "CodeBuddy rejected the request through its content moderation system. "
    "This may be caused by the system prompt or conversation history."
)

# Shorter message used inside the streaming content_filter delta.
MODERATION_STREAM_MESSAGE = (
    "CodeBuddy rejected the request through its content moderation system."
)

# Number of assistant-content characters to buffer while deciding whether a
# streaming response is a moderation refusal. The Mandarin refusal is short and
# self-contained, so a small buffer distinguishes it from a normal reply.
MODERATION_STREAM_BUFFER_CHARS = 200


def _extract_delta_content(sse_line: str) -> str:
    """Return the assistant ``delta.content`` string from an OpenAI SSE line."""
    obj = parse_sse_line(sse_line.strip())
    if not obj:
        return ""
    try:
        choices = obj.get("choices") or []
        if not choices:
            return ""
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        return content if isinstance(content, str) else ""
    except (AttributeError, IndexError, TypeError):
        return ""


async def _moderation_stream() -> AsyncGenerator[str, None]:
    """Yield an OpenAI-compatible content_filter SSE stream, ending with [DONE]."""
    chunk = {
        "id": "chatcmpl-codebuddy-filter",
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "choices": [
            {
                "index": 0,
                "delta": {"content": MODERATION_STREAM_MESSAGE},
                "finish_reason": "content_filter",
            }
        ],
    }
    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


def _key_fingerprint(token: Optional[str]) -> str:
    """Return a short, non-reversible SHA-256 fingerprint of an API key.

    Used only for safe diagnostics. The raw key is never logged.
    """
    if not token:
        return "none"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


def _content_types_and_lengths(messages: List[Any]) -> tuple[List[str], List[str], List[int], int]:
    """Return (roles, content_types, content_lengths, tool_content_count).

    Metadata only: describes the shape of each message without exposing text.
    """
    roles: List[str] = []
    content_types: List[str] = []
    content_lengths: List[int] = []
    tool_content_count = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        roles.append(str(msg.get("role", "unknown")))
        content = msg.get("content", "")
        if isinstance(content, str):
            content_types.append("str")
            content_lengths.append(len(content))
        elif isinstance(content, list):
            content_types.append("list")
            length = 0
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in {"tool_use", "tool_result"}:
                        tool_content_count += 1
                    if item.get("type") == "text":
                        length += len(str(item.get("text", "")))
            content_lengths.append(length)
        else:
            content_types.append(type(content).__name__)
            content_lengths.append(0)
    return roles, content_types, content_lengths, tool_content_count


def _log_request_diagnostics(
    *,
    payload: Dict[str, Any],
    prep_info: Dict[str, Any],
    client_wants_stream: bool,
    request_profile: str,
    key_fingerprint: str,
    outcome: str = "prepared",
    upstream_response_model: Optional[str] = None,
) -> None:
    """Log safe request metadata for successful and failed requests.

    Never logs message contents, raw keys, or the mapped-away model label as
    text beyond what the client requested. ``outcome`` tags the log point
    (e.g. "prepared", "success", "failed:<code>").

    ``requested_model`` (what the client asked for), ``mapped_model`` (what we
    send upstream), ``mapping_source`` (how it was resolved), and
    ``upstream_response_model`` (what CodeBuddy actually reported) are logged
    separately so a divergence between the selected UI model and the upstream
    model is visible rather than hidden.
    """
    messages = payload.get("messages", []) or []
    roles, content_types, content_lengths, tool_content_count = _content_types_and_lengths(messages)
    tool_count = tool_content_count + len(payload.get("tools", []) or [])
    logger.info(
        "CodeBuddy request outcome=%s requested_model=%s mapped_model=%s mapping_source=%s "
        "upstream_response_model=%s stream=%s "
        "roles=[%s] content_types=[%s] content_lengths=%s has_tools=%s has_tool_choice=%s "
        "tool_count=%d dropped_fields=[%s] system_prompt_sanitized=%s request_profile=%s "
        "key_fingerprint=%s",
        outcome,
        prep_info.get("requested_model", "unknown"),
        prep_info.get("mapped_model", payload.get("model", "unknown")),
        prep_info.get("mapping_source", "unknown"),
        upstream_response_model or "unknown",
        client_wants_stream,
        ",".join(roles),
        ",".join(content_types),
        content_lengths,
        "tools" in payload,
        "tool_choice" in payload,
        tool_count,
        ",".join(prep_info.get("dropped_fields", []) or []),
        prep_info.get("system_prompt_sanitized", False),
        request_profile,
        key_fingerprint,
    )


class CodeBuddyStreamService:
    """CodeBuddy streaming service; each method performs exactly one upstream attempt."""

    @staticmethod
    def _classify_status(status_code: int) -> UpstreamAttemptError:
        if status_code == 401:
            return UpstreamAttemptError("invalid", 401, "upstream_authentication_failed")
        if status_code in {403, 429}:
            return UpstreamAttemptError("cooldown", status_code, "upstream_temporarily_unavailable")
        if status_code >= 500:
            return UpstreamAttemptError("transient", status_code, "upstream_server_error")
        return UpstreamAttemptError("fatal", status_code, "upstream_request_rejected")

    async def open_stream_response(
        self,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        key_id: Optional[str] = None,
    ) -> StreamingResponse:
        """Establish the connection and verify the upstream status before returning a StreamingResponse."""
        client = await get_http_client()
        request = client.build_request(
            "POST", get_codebuddy_api_url(), json=payload, headers=headers
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise UpstreamAttemptError("transient", 504, "upstream_timeout") from exc
        except httpx.RequestError as exc:
            raise UpstreamAttemptError("transient", 502, "upstream_network_error") from exc

        if response.status_code != 200:
            status_code = response.status_code
            await response.aclose()
            raise self._classify_status(status_code)

        # Do not expose upstream response headers; only this proxy's fixed SSE headers
        # are sent downstream after the first chunk is safely available.

        async def converted_chunks():
            buffer = ""
            tool_call_index_map = {}
            async for chunk in response.aiter_text(chunk_size=8192):
                if not chunk:
                    continue
                buffer += chunk

                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    if not line.strip() or line.startswith(':'):
                        continue
                    if '[DONE]' in line:
                        yield line + '\n'
                        return

                    chunk_data = parse_sse_line(line)
                    if chunk_data:
                        converted_chunk = OpenAICompatibilityConverter.convert_sse_chunk_to_openai_format(
                            chunk_data, tool_call_index_map
                        )
                        line = f"data: {json.dumps(converted_chunk, ensure_ascii=False)}"
                    yield line + '\n'

            if buffer.strip():
                chunk_data = parse_sse_line(buffer.strip())
                if chunk_data:
                    converted_chunk = OpenAICompatibilityConverter.convert_sse_chunk_to_openai_format(
                        chunk_data, tool_call_index_map
                    )
                    buffer = f"data: {json.dumps(converted_chunk, ensure_ascii=False)}"
                yield buffer + '\n'

        stream = converted_chunks()

        # Pre-buffer the leading chunks so a Mandarin moderation refusal can be
        # detected before any assistant content is sent downstream. The refusal
        # is short, so a small buffer suffices; normal replies simply get
        # replayed afterwards in order.
        buffered_lines: List[str] = []
        accumulated_content = ""
        moderation_detected = False
        try:
            async for line in stream:
                buffered_lines.append(line)
                if '[DONE]' in line:
                    break
                accumulated_content += _extract_delta_content(line)
                if is_codebuddy_moderation_response(accumulated_content):
                    moderation_detected = True
                    break
                if len(accumulated_content) >= MODERATION_STREAM_BUFFER_CHARS:
                    break
        except httpx.TimeoutException as exc:
            await response.aclose()
            raise UpstreamAttemptError("transient", 504, "upstream_timeout") from exc
        except httpx.RequestError as exc:
            await response.aclose()
            raise UpstreamAttemptError("transient", 502, "upstream_network_error") from exc
        except Exception as exc:
            await response.aclose()
            raise UpstreamAttemptError("fatal", 502, "upstream_response_invalid") from exc

        if moderation_detected:
            # Nothing normal was sent yet: replace the whole stream with an
            # OpenAI-compatible content_filter stream. The key is valid, so do
            # not mark it failed or fail over.
            await response.aclose()

            async def moderation_core():
                async for item in _moderation_stream():
                    yield item

            return StreamingResponse(
                moderation_core(), media_type="text/event-stream", headers={
                    **SSE_HEADERS, "X-CodeBuddy-Moderation": "true"
                }
            )

        async def stream_core():
            try:
                for line in buffered_lines:
                    yield line
                async for chunk in stream:
                    yield chunk
            except httpx.RequestError:
                logger.warning("CodeBuddy upstream stream interrupted")
                if key_id is not None:
                    await codebuddy_api_key_manager.mark_transient_error(
                        key_id, "stream_interrupted"
                    )
                yield format_sse_error(
                    "Upstream stream interrupted", "upstream_stream_error"
                )
            except Exception:
                logger.error("Unexpected CodeBuddy stream processing error")
                if key_id is not None:
                    await codebuddy_api_key_manager.mark_transient_error(
                        key_id, "stream_processing_error"
                    )
                yield format_sse_error(
                    "Upstream stream interrupted", "upstream_stream_error"
                )
            finally:
                await response.aclose()

        return StreamingResponse(
            stream_core(), media_type="text/event-stream", headers=SSE_HEADERS
        )

    async def handle_non_stream_response(
        self, payload: Dict[str, Any], headers: Dict[str, str]
    ) -> Dict[str, Any]:
        """Perform a single non-streaming upstream request and aggregate the SSE response."""
        try:
            client = await get_http_client()
            response = await client.post(
                get_codebuddy_api_url(), json=payload, headers=headers
            )
        except httpx.TimeoutException as exc:
            raise UpstreamAttemptError("transient", 504, "upstream_timeout") from exc
        except httpx.RequestError as exc:
            raise UpstreamAttemptError("transient", 502, "upstream_network_error") from exc

        if response.status_code != 200:
            status_code = response.status_code
            await response.aclose()
            raise self._classify_status(status_code)

        try:
            aggregator = StreamResponseAggregator()
            raw_text = ""
            buffer = ""
            async for chunk in response.aiter_text():
                if not chunk:
                    continue
                raw_text += chunk
                buffer += chunk
                while '\n' in buffer:
                    line, buffer = buffer.split('\n', 1)
                    obj = parse_sse_line(line)
                    if obj:
                        aggregator.process_chunk(obj)

            if buffer.strip():
                obj = parse_sse_line(buffer.strip())
                if obj:
                    aggregator.process_chunk(obj)
            final = aggregator.finalize()
        except httpx.RequestError as exc:
            raise UpstreamAttemptError("transient", 502, "upstream_network_error") from exc
        except UpstreamAttemptError:
            raise
        except Exception as exc:
            raise UpstreamAttemptError("fatal", 502, "upstream_response_invalid") from exc

        # Detect a CodeBuddy moderation refusal in either the aggregated
        # assistant content or the raw upstream body (covers nested/non-SSE
        # bodies). This is not an upstream failure, so it must not trigger
        # key failover.
        aggregated_content = ""
        try:
            aggregated_content = final["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            aggregated_content = ""
        if is_codebuddy_moderation_response(aggregated_content) or \
                is_codebuddy_moderation_response(raw_text):
            raise CodeBuddyModerationError()
        return final

class RequestProcessor:
    """Request preprocessor - thread-safe request handling"""

    # Top-level request fields forwarded to CodeBuddy upstream. Everything else
    # (OpenAI-only fields the upstream rejects, or unknown UI fields) is dropped.
    # ``tools`` and ``tool_choice`` are forwarded only when present.
    _UPSTREAM_ALLOWLIST = {"model", "messages", "stream", "tools", "tool_choice"}

    @staticmethod
    def resolve_model(requested_model: Any) -> tuple[str, str]:
        """Resolve a client/UI model label to an upstream CodeBuddy model ID.

        Returns ``(mapped_model, mapping_source)`` where ``mapping_source`` is:
          * ``"exact"``          - matched a configured available model ID
          * ``"alias"``          - matched a configured alias (case-insensitive)
          * ``"default_empty"``  - no model supplied; used the default
          * ``"passthrough"``    - unknown label forwarded verbatim
          * ``"default"``        - unknown label mapped to the default model

        An unknown label (neither a known model ID nor a configured alias) is
        handled per CODEBUDDY_UNKNOWN_MODEL_POLICY:
          * ``passthrough`` (default): forward it verbatim and let CodeBuddy
            accept or reject it. The requested model is never silently rewritten.
          * ``reject``: raise :class:`UnknownModelError` (HTTP 400 upstream).
          * ``default``: fall back to CODEBUDDY_DEFAULT_MODEL.
        """
        try:
            available = set(get_available_models_list())
        except Exception:
            available = set()
        try:
            default_model = get_codebuddy_default_model()
        except Exception:
            default_model = "auto-chat"

        # No usable model supplied: there is nothing to pass through, so use
        # the configured default regardless of policy.
        if not isinstance(requested_model, str) or not requested_model.strip():
            return default_model, "default_empty"
        requested = requested_model.strip()

        # Exact match against a known upstream model ID (this covers a client
        # that explicitly requests "auto-chat").
        if requested in available:
            return requested, "exact"

        # Configured alias mapping (case-insensitive).
        try:
            aliases = get_codebuddy_model_aliases()
        except Exception:
            aliases = {}
        mapped = aliases.get(requested.lower())
        if mapped:
            return mapped, "alias"

        # Unknown label: behavior is governed by the configured policy. Never
        # silently rewrite an unknown model to the default.
        try:
            policy = get_codebuddy_unknown_model_policy()
        except Exception:
            policy = "passthrough"

        if policy == "reject":
            logger.info("Unknown requested model rejected (policy=reject)")
            raise UnknownModelError(requested)
        if policy == "default":
            logger.info("Unknown requested model mapped to default (policy=default)")
            return default_model, "default"
        # passthrough (default): forward the requested model verbatim so the
        # real upstream model is preserved and CodeBuddy decides if it is valid.
        return requested, "passthrough"

    @staticmethod
    def map_model(requested_model: Any) -> str:
        """Backward-compatible wrapper returning only the resolved model ID.

        May raise :class:`UnknownModelError` when the unknown-model policy is
        ``reject``.
        """
        mapped, _source = RequestProcessor.resolve_model(requested_model)
        return mapped

    @staticmethod
    def prepare_payload(request_body: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Build a strict, upstream-safe payload from the client request body.

        Instead of forwarding the client body verbatim, only allowlisted
        top-level fields are forwarded. This prevents OpenAI-only fields (e.g.
        response_format, reasoning_effort, stream_options) and unknown UI fields
        from reaching CodeBuddy, which is the common cause of upstream rejection.

        Returns ``(payload, prep_info)`` where ``prep_info`` carries safe
        metadata for diagnostics (never message content or keys):
          * system_prompt_sanitized: bool
          * requested_model: str
          * mapped_model: str
          * mapping_source: str (how the model was resolved)
          * dropped_fields: sorted list of ignored top-level field names

        May raise :class:`UnknownModelError` when the requested model is unknown
        and CODEBUDDY_UNKNOWN_MODEL_POLICY is ``reject``.
        """
        source = request_body if isinstance(request_body, dict) else {}

        requested_model = source.get("model")
        mapped_model, mapping_source = RequestProcessor.resolve_model(requested_model)

        messages = source.get("messages", []) or []

        # Sanitize only agent system prompts that tend to trigger false-positive
        # moderation. User/assistant/tool messages are never modified, and
        # legitimate short system prompts are left intact.
        try:
            sanitize_enabled = get_sanitize_agent_prompt()
            max_len = get_max_system_prompt_length()
        except Exception:
            sanitize_enabled, max_len = True, 2000
        messages, system_prompt_sanitized = sanitize_messages(
            messages, enabled=sanitize_enabled, max_system_prompt_length=max_len
        )

        # CodeBuddy requires at least two messages. Only add a default system
        # prompt when there is no system message already, to avoid duplicating
        # system messages or reordering the conversation.
        has_system = any(
            isinstance(m, dict) and m.get("role") == "system" for m in messages
        )
        if not has_system and len(messages) == 1 and messages[0].get("role") == "user":
            system_msg = {
                "role": "system",
                "content": "You are a helpful assistant. Reply in the same language as the user.",
            }
            messages = [system_msg] + messages

        # Apply keyword replacement to system messages only.
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                msg["content"] = apply_keyword_replacement_to_system_message(msg.get("content"))

        # Final transform before upstream: after all Anthropic/OpenAI/tool
        # conversion and system-prompt sanitization, guarantee every message
        # has a valid role and a content field. This inspects the exact messages
        # CodeBuddy will receive (e.g. Claude Code assistant tool_calls turns
        # with null content, or tool-result messages), preventing the upstream
        # "Message N must have 'role' and 'content' fields" rejection. Raises
        # MessageNormalizationError (handled by the caller) when a role cannot
        # be determined.
        messages = normalize_messages_for_upstream(messages)

        # Build the strict upstream payload. CodeBuddy only supports streaming,
        # so stream is always True upstream; the client's stream preference is
        # honored separately by the router (aggregate vs pass-through).
        payload: Dict[str, Any] = {
            "model": mapped_model,
            "messages": messages,
            "stream": True,
        }
        # Forward tools / tool_choice only when the client actually sent them.
        if source.get("tools"):
            payload["tools"] = source["tools"]
        if source.get("tool_choice") is not None:
            payload["tool_choice"] = source["tool_choice"]

        dropped_fields = sorted(
            key for key in source.keys()
            if key not in RequestProcessor._UPSTREAM_ALLOWLIST
        )

        prep_info = {
            "system_prompt_sanitized": system_prompt_sanitized,
            "requested_model": requested_model if isinstance(requested_model, str) else "unknown",
            "mapped_model": mapped_model,
            "mapping_source": mapping_source,
            "dropped_fields": dropped_fields,
        }
        return payload, prep_info
    
    @staticmethod
    def validate_request(request_body: Dict[str, Any]) -> None:
        """Validate request structure.

        Only structural invariants are enforced here: the body must be an
        object with a non-empty ``messages`` array of objects. Role/content
        completeness is intentionally NOT enforced at this point: valid
        OpenAI/Anthropic tool-use turns legitimately omit ``content`` (assistant
        messages carrying ``tool_calls``) or ``role`` (tool results identified
        by ``tool_call_id``). Those fields are guaranteed later by
        ``normalize_messages_for_upstream`` immediately before the upstream call,
        which repairs what it can and returns an indexed 400 for what it cannot.
        """
        if not isinstance(request_body, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object")

        messages = request_body.get("messages")
        if not messages or not isinstance(messages, list):
            raise HTTPException(status_code=400, detail="Messages field is required and must be an array")

        # Validate message container type only; field-level normalization runs
        # later against the final transformed messages.
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                raise HTTPException(status_code=400, detail=f"Message {i} must be an object")

@dataclass(repr=False)
class ResolvedCredential:
    """Unified upstream credential; the raw token must not be logged or serialized."""

    bearer_token: str
    user_id: Optional[str]
    source: str
    key_id: Optional[str] = None


class CredentialManager:
    """Resolve legacy credentials or TXT API keys according to the auth mode."""

    @staticmethod
    def get_legacy_credential() -> Optional[ResolvedCredential]:
        try:
            credential = codebuddy_token_manager.get_next_credential()
        except Exception:
            logger.exception("Failed to select a legacy CodeBuddy credential")
            return None
        if not credential or not credential.get("bearer_token"):
            return None
        return ResolvedCredential(
            bearer_token=credential["bearer_token"],
            user_id=credential.get("user_id"),
            source="credentials",
        )

    @staticmethod
    async def get_api_key(excluded_ids: Set[str]) -> Optional[ResolvedCredential]:
        selection = await codebuddy_api_key_manager.acquire(excluded_ids)
        if selection is None:
            return None
        return ResolvedCredential(
            bearer_token=selection.key,
            user_id=None,
            source="api_key_file",
            key_id=selection.key_id,
        )

    @staticmethod
    async def resolve_source() -> tuple[str, int]:
        from config import get_codebuddy_auth_mode

        mode = get_codebuddy_auth_mode()
        if mode == "credentials":
            return "credentials", 1

        eligible = await codebuddy_api_key_manager.eligible_count()
        if eligible > 0:
            return "api_key_file", eligible
        if mode == "auto":
            return "credentials", 1

        total = await codebuddy_api_key_manager.total_count()
        if total == 0:
            raise ApiKeyConfigurationError(
                "No API keys are configured in CODEBUDDY_API_KEYS_FILE"
            )
        raise ApiKeyConfigurationError("No API keys are currently available")


def openai_error_response(
    message: str, error_type: str, code: str, status_code: int
) -> JSONResponse:
    """Construct an OpenAI-compatible error that excludes sensitive upstream information."""
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "message": message,
                "type": error_type,
                "code": code,
            }
        },
    )


async def record_attempt_error(credential: ResolvedCredential, error: UpstreamAttemptError) -> None:
    if credential.source != "api_key_file" or credential.key_id is None:
        return
    if error.kind == "invalid":
        await codebuddy_api_key_manager.mark_invalid(credential.key_id)
    elif error.kind == "cooldown":
        await codebuddy_api_key_manager.mark_cooldown(
            credential.key_id, error.status_code
        )
    else:
        await codebuddy_api_key_manager.mark_transient_error(
            credential.key_id, error.code
        )

# --- API Endpoints ---

@router.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    x_conversation_id: Optional[str] = Header(None, alias="X-Conversation-ID"),
    x_conversation_request_id: Optional[str] = Header(None, alias="X-Conversation-Request-ID"),
    x_conversation_message_id: Optional[str] = Header(None, alias="X-Conversation-Message-ID"),
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID"),
    auth_context: ClientAuthContext = Depends(authenticate_inference)
):
    """CodeBuddy V1 chat completions API, supporting relay and per-request passthrough."""
    try:
        request_body = await request.json()
    except Exception:
        return openai_error_response(
            "Invalid JSON request body", "invalid_request_error", "invalid_json", 400
        )

    try:
        RequestProcessor.validate_request(request_body)
    except HTTPException as exc:
        return openai_error_response(
            str(exc.detail), "invalid_request_error", "invalid_request", exc.status_code
        )

    passthrough_credential: Optional[ResolvedCredential] = None
    if auth_context.mode == "passthrough":
        if not auth_context.passthrough_key:
            return openai_error_response(
                "A Bearer API key is required",
                "authentication_error",
                "missing_api_key",
                401,
            )
        source, max_attempts = "passthrough", 1
        passthrough_credential = ResolvedCredential(
            bearer_token=auth_context.passthrough_key,
            user_id=None,
            source="passthrough",
        )
    else:
        try:
            source, max_attempts = await CredentialManager.resolve_source()
        except (ApiKeyConfigurationError, ValueError):
            logger.error("CodeBuddy API key file authentication is not configured correctly")
            return openai_error_response(
                "Upstream API key file is empty, unavailable, or invalid",
                "configuration_error",
                "api_key_file_unavailable",
                503,
            )

    try:
        payload, prep_info = RequestProcessor.prepare_payload(request_body)
    except UnknownModelError as exc:
        logger.info(
            "Rejecting unknown requested model (policy=reject): requested_present=%s",
            bool(exc.requested_model),
        )
        return openai_error_response(
            f"Unknown model: {exc.requested_model}",
            "invalid_request_error",
            "unknown_model",
            400,
        )
    except MessageNormalizationError as exc:
        # A message reached the upstream boundary without a determinable role.
        # Reject locally with the offending index instead of forwarding malformed
        # data that CodeBuddy rejects with a generic "must have 'role' and
        # 'content'" error.
        logger.info(
            "Rejecting malformed upstream message: index=%d reason=%s",
            exc.index,
            exc.reason,
        )
        return openai_error_response(
            f"Message {exc.index} is malformed: {exc.reason}",
            "invalid_request_error",
            "invalid_message",
            400,
        )
    usage_stats_manager.record_model_usage(payload.get("model", "unknown"))
    service = CodeBuddyStreamService()
    client_wants_stream = bool(request_body.get("stream", False)) if isinstance(request_body, dict) else False
    excluded_ids: Set[str] = set()
    last_error: Optional[UpstreamAttemptError] = None

    try:
        request_profile = get_codebuddy_request_profile()
    except ValueError:
        return openai_error_response(
            "Upstream request profile is misconfigured",
            "configuration_error",
            "request_profile_invalid",
            500,
        )

    for _attempt in range(max_attempts):
        if source == "passthrough":
            credential = passthrough_credential
        elif source == "api_key_file":
            credential = await CredentialManager.get_api_key(excluded_ids)
        else:
            credential = CredentialManager.get_legacy_credential()

        if credential is None:
            break
        if credential.key_id is not None:
            excluded_ids.add(credential.key_id)

        try:
            upstream_key_header = get_upstream_api_key_header()
        except ValueError:
            return openai_error_response(
                "Upstream API key header mode is misconfigured",
                "configuration_error",
                "upstream_header_mode_invalid",
                500,
            )

        key_fingerprint = _key_fingerprint(credential.bearer_token)

        # Safe diagnostics: metadata only, never key/prompt/user content.
        _log_request_diagnostics(
            payload=payload,
            prep_info=prep_info,
            client_wants_stream=client_wants_stream,
            request_profile=request_profile,
            key_fingerprint=key_fingerprint,
            outcome="prepared",
        )

        headers = codebuddy_api_client.generate_codebuddy_headers(
            bearer_token=credential.bearer_token,
            user_id=credential.user_id,
            conversation_id=x_conversation_id,
            conversation_request_id=x_conversation_request_id,
            conversation_message_id=x_conversation_message_id,
            request_id=x_request_id,
            api_key_header=upstream_key_header,
            profile=request_profile,
        )

        try:
            if client_wants_stream:
                result = await service.open_stream_response(
                    payload, headers, credential.key_id
                )
            else:
                result = await service.handle_non_stream_response(payload, headers)
            if credential.key_id is not None:
                await codebuddy_api_key_manager.mark_success(credential.key_id)
            # For the non-stream path we have the aggregated upstream response
            # and can log the model CodeBuddy actually reported. It is logged
            # separately and never used to rewrite the response model.
            upstream_response_model = (
                result.get("model") if isinstance(result, dict) else None
            )
            _log_request_diagnostics(
                payload=payload,
                prep_info=prep_info,
                client_wants_stream=client_wants_stream,
                request_profile=request_profile,
                key_fingerprint=key_fingerprint,
                outcome="success",
                upstream_response_model=upstream_response_model,
            )
            return result
        except CodeBuddyModerationError:
            # The key is valid and the request reached upstream; moderation is
            # not a key failure. Mark success (no failover) and return a clear
            # OpenAI-compatible content_filter error. Streaming moderation is
            # already handled inside open_stream_response before this point.
            if credential.key_id is not None:
                await codebuddy_api_key_manager.mark_success(credential.key_id)
            logger.info("CodeBuddy moderation rejection detected (non-stream)")
            return JSONResponse(
                status_code=400,
                headers={"X-CodeBuddy-Moderation": "true"},
                content={
                    "error": {
                        "message": MODERATION_MESSAGE,
                        "type": "content_filter",
                        "param": None,
                        "code": "codebuddy_content_filter",
                    }
                },
            )
        except UpstreamAttemptError as error:
            last_error = error
            await record_attempt_error(credential, error)
            logger.warning(
                "CodeBuddy upstream attempt failed: source=%s code=%s status=%d",
                credential.source,
                error.code,
                error.status_code,
            )
            _log_request_diagnostics(
                payload=payload,
                prep_info=prep_info,
                client_wants_stream=client_wants_stream,
                request_profile=request_profile,
                key_fingerprint=key_fingerprint,
                outcome=f"failed:{error.code}",
            )
            if source != "api_key_file" or error.kind == "fatal":
                break

    if source == "credentials" and last_error is None:
        return openai_error_response(
            "No valid CodeBuddy credentials are available",
            "authentication_error",
            "credentials_unavailable",
            401,
        )

    if source == "passthrough" and last_error is not None:
        if last_error.status_code == 401:
            return openai_error_response(
                "Upstream CodeBuddy rejected the supplied API key",
                "authentication_error",
                "upstream_api_key_rejected",
                401,
            )
        error_type = (
            "rate_limit_error"
            if last_error.status_code == 429
            else "permission_error"
            if last_error.status_code == 403
            else "upstream_error"
        )
        return openai_error_response(
            "Upstream CodeBuddy request failed",
            error_type,
            last_error.code,
            last_error.status_code,
        )

    status_code = (
        502
        if source == "credentials" and last_error and last_error.status_code >= 500
        else last_error.status_code
        if source == "credentials" and last_error
        else 502
    )
    code = last_error.code if source == "credentials" and last_error else "upstream_keys_exhausted"
    return openai_error_response(
        "Upstream CodeBuddy request failed",
        "upstream_error",
        code,
        status_code,
    )

@router.get("/v1/models")
async def list_v1_models(
    _auth_context: ClientAuthContext = Depends(authenticate_inference),
):
    """Get the list of CodeBuddy V1 models"""
    try:
        return {
            "object": "list",
            "data": [{
                "id": model,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "codebuddy"
            } for model in get_available_models_list()]
        }
        
    except Exception as e:
        logger.error(f"Error getting V1 model list: {e}")
        raise HTTPException(status_code=500, detail="Failed to get model list")

@router.get("/v1/api-keys/status", summary="Get upstream API key pool status")
async def get_api_keys_status(_token: str = Depends(authenticate_admin)):
    """Return a safe status containing only masked keys and runtime statistics."""
    from config import get_codebuddy_auth_mode

    status = await codebuddy_api_key_manager.get_status()
    return {"auth_mode": get_codebuddy_auth_mode(), **status}


@router.post("/v1/api-keys/reload", summary="Reload upstream API keys")
async def reload_api_keys(_token: str = Depends(authenticate_admin)):
    """Force a re-read of the TXT API key file without restarting the service."""
    result = await codebuddy_api_key_manager.reload()
    return {"message": "API key file reloaded", **result}


@router.get("/v1/credentials", summary="List all available credentials")
async def list_credentials(_token: str = Depends(authenticate_admin)):
    """List detailed information for all available credentials, including expiration status"""
    try:
        credentials_info = codebuddy_token_manager.get_credentials_info()
        safe_credentials = []
        
        credentials = codebuddy_token_manager.get_all_credentials()
        
        for info in credentials_info:
            bearer_token = credentials[info['index']].get("bearer_token", "") if info['index'] < len(credentials) else ""
            
            # Format the time display
            if info['time_remaining'] is not None and info['time_remaining'] > 0:
                days, remainder = divmod(info['time_remaining'], 86400)
                hours, remainder = divmod(remainder, 3600)
                minutes = remainder // 60
                time_remaining_str = f"{days}d {hours}h" if days > 0 else f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
            else:
                time_remaining_str = "Expired" if info['time_remaining'] is not None else "Unknown"
            
            safe_credentials.append({
                **info,  # Expand all original info
                "time_remaining_str": time_remaining_str,
                "has_token": bool(bearer_token),
                "token_preview": f"{bearer_token[:10]}...{bearer_token[-4:]}" if len(bearer_token) > 14 else "Invalid Token"
            })
        
        return {"credentials": safe_credentials}
        
    except Exception as e:
        logger.error(f"Failed to get credential list: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials", summary="Add a new credential")
async def add_credential(
    request: Request,
    _token: str = Depends(authenticate_admin)
):
    """Add a new authentication credential"""
    try:
        data = await request.json()
        if not data.get("bearer_token"):
            raise HTTPException(status_code=422, detail="bearer_token is required")

        success = codebuddy_token_manager.add_credential(
            data.get("bearer_token"),
            data.get("user_id"),
            data.get("filename")
        )
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save credential file")
        
        return {"message": "Credential added successfully"}

    except Exception as e:
        logger.error(f"Failed to add credential: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/select", summary="Manually select a credential")
async def select_credential(
    request: Request,
    _token: str = Depends(authenticate_admin)
):
    """Manually select the specified credential"""
    try:
        data = await request.json()
        index = data.get("index")
        if index is None:
            raise HTTPException(status_code=422, detail="index is required")

        if not codebuddy_token_manager.set_manual_credential(index):
            raise HTTPException(status_code=400, detail="Invalid credential index")
        
        return {"message": f"Credential #{index + 1} selected successfully"}

    except Exception as e:
        logger.error(f"Failed to select credential: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/auto", summary="Resume automatic credential rotation")
async def resume_auto_rotation(_token: str = Depends(authenticate_admin)):
    """Resume automatic credential rotation"""
    try:
        codebuddy_token_manager.clear_manual_selection()
        return {"message": "Resumed automatic credential rotation"}

    except Exception as e:
        logger.error(f"Failed to resume automatic rotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/toggle-rotation", summary="Toggle automatic credential rotation")
async def toggle_auto_rotation(_token: str = Depends(authenticate_admin)):
    """Toggle the automatic rotation switch"""
    try:
        is_enabled = codebuddy_token_manager.toggle_auto_rotation()
        status = "enabled" if is_enabled else "disabled"
        message = f"Auto rotation {status}"
        return {
            "message": message,
            "auto_rotation_enabled": is_enabled
        }

    except Exception as e:
        logger.error(f"Failed to toggle automatic rotation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/v1/credentials/current", summary="Get current credential info")
async def get_current_credential(_token: str = Depends(authenticate_admin)):
    """Get information about the currently used credential"""
    try:
        info = codebuddy_token_manager.get_current_credential_info()
        return info

    except Exception as e:
        logger.error(f"Failed to get current credential info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/v1/credentials/delete", summary="Delete a credential by index")
async def delete_credential(request: Request, _token: str = Depends(authenticate_admin)):
    """Delete a credential file (by index) and remove it from the list"""
    try:
        data = await request.json()
        index = data.get("index")
        if index is None or not isinstance(index, int):
            raise HTTPException(status_code=422, detail="Valid integer index is required")

        if not codebuddy_token_manager.delete_credential_by_index(index):
            raise HTTPException(status_code=400, detail="Invalid index or failed to delete credential")

        return {"message": f"Credential #{index + 1} deleted successfully"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete credential: {e}")
        raise HTTPException(status_code=500, detail=str(e))
