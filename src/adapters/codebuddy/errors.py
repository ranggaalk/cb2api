"""Typed error hierarchy for CodeBuddyAdapterV2.

Every adapter failure maps to one of these so the router can translate it into
a precise OpenAI-compatible HTTP error. Crucially, the six upstream timeout
stages are distinct types (spec §12) so a failure is never collapsed into a
generic "fetch connect timeout". None of these carry raw upstream bodies, keys,
or message content — only safe codes and status numbers.
"""
from __future__ import annotations

from typing import Optional


class AdapterError(Exception):
    """Base class for every CodeBuddyAdapterV2 error.

    ``code`` is a stable machine-readable slug surfaced to the client;
    ``status_code`` is the HTTP status the router should return; ``message`` is a
    safe human-readable summary that never contains upstream bodies or secrets.
    """

    code: str = "adapter_error"
    status_code: int = 502
    error_type: str = "upstream_error"

    def __init__(self, message: Optional[str] = None):
        super().__init__(message or self.code)
        self.message = message or self.code


# --- Request-shaping errors (local HTTP 400, never reach upstream) ---


class UnknownModelError(AdapterError):
    """Requested model is unknown and CODEBUDDY_UNKNOWN_MODEL_POLICY=reject."""

    code = "unknown_model"
    status_code = 400
    error_type = "invalid_request_error"

    def __init__(self, requested_model: str):
        super().__init__(f"Unknown model: {requested_model}")
        self.requested_model = requested_model


class MessageNormalizationError(AdapterError):
    """A message cannot be given a valid role/content before upstream.

    Carries the offending message ``index`` so the router returns a local 400
    identifying exactly which message is malformed.
    """

    code = "invalid_message"
    status_code = 400
    error_type = "invalid_request_error"

    def __init__(self, index: int, reason: str):
        super().__init__(f"Message {index} is malformed: {reason}")
        self.index = index
        self.reason = reason


class InvalidToolConversationError(AdapterError):
    """The tool-call/tool-result structure is invalid (spec §7).

    Raised when validation of the fully-converted conversation fails: an orphan
    tool result, non-JSON function arguments, a leftover Anthropic block, etc.
    """

    code = "invalid_tool_conversation"
    status_code = 400
    error_type = "invalid_request_error"

    def __init__(self, index: int, reason: str):
        super().__init__(
            f"Invalid tool conversation structure at message {index}: {reason}"
        )
        self.index = index
        self.reason = reason


class InvalidToolArgumentsError(AdapterError):
    """A streamed tool call finished with arguments that are not valid JSON.

    Per spec §9 the adapter must NOT silently coerce invalid arguments to ``{}``;
    it raises this structured error instead.
    """

    code = "invalid_tool_arguments"
    status_code = 502
    error_type = "upstream_error"

    def __init__(self, tool_name: Optional[str] = None):
        super().__init__("Upstream tool call produced invalid JSON arguments")
        self.tool_name = tool_name


# --- Moderation (not an upstream failure; must not trigger key failover) ---


class ModerationError(AdapterError):
    """CodeBuddy rejected the request via its content moderation system."""

    code = "codebuddy_content_filter"
    status_code = 400
    error_type = "content_filter"


# --- Upstream transport errors ---


class UpstreamRejectedError(AdapterError):
    """Upstream returned a non-200 status that is not retryable."""

    def __init__(self, status_code: int, code: str = "upstream_request_rejected"):
        super().__init__("Upstream CodeBuddy request was rejected")
        self.status_code = status_code
        self.code = code


class UpstreamRetryableError(AdapterError):
    """Upstream returned a status eligible for a single pre-stream retry."""

    error_type = "upstream_error"

    def __init__(self, status_code: int, code: str = "upstream_server_error"):
        super().__init__("Upstream CodeBuddy request failed")
        self.status_code = status_code
        self.code = code


class UpstreamAuthError(AdapterError):
    """Upstream rejected the supplied API key (401)."""

    code = "upstream_api_key_rejected"
    status_code = 401
    error_type = "authentication_error"


# --- Distinct timeout stages (spec §12) ---


class UpstreamTimeoutError(AdapterError):
    """Base for the distinct upstream timeout stages."""

    error_type = "upstream_error"
    status_code = 504


class ConnectTimeoutError(UpstreamTimeoutError):
    code = "upstream_connect_timeout"


class PoolTimeoutError(UpstreamTimeoutError):
    code = "upstream_pool_timeout"


class WriteTimeoutError(UpstreamTimeoutError):
    code = "upstream_write_timeout"


class HeadersTimeoutError(UpstreamTimeoutError):
    code = "upstream_headers_timeout"


class FirstChunkTimeoutError(UpstreamTimeoutError):
    code = "upstream_first_chunk_timeout"


class StreamIdleTimeoutError(UpstreamTimeoutError):
    code = "upstream_stream_idle_timeout"


class QueueTimeoutError(UpstreamTimeoutError):
    """Timed out waiting for a concurrency slot before contacting upstream."""

    code = "upstream_queue_timeout"
    status_code = 503


class UpstreamNetworkError(UpstreamRetryableError):
    """A connection reset or other network error before the first event."""

    def __init__(self):
        super().__init__(status_code=502, code="upstream_network_error")
