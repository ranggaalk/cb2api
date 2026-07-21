"""CodeBuddyAdapterV2 orchestrator (spec §11, §13, §14, §15, §16).

Owns the full request lifecycle for ``/v1/chat/completions`` when
``CODEBUDDY_ADAPTER_VERSION=v2``:

  1. resolve the request-local upstream credential (passthrough key, or a legacy
     credential for relay mode);
  2. normalize + validate messages, resolve the model, sanitize tool schemas,
     build the strict upstream payload;
  3. acquire a concurrency slot (bounded, with a queue timeout);
  4. open the upstream stream with at most one pre-first-event retry;
  5. stream through incrementally, or aggregate for a non-streaming client;
  6. on downstream disconnect, cancel upstream and release the slot — no ghost
     requests, no retry after the stream has started.

Telemetry is emitted per stage with safe metadata only (spec §15). Nothing here
logs prompts, file contents, tool argument values, or raw keys.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

import httpx
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse

from src.auth import ClientAuthContext
from src.codebuddy_message_sanitizer import (
    is_codebuddy_moderation_response,
    sanitize_messages,
)
from src.keyword_replacer import apply_keyword_replacement_to_system_message

from .config import AdapterSettings, load_adapter_settings
from .errors import (
    AdapterError,
    ModerationError,
    QueueTimeoutError,
    UnknownModelError,
    UpstreamAuthError,
    UpstreamNetworkError,
    UpstreamRetryableError,
)
from .errors import ConnectTimeoutError
from .message_normalizer import (
    convert_anthropic_messages_to_openai,
    normalize_messages_for_upstream,
)
from .models import TelemetryContext, UpstreamCredential
from .request_mapper import build_payload, dropped_fields, resolve_model
from . import response_mapper
from .response_mapper import sse_chunk, sse_done
from .stream_decoder import (
    SSELineBuffer,
    StreamAggregator,
    _extract_delta_content,
    parse_sse_data_line,
)
from .tool_schema_adapter import sanitize_tools
from .transport import UpstreamTransport, key_fingerprint
from .validation import validate_conversation

logger = logging.getLogger(__name__)

# SSE response headers (spec §11): disable caching and proxy buffering so the
# stream is delivered incrementally.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "*",
}

# How many characters of assistant content to buffer while deciding whether a
# streaming reply is a moderation refusal (the refusal is short).
_MODERATION_BUFFER_CHARS = 200

_MODERATION_MESSAGE = (
    "CodeBuddy rejected the request through its content moderation system."
)


class CodeBuddyAdapterV2:
    """Stateless-per-request CodeBuddy provider adapter."""

    def __init__(self, transport: Optional[UpstreamTransport] = None) -> None:
        self.transport = transport or UpstreamTransport()
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._semaphore_limit: Optional[int] = None

    # --- Lifecycle -------------------------------------------------------

    async def startup(self) -> None:
        """Warm the shared HTTP client at application startup."""
        try:
            settings = load_adapter_settings()
        except Exception:
            logger.exception("CodeBuddyAdapterV2 startup: settings unavailable")
            return
        await self.transport.get_client(settings)

    async def shutdown(self) -> None:
        """Close the shared HTTP client at application shutdown."""
        await self.transport.aclose()

    def _get_semaphore(self, settings: AdapterSettings) -> asyncio.Semaphore:
        """Return the concurrency semaphore, (re)building it if the limit changed."""
        limit = settings.max_concurrent_upstream_requests
        if self._semaphore is None or self._semaphore_limit != limit:
            self._semaphore = asyncio.Semaphore(limit)
            self._semaphore_limit = limit
        return self._semaphore

    # --- Telemetry -------------------------------------------------------

    @staticmethod
    def _log_stage(stage: str, telemetry: TelemetryContext, **extra: Any) -> None:
        fields = telemetry.as_log_fields()
        fields.update(extra)
        logger.info(
            "codebuddy_v2 stage=%s %s",
            stage,
            " ".join(f"{k}={v}" for k, v in fields.items()),
        )

    # --- Entry point -----------------------------------------------------

    async def chat_completion(
        self,
        *,
        request: Request,
        request_body: Any,
        auth_context: ClientAuthContext,
        conversation_id: Optional[str] = None,
        conversation_request_id: Optional[str] = None,
        conversation_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ):
        """Handle one chat-completions request end to end."""
        telemetry = TelemetryContext(request_id=request_id or uuid.uuid4().hex)
        self._log_stage("request_received", telemetry)

        try:
            settings = load_adapter_settings()
        except ValueError:
            return self._error_json(
                "Upstream configuration is invalid",
                "configuration_error",
                "configuration_invalid",
                500,
            )

        # Structural request validation (object with non-empty messages array).
        if not isinstance(request_body, dict):
            return self._error_json(
                "Request body must be a JSON object",
                "invalid_request_error",
                "invalid_request",
                400,
            )
        messages_in = request_body.get("messages")
        if not messages_in or not isinstance(messages_in, list):
            return self._error_json(
                "Messages field is required and must be an array",
                "invalid_request_error",
                "invalid_request",
                400,
            )

        # Resolve the request-local upstream credential.
        credential, cred_error = await self._resolve_credential(auth_context, settings)
        if cred_error is not None:
            return cred_error
        telemetry.key_fingerprint = credential.fingerprint

        client_wants_stream = bool(request_body.get("stream", False))
        telemetry.client_wants_stream = client_wants_stream

        # Prepare the payload (model resolve → normalize → validate → tools).
        try:
            payload, prepared_telemetry = self._prepare_payload(
                request_body, settings, telemetry
            )
        except AdapterError as exc:
            self._log_stage("request_failed", telemetry, code=exc.code)
            return self._error_from_adapter(exc)

        self._log_stage("request_prepared", telemetry)

        # Acquire a concurrency slot (bounded), honoring the queue timeout.
        semaphore = self._get_semaphore(settings)
        self._log_stage("upstream_slot_wait_start", telemetry)
        try:
            await asyncio.wait_for(
                semaphore.acquire(), timeout=settings.upstream_queue_timeout
            )
        except asyncio.TimeoutError:
            self._log_stage("request_failed", telemetry, code="upstream_queue_timeout")
            return self._error_from_adapter(QueueTimeoutError())
        self._log_stage("upstream_slot_acquired", telemetry)

        slot_owned_by_caller = True
        try:
            headers = self.transport.build_headers(
                bearer_token=credential.bearer_token,
                settings=settings,
                user_id=credential.user_id,
                conversation_id=conversation_id,
                conversation_request_id=conversation_request_id,
                conversation_message_id=conversation_message_id,
                request_id=request_id,
            )

            self._log_stage("upstream_send_start", telemetry)
            response = await self._open_with_retry(settings, payload, headers, telemetry)
            self._log_stage("upstream_headers_received", telemetry)

            if client_wants_stream:
                # Hand slot + response ownership to the streaming generator,
                # which releases the slot and closes the response in its finally.
                slot_owned_by_caller = False
                generator = self._stream_response(
                    request=request,
                    response=response,
                    settings=settings,
                    telemetry=telemetry,
                    semaphore=semaphore,
                )
                return StreamingResponse(
                    generator, media_type="text/event-stream", headers=SSE_HEADERS
                )

            # Non-streaming: aggregate, then release the slot in finally.
            result = await self._aggregate_response(
                response=response, settings=settings, telemetry=telemetry
            )
            if isinstance(result, JSONResponse):
                return result
            self._log_stage("stream_finished", telemetry, mode="aggregate")
            return JSONResponse(content=result)
        except ModerationError:
            self._log_stage("request_failed", telemetry, code="content_filter")
            return JSONResponse(
                status_code=400,
                headers={"X-CodeBuddy-Moderation": "true"},
                content={
                    "error": {
                        "message": _MODERATION_MESSAGE,
                        "type": "content_filter",
                        "param": None,
                        "code": "codebuddy_content_filter",
                    }
                },
            )
        except AdapterError as exc:
            self._log_stage("request_failed", telemetry, code=exc.code)
            return self._error_from_adapter(exc)
        finally:
            if slot_owned_by_caller:
                semaphore.release()

    # --- Credential resolution ------------------------------------------

    async def _resolve_credential(
        self, auth_context: ClientAuthContext, settings: AdapterSettings
    ):
        """Resolve the upstream credential for this request.

        Passthrough mode uses the caller's Bearer token verbatim, only for this
        request (spec §2) — no storage, no rotation, no failover. Relay/legacy
        modes reuse the existing credential managers via a lazy import so this
        adapter stays isolated while remaining backward compatible.

        Returns ``(credential, None)`` on success or ``(None, JSONResponse)`` on
        failure.
        """
        if auth_context.mode == "passthrough":
            if not auth_context.passthrough_key:
                return None, self._error_json(
                    "A Bearer API key is required",
                    "authentication_error",
                    "missing_api_key",
                    401,
                )
            token = auth_context.passthrough_key
            return (
                UpstreamCredential(
                    bearer_token=token,
                    fingerprint=key_fingerprint(token),
                    source="passthrough",
                ),
                None,
            )

        # Relay / legacy credential resolution (backward compatibility).
        try:
            from src.codebuddy_router import CredentialManager
            from src.codebuddy_api_key_manager import ApiKeyConfigurationError

            source, _max_attempts = await CredentialManager.resolve_source()
            if source == "api_key_file":
                resolved = await CredentialManager.get_api_key(set())
            else:
                resolved = CredentialManager.get_legacy_credential()
        except Exception:
            logger.exception("CodeBuddyAdapterV2 credential resolution failed")
            return None, self._error_json(
                "Upstream API key is unavailable",
                "configuration_error",
                "api_key_file_unavailable",
                503,
            )

        if resolved is None or not resolved.bearer_token:
            return None, self._error_json(
                "No valid CodeBuddy credentials are available",
                "authentication_error",
                "credentials_unavailable",
                401,
            )
        return (
            UpstreamCredential(
                bearer_token=resolved.bearer_token,
                fingerprint=key_fingerprint(resolved.bearer_token),
                source=resolved.source,
                user_id=resolved.user_id,
                key_id=resolved.key_id,
            ),
            None,
        )

    # --- Payload preparation --------------------------------------------

    def _prepare_payload(
        self,
        request_body: Dict[str, Any],
        settings: AdapterSettings,
        telemetry: TelemetryContext,
    ):
        """Resolve model, normalize + validate messages, and build the payload."""
        # Model resolution (may raise UnknownModelError → local 400).
        resolved = resolve_model(request_body.get("model"), settings)
        telemetry.requested_model = resolved.requested or "unknown"
        telemetry.mapped_model = resolved.mapped
        telemetry.mapping_source = resolved.source

        raw_messages = request_body.get("messages", []) or []
        telemetry.message_count = len(raw_messages)

        # 1) Convert Anthropic tool blocks → OpenAI shape.
        messages = convert_anthropic_messages_to_openai(raw_messages)

        # 2) Sanitize agent system prompts (system messages only). Per spec §4,
        # sanitization is AUTO-SKIPPED for agentic/tool requests: when the client
        # sends tools (Claude Code and other coding agents), the system prompt
        # carries the tool, permission, and agent-loop instructions the model
        # needs, and replacing it with a generic prompt both breaks the agent and
        # is a likely cause of upstream rejection. The config flag only governs
        # simple web-chat requests that have no tools.
        has_tools = bool(request_body.get("tools"))
        messages, _sanitized = sanitize_messages(
            messages,
            enabled=settings.sanitize_agent_prompt and not has_tools,
            max_system_prompt_length=settings.max_system_prompt_length,
        )

        # 2b) CodeBuddy rejects a conversation with fewer than two messages
        # (a lone user turn returns HTTP 400). Prepend a short default system
        # message ONLY when there is no system message already and the single
        # message is a user turn, mirroring the legacy handler. This never
        # reorders or duplicates existing messages.
        has_system = any(
            isinstance(m, dict) and m.get("role") == "system" for m in messages
        )
        if (
            not has_system
            and len(messages) == 1
            and isinstance(messages[0], dict)
            and messages[0].get("role") == "user"
        ):
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant. Reply in the same language as the user.",
                }
            ] + messages

        # 3) Keyword replacement on system messages only.
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                msg["content"] = apply_keyword_replacement_to_system_message(
                    msg.get("content")
                )

        # 4) Guarantee role + content on every message.
        messages = normalize_messages_for_upstream(messages)
        self._log_stage("request_normalized", telemetry)

        # 5) Validate the fully-converted conversation (→ local 400).
        validate_conversation(messages)
        self._log_stage("request_validated", telemetry)

        # 6) Sanitize tool schemas; forward tools only when present.
        tools = sanitize_tools(request_body.get("tools")) if request_body.get("tools") else None

        payload = build_payload(request_body, resolved.mapped, messages)
        if tools:
            payload["tools"] = tools
        if request_body.get("tool_choice") is not None:
            payload["tool_choice"] = request_body["tool_choice"]

        # Telemetry: tool + content-size metrics and large-request warning.
        telemetry.tool_count = len(tools or []) + self._count_tool_messages(messages)
        telemetry.total_content_length = self._total_content_length(messages)
        self._maybe_warn_large_request(settings, telemetry, dropped_fields(request_body))

        return payload, telemetry

    @staticmethod
    def _count_tool_messages(messages: List[Dict[str, Any]]) -> int:
        return sum(
            1
            for m in messages
            if isinstance(m, dict) and (m.get("role") == "tool" or m.get("tool_calls"))
        )

    @staticmethod
    def _total_content_length(messages: List[Dict[str, Any]]) -> int:
        total = 0
        for m in messages:
            if not isinstance(m, dict):
                continue
            content = m.get("content")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(str(block.get("text", "")))
        return total

    def _maybe_warn_large_request(
        self,
        settings: AdapterSettings,
        telemetry: TelemetryContext,
        dropped: List[str],
    ) -> None:
        """Log a warning for large agentic requests without ever truncating."""
        if (
            telemetry.total_content_length > settings.warn_total_content_length
            or telemetry.message_count > settings.warn_message_count
            or telemetry.tool_count > settings.warn_tool_count
        ):
            telemetry.large_agentic_request = True
            logger.warning(
                "codebuddy_v2 large_agentic_request=true request_id=%s message_count=%d "
                "tool_count=%d total_content_length=%d dropped_fields=[%s]",
                telemetry.request_id,
                telemetry.message_count,
                telemetry.tool_count,
                telemetry.total_content_length,
                ",".join(dropped),
            )

    # --- Upstream open with single pre-first-event retry ----------------

    async def _open_with_retry(
        self,
        settings: AdapterSettings,
        payload: Dict[str, Any],
        headers: Dict[str, str],
        telemetry: TelemetryContext,
    ) -> httpx.Response:
        """Open the upstream stream, retrying at most once before any event.

        Only transient pre-event failures are retried (connect timeout, connect
        reset/network error, 408/429/5xx). The same credential is reused — no key
        rotation/fallback (9Router owns fallback, spec §14). Fatal statuses
        (400/401/403/404/422) are never retried.
        """
        attempts = 0
        while True:
            try:
                return await self.transport.open_stream(
                    settings=settings, payload=payload, headers=headers
                )
            except (ConnectTimeoutError, UpstreamNetworkError, UpstreamRetryableError) as exc:
                attempts += 1
                if attempts > 1:
                    raise
                logger.warning(
                    "codebuddy_v2 retrying upstream once before first event: code=%s",
                    exc.code,
                )
                continue

    # --- Streaming path --------------------------------------------------

    async def _stream_response(
        self,
        *,
        request: Request,
        response: httpx.Response,
        settings: AdapterSettings,
        telemetry: TelemetryContext,
        semaphore: asyncio.Semaphore,
    ) -> AsyncIterator[str]:
        """Yield OpenAI SSE frames, with moderation pre-buffering and cancellation.

        A downstream disconnect (client cancel) raises ``CancelledError``/
        ``GeneratorExit`` into this generator; the ``finally`` closes the upstream
        response and releases the concurrency slot so no ghost request continues.
        Once the first content/tool event is forwarded, the stream is never
        retried.
        """
        line_buffer = SSELineBuffer()
        accumulated_content = ""
        moderation_window_open = True
        buffered_frames: List[str] = []
        first_event_seen = False
        released = False

        def release() -> None:
            nonlocal released
            if not released:
                released = True
                semaphore.release()

        try:
            async for chunk in self.transport.iter_lines_with_idle_timeout(
                response,
                first_chunk_timeout=settings.first_chunk_timeout,
                idle_timeout=settings.stream_idle_timeout,
            ):
                if await request.is_disconnected():
                    self._log_stage("downstream_disconnected", telemetry)
                    break

                for line in line_buffer.feed(chunk):
                    frame = self._process_stream_line(line)
                    if frame is None:
                        continue

                    if not first_event_seen:
                        first_event_seen = True
                        self._log_stage("upstream_first_chunk", telemetry)

                    # Moderation detection window: buffer initial frames until we
                    # can tell a normal reply from a Mandarin refusal.
                    if moderation_window_open:
                        obj = parse_sse_data_line(line)
                        if obj is not None:
                            accumulated_content += _extract_delta_content(obj)
                        buffered_frames.append(frame)
                        if is_codebuddy_moderation_response(accumulated_content):
                            for item in self._moderation_frames(telemetry):
                                yield item
                            self._log_stage("stream_finished", telemetry, moderation=True)
                            return
                        if (
                            len(accumulated_content) >= _MODERATION_BUFFER_CHARS
                            or "[DONE]" in line
                        ):
                            moderation_window_open = False
                            for item in buffered_frames:
                                yield item
                            buffered_frames.clear()
                        continue

                    yield frame

            # Flush any buffered frames that never crossed the window threshold
            # (short, non-moderation responses) and any trailing partial line.
            if buffered_frames:
                for item in buffered_frames:
                    yield item
            tail = line_buffer.flush()
            if tail is not None:
                frame = self._process_stream_line(tail)
                if frame is not None:
                    yield frame
            # Ensure the stream is terminated exactly once.
            yield sse_done()
            self._log_stage("stream_finished", telemetry, mode="stream")
        except (asyncio.CancelledError, GeneratorExit):
            self._log_stage("upstream_cancelled", telemetry)
            raise
        except AdapterError as exc:
            # A timeout mid-stream: surface a safe SSE error frame.
            logger.warning("codebuddy_v2 stream error code=%s", exc.code)
            yield self._sse_error(exc.message, exc.code)
        except httpx.RequestError:
            logger.warning("codebuddy_v2 upstream stream interrupted")
            yield self._sse_error("Upstream stream interrupted", "upstream_stream_error")
        finally:
            await response.aclose()
            release()

    def _process_stream_line(self, line: str) -> Optional[str]:
        """Convert one upstream SSE line into a downstream OpenAI SSE frame.

        Comments/blank lines are dropped. ``[DONE]`` is suppressed here so the
        generator can emit exactly one terminator itself. Data frames have their
        tool-call IDs normalized to the OpenAI ``call_`` convention for
        consistency and are re-serialized.
        """
        stripped = line.strip()
        if not stripped or stripped.startswith(":"):
            return None
        if "[DONE]" in stripped:
            return None
        obj = parse_sse_data_line(line)
        if obj is None:
            return None
        self._normalize_chunk_tool_ids(obj)
        return sse_chunk(obj)

    @staticmethod
    def _normalize_chunk_tool_ids(obj: Dict[str, Any]) -> None:
        """Rewrite ``tooluse_`` tool-call IDs to ``call_`` in place (consistent)."""
        try:
            choices = obj.get("choices") or []
            if not choices:
                return
            delta = choices[0].get("delta") or {}
            tool_calls = delta.get("tool_calls")
            if not isinstance(tool_calls, list):
                return
            for tc in tool_calls:
                if isinstance(tc, dict) and tc.get("id"):
                    tc["id"] = response_mapper._tool_call_id_openai(tc["id"])
        except (AttributeError, IndexError, TypeError):
            return

    def _moderation_frames(self, telemetry: TelemetryContext) -> List[str]:
        created = self._now()
        chunk = response_mapper.moderation_chunk(
            response_id="chatcmpl-codebuddy-filter",
            created=created,
            message=_MODERATION_MESSAGE,
        )
        return [sse_chunk(chunk), sse_done()]

    @staticmethod
    def _sse_error(message: str, code: str) -> str:
        import json as _json

        return f'data: {_json.dumps({"error": {"message": message, "type": "stream_error", "code": code}}, ensure_ascii=False)}\n\n'

    # --- Non-streaming path ---------------------------------------------

    async def _aggregate_response(
        self,
        *,
        response: httpx.Response,
        settings: AdapterSettings,
        telemetry: TelemetryContext,
    ):
        """Consume the whole SSE stream and build one chat.completion object.

        Detects a moderation refusal in the aggregated content or the raw body
        and raises :class:`ModerationError` (mapped to a content_filter 400).
        """
        aggregator = StreamAggregator()
        line_buffer = SSELineBuffer()
        raw_text = ""
        first = True
        try:
            async for chunk in self.transport.iter_lines_with_idle_timeout(
                response,
                first_chunk_timeout=settings.first_chunk_timeout,
                idle_timeout=settings.stream_idle_timeout,
            ):
                if first:
                    first = False
                    self._log_stage("upstream_first_chunk", telemetry)
                raw_text += chunk
                for line in line_buffer.feed(chunk):
                    obj = parse_sse_data_line(line)
                    if obj is not None:
                        aggregator.process_chunk(obj)
            tail = line_buffer.flush()
            if tail is not None:
                obj = parse_sse_data_line(tail)
                if obj is not None:
                    aggregator.process_chunk(obj)
            content, tool_calls, finish_reason = aggregator.finalize()
        finally:
            await response.aclose()

        if is_codebuddy_moderation_response(content) or is_codebuddy_moderation_response(
            raw_text
        ):
            raise ModerationError()

        return response_mapper.build_non_stream_response(
            response_id=aggregator.id or f"chatcmpl-{uuid.uuid4().hex}",
            model=aggregator.model or telemetry.mapped_model,
            created=self._now(),
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=aggregator.usage,
            system_fingerprint=aggregator.system_fingerprint,
            reasoning_content=aggregator.reasoning_content,
        )

    # --- Error mapping ---------------------------------------------------

    @staticmethod
    def _now() -> int:
        return int(time.time())

    @staticmethod
    def _error_json(
        message: str, error_type: str, code: str, status_code: int
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content={"error": {"message": message, "type": error_type, "code": code}},
        )

    def _error_from_adapter(self, exc: AdapterError) -> JSONResponse:
        return self._error_json(exc.message, exc.error_type, exc.code, exc.status_code)
