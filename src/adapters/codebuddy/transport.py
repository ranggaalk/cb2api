"""Upstream transport for CodeBuddyAdapterV2 (spec §2, §11, §12, §14).

Owns the single shared ``httpx.AsyncClient`` and everything about talking to
CodeBuddy over the wire:

  * a process-wide pooled client created at startup and closed at shutdown —
    never one client per request (spec §11);
  * granular per-stage timeouts mapped to distinct error types so a failure is
    reported precisely (connect / pool / write / headers / first-chunk / idle),
    never a generic "connect timeout" (spec §12);
  * request-local upstream credential handling: the caller's key is used only
    for that request, never stored, and only its SHA-256 fingerprint is logged
    (spec §2);
  * true streaming via ``client.send(request, stream=True)`` — the body is never
    pre-read with ``aread()``/``.text``/``.json()`` before the caller consumes
    it (spec §11).

Header construction reuses the existing ``codebuddy_api_client`` so the web/cli
profile behavior stays identical to the legacy path.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
from typing import AsyncIterator, Dict, Optional

import httpx

from src.codebuddy_api_client import codebuddy_api_client

from .config import AdapterSettings
from .errors import (
    ConnectTimeoutError,
    FirstChunkTimeoutError,
    HeadersTimeoutError,
    PoolTimeoutError,
    StreamIdleTimeoutError,
    UpstreamAuthError,
    UpstreamNetworkError,
    UpstreamRejectedError,
    UpstreamRetryableError,
    WriteTimeoutError,
)

logger = logging.getLogger(__name__)


def key_fingerprint(token: Optional[str]) -> str:
    """Return a short, non-reversible SHA-256 fingerprint of an API key.

    Used only for safe diagnostics; the raw key is never logged.
    """
    if not token:
        return "none"
    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:8]


class UpstreamTransport:
    """Shared HTTP transport to the CodeBuddy upstream.

    A single instance is held by the adapter for the process lifetime. Timeouts
    are read from :class:`AdapterSettings` when the client is first built.
    """

    def __init__(self, ssl_verify: bool = False) -> None:
        self._client: Optional[httpx.AsyncClient] = None
        self._lock = asyncio.Lock()
        self._ssl_verify = ssl_verify

    async def get_client(self, settings: AdapterSettings) -> httpx.AsyncClient:
        """Return the shared client, creating it once under a lock."""
        if self._client is None:
            async with self._lock:
                if self._client is None:
                    self._client = httpx.AsyncClient(
                        verify=self._ssl_verify,
                        timeout=httpx.Timeout(
                            connect=settings.connect_timeout,
                            read=settings.headers_timeout,
                            write=settings.write_timeout,
                            pool=settings.pool_timeout,
                        ),
                        limits=httpx.Limits(
                            max_connections=100,
                            max_keepalive_connections=30,
                            keepalive_expiry=60.0,
                        ),
                    )
                    logger.info("CodeBuddyAdapterV2 shared HTTP client initialized")
        return self._client

    async def aclose(self) -> None:
        """Close the shared client (called at application shutdown)."""
        async with self._lock:
            if self._client is not None:
                await self._client.aclose()
                self._client = None
                logger.info("CodeBuddyAdapterV2 shared HTTP client closed")

    def build_headers(
        self,
        *,
        bearer_token: str,
        settings: AdapterSettings,
        user_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_request_id: Optional[str] = None,
        conversation_message_id: Optional[str] = None,
        request_id: Optional[str] = None,
    ) -> Dict[str, str]:
        """Build upstream headers, honoring the configured key header + profile."""
        return codebuddy_api_client.generate_codebuddy_headers(
            bearer_token=bearer_token,
            user_id=user_id,
            conversation_id=conversation_id,
            conversation_request_id=conversation_request_id,
            conversation_message_id=conversation_message_id,
            request_id=request_id,
            api_key_header=settings.upstream_api_key_header,
            profile=settings.request_profile,
        )

    @staticmethod
    def _classify_timeout(exc: httpx.TimeoutException):
        """Map an httpx timeout to a distinct adapter timeout error."""
        if isinstance(exc, httpx.ConnectTimeout):
            return ConnectTimeoutError()
        if isinstance(exc, httpx.PoolTimeout):
            return PoolTimeoutError()
        if isinstance(exc, httpx.WriteTimeout):
            return WriteTimeoutError()
        if isinstance(exc, httpx.ReadTimeout):
            # A read timeout while awaiting headers is a headers timeout; the
            # first-chunk/idle stages are enforced separately during streaming.
            return HeadersTimeoutError()
        return HeadersTimeoutError()

    @staticmethod
    def _classify_status(status_code: int):
        """Map a non-200 upstream status to a retryable/fatal adapter error.

        Retryable (a single pre-stream retry is allowed): 408, 429, 502, 503,
        504. Fatal (never retried): 400, 401, 403, 404, 422, and other 4xx.
        """
        if status_code == 401:
            return UpstreamAuthError()
        if status_code in (408, 429, 502, 503, 504):
            return UpstreamRetryableError(status_code, "upstream_temporarily_unavailable")
        if status_code >= 500:
            return UpstreamRetryableError(status_code, "upstream_server_error")
        return UpstreamRejectedError(status_code, "upstream_request_rejected")

    async def open_stream(
        self,
        *,
        settings: AdapterSettings,
        payload: dict,
        headers: Dict[str, str],
    ) -> httpx.Response:
        """Open a streaming upstream response and verify its status.

        Returns an open ``httpx.Response`` whose body has NOT been read; the
        caller must iterate it and ensure it is closed. Raises a distinct
        timeout/te transport error, or a status-classified error on non-200.
        """
        client = await self.get_client(settings)
        request = client.build_request(
            "POST", settings.chat_completions_url, json=payload, headers=headers
        )
        try:
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise self._classify_timeout(exc) from exc
        except httpx.RequestError as exc:
            raise UpstreamNetworkError() from exc

        if response.status_code != 200:
            status_code = response.status_code
            await response.aclose()
            raise self._classify_status(status_code)
        return response

    async def iter_lines_with_idle_timeout(
        self,
        response: httpx.Response,
        *,
        first_chunk_timeout: float,
        idle_timeout: float,
    ) -> AsyncIterator[str]:
        """Yield decoded text chunks, enforcing first-chunk and idle timeouts.

        The wait for the FIRST chunk uses ``first_chunk_timeout``; each
        subsequent gap uses ``idle_timeout``. These are distinct from the
        connect/headers timeouts so a stall after headers is reported as
        ``upstream_first_chunk_timeout`` / ``upstream_stream_idle_timeout``.
        """
        iterator = response.aiter_text(chunk_size=8192).__aiter__()
        first = True
        while True:
            timeout = first_chunk_timeout if first else idle_timeout
            try:
                chunk = await asyncio.wait_for(iterator.__anext__(), timeout=timeout)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:
                if first:
                    raise FirstChunkTimeoutError() from exc
                raise StreamIdleTimeoutError() from exc
            first = False
            if chunk:
                yield chunk
