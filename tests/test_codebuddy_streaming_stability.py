"""
Stability tests for the CodeBuddy relay: true streaming, SSE headers, shared
HTTP client reuse, upstream concurrency limiting (queue wait + 503), separated
timeouts (first-byte vs stream read), heartbeats, cancellation/slot release, and
no-retry-after-stream-start.

These lock in the latency/streaming/cancellation fixes so a future change cannot
silently reintroduce full-body buffering, a leaked concurrency slot, a single
global timeout, or a retry after the stream has started.

Uses mock upstream transports; no real CodeBuddy API key is required.
"""
import asyncio
import json

import httpx
import pytest

from src import auth, codebuddy_router
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def app():
    from fastapi import FastAPI

    from src import codebuddy_auth_router, settings_router

    application = FastAPI()
    application.include_router(codebuddy_router.router, prefix="/codebuddy")
    application.include_router(codebuddy_auth_router.router, prefix="/codebuddy")
    application.include_router(settings_router.router, prefix="/api")
    return application


@pytest.fixture
async def empty_pool(monkeypatch, tmp_path):
    path = tmp_path / "keys.txt"
    path.write_text("", encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(path), reload_interval=0)
    await manager.reload()
    monkeypatch.setattr(codebuddy_router, "codebuddy_api_key_manager", manager)
    return manager


@pytest.fixture(autouse=True)
def reset_semaphore(monkeypatch):
    """Reset the global upstream concurrency limiter between tests."""
    codebuddy_router._upstream_semaphore = None
    codebuddy_router._upstream_semaphore_size = 0
    yield
    codebuddy_router._upstream_semaphore = None
    codebuddy_router._upstream_semaphore_size = 0


def configure(monkeypatch, profile="web"):
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "passthrough")
    monkeypatch.setattr(auth, "get_server_password", lambda: RELAY_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: ADMIN_PASSWORD)
    monkeypatch.setattr(codebuddy_router, "get_upstream_api_key_header", lambda: "both")
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_request_profile", lambda: profile)
    monkeypatch.setattr(
        codebuddy_router.usage_stats_manager, "record_model_usage", lambda _m: None
    )


def install_upstream(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def get_client():
        return client

    monkeypatch.setattr(codebuddy_router, "get_http_client", get_client)
    return client


def install_transport(monkeypatch, transport):
    """Install a shared client backed by a custom async transport."""
    client = httpx.AsyncClient(transport=transport)

    async def get_client():
        return client

    monkeypatch.setattr(codebuddy_router, "get_http_client", get_client)
    return client


def sse_body(chunks, done=True):
    """Build an SSE ``httpx.Response`` from ``(delta, finish_reason)`` pairs."""
    parts = []
    for delta, finish in chunks:
        choice = {"index": 0, "delta": delta}
        if finish is not None:
            choice["finish_reason"] = finish
        obj = {
            "id": "chat-stab",
            "object": "chat.completion.chunk",
            "model": "auto-chat",
            "choices": [choice],
        }
        parts.append("data: " + json.dumps(obj, ensure_ascii=False))
    text = "\n\n".join(parts) + "\n\n"
    if done:
        text += "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


async def request(app, path, token=None, method="GET", json_body=None):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, headers=headers, json=json_body)


async def chat(app, token, stream=False, messages=None, tools=None):
    body = {
        "model": "auto-chat",
        "messages": messages or [{"role": "user", "content": "halo, siapa kamu"}],
        "stream": stream,
    }
    if tools is not None:
        body["tools"] = tools
    return await request(app, "/codebuddy/v1/chat/completions", token, "POST", body)


def parse_stream_events(text):
    events = []
    for raw in text.split("\n\n"):
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        events.append(json.loads(payload))
    return events


# --------------------------------------------------------------------------- #
# Custom transports for delay-based tests
# --------------------------------------------------------------------------- #


class _ChunkStream(httpx.AsyncByteStream):
    """Async byte stream that yields chunks with an inter-chunk delay."""

    def __init__(self, chunks, delay=0.0):
        self._chunks = chunks
        self._delay = delay

    async def __aiter__(self):
        for chunk in self._chunks:
            if self._delay:
                await asyncio.sleep(self._delay)
            yield chunk

    async def aclose(self):
        return None


class SlowFirstByteTransport(httpx.AsyncBaseTransport):
    """Delays returning the response object itself (simulates slow headers)."""

    def __init__(self, header_delay):
        self._header_delay = header_delay
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        await asyncio.sleep(self._header_delay)
        body = b"data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": "stop"}]}
        ).encode() + b"\n\ndata: [DONE]\n\n"
        return httpx.Response(
            200, stream=_ChunkStream([body]), headers={"content-type": "text/event-stream"}
        )


class SlowChunkTransport(httpx.AsyncBaseTransport):
    """Returns headers immediately but delays the first body chunk."""

    def __init__(self, first_chunk_delay):
        self._delay = first_chunk_delay
        self.calls = 0

    async def handle_async_request(self, request):
        self.calls += 1
        first = "data: " + json.dumps(
            {"choices": [{"index": 0, "delta": {"content": "hello"}}]}
        ) + "\n\n"
        second = "data: [DONE]\n\n"
        stream = _ChunkStream(
            [first.encode(), second.encode()], delay=self._delay
        )
        return httpx.Response(
            200, stream=stream, headers={"content-type": "text/event-stream"}
        )


# --------------------------------------------------------------------------- #
# 1. SSE headers: no-transform + X-Accel-Buffering: no
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sse_headers_disable_proxy_buffering(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body([({"content": "hi"}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "no-transform" in response.headers.get("cache-control", "")
    assert response.headers.get("x-accel-buffering") == "no"


# --------------------------------------------------------------------------- #
# 2. Streaming preserves multiple incremental chunks in order
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stream_preserves_incremental_chunks(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body(
            [
                ({"role": "assistant"}, None),
                ({"content": "A"}, None),
                ({"content": "B"}, None),
                ({"content": "C"}, None),
                ({}, "stop"),
            ]
        )

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    text = response.text
    assert text.count("[DONE]") == 1
    events = parse_stream_events(text)
    streamed = "".join(
        (e["choices"][0].get("delta", {}) or {}).get("content", "") for e in events
    )
    assert streamed == "ABC"


# --------------------------------------------------------------------------- #
# 3. Shared HTTP client is built once and reused
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_shared_http_client_is_reused(monkeypatch):
    # Force a fresh pool, then confirm repeated calls return the same instance.
    await codebuddy_router.close_http_client()
    try:
        first = await codebuddy_router.get_http_client()
        second = await codebuddy_router.get_http_client()
        assert first is second
    finally:
        await codebuddy_router.close_http_client()


def test_http_client_config_uses_separated_timeouts(monkeypatch):
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_stream_read_timeout_seconds", None, raising=False)
    from config import (
        get_codebuddy_connect_timeout_seconds,
        get_codebuddy_first_byte_timeout_seconds,
    )

    cfg = codebuddy_router._build_http_client_config()
    timeout = cfg["timeout"]
    # connect uses the connect timeout; unlimited stream read maps to no read
    # timeout so a long agentic stream is never capped by a global deadline.
    assert timeout.connect == get_codebuddy_connect_timeout_seconds()
    assert timeout.read is None  # default stream read timeout is 0 (unlimited)


# --------------------------------------------------------------------------- #
# 4. Concurrency limiter: queue wait recorded, timeout -> 503
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_queue_timeout_returns_503(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    # Exhaust the single slot and force a near-instant queue timeout.
    exhausted = asyncio.Semaphore(1)
    await exhausted.acquire()

    async def get_sema():
        return exhausted

    monkeypatch.setattr(codebuddy_router, "_get_upstream_semaphore", get_sema)
    monkeypatch.setattr(
        codebuddy_router,
        "get_codebuddy_upstream_queue_timeout_seconds",
        lambda: 0.05,
        raising=False,
    )
    monkeypatch.setattr(
        "config.get_codebuddy_upstream_queue_timeout_seconds", lambda: 0.05
    )

    def handler(_req):
        return sse_body([({"content": "unreachable"}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 503
    error = response.json()["error"]
    assert error["code"] == "upstream_queue_timeout"
    assert error["type"] == "server_overloaded"


@pytest.mark.asyncio
async def test_slot_released_after_non_stream(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    monkeypatch.setattr(
        "config.get_codebuddy_max_concurrent_upstream_requests", lambda: 2
    )

    def handler(_req):
        return sse_body([({"content": "ok"}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 200
    sema = await codebuddy_router._get_upstream_semaphore()
    # Both slots must be free again once the request completed.
    assert sema._value == 2


@pytest.mark.asyncio
async def test_slot_released_after_stream_completes(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    monkeypatch.setattr(
        "config.get_codebuddy_max_concurrent_upstream_requests", lambda: 2
    )

    def handler(_req):
        return sse_body([({"content": "hi"}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    # Fully drain the streaming body so the generator's finally runs.
    _ = response.text
    await upstream.aclose()

    assert response.status_code == 200
    sema = await codebuddy_router._get_upstream_semaphore()
    assert sema._value == 2


# --------------------------------------------------------------------------- #
# 5. Separated timeouts: first-byte deadline fails fast; stream read is separate
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_first_byte_timeout_is_enforced(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    monkeypatch.setattr(
        "config.get_codebuddy_first_byte_timeout_seconds", lambda: 0.1
    )
    monkeypatch.setattr(
        "config.get_codebuddy_stream_read_timeout_seconds", lambda: 0.0
    )
    monkeypatch.setattr(
        "config.get_codebuddy_heartbeat_interval_seconds", lambda: 0.0
    )

    transport = SlowFirstByteTransport(header_delay=0.5)
    upstream = install_transport(monkeypatch, transport)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    # Slow headers trip the first-byte deadline -> upstream failure, not a hang.
    assert response.status_code in (502, 504)
    assert transport.calls == 1


@pytest.mark.asyncio
async def test_heartbeat_emitted_while_waiting_and_not_content(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)
    # Generous first-byte deadline, frequent heartbeats, slow first chunk.
    monkeypatch.setattr(
        "config.get_codebuddy_first_byte_timeout_seconds", lambda: 5.0
    )
    monkeypatch.setattr(
        "config.get_codebuddy_stream_read_timeout_seconds", lambda: 0.0
    )
    monkeypatch.setattr(
        "config.get_codebuddy_heartbeat_interval_seconds", lambda: 0.05
    )

    transport = SlowChunkTransport(first_chunk_delay=0.18)
    upstream = install_transport(monkeypatch, transport)
    response = await chat(app, KEY_A, stream=True)
    text = response.text
    await upstream.aclose()

    assert response.status_code == 200
    # Heartbeat is an SSE comment, never a data event / assistant content.
    assert ": ping" in text
    events = parse_stream_events(text)
    streamed = "".join(
        (e["choices"][0].get("delta", {}) or {}).get("content", "") for e in events
    )
    assert streamed == "hello"
    # The ping comment must not have been parsed as a data event.
    for e in events:
        assert (e["choices"][0].get("delta", {}) or {}).get("content") != ": ping"


# --------------------------------------------------------------------------- #
# 6. No retry on fatal (HTTP 400) and single upstream call per stream
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_http_400_is_not_retried(monkeypatch, app, tmp_path):
    # Multiple keys available so retry, if it happened, would call upstream again.
    configure(monkeypatch)
    keys = tmp_path / "keys.txt"
    keys.write_text("key-one-000000000000\nkey-two-000000000000\n", encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(keys), reload_interval=0)
    await manager.reload()
    monkeypatch.setattr(codebuddy_router, "codebuddy_api_key_manager", manager)
    # Use api_key_file source (relay), not passthrough, to exercise failover.
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "relay")
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_request_profile", lambda: "web")

    calls = {"n": 0}

    def handler(_req):
        calls["n"] += 1
        return httpx.Response(
            400, json={"error": {"message": "bad", "code": "invalid_request"}}
        )

    upstream = install_upstream(monkeypatch, handler)
    response = await request(
        app,
        "/codebuddy/v1/chat/completions",
        RELAY_PASSWORD,
        "POST",
        {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    )
    await upstream.aclose()

    # A 400 is fatal: exactly one upstream call, no failover to the second key.
    assert calls["n"] == 1
    assert response.status_code in (400, 502)


# --------------------------------------------------------------------------- #
# 7. Tool-heavy request (30 tools) still streams
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_thirty_tools_still_streams(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    tools = [
        {
            "type": "function",
            "function": {
                "name": f"tool_{i}",
                "description": f"tool number {i}",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        for i in range(30)
    ]
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return sse_body([({"content": "done"}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True, tools=tools)
    text = response.text
    await upstream.aclose()

    assert response.status_code == 200
    assert len(seen["payload"]["tools"]) == 30
    events = parse_stream_events(text)
    streamed = "".join(
        (e["choices"][0].get("delta", {}) or {}).get("content", "") for e in events
    )
    assert streamed == "done"


# --------------------------------------------------------------------------- #
# 8. Config getters validate and clamp
# --------------------------------------------------------------------------- #


def test_config_getters_defaults_and_validation():
    import config

    assert config.get_codebuddy_connect_timeout_seconds() == 30.0
    assert config.get_codebuddy_first_byte_timeout_seconds() == 180.0
    # 0 is a valid (unlimited) stream read timeout.
    assert config.get_codebuddy_stream_read_timeout_seconds() == 0.0
    assert config.get_codebuddy_max_concurrent_upstream_requests() == 20
    assert config.get_codebuddy_upstream_queue_timeout_seconds() == 60.0
    assert config.get_codebuddy_heartbeat_interval_seconds() == 15.0
