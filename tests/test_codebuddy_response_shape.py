"""
Response-shape regression tests for /codebuddy/v1/chat/completions.

Locks in the OpenAI-compatible contract for both transports so a future change
to the aggregator or streaming path cannot silently regress it:

  * stream=false  -> a single ChatCompletion JSON object
                     (choices[0].message.role="assistant", non-empty content,
                     finish_reason="stop")
  * stream=true   -> text/event-stream, `data: <JSON>\\n\\n` events with text in
                     choices[0].delta.content, terminated by exactly one
                     `data: [DONE]`

Covers: sanitized system prompt still yields content, moderation maps to
content_filter (never an empty reply), tool-call-only responses return
tool_calls (never "no response"), and reasoning_content never replaces the
final assistant content.

Uses a mock upstream transport; no real CodeBuddy API key is required.
"""
import json

import httpx
import pytest

from src import auth, codebuddy_router
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager
from src.codebuddy_message_sanitizer import NEUTRAL_SYSTEM_PROMPT

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"

MODERATION_TEXT = (
    "抱歉，系统检测到您当前输入的信息存在敏感内容，我无法响应您的请求，请检查后重新输入。"
)


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


def sse_body(chunks, done=True):
    """Build an SSE ``httpx.Response`` from ``(delta, finish_reason)`` pairs."""
    parts = []
    for delta, finish in chunks:
        choice = {"index": 0, "delta": delta}
        if finish is not None:
            choice["finish_reason"] = finish
        obj = {
            "id": "chat-shape",
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


async def chat(app, token, stream=False, messages=None):
    body = {
        "model": "auto-chat",
        "messages": messages or [{"role": "user", "content": "halo, siapa kamu"}],
        "stream": stream,
    }
    return await request(app, "/codebuddy/v1/chat/completions", token, "POST", body)


def parse_stream_events(text):
    """Return the list of parsed JSON event objects (excluding [DONE])."""
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
# 1. stream=true yields content and exactly one [DONE]
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_stream_yields_content_and_single_done(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body(
            [
                ({"role": "assistant"}, None),          # role-only chunk (no content)
                ({"content": "Hello"}, None),
                ({"content": " world"}, None),
                ({}, "stop"),                            # trailing usage/stop chunk
            ]
        )

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    text = response.text
    assert text.count("[DONE]") == 1
    # No HTML or extra JSON wrapper.
    assert "<html" not in text.lower()

    events = parse_stream_events(text)
    streamed = "".join(
        (e["choices"][0].get("delta", {}) or {}).get("content", "") for e in events
    )
    assert streamed == "Hello world"


# --------------------------------------------------------------------------- #
# 2. stream=false yields a ChatCompletion with non-empty content
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_stream_returns_chat_completion(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body([({"content": "2 + 2 = 4"}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "2 + 2 = 4"
    assert choice["message"]["content"]  # non-empty
    assert choice["finish_reason"] == "stop"


# --------------------------------------------------------------------------- #
# 3. Sanitized system prompt still yields final content
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sanitized_prompt_still_returns_content(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["messages"] = json.loads(req.content)["messages"]
        return sse_body([({"content": "Halo! Saya asisten."}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(
        app,
        KEY_A,
        stream=False,
        messages=[
            {"role": "system", "content": "You are Claude Code, the official CLI."},
            {"role": "user", "content": "halo, siapa kamu"},
        ],
    )
    await upstream.aclose()

    assert response.status_code == 200
    # The agent system prompt was sanitized upstream...
    system_msg = next(m for m in seen["messages"] if m["role"] == "system")
    assert system_msg["content"] == NEUTRAL_SYSTEM_PROMPT
    # ...and the client still gets a real, non-empty assistant reply.
    data = response.json()
    assert data["choices"][0]["message"]["content"] == "Halo! Saya asisten."


# --------------------------------------------------------------------------- #
# 4. Moderation yields content_filter, never an empty response
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_stream_moderation_is_content_filter(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body([({"content": MODERATION_TEXT}, "stop")])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 400
    assert response.headers.get("X-CodeBuddy-Moderation") == "true"
    error = response.json()["error"]
    assert error["type"] == "content_filter"
    assert error["code"] == "codebuddy_content_filter"
    assert error["message"]  # non-empty explanation, not an empty reply
    # The Mandarin refusal must not be surfaced as assistant content.
    assert "choices" not in response.json()


@pytest.mark.asyncio
async def test_stream_moderation_is_content_filter(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body([({"content": MODERATION_TEXT}, None)])

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.headers.get("X-CodeBuddy-Moderation") == "true"
    text = response.text
    assert text.count("[DONE]") == 1
    events = parse_stream_events(text)
    assert events, "expected at least one content_filter event"
    assert events[0]["choices"][0]["finish_reason"] == "content_filter"
    # The raw Mandarin refusal must not leak into the stream.
    assert "敏感内容" not in text


# --------------------------------------------------------------------------- #
# 5. Tool-call-only response returns tool_calls, never "no response"
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tool_call_only_returns_tool_calls(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(_req):
        return sse_body(
            [
                (
                    {
                        "tool_calls": [
                            {
                                "id": "tooluse_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "Jakarta"}',
                                },
                            }
                        ]
                    },
                    "tool_calls",
                )
            ]
        )

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 200
    message = response.json()["choices"][0]["message"]
    tool_calls = message.get("tool_calls")
    assert isinstance(tool_calls, list) and len(tool_calls) == 1
    assert tool_calls[0]["function"]["name"] == "get_weather"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"city": "Jakarta"}
    assert response.json()["choices"][0]["finish_reason"] == "tool_calls"
    # Must not fabricate a placeholder like "no response".
    assert message.get("content") in ("", None)
    assert message.get("content") != "no response"


# --------------------------------------------------------------------------- #
# 6. reasoning_content never replaces the final assistant content
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_reasoning_content_does_not_replace_final_content(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)

    def handler(_req):
        return sse_body(
            [
                ({"reasoning_content": "Let me think step by step..."}, None),
                ({"reasoning_content": " still reasoning..."}, None),
                ({"content": "The answer is 42."}, "stop"),
            ]
        )

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 200
    content = response.json()["choices"][0]["message"]["content"]
    assert content == "The answer is 42."
    assert "reasoning" not in content
    assert "step by step" not in content
