"""
Tests for CodeBuddy false-positive moderation mitigation:

  * agent system-prompt sanitization (unit + end-to-end)
  * web/cli request header profiles
  * Mandarin moderation detection (non-stream + stream)
  * safe diagnostics (no raw key in logs)
  * message normalization / tool-call preservation
  * upstream status propagation for 9Router fallback

Uses a mock upstream transport; no real CodeBuddy API key is required.
"""
import json
import logging

import httpx
import pytest

import config
from src import auth, codebuddy_router
from src.codebuddy_api_client import codebuddy_api_client
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager
from src.codebuddy_message_sanitizer import (
    NEUTRAL_SYSTEM_PROMPT,
    is_codebuddy_moderation_response,
    sanitize_messages,
)

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"
KEY_B = "passthrough-account-beta-0002"

MODERATION_TEXT = (
    "抱歉，系统检测到您当前输入的信息存在敏感内容，我无法响应您的请求，请检查后重新输入。"
)


# --------------------------------------------------------------------------- #
# Unit tests: sanitizer + moderation detector (no I/O)
# --------------------------------------------------------------------------- #


def test_simple_user_prompt_not_modified():
    messages = [{"role": "user", "content": "halo, siapa kamu"}]
    result, changed = sanitize_messages(messages)
    assert changed is False
    assert result[0]["content"] == "halo, siapa kamu"


def test_user_message_never_changed_even_with_agent_markers():
    # A user message that mentions "Claude Code" must stay intact; only system
    # messages are ever sanitized.
    messages = [{"role": "user", "content": "explain what claude code cli does"}]
    result, changed = sanitize_messages(messages)
    assert changed is False
    assert result[0]["content"] == "explain what claude code cli does"


def test_claude_code_system_prompt_detected_and_replaced():
    messages = [
        {"role": "system", "content": "You are Claude Code, Anthropic's official CLI."},
        {"role": "user", "content": "hi"},
    ]
    result, changed = sanitize_messages(messages)
    assert changed is True
    assert result[0]["content"] == NEUTRAL_SYSTEM_PROMPT
    assert result[1]["content"] == "hi"


def test_over_length_system_prompt_replaced():
    long_prompt = "a" * 5000
    messages = [{"role": "system", "content": long_prompt}]
    result, changed = sanitize_messages(messages, max_system_prompt_length=2000)
    assert changed is True
    assert result[0]["content"] == NEUTRAL_SYSTEM_PROMPT


def test_short_normal_system_prompt_preserved():
    messages = [
        {"role": "system", "content": "You are a friendly translator."},
        {"role": "user", "content": "hola"},
    ]
    result, changed = sanitize_messages(messages)
    assert changed is False
    assert result[0]["content"] == "You are a friendly translator."


def test_sanitize_disabled_is_noop():
    messages = [{"role": "system", "content": "You are Claude Code"}]
    result, changed = sanitize_messages(messages, enabled=False)
    assert changed is False
    assert result[0]["content"] == "You are Claude Code"


def test_moderation_markers_detected():
    assert is_codebuddy_moderation_response(MODERATION_TEXT) is True
    assert is_codebuddy_moderation_response("系统检测到问题") is True


def test_normal_mandarin_not_flagged():
    assert is_codebuddy_moderation_response("你好，我是一个AI助手，很高兴认识你。") is False
    assert is_codebuddy_moderation_response("2 + 2 = 4") is False
    assert is_codebuddy_moderation_response("") is False
    assert is_codebuddy_moderation_response(None) is False


# --------------------------------------------------------------------------- #
# Header profile unit tests
# --------------------------------------------------------------------------- #


def test_web_profile_uses_browser_user_agent_and_both_headers():
    headers = codebuddy_api_client.generate_codebuddy_headers(
        bearer_token=KEY_A, profile="web"
    )
    assert "Mozilla/5.0" in headers["User-Agent"]
    assert "CLI" not in headers["User-Agent"]
    assert headers["Authorization"] == f"Bearer {KEY_A}"
    assert headers["X-API-Key"] == KEY_A
    # CLI/IDE identity headers must be absent on the web profile.
    assert "X-IDE-Type" not in headers
    assert "X-IDE-Name" not in headers
    assert "x-stainless-lang" not in headers


def test_cli_profile_preserves_legacy_headers():
    headers = codebuddy_api_client.generate_codebuddy_headers(
        bearer_token=KEY_A, profile="cli", api_key_header="bearer"
    )
    assert headers["User-Agent"] == "CLI/1.0.7 CodeBuddy/1.0.7"
    assert headers["X-IDE-Type"] == "CLI"
    assert headers["Authorization"] == f"Bearer {KEY_A}"
    # cli honors api_key_header; bearer-only means no X-API-Key.
    assert "X-API-Key" not in headers


# --------------------------------------------------------------------------- #
# Integration fixtures / helpers
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


def configure(monkeypatch, client_mode="passthrough", profile="web"):
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: client_mode)
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


def sse(text, finish="stop"):
    body = (
        'data: {"id":"chat-1","model":"auto-chat","choices":'
        f'[{{"delta":{{"content":{json.dumps(text)}}},"finish_reason":{json.dumps(finish)}}}]}}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


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


# --------------------------------------------------------------------------- #
# Integration: header profile end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_web_profile_sends_both_headers_upstream(monkeypatch, app, empty_pool):
    configure(monkeypatch, profile="web")
    seen = {}

    def handler(req):
        seen["authorization"] = req.headers.get("authorization")
        seen["x-api-key"] = req.headers.get("x-api-key")
        seen["user-agent"] = req.headers.get("user-agent")
        return sse("hi")

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen["authorization"] == f"Bearer {KEY_A}"
    assert seen["x-api-key"] == KEY_A
    assert "Mozilla/5.0" in seen["user-agent"]


@pytest.mark.asyncio
async def test_cli_profile_still_works(monkeypatch, app, empty_pool):
    configure(monkeypatch, profile="cli")
    monkeypatch.setattr(codebuddy_router, "get_upstream_api_key_header", lambda: "bearer")
    seen = {}

    def handler(req):
        seen["authorization"] = req.headers.get("authorization")
        seen["user-agent"] = req.headers.get("user-agent")
        seen["x-ide-type"] = req.headers.get("x-ide-type")
        return sse("hi")

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen["authorization"] == f"Bearer {KEY_A}"
    assert seen["user-agent"] == "CLI/1.0.7 CodeBuddy/1.0.7"
    assert seen["x-ide-type"] == "CLI"


# --------------------------------------------------------------------------- #
# Integration: sanitization end-to-end
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_agent_system_prompt_sanitized_upstream(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        body = json.loads(req.content)
        seen["messages"] = body["messages"]
        return sse("hi")

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(
        app,
        KEY_A,
        messages=[
            {"role": "system", "content": "You are Claude Code, the official CLI."},
            {"role": "user", "content": "halo, siapa kamu"},
        ],
    )
    await upstream.aclose()

    assert response.status_code == 200
    system_msg = next(m for m in seen["messages"] if m["role"] == "system")
    user_msg = next(m for m in seen["messages"] if m["role"] == "user")
    assert system_msg["content"] == NEUTRAL_SYSTEM_PROMPT
    # The user message must be forwarded verbatim.
    assert user_msg["content"] == "halo, siapa kamu"


@pytest.mark.asyncio
async def test_tool_calls_survive_normalization(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    tools = [
        {
            "type": "function",
            "function": {"name": "get_weather", "parameters": {"type": "object"}},
        }
    ]
    upstream = install_upstream(monkeypatch, handler)
    response = await request(
        app,
        "/codebuddy/v1/chat/completions",
        KEY_A,
        "POST",
        {
            "model": "auto-chat",
            "messages": [{"role": "user", "content": "weather?"}],
            "tools": tools,
            "stream": False,
        },
    )
    await upstream.aclose()

    assert response.status_code == 200
    assert seen["body"]["tools"] == tools


# --------------------------------------------------------------------------- #
# Integration: moderation detection
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_stream_moderation_becomes_content_filter(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(req):
        return sse(MODERATION_TEXT)

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 400
    assert response.headers.get("X-CodeBuddy-Moderation") == "true"
    payload = response.json()
    assert payload["error"]["type"] == "content_filter"
    assert payload["error"]["code"] == "codebuddy_content_filter"
    # The raw Mandarin refusal must not be surfaced as assistant content.
    assert "choices" not in payload


@pytest.mark.asyncio
async def test_stream_moderation_uses_content_filter_finish_reason(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)

    def handler(req):
        return sse(MODERATION_TEXT, finish=None)

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.headers.get("X-CodeBuddy-Moderation") == "true"
    text = response.text
    assert '"finish_reason": "content_filter"' in text or '"finish_reason":"content_filter"' in text
    assert "[DONE]" in text
    # The Mandarin refusal text must not be streamed as assistant content.
    assert "敏感内容" not in text


@pytest.mark.asyncio
async def test_normal_mandarin_response_not_flagged(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    normal = "你好，我是一个AI助手。"

    def handler(req):
        return sse(normal)

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.headers.get("X-CodeBuddy-Moderation") is None
    assert response.json()["choices"][0]["message"]["content"] == normal


@pytest.mark.asyncio
async def test_normal_stream_still_emits_done(monkeypatch, app, empty_pool):
    configure(monkeypatch)

    def handler(req):
        return sse("hello there")

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert "[DONE]" in response.text
    assert "hello there" in response.text


# --------------------------------------------------------------------------- #
# Integration: upstream status propagation for 9Router
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, 401), (403, 403), (429, 429), (500, 502), (503, 502)],
)
async def test_upstream_status_propagated(monkeypatch, app, empty_pool, status, expected):
    configure(monkeypatch)

    def handler(req):
        return httpx.Response(status, json={"error": "upstream"})

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == expected
    # Never leak the key or upstream body.
    assert KEY_A not in response.text


# --------------------------------------------------------------------------- #
# Safe diagnostics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_raw_key_not_in_logs(monkeypatch, app, empty_pool, caplog):
    configure(monkeypatch)

    def handler(req):
        return sse("hi")

    upstream = install_upstream(monkeypatch, handler)
    with caplog.at_level(logging.INFO, logger="src.codebuddy_router"):
        response = await chat(app, KEY_A, stream=False)
    await upstream.aclose()

    assert response.status_code == 200
    combined = "\n".join(record.getMessage() for record in caplog.records)
    assert KEY_A not in combined
    # A short fingerprint should be present instead.
    assert "key_fingerprint=" in combined
    assert "request_profile=web" in combined
