"""
Payload-compatibility regression tests for /codebuddy/v1/chat/completions.

Reproduces the reported bug: Claude Code works through 9Router + CodeBuddy2API,
but the Shiteru web chat fails ("Upstream CodeBuddy request failed" / empty
reply) because its request body carries OpenAI-only fields and/or a UI model
label that CodeBuddy rejects.

The fix builds a strict upstream allowlist (model, messages, stream, tools,
tool_choice) and maps model aliases before calling CodeBuddy. These tests lock
in that behavior with the two exact request shapes plus unit coverage of the
allowlist and model mapping.

Uses a mock upstream transport; no real CodeBuddy API key is required.
"""
import json

import httpx
import pytest

from src import auth, codebuddy_router
from src.codebuddy_router import RequestProcessor
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"

AVAILABLE_MODELS = ["claude-4.0", "gpt-5", "auto-chat"]

# OpenAI fields Shiteru-style clients send that CodeBuddy does not accept.
UNSUPPORTED_FIELDS = {
    "response_format": {"type": "json_object"},
    "parallel_tool_calls": False,
    "reasoning_effort": "high",
    "stream_options": {"include_usage": True},
    "service_tier": "auto",
    "store": True,
    "metadata": {"session": "abc"},
    "seed": 42,
    "logprobs": True,
    "top_logprobs": 5,
    "prediction": {"type": "content", "content": "x"},
    "modalities": ["text"],
    "temperature": 0.7,
    "top_p": 0.9,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "max_tokens": 1024,
    "user": "shiteru-web",
}


# --------------------------------------------------------------------------- #
# Unit: model mapping
# --------------------------------------------------------------------------- #


def _patch_models(monkeypatch, aliases=None, default="auto-chat", policy="passthrough"):
    monkeypatch.setattr(codebuddy_router, "get_available_models_list", lambda: list(AVAILABLE_MODELS))
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_default_model", lambda: default)
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_model_aliases", lambda: dict(aliases or {}))
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_unknown_model_policy", lambda: policy)


def test_known_model_passthrough(monkeypatch):
    _patch_models(monkeypatch)
    assert RequestProcessor.map_model("claude-4.0") == "claude-4.0"
    assert RequestProcessor.resolve_model("claude-4.0") == ("claude-4.0", "exact")


def test_alias_mapped_case_insensitively(monkeypatch):
    _patch_models(monkeypatch, aliases={"claude opus 4.7": "claude-4.0"})
    assert RequestProcessor.map_model("Claude Opus 4.7") == "claude-4.0"
    assert RequestProcessor.resolve_model("Claude Opus 4.7") == ("claude-4.0", "alias")


def test_unknown_label_passthrough_by_default(monkeypatch):
    # Default policy is passthrough: an unknown label is forwarded verbatim,
    # never silently rewritten to the default model. This is the core bug fix.
    _patch_models(monkeypatch, default="auto-chat", policy="passthrough")
    assert RequestProcessor.resolve_model("Claude Opus 4.7") == (
        "Claude Opus 4.7",
        "passthrough",
    )
    assert RequestProcessor.map_model("Claude Opus 4.7") == "Claude Opus 4.7"


def test_unknown_label_default_policy_falls_back(monkeypatch):
    # Only when policy=default does an unknown label map to the default model.
    _patch_models(monkeypatch, default="auto-chat", policy="default")
    assert RequestProcessor.resolve_model("Claude Opus 4.7") == ("auto-chat", "default")


def test_unknown_label_reject_policy_raises(monkeypatch):
    _patch_models(monkeypatch, default="auto-chat", policy="reject")
    with pytest.raises(codebuddy_router.UnknownModelError) as excinfo:
        RequestProcessor.resolve_model("Claude Opus 4.7")
    assert excinfo.value.requested_model == "Claude Opus 4.7"


def test_missing_model_uses_default_under_every_policy(monkeypatch):
    # A missing/blank model has nothing to pass through, so it uses the default
    # regardless of policy (and never raises under reject).
    for policy in ("passthrough", "default", "reject"):
        _patch_models(monkeypatch, default="auto-chat", policy=policy)
        assert RequestProcessor.resolve_model(None) == ("auto-chat", "default_empty")
        assert RequestProcessor.resolve_model("") == ("auto-chat", "default_empty")
        assert RequestProcessor.map_model(None) == "auto-chat"


def test_explicit_auto_chat_request_is_exact_not_default(monkeypatch):
    # A client explicitly asking for auto-chat matches exactly; it is not the
    # unknown-model fallback path.
    _patch_models(monkeypatch, default="auto-chat", policy="reject")
    assert RequestProcessor.resolve_model("auto-chat") == ("auto-chat", "exact")


def test_opus_4_7_1m_is_not_converted_to_auto_chat(monkeypatch):
    # Regression for the reported log: requested_model=claude-opus-4.7-1m must
    # NOT be silently mapped to auto-chat. Under the default passthrough policy
    # it is forwarded verbatim so the real upstream model is preserved.
    _patch_models(monkeypatch, default="auto-chat", policy="passthrough")
    mapped, source = RequestProcessor.resolve_model("claude-opus-4.7-1m")
    assert mapped == "claude-opus-4.7-1m"
    assert mapped != "auto-chat"
    assert source == "passthrough"


# --------------------------------------------------------------------------- #
# Unit: allowlist
# --------------------------------------------------------------------------- #


def test_prepare_payload_drops_unsupported_fields(monkeypatch):
    _patch_models(monkeypatch)
    monkeypatch.setattr(codebuddy_router, "get_sanitize_agent_prompt", lambda: True)
    monkeypatch.setattr(codebuddy_router, "get_max_system_prompt_length", lambda: 2000)

    body = {
        "model": "claude-4.0",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False,
        "tools": [{"type": "function", "function": {"name": "f"}}],
        "tool_choice": "auto",
        **UNSUPPORTED_FIELDS,
    }
    payload, prep = RequestProcessor.prepare_payload(body)

    # Only allowlisted fields survive.
    assert set(payload.keys()) == {"model", "messages", "stream", "tools", "tool_choice"}
    # Upstream always receives stream=True (CodeBuddy is SSE-only).
    assert payload["stream"] is True
    # Every unsupported field is reported as dropped.
    for field in UNSUPPORTED_FIELDS:
        assert field in prep["dropped_fields"]
    assert prep["mapped_model"] == "claude-4.0"


def test_prepare_payload_omits_absent_tools(monkeypatch):
    _patch_models(monkeypatch)
    monkeypatch.setattr(codebuddy_router, "get_sanitize_agent_prompt", lambda: True)
    monkeypatch.setattr(codebuddy_router, "get_max_system_prompt_length", lambda: 2000)

    body = {"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}
    payload, _prep = RequestProcessor.prepare_payload(body)
    assert "tools" not in payload
    assert "tool_choice" not in payload


# --------------------------------------------------------------------------- #
# Integration fixtures
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


def configure(monkeypatch, aliases=None, default="auto-chat", policy="passthrough"):
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "passthrough")
    monkeypatch.setattr(auth, "get_server_password", lambda: RELAY_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: ADMIN_PASSWORD)
    monkeypatch.setattr(codebuddy_router, "get_upstream_api_key_header", lambda: "both")
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_request_profile", lambda: "web")
    monkeypatch.setattr(codebuddy_router, "get_sanitize_agent_prompt", lambda: True)
    monkeypatch.setattr(codebuddy_router, "get_max_system_prompt_length", lambda: 2000)
    _patch_models(monkeypatch, aliases=aliases, default=default, policy=policy)
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


async def post(app, token, body):
    transport = httpx.ASGITransport(app=app)
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/codebuddy/v1/chat/completions", headers=headers, json=body
        )


# --------------------------------------------------------------------------- #
# The two exact request shapes
# --------------------------------------------------------------------------- #

# A Claude Code request: valid model, system + user messages, tools, streaming.
CLAUDE_CODE_REQUEST = {
    "model": "claude-4.0",
    "messages": [
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "list two prime numbers"},
    ],
    "stream": True,
    "tools": [
        {
            "type": "function",
            "function": {"name": "noop", "parameters": {"type": "object"}},
        }
    ],
    "tool_choice": "auto",
}

# A Shiteru web-chat request: UI display-label model + many OpenAI-only fields
# and stream=false. This is the shape that currently fails upstream.
SHITERU_WEB_REQUEST = {
    "model": "Claude Opus 4.7",
    "messages": [{"role": "user", "content": "halo, siapa kamu"}],
    "stream": False,
    "response_format": {"type": "text"},
    "reasoning_effort": "medium",
    "parallel_tool_calls": True,
    "stream_options": {"include_usage": True},
    "temperature": 0.6,
    "top_p": 1.0,
    "max_tokens": 800,
    "metadata": {"ui": "shiteru"},
    "seed": 7,
}


@pytest.mark.asyncio
async def test_claude_code_request_succeeds(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("2 and 3")

    upstream = install_upstream(monkeypatch, handler)
    response = await post(app, KEY_A, CLAUDE_CODE_REQUEST)
    await upstream.aclose()

    assert response.status_code == 200
    # Streaming client -> SSE passthrough.
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "[DONE]" in response.text
    # Upstream received only allowlisted fields, model unchanged, stream=True.
    body = seen["body"]
    assert set(body.keys()) == {"model", "messages", "stream", "tools", "tool_choice"}
    assert body["model"] == "claude-4.0"
    assert body["stream"] is True
    assert body["tools"] == CLAUDE_CODE_REQUEST["tools"]


@pytest.mark.asyncio
async def test_shiteru_web_request_now_succeeds(monkeypatch, app, empty_pool):
    # Map the UI label so it resolves to a real upstream model.
    configure(monkeypatch, aliases={"claude opus 4.7": "claude-4.0"})
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("Halo! Saya asisten AI.")

    upstream = install_upstream(monkeypatch, handler)
    response = await post(app, KEY_A, SHITERU_WEB_REQUEST)
    await upstream.aclose()

    # No more "Upstream CodeBuddy request failed": a proper ChatCompletion.
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    data = response.json()
    assert data["object"] == "chat.completion"
    choice = data["choices"][0]
    assert choice["message"]["role"] == "assistant"
    assert choice["message"]["content"] == "Halo! Saya asisten AI."
    assert choice["finish_reason"] == "stop"

    # Upstream must NOT have received any of the unsupported OpenAI fields...
    body = seen["body"]
    assert set(body.keys()) == {"model", "messages", "stream"}
    for field in (
        "response_format", "reasoning_effort", "parallel_tool_calls",
        "stream_options", "temperature", "top_p", "max_tokens", "metadata", "seed",
    ):
        assert field not in body
    # ...and the UI label must be mapped to a real model ID.
    assert body["model"] == "claude-4.0"
    # CodeBuddy is SSE-only: upstream stream is always True even for stream=false.
    assert body["stream"] is True


@pytest.mark.asyncio
async def test_shiteru_unmapped_label_passthrough_by_default(
    monkeypatch, app, empty_pool
):
    # No alias configured and default policy (passthrough): the unknown UI label
    # is forwarded verbatim, NOT silently rewritten to auto-chat.
    configure(monkeypatch, default="auto-chat", policy="passthrough")
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)
    response = await post(app, KEY_A, SHITERU_WEB_REQUEST)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen["body"]["model"] == "Claude Opus 4.7"
    assert seen["body"]["model"] != "auto-chat"


@pytest.mark.asyncio
async def test_opus_4_7_1m_not_rewritten_to_auto_chat_integration(
    monkeypatch, app, empty_pool
):
    # End-to-end regression for the reported log line
    # (requested_model=claude-opus-4.7-1m mapped_model=auto-chat): the real
    # model ID must reach upstream unchanged under the default policy.
    configure(monkeypatch, default="auto-chat", policy="passthrough")
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)
    body = {
        "model": "claude-opus-4.7-1m",
        "messages": [{"role": "user", "content": "halo"}],
        "stream": False,
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen["body"]["model"] == "claude-opus-4.7-1m"
    assert seen["body"]["model"] != "auto-chat"


@pytest.mark.asyncio
async def test_shiteru_unmapped_label_default_policy_uses_default(
    monkeypatch, app, empty_pool
):
    # Only with policy=default does the unknown label fall back to the default.
    configure(monkeypatch, default="auto-chat", policy="default")
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)
    response = await post(app, KEY_A, SHITERU_WEB_REQUEST)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen["body"]["model"] == "auto-chat"


@pytest.mark.asyncio
async def test_unknown_model_reject_policy_returns_400(monkeypatch, app, empty_pool):
    # policy=reject: an unknown model is rejected with HTTP 400 code=unknown_model
    # before any upstream call is made.
    configure(monkeypatch, default="auto-chat", policy="reject")
    called = {"upstream": False}

    def handler(req):
        called["upstream"] = True
        return sse("should not be called")

    upstream = install_upstream(monkeypatch, handler)
    response = await post(app, KEY_A, SHITERU_WEB_REQUEST)
    await upstream.aclose()

    assert response.status_code == 400
    assert called["upstream"] is False
    error = response.json()["error"]
    assert error["code"] == "unknown_model"
    assert error["type"] == "invalid_request_error"


@pytest.mark.asyncio
async def test_passthrough_preserves_upstream_400_no_fallback(
    monkeypatch, app, empty_pool
):
    # Requirement 10: if CodeBuddy rejects the passed-through model with a 400,
    # that rejection is surfaced rather than retried with a different model.
    configure(monkeypatch, default="auto-chat", policy="passthrough")
    seen = {"count": 0, "models": []}

    def handler(req):
        seen["count"] += 1
        seen["models"].append(json.loads(req.content)["model"])
        return httpx.Response(
            400,
            json={"error": {"message": "unknown model", "code": "model_not_found"}},
            headers={"content-type": "application/json"},
        )

    upstream = install_upstream(monkeypatch, handler)
    body = {
        "model": "claude-opus-4.7-1m",
        "messages": [{"role": "user", "content": "halo"}],
        "stream": False,
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    # Upstream saw exactly one attempt with the verbatim model; no fallback to
    # auto-chat was attempted.
    assert seen["count"] == 1
    assert seen["models"] == ["claude-opus-4.7-1m"]
    assert response.status_code == 400
