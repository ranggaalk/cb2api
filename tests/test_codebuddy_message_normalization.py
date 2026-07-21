"""
Upstream message-normalization regression tests.

Reproduces and locks the fix for the reported upstream failure during Claude
Code tool-use conversations:

    Message 13 must have 'role' and 'content' fields

CodeBuddy rejects any message missing a role or content. Valid OpenAI/Anthropic
tool-use turns legitimately omit those fields (an assistant turn carrying
``tool_calls`` has null/absent content; a tool result is identified by
``tool_call_id`` and may omit ``role``). ``normalize_messages_for_upstream``
runs as the final transform before the upstream request and guarantees every
message has a valid role and a content field, while preserving content
CodeBuddy already accepts (including multimodal arrays).

These tests assert both the unit behavior of the normalizer and the exact
messages that reach the (mocked) upstream through the full request path.

Uses a mock upstream transport; no real CodeBuddy API key is required.
"""
import json

import httpx
import pytest

from src import auth, codebuddy_router
from src.codebuddy_router import RequestProcessor
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager
from src.codebuddy_message_sanitizer import (
    MessageNormalizationError,
    normalize_messages_for_upstream,
)

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"

AVAILABLE_MODELS = ["claude-4.0", "gpt-5", "auto-chat"]


# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirror test_codebuddy_payload_allowlist.py)
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


def _patch_models(monkeypatch):
    monkeypatch.setattr(codebuddy_router, "get_available_models_list", lambda: list(AVAILABLE_MODELS))
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_default_model", lambda: "auto-chat")
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_model_aliases", lambda: {})
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_unknown_model_policy", lambda: "passthrough")


def configure(monkeypatch):
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "passthrough")
    monkeypatch.setattr(auth, "get_server_password", lambda: RELAY_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: ADMIN_PASSWORD)
    monkeypatch.setattr(codebuddy_router, "get_upstream_api_key_header", lambda: "both")
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_request_profile", lambda: "web")
    # Disable system-prompt sanitization so it does not perturb the message
    # sequences under test; normalization is independent of it.
    monkeypatch.setattr(codebuddy_router, "get_sanitize_agent_prompt", lambda: False)
    monkeypatch.setattr(codebuddy_router, "get_max_system_prompt_length", lambda: 2000)
    _patch_models(monkeypatch)
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


def _assert_all_valid_upstream(messages):
    """Every upstream message must have a non-empty role and a content field."""
    for i, msg in enumerate(messages):
        assert isinstance(msg, dict), f"message {i} is not an object"
        role = msg.get("role")
        assert isinstance(role, str) and role.strip(), f"message {i} has no valid role"
        assert "content" in msg, f"message {i} is missing content"
        assert msg["content"] is not None, f"message {i} has null content"


# --------------------------------------------------------------------------- #
# Unit: normalize_messages_for_upstream
# --------------------------------------------------------------------------- #


def test_assistant_tool_calls_missing_content_gets_empty_string():
    tool_calls = [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city":"Jakarta"}'},
        }
    ]
    messages = [
        {"role": "user", "content": "weather?"},
        {"role": "assistant", "tool_calls": tool_calls},  # no content field
    ]
    out = normalize_messages_for_upstream(messages)
    _assert_all_valid_upstream(out)
    assert out[1]["role"] == "assistant"
    assert out[1]["content"] == ""
    # tool_calls preserved verbatim.
    assert out[1]["tool_calls"] == tool_calls


def test_assistant_tool_calls_null_content_gets_empty_string():
    messages = [
        {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
    ]
    out = normalize_messages_for_upstream(messages)
    assert out[0]["content"] == ""
    assert out[0]["role"] == "assistant"


def test_tool_result_missing_role_inferred_from_tool_call_id():
    messages = [
        # No role, but tool_call_id identifies it as a tool result.
        {"tool_call_id": "call_abc", "content": "23C and sunny"},
    ]
    out = normalize_messages_for_upstream(messages)
    _assert_all_valid_upstream(out)
    assert out[0]["role"] == "tool"
    assert out[0]["tool_call_id"] == "call_abc"
    assert out[0]["content"] == "23C and sunny"


def test_tool_result_shape_is_preserved():
    messages = [
        {"role": "tool", "tool_call_id": "call_xyz", "content": "result text"},
    ]
    out = normalize_messages_for_upstream(messages)
    assert out[0] == {
        "role": "tool",
        "tool_call_id": "call_xyz",
        "content": "result text",
    }


def test_existing_empty_string_content_is_not_touched():
    messages = [{"role": "assistant", "content": "", "tool_calls": [{"id": "c1"}]}]
    out = normalize_messages_for_upstream(messages)
    assert out[0]["content"] == ""


def test_multimodal_content_array_is_preserved():
    multimodal = [
        {"type": "text", "text": "look at this"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]
    messages = [{"role": "user", "content": multimodal}]
    out = normalize_messages_for_upstream(messages)
    _assert_all_valid_upstream(out)
    # The array must be preserved verbatim, not flattened or replaced.
    assert out[0]["content"] == multimodal
    assert isinstance(out[0]["content"], list)


def test_undeterminable_role_raises_with_index():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
        {"content": "orphan with no role and no tool markers"},  # index 2
    ]
    with pytest.raises(MessageNormalizationError) as excinfo:
        normalize_messages_for_upstream(messages)
    assert excinfo.value.index == 2


def test_blank_role_string_is_treated_as_missing():
    messages = [{"role": "   ", "content": "x"}]
    with pytest.raises(MessageNormalizationError) as excinfo:
        normalize_messages_for_upstream(messages)
    assert excinfo.value.index == 0


def test_normalizer_does_not_mutate_input():
    messages = [{"role": "assistant", "tool_calls": [{"id": "c1"}]}]
    normalize_messages_for_upstream(messages)
    # Original list/message untouched (copy-on-write).
    assert "content" not in messages[0]


# --------------------------------------------------------------------------- #
# Integration: full request path, asserting what reaches upstream
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_claude_code_tool_loop_reaches_upstream_valid(monkeypatch, app, empty_pool):
    """Assistant tool_calls (no content) + tool result (no role) both survive."""
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("done")

    upstream = install_upstream(monkeypatch, handler)
    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [
            {"role": "user", "content": "weather in Jakarta?"},
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city":"Jakarta"}'},
                    }
                ],
            },
            {"tool_call_id": "call_1", "content": "31C, humid"},
            {"role": "user", "content": "and tomorrow?"},
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]
    _assert_all_valid_upstream(sent)
    # Assistant tool-call turn keeps its tool_calls and gains content: "".
    assert sent[1]["role"] == "assistant"
    assert sent[1]["content"] == ""
    assert sent[1]["tool_calls"][0]["id"] == "call_1"
    # Tool result got role tool inferred from tool_call_id.
    assert sent[2]["role"] == "tool"
    assert sent[2]["tool_call_id"] == "call_1"


@pytest.mark.asyncio
async def test_multiple_claude_code_tool_loops(monkeypatch, app, empty_pool):
    """Several back-to-back tool loops all normalize correctly."""
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)

    messages = [{"role": "user", "content": "start a multi-step task"}]
    for n in range(3):
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{n}",
                        "type": "function",
                        "function": {"name": "step", "arguments": f'{{"n":{n}}}'},
                    }
                ],
            }
        )
        # Tool result with no role (identified by tool_call_id).
        messages.append({"tool_call_id": f"call_{n}", "content": f"step {n} done"})
    messages.append({"role": "assistant", "content": "all steps complete"})

    body = {"model": "claude-4.0", "stream": False, "messages": messages}
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]
    _assert_all_valid_upstream(sent)
    # Each tool-call assistant turn has content "" and each tool result role tool.
    assistants = [m for m in sent if m["role"] == "assistant" and m.get("tool_calls")]
    tools = [m for m in sent if m["role"] == "tool"]
    assert len(assistants) == 3
    assert len(tools) == 3
    assert all(m["content"] == "" for m in assistants)
    assert all(m["tool_call_id"].startswith("call_") for m in tools)


@pytest.mark.asyncio
async def test_long_conversation_at_least_15_messages(monkeypatch, app, empty_pool):
    """A >=15-message Claude Code conversation reaches upstream fully valid.

    Regression for "Message 13 must have 'role' and 'content' fields": the
    message at index 13 is an assistant tool-call turn with no content.
    """
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("final answer")

    upstream = install_upstream(monkeypatch, handler)

    messages = [{"role": "system", "content": "You are helpful."}]
    # Build alternating user / assistant-tool-call / tool-result turns until
    # we have well over 15 messages, including one at index 13.
    for n in range(6):
        messages.append({"role": "user", "content": f"question {n}"})
        messages.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": f"call_{n}",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": f'{{"q":{n}}}'},
                    }
                ],
            }
        )
        messages.append({"tool_call_id": f"call_{n}", "content": f"answer {n}"})
    messages.append({"role": "user", "content": "summarize"})

    assert len(messages) >= 15
    # Sanity: index 13 is the kind of message that used to fail upstream.
    assert "content" not in messages[13] or messages[13].get("content") is None

    body = {"model": "claude-4.0", "stream": False, "messages": messages}
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]
    assert len(sent) >= 15
    _assert_all_valid_upstream(sent)


@pytest.mark.asyncio
async def test_multimodal_array_survives_full_path(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)
    multimodal = [
        {"type": "text", "text": "describe this image"},
        {"type": "image_url", "image_url": {"url": "https://example.com/a.png"}},
    ]
    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [{"role": "user", "content": multimodal}],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]
    # The multimodal content array reaches upstream unchanged.
    user_msg = next(m for m in sent if m["role"] == "user")
    assert user_msg["content"] == multimodal


@pytest.mark.asyncio
async def test_undeterminable_role_returns_local_400_with_index(
    monkeypatch, app, empty_pool
):
    """A message with no determinable role is rejected locally, not forwarded."""
    configure(monkeypatch)
    called = {"upstream": False}

    def handler(req):
        called["upstream"] = True
        return sse("should not be called")

    upstream = install_upstream(monkeypatch, handler)
    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [
            {"role": "user", "content": "hi"},
            {"content": "orphan with no role and no tool markers"},  # index 1
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 400
    assert called["upstream"] is False
    error = response.json()["error"]
    assert error["code"] == "invalid_message"
    assert error["type"] == "invalid_request_error"
    # The message index is identified in the error message.
    assert "1" in error["message"]


@pytest.mark.asyncio
async def test_prepare_payload_normalizes_missing_content(monkeypatch):
    _patch_models(monkeypatch)
    monkeypatch.setattr(codebuddy_router, "get_sanitize_agent_prompt", lambda: False)
    monkeypatch.setattr(codebuddy_router, "get_max_system_prompt_length", lambda: 2000)

    body = {
        "model": "claude-4.0",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"tool_call_id": "c1", "content": "tool output"},
        ],
    }
    payload, _prep = RequestProcessor.prepare_payload(body)
    _assert_all_valid_upstream(payload["messages"])
