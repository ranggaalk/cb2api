"""Integration tests for CodeBuddyAdapterV2 through the FastAPI app (spec §18).

These drive real HTTP requests through the router (which dispatches to the v2
adapter) against a mocked CodeBuddy upstream, so the full lifecycle is covered:
auth → normalize → validate → payload → upstream → stream/aggregate → response.
No real CodeBuddy API key is required.

The upstream is mocked by replacing the adapter transport's ``get_client`` with
an ``httpx.AsyncClient`` backed by ``httpx.MockTransport``. The handler can
capture the exact payload the adapter sends upstream for assertions.
"""
import json

import httpx
import pytest

from src import auth, codebuddy_router
from src.adapters.codebuddy import get_adapter

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"

MODERATION_TEXT = (
    "抱歉，系统检测到您当前输入的信息存在敏感内容，我无法响应您的请求，请检查后重新输入。"
)

SENTINEL = "SENTINEL_MARKER_9F3A"
PORTFOLIO_HTML = f"<!doctype html><html><body><h1>{SENTINEL}</h1></body></html>"


# --------------------------------------------------------------------------- #
# Fixtures / helpers
# --------------------------------------------------------------------------- #


@pytest.fixture
def app():
    from fastapi import FastAPI

    application = FastAPI()
    application.include_router(codebuddy_router.router, prefix="/codebuddy")
    return application


def configure(monkeypatch, unknown_policy="passthrough", sanitize=False):
    # Route to the v2 adapter.
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_adapter_version", lambda: "v2")
    # Passthrough auth: the client Bearer token is the upstream key.
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "passthrough")
    monkeypatch.setattr(auth, "get_server_password", lambda: RELAY_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: ADMIN_PASSWORD)

    import config as root_config
    from src.adapters.codebuddy import config as adapter_config

    monkeypatch.setattr(root_config, "get_codebuddy_unknown_model_policy", lambda: unknown_policy)
    monkeypatch.setattr(root_config, "get_sanitize_agent_prompt", lambda: sanitize)
    monkeypatch.setattr(root_config, "get_codebuddy_request_profile", lambda: "web")
    monkeypatch.setattr(root_config, "get_upstream_api_key_header", lambda: "bearer")
    # Ensure the adapter's settings loader sees the patched getters.
    assert adapter_config.load_adapter_settings().unknown_model_policy == unknown_policy


def install_upstream(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    adapter = get_adapter()

    async def get_client(_settings):
        return client

    monkeypatch.setattr(adapter.transport, "get_client", get_client)
    return client


def sse_body(chunks, done=True, model="auto-chat"):
    parts = []
    for delta, finish in chunks:
        choice = {"index": 0, "delta": delta}
        if finish is not None:
            choice["finish_reason"] = finish
        obj = {
            "id": "chat-v2",
            "object": "chat.completion.chunk",
            "model": model,
            "choices": [choice],
        }
        parts.append("data: " + json.dumps(obj, ensure_ascii=False))
    text = "\n\n".join(parts) + "\n\n"
    if done:
        text += "data: [DONE]\n\n"
    return httpx.Response(200, text=text, headers={"content-type": "text/event-stream"})


async def post_chat(app, token, body):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/codebuddy/v1/chat/completions",
            headers={"Authorization": f"Bearer {token}"},
            json=body,
        )


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
# Basic chat (spec §18.1-4)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_stream_basic(monkeypatch, app):
    configure(monkeypatch)
    install_upstream(monkeypatch, lambda _r: sse_body([({"content": "Halo!"}, "stop")]))

    resp = await post_chat(
        app, KEY_A, {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "Halo!"
    assert data["choices"][0]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_stream_basic_single_done(monkeypatch, app):
    configure(monkeypatch)
    install_upstream(
        monkeypatch,
        lambda _r: sse_body(
            [({"role": "assistant"}, None), ({"content": "Hello"}, None), ({"content": " world"}, None), ({}, "stop")]
        ),
    )

    resp = await post_chat(
        app,
        KEY_A,
        {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    text = resp.text
    assert text.count("[DONE]") == 1
    events = parse_stream_events(text)
    streamed = "".join(
        (e["choices"][0].get("delta", {}) or {}).get("content", "") for e in events
    )
    assert streamed == "Hello world"


@pytest.mark.asyncio
async def test_sse_headers_present(monkeypatch, app):
    configure(monkeypatch)
    install_upstream(monkeypatch, lambda _r: sse_body([({"content": "hi"}, "stop")]))
    resp = await post_chat(
        app,
        KEY_A,
        {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}], "stream": True},
    )
    assert resp.headers.get("cache-control") == "no-cache, no-transform"
    assert resp.headers.get("x-accel-buffering") == "no"


# --------------------------------------------------------------------------- #
# Model mapping (spec §18.5-8)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_opus_label_not_rewritten_to_auto_chat(monkeypatch, app):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return sse_body([({"content": "ok"}, "stop")])

    install_upstream(monkeypatch, handler)
    await post_chat(
        app,
        KEY_A,
        {"model": "claude-opus-4.7-1m", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert seen["payload"]["model"] == "claude-opus-4.7-1m"
    assert seen["payload"]["model"] != "auto-chat"


@pytest.mark.asyncio
async def test_unknown_model_reject_returns_400(monkeypatch, app):
    configure(monkeypatch, unknown_policy="reject")
    install_upstream(monkeypatch, lambda _r: sse_body([({"content": "x"}, "stop")]))
    resp = await post_chat(
        app, KEY_A, {"model": "totally-unknown", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "unknown_model"


# --------------------------------------------------------------------------- #
# Tool workflow (spec §18.14-21, §18.35-40)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_read_workflow_delivers_full_file_upstream(monkeypatch, app):
    """Read → tool_result(portfolio.html) → the NEXT request must carry the
    complete file to the model as a role:tool message."""
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return sse_body([({"content": "I can see the file."}, "stop")])

    install_upstream(monkeypatch, handler)

    body = {
        "model": "auto-chat",
        "messages": [
            {"role": "user", "content": "read portfolio.html"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Reading."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Read", "input": {"file_path": "portfolio.html"}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": PORTFOLIO_HTML}
                ],
            },
        ],
    }
    resp = await post_chat(app, KEY_A, body)
    assert resp.status_code == 200

    upstream_msgs = seen["payload"]["messages"]
    # The assistant tool_use became an OpenAI tool_calls message.
    asst = next(m for m in upstream_msgs if m.get("tool_calls"))
    assert asst["tool_calls"][0]["id"] == "toolu_1"
    assert asst["tool_calls"][0]["function"]["name"] == "Read"
    # The tool_result became a role:tool message carrying the FULL file.
    tool_msg = next(m for m in upstream_msgs if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "toolu_1"
    assert SENTINEL in tool_msg["content"]
    assert tool_msg["content"] == PORTFOLIO_HTML


@pytest.mark.asyncio
async def test_read_grep_edit_read_loop_preserves_every_result(monkeypatch, app):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return sse_body([({"content": "done"}, "stop")])

    install_upstream(monkeypatch, handler)

    def tu(i, name):
        return {
            "role": "assistant",
            "content": [{"type": "tool_use", "id": f"toolu_{i}", "name": name, "input": {"n": i}}],
        }

    def tr(i, payload):
        return {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": f"toolu_{i}", "content": payload}],
        }

    body = {
        "model": "auto-chat",
        "messages": [
            {"role": "user", "content": "improve portfolio"},
            tu(1, "Read"), tr(1, "READ_RESULT_1"),
            tu(2, "Grep"), tr(2, "GREP_RESULT_2"),
            tu(3, "Edit"), tr(3, "EDIT_RESULT_3"),
            tu(4, "Read"), tr(4, "READ_RESULT_4"),
        ],
    }
    resp = await post_chat(app, KEY_A, body)
    assert resp.status_code == 200

    tool_msgs = [m for m in seen["payload"]["messages"] if m.get("role") == "tool"]
    contents = [m["content"] for m in tool_msgs]
    assert contents == ["READ_RESULT_1", "GREP_RESULT_2", "EDIT_RESULT_3", "READ_RESULT_4"]
    # Each tool result is paired to its originating tool call id.
    assert [m["tool_call_id"] for m in tool_msgs] == ["toolu_1", "toolu_2", "toolu_3", "toolu_4"]


@pytest.mark.asyncio
async def test_multimodal_array_survives(monkeypatch, app):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return sse_body([({"content": "ok"}, "stop")])

    install_upstream(monkeypatch, handler)

    image_block = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}
    body = {
        "model": "auto-chat",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "what is this"}, image_block]}
        ],
    }
    resp = await post_chat(app, KEY_A, body)
    assert resp.status_code == 200
    user_msg = seen["payload"]["messages"][-1]
    assert isinstance(user_msg["content"], list)
    assert image_block in user_msg["content"]


@pytest.mark.asyncio
async def test_orphan_tool_result_returns_local_400(monkeypatch, app):
    configure(monkeypatch)
    # Upstream must never be called; make it explode if it is.
    def handler(_req):
        raise AssertionError("upstream must not be called for a local 400")

    install_upstream(monkeypatch, handler)

    body = {
        "model": "auto-chat",
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "ghost", "content": "x"}]},
        ],
    }
    resp = await post_chat(app, KEY_A, body)
    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "invalid_tool_conversation"


# --------------------------------------------------------------------------- #
# Tools passthrough + schema (spec §8, §18)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_tools_are_forwarded_with_names_preserved(monkeypatch, app):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["payload"] = json.loads(req.content)
        return sse_body([({"content": "ok"}, "stop")])

    install_upstream(monkeypatch, handler)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "Read",
                "parameters": {
                    "type": "object",
                    "properties": {"file_path": {"type": "string"}},
                    "required": ["file_path"],
                },
            },
        }
    ]
    body = {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}], "tools": tools}
    resp = await post_chat(app, KEY_A, body)
    assert resp.status_code == 200
    fwd = seen["payload"]["tools"]
    assert fwd[0]["function"]["name"] == "Read"
    assert fwd[0]["function"]["parameters"]["required"] == ["file_path"]


# --------------------------------------------------------------------------- #
# Moderation (spec §18)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_stream_moderation_is_content_filter(monkeypatch, app):
    configure(monkeypatch)
    install_upstream(monkeypatch, lambda _r: sse_body([({"content": MODERATION_TEXT}, "stop")]))
    resp = await post_chat(
        app, KEY_A, {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert resp.status_code == 400
    assert resp.headers.get("X-CodeBuddy-Moderation") == "true"
    assert resp.json()["error"]["type"] == "content_filter"


# --------------------------------------------------------------------------- #
# Streaming tool calls end-to-end (spec §9)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_non_stream_fragmented_tool_args_reconstructed(monkeypatch, app):
    configure(monkeypatch)
    install_upstream(
        monkeypatch,
        lambda _r: sse_body(
            [
                ({"tool_calls": [{"index": 0, "id": "tooluse_1", "function": {"name": "Read", "arguments": '{"file_'}}]}, None),
                ({"tool_calls": [{"index": 0, "function": {"arguments": 'path":"a.html"}'}}]}, None),
                ({}, "tool_calls"),
            ]
        ),
    )
    resp = await post_chat(
        app, KEY_A, {"model": "auto-chat", "messages": [{"role": "user", "content": "read a"}]}
    )
    assert resp.status_code == 200
    msg = resp.json()["choices"][0]["message"]
    tc = msg["tool_calls"][0]
    assert tc["id"] == "call_1"  # tooluse_ normalized to call_
    assert json.loads(tc["function"]["arguments"]) == {"file_path": "a.html"}
    assert resp.json()["choices"][0]["finish_reason"] == "tool_calls"


@pytest.mark.asyncio
async def test_security_key_never_appears_in_response(monkeypatch, app):
    configure(monkeypatch)
    install_upstream(monkeypatch, lambda _r: sse_body([({"content": "ok"}, "stop")]))
    resp = await post_chat(
        app, KEY_A, {"model": "auto-chat", "messages": [{"role": "user", "content": "hi"}]}
    )
    assert KEY_A not in resp.text
