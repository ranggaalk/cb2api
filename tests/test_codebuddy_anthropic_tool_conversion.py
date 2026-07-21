"""
Anthropic tool-block -> OpenAI conversion regression tests.

Reproduces and locks the fix for Claude Code tool-result context loss:

    - Claude Code runs Read on portfolio.html and the client sends the file
      back as an Anthropic ``tool_result`` block.
    - Without conversion, the block is forwarded to CodeBuddy verbatim, the
      OpenAI-speaking upstream never sees the file text, and the model reports
      "the file contents are not in context" and falls back to shell commands.

``convert_anthropic_messages_to_openai`` rewrites Anthropic content-block
messages into OpenAI-shaped messages before the upstream call:

  * assistant ``tool_use`` blocks -> a ``tool_calls`` list (arguments as a JSON
    string), keeping the ``tool_use.id`` verbatim;
  * user ``tool_result`` blocks -> standalone ``role: "tool"`` messages whose
    ``tool_call_id`` is the verbatim ``tool_use_id`` and whose ``content`` is
    the complete tool output.

These tests cover the exact portfolio.html workflow, a multi-step
Read -> Grep -> Edit -> Read loop, multimodal preservation, that system-prompt
sanitization never touches tool results, and that payload allowlisting keeps
content blocks intact.

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
    convert_anthropic_messages_to_openai,
    sanitize_messages,
    validate_upstream_messages,
)

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"

AVAILABLE_MODELS = ["claude-4.0", "gpt-5", "auto-chat"]

# A distinctive file body so we can assert it survives the pipeline exactly.
PORTFOLIO_HTML = (
    "<!DOCTYPE html>\n<html>\n<head><title>My Portfolio</title></head>\n"
    "<body>\n  <h1>Jane Developer</h1>\n  <p>Full-stack engineer.</p>\n"
    "  <!-- SENTINEL_MARKER_9F3A -->\n</body>\n</html>\n"
)


# --------------------------------------------------------------------------- #
# Fixtures / helpers (mirror the other integration suites)
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


def configure(monkeypatch, sanitize=False):
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "passthrough")
    monkeypatch.setattr(auth, "get_server_password", lambda: RELAY_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: ADMIN_PASSWORD)
    monkeypatch.setattr(codebuddy_router, "get_upstream_api_key_header", lambda: "both")
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_request_profile", lambda: "web")
    monkeypatch.setattr(codebuddy_router, "get_sanitize_agent_prompt", lambda: sanitize)
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


def _tool_use_msg(text, tool_id, name, tool_input):
    """An Anthropic assistant turn: optional text + a tool_use block."""
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.append(
        {"type": "tool_use", "id": tool_id, "name": name, "input": tool_input}
    )
    return {"role": "assistant", "content": content}


def _tool_result_msg(tool_use_id, result):
    """An Anthropic user turn carrying a tool_result block."""
    return {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": tool_use_id, "content": result}
        ],
    }


def _find_tool_messages(messages):
    return [m for m in messages if m.get("role") == "tool"]


# --------------------------------------------------------------------------- #
# Unit: converter shape
# --------------------------------------------------------------------------- #


def test_assistant_tool_use_becomes_openai_tool_calls():
    messages = [
        _tool_use_msg("I'll read it.", "toolu_01", "Read", {"file_path": "portfolio.html"}),
    ]
    out = convert_anthropic_messages_to_openai(messages)
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "I'll read it."
    assert isinstance(msg["tool_calls"], list) and len(msg["tool_calls"]) == 1
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_01"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "Read"
    # arguments is a JSON *string* and round-trips.
    assert json.loads(tc["function"]["arguments"]) == {"file_path": "portfolio.html"}


def test_assistant_tool_use_without_text_gets_empty_content():
    messages = [_tool_use_msg("", "toolu_x", "Read", {"file_path": "a"})]
    out = convert_anthropic_messages_to_openai(messages)
    assert out[0]["content"] == ""
    assert out[0]["tool_calls"][0]["id"] == "toolu_x"


def test_tool_result_becomes_standalone_tool_message():
    messages = [_tool_result_msg("toolu_01", PORTFOLIO_HTML)]
    out = convert_anthropic_messages_to_openai(messages)
    assert len(out) == 1
    assert out[0] == {
        "role": "tool",
        "tool_call_id": "toolu_01",
        "content": PORTFOLIO_HTML,
    }


def test_tool_result_with_block_list_content_is_flattened():
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_2",
                    "content": [{"type": "text", "text": PORTFOLIO_HTML}],
                }
            ],
        }
    ]
    out = convert_anthropic_messages_to_openai(messages)
    assert out[0]["role"] == "tool"
    assert out[0]["content"] == PORTFOLIO_HTML


def test_id_relationship_preserved_end_to_end():
    messages = [
        {"role": "user", "content": "read portfolio.html"},
        _tool_use_msg("Reading.", "toolu_ABC", "Read", {"file_path": "portfolio.html"}),
        _tool_result_msg("toolu_ABC", PORTFOLIO_HTML),
    ]
    out = convert_anthropic_messages_to_openai(messages)
    assistant = next(m for m in out if m.get("tool_calls"))
    tool = next(m for m in out if m.get("role") == "tool")
    # tool_use.id == tool_call.id == tool_result tool_call_id.
    assert assistant["tool_calls"][0]["id"] == "toolu_ABC"
    assert tool["tool_call_id"] == "toolu_ABC"


def test_tool_result_not_merged_into_unrelated_user_text():
    # A user turn that contains BOTH a tool_result and trailing text.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "toolu_9", "content": "RESULT"},
                {"type": "text", "text": "now edit it"},
            ],
        }
    ]
    out = convert_anthropic_messages_to_openai(messages)
    # The tool result is its own message; the text is a separate user message.
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    user_msgs = [m for m in out if m.get("role") == "user"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "RESULT"
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == [{"type": "text", "text": "now edit it"}]


def test_plain_string_content_passed_through():
    messages = [{"role": "user", "content": "hello"}]
    out = convert_anthropic_messages_to_openai(messages)
    assert out == messages


def test_multimodal_image_array_preserved_by_converter():
    multimodal = [
        {"type": "text", "text": "describe"},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}},
    ]
    messages = [{"role": "user", "content": multimodal}]
    out = convert_anthropic_messages_to_openai(messages)
    # No tool blocks -> preserved verbatim, never replaced with "".
    assert out[0]["content"] == multimodal


def test_no_duplication_of_tool_calls_or_results():
    messages = [
        _tool_use_msg("", "toolu_1", "Read", {"file_path": "a"}),
        _tool_result_msg("toolu_1", "A"),
        _tool_use_msg("", "toolu_2", "Read", {"file_path": "b"}),
        _tool_result_msg("toolu_2", "B"),
    ]
    out = convert_anthropic_messages_to_openai(messages)
    all_call_ids = [
        tc["id"] for m in out for tc in (m.get("tool_calls") or [])
    ]
    all_result_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert all_call_ids == ["toolu_1", "toolu_2"]
    assert all_result_ids == ["toolu_1", "toolu_2"]


# --------------------------------------------------------------------------- #
# Unit: validation (requirement 12/13)
# --------------------------------------------------------------------------- #


def test_validate_rejects_orphan_tool_result():
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "tool", "tool_call_id": "toolu_missing", "content": "x"},  # index 1
    ]
    with pytest.raises(MessageNormalizationError) as excinfo:
        validate_upstream_messages(messages)
    assert excinfo.value.index == 1


def test_validate_rejects_unconverted_anthropic_block():
    messages = [
        {
            "role": "user",
            "content": [{"type": "tool_result", "tool_use_id": "t", "content": "x"}],
        }
    ]
    with pytest.raises(MessageNormalizationError) as excinfo:
        validate_upstream_messages(messages)
    assert excinfo.value.index == 0


def test_validate_rejects_non_json_arguments():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{not json"}}
            ],
        }
    ]
    with pytest.raises(MessageNormalizationError) as excinfo:
        validate_upstream_messages(messages)
    assert excinfo.value.index == 0


def test_validate_accepts_well_formed_pairing():
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "ok"},
    ]
    # Should not raise.
    validate_upstream_messages(messages)


# --------------------------------------------------------------------------- #
# Requirement 17: sanitization only touches system messages
# --------------------------------------------------------------------------- #


def test_sanitization_never_removes_tool_result_content():
    messages = [
        {"role": "system", "content": "You are Claude Code, the official CLI. " + "x" * 3000},
        {"role": "tool", "tool_call_id": "c1", "content": PORTFOLIO_HTML},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{}"}}
        ]},
    ]
    out, changed = sanitize_messages(messages, enabled=True, max_system_prompt_length=2000)
    assert changed is True  # the system prompt was replaced
    # The tool result content is untouched.
    tool_msg = next(m for m in out if m.get("role") == "tool")
    assert tool_msg["content"] == PORTFOLIO_HTML
    # The assistant tool call is untouched.
    asst = next(m for m in out if m.get("tool_calls"))
    assert asst["tool_calls"][0]["id"] == "c1"


# --------------------------------------------------------------------------- #
# Requirement 14: the exact portfolio.html workflow (integration)
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_portfolio_read_workflow_delivers_full_file(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("I can see the portfolio file.")

    upstream = install_upstream(monkeypatch, handler)

    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [
            {"role": "user", "content": "please read portfolio.html"},
            _tool_use_msg("I'll read it.", "toolu_read1", "Read", {"file_path": "portfolio.html"}),
            _tool_result_msg("toolu_read1", PORTFOLIO_HTML),
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]

    # The assistant Read tool call reached upstream as an OpenAI tool call.
    asst = next(m for m in sent if m.get("tool_calls"))
    assert asst["tool_calls"][0]["id"] == "toolu_read1"
    assert asst["tool_calls"][0]["function"]["name"] == "Read"

    # The file contents are present, complete, and unmodified in a tool message.
    tool_msgs = _find_tool_messages(sent)
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "toolu_read1"
    assert tool_msgs[0]["content"] == PORTFOLIO_HTML
    assert "SENTINEL_MARKER_9F3A" in tool_msgs[0]["content"]


@pytest.mark.asyncio
async def test_portfolio_workflow_then_edit_call_converts(monkeypatch, app, empty_pool):
    # After the file is read, the model proposes an Edit call: that Anthropic
    # tool_use must also convert cleanly.
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("Editing now.")

    upstream = install_upstream(monkeypatch, handler)

    edit_input = {"file_path": "portfolio.html", "old_string": "Jane Developer", "new_string": "Jane D."}
    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [
            {"role": "user", "content": "read then rename the heading"},
            _tool_use_msg("Reading.", "toolu_read1", "Read", {"file_path": "portfolio.html"}),
            _tool_result_msg("toolu_read1", PORTFOLIO_HTML),
            _tool_use_msg("Now editing.", "toolu_edit1", "Edit", edit_input),
            _tool_result_msg("toolu_edit1", "Edit applied."),
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]
    call_ids = [tc["id"] for m in sent for tc in (m.get("tool_calls") or [])]
    assert call_ids == ["toolu_read1", "toolu_edit1"]
    # The Edit arguments survived as valid JSON.
    edit_call = next(
        tc for m in sent for tc in (m.get("tool_calls") or []) if tc["id"] == "toolu_edit1"
    )
    assert json.loads(edit_call["function"]["arguments"]) == edit_input
    # Both tool results are present and none dropped.
    tool_msgs = _find_tool_messages(sent)
    assert [t["tool_call_id"] for t in tool_msgs] == ["toolu_read1", "toolu_edit1"]
    assert tool_msgs[0]["content"] == PORTFOLIO_HTML


# --------------------------------------------------------------------------- #
# Requirement 15: multi-step Read -> Grep -> Edit -> Read, nothing dropped
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_read_grep_edit_read_loop_preserves_every_result(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("done")

    upstream = install_upstream(monkeypatch, handler)

    steps = [
        ("toolu_read1", "Read", {"file_path": "portfolio.html"}, PORTFOLIO_HTML),
        ("toolu_grep1", "Grep", {"pattern": "Jane", "path": "portfolio.html"}, "portfolio.html:4:  <h1>Jane Developer</h1>"),
        ("toolu_edit1", "Edit", {"file_path": "portfolio.html", "old_string": "Jane", "new_string": "Janet"}, "Edit applied."),
        ("toolu_read2", "Read", {"file_path": "portfolio.html"}, PORTFOLIO_HTML.replace("Jane", "Janet")),
    ]

    messages = [{"role": "user", "content": "read, grep, edit, then re-read"}]
    for tool_id, name, tool_input, result in steps:
        messages.append(_tool_use_msg("", tool_id, name, tool_input))
        messages.append(_tool_result_msg(tool_id, result))

    body = {"model": "claude-4.0", "stream": False, "messages": messages}
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]

    # Every tool call id appears exactly once, in order.
    call_ids = [tc["id"] for m in sent for tc in (m.get("tool_calls") or [])]
    assert call_ids == [s[0] for s in steps]

    # Every tool result is present, in order, with its exact content.
    tool_msgs = _find_tool_messages(sent)
    assert [t["tool_call_id"] for t in tool_msgs] == [s[0] for s in steps]
    for tmsg, (_id, _name, _input, result) in zip(tool_msgs, steps):
        assert tmsg["content"] == result

    # No duplication: as many tool results as tool calls.
    assert len(tool_msgs) == len(call_ids) == 4


# --------------------------------------------------------------------------- #
# Requirement 16/18: multimodal + allowlist preserve content blocks
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_multimodal_array_survives_full_path(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)
    multimodal = [
        {"type": "text", "text": "what is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
    ]
    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [
            {"role": "user", "content": multimodal},
            {"role": "assistant", "content": "A cat."},
            {"role": "user", "content": "thanks"},
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    sent = seen["body"]["messages"]
    first_user = next(m for m in sent if m.get("role") == "user")
    # The multimodal content array reaches upstream unchanged (req 16, 18).
    assert first_user["content"] == multimodal


@pytest.mark.asyncio
async def test_allowlist_keeps_tool_calls_inside_messages(monkeypatch, app, empty_pool):
    # Top-level allowlist drops unsupported fields but must not strip the
    # content/tool_calls INSIDE the message objects.
    configure(monkeypatch)
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return sse("ok")

    upstream = install_upstream(monkeypatch, handler)
    body = {
        "model": "claude-4.0",
        "stream": False,
        "temperature": 0.5,  # unsupported top-level field, must be dropped
        "metadata": {"ui": "shiteru"},
        "messages": [
            {"role": "user", "content": "read it"},
            _tool_use_msg("Reading.", "toolu_1", "Read", {"file_path": "portfolio.html"}),
            _tool_result_msg("toolu_1", PORTFOLIO_HTML),
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 200
    body_sent = seen["body"]
    # Unsupported top-level fields were dropped...
    assert "temperature" not in body_sent
    assert "metadata" not in body_sent
    # ...but the tool call and tool result inside the messages survived.
    assert any(m.get("tool_calls") for m in body_sent["messages"])
    tool_msgs = _find_tool_messages(body_sent["messages"])
    assert tool_msgs and tool_msgs[0]["content"] == PORTFOLIO_HTML


# --------------------------------------------------------------------------- #
# Requirement 13: orphan tool result -> local 400 with index, not sent upstream
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_orphan_tool_result_returns_local_400(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    called = {"upstream": False}

    def handler(req):
        called["upstream"] = True
        return sse("should not be called")

    upstream = install_upstream(monkeypatch, handler)
    # A tool_result whose tool_use_id has no preceding tool_use.
    body = {
        "model": "claude-4.0",
        "stream": False,
        "messages": [
            {"role": "user", "content": "hi"},
            _tool_result_msg("toolu_orphan", "dangling result"),
        ],
    }
    response = await post(app, KEY_A, body)
    await upstream.aclose()

    assert response.status_code == 400
    assert called["upstream"] is False
    error = response.json()["error"]
    assert error["code"] == "invalid_message"
    assert error["type"] == "invalid_request_error"
