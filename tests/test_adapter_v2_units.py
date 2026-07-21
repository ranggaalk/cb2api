"""Unit tests for CodeBuddyAdapterV2 pure modules (spec §18).

These exercise the deterministic building blocks with no HTTP: model
resolution, Anthropic→OpenAI message conversion, normalization, tool-conversation
validation, tool-schema sanitization, the streaming tool-call state machine,
SSE aggregation, and response shaping. They require no CodeBuddy API key.
"""
import json

import pytest

from src.adapters.codebuddy import response_mapper
from src.adapters.codebuddy.config import AdapterSettings
from src.adapters.codebuddy.errors import (
    InvalidToolArgumentsError,
    InvalidToolConversationError,
    MessageNormalizationError,
    UnknownModelError,
)
from src.adapters.codebuddy.message_normalizer import (
    convert_anthropic_messages_to_openai,
    normalize_messages_for_upstream,
)
from src.adapters.codebuddy.request_mapper import build_payload, dropped_fields, resolve_model
from src.adapters.codebuddy.stream_decoder import (
    SSELineBuffer,
    StreamAggregator,
    parse_sse_data_line,
)
from src.adapters.codebuddy.tool_call_state import ToolCallAccumulator
from src.adapters.codebuddy.tool_schema_adapter import sanitize_tools
from src.adapters.codebuddy.validation import validate_conversation


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def make_settings(**overrides) -> AdapterSettings:
    base = dict(
        adapter_version="v2",
        request_profile="web",
        upstream_api_key_header="bearer",
        sanitize_agent_prompt=False,
        max_system_prompt_length=2000,
        default_model="auto-chat",
        unknown_model_policy="passthrough",
        connect_timeout=30.0,
        pool_timeout=30.0,
        write_timeout=60.0,
        headers_timeout=300.0,
        first_chunk_timeout=300.0,
        stream_idle_timeout=600.0,
        max_concurrent_upstream_requests=20,
        upstream_queue_timeout=60.0,
        warn_total_content_length=50000,
        warn_message_count=40,
        warn_tool_count=30,
        api_endpoint="https://www.codebuddy.ai",
    )
    base.update(overrides)
    return AdapterSettings(**base)


AVAILABLE = ["claude-4.0", "gpt-5", "auto-chat"]


def _tool_use_msg(text, tool_id, name, tool_input):
    content = []
    if text:
        content.append({"type": "text", "text": text})
    content.append({"type": "tool_use", "id": tool_id, "name": name, "input": tool_input})
    return {"role": "assistant", "content": content}


def _tool_result_msg(tool_use_id, result, extra_text=None):
    content = [{"type": "tool_result", "tool_use_id": tool_use_id, "content": result}]
    if extra_text:
        content.append({"type": "text", "text": extra_text})
    return {"role": "user", "content": content}


# --------------------------------------------------------------------------- #
# Model mapping (spec §18.5-8)
# --------------------------------------------------------------------------- #


def test_opus_label_is_not_rewritten_to_auto_chat():
    settings = make_settings()
    resolved = resolve_model("claude-opus-4.7-1m", settings, AVAILABLE, {})
    assert resolved.mapped == "claude-opus-4.7-1m"
    assert resolved.source == "passthrough"
    assert resolved.mapped != "auto-chat"


def test_unknown_model_passthrough_by_default():
    settings = make_settings(unknown_model_policy="passthrough")
    resolved = resolve_model("mystery-model", settings, AVAILABLE, {})
    assert resolved.mapped == "mystery-model"
    assert resolved.source == "passthrough"


def test_unknown_model_reject_policy_raises():
    settings = make_settings(unknown_model_policy="reject")
    with pytest.raises(UnknownModelError):
        resolve_model("mystery-model", settings, AVAILABLE, {})


def test_unknown_model_default_policy_falls_back():
    settings = make_settings(unknown_model_policy="default", default_model="auto-chat")
    resolved = resolve_model("mystery-model", settings, AVAILABLE, {})
    assert resolved.mapped == "auto-chat"
    assert resolved.source == "default"


def test_alias_is_mapped_case_insensitively():
    settings = make_settings()
    resolved = resolve_model(
        "Claude Opus 4.7", settings, AVAILABLE, {"claude opus 4.7": "claude-4.0"}
    )
    assert resolved.mapped == "claude-4.0"
    assert resolved.source == "alias"


def test_exact_auto_chat_is_exact_not_default():
    settings = make_settings()
    resolved = resolve_model("auto-chat", settings, AVAILABLE, {})
    assert resolved.source == "exact"


def test_missing_model_uses_default_under_every_policy():
    for policy in ("passthrough", "reject", "default"):
        settings = make_settings(unknown_model_policy=policy, default_model="auto-chat")
        resolved = resolve_model(None, settings, AVAILABLE, {})
        assert resolved.mapped == "auto-chat"
        assert resolved.source == "default_empty"


def test_dropped_fields_excludes_allowlisted():
    body = {
        "model": "x",
        "messages": [],
        "stream": True,
        "tools": [],
        "tool_choice": "auto",
        "response_format": {"type": "json"},
        "reasoning_effort": "high",
    }
    assert dropped_fields(body) == ["reasoning_effort", "response_format"]


def test_build_payload_always_streams_upstream():
    body = {"stream": False, "tools": [{"x": 1}], "tool_choice": "auto"}
    payload = build_payload(body, "auto-chat", [{"role": "user", "content": "hi"}])
    assert payload["stream"] is True
    assert payload["model"] == "auto-chat"
    assert payload["tools"] == [{"x": 1}]
    assert payload["tool_choice"] == "auto"


# --------------------------------------------------------------------------- #
# Messages: role/content guarantees (spec §18.9-13)
# --------------------------------------------------------------------------- #


def test_string_content_survives_normalization():
    out = normalize_messages_for_upstream([{"role": "user", "content": "hello"}])
    assert out[0]["content"] == "hello"


def test_array_content_is_not_replaced_with_empty_string():
    multimodal = [
        {"type": "text", "text": "look"},
        {"type": "image", "source": {"type": "base64", "data": "AAAA"}},
    ]
    out = convert_anthropic_messages_to_openai(
        [{"role": "user", "content": multimodal}]
    )
    assert out[0]["content"] == multimodal  # preserved verbatim, not ""


def test_assistant_tool_calls_get_empty_content():
    out = normalize_messages_for_upstream(
        [{"role": "assistant", "tool_calls": [{"id": "c1"}]}]
    )
    assert out[0]["content"] == ""
    assert out[0]["role"] == "assistant"


def test_missing_role_inferred_from_tool_call_id():
    out = normalize_messages_for_upstream(
        [
            {"role": "assistant", "tool_calls": [{"id": "c1"}], "content": ""},
            {"tool_call_id": "c1", "content": "result"},
        ]
    )
    assert out[1]["role"] == "tool"


def test_undeterminable_role_raises_indexed_error():
    with pytest.raises(MessageNormalizationError) as exc:
        normalize_messages_for_upstream(
            [{"role": "user", "content": "ok"}, {"content": "who am i"}]
        )
    assert exc.value.index == 1


def test_every_normalized_message_has_role_and_content():
    out = normalize_messages_for_upstream(
        [
            {"role": "system", "content": "s"},
            {"role": "user", "content": "u"},
            {"role": "assistant", "tool_calls": [{"id": "c1"}]},
            {"tool_call_id": "c1", "content": "r"},
        ]
    )
    for msg in out:
        assert msg.get("role")
        assert "content" in msg


# --------------------------------------------------------------------------- #
# Tool conversion (spec §18.14-21)
# --------------------------------------------------------------------------- #


def test_tool_use_becomes_openai_tool_calls():
    out = convert_anthropic_messages_to_openai(
        [_tool_use_msg("Reading", "toolu_1", "Read", {"file_path": "a.html"})]
    )
    assert len(out) == 1
    msg = out[0]
    assert msg["role"] == "assistant"
    assert msg["content"] == "Reading"
    tc = msg["tool_calls"][0]
    assert tc["id"] == "toolu_1"
    assert tc["type"] == "function"
    assert tc["function"]["name"] == "Read"
    assert json.loads(tc["function"]["arguments"]) == {"file_path": "a.html"}


def test_tool_result_becomes_role_tool():
    out = convert_anthropic_messages_to_openai(
        [_tool_result_msg("toolu_1", "<html>full file</html>")]
    )
    assert len(out) == 1
    assert out[0]["role"] == "tool"
    assert out[0]["tool_call_id"] == "toolu_1"
    assert out[0]["content"] == "<html>full file</html>"


def test_tool_use_id_matches_tool_result_id():
    msgs = [
        _tool_use_msg("", "toolu_ABC", "Read", {"file_path": "a"}),
        _tool_result_msg("toolu_ABC", "data"),
    ]
    out = convert_anthropic_messages_to_openai(msgs)
    assert out[0]["tool_calls"][0]["id"] == out[1]["tool_call_id"] == "toolu_ABC"


def test_assistant_tool_call_content_empty_when_no_text():
    out = convert_anthropic_messages_to_openai(
        [_tool_use_msg("", "toolu_1", "Read", {"file_path": "a"})]
    )
    assert out[0]["content"] == ""


def test_tool_result_preserves_full_file_contents():
    big = "<html>" + ("x" * 10000) + "</html>"
    out = convert_anthropic_messages_to_openai([_tool_result_msg("toolu_1", big)])
    assert out[0]["content"] == big


def test_tool_result_not_merged_into_user_message():
    # A user turn mixing text + tool_result must split, never merge the result
    # into the text message.
    out = convert_anthropic_messages_to_openai(
        [_tool_result_msg("toolu_1", "RESULT", extra_text="and please continue")]
    )
    tool_msgs = [m for m in out if m.get("role") == "tool"]
    user_msgs = [m for m in out if m.get("role") == "user"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["content"] == "RESULT"
    # The trailing user text is a separate message, not merged with the result.
    assert user_msgs and all("RESULT" not in str(m["content"]) for m in user_msgs)


def test_multiple_tool_calls_in_one_assistant_message():
    content = [
        {"type": "tool_use", "id": "t1", "name": "Read", "input": {"p": 1}},
        {"type": "tool_use", "id": "t2", "name": "Grep", "input": {"q": "x"}},
    ]
    out = convert_anthropic_messages_to_openai(
        [{"role": "assistant", "content": content}]
    )
    assert len(out) == 1
    ids = [tc["id"] for tc in out[0]["tool_calls"]]
    assert ids == ["t1", "t2"]


def test_multiple_tool_results_in_one_user_array():
    content = [
        {"type": "tool_result", "tool_use_id": "t1", "content": "r1"},
        {"type": "tool_result", "tool_use_id": "t2", "content": "r2"},
    ]
    out = convert_anthropic_messages_to_openai([{"role": "user", "content": content}])
    assert [m["tool_call_id"] for m in out] == ["t1", "t2"]
    assert [m["content"] for m in out] == ["r1", "r2"]


def test_no_duplication_of_tool_calls_or_results():
    msgs = [
        _tool_use_msg("", "t1", "Read", {"p": 1}),
        _tool_result_msg("t1", "r1"),
    ]
    out = convert_anthropic_messages_to_openai(msgs)
    all_tc_ids = [tc["id"] for m in out for tc in m.get("tool_calls", [])]
    all_result_ids = [m["tool_call_id"] for m in out if m.get("role") == "tool"]
    assert all_tc_ids == ["t1"]
    assert all_result_ids == ["t1"]


def test_tool_result_content_list_is_flattened_completely():
    content = [
        {
            "type": "tool_result",
            "tool_use_id": "t1",
            "content": [
                {"type": "text", "text": "part1 "},
                {"type": "text", "text": "part2"},
            ],
        }
    ]
    out = convert_anthropic_messages_to_openai([{"role": "user", "content": content}])
    assert out[0]["content"] == "part1 part2"


# --------------------------------------------------------------------------- #
# Validation (spec §7, §18)
# --------------------------------------------------------------------------- #


def test_validate_accepts_well_formed_tool_conversation():
    msgs = [
        {"role": "user", "content": "read a"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": '{"p":1}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "t1", "content": "data"},
    ]
    validate_conversation(msgs)  # must not raise


def test_validate_rejects_orphan_tool_result():
    msgs = [{"role": "tool", "tool_call_id": "ghost", "content": "x"}]
    with pytest.raises(InvalidToolConversationError) as exc:
        validate_conversation(msgs)
    assert exc.value.index == 0


def test_validate_rejects_non_json_arguments():
    msgs = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "t1",
                    "type": "function",
                    "function": {"name": "Read", "arguments": "{not json"},
                }
            ],
        }
    ]
    with pytest.raises(InvalidToolConversationError):
        validate_conversation(msgs)


def test_validate_rejects_leftover_anthropic_block():
    msgs = [
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1"}]}
    ]
    with pytest.raises(InvalidToolConversationError):
        validate_conversation(msgs)


# --------------------------------------------------------------------------- #
# Tool schema adapter (spec §8, §18.30-34)
# --------------------------------------------------------------------------- #


def test_schema_preserves_name_and_required():
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
    out = sanitize_tools(tools)
    assert out[0]["function"]["name"] == "Read"
    assert out[0]["function"]["parameters"]["required"] == ["file_path"]


def test_schema_resolves_local_ref():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "AskUserQuestion",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"$ref": "#/$defs/Question"}},
                    "required": ["q"],
                    "$defs": {
                        "Question": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        }
                    },
                },
            },
        }
    ]
    out = sanitize_tools(tools)
    q = out[0]["function"]["parameters"]["properties"]["q"]
    # $ref inlined; $defs stripped from output.
    assert q["type"] == "object"
    assert q["properties"]["text"]["type"] == "string"
    assert "$defs" not in out[0]["function"]["parameters"]


def test_schema_drops_unsupported_keyword_but_keeps_structure():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Write",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                    "$comment": "internal",
                    "unevaluatedProperties": False,
                },
            },
        }
    ]
    out = sanitize_tools(tools)
    params = out[0]["function"]["parameters"]
    assert "$comment" not in params
    assert "unevaluatedProperties" not in params
    assert params["properties"]["content"]["type"] == "string"


def test_schema_handles_cyclic_ref_without_infinite_loop():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "Tree",
                "parameters": {
                    "type": "object",
                    "properties": {"child": {"$ref": "#/$defs/Node"}},
                    "$defs": {
                        "Node": {
                            "type": "object",
                            "properties": {"child": {"$ref": "#/$defs/Node"}},
                        }
                    },
                },
            },
        }
    ]
    out = sanitize_tools(tools)  # must terminate
    assert out[0]["function"]["name"] == "Tree"


# --------------------------------------------------------------------------- #
# Streaming tool-call state machine (spec §9, §18.22-29)
# --------------------------------------------------------------------------- #


def test_fragmented_arguments_are_joined():
    acc = ToolCallAccumulator()
    acc.process_delta_tool_calls(
        [{"index": 0, "id": "t1", "function": {"name": "Read", "arguments": '{"file_'}}]
    )
    acc.process_delta_tool_calls([{"index": 0, "function": {"arguments": 'path":"a'}}])
    acc.process_delta_tool_calls([{"index": 0, "function": {"arguments": '.html"}'}}])
    calls = acc.finalize()
    assert len(calls) == 1
    assert json.loads(calls[0]["function"]["arguments"]) == {"file_path": "a.html"}


def test_arguments_parsed_only_after_completion():
    acc = ToolCallAccumulator()
    acc.process_delta_tool_calls(
        [{"index": 0, "id": "t1", "function": {"name": "Read", "arguments": '{"a":'}}]
    )
    # Mid-stream: nothing is parsed/validated yet, so no error despite bad JSON.
    assert acc.has_pending()


def test_nested_and_escaped_arguments():
    acc = ToolCallAccumulator()
    payload = '{"query":"a\\"b","opts":{"deep":[1,2,{"k":"v"}]}}'
    for frag in [payload[:5], payload[5:15], payload[15:]]:
        acc.process_delta_tool_calls(
            [{"index": 0, "id": "t1", "function": {"name": "Grep", "arguments": frag}}]
        )
    calls = acc.finalize()
    assert json.loads(calls[0]["function"]["arguments"]) == {
        "query": 'a"b',
        "opts": {"deep": [1, 2, {"k": "v"}]},
    }


def test_two_concurrent_tool_calls_are_separated():
    acc = ToolCallAccumulator()
    acc.process_delta_tool_calls(
        [
            {"index": 0, "id": "t1", "function": {"name": "Read", "arguments": '{"p":1}'}},
            {"index": 1, "id": "t2", "function": {"name": "Grep", "arguments": '{"q":"x"}'}},
        ]
    )
    calls = acc.finalize()
    assert [c["id"] for c in calls] == ["t1", "t2"]


def test_tool_call_id_preserved_and_not_emitted_twice():
    acc = ToolCallAccumulator()
    acc.process_delta_tool_calls(
        [{"index": 0, "id": "toolu_keep", "function": {"name": "Read", "arguments": "{}"}}]
    )
    first = acc.finalize()
    second = acc.finalize()  # already emitted → nothing new
    assert first[0]["id"] == "toolu_keep"
    assert second == []


def test_invalid_complete_json_raises_not_empty_object():
    acc = ToolCallAccumulator()
    acc.process_delta_tool_calls(
        [{"index": 0, "id": "t1", "function": {"name": "Read", "arguments": "{broken"}}]
    )
    with pytest.raises(InvalidToolArgumentsError):
        acc.finalize()


def test_empty_arguments_normalize_to_empty_object():
    acc = ToolCallAccumulator()
    acc.process_delta_tool_calls(
        [{"index": 0, "id": "t1", "function": {"name": "NoArgs", "arguments": ""}}]
    )
    calls = acc.finalize()
    assert calls[0]["function"]["arguments"] == "{}"


# --------------------------------------------------------------------------- #
# SSE parsing + aggregation (spec §10)
# --------------------------------------------------------------------------- #


def test_parse_sse_line_ignores_done_and_comments():
    assert parse_sse_data_line("data: [DONE]") is None
    assert parse_sse_data_line(": keep-alive") is None
    assert parse_sse_data_line("") is None
    assert parse_sse_data_line('data: {"a":1}') == {"a": 1}


def test_line_buffer_splits_across_chunks():
    buf = SSELineBuffer()
    assert buf.feed("data: {") == []
    lines = buf.feed('"a":1}\n\n')
    assert 'data: {"a":1}' in lines


def test_aggregator_concatenates_content_and_reasoning_separately():
    agg = StreamAggregator()
    agg.process_chunk({"choices": [{"delta": {"reasoning_content": "think"}}]})
    agg.process_chunk({"choices": [{"delta": {"content": "answer"}}]})
    content, tool_calls, finish = agg.finalize()
    assert content == "answer"
    assert agg.reasoning_content == "think"
    assert tool_calls == []


def test_aggregator_reconstructs_split_tool_call():
    agg = StreamAggregator()
    agg.process_chunk(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "t1", "function": {"name": "Read", "arguments": '{"p"'}}]}}]}
    )
    agg.process_chunk(
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": ':1}'}}]}, "finish_reason": "tool_calls"}]}
    )
    content, tool_calls, finish = agg.finalize()
    assert finish == "tool_calls"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"p": 1}


# --------------------------------------------------------------------------- #
# Response mapper (spec §10)
# --------------------------------------------------------------------------- #


def test_non_stream_response_shape():
    resp = response_mapper.build_non_stream_response(
        response_id="id1",
        model="auto-chat",
        created=123,
        content="hi",
        tool_calls=[],
        finish_reason="stop",
    )
    assert resp["object"] == "chat.completion"
    assert resp["choices"][0]["message"]["content"] == "hi"
    assert resp["choices"][0]["finish_reason"] == "stop"


def test_non_stream_reasoning_not_folded_into_content():
    resp = response_mapper.build_non_stream_response(
        response_id="id1",
        model="m",
        created=1,
        content="final",
        tool_calls=[],
        finish_reason="stop",
        reasoning_content="secret thoughts",
    )
    msg = resp["choices"][0]["message"]
    assert msg["content"] == "final"
    assert msg["reasoning_content"] == "secret thoughts"


def test_tooluse_id_normalized_to_call_prefix():
    resp = response_mapper.build_non_stream_response(
        response_id="id1",
        model="m",
        created=1,
        content="",
        tool_calls=[{"id": "tooluse_abc", "function": {"name": "Read", "arguments": "{}"}}],
        finish_reason="tool_calls",
    )
    assert resp["choices"][0]["message"]["tool_calls"][0]["id"] == "call_abc"
