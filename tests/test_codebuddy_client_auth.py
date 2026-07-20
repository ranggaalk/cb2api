import json
import logging

import httpx
import pytest
from fastapi import FastAPI

import config
from src import auth, codebuddy_auth_router, codebuddy_router, settings_router
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager
from src.codebuddy_api_client import codebuddy_api_client
from src.logging_utils import SensitiveHeaderFilter, redact_sensitive

RELAY_PASSWORD = "relay-password"
ADMIN_PASSWORD = "admin-password"
KEY_A = "passthrough-account-alpha-0001"
KEY_B = "passthrough-account-beta-0002"


def sse_response(text="ok"):
    body = (
        'data: {"id":"chat-1","model":"auto-chat","choices":'
        f'[{{"delta":{{"content":"{text}"}},"finish_reason":"stop"}}]}}\n\n'
        "data: [DONE]\n\n"
    )
    return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})


@pytest.fixture
def app():
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


def configure(monkeypatch, client_mode="passthrough", header_mode="bearer", profile="cli"):
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: client_mode)
    monkeypatch.setattr(auth, "get_server_password", lambda: RELAY_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: ADMIN_PASSWORD)
    monkeypatch.setattr(codebuddy_router, "get_upstream_api_key_header", lambda: header_mode)
    # Pin the request profile to cli so the upstream header-mode mapping stays
    # deterministic in these tests (the web profile always sends both headers).
    monkeypatch.setattr(codebuddy_router, "get_codebuddy_request_profile", lambda: profile)
    monkeypatch.setattr(
        codebuddy_router.usage_stats_manager,
        "record_model_usage",
        lambda _model: None,
    )


def install_upstream(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def get_client():
        return client

    monkeypatch.setattr(codebuddy_router, "get_http_client", get_client)
    return client


async def request(app, path, token=None, method="GET", json_body=None):
    headers = {}
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.request(method, path, headers=headers, json=json_body)


async def chat(app, token, stream=False, content="hello"):
    return await request(
        app,
        "/codebuddy/v1/chat/completions",
        token,
        "POST",
        {
            "model": "auto-chat",
            "messages": [{"role": "user", "content": content}],
            "stream": stream,
        },
    )


@pytest.mark.asyncio
async def test_passthrough_forwards_request_key(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = []

    def handler(upstream_request):
        seen.append(upstream_request.headers.get("authorization"))
        return sse_response()

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen == [f"Bearer {KEY_A}"]
    assert await empty_pool.total_count() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mode", "authorization", "x_api_key"),
    [
        ("bearer", f"Bearer {KEY_A}", None),
        ("x-api-key", None, KEY_A),
        ("both", f"Bearer {KEY_A}", KEY_A),
    ],
)
async def test_upstream_header_mapping(
    monkeypatch, app, empty_pool, mode, authorization, x_api_key
):
    configure(monkeypatch, header_mode=mode)
    seen = []

    def handler(upstream_request):
        seen.append(
            (
                upstream_request.headers.get("authorization"),
                upstream_request.headers.get("x-api-key"),
            )
        )
        return sse_response()

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == 200
    assert seen == [(authorization, x_api_key)]


@pytest.mark.asyncio
async def test_concurrent_passthrough_keys_do_not_mix(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = {}

    def handler(upstream_request):
        body = json.loads(upstream_request.content)
        content = body["messages"][-1]["content"]
        seen[content] = upstream_request.headers["authorization"]
        return sse_response(content)

    upstream = install_upstream(monkeypatch, handler)
    first, second = await asyncio.gather(
        chat(app, KEY_A, content="a"), chat(app, KEY_B, content="b")
    )
    await upstream.aclose()

    assert first.status_code == second.status_code == 200
    assert seen == {"a": f"Bearer {KEY_A}", "b": f"Bearer {KEY_B}"}


@pytest.mark.asyncio
async def test_passthrough_stream_uses_one_request_key(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    seen = []

    def handler(upstream_request):
        seen.append(upstream_request.headers["authorization"])
        return sse_response("streamed")

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert "streamed" in response.text
    assert seen == [f"Bearer {KEY_A}"]
    assert await empty_pool.total_count() == 0


@pytest.mark.asyncio
async def test_passthrough_stream_does_not_restart_after_first_chunk(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)
    attempts = []

    class InterruptedStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"chat-1","model":"auto-chat","choices":[{"delta":{"content":"first"}}]}\n\n'
            raise httpx.ReadError("interrupted")

    def handler(upstream_request):
        attempts.append(upstream_request.headers["authorization"])
        return httpx.Response(200, stream=InterruptedStream())

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert "first" in response.text
    assert "upstream_stream_error" in response.text
    assert attempts == [f"Bearer {KEY_A}"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("upstream_status", "expected_status"),
    [(401, 401), (403, 403), (429, 429), (500, 500), (503, 503)],
)
async def test_passthrough_preserves_status_without_failover(
    monkeypatch, app, empty_pool, upstream_status, expected_status, caplog
):
    configure(monkeypatch)
    attempts = []

    def handler(upstream_request):
        attempts.append(upstream_request.headers["authorization"])
        return httpx.Response(upstream_status, text=f"rejected {KEY_A}")

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == expected_status
    assert attempts == [f"Bearer {KEY_A}"]
    assert KEY_A not in response.text + caplog.text
    if upstream_status == 401:
        assert response.json()["error"]["code"] == "upstream_api_key_rejected"


@pytest.mark.asyncio
async def test_passthrough_timeout_is_single_504_attempt(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    attempts = []

    def handler(upstream_request):
        attempts.append(upstream_request.headers["authorization"])
        raise httpx.ConnectTimeout("timeout", request=upstream_request)

    upstream = install_upstream(monkeypatch, handler)
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == 504
    assert attempts == [f"Bearer {KEY_A}"]
    assert KEY_A not in response.text


@pytest.mark.asyncio
async def test_hybrid_exact_match_selects_relay_or_passthrough(
    monkeypatch, app, tmp_path
):
    configure(monkeypatch, client_mode="hybrid")
    path = tmp_path / "keys.txt"
    pool_key = "pool-key-secret-0001"
    path.write_text(pool_key, encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(path), reload_interval=0)
    await manager.reload()
    monkeypatch.setattr(codebuddy_router, "codebuddy_api_key_manager", manager)
    monkeypatch.setattr(config, "get_codebuddy_auth_mode", lambda: "api_key_file")
    seen = []

    def handler(upstream_request):
        seen.append(upstream_request.headers["authorization"])
        return sse_response()

    upstream = install_upstream(monkeypatch, handler)
    relay_response = await chat(app, RELAY_PASSWORD)
    passthrough_response = await chat(app, KEY_A)
    await upstream.aclose()

    assert relay_response.status_code == passthrough_response.status_code == 200
    assert seen == [f"Bearer {pool_key}", f"Bearer {KEY_A}"]


@pytest.mark.asyncio
async def test_models_and_invalid_authorization(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    valid = await request(app, "/codebuddy/v1/models", KEY_A)
    missing = await request(app, "/codebuddy/v1/models")

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        malformed = await client.get(
            "/codebuddy/v1/models", headers={"Authorization": f"Basic {KEY_A}"}
        )

    assert valid.status_code == 200
    assert missing.status_code == malformed.status_code == 401


@pytest.mark.asyncio
async def test_admin_password_is_separate(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    denied_key = await request(app, "/codebuddy/v1/api-keys/status", KEY_A)
    denied_relay = await request(
        app, "/codebuddy/v1/api-keys/status", RELAY_PASSWORD
    )
    accepted = await request(app, "/codebuddy/v1/api-keys/status", ADMIN_PASSWORD)

    assert denied_key.status_code == denied_relay.status_code == 403
    assert accepted.status_code == 200


@pytest.mark.asyncio
async def test_admin_password_falls_back_to_relay(monkeypatch, app, empty_pool):
    configure(monkeypatch)
    monkeypatch.setattr(auth, "get_admin_password", lambda: RELAY_PASSWORD)
    response = await request(
        app, "/codebuddy/v1/api-keys/status", RELAY_PASSWORD
    )
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_oauth_management_requires_admin_password(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)

    async def fake_start():
        return {"success": True, "verification_uri_complete": "https://example.test"}

    monkeypatch.setattr(codebuddy_auth_router, "start_codebuddy_auth", fake_start)
    denied = await request(app, "/codebuddy/auth/start", KEY_A)
    accepted = await request(app, "/codebuddy/auth/start", ADMIN_PASSWORD)

    assert denied.status_code == 403
    assert accepted.status_code == 200


@pytest.mark.asyncio
async def test_relay_still_validates_server_password(monkeypatch, app, tmp_path):
    configure(monkeypatch, client_mode="relay")
    path = tmp_path / "relay-keys.txt"
    path.write_text("relay-pool-key-0001", encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(path), reload_interval=0)
    await manager.reload()
    monkeypatch.setattr(codebuddy_router, "codebuddy_api_key_manager", manager)
    monkeypatch.setattr(config, "get_codebuddy_auth_mode", lambda: "api_key_file")

    upstream = install_upstream(monkeypatch, lambda _request: sse_response("relay"))
    accepted = await chat(app, RELAY_PASSWORD)
    denied = await chat(app, KEY_A)
    await upstream.aclose()

    assert accepted.status_code == 200
    assert denied.status_code == 403


@pytest.mark.asyncio
async def test_passthrough_does_not_call_global_credential_managers(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)

    async def forbidden_acquire(*_args, **_kwargs):
        raise AssertionError("passthrough must not acquire from TXT pool")

    def forbidden_legacy():
        raise AssertionError("passthrough must not select legacy credentials")

    monkeypatch.setattr(empty_pool, "acquire", forbidden_acquire)
    monkeypatch.setattr(
        codebuddy_router.codebuddy_token_manager,
        "get_next_credential",
        forbidden_legacy,
    )
    upstream = install_upstream(monkeypatch, lambda _request: sse_response())
    response = await chat(app, KEY_A)
    await upstream.aclose()

    assert response.status_code == 200


@pytest.mark.asyncio
async def test_settings_masks_secrets_and_sentinel_does_not_overwrite(
    monkeypatch, app, empty_pool
):
    configure(monkeypatch)
    monkeypatch.setitem(config._config_cache, "CODEBUDDY_PASSWORD", RELAY_PASSWORD)
    monkeypatch.setitem(config._config_cache, "CODEBUDDY_ADMIN_PASSWORD", ADMIN_PASSWORD)
    monkeypatch.setattr(config, "save_config_to_json", lambda: None)

    response = await request(app, "/api/settings", ADMIN_PASSWORD)
    assert response.status_code == 200
    settings = response.json()["settings"]
    assert settings["CODEBUDDY_PASSWORD"] == "********"
    assert settings["CODEBUDDY_ADMIN_PASSWORD"] == "********"
    assert RELAY_PASSWORD not in response.text
    assert ADMIN_PASSWORD not in response.text

    config.update_settings(
        {
            "CODEBUDDY_PASSWORD": "********",
            "CODEBUDDY_ADMIN_PASSWORD": "********",
        }
    )
    assert config._config_cache["CODEBUDDY_PASSWORD"] == RELAY_PASSWORD
    assert config._config_cache["CODEBUDDY_ADMIN_PASSWORD"] == ADMIN_PASSWORD


def test_header_builder_and_log_redaction():
    headers = codebuddy_api_client.generate_codebuddy_headers(
        KEY_A, api_key_header="both"
    )
    assert headers["Authorization"] == f"Bearer {KEY_A}"
    assert headers["X-API-Key"] == KEY_A

    message = f"Authorization: Bearer {KEY_A}, X-API-Key: {KEY_B}"
    redacted = redact_sensitive(message)
    assert KEY_A not in redacted
    assert KEY_B not in redacted
    assert redacted.count("[REDACTED]") == 2

    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, (), None)
    assert SensitiveHeaderFilter().filter(record)
    assert KEY_A not in record.getMessage()
