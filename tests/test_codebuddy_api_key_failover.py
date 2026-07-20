import json

import httpx
import pytest
from fastapi import FastAPI

import config
from src import auth, codebuddy_router
from src.codebuddy_api_key_manager import CodeBuddyApiKeyManager


SERVER_PASSWORD = "relay-test-password"
KEY_A = "account-alpha-secret-0001"
KEY_B = "account-beta-secret-0002"


class BrokenStream(httpx.AsyncByteStream):
    def __init__(self, fail_before_first=False):
        self.fail_before_first = fail_before_first

    async def __aiter__(self):
        if self.fail_before_first:
            raise httpx.ReadError("upstream disconnected")
        yield b'data: {"id":"chat-1","model":"auto-chat","choices":[{"delta":{"content":"first"}}]}\n\n'
        raise httpx.ReadError("upstream disconnected")



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
    return application


async def setup_pool(monkeypatch, tmp_path, keys=(KEY_A, KEY_B), mode="api_key_file"):
    path = tmp_path / "keys.txt"
    path.write_text("\n".join(keys), encoding="utf-8")
    manager = CodeBuddyApiKeyManager(str(path), cooldown_seconds=60, reload_interval=0)
    await manager.reload()
    monkeypatch.setattr(codebuddy_router, "codebuddy_api_key_manager", manager)
    monkeypatch.setattr(config, "get_codebuddy_auth_mode", lambda: mode)
    monkeypatch.setattr(auth, "get_client_auth_mode", lambda: "relay")
    monkeypatch.setattr(auth, "get_server_password", lambda: SERVER_PASSWORD)
    monkeypatch.setattr(auth, "get_admin_password", lambda: SERVER_PASSWORD)
    return manager


def install_upstream(monkeypatch, handler):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def get_client():
        return client

    monkeypatch.setattr(codebuddy_router, "get_http_client", get_client)
    return client


async def post_chat(app, stream=False):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        return await client.post(
            "/codebuddy/v1/chat/completions",
            headers={"Authorization": f"Bearer {SERVER_PASSWORD}"},
            json={
                "model": "auto-chat",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": stream,
            },
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("first_status", "expected_state"),
    [(401, "invalid"), (429, "cooldown"), (500, "active")],
)
async def test_non_stream_failover(monkeypatch, tmp_path, app, first_status, expected_state):
    manager = await setup_pool(monkeypatch, tmp_path)
    used = []

    def handler(request):
        token = request.headers["authorization"].removeprefix("Bearer ")
        used.append(token)
        return httpx.Response(first_status) if token == KEY_A else sse_response()

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app)
    await upstream.aclose()

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert used == [KEY_A, KEY_B]
    status = await manager.get_status()
    assert status["keys"][0]["status"] == expected_state
    assert status["keys"][0]["error_count"] == 1
    assert status["keys"][1]["request_count"] == 1


@pytest.mark.asyncio
async def test_timeout_failover_uses_next_key_once(monkeypatch, tmp_path, app):
    manager = await setup_pool(monkeypatch, tmp_path)
    used = []

    def handler(request):
        token = request.headers["authorization"].removeprefix("Bearer ")
        used.append(token)
        if token == KEY_A:
            raise httpx.ConnectTimeout("upstream timeout", request=request)
        return sse_response("recovered")

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app)
    await upstream.aclose()

    assert response.status_code == 200
    assert used == [KEY_A, KEY_B]
    status = await manager.get_status()
    assert status["keys"][0]["error_count"] == 1
    assert status["keys"][1]["request_count"] == 1


@pytest.mark.asyncio
async def test_stream_failover_happens_before_first_chunk(monkeypatch, tmp_path, app):
    manager = await setup_pool(monkeypatch, tmp_path)
    used = []

    def handler(request):
        token = request.headers["authorization"].removeprefix("Bearer ")
        used.append(token)
        return httpx.Response(401) if token == KEY_A else sse_response("streamed")

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert "streamed" in response.text
    assert used == [KEY_A, KEY_B]
    status = await manager.get_status()
    assert status["keys"][0]["status"] == "invalid"


@pytest.mark.asyncio
async def test_stream_failover_on_error_before_first_chunk(monkeypatch, tmp_path, app):
    await setup_pool(monkeypatch, tmp_path)
    used = []

    def handler(request):
        token = request.headers["authorization"].removeprefix("Bearer ")
        used.append(token)
        if token == KEY_A:
            return httpx.Response(200, stream=BrokenStream(fail_before_first=True))
        return sse_response("recovered")

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert "recovered" in response.text
    assert used == [KEY_A, KEY_B]


@pytest.mark.asyncio
async def test_stream_does_not_switch_key_after_first_chunk(monkeypatch, tmp_path, app):
    manager = await setup_pool(monkeypatch, tmp_path)
    used = []

    def handler(request):
        token = request.headers["authorization"].removeprefix("Bearer ")
        used.append(token)
        return httpx.Response(200, stream=BrokenStream())

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app, stream=True)
    await upstream.aclose()

    assert response.status_code == 200
    assert "first" in response.text
    assert "upstream_stream_error" in response.text
    assert used == [KEY_A]
    status = await manager.get_status()
    assert status["keys"][0]["error_count"] == 1
    assert status["keys"][1]["request_count"] == 0


@pytest.mark.asyncio
async def test_all_keys_exhausted_is_openai_error_and_sanitized(monkeypatch, tmp_path, app, caplog):
    await setup_pool(monkeypatch, tmp_path)

    def handler(request):
        return httpx.Response(401, text=f"invalid key {request.headers['authorization']}")

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app)
    await upstream.aclose()

    assert response.status_code == 502
    assert response.json()["error"]["code"] == "upstream_keys_exhausted"
    combined = response.text + caplog.text
    assert KEY_A not in combined
    assert KEY_B not in combined


@pytest.mark.asyncio
async def test_auto_mode_falls_back_to_legacy_credentials(monkeypatch, tmp_path, app):
    await setup_pool(monkeypatch, tmp_path, keys=(), mode="auto")
    calls = []

    def legacy_credential():
        calls.append(True)
        return {"bearer_token": "legacy-bearer", "user_id": "legacy-user"}

    monkeypatch.setattr(
        codebuddy_router.codebuddy_token_manager,
        "get_next_credential",
        legacy_credential,
    )

    def handler(request):
        assert request.headers["authorization"] == "Bearer legacy-bearer"
        return sse_response("legacy")

    upstream = install_upstream(monkeypatch, handler)
    response = await post_chat(app)
    await upstream.aclose()

    assert response.status_code == 200
    assert calls == [True]
    assert response.json()["choices"][0]["message"]["content"] == "legacy"


@pytest.mark.asyncio
async def test_api_key_admin_endpoints_are_protected_and_sanitized(monkeypatch, tmp_path, app):
    await setup_pool(monkeypatch, tmp_path)
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        denied = await client.get("/codebuddy/v1/api-keys/status")
        status = await client.get(
            "/codebuddy/v1/api-keys/status",
            headers={"Authorization": f"Bearer {SERVER_PASSWORD}"},
        )
        reload_response = await client.post(
            "/codebuddy/v1/api-keys/reload",
            headers={"Authorization": f"Bearer {SERVER_PASSWORD}"},
        )

    assert denied.status_code in {401, 403}
    assert status.status_code == 200
    assert reload_response.status_code == 200
    serialized = json.dumps(status.json()) + json.dumps(reload_response.json())
    assert KEY_A not in serialized
    assert KEY_B not in serialized
    assert status.json()["keys"][0]["masked_key"] == "acco...0001"
