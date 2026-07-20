# CodeBuddy2API

CodeBuddy2API wraps the official CodeBuddy API in an OpenAI-compatible service. It calls CodeBuddy directly and provides one consistent interface for standard OpenAI clients, 9Router, and other integrations.

## Features

- **OpenAI-compatible API** — standard `/v1/chat/completions` and `/v1/models` endpoints.
- **Streaming and non-streaming** — aggregates CodeBuddy's native stream when a client requests a non-streaming response.
- **Asynchronous service** — built with FastAPI, HTTPX, and `asyncio`.
- **Three client authentication modes** — relay, passthrough, and hybrid.
- **TXT API key pool** — hot reload, asynchronous round-robin, cooldown, invalid-key tracking, and bounded failover.
- **Legacy credential support** — existing credential JSON files remain supported.
- **Separate admin password** — dashboard and management endpoints do not accept request-scoped CodeBuddy keys.
- **Web dashboard** — manage credentials, inspect statistics, test the API, and update settings.
- **One-command Docker deployment** — automatically installs Docker on Ubuntu/Debian, creates secure secrets, starts the service, and verifies health.

## Quick Start

### Requirements

For local Python execution:

- Python 3.8 or newer
- pip

For the recommended deployment:

- Ubuntu or Debian
- root or sudo access when Docker is not installed

### One-command server deployment

```bash
git clone https://github.com/xueyue33/codebuddy2api.git
cd codebuddy2api
chmod +x deploy.sh
./deploy.sh
```

The deployment script:

- installs Docker Engine and the Compose plugin from Docker's official repository when needed;
- creates `.env` with separate random relay and admin passwords;
- uses `CODEBUDDY_CLIENT_AUTH_MODE=passthrough` for a new deployment;
- creates `config/` and `.codebuddy_creds/` without overwriting existing data;
- builds and starts one container with `restart: unless-stopped`;
- waits for `/health` and prints container logs if startup fails.

The script is idempotent. Running it again does not overwrite `.env`, `config/codebuddy_api_keys.txt`, or stored credentials.

Common overrides:

```bash
./deploy.sh --port 18001 --client-auth-mode hybrid --upstream-header both

DEPLOY_PORT=18001 \
DEPLOY_CLIENT_AUTH_MODE=passthrough \
DEPLOY_UPSTREAM_HEADER=bearer \
./deploy.sh
```

Supported options:

```text
--env-file PATH
--port PORT
--client-auth-mode relay|passthrough|hybrid
--upstream-header bearer|x-api-key|both
--relay-password VALUE
--admin-password VALUE
--no-build
--skip-docker-install
--timeout SECONDS
--bootstrap-only
```

Use `--bootstrap-only` to create and validate files without starting Docker.

### Manual Docker deployment

```bash
cp .env.example .env
cp config/codebuddy_api_keys.example.txt config/codebuddy_api_keys.txt
# Edit .env and add real keys if relay TXT mode is used.
docker compose up -d --build
curl http://127.0.0.1:8001/health
```

### Local Python execution

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env before starting the service.
python web.py
```

Windows users can run `start.bat`.

## Client Authentication Modes

`CODEBUDDY_CLIENT_AUTH_MODE` determines how the incoming Bearer token is interpreted.

### `relay`

The client sends:

```http
Authorization: Bearer <CODEBUDDY_PASSWORD>
```

CodeBuddy2API validates the relay password and selects an upstream credential using `CODEBUDDY_AUTH_MODE`.

### `passthrough`

The client sends a real CodeBuddy API key:

```http
Authorization: Bearer <CODEBUDDY_API_KEY>
```

That key is used only for the current request. It is not saved, cached, added to the TXT pool, or recorded in credential statistics. CodeBuddy2API does not rotate or fail over to another key in this mode.

### `hybrid`

- A Bearer token exactly equal to `CODEBUDDY_PASSWORD` uses relay mode.
- Any other Bearer token uses passthrough mode.
- No prefix or key-format guessing is performed.

`CODEBUDDY_CLIENT_AUTH_MODE` and `CODEBUDDY_AUTH_MODE` are separate settings:

- `CODEBUDDY_CLIENT_AUTH_MODE` defines the meaning of the client Bearer token.
- `CODEBUDDY_AUTH_MODE` selects TXT or legacy credentials only when the effective client path is relay.

## Upstream Authentication Header

`CODEBUDDY_UPSTREAM_API_KEY_HEADER` controls how the selected/request-scoped key is sent upstream:

- `bearer` — `Authorization: Bearer <key>`
- `x-api-key` — `X-API-Key: <key>`
- `both` — sends both headers

The default is `bearer`, which matches the previously verified implementation.

## TXT API Key Pool

Create the real key file:

```bash
cp config/codebuddy_api_keys.example.txt config/codebuddy_api_keys.txt
```

Format:

```text
# One upstream CodeBuddy API key per line.
api_key_account_1
api_key_account_2
api_key_account_3
```

The parser:

- trims whitespace;
- ignores empty lines;
- ignores lines beginning with `#`;
- removes duplicates while preserving order.

Pool behavior:

- `401` marks a key invalid;
- `403` and `429` place a key in cooldown;
- timeouts, connection errors, and `5xx` responses try the next unused key;
- a request never retries the same key;
- a streaming request remains pinned after its first downstream chunk;
- changes to the TXT file are reloaded without a container restart.

`CODEBUDDY_AUTH_MODE` values:

- `api_key_file` — use TXT keys only;
- `credentials` — use legacy JSON credentials only;
- `auto` — prefer TXT keys and fall back to legacy credentials.

## 9Router Passthrough Configuration

Multiple 9Router providers can share one Base URL while using different CodeBuddy keys.

```text
Provider A
Name       : CodeBuddy A
Prefix     : cba
Base URL   : http://codebuddy2api:8001/codebuddy/v1
API Key    : CODEBUDDY_KEY_A
Model ID   : glm-5.2

Provider B
Name       : CodeBuddy B
Prefix     : cbb
Base URL   : http://codebuddy2api:8001/codebuddy/v1
API Key    : CODEBUDDY_KEY_B
Model ID   : glm-5.2

Provider C
Name       : CodeBuddy C
Prefix     : cbc
Base URL   : http://codebuddy2api:8001/codebuddy/v1
API Key    : CODEBUDDY_KEY_C
Model ID   : glm-5.2
```

9Router controls combo, fallback, and round-robin. CodeBuddy2API remains a request-scoped protocol adapter.

## API Usage

### Passthrough, non-streaming

```bash
curl -X POST "http://127.0.0.1:8001/codebuddy/v1/chat/completions" \
  -H "Authorization: Bearer CODEBUDDY_KEY_A" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### Passthrough, streaming

```bash
curl -N -X POST "http://127.0.0.1:8001/codebuddy/v1/chat/completions" \
  -H "Authorization: Bearer CODEBUDDY_KEY_A" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": true
  }'
```

### Legacy relay request

```bash
curl -X POST "http://127.0.0.1:8001/codebuddy/v1/chat/completions" \
  -H "Authorization: Bearer relay_master_secret" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto-chat",
    "messages": [{"role": "user", "content": "Hello"}],
    "stream": false
  }'
```

### Models

```bash
curl "http://127.0.0.1:8001/codebuddy/v1/models" \
  -H "Authorization: Bearer CODEBUDDY_KEY_A"
```

### Admin key-pool status and reload

```bash
curl "http://127.0.0.1:8001/codebuddy/v1/api-keys/status" \
  -H "Authorization: Bearer admin_secret"

curl -X POST "http://127.0.0.1:8001/codebuddy/v1/api-keys/reload" \
  -H "Authorization: Bearer admin_secret"
```

If `CODEBUDDY_ADMIN_PASSWORD` is empty, admin authentication falls back to `CODEBUDDY_PASSWORD` for backward compatibility.

## Python Client Example

```python
from openai import OpenAI

client = OpenAI(
    api_key="CODEBUDDY_KEY_A",
    base_url="http://127.0.0.1:8001/codebuddy/v1",
)

response = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "What is 2 + 2?"}],
)
print(response.choices[0].message.content)

stream = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "Write a Python hello-world program."}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="")
```

## API Endpoints

| Method | Endpoint | Authentication | Description |
| --- | --- | --- | --- |
| POST | `/codebuddy/v1/chat/completions` | Inference | OpenAI-compatible chat completions |
| GET | `/codebuddy/v1/models` | Inference | Configured model list |
| GET | `/codebuddy/v1/api-keys/status` | Admin | Masked TXT key-pool status |
| POST | `/codebuddy/v1/api-keys/reload` | Admin | Force a TXT key reload |
| GET/POST | `/codebuddy/v1/credentials*` | Admin | Legacy credential management |
| GET/POST | `/api/settings` | Admin | Read/update settings |
| GET | `/api/stats` | Admin | Usage statistics |
| GET | `/health` | None | Service health |

## Configuration

| Environment variable | Default | Description |
| --- | --- | --- |
| `CODEBUDDY_PASSWORD` | none | Relay/client password |
| `CODEBUDDY_ADMIN_PASSWORD` | empty | Admin password; falls back to relay password |
| `CODEBUDDY_CLIENT_AUTH_MODE` | `relay` | `relay`, `passthrough`, or `hybrid` |
| `CODEBUDDY_UPSTREAM_API_KEY_HEADER` | `bearer` | `bearer`, `x-api-key`, or `both` |
| `CODEBUDDY_HOST` | `127.0.0.1` | Service bind address |
| `CODEBUDDY_PORT` | `8001` | Service/host port |
| `CODEBUDDY_API_ENDPOINT` | `https://www.codebuddy.ai` | Upstream endpoint |
| `CODEBUDDY_CREDS_DIR` | `.codebuddy_creds` | Legacy credential directory |
| `CODEBUDDY_AUTH_MODE` | `auto` | Relay upstream source mode |
| `CODEBUDDY_API_KEYS_FILE` | `./config/codebuddy_api_keys.txt` | TXT key file |
| `CODEBUDDY_API_KEY_ROTATION` | `round_robin` | TXT key rotation strategy |
| `CODEBUDDY_API_KEY_RELOAD_INTERVAL` | `5` | TXT reload interval in seconds |
| `CODEBUDDY_API_KEY_COOLDOWN_SECONDS` | `300` | Cooldown after 403/429 |
| `CODEBUDDY_ROTATION_COUNT` | `1` | Legacy credential rotation frequency |
| `CODEBUDDY_LOG_LEVEL` | `INFO` | Application log level |
| `CODEBUDDY_MODELS` | model list | Models reported to clients |
| `CODEBUDDY_SSL_VERIFY` | `false` | Upstream TLS verification toggle |

## Project Structure

```text
codebuddy2api/
├── src/
│   ├── auth.py
│   ├── codebuddy_api_client.py
│   ├── codebuddy_api_key_manager.py
│   ├── codebuddy_auth_router.py
│   ├── codebuddy_router.py
│   ├── codebuddy_token_manager.py
│   ├── frontend_router.py
│   ├── keyword_replacer.py
│   ├── logging_utils.py
│   ├── settings_router.py
│   └── usage_stats_manager.py
├── frontend/admin.html
├── config/codebuddy_api_keys.example.txt
├── tests/
├── web.py
├── config.py
├── deploy.sh
├── docker-compose.yml
├── Dockerfile
├── entrypoint.sh
├── requirements.txt
└── README.md
```

## Operations

```bash
# Follow logs
docker compose logs -f codebuddy2api

# Restart
docker compose restart codebuddy2api

# Stop
docker compose down

# Upgrade
git pull
./deploy.sh
```

## Testing

```bash
python -m pip install -r requirements-dev.txt
python -m compileall -q .
python -m pytest -q
bash -n deploy.sh
docker compose config
```

## Security Notes

- Request-scoped passthrough keys are not persisted or added to global managers.
- Admin endpoints require the separate admin password when configured.
- Raw keys are excluded from status responses and application logs.
- Sensitive header redaction covers Bearer authorization and `X-API-Key` values.
- `.env` and real TXT key files are excluded from Git and Docker image builds.
- The generated `.env` uses permission `600`.

## Troubleshooting

### API key file is unavailable

- Check `CODEBUDDY_API_KEYS_FILE`, the volume mount, and file permissions.
- Ensure the file contains at least one non-comment key in relay TXT mode.
- Call `/codebuddy/v1/api-keys/reload`, then inspect the masked status endpoint.

### No valid legacy credentials

- Add at least one valid JSON credential to `.codebuddy_creds`.
- Use the admin dashboard OAuth flow or credential form.

### Upstream 401 or 403

The CodeBuddy key or legacy Bearer token may be invalid, expired, blocked, or unauthorized. Obtain a valid credential and retry.

### Invalid relay/admin password

Confirm that the client Bearer token matches `CODEBUDDY_PASSWORD` for relay inference or `CODEBUDDY_ADMIN_PASSWORD` for management endpoints.

### Detailed logs

Set `CODEBUDDY_LOG_LEVEL=DEBUG` and restart the service. Never share logs without checking them for sensitive deployment data.
