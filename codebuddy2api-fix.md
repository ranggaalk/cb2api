# PROMPT 1 — CodeBuddy2API: Audit Latency, Streaming, Cancellation, dan Connection Pool

Repository:

```text
https://github.com/xueyue33/codebuddy2api
```

## Konteks Masalah

CodeBuddy2API digunakan dengan alur:

```text
Claude Code / Shiteru
→ 9Router
→ CodeBuddy2API
→ https://www.codebuddy.ai/v2/chat/completions
```

Chat biasa relatif cepat. Namun request Claude Code yang memiliki banyak messages, tool calls, tool results, dan sekitar 30 tools terkadang:

- lama setelah log `outcome=prepared`;
- menunggu lama sebelum respons pertama;
- menyebabkan 9Router menampilkan `502 fetch connect timeout`;
- meninggalkan request yang masih berjalan setelah client berhenti;
- menghasilkan beberapa request upstream berdekatan;
- terasa stuck saat melanjutkan percakapan lama.

Contoh request berat:

```text
requested_model=claude-opus-4.7-1m
mapped_model=claude-opus-4.7-1m
stream=True
message_count=20+
tool_count=30
has_tools=True
system_prompt_sanitized=False
request_profile=cli
```

Model mapping, tool-result conversion, dan basic streaming sudah bekerja. Jangan merusak perbaikan yang sudah ada.

## Tujuan

Perbaiki observability dan stabilitas CodeBuddy2API agar:

1. Tahap yang lambat dapat diketahui secara pasti.
2. Request tidak diam setelah `outcome=prepared`.
3. Streaming benar-benar diteruskan secara incremental.
4. Request upstream dibatalkan ketika downstream disconnect.
5. Tidak ada ghost request setelah Claude Code dibatalkan.
6. Connection pool dipakai ulang dengan benar.
7. Semaphore atau concurrency limiter tidak bocor.
8. Request besar tidak membuat task/socket menumpuk tanpa batas.
9. Error upstream tetap dikembalikan dengan status yang tepat.
10. API key, prompt, dan isi file tidak bocor ke log.

## 1. Audit Sebelum Mengubah Kode

Temukan dan dokumentasikan:

- endpoint `chat/completions`;
- lokasi log `outcome=prepared`;
- pembentukan payload upstream;
- pembuatan `httpx.AsyncClient`;
- semaphore/concurrency limiter;
- pemanggilan CodeBuddy upstream;
- pembuatan `StreamingResponse`;
- pembacaan SSE upstream;
- cancellation handling;
- error propagation;
- lifecycle startup dan shutdown.

Jangan langsung menulis ulang streaming. Audit terlebih dahulu apakah body upstream masih dibuffer atau sudah diteruskan incremental.

## 2. Request ID dan Stage Telemetry

Setiap inbound request harus memiliki `request_id`.

Tambahkan log pada tahap:

```text
request_received
request_normalized
request_prepared
upstream_slot_wait_start
upstream_slot_acquired
upstream_send_start
upstream_headers_received
upstream_first_chunk
upstream_stream_finished
downstream_disconnected
upstream_cancelled
request_failed
```

Metadata aman:

```text
request_id
requested_model
mapped_model
stream
message_count
tool_count
total_content_length
key_fingerprint
queue_wait_ms
pool_wait_ms
time_to_upstream_headers_ms
time_to_first_chunk_ms
stream_duration_ms
chunk_count
upstream_status
finish_reason
```

Contoh:

```text
request_id=abc123 stage=upstream_slot_wait_start
request_id=abc123 stage=upstream_slot_acquired queue_wait_ms=125
request_id=abc123 stage=upstream_send_start
request_id=abc123 stage=upstream_headers_received status=200 elapsed_ms=2180
request_id=abc123 stage=upstream_first_chunk elapsed_ms=3475
request_id=abc123 stage=upstream_stream_finished chunks=82 duration_ms=18230
```

Jangan log:

- raw API key;
- Authorization;
- X-Api-Key;
- isi prompt;
- isi file;
- isi tool result;
- cookie atau credential.

## 3. Shared HTTP Client

Pastikan aplikasi menggunakan satu reusable `httpx.AsyncClient`.

Contoh konfigurasi awal:

```python
httpx.AsyncClient(
    timeout=httpx.Timeout(
        connect=30.0,
        read=None,
        write=60.0,
        pool=30.0,
    ),
    limits=httpx.Limits(
        max_connections=100,
        max_keepalive_connections=30,
        keepalive_expiry=60.0,
    ),
)
```

Ketentuan:

- dibuat saat startup;
- ditutup saat shutdown;
- tidak membuat client baru pada setiap request;
- tidak menutup shared client setelah satu request;
- connection pool memiliki limit yang jelas;
- waktu menunggu pool dicatat.

## 4. Audit True Streaming

Untuk `stream=true`, pastikan jalur upstream menggunakan:

```python
client.stream(...)
```

atau:

```python
client.send(request, stream=True)
```

Dilarang pada jalur streaming:

```python
await response.aread()
response.text
response.json()
list(response.aiter_lines())
```

Jangan mengumpulkan seluruh chunk sebelum mengirim ke downstream.

Koneksi upstream harus tetap terbuka selama downstream membaca stream.

Response SSE harus memakai:

```http
Content-Type: text/event-stream
Cache-Control: no-cache, no-transform
X-Accel-Buffering: no
```

Pastikan GZip tidak membuffer endpoint SSE.

## 5. Jangan Mengirim Heartbeat Sebelum Status Upstream Diketahui Secara Aman

Audit kemungkinan penggunaan SSE heartbeat.

Heartbeat boleh dipakai untuk menjaga koneksi:

```text
: ping

```

tetapi jangan langsung mengirim HTTP 200 ke downstream sebelum diketahui bahwa upstream menerima request.

Gunakan strategi aman:

1. buka koneksi upstream;
2. tunggu status/header upstream;
3. jika status error, kembalikan error HTTP yang benar;
4. jika status berhasil, mulai StreamingResponse;
5. selama menunggu chunk pertama, heartbeat SSE boleh dikirim setiap 15–20 detik.

Heartbeat:

- tidak boleh masuk ke assistant content;
- tidak boleh merusak `[DONE]`;
- harus berhenti saat client disconnect;
- tidak boleh ditulis paralel tanpa sinkronisasi.

## 6. Cancellation dan Disconnect

Jika downstream disconnect:

- batalkan request upstream;
- tutup response upstream;
- lepaskan semaphore;
- hentikan heartbeat;
- jangan lanjutkan retry;
- jangan meninggalkan background task.

Tangani:

```python
asyncio.CancelledError
```

Pastikan cleanup menggunakan `finally`.

Contoh kebutuhan:

```python
try:
    ...
except asyncio.CancelledError:
    logger.info(
        "request_id=%s stage=downstream_disconnected",
        request_id,
    )
    raise
finally:
    if upstream_response is not None:
        await upstream_response.aclose()
```

Tambahkan test bahwa setelah client disconnect, tidak ada request upstream yang terus berjalan.

## 7. Semaphore dan Concurrency

Audit semua semaphore.

Pastikan:

- semaphore selalu dilepas;
- cancellation tidak menyebabkan slot bocor;
- tidak ada semaphore ganda tanpa alasan;
- queue wait dicatat;
- request tidak mengantre tanpa batas.

Tambahkan konfigurasi:

```env
CODEBUDDY_MAX_CONCURRENT_UPSTREAM_REQUESTS=20
CODEBUDDY_UPSTREAM_QUEUE_TIMEOUT_SECONDS=60
```

Jika queue timeout:

```json
{
  "error": {
    "message": "CodeBuddy relay is temporarily at capacity",
    "type": "server_overloaded",
    "code": "upstream_queue_timeout"
  }
}
```

Gunakan HTTP `503`.

## 8. Timeout Terpisah

Tambahkan konfigurasi terpisah:

```env
CODEBUDDY_CONNECT_TIMEOUT_SECONDS=30
CODEBUDDY_POOL_TIMEOUT_SECONDS=30
CODEBUDDY_WRITE_TIMEOUT_SECONDS=60
CODEBUDDY_FIRST_BYTE_TIMEOUT_SECONDS=180
CODEBUDDY_STREAM_READ_TIMEOUT_SECONDS=0
```

Interpretasi:

```text
connect timeout
→ gagal membuka koneksi

pool timeout
→ menunggu slot connection pool terlalu lama

first-byte timeout
→ upstream terlalu lama mengirim header/chunk awal

stream read timeout
→ timeout setelah stream berjalan
```

Nilai `0` untuk stream read berarti unlimited jika implementasi mendukung.

Jangan memakai satu timeout global 15 detik untuk seluruh agentic request.

## 9. Jangan Retry Internal pada Passthrough

Pada:

```env
CODEBUDDY_CLIENT_AUTH_MODE=passthrough
```

CodeBuddy2API tidak boleh berpindah ke key lain.

Tidak boleh retry untuk:

```text
400 invalid_request
400 content_filter
401 invalid API key
403 permission denied
```

Retry terbatas hanya boleh terjadi sebelum first chunk untuk:

```text
408
429
502
503
504
connect timeout
connection reset
```

Ketentuan:

- maksimal 2 attempt;
- tidak retry setelah first chunk;
- tidak retry setelah tool call diteruskan;
- tidak retry setelah client disconnect;
- tidak retry payload malformed.

## 10. Pertahankan Tool Messages

Jangan menghapus atau merangkum tool results secara diam-diam.

Pertahankan perbaikan yang memastikan:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [...]
}
```

dan:

```json
{
  "role": "tool",
  "tool_call_id": "call_xxx",
  "content": "..."
}
```

Sebelum upstream request, validasi:

- semua message memiliki `role`;
- semua message memiliki `content`;
- setiap tool result memiliki pasangan tool call;
- tool call arguments merupakan JSON string valid;
- content array yang valid tidak dibuang.

## 11. Large Request Warning

Tambahkan warning metadata bila request besar:

```env
CODEBUDDY_WARN_TOTAL_CONTENT_LENGTH=50000
CODEBUDDY_WARN_MESSAGE_COUNT=40
CODEBUDDY_WARN_TOOL_COUNT=30
```

Log hanya warning, jangan memotong request otomatis.

Contoh:

```text
request_id=abc123 large_agentic_request=true message_count=42 tool_count=30 total_content_length=68420
```

## 12. Tests

Tambahkan automated tests:

1. Upstream memberi tiga chunk dengan delay dan downstream menerima incremental.
2. Header SSE benar.
3. Tidak ada buffering body pada stream path.
4. Shared HTTP client digunakan ulang.
5. Queue wait dicatat.
6. Queue timeout menghasilkan 503.
7. Client disconnect membatalkan upstream.
8. Semaphore dilepas setelah cancellation.
9. First-byte timeout berbeda dari stream read timeout.
10. Heartbeat tidak menjadi assistant content.
11. Tidak ada retry setelah first chunk.
12. HTTP 400 tidak di-retry.
13. HTTP 429 hanya di-retry secara terbatas.
14. Request dengan 30 tools tetap dapat streaming.
15. Request dengan 25+ messages tidak kehilangan tool result.
16. Non-streaming tetap menghasilkan `message.content`.
17. Raw key, prompt, dan isi file tidak muncul di log.

## Acceptance Criteria

Pekerjaan selesai jika:

- tahap lambat dapat diketahui dari log;
- log tidak berhenti tanpa penjelasan setelah `outcome=prepared`;
- streaming tetap incremental;
- client disconnect menghentikan upstream;
- tidak ada ghost request;
- semaphore dan connection pool tidak bocor;
- request tool-heavy tidak menumpuk tanpa batas;
- error HTTP tetap dipertahankan;
- mode passthrough tidak merotasi key internal;
- semua test lulus.

## Output Agent

Sebelum implementasi:

- jelaskan alur request saat ini;
- tunjukkan kemungkinan lokasi delay;
- sebutkan file yang akan diubah.

Setelah implementasi:

- daftar file yang diubah;
- arsitektur shared client;
- lifecycle streaming;
- konfigurasi environment baru;
- contoh telemetry;
- hasil test;
- perintah deployment.

Implementasikan perubahan secara langsung. Jangan hanya memberikan saran.
```

---

# PROMPT 2 — 9Router: Retry Policy, Timeout, Fallback, dan Streaming Cancellation

Repository:

```text
https://github.com/decolua/9router
```

## Konteks Masalah

9Router digunakan dengan alur:

```text
Claude Code / Shiteru
→ 9Router
→ http://cb2api:8001/codebuddy/v1
→ CodeBuddy2API
→ CodeBuddy
```

Chat biasa bekerja. Namun workflow Claude Code dengan tool calls kadang menampilkan:

```text
API error · Retrying · attempt 1/10
```

Dashboard 9Router kadang menampilkan:

```text
[502]: fetch connect timeout
13s–15s
```

atau menandai banyak key sebagai:

```text
unavailable
```

Beberapa error yang pernah terjadi:

```text
400 invalid_request
Message must have role and content

502 fetch connect timeout

upstream response lambat saat request tool-heavy
```

Saat ini retry sampai 10 kali membuat kegagalan terasa sangat lama dan dapat menghasilkan request berulang.

## Tujuan

Perbaiki 9Router agar:

1. HTTP 400 tidak di-retry.
2. Retry dibatasi berdasarkan jenis error.
3. Connect timeout terpisah dari first-byte dan stream timeout.
4. Tidak melakukan fallback setelah stream dimulai.
5. Tidak menggandakan tool call.
6. Downstream disconnect membatalkan upstream.
7. Provider tidak ditandai unavailable terlalu agresif.
8. Semua attempt memiliki telemetry.
9. Base URL internal Docker digunakan secara stabil.
10. API key tidak bocor ke log.

## 1. Audit Retry dan Timeout

Temukan:

- fungsi request upstream;
- default connect timeout;
- retry loop;
- fallback loop;
- provider health state;
- unavailable/cooldown handling;
- streaming parser;
- cancellation handling;
- logika kapan attempt dianggap gagal.

Jelaskan alasan angka timeout sekitar 13–15 detik yang terlihat di dashboard.

## 2. Pisahkan Timeout

Tambahkan konfigurasi:

```env
ROUTER_CONNECT_TIMEOUT_SECONDS=30
ROUTER_FIRST_BYTE_TIMEOUT_SECONDS=180
ROUTER_STREAM_IDLE_TIMEOUT_SECONDS=600
ROUTER_TOTAL_REQUEST_TIMEOUT_SECONDS=0
```

Makna:

```text
connect timeout
→ waktu membuka koneksi ke CodeBuddy2API

first-byte timeout
→ waktu menunggu header/chunk pertama

stream idle timeout
→ maksimum waktu tanpa event setelah stream dimulai

total timeout
→ 0 berarti tidak membatasi seluruh agentic request
```

Jangan memakai connect timeout sebagai total timeout.

## 3. Retry Policy Berdasarkan Status

Jangan retry:

```text
400 invalid_request
400 content_filter
401 invalid credentials
403 permission denied
404 model/endpoint not found
422 validation error
```

Boleh retry/fallback secara terbatas:

```text
408 request timeout
429 rate limited
502 bad gateway
503 service unavailable
504 gateway timeout
connect timeout
connection reset
DNS failure sementara
```

Konfigurasi:

```env
ROUTER_MAX_RETRY_ATTEMPTS=2
ROUTER_RETRY_BASE_DELAY_MS=500
ROUTER_RETRY_MAX_DELAY_MS=3000
```

Jangan gunakan 10 attempt sebagai default.

## 4. Jangan Retry Setelah Stream Dimulai

Setelah salah satu dari berikut diterima:

- assistant content;
- reasoning chunk;
- tool call;
- role chunk;
- event SSE valid;

request dianggap sudah dimulai.

Setelah itu:

- jangan fallback ke provider lain;
- jangan retry request dari awal;
- jangan mengirim tool call dua kali;
- jangan menandai key lain sebagai gagal karena stream yang sama;
- jika stream terputus, return error stream ke client.

Tambahkan state:

```text
stream_started=true
```

dan log:

```text
retry_skipped_reason=stream_already_started
```

## 5. Cancellation

Jika Claude Code atau Shiteru membatalkan request:

- abort fetch upstream;
- hentikan retry;
- hentikan fallback;
- jangan lanjutkan request pada background;
- jangan menandai key unavailable karena downstream cancel.

Gunakan `AbortController` atau mekanisme cancellation runtime yang sesuai.

Log:

```text
request_id=...
downstream_disconnected=true
upstream_aborted=true
retry_cancelled=true
```

## 6. Provider Health dan Unavailable State

Jangan menandai key unavailable permanen hanya karena satu connect timeout ke relay lokal.

Bedakan:

```text
invalid_key
rate_limited
provider_unavailable
relay_connect_timeout
client_cancelled
payload_invalid
```

Aturan:

```text
400 payload_invalid
→ jangan mengubah status key

client cancelled
→ jangan mengubah status key

connect timeout lokal
→ cooldown pendek, bukan invalid

401
→ invalid key

429
→ cooldown sesuai retry-after

5xx
→ temporary unavailable
```

Tambahkan konfigurasi:

```env
ROUTER_CONNECT_FAILURE_COOLDOWN_SECONDS=15
ROUTER_5XX_COOLDOWN_SECONDS=30
ROUTER_429_DEFAULT_COOLDOWN_SECONDS=60
```

## 7. Internal Docker Endpoint

Untuk provider CodeBuddy2API pada VPS yang sama, gunakan:

```text
http://cb2api:8001/codebuddy/v1
```

Jangan menggunakan:

```text
https://cb2api.heracles.id/codebuddy/v1
http://127.0.0.1:8001/codebuddy/v1
```

Tambahkan startup validation atau provider test yang memeriksa:

```text
GET http://cb2api:8001/health
```

Namun health check tidak boleh terlalu sering atau memicu load besar.

## 8. Safe Telemetry

Setiap inbound request memiliki `request_id`.

Catat:

```text
request_id
provider
key_fingerprint
attempt
max_attempts
base_url_host
connect_ms
time_to_headers_ms
time_to_first_chunk_ms
stream_duration_ms
stream_started
upstream_status
error_type
fallback_reason
cooldown_seconds
downstream_disconnected
```

Jangan log:

- raw API key;
- Authorization;
- prompt;
- isi file;
- tool result;
- cookie.

Contoh:

```text
request_id=abc123 provider=codebuddy attempt=1 stage=connect_start
request_id=abc123 provider=codebuddy attempt=1 connected_ms=12
request_id=abc123 provider=codebuddy first_chunk_ms=2450
request_id=abc123 stream_started=true
request_id=abc123 retry_skipped_reason=stream_already_started
```

## 9. Preserve Error Detail

Jangan mengubah seluruh error menjadi:

```text
fetch connect timeout
```

Jika upstream memberi status HTTP, pertahankan:

```json
{
  "error": {
    "message": "...",
    "type": "invalid_request_error",
    "code": "invalid_request"
  }
}
```

Bedakan error:

```text
connect_timeout
first_byte_timeout
stream_idle_timeout
upstream_http_error
client_cancelled
payload_invalid
```

## 10. Concurrency

Audit apakah fallback atau retry dijalankan paralel.

Jangan menjalankan banyak provider secara paralel untuk satu request kecuali memang mode hedging diaktifkan secara eksplisit.

Tambahkan konfigurasi:

```env
ROUTER_MAX_CONCURRENT_REQUESTS_PER_PROVIDER=4
ROUTER_MAX_CONCURRENT_REQUESTS_PER_KEY=2
```

Pastikan tool-heavy request tidak menggandakan attempt secara paralel.

## 11. Tests

Tambahkan automated tests:

1. HTTP 400 tidak di-retry.
2. HTTP 422 tidak di-retry.
3. HTTP 429 di-retry maksimal sesuai konfigurasi.
4. Connect timeout dibedakan dari first-byte timeout.
5. Setelah first chunk, tidak ada retry.
6. Tool call pertama menandai stream sudah dimulai.
7. Client disconnect membatalkan upstream.
8. Client disconnect tidak menandai key unavailable.
9. Connect timeout hanya memberi cooldown pendek.
10. 401 menandai key invalid.
11. 429 memakai `Retry-After` bila tersedia.
12. Maksimal retry default adalah 2.
13. Tidak ada retry paralel.
14. Error body OpenAI-compatible dipertahankan.
15. Raw API key tidak muncul di log.
16. Provider internal `http://cb2api:8001` dapat divalidasi.
17. Stream SSE normal diteruskan incremental.
18. `[DONE]` hanya dikirim sekali.

## Acceptance Criteria

Pekerjaan selesai jika:

- HTTP 400 langsung dikembalikan tanpa retry;
- `attempt 1/10` tidak lagi menjadi default;
- connect timeout tidak membatasi seluruh request;
- first-byte timeout cocok untuk request agentic;
- tidak ada fallback setelah stream dimulai;
- tidak ada tool call ganda;
- client disconnect menghentikan seluruh attempt;
- key tidak ditandai unavailable karena payload invalid;
- retry/fallback dapat dijelaskan dari log;
- semua test lulus.

## Output Agent

Sebelum implementasi:

- analisis retry saat ini;
- analisis timeout saat ini;
- jelaskan kapan key ditandai unavailable;
- sebutkan file yang akan diubah.

Setelah implementasi:

- daftar file yang diubah;
- tabel retry policy;
- konfigurasi environment baru;
- contoh telemetry;
- hasil test;
- perintah deployment;
- catatan backward compatibility.

Implementasikan perubahan secara langsung. Jangan hanya memberikan saran.
```