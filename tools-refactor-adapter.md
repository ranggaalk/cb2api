# Refactor CodeBuddy2API Menjadi Adapter V2 yang Stabil untuk Claude Code dan 9Router

Kerjakan langsung pada repository:

```text
https://github.com/xueyue33/codebuddy2api
```

Gunakan implementasi provider berikut hanya sebagai referensi teknis untuk normalisasi messages, tool schema, tool calls, streaming, dan respons CodeBuddy:

```text
https://github.com/priyo000/etteum-pool
```

Fokus referensi:

```text
src/proxy/providers/codebuddy.ts
```

Jangan menyalin dashboard, database, Camoufox, sistem login, account pool, atau arsitektur Etteum secara keseluruhan.

## Latar Belakang

CodeBuddy2API saat ini digunakan dengan alur:

```text
Claude Code
→ 9Router
→ CodeBuddy2API
→ https://www.codebuddy.ai/v2/chat/completions
```

9Router tetap bertanggung jawab atas:

- round-robin API key;
- fallback provider;
- pemilihan model;
- API key per provider.

CodeBuddy2API hanya bertindak sebagai adapter stateless:

```text
OpenAI/Claude-compatible request
→ CodeBuddy-compatible request
→ CodeBuddy response
→ OpenAI-compatible response
```

API key CodeBuddy dikirim oleh 9Router pada setiap request:

```http
Authorization: Bearer <CODEBUDDY_API_KEY>
```

Adapter harus menggunakan key tersebut hanya untuk request terkait.

## Masalah yang Pernah Terjadi

Implementasi lama mengalami masalah berikut:

```text
- Message N must have role and content fields
- Invalid tool parameters
- tool_result hilang dari context
- model tidak dapat melihat isi file yang sudah dibaca
- tool call dan tool result tidak cocok
- function.arguments diproses ketika JSON masih berupa fragmen
- request tool-heavy terus berulang
- retry setelah stream mulai
- beberapa request upstream berjalan setelah client disconnect
- request berhenti setelah outcome=prepared
- timeout sekitar 60 detik sebelum first byte
- 9Router menampilkan fetch connect timeout
- conversation dengan 30 tools menjadi sangat lambat
- unknown model pernah diubah diam-diam menjadi auto-chat
- system prompt Claude Code pernah disanitasi sehingga instruksi tool hilang
```

Jangan menambahkan patch baru ke handler lama tanpa struktur yang jelas.

## Tujuan Utama

Bangun `CodeBuddyAdapterV2` yang terisolasi, dapat diuji, dan tetap kompatibel dengan endpoint serta konfigurasi deployment yang sudah ada.

Target arsitektur:

```text
src/
├── codebuddy_router.py
└── adapters/
    └── codebuddy/
        ├── __init__.py
        ├── adapter.py
        ├── config.py
        ├── models.py
        ├── request_mapper.py
        ├── message_normalizer.py
        ├── tool_schema_adapter.py
        ├── tool_call_state.py
        ├── transport.py
        ├── stream_decoder.py
        ├── response_mapper.py
        ├── validation.py
        └── errors.py
```

Struktur boleh disesuaikan dengan repository, tetapi pemisahan tanggung jawab harus tetap jelas.

---

# 1. Feature Flag dan Backward Compatibility

Tambahkan:

```env
CODEBUDDY_ADAPTER_VERSION=v2
```

Nilai yang didukung:

```text
legacy
v2
```

Default sementara:

```env
CODEBUDDY_ADAPTER_VERSION=v2
```

Routing:

```python
if settings.codebuddy_adapter_version == "v2":
    return await codebuddy_adapter_v2.chat_completion(...)
return await legacy_handler(...)
```

Jangan menghapus implementasi lama sampai seluruh regression test v2 lulus.

Endpoint lama harus tetap tersedia:

```text
POST /codebuddy/v1/chat/completions
GET  /codebuddy/v1/models
GET  /health
```

---

# 2. API Key Passthrough Harus Request-Local

Mode utama:

```env
CODEBUDDY_CLIENT_AUTH_MODE=passthrough
```

Perilaku:

```text
Authorization Bearer dari request 9Router
→ dipakai sebagai API key CodeBuddy upstream
→ hanya untuk request tersebut
```

Ketentuan wajib:

- jangan menyimpan key ke global state;
- jangan memasukkan key ke TXT pool;
- jangan memasukkan key ke database;
- jangan melakukan rotasi internal;
- jangan mencampur key antar-request;
- streaming harus menggunakan key yang sama sampai selesai;
- jangan fallback ke key lain;
- jangan mencetak raw key ke log;
- jangan mencetak header Authorization;
- jangan mencetak `X-Api-Key`.

Gunakan fingerprint aman:

```python
sha256(api_key.encode()).hexdigest()[:8]
```

Header upstream harus configurable:

```env
CODEBUDDY_UPSTREAM_API_KEY_HEADER=bearer
```

Dukungan:

```text
bearer
x-api-key
both
```

Untuk mode `bearer`:

```http
Authorization: Bearer <KEY>
```

Untuk mode `both`:

```http
Authorization: Bearer <KEY>
X-Api-Key: <KEY>
```

---

# 3. Model Resolver yang Ketat

Jangan mengubah unknown model menjadi `auto-chat` secara diam-diam.

Tambahkan:

```env
CODEBUDDY_UNKNOWN_MODEL_POLICY=passthrough
CODEBUDDY_DEFAULT_MODEL=claude-opus-4.7-1m
CODEBUDDY_MODEL_ALIASES=Claude Opus 4.7=claude-opus-4.7-1m
```

Nilai policy:

```text
passthrough
reject
default
```

Perilaku:

```text
passthrough
→ teruskan requested model tanpa perubahan

reject
→ return HTTP 400 unknown_model

default
→ gunakan CODEBUDDY_DEFAULT_MODEL
```

Default harus:

```env
CODEBUDDY_UNKNOWN_MODEL_POLICY=passthrough
```

Log aman:

```text
requested_model
mapped_model
mapping_source
upstream_response_model
```

Jangan mengubah metadata respons untuk menyembunyikan model upstream.

---

# 4. Jangan Sanitasi System Prompt Claude Code

Untuk request agentic dengan tools, system prompt Claude Code harus dipertahankan.

Default:

```env
CODEBUDDY_SANITIZE_AGENT_PROMPT=false
CODEBUDDY_REQUEST_PROFILE=cli
```

Ketentuan:

- jangan mengganti system prompt Claude Code dengan prompt generik;
- jangan menghapus instruksi tool;
- jangan menghapus permission instructions;
- jangan menghapus aturan agent loop;
- jangan menghapus context yang diperlukan model.

Jika sanitasi masih ingin dipertahankan untuk web chat sederhana:

```text
has_tools=true atau Claude Code terdeteksi
→ sanitasi otomatis dilewati
```

Sanitasi hanya boleh menyentuh system message, tidak pernah:

```text
user
assistant
tool
tool_result
```

---

# 5. Message Normalizer

Bangun normalizer deterministik untuk format OpenAI dan Anthropic content blocks.

Dukung:

```text
string content
array content
text blocks
image blocks jika sudah didukung
tool_use
tool_result
assistant tool_calls
OpenAI role=tool
```

Jangan pernah melakukan pola berikut:

```python
if not isinstance(content, str):
    content = ""
```

Karena itu akan membuang `tool_result`, file content, dan array multimodal.

Setiap final upstream message wajib memiliki:

```text
role
content
```

Assistant yang hanya berisi tool calls:

```json
{
  "role": "assistant",
  "content": "",
  "tool_calls": [
    {
      "id": "toolu_xxx",
      "type": "function",
      "function": {
        "name": "Read",
        "arguments": "{\"file_path\":\"portfolio.html\"}"
      }
    }
  ]
}
```

Tool result:

```json
{
  "role": "tool",
  "tool_call_id": "toolu_xxx",
  "content": "isi hasil tool"
}
```

Ketentuan:

- pertahankan urutan message;
- jangan menduplikasi user message;
- jangan menduplikasi tool call;
- jangan menduplikasi tool result;
- jangan menggabungkan tool result ke user message yang tidak terkait;
- jangan membuang content array;
- jangan membuat message hanya berisi `tool_calls` tanpa `role` dan `content`;
- jangan membuat message hanya berisi `tool_call_id`.

---

# 6. Konversi Anthropic Tool Blocks

Input seperti:

```json
{
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "Saya akan membaca file."
    },
    {
      "type": "tool_use",
      "id": "toolu_123",
      "name": "Read",
      "input": {
        "file_path": "portfolio.html"
      }
    }
  ]
}
```

Harus dikonversi menjadi:

```json
{
  "role": "assistant",
  "content": "Saya akan membaca file.",
  "tool_calls": [
    {
      "id": "toolu_123",
      "type": "function",
      "function": {
        "name": "Read",
        "arguments": "{\"file_path\":\"portfolio.html\"}"
      }
    }
  ]
}
```

Input tool result:

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_123",
      "content": "<html>...</html>"
    }
  ]
}
```

Harus menjadi:

```json
{
  "role": "tool",
  "tool_call_id": "toolu_123",
  "content": "<html>...</html>"
}
```

Jika satu user content array berisi kombinasi:

```text
text
tool_result
text
```

pecah menjadi beberapa message yang urut dan valid tanpa kehilangan isi.

Pertahankan hubungan:

```text
tool_use.id
↔ tool_result.tool_use_id
```

---

# 7. Validasi Urutan Tool Call

Sebelum request dikirim upstream, validasi seluruh conversation.

Aturan:

1. Setiap tool result harus memiliki preceding assistant tool call dengan ID yang sama.
2. Setiap assistant tool call harus memiliki `content`, minimal string kosong.
3. Setiap tool result harus memiliki `role=tool`.
4. Setiap tool result harus memiliki `tool_call_id`.
5. Tool call ID tidak boleh berubah selama conversion.
6. Function arguments harus berupa JSON string valid.
7. Jangan mengirim unsupported Anthropic blocks ke upstream.

Jika invalid, return HTTP `400` lokal:

```json
{
  "error": {
    "message": "Invalid tool conversation structure at message 13",
    "type": "invalid_request_error",
    "code": "invalid_tool_conversation"
  }
}
```

Jangan mengirim payload malformed ke CodeBuddy.

---

# 8. Tool Schema Adapter

Port konsep sanitasi tool schema dari Etteum.

Dukung:

```text
type
properties
required
items
enum
additionalProperties
anyOf
oneOf
allOf
$ref
$defs
definitions
```

Ketentuan:

- resolve local `$ref` bila diperlukan;
- jangan membuang required fields;
- jangan mengganti nama parameter;
- jangan mengubah struktur nested object secara sembarangan;
- jangan menghapus array item schema;
- jangan mengirim field JSON Schema yang tidak didukung upstream tanpa sanitasi;
- pertahankan nama tool persis seperti yang diterima dari Claude Code.

Buat mapping schema yang deterministik dan memiliki unit test.

---

# 9. State Machine untuk Streaming Tool Calls

Jangan memproses `function.arguments` per chunk.

Arguments dapat datang seperti:

```text
chunk 1: {"file_
chunk 2: path":"port
chunk 3: folio.html"}
```

Buat state:

```python
@dataclass
class ToolCallState:
    index: int
    id: str | None
    name: str | None
    argument_fragments: list[str]
    emitted: bool = False
```

Kelompokkan fragmen berdasarkan:

```text
tool_call index
tool_call ID
```

Alur:

```text
chunk pertama
→ buat state

chunk berikutnya
→ append argument fragment

finish_reason=tool_calls atau stream selesai
→ gabungkan fragments
→ parse JSON
→ validasi terhadap schema
→ emit satu tool call lengkap
```

Ketentuan:

- jangan emit partial JSON sebagai tool input;
- jangan parse arguments sebelum lengkap;
- jangan mengganti invalid JSON menjadi `{}`;
- jangan mengambil substring JSON pertama;
- jangan menghapus parameter yang dianggap tidak dikenal;
- jangan mengirim tool call dua kali;
- jangan mengubah tool call ID;
- jangan mengirim `input` sebagai string pada format Anthropic;
- `input` harus object setelah JSON selesai di-parse.

Jika JSON tetap invalid setelah tool call selesai, return error terstruktur. Jangan memperbaikinya secara heuristik menjadi `{}`.

---

# 10. Response Mapper

Dukung respons:

```text
assistant text
reasoning_content bila ada
tool_calls
finish_reason
usage
moderation response
normal completion
streaming completion
```

Untuk OpenAI streaming:

```json
{
  "choices": [
    {
      "index": 0,
      "delta": {
        "content": "Halo"
      },
      "finish_reason": null
    }
  ]
}
```

Untuk tool calls, emit fragmen yang valid dan konsisten.

Untuk non-streaming:

```json
{
  "id": "...",
  "object": "chat.completion",
  "model": "claude-opus-4.7-1m",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "...",
        "tool_calls": []
      },
      "finish_reason": "stop"
    }
  ]
}
```

Jangan return objek `chat.completion.chunk` pada `stream=false`.

Jangan menganggap `reasoning_content` sebagai final answer.

---

# 11. True End-to-End Streaming

Gunakan shared `httpx.AsyncClient`.

Contoh konfigurasi:

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

Lifecycle:

```text
startup → buat client
shutdown → tutup client
```

Jangan membuat client baru untuk setiap request.

Pada stream path gunakan:

```python
client.stream(...)
```

atau:

```python
client.send(request, stream=True)
```

Dilarang:

```python
await response.aread()
response.text
response.json()
list(response.aiter_lines())
```

sebelum downstream menerima stream.

Response SSE harus memiliki:

```http
Content-Type: text/event-stream
Cache-Control: no-cache, no-transform
X-Accel-Buffering: no
```

Pastikan GZip tidak membuffer SSE.

Akhiri normal stream tepat satu kali:

```text
data: [DONE]
```

---

# 12. Timeout Terpisah

Tambahkan:

```env
CODEBUDDY_CONNECT_TIMEOUT_SECONDS=30
CODEBUDDY_POOL_TIMEOUT_SECONDS=30
CODEBUDDY_WRITE_TIMEOUT_SECONDS=60
CODEBUDDY_HEADERS_TIMEOUT_SECONDS=300
CODEBUDDY_FIRST_CHUNK_TIMEOUT_SECONDS=300
CODEBUDDY_STREAM_IDLE_TIMEOUT_SECONDS=600
CODEBUDDY_MAX_CONCURRENT_UPSTREAM_REQUESTS=20
CODEBUDDY_UPSTREAM_QUEUE_TIMEOUT_SECONDS=60
```

Bedakan error:

```text
upstream_connect_timeout
upstream_pool_timeout
upstream_headers_timeout
upstream_first_chunk_timeout
upstream_stream_idle_timeout
upstream_queue_timeout
```

Jangan menyebut semua timeout sebagai:

```text
fetch connect timeout
```

---

# 13. Cancellation dan Disconnect

Jika downstream disconnect:

- hentikan stream;
- batalkan request upstream;
- tutup response upstream;
- lepaskan semaphore;
- jangan menyimpan task di background;
- jangan retry;
- jangan lanjutkan request setelah Claude Code dihentikan.

Tangani:

```python
asyncio.CancelledError
```

Gunakan `finally` untuk cleanup.

Log:

```text
request_id=...
stage=downstream_disconnected
stage=upstream_cancelled
```

Tidak boleh ada ghost request setelah client berhenti.

---

# 14. Retry Policy Internal

Dalam mode passthrough, adapter idealnya tidak melakukan rotasi atau fallback.

Tidak boleh retry:

```text
400 invalid_request
400 content_filter
401
403
404
422
```

Retry maksimal satu kali hanya sebelum event pertama untuk:

```text
408
429
502
503
504
connect reset
connect timeout
```

Setelah salah satu event berikut diteruskan:

```text
assistant content
reasoning delta
tool call
role delta
SSE event valid
```

maka:

```text
stream_started=true
```

Setelah itu:

- jangan retry;
- jangan fallback;
- jangan mengulang tool call;
- jangan mengirim duplicate content.

9Router tetap menjadi pihak utama yang menentukan fallback.

---

# 15. Telemetry Bertahap

Setiap request memiliki `request_id`.

Log stage:

```text
request_received
request_normalized
request_validated
request_prepared
upstream_slot_wait_start
upstream_slot_acquired
upstream_send_start
upstream_headers_received
upstream_first_chunk
stream_finished
downstream_disconnected
request_failed
```

Metadata aman:

```text
request_id
requested_model
mapped_model
mapping_source
message_count
tool_count
total_content_length
key_fingerprint
queue_wait_ms
time_to_headers_ms
time_to_first_chunk_ms
stream_duration_ms
chunk_count
finish_reason
upstream_status
```

Untuk debug tool structure, log hanya:

```text
message_index
role
content_type
content_block_types
has_tool_calls
tool_call_ids
tool_result_ids
content_length
```

Jangan log:

- prompt;
- isi file;
- tool argument values;
- raw API key;
- Authorization;
- X-Api-Key;
- cookies;
- complete tool result.

---

# 16. Large Agentic Request Warning

Tambahkan:

```env
CODEBUDDY_WARN_TOTAL_CONTENT_LENGTH=50000
CODEBUDDY_WARN_MESSAGE_COUNT=40
CODEBUDDY_WARN_TOOL_COUNT=30
```

Jika melewati batas, log warning:

```text
large_agentic_request=true
```

Jangan memotong context secara diam-diam.

Jangan menghapus tools secara otomatis.

Jangan meringkas tool result tanpa permintaan eksplisit.

---

# 17. `.env.example`

Tambahkan atau perbarui:

```env
CODEBUDDY_ADAPTER_VERSION=v2

CODEBUDDY_HOST=0.0.0.0
CODEBUDDY_PORT=8001

CODEBUDDY_API_ENDPOINT=https://www.codebuddy.ai
CODEBUDDY_CLIENT_AUTH_MODE=passthrough
CODEBUDDY_REQUEST_PROFILE=cli
CODEBUDDY_SANITIZE_AGENT_PROMPT=false

CODEBUDDY_UNKNOWN_MODEL_POLICY=passthrough
CODEBUDDY_DEFAULT_MODEL=claude-opus-4.7-1m
CODEBUDDY_MODEL_ALIASES=Claude Opus 4.7=claude-opus-4.7-1m

CODEBUDDY_UPSTREAM_API_KEY_HEADER=bearer

CODEBUDDY_CONNECT_TIMEOUT_SECONDS=30
CODEBUDDY_POOL_TIMEOUT_SECONDS=30
CODEBUDDY_WRITE_TIMEOUT_SECONDS=60
CODEBUDDY_HEADERS_TIMEOUT_SECONDS=300
CODEBUDDY_FIRST_CHUNK_TIMEOUT_SECONDS=300
CODEBUDDY_STREAM_IDLE_TIMEOUT_SECONDS=600

CODEBUDDY_MAX_CONCURRENT_UPSTREAM_REQUESTS=20
CODEBUDDY_UPSTREAM_QUEUE_TIMEOUT_SECONDS=60

CODEBUDDY_WARN_TOTAL_CONTENT_LENGTH=50000
CODEBUDDY_WARN_MESSAGE_COUNT=40
CODEBUDDY_WARN_TOOL_COUNT=30

CODEBUDDY_LOG_LEVEL=INFO
```

---

# 18. Automated Tests

Gunakan mock upstream. Jangan membutuhkan API key CodeBuddy asli.

Test wajib:

## Basic chat

1. Chat sederhana non-streaming.
2. Chat sederhana streaming.
3. Stream selesai dengan satu `[DONE]`.
4. `stream=false` menghasilkan `message.content`.

## Model mapping

5. `claude-opus-4.7-1m` tidak berubah menjadi `auto-chat`.
6. Unknown model passthrough.
7. Policy reject menghasilkan HTTP 400.
8. Alias UI dipetakan dengan benar.

## Messages

9. Semua final message memiliki `role`.
10. Semua final message memiliki `content`.
11. Content string tetap utuh.
12. Content array tidak dibuang.
13. Multimodal content tetap valid bila didukung.

## Tool conversion

14. Anthropic `tool_use` menjadi OpenAI `tool_calls`.
15. Anthropic `tool_result` menjadi role `tool`.
16. `tool_use.id` cocok dengan `tool_result.tool_use_id`.
17. Assistant tool call memiliki `content=""`.
18. Tool result mempertahankan isi file.
19. Tool result tidak digabung ke user message lain.
20. Multiple tool calls pada satu assistant response.
21. Multiple tool results pada satu user content array.

## Streaming tool calls

22. Arguments yang terpecah dalam banyak chunk digabung.
23. Arguments baru di-parse setelah selesai.
24. Nested JSON arguments.
25. Escaped string arguments.
26. Dua tool calls bersamaan.
27. Tool call ID dipertahankan.
28. Tool call tidak di-emit dua kali.
29. Invalid complete JSON menghasilkan error, bukan `{}`.
30. `AskUserQuestion` schema tetap valid.
31. `Read` schema tetap valid.
32. `Write` schema tetap valid.
33. `Edit` schema tetap valid.
34. `Bash` schema tetap valid.

## Tool workflow

35. `Read → tool_result → assistant` melihat isi file.
36. `Read → AskUserQuestion → user answer → Edit`.
37. `Read → Grep → Edit → Read`.
38. Conversation dengan 25+ messages.
39. Request dengan 30 tool definitions.
40. Tool result tidak hilang setelah banyak iterasi.

## Streaming transport

41. Mock upstream mengirim tiga chunk dengan delay.
42. Downstream menerima chunk secara incremental.
43. Response tidak dibuffer sampai selesai.
44. Shared HTTP client digunakan ulang.
45. SSE headers benar.
46. GZip tidak membuffer SSE.
47. First chunk timeout berbeda dari connect timeout.
48. Stream idle timeout berbeda dari headers timeout.

## Cancellation

49. Client disconnect membatalkan upstream.
50. Semaphore dilepas setelah cancellation.
51. Tidak ada ghost request.
52. Tidak ada retry setelah disconnect.
53. Tidak ada retry setelah first chunk.

## Errors

54. HTTP 400 tidak di-retry.
55. HTTP 401 tidak di-retry.
56. HTTP 403 tidak di-retry.
57. HTTP 422 tidak di-retry.
58. HTTP 429 boleh di-retry maksimal sekali sebelum stream.
59. HTTP 5xx boleh di-retry maksimal sekali sebelum stream.
60. Error upstream mempertahankan status dan body yang sudah disanitasi.

## Security

61. Concurrent request dengan dua key berbeda tidak tertukar.
62. Raw API key tidak muncul di log.
63. Authorization tidak muncul di log.
64. Isi file tidak muncul di log.
65. Tool argument values tidak muncul di log.

---

# 19. Skenario Acceptance Manual

Uji melalui:

```text
Claude Code
→ 9Router
→ CodeBuddyAdapterV2
```

Base URL 9Router:

```text
http://cb2api:8001/codebuddy/v1
```

Skenario wajib:

```text
1. "Halo, siapa kamu?"
2. "Baca portfolio.html"
3. "Buat portfolio ini lebih bagus"
4. Model mengirim AskUserQuestion
5. User menjawab pertanyaan
6. Model membaca file
7. Model mengedit atau overwrite file
8. Model membaca ulang hasil
9. Model selesai tanpa retry
```

Uji juga:

```text
- cancel request saat upstream belum memberi first token;
- cancel saat tool loop berjalan;
- lanjutkan conversation dengan 25+ messages;
- dua Claude Code session paralel dengan API key berbeda;
- prompt dengan 30 tools;
- upstream menunggu 90 detik sebelum first chunk.
```

Tidak boleh terjadi:

```text
Message must have role and content
Invalid tool parameters karena fragment JSON
tool result hilang
file sudah dibaca tetapi isi tidak terlihat model
duplicate tool call
ghost request
retry setelah stream dimulai
model otomatis berubah ke auto-chat
system prompt Claude Code dihapus
```

---

# 20. Acceptance Criteria

Implementasi selesai jika:

- CodeBuddy2API memiliki adapter v2 terpisah;
- router tidak lagi berisi seluruh logika provider;
- API key passthrough aman untuk concurrent request;
- model tidak diubah diam-diam;
- tool calls dan tool results selalu valid;
- file content tidak hilang dari context;
- streamed arguments direkonstruksi dengan state machine;
- invalid arguments tidak diubah menjadi `{}`;
- streaming diteruskan incremental;
- cancellation menghentikan upstream;
- tidak ada request berjalan setelah client disconnect;
- timeout dibedakan berdasarkan tahap;
- tidak ada retry setelah stream mulai;
- request tool-heavy dengan 30 tools berhasil;
- seluruh automated test lulus;
- implementasi legacy masih bisa dipilih melalui feature flag.

---

# Output Wajib dari Agent

Sebelum implementasi, tampilkan:

```text
1. Analisis arsitektur lama.
2. Root cause utama.
3. Daftar kode yang akan dipertahankan.
4. Daftar kode yang akan dipindahkan ke adapter.
5. Referensi logika yang akan di-port dari Etteum.
6. Daftar file yang akan dibuat atau diubah.
```

Setelah implementasi, tampilkan:

```text
1. Daftar file yang dibuat atau diubah.
2. Diagram alur request v2.
3. Penjelasan message normalizer.
4. Penjelasan tool-call state machine.
5. Penjelasan shared HTTP client.
6. Penjelasan cancellation.
7. Contoh `.env`.
8. Hasil lint.
9. Hasil type checking.
10. Hasil automated tests.
11. Curl streaming.
12. Curl non-streaming.
13. Contoh konfigurasi 9Router.
14. Perintah deployment Docker.
15. Catatan backward compatibility.
16. Risiko yang masih tersisa.
```

Implementasikan perubahan secara langsung pada repository.

Jangan hanya memberikan analisis, pseudocode, atau potongan kode.

Jangan memodifikasi 9Router dalam pekerjaan ini.

Jangan menyalin seluruh Etteum.

Port hanya logika adapter/provider yang relevan dan tulis ulang agar sesuai dengan struktur Python CodeBuddy2API.