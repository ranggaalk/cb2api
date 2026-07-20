# Brief Update — Per-Request CodeBuddy API Key Passthrough

Repository:

```text
https://github.com/xueyue33/codebuddy2api
```

## Objective

Tambahkan mode baru agar client seperti 9Router dapat mengirim **API key CodeBuddy asli pada setiap request**.

Dalam mode ini, CodeBuddy2API tidak memilih key dari TXT dan tidak memakai `CODEBUDDY_PASSWORD` sebagai API key client untuk endpoint inference.

Alur:

```text
9Router provider 1 — CodeBuddy key A ┐
9Router provider 2 — CodeBuddy key B ├─ round-robin/fallback oleh 9Router
9Router provider 3 — CodeBuddy key C ┘
                ↓
Authorization: Bearer <CODEBUDDY_API_KEY>
                ↓
CodeBuddy2API
                ↓
meneruskan key request tersebut ke CodeBuddy Global
```

CodeBuddy2API hanya bertindak sebagai protocol adapter/OpenAI-compatible relay. Rotasi, fallback, dan pemilihan key dilakukan oleh 9Router.

## Required Modes

Pertahankan metode lama dan tambahkan metode baru:

```env
CODEBUDDY_CLIENT_AUTH_MODE=relay
```

Nilai yang didukung:

```text
relay
passthrough
hybrid
```

Perilaku:

### `relay`

Perilaku lama:

- Client mengirim `Authorization: Bearer <CODEBUDDY_PASSWORD>`.
- Relay memvalidasi `CODEBUDDY_PASSWORD`.
- Upstream credential dipilih dari TXT atau credential manager lama.

### `passthrough`

Perilaku baru:

- Client mengirim `Authorization: Bearer <CODEBUDDY_API_KEY_ASLI>`.
- Jangan bandingkan bearer tersebut dengan `CODEBUDDY_PASSWORD`.
- Gunakan bearer tersebut hanya untuk request upstream yang sedang berjalan.
- Jangan menyimpan, mencatat, cache, atau memasukkannya ke key pool.
- Satu request/stream harus memakai key yang sama sampai selesai.
- Relay tidak melakukan round-robin atau failover antar-key.

### `hybrid`

- Jika bearer sama dengan `CODEBUDDY_PASSWORD`, gunakan key pool TXT/credential lama.
- Jika bearer berbeda, perlakukan bearer sebagai upstream CodeBuddy API key per-request.
- Jangan menebak berdasarkan prefix key.
- Default tetap `relay`.

Default:

```env
CODEBUDDY_CLIENT_AUTH_MODE=relay
```

Ini menjaga backward compatibility.

## Separate Admin Authentication

Jangan gunakan API key CodeBuddy per-request untuk mengakses dashboard atau endpoint admin.

Tambahkan:

```env
CODEBUDDY_ADMIN_PASSWORD=admin_secret
```

Ketentuan:

- Dashboard, credential management, status key pool, reload, dan settings harus memakai `CODEBUDDY_ADMIN_PASSWORD`.
- Untuk backward compatibility, bila `CODEBUDDY_ADMIN_PASSWORD` kosong, fallback ke `CODEBUDDY_PASSWORD`.
- Endpoint inference tetap mengikuti `CODEBUDDY_CLIENT_AUTH_MODE`.

## Incoming Key Extraction

Untuk endpoint inference:

```text
POST /codebuddy/v1/chat/completions
GET  /codebuddy/v1/models
```

Ambil key dari:

```http
Authorization: Bearer <key>
```

Ketentuan:

- Tolak header kosong atau format non-Bearer dengan `401`.
- Jangan menampilkan raw key di log, traceback, response, metrics, atau dashboard.
- Jika perlu identifikasi debugging, gunakan fingerprint SHA-256 pendek atau masked key.
- Jangan simpan incoming key ke file TXT, JSON credential, database, atau memory global.
- Key hanya boleh berada pada scope request.

## Upstream Authentication

Gunakan incoming per-request key untuk membangun header CodeBuddy Global.

Tambahkan konfigurasi:

```env
CODEBUDDY_UPSTREAM_API_KEY_HEADER=both
```

Nilai:

```text
x-api-key
bearer
both
```

Mapping:

```text
x-api-key:
X-API-Key: <incoming-key>

bearer:
Authorization: Bearer <incoming-key>

both:
X-API-Key: <incoming-key>
Authorization: Bearer <incoming-key>
```

Gunakan default yang sesuai dengan implementasi upstream yang sudah terbukti bekerja pada project. Jangan meneruskan header client secara mentah; bangun ulang header upstream secara eksplisit agar header internal tidak ikut bocor.

## Request Lifecycle

Untuk mode `passthrough`:

1. Extract bearer dari request.
2. Validasi format dasar.
3. Buat request-scoped upstream client/context.
4. Kirim key ke CodeBuddy upstream.
5. Pertahankan key yang sama sampai response atau stream selesai.
6. Hapus reference key setelah request selesai.
7. Jangan retry menggunakan key lain.
8. Boleh retry koneksi dengan key yang sama hanya jika aman dan belum ada chunk stream yang dikirim.
9. Setelah chunk pertama dikirim, jangan restart request.

## Error Handling

Teruskan status upstream secara aman:

- `401`: key CodeBuddy tidak valid.
- `403`: key tidak punya izin atau diblokir.
- `429`: key mencapai limit.
- `5xx`/timeout: upstream unavailable.

Format OpenAI-compatible:

```json
{
  "error": {
    "message": "Upstream CodeBuddy rejected the supplied API key",
    "type": "authentication_error",
    "code": "upstream_api_key_rejected"
  }
}
```

Jangan sertakan key atau upstream authorization header pada error.

9Router harus dapat membaca status `401`, `403`, `429`, dan `5xx` agar fallback/round-robin bekerja sesuai konfigurasi 9Router. Jangan mengubah semuanya menjadi HTTP `200`.

## Existing TXT Mode

Fitur TXT sebelumnya harus tetap tersedia untuk mode `relay`:

```env
CODEBUDDY_AUTH_MODE=api_key_file
CODEBUDDY_API_KEYS_FILE=./config/codebuddy_api_keys.txt
```

Ketentuan prioritas:

```text
CODEBUDDY_CLIENT_AUTH_MODE=passthrough
→ selalu gunakan key dari request
→ abaikan TXT untuk inference

CODEBUDDY_CLIENT_AUTH_MODE=relay
→ validasi CODEBUDDY_PASSWORD
→ gunakan TXT/credential manager

CODEBUDDY_CLIENT_AUTH_MODE=hybrid
→ CODEBUDDY_PASSWORD memakai pool
→ bearer lain memakai passthrough
```

## 9Router Configuration

Setiap provider 9Router memakai Base URL yang sama, tetapi API key berbeda.

Provider A:

```text
Name       : CodeBuddy A
Prefix     : cba
Base URL   : http://codebuddy2api:8001/codebuddy/v1
API Key    : CODEBUDDY_KEY_A
Model ID   : glm-5.2
```

Provider B:

```text
Name       : CodeBuddy B
Prefix     : cbb
Base URL   : http://codebuddy2api:8001/codebuddy/v1
API Key    : CODEBUDDY_KEY_B
Model ID   : glm-5.2
```

Provider C:

```text
Name       : CodeBuddy C
Prefix     : cbc
Base URL   : http://codebuddy2api:8001/codebuddy/v1
API Key    : CODEBUDDY_KEY_C
Model ID   : glm-5.2
```

9Router kemudian mengatur combo/fallback/round-robin. Relay tidak perlu mengetahui daftar semua key.

## Security Requirements

- Jangan log raw incoming `Authorization`.
- Tambahkan redaction untuk header sensitif.
- Jangan mencantumkan key pada access log.
- Jangan menyimpan key per-request ke usage statistics.
- Jangan mengembalikan key melalui exception.
- Jangan mengirim key client ke endpoint selain CodeBuddy upstream.
- Jangan memperbolehkan bearer CodeBuddy membuka endpoint admin.
- Dashboard/admin tetap memakai password terpisah.
- Tambahkan test yang memastikan raw key tidak muncul pada log dan response.

## Tests

Tambahkan automated tests untuk:

1. `relay` tetap memvalidasi `CODEBUDDY_PASSWORD`.
2. `passthrough` menerima bearer yang berbeda dari `CODEBUDDY_PASSWORD`.
3. Incoming bearer diteruskan ke upstream dengan header yang benar.
4. Dua request bersamaan memakai key masing-masing tanpa tertukar.
5. Streaming mempertahankan key request yang sama.
6. Key tidak masuk ke manager global atau key pool.
7. `401`, `403`, `429`, dan `5xx` tidak diubah menjadi `200`.
8. Raw API key tidak muncul di log, response, traceback, atau statistics.
9. Endpoint admin menolak API key CodeBuddy.
10. `hybrid` membedakan relay password dan passthrough key.
11. TXT/key-pool lama masih bekerja pada mode `relay`.
12. `/v1/models` dapat divalidasi memakai per-request key.
13. Missing/invalid Authorization menghasilkan `401`.

Gunakan mock upstream. Jangan membutuhkan key CodeBuddy asli untuk unit test.

## Documentation Updates

Update:

```text
README.md
.env.example
docker-compose.yml
```

Contoh mode passthrough:

```env
CODEBUDDY_CLIENT_AUTH_MODE=passthrough
CODEBUDDY_ADMIN_PASSWORD=admin_secret
CODEBUDDY_UPSTREAM_API_KEY_HEADER=both
```

Dalam mode passthrough, `CODEBUDDY_PASSWORD` boleh tetap ada untuk kompatibilitas, tetapi tidak digunakan untuk autentikasi endpoint inference.

## Acceptance Criteria

Implementasi selesai jika:

- 9Router dapat mengirim API key CodeBuddy asli sebagai Bearer token.
- Relay menggunakan key tersebut hanya untuk request terkait.
- Beberapa provider 9Router dengan key berbeda dapat memakai satu Base URL relay.
- 9Router dapat melakukan fallback berdasarkan status upstream yang sebenarnya.
- Relay tidak melakukan rotasi internal pada mode passthrough.
- TXT dan credential mode lama tetap bekerja.
- Dashboard/admin tidak dapat dibuka memakai API key CodeBuddy.
- Streaming stabil dan tidak mencampur key antar-request.
- Tidak ada raw API key di log atau response.
- Semua tests lulus.

## Required Agent Output

Sebelum implementasi:

- Analisis alur autentikasi client saat ini.
- Analisis lokasi pemilihan upstream credential.
- Sebutkan file yang perlu diubah.

Setelah implementasi:

- Daftar file yang dibuat/diubah.
- Ringkasan perubahan arsitektur.
- Contoh `.env`.
- Contoh konfigurasi 9Router.
- Curl mode passthrough.
- Curl mode relay lama.
- Hasil tests.
- Catatan keamanan dan backward compatibility.
