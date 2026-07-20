# Brief Update — Multi API Key TXT Rotation

Update repository:

```text
https://github.com/xueyue33/codebuddy2api
```

## Goal

Tambahkan dukungan banyak upstream CodeBuddy API key dari satu file TXT, dengan satu API key per baris.

Alur akhir:

```text
9Router
→ satu CODEBUDDY_PASSWORD
→ satu instance CodeBuddy2API
→ rotasi otomatis API key dari file TXT
→ CodeBuddy Global
```

Tidak boleh membutuhkan banyak container atau banyak provider di 9Router.

## API Key File

Gunakan file default:

```text
config/codebuddy_api_keys.txt
```

Format:

```text
api_key_akun_1
api_key_akun_2
api_key_akun_3
```

Ketentuan:

- Satu API key per baris.
- Abaikan baris kosong.
- Abaikan baris yang diawali `#`.
- Trim whitespace.
- Hapus key duplikat.
- Jangan pernah menampilkan raw API key di log atau response.
- Tambahkan file tersebut ke `.gitignore`.

Tambahkan contoh aman:

```text
config/codebuddy_api_keys.example.txt
```

## Environment Variables

Tambahkan:

```env
CODEBUDDY_AUTH_MODE=api_key_file
CODEBUDDY_API_KEYS_FILE=./config/codebuddy_api_keys.txt
CODEBUDDY_API_KEY_ROTATION=round_robin
CODEBUDDY_API_KEY_RELOAD_INTERVAL=5
CODEBUDDY_API_KEY_COOLDOWN_SECONDS=300
```

Pertahankan:

```env
CODEBUDDY_PASSWORD=relay_master_secret
```

`CODEBUDDY_PASSWORD` tetap menjadi API key untuk client atau 9Router. Jangan gunakan sebagai upstream CodeBuddy API key.

Mode yang harus didukung:

```text
api_key_file
credentials
auto
```

Perilaku:

- `api_key_file`: hanya memakai key dari TXT.
- `credentials`: memakai sistem credential lama.
- `auto`: memakai key TXT jika tersedia, lalu fallback ke credential lama.
- Default gunakan `auto` agar backward-compatible.

## Key Pool Manager

Buat manager baru, misalnya:

```text
src/codebuddy_api_key_manager.py
```

Fitur minimum:

- Membaca key dari TXT.
- Hot reload tanpa restart container.
- Round-robin.
- Aman untuk concurrent async requests.
- Satu request atau stream memakai satu key sampai selesai.
- Menyimpan status runtime tiap key:
  - active
  - cooldown
  - invalid
  - request_count
  - error_count
  - last_used_at
  - cooldown_until
  - last_error

Gunakan fingerprint atau masked key pada log, misalnya:

```text
abcd...wxyz
```

## Failover

Perilaku error:

- `401`: tandai key invalid dan coba key berikutnya.
- `403`: cooldown lalu coba key berikutnya.
- `429`: cooldown sesuai konfigurasi lalu coba key berikutnya.
- Timeout, connection error, atau `5xx`: coba key berikutnya.
- Maksimum retry sebanyak jumlah key aktif.
- Jangan retry key yang sama dalam satu request.
- Jangan pindah key setelah chunk streaming pertama dikirim.
- Jika semua key gagal, return error OpenAI-compatible tanpa data sensitif.

## Existing Endpoints

Jangan merusak endpoint yang sudah ada:

```text
POST /codebuddy/v1/chat/completions
GET  /codebuddy/v1/models
GET  /health
```

Client tetap mengirim:

```http
Authorization: Bearer <CODEBUDDY_PASSWORD>
```

9Router tidak perlu mengetahui upstream key yang sedang digunakan.

## Admin Status

Tambahkan endpoint yang dilindungi `CODEBUDDY_PASSWORD`:

```text
GET  /codebuddy/v1/api-keys/status
POST /codebuddy/v1/api-keys/reload
```

Endpoint status hanya menampilkan masked key, status, request count, error count, waktu penggunaan terakhir, dan cooldown.

Endpoint reload harus memaksa pembacaan ulang TXT tanpa restart.

## Docker

Pastikan folder config di-mount:

```yaml
services:
  codebuddy2api:
    volumes:
      - ./config:/app/config
      - ./.codebuddy_creds:/app/.codebuddy_creds
```

Jangan membuat container baru untuk setiap API key.

## Backward Compatibility

- Jangan hapus sistem credential lama.
- Mode `credentials` harus tetap bekerja.
- Mode `auto` harus fallback ke credential lama ketika TXT kosong.
- Mode `api_key_file` harus memberikan error konfigurasi yang jelas jika file kosong atau tidak tersedia.

## Tests

Tambahkan test minimum untuk:

- Parsing TXT.
- Komentar dan baris kosong.
- Deduplication.
- Round-robin.
- Concurrent requests.
- Streaming memakai satu key.
- Failover 401, 429, dan 5xx.
- Hot reload.
- Key yang dihapus tidak digunakan lagi.
- Tidak ada raw API key pada log atau response.
- Backward compatibility mode lama.

## Acceptance Criteria

Update dianggap selesai jika:

- Menambah key cukup dengan menambahkan satu baris ke TXT.
- Tidak perlu restart container.
- Tidak perlu menambah provider baru di 9Router.
- Rotasi round-robin bekerja.
- Failover otomatis bekerja.
- Streaming tetap stabil.
- API key tidak bocor.
- Sistem credential lama tetap berjalan.
- Deployment tetap satu container.

## Required Output From Agent

Setelah implementasi, tampilkan:

- Daftar file yang dibuat atau diubah.
- Ringkasan arsitektur.
- Contoh `.env`.
- Contoh file TXT.
- Perintah deployment atau upgrade.
- Curl status dan reload.
- Curl chat streaming dan non-streaming.
- Hasil test.
