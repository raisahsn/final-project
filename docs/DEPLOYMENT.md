# Panduan Deploy Lengkap — Tokopedia Review NLP (untuk pemula)

Dokumen ini dibuat **serinci mungkin** untuk kamu yang baru pertama kali deploy.
Tujuannya: supaya **model**, **API (FastAPI)**, dan **UI (Streamlit)** bisa diakses
publik lewat internet.

Kalau kamu cuma mau jalan cepat, baca bagian **"Jalur Cepat"** di bawah, lalu loncat
ke bagian yang kamu butuh.

---

## Daftar Isi

1. [Konsep dasar (wajib baca kalau baru)](#1-konsep-dasar)
2. [Jalur cepat (TL;DR)](#2-jalur-cepat-tldr)
3. [Arsitektur aplikasi](#3-arsitektur-aplikasi)
4. [Prasyarat (akun & tools)](#4-prasyarat)
5. [Struktur folder project](#5-struktur-folder-project)
6. [Menyiapkan artefak model](#6-menyiapkan-artefak-model)
7. [Menjalankan di komputer sendiri (lokal)](#7-menjalankan-di-komputer-sendiri-lokal)
8. [Deploy ke Railway (utama)](#8-deploy-ke-railway-utama)
9. [CI/CD otomatis dengan GitHub Actions](#9-cicd-otomatis-dengan-github-actions)
10. [Alternatif platform deploy](#10-alternatif-platform-deploy)
11. [Validasi & pengujian](#11-validasi--pengujian)
12. [Update model (retrain → redeploy)](#12-update-model-retrain--redeploy)
13. [Monitoring, log, rollback](#13-monitoring-log-rollback)
14. [Biaya & batasan](#14-biaya--batasan)
15. [Troubleshooting](#15-troubleshooting)
16. [Glosarium (istilah)](#16-glosarium)
17. [Checklist akhir](#17-checklist-akhir)

---

## 1. Konsep dasar

Sebelum deploy, pahami dulu 4 komponen utama:

| Komponen | Apa itu | Di project ini |
|---|---|---|
| **Model** | "Otak" AI yang sudah dilatih untuk menebak sentimen & kategori review | File di `models/sentiment_model` dan `models/category_model` |
| **API** | Pintu masuk HTTP. Orang/aplikasi lain mengirim teks review, API membalas prediksi | FastAPI di `src/api/main.py`, endpoint `/predict`, `/predict/batch`, `/health` |
| **UI** | Tampilan web yang bisa diklik manusia. UI memanggil API di belakang layar | Streamlit di `src/streamlit_app/app.py` |
| **Database** | Tempat menyimpan riwayat prediksi | SQLite lokal (default) atau PostgreSQL |

Alur kerja:

```
Pengguna buka UI  →  UI kirim teks ke API  →  API load model & prediksi  →  API simpan ke DB  →  API balas ke UI  →  UI tampilkan hasil
```

**Apa itu "deploy"?**
Menjalankan aplikasi di server yang online 24 jam, supaya bisa diakses lewat URL publik
(misalnya `https://tokopedia-api.up.railway.app`), bukan cuma di laptop kamu.

**Apa itu "CI/CD"?**
- **CI (Continuous Integration)**: setiap kamu push kode, otomatis di-lint & di-test.
- **CD (Continuous Deployment)**: kalau test lolos, otomatis di-deploy ke server.

Di project ini CI/CD pakai **GitHub Actions** + **Railway**.

---

## 2. Jalur cepat (TL;DR)

Untuk yang sudah paham, ini versi 5 langkah:

```bash
# 1. Export model dari notebook → taruh ke models/
unzip -o tokopedia_artifacts.zip
mv tokopedia_artifacts/sentiment_model models/sentiment_model
mv tokopedia_artifacts/category_model models/category_model
git add models && git commit -m "chore: update model" && git push

# 2. Di Railway: buat project + 2 service (api, ui) dari repo GitHub
# 3. Service api  → biarkan Start Command default (Dockerfile sudah pakai $PORT)
#    Service ui   → Start Command override:
#    streamlit run src/streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0
# 4. Service ui → set Variable API_URL = domain publik service api
# 5. Generate domain publik untuk api & ui → selesai
```

Detail setiap langkah ada di bagian berikut.

---

## 3. Arsitektur aplikasi

```
┌─────────────────────┐      HTTP/JSON      ┌────────────────────┐
│  Streamlit UI       │  ─────────────────► │  FastAPI Service   │
│  (port $PORT)       │   API_URL           │  (port $PORT)      │
└─────────────────────┘                     │  ├─ /health        │
                                            │  ├─ /models        │
                                            │  ├─ /predict       │
                                            │  ├─ /predict/batch │
                                            │  └─ /predictions   │
                                            └─────────┬──────────┘
                                                      │ load on startup
                                                      ▼
                                            ┌────────────────────┐
                                            │  Model Artifacts   │
                                            │  models/           │
                                            │  ├ sentiment_model │
                                            │  └ category_model  │
                                            └────────────────────┘
```

Penjelasan endpoint API:

| Endpoint | Method | Fungsi |
|---|---|---|
| `/health` | GET | Cek apakah API hidup & model sudah ke-load |
| `/models` | GET | Info model yang aktif |
| `/predict` | POST | Prediksi 1 review |
| `/predict/batch` | POST | Prediksi banyak review sekaligus |
| `/predictions` | GET | Ambil riwayat prediksi (dengan pagination) |
| `/docs` | GET | Halaman dokumentasi interaktif (Swagger) |

---

## 4. Prasyarat

### 4.1 Akun yang dibutuhkan

| Akun | Untuk apa | Cara daftar |
|---|---|---|
| **GitHub** | Menyimpan kode & menjalankan CI/CD | <https://github.com> → Sign up |
| **Railway** | Menjalankan API & UI online | <https://railway.app> → login pakai GitHub |

### 4.2 Tools di komputer (opsional, untuk cek lokal)

| Tool | Fungsi | Install |
|---|---|---|
| **Git** | Upload kode ke GitHub | <https://git-scm.com/downloads> |
| **Python 3.10/3.11** | Menjalankan notebook & API | <https://www.python.org/downloads> |
| **Docker Desktop** | Menjalankan API+UI dalam container | <https://www.docker.com/products/docker-desktop> |
| **Node.js 20** | Hanya untuk Railway CLI (CI/CD) | <https://nodejs.org> |

> Kamu **tidak wajib** install semua. Untuk deploy ke Railway, yang wajib hanya
> GitHub + Railway. Tools lokal hanya untuk testing sebelum deploy.

---

## 5. Struktur folder project

```
raisah/
├── notebooks/
│   └── tokopedia_full_pipeline_v2.ipynb   # tempat training model
├── src/
│   ├── api/
│   │   └── main.py                        # FastAPI app
│   ├── streamlit_app/
│   │   └── app.py                         # Streamlit UI
│   └── tokopedia_ml/
│       ├── config.py                      # path model, hyperparameter
│       ├── inference.py                   # load model & fungsi predict
│       └── preprocessing.py               # fungsi bersih-bersih teks
├── models/
│   ├── sentiment_model/                   # artefak model sentimen
│   └── category_model/                    # artefak model kategori
├── .github/workflows/
│   ├── ci-cd.yml                          # CI: lint + test + build
│   └── deploy-railway.yml                 # CD: deploy ke Railway
├── Dockerfile                             # resep build image (dipakai Railway)
├── docker-compose.yml                     # jalan lokal: api + ui + db
├── requirements.txt                       # daftar library Python
├── render.yaml                            # config deploy ke Render (alternatif)
└── docs/
    └── DEPLOYMENT.md                      # dokumen ini
```

---

## 6. Menyiapkan artefak model

Model harus di-export dari notebook menjadi **artefak** (file yang bisa di-load API).

### 6.1 Jalankan notebook

Buka `notebooks/tokopedia_full_pipeline_v2.ipynb` (di Google Colab atau Jupyter lokal),
lalu **Run All** sampai selesai.

Cell paling bawah (**Simpan artefak model terbaik**) akan menghasilkan:

```
tokopedia_artifacts.zip
tokopedia_artifacts/sentiment_model/
tokopedia_artifacts/category_model/
```

Tiap folder model berisi:

```
sentiment_model/
├── model.keras          # bobot model
├── tokenizer.json       # kamus kata → angka
├── label_encoder.pkl    # mapping label (positif/negatif/netral)
└── config.json          # metadata (max_len, vocab_size, dll)
```

### 6.2 Download artefak (kalau di Colab)

Cell terakhir otomatis memanggil `files.download(...)`, jadi file zip akan ter-download
ke komputer kamu. Kalau tidak otomatis, klik kanan file di panel Colab → Download.

### 6.3 Masukkan artefak ke folder `models/`

API membaca model dari folder `models/`. Jadi artefak harus dipindah ke sana.

Di komputer kamu (terminal):

```bash
cd /path/ke/raisah

# hapus model lama (kalau ada)
rm -rf models/sentiment_model models/category_model

# ekstrak zip hasil training
unzip -o tokopedia_artifacts.zip

# pindahkan ke folder models/
mv tokopedia_artifacts/sentiment_model models/sentiment_model
mv tokopedia_artifacts/category_model models/category_model

# simpan ke git
git add models/sentiment_model models/category_model
git commit -m "chore: update trained model artifacts"
git push
```

> Model TextCNN + tokenizer ukurannya kecil (beberapa MB), jadi aman di-commit ke Git.
> Kalau suatu hari artefak terlalu besar, simpan di object storage (lihat bagian 10).

---

## 7. Menjalankan di komputer sendiri (lokal)

Sebelum deploy ke internet, selalu cek dulu di laptop. Ada 2 cara: **tanpa Docker**
(lebih ringan) dan **pakai Docker** (lebih mirip kondisi server).

### 7.1 Cara A — tanpa Docker

```bash
cd /path/ke/raisah

# buat virtual environment (sekali saja)
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# install library
pip install -r requirements.txt

# jalankan API (terminal 1)
bash run_api.sh
# API jalan di http://localhost:8000

# jalankan UI (terminal 2, terpisah)
bash run_streamlit.sh
# UI jalan di http://localhost:8501
```

Buka browser:
- API docs: <http://localhost:8000/docs>
- UI: <http://localhost:8501>

### 7.2 Cara B — pakai Docker Compose

Cara ini menjalankan API + UI + PostgreSQL sekaligus, mirip kondisi di server.

```bash
cd /path/ke/raisah
docker compose up --build
```

Tunggu sampai log berhenti (bisa 1–3 menit pertama kali karena download image).
Lalu buka:

- API: <http://localhost:8000>
- UI: <http://localhost:8501>

Untuk berhenti: tekan `Ctrl+C`, lalu:

```bash
docker compose down
```

### 7.3 Uji cepat

Di terminal lain:

```bash
# cek API hidup
curl http://localhost:8000/health

# coba prediksi
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Barang bagus tapi pengiriman lambat"}'
```

Kalau `/health` membalas `{"status":"ok","ready":true,...}`, berarti model sudah ke-load.

---

## 8. Deploy ke Railway (utama)

Railway dipilih karena:
- Otomatis mendeteksi `Dockerfile`.
- Menyediakan env `PORT` otomatis.
- Ada plugin PostgreSQL.
- Bisa auto-deploy dari GitHub.

Kita akan membuat **2 service** dalam **1 project**: `api` dan `ui`.

### 8.1 Buat project Railway

1. Buka <https://railway.app> → login pakai GitHub.
2. Klik **New Project** → **Deploy from GitHub repo**.
3. Pilih repo `raisah` (atau nama repo kamu).
4. Railway otomatis membuat 1 service. Klik service itu → tab **Settings** →
   ubah **Name** menjadi `api`.
5. Tambah service kedua: klik **+ New** (di dalam project) → **GitHub Repo** →
   pilih repo yang sama → ubah namanya menjadi `ui`.

Sekarang kamu punya 2 service: `api` dan `ui`.

### 8.2 Konfigurasi service `api`

Masuk ke service `api` → tab **Settings**:

| Setting | Nilai |
|---|---|
| **Build / Builder** | Dockerfile (otomatis terdeteksi) |
| **Start Command** | **Biarkan kosong/default** — Dockerfile sudah menjalankan `uvicorn` dengan `$PORT` |
| **Health Check Path** | `/health` |

Masuk ke tab **Variables**, tambahkan:

| Key | Value | Keterangan |
|---|---|---|
| `DATABASE_URL` | (kosongkan dulu) | Kalau kosong, API pakai SQLite lokal. Nanti bisa diisi Postgres. |

> Catatan: `PYTHONPATH` dan `PORT` tidak perlu di-set manual. `PORT` diisi otomatis
> oleh Railway, dan `PYTHONPATH` sudah di-set di dalam Dockerfile.

Lalu masuk ke **Settings → Networking** → klik **Generate Domain**.
Kamu akan dapat URL publik, misalnya:

```
https://tokopedia-api-production.up.railway.app
```

Salin URL ini — nanti dipakai oleh service `ui`.

Tunggu deploy selesai (lihat tab **Deployments**). Kalau status **Success**, tes:

```bash
curl https://<DOMAIN_API_KAMU>/health
```

### 8.3 Konfigurasi service `ui`

Masuk ke service `ui` → tab **Settings**:

| Setting | Nilai |
|---|---|
| **Build / Builder** | Dockerfile (otomatis terdeteksi) |
| **Start Command** | **Override** dengan perintah di bawah |

Start Command untuk `ui`:

```bash
streamlit run src/streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0
```

Masuk ke tab **Variables**, tambahkan:

| Key | Value |
|---|---|
| `API_URL` | `https://<DOMAIN_API_KAMU>` (dari langkah 8.2) |

Lalu **Settings → Networking → Generate Domain** untuk `ui`, misalnya:

```
https://tokopedia-ui-production.up.railway.app
```

Buka URL itu di browser → UI harus bisa memanggil API.

> CORS: API sudah mengizinkan semua origin (`allow_origins=["*"]`), jadi Streamlit
> di domain berbeda tetap bisa memanggil API.

### 8.4 (Opsional) Tambah PostgreSQL

Kalau mau riwayat prediksi tersimpan permanen (tidak hilang saat redeploy):

1. Di project Railway → **+ New → Database → PostgreSQL**.
2. Railway otomatis membuat service database dan menyediakan variabel `DATABASE_URL`.
3. Di service `api` → **Variables** → tambahkan `DATABASE_URL` → pilih **Add Reference**
   → pilih database Postgres → variabel `DATABASE_URL`.

Setelah redeploy, API akan pakai Postgres.

### 8.5 Cara ambil Service ID (dipakai untuk CI/CD)

Service ID ada di URL saat kamu membuka sebuah service di Railway:

```
https://railway.app/project/<PROJECT_ID>/service/<SERVICE_ID>
```

Bagian `<SERVICE_ID>` adalah yang kamu butuhkan untuk secret GitHub
(`RAILWAY_API_SERVICE_ID` dan `RAILWAY_UI_SERVICE_ID`).

---

## 9. CI/CD otomatis dengan GitHub Actions

Project ini punya 2 workflow:

| Workflow | File | Fungsi |
|---|---|---|
| **CI/CD** | `.github/workflows/ci-cd.yml` | Lint + test + build setiap push/PR |
| **Deploy to Railway** | `.github/workflows/deploy-railway.yml` | Deploy `api` & `ui` setiap push ke `main` |

### 9.1 Apa itu GitHub Actions?

GitHub Actions = robot yang menjalankan perintah otomatis setiap ada perubahan kode.
Perintahnya ditulis dalam file `.yml` di folder `.github/workflows/`.

### 9.2 Menyiapkan secret (wajib untuk deploy)

Buka repo GitHub → **Settings → Secrets and variables → Actions → New repository secret**.
Tambahkan 3 secret:

| Nama secret | Isi | Cara dapat |
|---|---|---|
| `RAILWAY_API_TOKEN` | Token Railway | Railway → klik avatar → **Account Settings** → **Tokens** → **Create Token** |
| `RAILWAY_API_SERVICE_ID` | Service ID `api` | Dari URL service `api` (lihat 8.5) |
| `RAILWAY_UI_SERVICE_ID` | Service ID `ui` | Dari URL service `ui` |

> ⚠️ Pakai **account-scoped token**, bukan project token. Project/team token sering
> ditolak (error 401) untuk deploy non-interaktif di CI.

### 9.3 Cara kerja workflow deploy

Isi `deploy-railway.yml` (ringkas):

```yaml
- run: npx --yes @railway/cli up --service=${{ secrets.RAILWAY_API_SERVICE_ID }} --ci
  env:
    RAILWAY_API_TOKEN: ${{ secrets.RAILWAY_API_TOKEN }}
```

Penjelasan:
- `npx @railway/cli` → menjalankan Railway CLI tanpa install permanen.
- `up --service=<ID>` → deploy ke service tertentu.
- `--ci` → mode non-interaktif (wajib di CI, supaya tidak minta input).
- `RAILWAY_API_TOKEN` → token login (account-scoped), tidak perlu `railway link`.

### 9.4 Cara trigger

- **Otomatis**: setiap `git push` ke branch `main`.
- **Manual**: GitHub → tab **Actions** → pilih **Deploy to Railway** → **Run workflow**.

### 9.5 Membaca log deploy

GitHub → tab **Actions** → klik run yang sedang berjalan → klik job `deploy-api` atau
`deploy-ui` → lihat log. Kalau gagal, pesan error biasanya ada di sini.

Di sisi Railway, log build/runtime ada di service → tab **Deployments** → klik deployment
→ **View Logs**.

### 9.6 Alternatif tanpa token: Railway auto-deploy

Kalau tidak mau repot token/CLI:

1. Di Railway → service `api` → **Settings → Source**.
2. Aktifkan **Auto Deploy** untuk branch `main`.
3. Ulangi untuk service `ui`.

Setiap push ke `main` akan otomatis rebuild & redeploy. Kalau pakai cara ini, file
`deploy-railway.yml` bisa dihapus.

---

## 10. Alternatif platform deploy

| Platform | Cocok untuk | Catatan |
|---|---|---|
| **Railway** | API + UI (rekomendasi) | Dockerfile auto-detect, env `PORT`, Postgres plugin |
| **Render** | API + UI | Sudah ada `render.yaml` di repo |
| **Google Cloud Run** | API + UI | Bayar per request; build image → push GCR → deploy 2 service |
| **Fly.io** | API + UI | Butuh `fly.toml` per service |
| **Hugging Face Spaces** | UI saja | Cocok untuk Streamlit; TensorFlow berat, API sebaiknya terpisah |
| **VPS (DigitalOcean/Linode)** | API + UI | Pakai `docker compose up -d` + reverse proxy (Caddy/Nginx) |

### 10.1 Render (sudah ada `render.yaml`)

`render.yaml` sudah mendefinisikan service `tokopedia-api`. Untuk UI, tambahkan service
kedua di file yang sama:

```yaml
  - type: web
    name: tokopedia-ui
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: streamlit run src/streamlit_app/app.py --server.port=$PORT --server.address=0.0.0.0
    envVars:
      - key: API_URL
        value: https://tokopedia-api.onrender.com
```

### 10.2 Artefak di object storage (kalau model besar)

Kalau `models/` terlalu besar untuk Git, simpan di S3/GCS dan download saat container
start. Contoh start command:

```bash
python - <<'PY'
import os, urllib.request, zipfile
url = os.environ["MODELS_URL"]
urllib.request.urlretrieve(url, "/tmp/models.zip")
zipfile.ZipFile("/tmp/models.zip").extractall("/app/models")
PY
```

Lalu set variabel `MODELS_URL` di service `api`.

---

## 11. Validasi & pengujian

Setelah deploy, pastikan semua jalan:

```bash
# 1. API hidup & model ke-load
curl https://<DOMAIN_API>/health
# harus: {"status":"ok","ready":true,...}

# 2. Info model
curl https://<DOMAIN_API>/models

# 3. Prediksi tunggal
curl -X POST https://<DOMAIN_API>/predict \
  -H "Content-Type: application/json" \
  -d '{"review_text": "Barang bagus tapi pengiriman lambat"}'

# 4. Prediksi batch
curl -X POST https://<DOMAIN_API>/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["mantap", "jelek banget", "biasa saja"]}'
```

Di browser:
- Buka `https://<DOMAIN_API>/docs` → coba endpoint lewat Swagger UI.
- Buka `https://<DOMAIN_UI>` → coba tab **Predict** dan **Batch**.

---

## 12. Update model (retrain → redeploy)

Kalau kamu melatih ulang model dan mendapat artefak baru:

```bash
# 1. Ganti artefak lama
rm -rf models/sentiment_model models/category_model
unzip -o tokopedia_artifacts.zip
mv tokopedia_artifacts/sentiment_model models/sentiment_model
mv tokopedia_artifacts/category_model models/category_model

# 2. Commit & push
git add models
git commit -m "chore: retrain model v2"
git push

# 3. Otomatis ter-deploy (kalau CI/CD aktif)
```

Kalau pakai auto-deploy Railway, langkah 3 otomatis. Kalau manual, trigger workflow
atau klik **Redeploy** di Railway.

---

## 13. Monitoring, log, rollback

### 13.1 Log
- **Railway**: service → **Deployments** → pilih deployment → **View Logs** (build & runtime).
- **GitHub Actions**: tab **Actions** → pilih run → lihat log tiap step.

### 13.2 Health check
Railway otomatis memanggil `/health` (kalau di-set). Kalau gagal terus, deployment
dianggap gagal.

### 13.3 Rollback
Kalau deploy baru rusak:
- **Railway**: **Deployments** → pilih deployment lama yang sukses → **Redeploy**.
- **Git**: revert commit, lalu push:

```bash
git revert HEAD
git push
```

---

## 14. Biaya & batasan

- **Railway** punya free tier terbatas (kredit bulanan). Image TensorFlow cukup besar,
  jadi hemat resource:
  - Pakai TextCNN (sudah default), hindari model berat seperti IndoBERT.
  - Matikan service yang tidak dipakai.
- **SQLite** (default) cocok untuk demo. Untuk produksi, pakai Postgres (set `DATABASE_URL`).
- **Cold start**: service gratis bisa "tidur" kalau tidak dipakai; request pertama
  setelah tidur bisa lambat beberapa detik.

---

## 15. Troubleshooting

| Masalah | Penyebab / solusi |
|---|---|
| `/health` return `ready: false` | Artefak model belum ada di `models/`. Pastikan folder `models/sentiment_model` & `models/category_model` ikut ter-build (commit ke Git atau download saat start). |
| `ModelLoadError` saat load `model.keras` | Versi TensorFlow/Keras beda antara training & inference. Notebook sekarang menyimpan format `.keras`; pastikan `tensorflow==2.16.1` + `tf-keras==2.16.0` (sudah di `requirements.txt`). |
| UI error "Cannot connect to API" | `API_URL` belum di-set atau salah. Cek Variables service `ui`; pastikan mengarah ke domain publik service `api` (pakai `https://`). |
| Crash / out of memory | Image TensorFlow besar. Pakai TextCNN (default). Pertimbangkan upgrade plan atau pindah ke Render/Cloud Run dengan memori lebih besar. |
| Streamlit tidak jalan di Railway | Pastikan Start Command `ui` pakai `--server.port=$PORT` dan `--server.address=0.0.0.0`. |
| Data prediksi hilang setelah redeploy | SQLite ephemeral. Pakai Postgres (set `DATABASE_URL`). |
| Deploy workflow gagal 401 Unauthorized | Token bukan account-scoped. Buat token baru di **Account Settings → Tokens** dan update secret `RAILWAY_API_TOKEN`. |
| `railway up` minta `railway link` | Pastikan command memakai `--service=<ID>` dan `--ci`, serta `RAILWAY_API_TOKEN` account-scoped. Tidak perlu `railway link`. |
| Prediksi API jelek padahal model bagus | Preprocessing API berbeda dari notebook. Pastikan `tokopedia_ml.preprocessing.clean_text` sinkron dengan preprocessing di notebook (negation handling, emoji removal, no-stemming untuk DL). |
| CORS error di browser | API sudah `allow_origins=["*"]`. Kalau masih error, cek URL `API_URL` di UI (jangan pakai `http://localhost`). |

---

## 16. Glosarium

| Istilah | Arti |
|---|---|
| **API** | Antarmuka HTTP untuk berkomunikasi dengan model |
| **Artefak** | File hasil training (model + tokenizer + encoder + config) |
| **CI/CD** | Otomatisasi test (CI) dan deploy (CD) |
| **Container / Docker** | Cara membungkus aplikasi beserta semua dependensinya |
| **Deploy** | Menjalankan aplikasi di server online |
| **Endpoint** | URL spesifik di API (misal `/predict`) |
| **Env var / Variable** | Pengaturan yang dibaca aplikasi saat jalan (misal `API_URL`) |
| **Health check** | Panggilan berkala untuk memastikan aplikasi masih hidup |
| **Port** | "Nomor pintu" tempat aplikasi mendengarkan request |
| **Rollback** | Kembali ke versi sebelumnya yang stabil |
| **Secret** | Nilai rahasia (token/password) yang disimpan aman di GitHub/Railway |
| **Service** | Satu unit aplikasi yang berjalan (misal service `api` atau `ui`) |
| **UI** | Tampilan web yang bisa diklik pengguna |

---

## 17. Checklist akhir

Sebelum bilang "selesai deploy", pastikan:

- [ ] Artefak model terbaru ada di `models/sentiment_model` & `models/category_model`.
- [ ] `docker compose up` (atau `run_api.sh` + `run_streamlit.sh`) berhasil di lokal.
- [ ] `/health` lokal return `ready: true`.
- [ ] Kode ter-push ke GitHub branch `main`.
- [ ] Service `api` di Railway jalan & punya domain publik; `/health` OK.
- [ ] Service `ui` di Railway jalan; Start Command override sudah benar.
- [ ] Variable `API_URL` di service `ui` mengarah ke domain `api`.
- [ ] UI bisa melakukan prediksi single & batch.
- [ ] Secret CI/CD (`RAILWAY_API_TOKEN`, `RAILWAY_API_SERVICE_ID`, `RAILWAY_UI_SERVICE_ID`)
      sudah di-set **atau** auto-deploy Railway aktif.
- [ ] Workflow `Deploy to Railway` berhasil (hijau) di tab Actions.

Kalau semua ter-centang, aplikasi kamu sudah online dan bisa diakses publik. 🎉
