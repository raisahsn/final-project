# Tokopedia Review Sentiment & Category Classification

Deep Learning deployment pipeline untuk klasifikasi sentimen dan kategori review produk Tokopedia.

## 📁 Project Structure

```
raisah/
├── notebooks/
│   └── tokopedia_merged.ipynb       # Notebook riset/training
├── src/
│   ├── tokopedia_ml/                # Preprocessing, model builders, inference
│   ├── api/                         # FastAPI REST API
│   └── streamlit_app/               # Streamlit web UI
├── models/                          # Model artifacts (diisi dari Colab)
│   ├── sentiment_model/
│   └── category_model/
├── colab/
│   ├── save_artifacts.py            # Script export model dari Colab
│   └── README.md                    # Panduan export dari Colab
├── tests/                           # Unit & integration tests
├── .github/workflows/ci-cd.yml      # GitHub Actions pipeline
├── Dockerfile
├── docker-compose.yml
├── render.yaml                      # Render deployment blueprint
└── requirements.txt
```

## 🚀 Quick Start

### 1. Export Model Artifacts dari Google Colab

Model telah dilatih di Google Colab. Untuk deployment, export artifact dengan cara:

1. Upload `colab/save_artifacts.py` ke sesi Colab.
2. Jalankan setelah semua training cell selesai:

```python
%run save_artifacts.py
```

3. Download file `tokopedia_artifacts.zip`.
4. Extract ke project root:

```bash
unzip ~/Downloads/tokopedia_artifacts.zip
```

Struktur yang diharapkan:

```
models/
├── sentiment_model/
│   ├── model.keras
│   ├── tokenizer.json
│   ├── label_encoder.pkl
│   └── config.json
└── category_model/
    ├── model.keras
│   ├── tokenizer.json
│   ├── label_encoder.pkl
│   └── config.json
```

### 2. Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Run FastAPI

```bash
PYTHONPATH=src uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Swagger docs: http://localhost:8000/docs

### 4. Run Streamlit

```bash
PYTHONPATH=src streamlit run src/streamlit_app/app.py
```

Buka: http://localhost:8501

## 🐳 Docker

Jalankan API + Streamlit + PostgreSQL sekaligus:

```bash
docker-compose up --build
```

- API: http://localhost:8000
- Streamlit: http://localhost:8501
- PostgreSQL: localhost:5432

## 🧪 Testing

```bash
PYTHONPATH=src pytest tests/ -v
```

## 🔄 CI/CD

GitHub Actions pipeline (`.github/workflows/ci-cd.yml`) menjalankan:

1. Lint dengan Ruff
2. Format check dengan Black
3. Unit & integration tests dengan pytest
4. Docker image build

Aktifkan push ke GHCR dengan meng-uncomment bagian `push` job dan menyediakan `GITHUB_TOKEN`.

## ☁️ Free Deployment untuk Bootcamp

Rekomendasi: **Render** untuk FastAPI + **Streamlit Community Cloud** untuk UI.
Keduanya punya free tier dan cukup untuk tugas bootcamp.

### Persiapan

Pastikan model artifacts sudah ada di folder `models/`. Untuk deploy, file model harus ikut ke GitHub. Kalau model kecil (<100MB total), uncomment atau hapus baris ini di `.gitignore`:

```gitignore
# Hapus/comment baris ini supaya model ikut ke GitHub
models/sentiment_model/*
!models/sentiment_model/.gitkeep
models/category_model/*
!models/category_model/.gitkeep
```

Kalau model >100MB, gunakan **Git LFS** atau upload manual ke platform.

### Deploy FastAPI ke Render

1. Push repo ke GitHub (pastikan model artifacts ikut terpush).
2. Buka [render.com](https://render.com), login dengan GitHub.
3. Klik **New** → **Web Service**.
4. Pilih repository project ini.
5. Render akan otomatis membaca `render.yaml`:
   - **Name**: `tokopedia-api`
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `PYTHONPATH=src uvicorn api.main:app --host 0.0.0.0 --port 8000`
   - **Plan**: Free
6. Klik **Create Web Service**.
7. Tunggu deploy selesai, catat URL-nya (misal `https://tokopedia-api-xxxxx.onrender.com`).

### Deploy Streamlit ke Streamlit Community Cloud

1. Buka [share.streamlit.io](https://share.streamlit.io), login dengan GitHub.
2. Klik **New app**.
3. Pilih repository, branch `main`, file `src/streamlit_app/app.py`.
4. Tambahkan environment variable:
   - `API_URL`: URL Render API kamu (contoh: `https://tokopedia-api-xxxxx.onrender.com`)
5. Klik **Deploy**.
6. Selesai! Streamlit akan berjalan di URL bawaan platform.

### Catatan Penting Free Tier

- **Render free**: service sleep setelah 15 menit idle. Akses pertama bisa lambat (~30 detik) karena cold start.
- **Streamlit Cloud free**: juga sleep saat idle, tapi cukup untuk demo.
- **Database**: pakai SQLite (`predictions.db`) yang tersimpan di disk Render. Untuk production sebenarnya pakai PostgreSQL, tapi untuk tugas SQLite sudah cukup.
- **RAM**: TensorFlow butuh memori. Kalau Render free (512MB) tidak cukup, pertimbangkan Railway atau upgrade ke paid plan.

## 📡 API Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| GET | `/health` | Health check + status model |
| GET | `/models` | Metadata model yang dimuat |
| POST | `/predict` | Prediksi single review |
| POST | `/predict/batch` | Prediksi batch review |
| GET | `/predictions` | Daftar prediksi tersimpan (paginasi) |

## 🗄️ Database

Default menggunakan SQLite (`predictions.db` di root). Untuk PostgreSQL, set environment variable:

```bash
export DATABASE_URL=postgresql://user:password@host:5432/dbname
```

## 🛠️ Model yang Dideploy

- **Sentiment**: BiLSTM Tuned (3 kelas: `negative`, `neutral`, `positive`)
- **Category**: BiLSTM (4 kelas: `produk`, `produk_dan_pengiriman`, `pengiriman`, `umum`)

Kedua model menggunakan preprocessing yang sama dengan notebook riset.

## 🛟 Troubleshooting

### API mengembalikan 500 atau 503 setelah menambahkan model

Ini biasanya terjadi karena model yang disimpan di Colab tidak compatible dengan environment inference. API sekarang sudah memberikan error message yang detail di response dan log.

Cek status model:
```bash
curl http://localhost:8000/health
```

Kalau muncul error seperti:
- `Could not locate class 'Functional'`
- `Could not deserialize class 'Functional'`
- `tf_keras.src.engine.functional`

Artinya model disimpan dengan `tf_keras` di Colab. Solusi:

1. Pastikan `tf-keras==2.16.0` sudah terinstall:
   ```bash
   pip install tf-keras==2.16.0
   ```

2. Atau export ulang dari Colab dengan `colab/save_artifacts.py` yang sudah diupdate.

### Model gagal load karena versi scikit-learn berbeda

Kalau muncul warning/error saat load `label_encoder.pkl`, pastikan versi scikit-learn sama dengan Colab. Saat ini pinned di `requirements.txt`:
- `scikit-learn==1.6.1`

### File model tidak ditemukan

Pastikan struktur folder benar:
```bash
ls models/sentiment_model
ls models/category_model
```

Harus ada `model.keras`, `tokenizer.json`, `label_encoder.pkl`, `config.json`.

### Docker build failed karena memory penuh

TensorFlow sangat besar. Build Docker lokal butuh RAM 6-8GB. Solusi:
1. Naikin memory limit Docker Desktop ke 8GB
2. Atau skip Docker local, langsung deploy ke Render

## 📝 Notes

- File model (`*.keras`) tidak di-commit ke Git secara default (lihat `.gitignore`).
- Gunakan Git LFS jika ingin menyimpan model di repository.
- Prediksi disimpan otomatis ke database saat memanggil `/predict` atau `/predict/batch`.
