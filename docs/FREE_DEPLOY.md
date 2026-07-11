# Deploy Gratis & Cepat (Render + Streamlit Cloud)

Panduan paling **satset** dan **100% gratis** untuk membuat model + API + UI online.
Cocok untuk demo / tugas / portofolio.

```
Browser → Streamlit Cloud (UI, gratis) ──HTTP──► Render (API, gratis) → model (di-commit ke Git)
```

Kenapa kombinasi ini:
- **Render** punya free web service, baca `render.yaml` yang sudah ada di repo.
- **Streamlit Community Cloud** gratis untuk UI.
- **Model TextCNN kecil** (beberapa MB) → aman di-commit ke Git, tidak perlu object storage.

> Kekurangan free tier: service Render akan **tidur** setelah ~15 menit tidak dipakai.
> Request pertama setelah tidur butuh ~30–60 detik (cold start). Cukup untuk demo.

---

## 1. Commit model ke Git

`.gitignore` sudah diubah supaya `models/` boleh di-commit (model kecil).

```bash
# pastikan artefak ada di models/ (lihat docs/DEPLOYMENT.md bagian 6)
ls models/sentiment_model models/category_model

git add .gitignore models/sentiment_model models/category_model
git commit -m "chore: commit model artifacts for free deploy"
git push
```

> Kalau suatu hari model jadi besar (>100 MB), jangan commit — pakai `MODELS_URL`
> (lihat `docs/DEPLOYMENT.md` bagian 10.2).

---

## 2. Deploy API ke Render

1. Buka <https://render.com> → **Sign up / Login** (bisa pakai GitHub).
2. Klik **New → Blueprint** → pilih repo ini.
3. Render membaca `render.yaml` dan membuat service `tokopedia-api` (plan **free**).
4. Klik **Apply** → tunggu build (install `requirements.txt`, termasuk TensorFlow — bisa 5–10 menit pertama kali).
5. Setelah **Live**, salin URL service, misalnya:
   `https://tokopedia-api.onrender.com`

Tes:

```bash
curl https://tokopedia-api.onrender.com/health
# harus: {"status":"ok","ready":true,...}
```

> `render.yaml` sudah pakai `--port $PORT` dan `PYTHONPATH=src`. Database default SQLite
> (ephemeral). Untuk permanen, tambah Render PostgreSQL dan set `DATABASE_URL`.

---

## 3. Deploy UI ke Streamlit Community Cloud

Ikuti `docs/STREAMLIT_CLOUD.md`. Ringkasnya:

1. Buka <https://share.streamlit.io> → **Create app**.
2. Repository: repo ini · Branch: `main` · Main file path: `src/streamlit_app/app.py`.
3. **Advanced settings → Secrets**, isi:

   ```toml
   API_URL = "https://tokopedia-api.onrender.com"
   ```

   (ganti dengan URL Render kamu)

4. **Deploy** → app jadi `https://<nama>.streamlit.app`.

---

## 4. Selesai — akses

- **UI**: `https://<nama>.streamlit.app`
- **API docs**: `https://tokopedia-api.onrender.com/docs`

Buka UI, coba prediksi. Request pertama bisa lambat (cold start Render); setelah itu normal.

---

## 5. (Opsional) Jaga API tetap hangat

Supaya API Render tidak sering tidur, ping `/health` berkala pakai cron gratis
(misalnya <https://cron-job.org> atau GitHub Actions scheduled):

```bash
curl https://tokopedia-api.onrender.com/health
```

Jangan terlalu sering (cukup tiap 10–15 menit) agar tidak melanggar fair use.

---

## 6. Alternatif gratis lain

| Platform | Catatan |
|---|---|
| **Koyeb** (free tier) | Deploy dari Dockerfile; instance nano gratis |
| **Fly.io** | Free allowance, butuh kartu kredit untuk verifikasi |
| **Google Cloud Run** | Free tier besar, tapi wajib billing account |
| **Hugging Face Spaces (Docker)** | Gratis, cocok untuk ML; perlu `README.md` header YAML |
| **Railway** | Paling satset, tapi free tier terbatas / butuh kartu |

Untuk yang paling simpel tanpa kartu kredit: **Render + Streamlit Cloud** (panduan ini).
