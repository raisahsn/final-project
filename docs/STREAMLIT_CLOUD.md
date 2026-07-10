# Deploy UI ke Streamlit Community Cloud

Panduan ini khusus untuk men-deploy **UI (Streamlit)** ke
**Streamlit Community Cloud** (gratis). API (FastAPI) tetap di-deploy di tempat lain
(misalnya Railway) karena butuh TensorFlow & model artifacts.

> Streamlit Community Cloud hanya menjalankan aplikasi Streamlit. Ia **tidak** menjalankan
> FastAPI. Jadi arsitekturnya:
>
> ```
> Browser → Streamlit Cloud (UI) ──HTTP──► API publik (Railway/Render/dll)
> ```

---

## 1. Prasyarat

- Repo ini sudah di-push ke **GitHub** (public atau private yang kamu punya akses).
- **API sudah online** dan punya URL publik, misalnya:
  `https://tokopedia-api-production.up.railway.app`
  (lihat `docs/DEPLOYMENT.md` bagian 8 untuk deploy API ke Railway).
- Akun Streamlit Community Cloud: <https://share.streamlit.io> (login pakai GitHub).

---

## 2. Kenapa ada `requirements.txt` terpisah untuk UI

UI cuma butuh paket ringan (`streamlit`, `pandas`, `plotly`, `requests`, `openpyxl`),
**tidak butuh TensorFlow**. Karena itu dibuat file khusus:

```
src/streamlit_app/requirements.txt
```

Streamlit Community Cloud akan otomatis memakai `requirements.txt` yang berada **di folder
yang sama dengan entrypoint** (`app.py`). Jadi build UI jadi cepat & ringan, sementara
API tetap memakai `requirements.txt` di root (lengkap, untuk Docker/Railway).

---

## 3. Cara kerja `API_URL` di UI

`src/streamlit_app/app.py` membaca URL API dengan prioritas:

1. **Streamlit secrets** → `st.secrets["API_URL"]` (dipakai di Community Cloud).
2. **Environment variable** → `API_URL` (dipakai di Docker/Railway/lokal).
3. Default → `http://localhost:8000`.

Jadi di Streamlit Cloud kamu cukup mengisi **Secrets** bernama `API_URL`.

---

## 4. Langkah deploy (klik demi klik)

1. Pastikan perubahan sudah ter-push ke GitHub:

   ```bash
   git add .
   git commit -m "feat: siapkan UI untuk Streamlit Cloud"
   git push
   ```

2. Buka <https://share.streamlit.io> → **Sign in** pakai GitHub.

3. Klik **Create app** (atau **New app**).

4. Isi form:

   | Field | Isi |
   |---|---|
   | **Repository** | `<username>/raisah` (atau nama repo kamu) |
   | **Branch** | `main` |
   | **Main file path** | `src/streamlit_app/app.py` |
   | **App URL (optional)** | misal `tokopedia-review-ui` |

5. Klik **Advanced settings**:

   - **Python version**: pilih **3.11** (atau 3.10). Catatan: versi Python tidak bisa
     diubah setelah deploy — kalau mau ganti harus hapus app & deploy ulang.
   - **Secrets**: tempel konfigurasi berikut (format TOML):

     ```toml
     API_URL = "https://tokopedia-api-production.up.railway.app"
     ```

     Ganti URL-nya dengan domain API kamu yang sebenarnya.

6. Klik **Deploy**.

Tunggu proses build (biasanya 1–3 menit). Kalau sukses, app akan terbuka di URL seperti:

```
https://tokopedia-review-ui.streamlit.app
```

---

## 5. Menjalankan lokal dengan secrets (opsional)

Kalau mau meniru perilaku Streamlit Cloud di laptop:

```bash
# 1. Salin contoh secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 2. Edit isinya, arahkan API_URL ke API lokal atau publik
#    API_URL = "http://localhost:8000"

# 3. Jalankan UI
bash run_streamlit.sh
```

> `.streamlit/secrets.toml` sudah ada di `.gitignore`, jadi tidak akan ter-commit.

---

## 6. Update / redeploy

Streamlit Community Cloud **otomatis redeploy** setiap kali kamu push ke branch yang
terhubung (`main`). Jadi untuk update UI:

```bash
git add .
git commit -m "update UI"
git push
```

Tunggu ±1 menit, app akan rebuild sendiri.

Kalau mengubah **Secrets** atau **Python version**:
- Secrets: buka app → **Settings → Secrets** → edit → **Save** (app akan restart).
- Python version: tidak bisa diubah. Hapus app lalu deploy ulang.

---

## 7. Troubleshooting

| Masalah | Penyebab / solusi |
|---|---|
| Build gagal karena paket berat (TensorFlow) | Streamlit Cloud memakai `requirements.txt` di root, bukan yang di `src/streamlit_app/`. Pastikan **Main file path** = `src/streamlit_app/app.py` supaya file `requirements.txt` ringan di folder itu yang dipakai. |
| UI error "Cannot connect to API" | `API_URL` di Secrets belum benar. Buka **Settings → Secrets**, pastikan `API_URL` mengarah ke domain API publik (pakai `https://`, tanpa trailing slash). |
| `ModuleNotFoundError: plotly` / `openpyxl` | Pastikan `src/streamlit_app/requirements.txt` berisi paket tersebut dan sudah ter-push. |
| App stuck "Your app is in the oven" | Tunggu beberapa menit. Kalau >10 menit, cek log build (menu **⋮ → Logs**). |
| CORS error | API sudah `allow_origins=["*"]`. Kalau masih error, cek `API_URL` (jangan `localhost`) dan pastikan API bisa diakses publik (`curl <API_URL>/health`). |
| Perubahan kode tidak muncul | Push ke branch `main`. Kalau sudah push tapi belum berubah, buka app → **⋮ → Reboot**. |

---

## 8. Checklist

- [ ] API sudah online & `/health` return `ready: true`.
- [ ] Repo ter-push ke GitHub.
- [ ] `src/streamlit_app/requirements.txt` ada (ringan, tanpa TensorFlow).
- [ ] Main file path di Streamlit Cloud = `src/streamlit_app/app.py`.
- [ ] Secrets `API_URL` sudah diisi dengan domain API publik.
- [ ] App bisa dibuka & melakukan prediksi single/batch.

---

## 9. Alternatif kalau Streamlit Cloud tidak cukup

Streamlit Community Cloud gratis tapi punya batas resource. Kalau UI butuh lebih:
- Deploy UI sebagai **service terpisah di Railway/Render** (pakai Dockerfile, lihat
  `docs/DEPLOYMENT.md` bagian 8.3).
- Atau deploy UI + API dalam satu container (docker-compose) di VPS.
