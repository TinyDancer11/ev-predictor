# ⚡ EV Adoption Predictor — Streamlit App

Aplikasi prediksi adopsi kendaraan listrik (EV) berbasis survei menggunakan CatBoost.

---

## 📁 Struktur Folder

```
ev_prediction_app/
├── app.py                    # Aplikasi Streamlit utama
├── requirements.txt          # Dependencies Python
├── catboost_ev_model.pkl     # Model CatBoost
├── feature_columns.pkl       # Daftar nama fitur
└── scaler.pkl                # StandardScaler
```

---

## 🚀 Cara Menjalankan (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan aplikasi
```bash
streamlit run app.py
```

Buka browser di: **http://localhost:8501**

---

## ☁️ Deploy ke Streamlit Community Cloud (Gratis)

1. **Push ke GitHub**
   ```bash
   git init
   git add .
   git commit -m "first commit"
   git remote add origin https://github.com/username/ev-predictor.git
   git push -u origin main
   ```

2. **Buka** https://share.streamlit.io

3. **Connect** akun GitHub kamu

4. **New app** → pilih repo → pilih branch `main` → main file: `app.py`

5. **Deploy!** → dapat URL publik gratis seperti:
   `https://username-ev-predictor-app-xxxxx.streamlit.app`

> **Catatan:** File `.pkl` harus ikut di-commit ke GitHub agar bisa dibaca saat deploy.

---

## 🐍 Versi Python yang Direkomendasikan
Python **3.10** atau **3.11** (hindari 3.12 untuk kompatibilitas catboost)
