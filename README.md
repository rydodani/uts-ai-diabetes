# 🩺 Sistem Klasifikasi Diabetes Berbasis Decision Tree
### Project Ujian Tengah Semester (UTS) — Mata Kuliah Kecerdasan Buatan
**Program Studi D3 Manajemen Informatika | PSDKU Kota Kediri — Politeknik Negeri Malang**  
**Tahun Akademik 2026/2027**  
**Dosen Pengampu:** Abidatul Izzah, M.Kom  

---

## 📌 Ringkasan Eksekutif
Aplikasi web ini dibangun menggunakan bahasa pemrograman **Python**, web framework **Flask**, dan algoritma Machine Learning **Decision Tree (C4.5/CART)** dengan kriteria *Entropy*. Sistem ini memprediksi risiko penyakit diabetes melitus secara dini berdasarkan parameter klinis pasien dari dataset **Pima Indians Diabetes**.

Proyek ini dirancang secara komprehensif untuk memenuhi **100% ketentuan Soal UTS Kecerdasan Buatan** topik Klasifikasi Data, membandingkan hasil pengujian secara empiris dengan artikel jurnal ilmiah **JDMIS 2026**, menyediakan antarmuka modern yang responsif, serta siap di-deploy secara online ke platform cloud **Render**.

---

## 🎯 Pemenuhan Seluruh Kriteria Soal UTS (Klasifikasi Data)

| No | Kriteria Wajib pada Soal UTS | Status | Implementasi pada Aplikasi |
|---|---|:---:|---|
| 1 | **Menjelaskan sumber & karakteristik dataset** | ✅ Terpenuhi | Halaman `/dataset` memuat asal usul dataset (UCI & Kaggle, NIDDK Pima Indians), 768 baris, 8 prediktor medis, 1 target label, dan statistik deskriptif lengkap. |
| 2 | **Melakukan preprocessing data** | ✅ Terpenuhi | Halaman `/preprocessing` merinci deteksi anomali nilai 0 biologis pada parameter fisiologis serta penerapan imputasi median data latih. |
| 3 | **Membagi data training dan testing** | ✅ Terpenuhi | Halaman `/preprocessing` menampilkan metode Hold-out 80:20 (614 data latih dan 154 data uji terisolasi). |
| 4 | **Melatih model klasifikasi Decision Tree** | ✅ Terpenuhi | Halaman `/modeling` menampilkan pohon keputusan dengan kriteria *Entropy* dan `max_depth=3`, dilengkapi diagram pohon interaktif & aturan *If-Then*. |
| 5 | **Evaluasi model dengan matriks konfusi (Accuracy, Precision, Recall)** | ✅ Terpenuhi | Halaman `/evaluation` menyajikan Confusion Matrix riil (TN=83, FP=16, FN=20, TP=35), Akurasi 76.62%, Presisi 69%, Recall 64%, dan F1-Score 0.66. |
| 6 | **Melakukan pengujian terhadap data baru** | ✅ Terpenuhi | Halaman `/predict` menyediakan form interaktif 8 variabel klinis, tombol preset kasus cepat, kalkulasi probabilitas, dan jalur penalaran pohon. |
| 7 | **Membandingkan hasil dengan artikel jurnal** | ✅ Terpenuhi | Halaman `/comparison` membedah tabel komparasi side-by-side, persamaan, perbedaan, dan penyebab perbedaan vs paper JDMIS 2026. |
| 8 | **Aplikasi berbasis web Flask & siap deploy online** | ✅ Terpenuhi | Dilengkapi `Procfile`, `requirements.txt`, `render.yaml`, web server `gunicorn`, dan panduan deployment online ke Render. |
| 9 | **Desain modern, responsif & siap presentasi 10 menit** | ✅ Terpenuhi | Halaman `/presentation` menyediakan Pitch Deck 6 slide terstruktur per menit lengkap dengan live timer 10:00 dan catatan pembicara mahasiswa. |

---

## 📖 Jurnal Ilmiah Referensi Utama
* **Judul Artikel:** *Prediksi Diabetes Berbasis Decision Tree Dengan Menggunakan Dataset Pima Indians Diabetes*
* **Penulis:** Yustri Insani, Marcel Filemon Naibaho, Sardo Pardingotan Sipayung
* **Afiliasi:** Universitas Katolik Santo Thomas, Medan, Indonesia
* **Jurnal:** *JDMIS: Journal of Data Mining and Information Systems*, Vol. 4, No. 1, 2026, hlm. 40–45
* **ISSN:** 2986-3473 | **DOI:** [10.54259/jdmis.v4i1.7107](https://doi.org/10.54259/jdmis.v4i1.7107)

---

## 📊 Hasil Evaluasi Nyata (Tanpa Mengarang Angka)

Pengujian dilakukan pada **154 data pengujian** yang belum pernah dilihat oleh model saat pelatihan:

```text
========================================================================
METRIK EVALUASI          JURNAL JDMIS 2026   MODEL BASELINE   MODEL OPTIMASI (IMPUTASI)
========================================================================
True Negative (TN)       83 sampel           83 sampel        85 sampel (+2)
False Positive (FP)      16 sampel           16 sampel        14 sampel (-2)
False Negative (FN)      20 sampel           20 sampel        21 sampel (+1)
True Positive (TP)       35 sampel           35 sampel        34 sampel (-1)
------------------------------------------------------------------------
Akurasi (Accuracy)       76.62%              76.62%           77.27% (+0.65%)
Presisi (Precision)      0.69 (69%)          68.63% (~69%)    70.83% (~71%)
Recall (Sensitivitas)    0.64 (64%)          63.64% (~64%)    61.82% (~62%)
F1-Score                 0.66                0.6604           0.6602
Spesifisitas (Sehat)     -                   83.84%           85.86% (+2.02%)
========================================================================
```

---

## 📁 Struktur Folder Proyek
```text
c:/UTS_AI/
│
├── data/
│   ├── diabetes.csv                    # Dataset Pima Indians Diabetes asli (768 baris)
│   └── pima-indians-diabetes.csv
│
├── referensi/
│   ├── Article+4045-0401.pdf           # Jurnal referensi utama (JDMIS 2026)
│   └── UTS Kecerdasan Buatan 2026.pdf  # Lembar soal acuan UTS
│
├── model/
│   ├── train_model.py                  # Skrip pelatihan model, kalkulasi metrik riil & ekspor visual
│   ├── decision_tree_baseline.joblib   # Model Decision Tree baseline terlatih
│   ├── decision_tree_optimized.joblib  # Model Decision Tree dengan imputasi median
│   ├── median_imputer.joblib           # Transformer imputasi median data latih
│   └── metrics.json                    # Data metrik kalkulasi riil format JSON
│
├── static/
│   ├── css/
│   │   └── style.css                   # Desain modern, glassmorphism, responsive, dark theme
│   ├── js/
│   │   ├── main.js                     # Logika form prediksi interaktif & preset cepat
│   │   └── presentation.js             # Navigasi presentasi 6 slide & timer 10 menit
│   └── img/
│       ├── decision_tree_baseline.png  # Gambar visualisasi diagram pohon baseline (High-Res)
│       ├── decision_tree_baseline.svg  # Format vektor SVG
│       ├── decision_tree_optimized.png # Gambar visualisasi diagram pohon optimasi
│       └── decision_tree_optimized.svg
│
├── templates/
│   ├── base.html                       # Layout induk dengan navbar modern & footer
│   ├── index.html                      # Dashboard utama & checklist pemenuhan UTS
│   ├── dataset.html                    # Eksplorasi karakteristik & statistik 8 fitur
│   ├── preprocessing.html              # Penjelasan penanganan data & hold-out split
│   ├── modeling.html                   # Pemodelan pohon keputusan & aturan If-Then
│   ├── evaluation.html                 # Confusion Matrix interaktif & metrik kalkulasi riil
│   ├── comparison.html                 # Analisis perbandingan berdampingan vs jurnal JDMIS
│   ├── predict.html                    # Form uji coba data pasien baru (Live Prediction)
│   └── presentation.html               # Mode Presentasi UTS 10 Menit (Pitch Deck)
│
├── app.py                              # Server web Flask utama & API endpoint
├── requirements.txt                    # Dependensi pustaka Python
├── Procfile                            # Konfigurasi worker web Render / Heroku
├── runtime.txt                         # Versi Python deployment (3.11.9)
├── render.yaml                         # Blueprint konfigurasi otomatis Render
├── .gitignore                          # Daftar file yang diabaikan Git
└── README.md                           # Dokumentasi lengkap proyek
```

---

## 💻 Cara Menjalankan Aplikasi Secara Lokal (Offline)

### 1. Prasyarat Sistem
* Python 3.10 atau versi lebih baru
* pip (Python package installer)

### 2. Langkah Instalasi & Menjalankan Server
Buka terminal / PowerShell di direktori proyek `c:\UTS_AI`:

```bash
# 1. Masuk ke direktori proyek
cd c:\UTS_AI

# 2. (Opsional) Buat virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Instal dependensi
pip install -r requirements.txt

# 4. Jalankan pelatihan model untuk memastikan artefak terbaru
python model/train_model.py

# 5. Jalankan aplikasi web Flask
python app.py
```

Setelah server aktif, buka peramban web (browser) dan akses alamat:
👉 **`http://localhost:5000`** atau **`http://127.0.0.1:5000`**

---

## 🌐 Panduan Deployment Online ke Platform Cloud Render (Gratis)

Aplikasi ini telah disiapkan dengan standar *production-ready* (`Procfile`, `render.yaml`, `gunicorn`, `runtime.txt`). Berikut langkah mudah mempublikasikan aplikasi agar dapat diakses oleh Dosen dan publik:

### Langkah 1: Push Proyek ke GitHub
1. Buat repositori baru di akun [GitHub](https://github.com) Anda (misalnya bernama `uts-ai-diabetes-classification`).
2. Jalankan perintah git berikut di folder `c:\UTS_AI`:
   ```bash
   git init
   git add .
   git commit -m "Initial commit - UTS AI Decision Tree Diabetes Web App"
   git branch -M main
   git remote add origin https://github.com/<username-anda>/<nama-repo-anda>.git
   git push -u origin main
   ```

### Langkah 2: Deploy di Render (render.com)
1. Buka [https://render.com](https://render.com) dan masuk menggunakan akun GitHub Anda.
2. Di dashboard Render, klik tombol **"New +"** lalu pilih **"Web Service"**.
3. Pilih repositori GitHub proyek yang baru saja Anda buat.
4. Isi konfigurasi layanan:
   * **Name:** `uts-ai-diabetes-prediction` (atau nama pilihan Anda)
   * **Region:** Singapore (Southeast Asia) — paling cepat diakses dari Indonesia
   * **Branch:** `main`
   * **Runtime:** `Python 3`
   * **Build Command:** `pip install -r requirements.txt && python model/train_model.py`
   * **Start Command:** `gunicorn app:app`
   * **Instance Type:** `Free`
5. Klik **"Create Web Service"**.
6. Tunggu proses build selesai sekitar 2-3 menit. Render akan memberikan URL publik gratis seperti:  
   👉 `https://uts-ai-diabetes-prediction.onrender.com`

---

## 🎤 Strategi Presentasi 10 Menit di Depan Dosen Penguji

Buka menu **"⏱️ Presentasi 10 Menit"** di navbar aplikasi web. Gunakan panduan waktu berikut:

1. **Menit 00:00 – 01:30 (Slide 1):** Sapa dosen penguji, sampaikan judul proyek, latar belakang perlunya diagnosis dini diabetes, serta karakteristik dataset Pima Indians (768 data, 8 variabel medis dari UCI).
2. **Menit 01:30 – 03:00 (Slide 2):** Jelaskan preprocessing (anomali nilai 0 fisiologis) dan pembagian data menggunakan Hold-out 80% train (614 data) dan 20% test (154 data) yang terisolasi ketat.
3. **Menit 03:00 – 05:00 (Slide 3):** Tunjukkan arsitektur Decision Tree berbasis Entropy, kenapa `max_depth` dibatasi 3 level (agar tidak overfitting dan mudah dipahami dokter), serta perlihatkan bahwa Glukosa menjadi simpul akar pada nilai $\le 127.5$ mg/dL.
4. **Menit 05:00 – 07:00 (Slide 4):** Paparkan Confusion Matrix riil (TN=83, FP=16, FN=20, TP=35) dan metrik evaluasi nyata: Akurasi 76.62%, Presisi 69%, Recall 64%. Tekankan bahwa hasil ini dihitung secara nyata tanpa rekayasa.
5. **Menit 07:00 – 08:30 (Slide 5):** Jelaskan komparasi dengan jurnal JDMIS (Poin 8 UTS): persamaan arsitektur dan baseline, perbedaan penambahan model optimasi imputasi median yang menaikkan akurasi ke 77.27%, serta alasan medisnya.
6. **Menit 08:30 – 10:00 (Slide 6):** Buka form prediksi, klik salah satu tombol **"Preset Cepat"** untuk demonstrasi langsung di hadapan dosen, tunjukkan jalur penelusuran keputusan pohon, sampaikan kesimpulan, dan tutup presentasi tepat waktu.

---
*Dibuat dengan dedikasi akademik untuk memenuhi Ujian Tengah Semester (UTS) Mata Kuliah Kecerdasan Buatan.*
