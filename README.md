# NLP Airbnb App Reviews

Analisis review aplikasi Airbnb (12.906 review mentah): klasifikasi sentimen biner dan topic modeling keluhan.

## Hasil Ringkas

| Item | Nilai |
|---|---|
| Data mentah | 12.906 review |
| Data final (unik, binary) | 9.435 baris (71,6% positif / 28,4% negatif) |
| Model | TF-IDF (word 1-2 gram + char_wb 2-5 gram) + Logistic Regression, `C=5`, `class_weight="balanced"` |
| CV macro-F1 (5-fold x 3 repeat) | 0.9339 |
| Hold-out macro-F1 | 0.9308 |
| Hold-out akurasi | 94,0% (baseline tebak-positif: 71,6%) |
| Recall negatif / positif | 0.93 / 0.95 |

Confusion matrix hold-out (1.887 baris):

|  | Pred negatif | Pred positif |
|---|---|---|
| **Aktual negatif** | 497 | 38 |
| **Aktual positif** | 70 | 1.282 |

### Topik keluhan (NMF, K=10, review negatif >= 4 kata, n=2.474)

Porsi = topik dominan per review (argmax).

| Topik | Porsi |
|---|---|
| Verifikasi (ID, nomor HP/OTP, "verification does not work") | 18,7% |
| Layanan pelanggan / support | 17,3% |
| Pembayaran | 8,5% |
| Pembatalan & refund | 8,1% |
| Crash setelah update | 6,5% |
| Booking/search (campuran) | 21,7% |
| Host/review (campuran) | 19,2% |

Dua topik campuran **bukan temuan**, hanya sisa yang tidak masuk topik lain.

Temuan temporal: lonjakan review crash pada 17-18 Des 2025 (55 dari 71 review dalam dua hari). Ini indikasi insiden rilis dari pola data, belum diverifikasi dari sumber luar. Sebagian review di topik ini salah masuk (kata "update" muncul di konteks kebijakan), jadi 71 kemungkinan overcount.

## Struktur Project

```
NLP AIR BNB/
├── data/
│   ├── raw/                  airbnb_app_reviews_raw.csv (jangan diedit)
│   └── processed/            reviews_clean.csv, negative_topics.csv
├── notebooks/                01_eda.ipynb
├── src/
│   ├── preprocess.py         cleaning, filter bahasa, dedup, label
│   ├── train.py              model final + evaluasi hold-out
│   ├── tune.py               perbandingan fitur dan C (repeated CV)
│   ├── evaluate.py           error analysis, fitur terkuat, CV
│   └── topics.py             NMF topic modeling
├── models/                   model hasil training (.joblib)
├── reports/figures/
├── requirements.txt
└── README.md
```

## Setup

Python 3.13, Windows PowerShell.

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -c "import nltk; [nltk.download(x) for x in ['stopwords','punkt','punkt_tab','wordnet']]"
```

Taruh `airbnb_app_reviews_raw.csv` di `data\raw\`.

## Cara Menjalankan

Semua perintah dari root project, dengan urutan ini:

```powershell
python src\preprocess.py      # bikin data\processed\reviews_clean.csv
python src\train.py           # train + hold-out, simpan models\tfidf_word_char_lr.joblib
python src\tune.py            # opsional: ulang perbandingan konfigurasi
python src\evaluate.py        # opsional: error analysis dan CV
python src\topics.py          # eksplorasi K = 6, 8, 10, 12
python src\topics.py 10       # fit final K=10, tulis data\processed\negative_topics.csv
```

Jangan skip `preprocess.py`: `train.py` dan `topics.py` membaca `reviews_clean.csv`.

## Metodologi

**Label.** Target = rating bintang: 1-2 negatif, 4-5 positif, rating 3 dibuang (hanya 357 baris, terlalu sedikit untuk kelas sendiri). Kolom `sentiment_category` di data mentah tidak dipakai karena hanya cocok 78,7% dengan rating (contoh: 486 review rating 1 dilabel Positive).

**Cleaning** (`preprocess.py`):
- lowercase, buang URL, emoji, dan karakter non-ASCII
- kontraksi diubah ke `not` (`don't` -> `do not`) agar negasi tidak hilang; stopword **tidak** dibuang
- huruf berulang dipadatkan (`gooood` -> `good`)
- filter bahasa konservatif: buang jika >= 6 kata, rasio stopword Inggris <= 15%, dan `langdetect` bukan Inggris dengan prob > 0,99 (107 baris)
- dedup exact-match; teks sama dengan label beda diambil mayoritas (12 teks bentrok), tie dibuang

**Alur pengurangan data:** 12.906 -> 12.823 (teks kosong setelah cleaning) -> 12.713 (non-Inggris) -> 12.359 (rating 3) -> 9.436 (dedup) -> 9.435 (tie).

**Model.** Pilihan dibuat lewat repeated stratified CV (5-fold x 3). `word+char, C=5` (0.9339) vs baseline `word, C=2` (0.9218). Evaluasi akhir pada split 80/20 stratified, `random_state=42`. Metrik utama macro-F1, bukan akurasi.

**Topic modeling.** NMF + TF-IDF pada review negatif >= 4 kata. Negasi dan kata intensitas sentimen (`bad`, `worst`, `terrible`, dst.) dibuang dari vocabulary karena seluruh korpus negatif dan kata-kata itu hanya membentuk topik sentimen generik. K=10 dipilih secara manual dari perbandingan K=6, 8, 10, 12. NMF tidak punya metrik benar-salah.

## Keterbatasan

1. **Hold-out tidak sepenuhnya bersih.** `C=5` dipilih lewat CV di seluruh data, termasuk split hold-out. Efeknya kecil (hanya 10 konfigurasi) tapi ada.
2. **Near-duplicate belum dibuang** ("very good" vs "very good app"). Skor adalah batas atas yang agak optimis.
3. **Hanya biner.** Model tidak memprediksi netral.
4. **Label = rating, bukan sentimen teks.** Sebagian error adalah noise rating (komplain berbintang 5), bukan salah model.
5. **Kelemahan bag-of-words yang tersisa:** negasi dan idiom (`never a bad experience`, `no good app`), kontras temporal (`it used to be good but now it's bad`), dan kata asing out-of-vocabulary (jatuh ke bias intersep).
6. **Filter bahasa hanya menjaring teks >= 6 kata.** Review non-Inggris pendek (`bakwas`, `penipuan`) tetap lolos.
7. **Topik:** dasar hanya 2.474 review; persentase dihitung dari topik dominan; dua topik campuran tidak informatif; segmen sudut pandang host terlalu kecil (regex menangkap 83 review, presisi sekitar 70%) untuk dianalisis kuantitatif.
8. **Tren bulanan:** hanya 148-255 review negatif per bulan, per topik sering di bawah 15. Hanya lonjakan Desember 2025 yang kuat secara statistik. Bulan Oktober 2026 belum penuh.

## Langkah Lanjutan

- Dedup near-duplicate sebelum split
- Transformer multilingual (mis. `cardiffnlp/twitter-xlm-roberta-base-sentiment`) untuk negasi dan bahasa asing
- Threshold tuning untuk mengatur trade-off recall negatif