# Final Project — Image Retrieval (CBIR) Daun Tanaman

**Content-Based Image Retrieval (CBIR)** menggunakan kombinasi fitur **warna (HSV Histogram)**, **tekstur (LBP)**, dan **bentuk/tepi (HOG)**, dengan tiga skenario segmentasi (tanpa segmentasi, Otsu Thresholding, K-Means) dan empat metrik jarak (Euclidean, Manhattan, Canberra, Cosine).

| Item | Keterangan |
|---|---|
| Nama | Achmal Farhan Ashidiqy |
| Temu Kembali Citra (Image Retrieval) |
| Dataset | [Plant Leaf Image Dataset (8 Classes)](https://www.kaggle.com/datasets/mahmoudshaheen1134/plant-leaf-image-dataset) |


---

## 1. Ringkasan Singkat (versi non-teknis)

Sistem ini bekerja seperti "pencarian Google Images", tetapi **murni berbasis isi gambar**, bukan nama file atau tag.

1. Sistem "belajar" dari **2.868 foto daun** (galeri/reference).
2. Ketika diberi **1 foto daun baru** (query), sistem mengubahnya menjadi deretan angka yang mendeskripsikan **warna, tekstur, dan pola tepi daun**.
3. Angka query dibandingkan dengan seluruh angka di galeri, lalu diurutkan dari yang paling mirip.
4. Diambil **Top-5 / Top-10 / Top-20** hasil terbaik, lalu diukur seberapa sering hasil itu benar kelasnya.

Hasil utama: dengan **jarak Manhattan**, hasil yang benar rata-rata muncul di **peringkat teratas (MRR 0,9266)** dan **Recall@10 = 0,9611** pada 720 gambar uji.

---

## 2. Tujuan Proyek

- Membangun pipeline CBIR lengkap: **preprocessing → segmentasi → ekstraksi fitur → similarity search → evaluasi**.
- Membandingkan **3 metode segmentasi** dan **4 metrik jarak** secara kuantitatif.
- Menentukan konfigurasi terbaik berdasarkan metrik evaluasi retrieval (bukan hanya visual).

---

## 3. Dataset

**Sumber:** Plant Leaf Image Dataset (Kaggle) — file `plant-leaf-image.zip` di Google Drive, diekstrak ke `/content/plant_leaf_dataset2`.

**Total: 3.588 gambar, 8 kelas** (melewati batas minimum 300 gambar yang ditetapkan di notebook).

| Kelas | Jumlah Gambar |
|---|---:|
| Apple | 519 |
| Berry | 340 |
| Fig | 508 |
| Guava | 520 |
| Orange | 547 |
| Palm | 468 |
| Persimmon | 527 |
| Tomato | 159 |
| **Total** | **3.588** |

Catatan: kelas **Tomato** adalah kelas minoritas (159 gambar), sehingga komposisi data tidak seimbang (imbalanced).

---

## 4. Pembagian Data (Stratified Split 80:20)

Split dilakukan **per kelas** agar proporsi tiap kelas tetap terjaga di galeri dan query, dengan `SEED = 42` (hasil dapat direproduksi). Setiap kelas dijamin memiliki minimal 1 gambar query.

| Kelas | Gallery | Query |
|---|---:|---:|
| Apple | 415 | 104 |
| Berry | 272 | 68 |
| Fig | 406 | 102 |
| Guava | 416 | 104 |
| Orange | 437 | 110 |
| Palm | 374 | 94 |
| Persimmon | 421 | 106 |
| Tomato | 127 | 32 |
| **Total** | **2.868 (79,93%)** | **720 (20,07%)** |

Validasi: jumlah kelas di gallery dan query sama-sama 8, dan himpunan kelasnya identik (diperiksa dengan `assert`). Hasil split disimpan ke `gallery_set.csv` dan `query_set.csv`.

---

## 5. Pipeline Sistem

```
Foto query
   │
   ├─ Preprocessing : BGR→RGB, resize aspect-ratio + padding 224×224, CLAHE (LAB)
   ├─ Segmentasi    : none / Otsu / K-Means (HSV, k=2)
   ├─ Fitur Warna   : HSV 3D histogram 8×8×8 → 512 dimensi (ternormalisasi)
   ├─ Fitur Tekstur : LBP uniform (radius 2, 16 titik) → 18 dimensi
   ├─ Fitur Bentuk  : HOG (9 orientasi, cell 8×8, block 2×2, L2-Hys) → 26.244 dimensi
   ├─ Kombinasi     : concat [w·color, w·lbp, w·hog] → 26.774 dimensi, L2-normalized
   ├─ Similarity    : Euclidean / Manhattan / Canberra / Cosine (brute-force linear scan)
   └─ Output        : Top-K (5, 10, 20) + metrik Precision, Recall, mAP, MRR
```

### 5.1 Preprocessing
- **Resize with padding**: skala mengikuti sisi terpanjang, sisa area diisi hitam (0,0,0) → tidak ada distorsi rasio daun.
- **CLAHE** (Contrast Limited Adaptive Histogram Equalization): diterapkan pada kanal **L** ruang warna **LAB**, `clipLimit=2.0`, `tileGridSize=(8,8)`, lalu dikembalikan ke RGB. Tujuannya menstabilkan pencahayaan antar foto.
- Turunan citra: **HSV** (untuk histogram & segmentasi K-Means) dan **grayscale** (untuk Otsu, LBP, HOG).

### 5.2 Segmentasi (3 skenario)
| Metode | Cara kerja |
|---|---|
| No Segmentation | Seluruh citra 224×224 dipakai sebagai objek |
| Otsu Thresholding | Gaussian blur 5×5 → threshold Otsu biner → mask 0/1 |
| K-Means Clustering | Piksel HSV di-cluster `k=2`; cluster terbesar dianggap **background**, cluster lainnya menjadi objek |

Mask diterapkan ke citra RGB (piksel background menjadi hitam) sebelum ekstraksi fitur.

### 5.3 Ekstraksi Fitur
| Descriptor | Konfigurasi | Dimensi |
|---|---|---:|
| Color Histogram | HSV, bins 8×8×8, dinormalisasi | 512 |
| LBP | uniform, radius 2, 16 titik, histogram dinormalisasi | 18 |
| HOG | 9 orientasi, pixel/cell 8×8, cell/block 2×2, L2-Hys | 26.244 |
| **Kombinasi** | concat berbobot + normalisasi L2 | **26.774** |

Bobot fitur bersifat konfigurabel (`weights=(w_color, w_lbp, w_hog)`); pada eksekusi digunakan bobot default 1.0 untuk ketiganya.

### 5.4 Similarity Search
Empat metrik jarak diuji: **Euclidean**, **Manhattan (L1)**, **Canberra**, dan **Cosine**. Pencarian bersifat *exhaustive* (setiap query dibandingkan dengan seluruh 2.868 vektor galeri), lalu diurutkan ascending dan diambil Top-K.

---

## 6. Hasil Eksperimen

### 6.1 Pengaruh Segmentasi (Recall@10, jarak Manhattan)

Eksperimen ini memakai pipeline fitur versi pertama (HOG 128×128 dengan cell 16×16, kombinasi fitur tanpa normalisasi L2).

| Metode Segmentasi | Recall@10 |
|---|---:|
| No Segmentation | **0,9972** |
| Otsu Thresholding | 0,9903 |
| K-Means Clustering | 0,9736 |

### 6.2 Pengaruh Metrik Jarak (pipeline final: K-Means + HSV + L2-normalized, 26.774 dimensi)

**Recall@K** (query dianggap benar jika label yang sama muncul di Top-K):

| Distance | Recall@5 | Recall@10 | Recall@20 |
|---|---:|---:|---:|
| Euclidean | 0,8042 | 0,8583 | 0,9431 |
| **Manhattan** | **0,9458** | **0,9611** | 0,9708 |
| Canberra | 0,9444 | 0,9625 | **0,9764** |
| Cosine | 0,8042 | 0,8583 | 0,9431 |

**Evaluasi lengkap** (720 query; mAP & MRR dihitung dari Top-20 sehingga nilainya sama untuk setiap K):

| Distance | K | Precision@K | Recall@K | mAP | MRR |
|---|---:|---:|---:|---:|---:|
| Euclidean | 5 | 0,4694 | 0,0059 | 0,5842 | 0,7681 |
| Euclidean | 10 | 0,3733 | 0,0093 | 0,5842 | 0,7681 |
| Euclidean | 20 | 0,3174 | 0,0158 | 0,5842 | 0,7681 |
| **Manhattan** | 5 | **0,7675** | 0,0098 | **0,8036** | **0,9266** |
| **Manhattan** | 10 | **0,6488** | 0,0165 | **0,8036** | **0,9266** |
| **Manhattan** | 20 | 0,5313 | 0,0268 | **0,8036** | **0,9266** |
| Canberra | 5 | 0,7550 | 0,0097 | 0,7859 | 0,9234 |
| Canberra | 10 | 0,6444 | 0,0164 | 0,7859 | 0,9234 |
| Canberra | 20 | 0,5333 | 0,0271 | 0,7859 | 0,9234 |
| Cosine | 5 | 0,4694 | 0,0059 | 0,5842 | 0,7681 |
| Cosine | 10 | 0,3733 | 0,0093 | 0,5842 | 0,7681 |
| Cosine | 20 | 0,3174 | 0,0158 | 0,5842 | 0,7681 |

### 6.3 Kesimpulan Terukur

**Metrik jarak terbaik: Manhattan** — mAP **0,8036**, MRR **0,9266** (disimpan otomatis oleh notebook sebagai `best_distance`).

- Manhattan mengungguli Euclidean/Cosine sebesar **+0,2194 mAP** (0,8036 vs 0,5842).
- Precision@10 Manhattan **0,6488**, artinya rata-rata **6–7 dari 10** hasil teratas memang kelas yang benar.
- MRR 0,9266 (maksimum 1,0) berarti hasil yang benar rata-rata berada sangat dekat dengan **peringkat teratas**.
- **Euclidean dan Cosine menghasilkan angka identik**. Ini konsekuensi matematis dari fitur yang sudah dinormalisasi L2: untuk vektor bernorma 1, `1 − cos` berbanding lurus dengan `‖a − b‖²`, sehingga urutan peringkatnya sama persis.
- **Canberra** sedikit lebih baik pada Recall@20 (0,9764 vs 0,9708), tetapi kalah pada mAP, MRR, dan Precision@K.

---

## 7. Catatan Penting & Keterbatasan (transparansi hasil)

Bagian ini penting agar pembaca tidak salah menafsirkan angka:

1. **Dua pipeline fitur berbeda di dalam notebook.** Eksperimen segmentasi (§6.1) memakai HOG 1.764 dimensi tanpa normalisasi L2, sedangkan pipeline final (§6.2) memakai HOG 26.244 dimensi dengan normalisasi L2. Karena itu angkanya **tidak apple-to-apple**: Recall@10 *No Segmentation* (0,9972) justru lebih tinggi daripada pipeline final K-Means (0,9611). Notebook melanjutkan dengan konfigurasi **K-Means + HSV + L2** sebagai pipeline final, bukan karena Recall@10-nya lebih tinggi.
2. **Recall@K pada tabel evaluasi lengkap terlihat sangat kecil** (0,0059–0,0271) karena `total_relevant` didefinisikan sebagai **jumlah seluruh gambar kelas tersebut di galeri** (ratusan gambar), padahal hanya Top-K yang diambil. Untuk menilai kualitas peringkat, **Precision@K, mAP, dan MRR lebih informatif**.
3. **mAP dan MRR dihitung sekali dari Top-20** untuk setiap metrik jarak, sehingga nilainya konstan di tiap K. Kolom K hanya memengaruhi Precision@K dan Recall@K.
4. **Label tunggal (single-label).** Setiap gambar hanya punya satu kelas; kemiripan antar kelas yang secara visual serupa tidak dinilai.
5. **Tidak ada pengecekan duplikat/near-duplicate** antara galeri dan query. Jika ada gambar yang sama persis di kedua set, skor bisa optimistic.
6. **Dataset imbalanced** (Tomato 159 vs Orange 547); metrik dilaporkan sebagai rata-rata global, bukan rata-rata per kelas.
7. **Pencarian brute-force**, belum menggunakan indexing Approximate Nearest Neighbor (ANN). Waktu ekstraksi fitur galeri ±13 menit dan query ±3 menit di Colab; total eksperimen segmentasi K-Means memakan ±11 menit untuk galeri saja.

---

## 8. Struktur Notebook

| Bagian | Isi |
|---|---|
| 0 | Identitas, kelas, dataset |
| 1 | Mount Google Drive |
| 2 | Import library & set random seed (42) |
| 3 | Path dataset, ekstraksi ZIP, auto-detect root, load 3.588 gambar |
| 4 | Stratified split 80:20 + validasi + simpan CSV |
| 5 | Preprocessing (resize+padding, CLAHE) & uji visual RGB/HSV/Grayscale |
| 6 | Fungsi ekstraksi fitur & 3 metode segmentasi + visualisasi perbandingan |
| 7 | Eksperimen Recall@10 per metode segmentasi + cache fitur |
| 8 | Ekstraksi fitur final (K-Means + HSV + L2) untuk galeri & query |
| 9 | Similarity search 4 metrik × 3 nilai K |
| 10 | Evaluasi Precision/Recall/mAP/MRR + simpan hasil |
| 11 | Analisis metrik jarak terbaik + grafik mAP |
| 12 | Visualisasi Top-10 hasil retrieval untuk 3 contoh query |

---

## 9. Output yang Dihasilkan Notebook

| Path | Isi |
|---|---|
| `/content/splits/gallery_set.csv` | Daftar path + label galeri (2.868 baris) |
| `/content/splits/query_set.csv` | Daftar path + label query (720 baris) |
| `features_segmentation/gallery_feats_*.npy` | Fitur galeri per metode segmentasi (cache) |
| `cbir_cache_35/gallery_features.npy` | Fitur galeri final (2868 × 26774) |
| `cbir_cache_35/query_features.npy` | Fitur query final (720 × 26774) |
| `cbir_cache_35/retrieval_results_35.csv` | Recall@K per metrik jarak |
| `cbir_cache_36/evaluation_metrics.csv` / `.pkl` | Tabel evaluasi lengkap |
| `*.png` (figure inline) | Visualisasi preprocessing, segmentasi, grafik mAP, Top-10 retrieval |

---

## 10. Cara Menjalankan

1. Buka notebook di **Google Colab**.
2. Unggah `plant-leaf-image.zip` ke Google Drive, lalu sesuaikan:
   ```python
   ZIP_PATH = "/content/drive/MyDrive/SMT 7/TKC/plant-leaf-image.zip"
   ```
3. Jalankan **Runtime → Run all**. Notebook bersifat *safe re-run*: ekstraksi dataset dan ekstraksi fitur dilewati jika file cache sudah ada.
4. Dependensi utama:
   ```bash
   pip install opencv-python scikit-image scikit-learn pandas numpy matplotlib tqdm
   ```
5. Path `EXTRACT_PATH`, `SPLIT_DIR`, dan folder cache dapat diubah bila tidak memakai Colab.

---

## 11. Teknologi yang Digunakan

`Python` · `OpenCV` · `scikit-image` (LBP, HOG) · `scikit-learn` (KMeans, normalisasi L2) · `NumPy` · `pandas` · `Matplotlib` · `tqdm` · `Google Colab` · `Google Drive`

---

## 12. Ringkasan Kompetensi yang Ditunjukkan

- Perancangan **pipeline CBIR end-to-end** (data ingestion → evaluasi → visualisasi hasil).
- **Image preprocessing**: aspect-ratio preserving resize, padding, CLAHE pada LAB.
- **Image segmentation**: Otsu thresholding dan K-Means clustering pada ruang HSV.
- **Feature engineering**: color histogram, LBP, HOG, pembobotan & normalisasi vektor.
- **Information retrieval evaluation**: Precision@K, Recall@K, mAP, MRR, analisis perbandingan metrik jarak.
- **Engineering hygiene**: random seed tetap, cache fitur `.npy`, validasi split dengan `assert`, pemisahan tahap offline (galeri) dan online (query).

---

## 13. Tutorial Push ke GitHub

### 13.1 Persiapan
Letakkan file berikut dalam satu folder project:
```
final-project-cbir/
├── finallllll-IR_FinalProject_Achmal_Farhan_Ashidiqy.ipynb
├── README.md
└── .gitignore
```

> Ukuran notebook ±4,5 MB (mayoritas berisi gambar output). Ini masih di bawah batas GitHub 100 MB, jadi **tidak perlu Git LFS**.

### 13.2 Inisialisasi repository lokal
```bash
cd final-project-cbir

git init
git config user.name  "Nama Kamu"
git config user.email "email@kamu.com"
```

### 13.3 Buat `.gitignore`
```bash
cat > .gitignore <<'EOF'
# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# Data & cache (jangan di-push, ukurannya besar)
*.zip
plant_leaf_dataset*/
features_segmentation/
cbir_cache_*/
splits/
*.npy

# OS
.DS_Store
Thumbs.db
EOF
```

### 13.4 Commit pertama
```bash
git add .
git status              # pastikan hanya notebook + README + .gitignore
git commit -m "Final Project CBIR: plant leaf image retrieval (8 classes)"
git branch -M main
```

### 13.5 Buat repository di GitHub
1. Buka https://github.com/new
2. **Repository name**: `final-project-cbir`
3. Pilih **Public** (agar bisa dilihat HRD/dosen)
4. **Jangan** centang "Add a README file" (sudah ada README lokal)
5. Klik **Create repository**

### 13.6 Hubungkan dan push
```bash
git remote add origin https://github.com/USERNAME/final-project-cbir.git
git push -u origin main
```

### 13.7 Autentikasi (wajib sejak GitHub menghapus password biasa)
**Opsi A — Personal Access Token (paling mudah)**
1. GitHub → **Settings → Developer settings → Personal access tokens → Tokens (classic) → Generate new token**
2. Centang scope **`repo`**, set masa berlaku, lalu **Generate**
3. Saat `git push` meminta password, tempel **token** tersebut (bukan password akun)

Simpan agar tidak berulang:
```bash
git config --global credential.helper store
```

**Opsi B — SSH**
```bash
ssh-keygen -t ed25519 -C "email@kamu.com"
cat ~/.ssh/id_ed25519.pub      # copy ke GitHub → Settings → SSH and GPG keys → New SSH key
git remote set-url origin git@github.com:USERNAME/final-project-cbir.git
git push -u origin main
```

**Opsi C — GitHub CLI**
```bash
gh auth login
gh repo create final-project-cbir --public --source=. --remote=origin --push
```

### 13.8 Update berikutnya
```bash
git status
git add .
git commit -m "Update: perbaikan analisis metrik"
git push
```

### 13.9 Troubleshooting

| Masalah | Solusi |
|---|---|
| `remote origin already exists` | `git remote set-url origin <URL_BARU>` |
| `Updates were rejected (non-fast-forward)` | `git pull --rebase origin main` lalu `git push` |
| `Authentication failed` | Gunakan token, bukan password; cek token belum kedaluwarsa |
| `File too large (>100 MB)` | Jangan commit data/cache; jika perlu: `git lfs track "*.zip"` |
| `src refspec main does not match any` | Belum ada commit: `git add . && git commit -m "init"` |
| Ingin notebook lebih ringan | Colab: **Edit → Clear all outputs**, lalu commit ulang |

### 13.10 Opsional — tombol "Open in Colab"
Tambahkan di bagian atas README:
```markdown
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/USERNAME/final-project-cbir/blob/main/finallllll-IR_FinalProject_Achmal_Farhan_Ashidiqy.ipynb)
```

---

*Seluruh angka pada README ini diambil langsung dari output eksekusi notebook (`finallllll-IR_FinalProject_Achmal_Farhan_Ashidiqy.ipynb`), bukan estimasi.*
