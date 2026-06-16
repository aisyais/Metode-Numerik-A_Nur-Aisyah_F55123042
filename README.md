# Metode-Numerik-A_Nur-Aisyah_F55123042

# UAS Metode Numerik 

Repositori ini berisi implementasi Python dari soal-soal UAS Metode Numerik, mencakup berbagai metode penyelesaian sistem persamaan linear dan nonlinear. Semua kode ditulis dari awal mengikuti alur algoritma yang diajarkan, disertai output detail dan visualisasi.

---

## Struktur Folder

```
UAS METODE NUMERIK/
├── soal 11-1.py        # Thomas Algorithm & Gauss-Seidel
├── soal 11-2.py        # LU Decomposition & Invers Matriks
├── soal 11-3.py        # Thomas Algorithm (Crank-Nicolson)
├── soal 11-4.py        # Verifikasi Cholesky Decomposition
├── soal 11-5.py        # Cholesky: Solve sistem simetris
├── soal 11-6.py        # Cholesky: Matriks 3×3 berbeda
├── soal 11.7.py        # Cholesky pada matriks diagonal
├── soal 11.8.py        # Gauss-Seidel + Overrelaxation
├── soal 11-9.py        # Gauss-Seidel: Konsentrasi reaktor
├── soal 11-10.py       # Jacobi Iteration
├── soal 11-11.py       # Gauss-Seidel: Sistem 3×3
├── soal 11-12.py       # Gauss-Seidel + Underrelaxation
├── soal 11-13.py       # Gauss-Seidel + Overrelaxation
├── soal 11-14.py       # Analisis konvergensi (slope ±1)
├── soal 11-15.py       # Identifikasi sistem tidak konvergen
├── soal 11-16.py       # Invers & Condition Number
├── soal 11-17.py       # Newton-Raphson 2D (Nonlinear)
├── soal 11-18.py       # Produksi Elektronik (Linear)
├── soal 11-19.py       # Hilbert Matrix & Condition Number
├── soal 11-20.py       # Vandermonde Matrix & Condition Number
├── soal 11-21.py       # Augmented Matrix [A | I]
├── soal  11-22.py      # Bentuk Matriks & Invers
├── soal 11.23.py       # Analisis Jumlah Operasi (Thomas vs Gauss)
├── soal 11-24.py       # Program Algoritma Thomas
├── soal 11-25.py       # Program Dekomposisi Cholesky
├── soal 11-26.py       # Program Gauss-Seidel
├── soal 11-27.py       # PDE → Sistem Linear (Beda Hingga)
├── soal 11-28.py       # Solver Sistem Pentadiagonal
├── plot_11_17.png      # Visualisasi kurva nonlinear
├── plot_11_23.png      # Plot perbandingan operasi Thomas vs Gauss
├── plot_11_27.png      # Distribusi konsentrasi PDE
└── plot_11_28.png      # Struktur matriks pentadiagonal
```

---

## Konsep Materi per Soal

### 11.1 — Thomas Algorithm & Gauss-Seidel (Tridiagonal)

Matriks tridiagonal adalah matriks yang hanya memiliki elemen tak-nol pada diagonal utama, satu diagonal di atasnya, dan satu di bawahnya. Dua metode diterapkan:

**Thomas Algorithm** bekerja dalam dua langkah — *forward sweep* untuk mengeliminasi sub-diagonal satu per satu, lalu *back substitution* untuk menghitung solusi dari bawah ke atas. Kompleksitasnya O(n), jauh lebih efisien dari Gauss biasa yang O(n³).

**Gauss-Seidel** adalah metode iteratif yang memperbarui setiap variabel secara berurutan menggunakan nilai terbaru yang tersedia, sampai error relatif antar iterasi turun di bawah toleransi yang ditentukan (es = 5%).

---

### 11.2 — LU Decomposition & Invers Matriks

Dekomposisi LU memfaktorkan matriks A menjadi dua matriks segitiga: L (*lower triangular*) dan U (*upper triangular*), sehingga A = L·U. Ini berguna karena sekali difaktorkan, sistem A·x = b bisa diselesaikan untuk berbagai vektor b dengan efisien.

Invers matriks dihitung kolom per kolom: setiap kolom A⁻¹ adalah solusi dari A·xᵢ = eᵢ, di mana eᵢ adalah unit vector (kolom ke-i dari matriks identitas). Setiap sistem diselesaikan dengan *forward substitution* pada L, lalu *back substitution* pada U.

---

### 11.3 — Thomas Algorithm (Crank-Nicolson)

Sama dengan 11.1 tetapi koefisiennya berasal dari diskritisasi persamaan diferensial panas menggunakan skema Crank-Nicolson. Soal ini menunjukkan bahwa PDE transien yang didiskritisasi secara implisit menghasilkan sistem tridiagonal yang perlu diselesaikan di setiap langkah waktu — persis kasus di mana Thomas Algorithm sangat unggul.

---

### 11.4 — Verifikasi Cholesky Decomposition

Cholesky Decomposition hanya berlaku untuk matriks yang **simetris dan positif definit**: A = L·Lᵀ, di mana L adalah matriks lower triangular. Soal ini memverifikasi hasil dekomposisi dari Example 11.2 buku dengan mengecek apakah L·Lᵀ kembali menghasilkan A asli.

---

### 11.5 & 11.6 — Penyelesaian Sistem dengan Cholesky

Setelah L didapat, sistem A·x = b diselesaikan dalam dua tahap:
1. **Forward substitution** pada L·d = b untuk mendapat d
2. **Back substitution** pada Lᵀ·x = d untuk mendapat x

Metode ini sekitar dua kali lebih efisien dari LU Decomposition umum karena memanfaatkan simetri — hanya setengah matriks yang perlu disimpan dan dihitung.

---

### 11.7 — Cholesky pada Matriks Diagonal

Ketika A diagonal, hasil Cholesky seharusnya juga diagonal dengan elemen-elemen `l_ii = √(a_ii)`. Soal ini memverifikasi bahwa rumus umum Cholesky menyederhanakan diri secara natural untuk kasus khusus ini, dan semua elemen off-diagonal L memang bernilai nol.

---

### 11.8 — Gauss-Seidel dengan Successive Over-Relaxation (SOR)

Gauss-Seidel standar kadang lambat konvergen. SOR mempercepat konvergensi dengan menerapkan faktor relaksasi λ:

`x_i^baru = λ · x_i^GS + (1 - λ) · x_i^lama`

- λ > 1 (*overrelaxation*): mempercepat konvergensi jika sistem sudah stabil
- λ < 1 (*underrelaxation*): menstabilkan sistem yang sulit konvergen
- λ = 1: Gauss-Seidel standar

Soal ini menggunakan λ = 1.2 pada sistem tridiagonal dari 11.1.

---

### 11.9 & 11.10 — Konsentrasi Reaktor: Gauss-Seidel vs Jacobi

Dua metode iteratif dibandingkan pada sistem yang merepresentasikan kesetimbangan massa di tiga reaktor bersambung:

**Gauss-Seidel** (11.9): setiap variabel diperbarui segera menggunakan nilai terbaru yang sudah dihitung di iterasi yang sama.

**Jacobi** (11.10): semua variabel diperbarui *serentak* menggunakan seluruh nilai dari iterasi sebelumnya. Lebih mudah diparalelkan, tetapi umumnya lebih lambat konvergen dibanding Gauss-Seidel.

---

### 11.11 — Gauss-Seidel: Sistem 3×3

Penerapan Gauss-Seidel pada sistem yang sudah diagonal dominan — kondisi yang menjamin konvergensi tanpa perlu menyusun ulang baris. Syarat diagonal dominan: nilai absolut elemen diagonal lebih besar dari jumlah nilai absolut elemen lain di baris yang sama.

---

### 11.12 & 11.13 — Relaxation: Under vs Over

Dua soal ini membandingkan Gauss-Seidel tanpa relaksasi dan dengan relaksasi, pada sistem yang perlu disusun ulang terlebih dahulu agar diagonal dominan:
- **11.12**: underrelaxation (λ = 0.95) untuk menstabilkan konvergensi
- **11.13**: overrelaxation (λ = 1.2) untuk mempercepat konvergensi

---

### 11.14 — Analisis Konvergensi: Kasus Batas Slope ±1

Ketika dua persamaan linear memiliki slope +1 dan −1, diagonal tidak dominan dan konvergensi Gauss-Seidel tidak terjamin secara teoritis. Soal ini mensimulasikan iterasi untuk menunjukkan perilaku ini secara langsung, sesuai dengan Figure 11.5 di buku.

---

### 11.15 — Identifikasi Sistem yang Tidak Konvergen

Diberikan tiga set persamaan, masing-masing dicek apakah memenuhi syarat diagonal dominan. Sistem yang gagal syarat ini kemungkinan besar tidak akan konvergen dengan Gauss-Seidel, dan dibuktikan dengan menjalankan iterasi secara langsung.

---

### 11.16 — Invers Matriks & Condition Number

**Condition number** (κ) mengukur seberapa sensitif solusi terhadap error kecil pada data masukan. Dihitung menggunakan *row-sum norm* (norma tak-hingga):

κ = ‖A‖∞ · ‖A⁻¹‖∞

Nilai κ yang besar menandakan matriks *ill-conditioned*: perubahan kecil pada b bisa menyebabkan perubahan besar pada x. Secara kasar, log₁₀(κ) menyatakan berapa digit presisi yang hilang dalam komputasi.

---

### 11.17 — Newton-Raphson untuk Sistem Nonlinear (dengan Plot)

Sistem dua persamaan nonlinear:
- f(x, y) = 4 − y − 2x² = 0
- g(x, y) = 8 − y² − 4x = 0

diselesaikan dengan **Newton-Raphson 2D**. Metode ini membutuhkan matriks Jacobian (turunan parsial) dan update:

[x, y]^baru = [x, y]^lama − J⁻¹ · F(x, y)

Karena sistem nonlinear bisa memiliki lebih dari satu solusi, pencarian dilakukan dari berbagai titik awal. Hasil divisualisasikan sebagai kurva level f = 0 dan g = 0 yang berpotongan di titik-titik solusi.

---

### 11.18 — Masalah Produksi: Eliminasi Gauss

Kasus nyata: menentukan berapa unit transistor, resistor, dan chip yang bisa diproduksi dari material yang tersedia (tembaga, seng, kaca). Dimodelkan sebagai sistem 3×3 yang diselesaikan dengan eliminasi Gauss dengan partial pivoting, lalu diverifikasi dengan `numpy.linalg.solve`.

---

### 11.19 — Hilbert Matrix & Ill-Conditioning

Matriks Hilbert H[i,j] = 1/(i+j−1) terkenal sebagai contoh ekstrem matriks *ill-conditioned*. Untuk ukuran 10×10, condition numbernya mencapai ∼10¹³, artinya sekitar 13 digit presisi hilang. Soal ini menghitung *spectral condition number* (berbasis nilai singular / SVD) dan menunjukkan secara numerik bahwa solusi yang dihitung jauh dari solusi eksak meskipun inputnya tepat.

---

### 11.20 — Vandermonde Matrix

Matriks Vandermonde V[i,j] = xᵢʲ⁻¹ muncul dalam interpolasi polinomial. Soal ini mengulangi analisis dari 11.19 untuk matriks Vandermonde 6×6 dan membandingkan tingkat ill-conditioning-nya dengan Hilbert 10×10.

---

### 11.21 — Augmented Matrix [A | I]

Membuat augmented matrix dengan menggabungkan matriks koefisien A dan matriks identitas I secara horizontal. Ini adalah langkah awal dalam algoritma **Gauss-Jordan** untuk mencari invers matriks: setelah operasi baris membawa sisi kiri menjadi I, sisi kanan menjadi A⁻¹.

Perintah satu baris Python: `np.hstack([A, np.eye(A.shape[0])])`

---

### 11.22 — Sistem Persamaan → Bentuk Matriks + Invers

Menyusun ulang sistem persamaan yang tidak dalam bentuk standar ke bentuk matriks [A]{x} = {b}, lalu menyelesaikannya sekaligus menghitung transpose dan invers matriks koefisien. Determinan dihitung untuk memverifikasi bahwa matriks non-singular (invers ada).

---

### 11.23 — Analisis Jumlah Operasi: Thomas vs Gauss (dengan Plot)

Perbandingan efisiensi komputasi secara analitis:

| Metode | Operasi | Orde |
|--------|---------|------|
| Thomas Algorithm | ≈ 8n − 7 | O(n) |
| Eliminasi Gauss | ≈ 2n³/3 | O(n³) |

Untuk n = 20 saja, Thomas sudah ∼38× lebih cepat. Plot dibuat untuk n = 2 hingga 20 dalam skala linear dan logaritmik untuk memvisualisasikan perbedaan pertumbuhan ini.

---

### 11.24 — Program Thomas Algorithm (User-Friendly)

Implementasi Thomas Algorithm sebagai fungsi yang dapat digunakan ulang, dengan parameter berupa empat array diagonal (sub, utama, super) dan vektor RHS. Program menampilkan setiap langkah forward sweep dan back substitution secara detail, lalu diuji dengan Example 11.1 dan soal 11.1.

---

### 11.25 — Program Dekomposisi Cholesky

Implementasi lengkap Cholesky Decomposition sebagai fungsi terpisah yang dapat digunakan untuk matriks simetris positif definit sembarang. Program menampilkan setiap elemen L yang dihitung, hasil forward dan back substitution, serta verifikasi L·Lᵀ = A.

---

### 11.26 — Program Gauss-Seidel (dengan Opsi Relaxation)

Implementasi Gauss-Seidel yang lengkap dengan parameter yang dapat dikonfigurasi: toleransi, iterasi maksimum, faktor relaxation λ, dan tebakan awal. Program secara otomatis mengecek diagonal dominan sebelum iterasi dimulai. Diuji dengan Example 11.3 (solusi eksak x₁=3, x₂=−2.5, x₃=7) dan soal konsentrasi reaktor 11.9.

---

### 11.27 — PDE ke Sistem Linear dengan Beda Hingga (dengan Plot)

Persamaan diferensial adveksi-difusi-reaksi satu dimensi:

`0 = D·(d²c/dx²) − U·(dc/dx) − k·c`

didiskritisasi menggunakan aproksimasi beda terpusat (*central difference*) pada setiap titik interior grid. Ini mengubah PDE menjadi sistem persamaan linear tridiagonal yang diselesaikan dengan `numpy.linalg.solve`. Solusi numerik kemudian dibandingkan dengan solusi analitik eksak dan divisualisasikan dalam plot.

---

### 11.28 — Solver Sistem Pentadiagonal (dengan Visualisasi)

Matriks pentadiagonal memiliki *bandwidth* = 5: elemen tak-nol hanya ada pada diagonal utama dan dua diagonal di setiap sisinya. Solver khusus ini mengeksploitasi struktur tersebut sehingga kompleksitasnya O(n), berbeda dengan Gauss biasa yang O(n³). Setiap baris eliminasi hanya mempengaruhi dua baris berikutnya, dan back substitution membutuhkan dua nilai ke depan (xᵢ₊₁ dan xᵢ₊₂). Struktur matriks divisualisasikan dengan heatmap berwarna.

---

## Cara Menjalankan

```bash
# Install dependensi
pip install numpy scipy matplotlib

# Jalankan salah satu soal
python "soal 11-1.py"
python "soal 11-17.py"   # akan menghasilkan plot_11_17.png
python "soal 11.23.py"   # akan menghasilkan plot_11_23.png
```

> Semua file dijalankan secara mandiri dan tidak memiliki dependensi antar file.

---

## Dependencies

- Python 3.x
- NumPy
- SciPy (hanya soal 11.17 dan 11.19)
- Matplotlib (soal yang menghasilkan plot)
