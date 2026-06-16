"""
=============================================================
SOAL 11.21 - Augmented Matrix [A | I]
=============================================================

PERMASALAHAN:
Diberikan matriks persegi [A], tulis perintah untuk membuat
matriks baru [Aug] yang merupakan matriks [A] yang di-augment
dengan matriks identitas [I].

Bentuk: [Aug] = [A | I]

Ini berguna dalam:
  - Mencari invers matriks dengan eliminasi Gauss-Jordan
  - Verifikasi invers matriks

PERINTAH MATLAB (referensi soal):
  Aug = [A, eye(size(A))]

EKUIVALEN PYTHON/NUMPY:
  Aug = np.hstack([A, np.eye(A.shape[0])])

Demonstrasi pada matriks contoh 3x3 dan 4x4.
=============================================================
"""

import numpy as np

print("=" * 60)
print("SOAL 11.21 - Membuat Augmented Matrix [A | I]")
print("=" * 60)

# ─── Penjelasan konsep ────────────────────────────────────
print("""
KONSEP:
Augmented matrix [A | I] digunakan dalam algoritma Gauss-Jordan
untuk mencari invers matriks.

Jika kita lakukan operasi baris pada [A | I] hingga sisi kiri
menjadi I (identitas), maka sisi kanan menjadi A⁻¹.

[ A | I ] → operasi baris → [ I | A⁻¹ ]
""")

# ─── Perintah satu baris (ekuivalen MATLAB) ───────────────
print("=" * 60)
print("SATU BARIS PERINTAH PYTHON:")
print("=" * 60)
print("""
  # MATLAB   : Aug = [A, eye(size(A))]
  # Python   : Aug = np.hstack([A, np.eye(A.shape[0])])
""")

# ─── Contoh 1: Matriks 3x3 ───────────────────────────────
print("=" * 60)
print("CONTOH 1: Matriks 3x3")
print("=" * 60)

A3 = np.array([
    [2, 1, -1],
    [-3, -1, 2],
    [-2, 1, 2]
], dtype=float)

n3 = A3.shape[0]
I3 = np.eye(n3)

# ─── SATU BARIS PERINTAH ──────────────────────────────────
Aug3 = np.hstack([A3, I3])

print("\nMatriks A:")
print(np.array2string(A3, precision=1))

print("\nMatriks Identitas I:")
print(np.array2string(I3, precision=1))

print("\nAugmented Matrix [A | I] (satu baris perintah):")
print("Aug = np.hstack([A, np.eye(A.shape[0])])")
print()
# Tampilkan dengan pemisah visual
header = "  [ " + "  ".join([f"A col{j+1}" for j in range(n3)]) + \
         " | " + "  ".join([f"I col{j+1}" for j in range(n3)]) + " ]"
print(header)
for row in Aug3:
    left = "  ".join([f"{v:6.1f}" for v in row[:n3]])
    right = "  ".join([f"{v:6.1f}" for v in row[n3:]])
    print(f"  [ {left} | {right} ]")

# ─── Verifikasi: Gauss-Jordan untuk cari invers ───────────
print("\nVerifikasi: Gauss-Jordan pada [A | I] → [I | A⁻¹]")

Aug_gj = Aug3.copy()
n = n3

# Forward elimination
for k in range(n):
    # Pivot
    max_row = np.argmax(abs(Aug_gj[k:, k])) + k
    Aug_gj[[k, max_row]] = Aug_gj[[max_row, k]]
    
    # Normalisasi baris pivot
    Aug_gj[k] /= Aug_gj[k, k]
    
    # Eliminasi kolom
    for i in range(n):
        if i != k:
            Aug_gj[i] -= Aug_gj[i, k] * Aug_gj[k]

A_inv_gj = Aug_gj[:, n:]
A_inv_np = np.linalg.inv(A3)

print("\n  A⁻¹ dari Gauss-Jordan:")
print(np.array2string(A_inv_gj, precision=6))

print("\n  A⁻¹ dari numpy (verifikasi):")
print(np.array2string(A_inv_np, precision=6))

print("\n  Konfirmasi A × A⁻¹ = I:")
check = np.dot(A3, A_inv_gj)
print(np.array2string(check, precision=4, suppress_small=True))

# ─── Contoh 2: Matriks 4x4 ───────────────────────────────
print("\n" + "=" * 60)
print("CONTOH 2: Matriks 4x4 (dari soal 11.16b)")
print("=" * 60)

A4 = np.array([
    [1,  4,  9, 16],
    [4,  9, 16, 25],
    [9, 16, 25, 36],
    [16, 25, 36, 49]
], dtype=float)

Aug4 = np.hstack([A4, np.eye(A4.shape[0])])

print("\nMatriks A (4x4):")
print(np.array2string(A4, precision=0))

print("\nAugmented Matrix [A | I] (4x8):")
print(f"Ukuran: {Aug4.shape}")
print(np.array2string(Aug4, precision=1, suppress_small=True))

print("\n" + "=" * 60)
print("RINGKASAN:")
print("=" * 60)
print("""
  Perintah SATU BARIS Python untuk augmented matrix:

      Aug = np.hstack([A, np.eye(A.shape[0])])

  Ekuivalen MATLAB:
      Aug = [A, eye(size(A))]

  Kegunaan utama:
  - Mencari A⁻¹ dengan Gauss-Jordan
  - Basis algoritma invers iteratif
""")