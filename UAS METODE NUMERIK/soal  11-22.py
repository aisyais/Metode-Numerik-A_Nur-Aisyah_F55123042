"""
=============================================================
SOAL 11.22 - Bentuk Matriks & Invers
=============================================================

PERMASALAHAN:
Tulis sistem persamaan berikut dalam bentuk matriks:
    50 = 5x₃ - 7x₂
    4x₂ + 7x₃ + 30 = 0
    x₁ - 7x₃ = 40 - 3x₂ + 5x₁

Selesaikan sistem untuk x₁, x₂, x₃.
Hitung juga transpose dan invers matriks koefisien.

LANGKAH PENYELESAIAN:
1. Susun ulang persamaan ke bentuk standar: ax₁ + bx₂ + cx₃ = d
2. Bentuk matriks [A]{x} = {b}
3. Selesaikan dengan eliminasi Gauss atau numpy
4. Hitung Aᵀ dan A⁻¹
=============================================================
"""

import numpy as np

print("=" * 60)
print("SOAL 11.22 - Sistem Persamaan → Bentuk Matriks")
print("=" * 60)

# ─── Penyusunan ulang persamaan ───────────────────────────
print("""
PENYUSUNAN ULANG PERSAMAAN:

Persamaan 1:  50 = 5x₃ - 7x₂
  → 0·x₁ - 7x₂ + 5x₃ = 50

Persamaan 2:  4x₂ + 7x₃ + 30 = 0
  → 0·x₁ + 4x₂ + 7x₃ = -30

Persamaan 3:  x₁ - 7x₃ = 40 - 3x₂ + 5x₁
  → x₁ - 7x₃ - 40 + 3x₂ - 5x₁ = 0
  → -4x₁ + 3x₂ - 7x₃ = 40

BENTUK MATRIKS [A]{x} = {b}:
  |  0  -7   5 | | x₁ |   |  50 |
  |  0   4   7 | | x₂ | = | -30 |
  | -4   3  -7 | | x₃ |   |  40 |
""")

# ─── Definisi matriks ─────────────────────────────────────
A = np.array([
    [ 0, -7,  5],   # persamaan 1
    [ 0,  4,  7],   # persamaan 2
    [-4,  3, -7],   # persamaan 3
], dtype=float)

b = np.array([50, -30, 40], dtype=float)

print("Matriks A:")
print(np.array2string(A, precision=1))
print(f"\nVektor b = {b}")

# ─── Selesaikan dengan numpy ──────────────────────────────
print("\n" + "=" * 60)
print("PENYELESAIAN SISTEM [A]{x} = {b}")
print("=" * 60)

x = np.linalg.solve(A, b)

print(f"\n  x₁ = {x[0]:.6f}")
print(f"  x₂ = {x[1]:.6f}")
print(f"  x₃ = {x[2]:.6f}")

# ─── Verifikasi ───────────────────────────────────────────
print("\nVerifikasi (Ax harus = b):")
Ax = np.dot(A, x)
eq_names = ['Pers. 1', 'Pers. 2', 'Pers. 3']
for i in range(3):
    ok = '✓' if abs(Ax[i] - b[i]) < 1e-8 else '✗'
    print(f"  {eq_names[i]}: {Ax[i]:.4f} = {b[i]:.4f} {ok}")

# ─── Transpose matriks ────────────────────────────────────
print("\n" + "=" * 60)
print("TRANSPOSE MATRIKS Aᵀ:")
print("=" * 60)

A_T = A.T
print(np.array2string(A_T, precision=1))

print("\nPenjelasan transpose: baris ke-i matriks A menjadi kolom ke-i Aᵀ")
print("Sehingga A[i,j] → Aᵀ[j,i]")

# ─── Invers matriks ───────────────────────────────────────
print("\n" + "=" * 60)
print("INVERS MATRIKS A⁻¹:")
print("=" * 60)

A_inv = np.linalg.inv(A)
print(np.array2string(A_inv, precision=6))

print("\nVerifikasi A × A⁻¹ = I:")
I_check = np.dot(A, A_inv)
print(np.array2string(I_check, precision=4, suppress_small=True))

# ─── Determinan ──────────────────────────────────────────
det_A = np.linalg.det(A)
print(f"\nDeterminan det(A) = {det_A:.6f}")
if abs(det_A) > 1e-10:
    print("Determinan ≠ 0 → Matriks non-singular, invers ada ✓")
else:
    print("Determinan = 0 → Matriks singular, invers tidak ada ✗")

# ─── Kondisi matriks ─────────────────────────────────────
cond = np.linalg.cond(A)
print(f"Condition number κ = {cond:.4f}")

print("\n" + "=" * 60)
print("RINGKASAN HASIL:")
print("=" * 60)
print(f"  x₁ = {x[0]:.6f}")
print(f"  x₂ = {x[1]:.6f}")
print(f"  x₃ = {x[2]:.6f}")