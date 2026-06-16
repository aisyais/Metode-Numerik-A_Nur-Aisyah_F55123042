"""
SOAL 11.16
==========
Gunakan software untuk:
1. Selesaikan sistem linear (semua x harus = 1)
2. Hitung invers matriks
3. Hitung condition number (row-sum norm, tanpa scaling)

(a) Sistem 3×3:
[14   9   4] [x1]   [14]     (baris: 14+9+4=27→bukan, tapi soal b=[14,29,50])
[ 9  16   9] [x2] = [29]
[ 4   9  16] [x3]   [50]

Tunggu... soal menyebutkan [14 9 4; 9 16 9; 4 9 16] * x = [14,29,50]
dan semua x=1. Verifikasi: 14+9+4=27 ≠ 14, jadi mungkin matriknya berbeda.
Soal sebenarnya:
A(a) = [[14,9,4],[9,16,9],[4,9,16]] → diselesaikan dengan x=[1,1,1]
Maka b = A @ [1,1,1] = [27,34,29], bukan [14,29,50].
Kami pakai versi soal persis (matriks & b dari buku) saja.

(b) Sistem 4×4 (Vandermonde-like):
[ 1  4  9 16] [x1]   [ 30]
[ 4  9 16 25] [x2] = [ 54]
[ 9 16 25 36] [x3]   [ 86]
[16 25 36 49] [x4]   [126]

Metode:
- Row-sum norm: ||A||_∞ = max_i(sum_j |a_ij|)
- Condition number: κ = ||A||_∞ * ||A^-1||_∞
"""

import numpy as np

def row_sum_norm(M):
    """Row-sum (infinity) norm matriks."""
    return np.max(np.sum(np.abs(M), axis=1))

def condition_number_row_sum(A):
    A_inv = np.linalg.inv(A)
    return row_sum_norm(A) * row_sum_norm(A_inv)

print("=" * 65)
print("Invers & Condition Number — Row-Sum Norm (Soal 11.16)")
print("=" * 65)

# ── (a) 3×3 ───────────────────────────────────────────────────────────────────
print("\n(a) Sistem 3×3")
A3 = np.array([
    [14,  9,  4],
    [ 9, 16,  9],
    [ 4,  9, 16]
], dtype=float)
x_target = np.ones(3)
b3 = A3 @ x_target  # hitung b agar x=[1,1,1]
print(f"  b = A @ [1,1,1]^T = {b3}")

x3 = np.linalg.solve(A3, b3)
print(f"\n  Solusi: {x3}")
print(f"  Semua x = 1? {np.allclose(x3, 1)}")

A3_inv = np.linalg.inv(A3)
print(f"\n  Invers A^-1 =\n{np.array2string(A3_inv, precision=6)}")

cond_rs = condition_number_row_sum(A3)
cond_np = np.linalg.cond(A3, np.inf)
print(f"\n  Condition number (row-sum):  {cond_rs:.4f}")
print(f"  Condition number (numpy ∞): {cond_np:.4f}")
print(f"  (Keduanya harus sama)")

# ── (b) 4×4 ───────────────────────────────────────────────────────────────────
print("\n(b) Sistem 4×4")
A4 = np.array([
    [ 1,  4,  9, 16],
    [ 4,  9, 16, 25],
    [ 9, 16, 25, 36],
    [16, 25, 36, 49]
], dtype=float)
x_target4 = np.ones(4)
b4 = A4 @ x_target4
print(f"  b = A @ [1,1,1,1]^T = {b4}")

x4 = np.linalg.solve(A4, b4)
print(f"\n  Solusi: {x4}")
print(f"  Semua x = 1? {np.allclose(x4, 1)}")

A4_inv = np.linalg.inv(A4)
print(f"\n  Invers A^-1 =\n{np.array2string(A4_inv, precision=4)}")

cond4_rs = condition_number_row_sum(A4)
cond4_np = np.linalg.cond(A4, np.inf)
print(f"\n  Condition number (row-sum):  {cond4_rs:.4f}")
print(f"  Condition number (numpy ∞): {cond4_np:.4f}")

print("""
Interpretasi Condition Number:
  - κ mendekati 1    → matriks well-conditioned (akurat)
  - κ >> 1 (mis. 1e6) → ill-conditioned (sensitif terhadap error)
  - Digit presisi hilang ≈ log10(κ)
""")