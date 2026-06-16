"""
SOAL 11.2
=========
Tentukan invers matriks untuk sistem tridiagonal dari Soal 11.1
menggunakan LU decomposition dan unit vectors.

Matriks:
A = [[ 0.8  -0.4   0  ]
     [-0.4   0.8  -0.4]
     [ 0    -0.4   0.8]]

Metode:
- LU Decomposition: faktorkan A = L * U
  L = lower triangular, U = upper triangular (tanpa pivoting)
- Invers dihitung dengan menyelesaikan A * col_i = e_i
  untuk setiap unit vector e_i (kolom identitas).
- Setiap sistem L*y = e_i (forward sub) lalu U*x = y (back sub).
"""

import numpy as np

A = np.array([
    [ 0.8, -0.4,  0.0],
    [-0.4,  0.8, -0.4],
    [ 0.0, -0.4,  0.8]
], dtype=float)

n = len(A)

# ── Langkah 1: LU Decomposition (Doolittle) ────────────────────────────────────
print("=" * 60)
print("Langkah 1: LU Decomposition (tanpa pivoting)")
print("=" * 60)

L = np.eye(n)
U = A.copy()

for k in range(n - 1):
    for i in range(k + 1, n):
        if U[k, k] == 0:
            raise ValueError("Pivot nol — perlu pivoting!")
        faktor = U[i, k] / U[k, k]
        L[i, k] = faktor
        U[i, :] = U[i, :] - faktor * U[k, :]

print("\nMatriks L (lower triangular):")
print(np.array2string(L, precision=6))
print("\nMatriks U (upper triangular):")
print(np.array2string(U, precision=6))
print("\nVerifikasi L*U = A:")
print(np.array2string(L @ U, precision=6))

# ── Langkah 2: Hitung invers kolom per kolom ──────────────────────────────────
print("\n" + "=" * 60)
print("Langkah 2: Hitung Invers dengan Unit Vectors")
print("=" * 60)

A_inv = np.zeros((n, n))

for col in range(n):
    e = np.zeros(n)
    e[col] = 1.0
    print(f"\n  Kolom {col+1}: e = {e}")

    # Forward substitution: L * y = e
    y = np.zeros(n)
    for i in range(n):
        y[i] = (e[i] - sum(L[i, j] * y[j] for j in range(i))) / L[i, i]
    print(f"    Forward sub → y = {y}")

    # Back substitution: U * x = y
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(U[i, j] * x[j] for j in range(i + 1, n))) / U[i, i]
    print(f"    Back sub    → x = {x}")

    A_inv[:, col] = x

# ── Hasil ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Hasil: Matriks Invers A^-1")
print("=" * 60)
print(np.array2string(A_inv, precision=4))

print("\nVerifikasi A * A^-1 = I:")
print(np.array2string(A @ A_inv, precision=6, suppress_small=True))

print("\nPembanding (numpy.linalg.inv):")
print(np.array2string(np.linalg.inv(A), precision=4))