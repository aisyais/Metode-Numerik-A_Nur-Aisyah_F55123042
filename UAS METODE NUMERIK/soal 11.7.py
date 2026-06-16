"""
SOAL 11.7
=========
Hitung Cholesky decomposition untuk matriks diagonal:

A = [[9, 0, 0],
     [0, 25, 0],
     [0, 0, 4]]

Analisis apakah hasilnya masuk akal berdasarkan persamaan
Cholesky (Eq. 11.3 dan 11.4).

Insight:
- Untuk matriks diagonal, L harus juga diagonal.
- l_ii = sqrt(a_ii)
- Semua off-diagonal L = 0
- Hasilnya: L = [[3,0,0],[0,5,0],[0,0,2]]
"""

import numpy as np

A = np.array([
    [9,  0,  0],
    [0, 25,  0],
    [0,  0,  4]
], dtype=float)

n = len(A)

print("=" * 60)
print("Cholesky Decomposition — Matriks Diagonal (Soal 11.7)")
print("=" * 60)
print(f"\nA =\n{A}")

# Cholesky manual
L = np.zeros((n, n))
for i in range(n):
    s = sum(L[i,k]**2 for k in range(i))
    L[i,i] = np.sqrt(A[i,i] - s)
    for j in range(i+1, n):
        s2 = sum(L[j,k]*L[i,k] for k in range(i))
        L[j,i] = (A[j,i] - s2) / L[i,i]

print(f"\nL =\n{L}")
print(f"\nL^T =\n{L.T}")

print("\n--- Analisis ---")
print("Untuk matriks diagonal:")
print(f"  l_11 = sqrt(a_11) = sqrt(9)  = {np.sqrt(9)}")
print(f"  l_22 = sqrt(a_22) = sqrt(25) = {np.sqrt(25)}")
print(f"  l_33 = sqrt(a_33) = sqrt(4)  = {np.sqrt(4)}")
print("\nSemua elemen off-diagonal L = 0 (karena A diagonal).")
print("Hasil masuk akal: Cholesky dari matriks diagonal adalah")
print("akar kuadrat elemen diagonalnya.")

print(f"\nVerifikasi L * L^T = A:")
print(np.array2string(L @ L.T, suppress_small=True))
print(f"Cocok dengan A: {np.allclose(L @ L.T, A)}")