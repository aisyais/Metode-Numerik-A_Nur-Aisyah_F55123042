"""
SOAL 11.4
=========
Konfirmasi validitas Cholesky decomposition dari Example 11.2
dengan memeriksa apakah produk [L] * [L]^T menghasilkan [A].

Matriks A dari Example 11.2:
A = [[6,  15,  55 ],
     [15, 55,  225],
     [55, 225, 979]]

Metode: Cholesky Decomposition
- Hanya berlaku untuk matriks simetris dan positif definit.
- A = L * L^T, di mana L adalah matriks lower triangular.
- Formula elemen L:
    l_ii = sqrt(a_ii - sum(l_ik^2, k=0..i-1))
    l_ji = (a_ji - sum(l_jk * l_ik, k=0..i-1)) / l_ii  untuk j > i
"""

import numpy as np

# Matriks dari Example 11.2
A = np.array([
    [  6,   15,   55],
    [ 15,   55,  225],
    [ 55,  225,  979]
], dtype=float)

print("=" * 60)
print("Cholesky Decomposition — Verifikasi Example 11.2")
print("=" * 60)
print("\nMatriks A:")
print(A)

# ── Cek syarat: simetris dan positif definit ───────────────────────────────────
print("\nCek syarat Cholesky:")
print(f"  Simetris: {np.allclose(A, A.T)}")
eigenvalues = np.linalg.eigvalsh(A)
print(f"  Eigenvalues: {eigenvalues}")
print(f"  Positif definit (semua eigenvalue > 0): {np.all(eigenvalues > 0)}")

# ── Cholesky manual ────────────────────────────────────────────────────────────
print("\n--- Hitung L secara manual ---")
n = len(A)
L = np.zeros((n, n))

for i in range(n):
    # Elemen diagonal
    jumlah = sum(L[i, k]**2 for k in range(i))
    L[i, i] = np.sqrt(A[i, i] - jumlah)
    print(f"  L[{i},{i}] = sqrt({A[i,i]:.4f} - {jumlah:.4f}) = {L[i,i]:.6f}")

    # Elemen bawah diagonal
    for j in range(i + 1, n):
        jumlah2 = sum(L[j, k] * L[i, k] for k in range(i))
        L[j, i] = (A[j, i] - jumlah2) / L[i, i]
        print(f"  L[{j},{i}] = ({A[j,i]:.4f} - {jumlah2:.4f}) / {L[i,i]:.6f} = {L[j,i]:.6f}")

print("\nMatriks L:")
print(np.array2string(L, precision=6))

# ── Verifikasi L * L^T = A ────────────────────────────────────────────────────
print("\n--- Verifikasi: L * L^T ---")
hasil = L @ L.T
print(np.array2string(hasil, precision=4))

print("\nA asli:")
print(A)

print(f"\nApakah L * L^T ≈ A? {np.allclose(L @ L.T, A)}")
print(f"Max error: {np.max(np.abs(L @ L.T - A)):.2e}")

# Pembanding numpy
L_np = np.linalg.cholesky(A)
print("\nPembanding (numpy.linalg.cholesky):")
print(np.array2string(L_np, precision=6))