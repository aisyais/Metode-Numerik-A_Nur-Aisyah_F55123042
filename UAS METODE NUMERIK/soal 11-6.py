"""
SOAL 11.6
=========
Lakukan Cholesky decomposition secara manual (by hand) untuk:

A = [[ 8,  20,  15],    b = [ 50 ]
     [20,  80,  50],        [250 ]
     [15,  50,  60]]        [100 ]

dan selesaikan sistem A * x = b.

Metode: sama dengan soal 11.5 (Cholesky + forward/back sub).
"""

import numpy as np

A = np.array([
    [ 8,  20,  15],
    [20,  80,  50],
    [15,  50,  60]
], dtype=float)

b = np.array([50.0, 250.0, 100.0])
n = len(b)

print("=" * 60)
print("Cholesky Decomposition by Hand (Soal 11.6)")
print("=" * 60)

# Cek simetris dan positif definit
print(f"\nSimetris: {np.allclose(A, A.T)}")
ev = np.linalg.eigvalsh(A)
print(f"Eigenvalues: {ev}")
print(f"Positif definit: {np.all(ev > 0)}")

# ── Cholesky ───────────────────────────────────────────────────────────────────
print("\n--- Dekomposisi A = L * L^T ---")
L = np.zeros((n, n))

for i in range(n):
    s = sum(L[i,k]**2 for k in range(i))
    L[i,i] = np.sqrt(A[i,i] - s)
    print(f"\n  Diagonal L[{i},{i}]:")
    print(f"    = sqrt(A[{i},{i}] - sum(L[{i},k]^2))")
    print(f"    = sqrt({A[i,i]} - {s:.4f}) = {L[i,i]:.6f}")
    for j in range(i+1, n):
        s2 = sum(L[j,k]*L[i,k] for k in range(i))
        L[j,i] = (A[j,i] - s2) / L[i,i]
        print(f"  Off-diag L[{j},{i}] = ({A[j,i]} - {s2:.4f}) / {L[i,i]:.6f} = {L[j,i]:.6f}")

print(f"\nL =\n{np.array2string(L, precision=6)}")

# ── Solve ──────────────────────────────────────────────────────────────────────
print("\n--- Forward Sub: L * d = b ---")
d = np.zeros(n)
for i in range(n):
    s = sum(L[i,j]*d[j] for j in range(i))
    d[i] = (b[i] - s) / L[i,i]
    print(f"  d[{i}] = ({b[i]} - {s:.4f}) / {L[i,i]:.6f} = {d[i]:.6f}")

print("\n--- Back Sub: L^T * x = d ---")
Lt = L.T
x = np.zeros(n)
for i in range(n-1, -1, -1):
    s = sum(Lt[i,j]*x[j] for j in range(i+1, n))
    x[i] = (d[i] - s) / Lt[i,i]
    print(f"  x[{i}] = ({d[i]:.4f} - {s:.4f}) / {Lt[i,i]:.6f} = {x[i]:.6f}")

print("\n" + "=" * 60)
print("Hasil:")
for i, xi in enumerate(x):
    print(f"  x{i+1} = {xi:.6f}")

print(f"\nVerifikasi: A @ x = {A @ x}")
print(f"          b     = {b}")