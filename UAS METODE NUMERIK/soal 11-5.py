"""
SOAL 11.5
=========
Lakukan Cholesky decomposition untuk sistem simetris berikut,
dan gunakan hasilnya untuk menyelesaikan sistem persamaan:

A = [[ 6,   15,   55 ],    b = [152.6 ]
     [15,   55,  225 ],        [585.6 ]
     [55,  225,  979 ]]        [2488.8]

Metode: Cholesky Decomposition
- Faktorkan A = L * L^T
- Selesaikan L * d = b (forward substitution)
- Selesaikan L^T * a = d (back substitution)
"""

import numpy as np

A = np.array([
    [  6,   15,   55 ],
    [ 15,   55,  225 ],
    [ 55,  225,  979 ]
], dtype=float)

b = np.array([152.6, 585.6, 2488.8])
n = len(b)

print("=" * 60)
print("Cholesky Decomposition + Solve (Soal 11.5)")
print("=" * 60)
print("\nSistem A * a = b:")
print(f"  A =\n{A}")
print(f"  b = {b}")

# ── Langkah 1: Cholesky Decomposition ─────────────────────────────────────────
print("\n--- Langkah 1: Hitung L (A = L * L^T) ---")
L = np.zeros((n, n))
for i in range(n):
    jumlah = sum(L[i,k]**2 for k in range(i))
    L[i,i] = np.sqrt(A[i,i] - jumlah)
    print(f"  L[{i},{i}] = sqrt({A[i,i]} - {jumlah:.4f}) = {L[i,i]:.6f}")
    for j in range(i+1, n):
        jumlah2 = sum(L[j,k]*L[i,k] for k in range(i))
        L[j,i] = (A[j,i] - jumlah2) / L[i,i]
        print(f"  L[{j},{i}] = ({A[j,i]} - {jumlah2:.4f}) / {L[i,i]:.6f} = {L[j,i]:.6f}")

print(f"\nL =\n{np.array2string(L, precision=6)}")
print(f"L^T =\n{np.array2string(L.T, precision=6)}")

# ── Langkah 2: Forward substitution — L * d = b ────────────────────────────────
print("\n--- Langkah 2: Forward Sub — L * d = b ---")
d = np.zeros(n)
for i in range(n):
    jumlah = sum(L[i,j]*d[j] for j in range(i))
    d[i] = (b[i] - jumlah) / L[i,i]
    print(f"  d[{i}] = ({b[i]} - {jumlah:.4f}) / {L[i,i]:.6f} = {d[i]:.6f}")

print(f"\n  d = {d}")

# ── Langkah 3: Back substitution — L^T * a = d ────────────────────────────────
print("\n--- Langkah 3: Back Sub — L^T * a = d ---")
Lt = L.T
a = np.zeros(n)
for i in range(n-1, -1, -1):
    jumlah = sum(Lt[i,j]*a[j] for j in range(i+1, n))
    a[i] = (d[i] - jumlah) / Lt[i,i]
    print(f"  a[{i}] = ({d[i]:.4f} - {jumlah:.4f}) / {Lt[i,i]:.6f} = {a[i]:.6f}")

print("\n" + "=" * 60)
print("Hasil:")
labels = ["a0", "a1", "a2"]
for label, val in zip(labels, a):
    print(f"  {label} = {val:.6f}")

print("\nVerifikasi A @ a = b:")
print(f"  A @ a = {A @ a}")
print(f"  b     = {b}")
print(f"  Error = {np.max(np.abs(A @ a - b)):.2e}")