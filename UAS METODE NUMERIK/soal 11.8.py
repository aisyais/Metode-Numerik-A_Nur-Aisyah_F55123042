"""
SOAL 11.8
=========
Selesaikan sistem tridiagonal dari Soal 11.1 menggunakan
Gauss-Seidel dengan overrelaxation (λ = 1.2), toleransi es = 5%.

Sistem:
[ 0.8  -0.4   0  ] [x1]   [41 ]
[-0.4   0.8  -0.4] [x2] = [25 ]
[ 0    -0.4   0.8] [x3]   [105]

Metode: Gauss-Seidel dengan Successive Over-Relaxation (SOR)
- Nilai baru diperbarui dengan:
    x_i^new = λ * x_i^GS + (1 - λ) * x_i^old
  di mana x_i^GS adalah nilai Gauss-Seidel tanpa relaksasi.
- λ = 1.2 → overrelaxation (mempercepat konvergensi)
- λ = 1   → Gauss-Seidel biasa
- λ < 1   → underrelaxation (memperlambat, menstabilkan)
"""

import numpy as np

A = np.array([
    [ 0.8, -0.4,  0.0],
    [-0.4,  0.8, -0.4],
    [ 0.0, -0.4,  0.8]
], dtype=float)
b = np.array([41.0, 25.0, 105.0])
n = len(b)

lam = 1.2   # faktor relaksasi
es  = 5.0   # toleransi %
max_iter = 100

print("=" * 60)
print(f"Gauss-Seidel + Overrelaxation (λ={lam}, es={es}%)")
print("=" * 60)

x = np.zeros(n)
print(f"\n{'Iter':>4} | {'x1':>10} | {'x2':>10} | {'x3':>10} | {'ea_max':>10}")
print("-" * 55)

for it in range(1, max_iter + 1):
    x_old = x.copy()

    for i in range(n):
        # Nilai Gauss-Seidel murni
        sigma = sum(A[i,j] * x[j] for j in range(n) if j != i)
        x_gs = (b[i] - sigma) / A[i,i]
        # Terapkan relaksasi
        x[i] = lam * x_gs + (1 - lam) * x_old[i]

    # Error relatif
    ea = []
    for i in range(n):
        if x[i] != 0:
            ea.append(abs((x[i] - x_old[i]) / x[i]) * 100)
        else:
            ea.append(0.0)
    ea_max = max(ea)

    print(f"  {it:>3} | {x[0]:>10.5f} | {x[1]:>10.5f} | {x[2]:>10.5f} | {ea_max:>9.4f}%")

    if ea_max < es and it > 1:
        print(f"\n  Konvergen pada iterasi ke-{it}!")
        break

print("\n" + "=" * 60)
print("Hasil:")
for i, xi in enumerate(x):
    print(f"  x{i+1} = {xi:.5f}")

# Solusi eksak untuk perbandingan
x_eksak = np.linalg.solve(A, b)
print("\nSolusi eksak (numpy):")
for i, xi in enumerate(x_eksak):
    print(f"  x{i+1} = {xi:.5f}")