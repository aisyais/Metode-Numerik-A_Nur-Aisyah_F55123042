"""
SOAL 11.11
==========
Selesaikan sistem berikut dengan Gauss-Seidel hingga es < 5%:

 10x1 +  2x2 -   x3 =  27
 -3x1 -  6x2 +  2x3 = -61.5
   x1 +   x2 +  5x3 = -21.5

Metode: Gauss-Seidel
- Sistem sudah diagonal dominan → tidak perlu susun ulang.
- Update berurutan dengan nilai terbaru.
"""

import numpy as np

A = np.array([
    [10,  2, -1],
    [-3, -6,  2],
    [ 1,  1,  5]
], dtype=float)

b = np.array([27.0, -61.5, -21.5])
n = len(b)
es = 5.0

print("=" * 60)
print("Gauss-Seidel — Sistem 3×3 (Soal 11.11)")
print("=" * 60)

print("\nPersamaan:")
print("  10x1 + 2x2 - x3  = 27")
print("  -3x1 - 6x2 + 2x3 = -61.5")
print("  x1  + x2  + 5x3  = -21.5")

print("\nCek diagonal dominan:")
for i in range(n):
    off = sum(abs(A[i,j]) for j in range(n) if j != i)
    ok = "✓" if abs(A[i,i]) >= off else "✗"
    print(f"  Baris {i}: |{A[i,i]}| vs {off} → {ok}")

x = np.zeros(n)
print(f"\n{'Iter':>4} | {'x1':>10} | {'x2':>10} | {'x3':>10} | {'ea_max':>10}")
print("-" * 55)

for it in range(1, 100):
    x_old = x.copy()

    x[0] = (b[0] - A[0,1]*x[1] - A[0,2]*x[2]) / A[0,0]
    x[1] = (b[1] - A[1,0]*x[0] - A[1,2]*x[2]) / A[1,1]
    x[2] = (b[2] - A[2,0]*x[0] - A[2,1]*x[1]) / A[2,2]

    ea = [abs((x[i]-x_old[i])/x[i])*100 if x[i] != 0 else 0 for i in range(n)]
    ea_max = max(ea)
    print(f"  {it:>3} | {x[0]:>10.5f} | {x[1]:>10.5f} | {x[2]:>10.5f} | {ea_max:>9.4f}%")

    if ea_max < es and it > 1:
        print(f"\n  Konvergen pada iterasi ke-{it}!")
        break

print("\n" + "=" * 60)
print("Hasil:")
for i, xi in enumerate(x):
    print(f"  x{i+1} = {xi:.5f}")

x_eksak = np.linalg.solve(A, b)
print(f"\nSolusi eksak: {x_eksak}")