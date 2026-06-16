"""
SOAL 11.12
==========
Selesaikan sistem berikut dengan Gauss-Seidel:
(a) tanpa relaksasi
(b) dengan underrelaxation (λ = 0.95)
Toleransi es = 5%. Susun ulang jika perlu agar konvergen.

Persamaan asli:
  -3x1 +  x2 + 12x3 = 50
   6x1 -  x2 -   x3 = 3
   6x1 + 9x2 +   x3 = 40

Metode:
- Periksa diagonal dominan → susun ulang baris jika belum.
- Gauss-Seidel standar, lalu bandingkan dengan SOR (λ=0.95).
- λ < 1 (underrelaxation) membantu menstabilkan sistem
  yang lambat konvergen atau hampir tidak konvergen.
"""

import numpy as np

# Persamaan asli — perlu disusun ulang agar diagonal dominan
# Baris 1: -3x1 + x2 + 12x3 = 50  → dominan di x3 (koef 12)
# Baris 2: 6x1 - x2 - x3 = 3      → dominan di x1 (koef 6)
# Baris 3: 6x1 + 9x2 + x3 = 40    → dominan di x2 (koef 9)
# Susun ulang: baris 2 → baris 1, baris 3 → baris 2, baris 1 → baris 3

A = np.array([
    [ 6, -1, -1],
    [ 6,  9,  1],
    [-3,  1, 12]
], dtype=float)

b = np.array([3.0, 40.0, 50.0])
n = len(b)

print("=" * 60)
print("Gauss-Seidel ± Underrelaxation (Soal 11.12)")
print("=" * 60)

print("\nSetelah disusun ulang:")
print("  6x1 - x2 - x3  = 3")
print("  6x1 + 9x2 + x3 = 40")
print("  -3x1 + x2 + 12x3 = 50")

print("\nCek diagonal dominan:")
for i in range(n):
    off = sum(abs(A[i,j]) for j in range(n) if j != i)
    ok = "✓" if abs(A[i,i]) >= off else "~"
    print(f"  Baris {i}: |{A[i,i]}| vs {off} → {ok}")

x_eksak = np.linalg.solve(A, b)
es = 5.0

for label, lam in [("(a) Tanpa relaksasi (λ=1.0)", 1.0),
                   ("(b) Underrelaxation (λ=0.95)", 0.95)]:
    print(f"\n{'='*60}")
    print(label)
    print(f"{'='*60}")

    x = np.zeros(n)
    print(f"\n{'Iter':>4} | {'x1':>10} | {'x2':>10} | {'x3':>10} | {'ea_max':>10}")
    print("-" * 55)

    for it in range(1, 200):
        x_old = x.copy()

        for i in range(n):
            sigma = sum(A[i,j]*x[j] for j in range(n) if j != i)
            x_gs = (b[i] - sigma) / A[i,i]
            x[i] = lam * x_gs + (1 - lam) * x_old[i]

        ea = [abs((x[i]-x_old[i])/x[i])*100 if x[i] != 0 else 0 for i in range(n)]
        ea_max = max(ea)
        print(f"  {it:>3} | {x[0]:>10.5f} | {x[1]:>10.5f} | {x[2]:>10.5f} | {ea_max:>9.4f}%")

        if ea_max < es and it > 1:
            print(f"\n  Konvergen pada iterasi ke-{it}!")
            break
    else:
        print("\n  TIDAK konvergen dalam 200 iterasi.")

    print(f"  Hasil akhir: x1={x[0]:.5f}, x2={x[1]:.5f}, x3={x[2]:.5f}")

print(f"\nSolusi eksak: {x_eksak}")