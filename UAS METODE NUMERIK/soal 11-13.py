"""
SOAL 11.13
==========
Selesaikan sistem berikut menggunakan Gauss-Seidel:
(a) tanpa relaksasi
(b) dengan overrelaxation (λ = 1.2)
Toleransi es = 5%. Susun ulang jika perlu.

Persamaan asli:
   2x1 -  6x2 -   x3 = -38
  -3x1 -   x2 +  7x3 = -34
  -8x1 +   x2 -  2x3 = -20

Metode:
- Cek diagonal dominan → susun ulang baris.
- Baris 3: -8x1 → dominan x1
- Baris 1:  2x1, -6x2 → dominan x2 (setelah tukar)
- Baris 2: 7x3 → dominan x3
"""

import numpy as np

# Susun ulang: baris 3→1, baris 1→2, baris 2→3
A = np.array([
    [-8,  1, -2],
    [ 2, -6, -1],
    [-3, -1,  7]
], dtype=float)
b = np.array([-20.0, -38.0, -34.0])
n = len(b)

print("=" * 60)
print("Gauss-Seidel ± Overrelaxation (Soal 11.13)")
print("=" * 60)
print("\nSetelah disusun ulang:")
print("  -8x1 +  x2 -  2x3 = -20")
print("   2x1 -  6x2 -   x3 = -38")
print("  -3x1 -   x2 +  7x3 = -34")

print("\nCek diagonal dominan:")
for i in range(n):
    off = sum(abs(A[i,j]) for j in range(n) if j != i)
    ok = "✓" if abs(A[i,i]) >= off else "~"
    print(f"  Baris {i}: |{A[i,i]}| vs {off} → {ok}")

x_eksak = np.linalg.solve(A, b)
es = 5.0

for label, lam in [("(a) Tanpa relaksasi (λ=1.0)", 1.0),
                   ("(b) Overrelaxation (λ=1.2)", 1.2)]:
    print(f"\n{'='*60}")
    print(label)
    print("="*60)

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

    print(f"  Hasil: x1={x[0]:.4f}, x2={x[1]:.4f}, x3={x[2]:.4f}")

print(f"\nSolusi eksak: {x_eksak}")