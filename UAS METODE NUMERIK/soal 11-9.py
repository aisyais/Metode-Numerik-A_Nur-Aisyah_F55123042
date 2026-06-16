"""
SOAL 11.9
=========
Selesaikan sistem persamaan konsentrasi reaktor berikut
menggunakan Gauss-Seidel (es = 5%):

 15c1 -  3c2 -   c3 = 3800
 -3c1 + 18c2 -  6c3 = 1200
 -4c1 -   c2 + 12c3 = 2350

c1, c2, c3 = konsentrasi (g/m³) di masing-masing reaktor.
RHS = laju masukan massa (g/d).

Metode: Gauss-Seidel
- Sistem ini diagonal dominan → langsung konvergen.
- Update berurutan: c1 → c2 → c3 → c1 → ... dst.
"""

import numpy as np

A = np.array([
    [15, -3,  -1],
    [-3, 18,  -6],
    [-4, -1,  12]
], dtype=float)

b = np.array([3800.0, 1200.0, 2350.0])
n = len(b)
es = 5.0

print("=" * 60)
print("Gauss-Seidel — Konsentrasi Reaktor (Soal 11.9)")
print("=" * 60)

# Cek diagonal dominan
print("\nCek diagonal dominan:")
for i in range(n):
    off = sum(abs(A[i,j]) for j in range(n) if j != i)
    ok = "✓" if abs(A[i,i]) >= off else "✗"
    print(f"  Baris {i}: |{A[i,i]}| >= {off}? {ok}")

c = np.zeros(n)
print(f"\n{'Iter':>4} | {'c1':>10} | {'c2':>10} | {'c3':>10} | {'ea_max':>10}")
print("-" * 55)

for it in range(1, 100):
    c_old = c.copy()

    c[0] = (b[0] - A[0,1]*c[1] - A[0,2]*c[2]) / A[0,0]
    c[1] = (b[1] - A[1,0]*c[0] - A[1,2]*c[2]) / A[1,1]
    c[2] = (b[2] - A[2,0]*c[0] - A[2,1]*c[1]) / A[2,2]

    ea = [abs((c[i]-c_old[i])/c[i])*100 if c[i] != 0 else 0 for i in range(n)]
    ea_max = max(ea)
    print(f"  {it:>3} | {c[0]:>10.4f} | {c[1]:>10.4f} | {c[2]:>10.4f} | {ea_max:>9.4f}%")

    if ea_max < es and it > 1:
        print(f"\n  Konvergen pada iterasi ke-{it}!")
        break

print("\n" + "=" * 60)
print("Hasil:")
labels = ["c1", "c2", "c3"]
for label, ci in zip(labels, c):
    print(f"  {label} = {ci:.4f} g/m³")

# Verifikasi
print(f"\nVerifikasi A @ c = {A @ c}")
print(f"          b     = {b}")

x_eksak = np.linalg.solve(A, b)
print(f"\nSolusi eksak: {x_eksak}")