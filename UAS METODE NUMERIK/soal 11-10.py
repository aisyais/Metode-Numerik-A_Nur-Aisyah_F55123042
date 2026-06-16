"""
SOAL 11.10
==========
Ulangi Soal 11.9 tetapi gunakan Jacobi Iteration.

Sistem (sama dengan 11.9):
 15c1 -  3c2 -   c3 = 3800
 -3c1 + 18c2 -  6c3 = 1200
 -4c1 -   c2 + 12c3 = 2350

Metode: Jacobi Iteration
- Berbeda dari Gauss-Seidel: semua variabel diupdate SERENTAK
  menggunakan nilai dari iterasi SEBELUMNYA.
- Formula: x_i^(k+1) = (b_i - sum(A_ij * x_j^(k), j≠i)) / A_ii
- Lebih lambat dari Gauss-Seidel karena tidak langsung memakai
  nilai baru yang baru dihitung.
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
print("Jacobi Iteration — Konsentrasi Reaktor (Soal 11.10)")
print("=" * 60)

print("\nPerbedaan Jacobi vs Gauss-Seidel:")
print("  Gauss-Seidel: gunakan nilai c_i terbaru segera")
print("  Jacobi      : simpan semua nilai lama, update serentak\n")

c = np.zeros(n)
print(f"{'Iter':>4} | {'c1':>10} | {'c2':>10} | {'c3':>10} | {'ea_max':>10}")
print("-" * 55)

for it in range(1, 200):
    c_old = c.copy()

    # Update SERENTAK — semua pakai c_old
    c_new = np.zeros(n)
    for i in range(n):
        sigma = sum(A[i,j] * c_old[j] for j in range(n) if j != i)
        c_new[i] = (b[i] - sigma) / A[i,i]

    c = c_new

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

x_eksak = np.linalg.solve(A, b)
print(f"\nSolusi eksak: c1={x_eksak[0]:.4f}, c2={x_eksak[1]:.4f}, c3={x_eksak[2]:.4f}")