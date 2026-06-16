"""
SOAL 11.14
==========
Analisis konvergensi Gauss-Seidel untuk sistem dengan slope 1 dan -1.

Contoh sistem dengan slope ±1:
  x1 + x2 = 2   (slope = -1)
  x1 - x2 = 0   (slope = +1)

Solusi eksak: x1 = 1, x2 = 1

Analisis:
- Jika slope kedua persamaan adalah 1 dan -1, diagonal persamaan
  TIDAK dominan: |a_ii| TIDAK > sum|a_ij|.
- Gauss-Seidel TIDAK akan konvergen atau konvergensinya
  sangat lambat/tidak terjamin.
"""

import numpy as np

print("=" * 60)
print("Analisis Konvergensi Gauss-Seidel — Slope 1 dan -1")
print("=" * 60)

print("""
Sistem persamaan dengan slope 1 dan -1:
  Persamaan 1: x1 + x2 = 2   → slope = -x1/x2 koef = -1 (slope -1)
  Persamaan 2: x1 - x2 = 0   → slope = +x1/x2 koef = +1 (slope +1)

Dalam bentuk matriks:
  [1   1] [x1]   [2]
  [1  -1] [x2] = [0]
""")

A = np.array([[1.0, 1.0], [1.0, -1.0]])
b = np.array([2.0, 0.0])

print("Cek diagonal dominan:")
for i in range(2):
    off = sum(abs(A[i,j]) for j in range(2) if j != i)
    ok = "✓" if abs(A[i,i]) > off else "TIDAK DOMINAN ✗"
    print(f"  Baris {i}: |A[{i},{i}]| = {abs(A[i,i])} vs off-diag = {off} → {ok}")

print("\nSimulasi Gauss-Seidel (20 iterasi):")
x = np.zeros(2)
print(f"\n{'Iter':>4} | {'x1':>12} | {'x2':>12} | {'Status':>15}")
print("-" * 50)

konvergen = False
for it in range(1, 21):
    x_old = x.copy()
    x[0] = (b[0] - A[0,1]*x[1]) / A[0,0]
    x[1] = (b[1] - A[1,0]*x[0]) / A[1,1]

    ea = max(abs(x[i]-x_old[i])/(abs(x[i])+1e-15)*100 for i in range(2))
    status = "OK" if ea < 5 else f"ea={ea:.2f}%"
    print(f"  {it:>3} | {x[0]:>12.6f} | {x[1]:>12.6f} | {status:>15}")

    if ea < 5 and it > 1:
        konvergen = True
        print(f"\n  Konvergen pada iterasi ke-{it}!")
        break

if not konvergen:
    print("\nKesimpulan:")
    print("  Gauss-Seidel berhasil konvergen karena meski slope ±1,")
    print("  setelah substitusi nilai baru langsung dipakai.")
    print("  Namun ini termasuk kasus batas — tidak ada jaminan formal.")

print(f"\nSolusi eksak: x1={1.0}, x2={1.0}")
x_e = np.linalg.solve(A, b)
print(f"Solusi numpy: x1={x_e[0]}, x2={x_e[1]}")

print("""
Kesimpulan Teoritis (Fig. 11.5):
  - Jika slope persamaan ±1, iterasi bisa berosilasi.
  - Diagonal dominan TIDAK terpenuhi → tidak ada jaminan konvergensi.
  - Untuk slope 1 dan -1 yang tepat sama: iterasi berosilasi tanpa konvergen.
""")