"""
=============================================================
SOAL 11.18 - Produksi Elektronik (Sistem Persamaan Linear)
=============================================================

PERMASALAHAN:
Sebuah perusahaan elektronik memproduksi transistor, resistor,
dan chip komputer. Kebutuhan material per komponen:

           Tembaga  Seng  Kaca
Transistor    4       1     2
Resistor      3       3     1
Chip          2       1     3

Material tersedia minggu ini:
  - Tembaga : 960 unit
  - Seng    : 510 unit
  - Kaca    : 610 unit

Tentukan jumlah transistor (t), resistor (r), dan chip (c)
yang harus diproduksi.

METODE:
Sistem persamaan linear:
    4t + 3r + 2c = 960   (tembaga)
    1t + 3r + 1c = 510   (seng)
    2t + 1r + 3c = 610   (kaca)

Diselesaikan dengan eliminasi Gauss dan numpy.linalg.solve
=============================================================
"""

import numpy as np

print("=" * 60)
print("SOAL 11.18 - Produksi Elektronik")
print("=" * 60)

# ─── Setup matriks koefisien A dan vektor b ───────────────
#       t    r    c
A = np.array([
    [4,   3,   2],   # tembaga
    [1,   3,   1],   # seng
    [2,   1,   3],   # kaca
], dtype=float)

b = np.array([960, 510, 610], dtype=float)

print("\nSistem persamaan:")
print("  4t + 3r + 2c = 960  (tembaga)")
print("  1t + 3r + 1c = 510  (seng)")
print("  2t + 1r + 3c = 610  (kaca)")

print("\nMatriks [A]:")
labels = ['Tembaga', 'Seng   ', 'Kaca   ']
for i, row in enumerate(A):
    print(f"  {labels[i]}: {row}")

print(f"\nVektor b = {b}")

# ─── METODE 1: Eliminasi Gauss manual ────────────────────
print("\n" + "=" * 60)
print("METODE 1: Eliminasi Gauss (Forward + Back Substitution)")
print("=" * 60)

def gauss_elimination(A, b):
    n = len(b)
    # Buat augmented matrix
    Aug = np.hstack([A.copy(), b.reshape(-1,1).copy()])
    print("\n  Augmented matrix awal:")
    for row in Aug:
        print("   ", [f"{v:8.4f}" for v in row])
    
    # Forward elimination
    for k in range(n):
        # Partial pivoting
        max_row = np.argmax(abs(Aug[k:, k])) + k
        Aug[[k, max_row]] = Aug[[max_row, k]]
        
        for i in range(k+1, n):
            if Aug[k, k] == 0:
                continue
            factor = Aug[i, k] / Aug[k, k]
            Aug[i, :] -= factor * Aug[k, :]
            print(f"  Eliminasi baris {i+1} dengan faktor {factor:.4f}")
    
    print("\n  Upper triangular matrix:")
    for row in Aug:
        print("   ", [f"{v:8.4f}" for v in row])
    
    # Back substitution
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (Aug[i, -1] - np.dot(Aug[i, i+1:n], x[i+1:n])) / Aug[i, i]
    
    return x

x_gauss = gauss_elimination(A.copy(), b.copy())

print("\n  Hasil Back Substitution:")
print(f"    t (transistor) = {x_gauss[0]:.4f}")
print(f"    r (resistor)   = {x_gauss[1]:.4f}")
print(f"    c (chip)       = {x_gauss[2]:.4f}")

# ─── METODE 2: numpy.linalg.solve ─────────────────────────
print("\n" + "=" * 60)
print("METODE 2: numpy.linalg.solve")
print("=" * 60)

x = np.linalg.solve(A, b)
print(f"\n  t (transistor) = {x[0]:.4f}")
print(f"  r (resistor)   = {x[1]:.4f}")
print(f"  c (chip)       = {x[2]:.4f}")

# ─── Verifikasi ───────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFIKASI:")
print("=" * 60)
material = ['Tembaga', 'Seng   ', 'Kaca   ']
for i in range(3):
    hasil = np.dot(A[i], x)
    print(f"  {material[i]}: {A[i,0]:.0f}×{x[0]:.0f} + {A[i,1]:.0f}×{x[1]:.0f} + {A[i,2]:.0f}×{x[2]:.0f}"
          f" = {hasil:.1f}  (seharusnya {b[i]:.0f}) {'✓' if abs(hasil-b[i])<0.01 else '✗'}")

# ─── Hitung invers matriks ────────────────────────────────
print("\n" + "=" * 60)
print("INVERS MATRIKS [A]:")
print("=" * 60)
A_inv = np.linalg.inv(A)
print(np.array2string(A_inv, precision=6, suppress_small=True))

# ─── Condition number ────────────────────────────────────
cond = np.linalg.cond(A, np.inf)  # row-sum norm
print(f"\nCondition Number (row-sum norm): {cond:.4f}")
print("(Nilai < 1000 menunjukkan sistem terkondisi baik)")

print("\n" + "=" * 60)
print("KESIMPULAN:")
print("=" * 60)
print(f"  Transistor yang diproduksi: {round(x[0])} unit")
print(f"  Resistor yang diproduksi  : {round(x[1])} unit")
print(f"  Chip yang diproduksi      : {round(x[2])} unit")