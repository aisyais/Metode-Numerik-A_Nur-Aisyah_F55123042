"""
SOAL 11.15
==========
Identifikasi set persamaan yang TIDAK dapat diselesaikan
dengan metode iteratif Gauss-Seidel dan tunjukkan tidak konvergen.

Set One:   8x + 3y +  z = 12    Set Two:   x +  y + 5z = 7
          -6x      + 7z =  1              x + 4y -  z = 4
           2x + 4y -  z =  5             3x +  y -  z = 4

Set Three: 2x + 3y + 5z = 7
          -2x + 4y - 5z = -3
               2y -   z =  1

Kriteria konvergensi: diagonal dominan
  |a_ii| > sum(|a_ij|) untuk j ≠ i
Jika tidak terpenuhi → kemungkinan tidak konvergen.
"""

import numpy as np

sets = {
    "Set One": {
        "A": np.array([[ 8,  3,  1],
                       [-6,  0,  7],
                       [ 2,  4, -1]], dtype=float),
        "b": np.array([12.0, 1.0, 5.0])
    },
    "Set Two": {
        "A": np.array([[1,  1,  5],
                       [1,  4, -1],
                       [3,  1, -1]], dtype=float),
        "b": np.array([7.0, 4.0, 4.0])
    },
    "Set Three": {
        "A": np.array([[ 2,  3,  5],
                       [-2,  4, -5],
                       [ 0,  2, -1]], dtype=float),
        "b": np.array([7.0, -3.0, 1.0])
    }
}

print("=" * 65)
print("Identifikasi Konvergensi Gauss-Seidel (Soal 11.15)")
print("=" * 65)

for nama, data in sets.items():
    A = data["A"]
    b = data["b"]
    n = len(b)

    print(f"\n{'='*65}")
    print(f"{nama}")
    print("="*65)
    print("\nCek diagonal dominan:")
    dominan = True
    for i in range(n):
        off = sum(abs(A[i,j]) for j in range(n) if j != i)
        ok = abs(A[i,i]) >= off
        if not ok:
            dominan = False
        print(f"  Baris {i}: |{A[i,i]}| vs {off} → {'✓' if ok else '✗ TIDAK DOMINAN'}")

    print(f"\nKesimpulan diagonal dominan: {'Ya' if dominan else 'TIDAK — risiko tidak konvergen'}")

    # Coba 10 iterasi
    x = np.zeros(n)
    print(f"\n{'Iter':>4} | {'x1':>12} | {'x2':>12} | {'x3':>12}")
    print("-" * 50)

    konvergen = False
    for it in range(1, 11):
        x_old = x.copy()
        for i in range(n):
            sigma = sum(A[i,j]*x[j] for j in range(n) if j != i)
            if A[i,i] != 0:
                x[i] = (b[i] - sigma) / A[i,i]

        print(f"  {it:>3} | {x[0]:>12.4f} | {x[1]:>12.4f} | {x[2]:>12.4f}")

        ea = [abs((x[i]-x_old[i])/(x[i]+1e-15))*100 for i in range(n)]
        if max(ea) < 5 and it > 1:
            konvergen = True
            print(f"  → Konvergen pada iterasi {it}")
            break

    if not konvergen:
        print(f"  → Tidak konvergen dalam 10 iterasi")

    # Solusi eksak jika ada
    try:
        x_e = np.linalg.solve(A, b)
        print(f"  Solusi eksak: {x_e}")
    except np.linalg.LinAlgError:
        print("  Sistem singular!")