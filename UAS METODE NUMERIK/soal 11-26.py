"""
=============================================================
SOAL 11.26 - Program Metode Gauss-Seidel
=============================================================

PERMASALAHAN:
Buat program user-friendly untuk Metode Gauss-Seidel (dengan
opsi over-relaxation).

Uji dengan menyelesaikan kembali Example 11.3 dari buku:
  3x₁ - 0.1x₂ - 0.2x₃ =  7.85
  0.1x₁ + 7x₂ - 0.3x₃ = -19.3
  0.3x₁ - 0.2x₂ + 10x₃ = 71.4

Solusi eksak: x₁=3, x₂=-2.5, x₃=7

METODE GAUSS-SEIDEL:
Untuk persamaan ke-i: 
  xᵢ^(k+1) = (bᵢ - Σⱼ₍ⱼ≠ᵢ₎ aᵢⱼxⱼ) / aᵢᵢ

Dengan over-relaxation (faktor λ):
  xᵢ^(k+1) = λ·xᵢ^(k+1) + (1-λ)·xᵢ^(k)

Konvergensi dijamin bila matriks diagonally dominant:
  |aᵢᵢ| > Σⱼ₍ⱼ≠ᵢ₎ |aᵢⱼ|  untuk semua i
=============================================================
"""

import numpy as np

# ─── Implementasi Gauss-Seidel ────────────────────────────
def gauss_seidel(A, b, x0=None, tol=0.05, max_iter=100,
                 lam=1.0, verbose=True):
    """
    Menyelesaikan [A]{x} = {b} dengan metode Gauss-Seidel.
    
    Parameter:
        A       : matriks koefisien (n×n)
        b       : vektor RHS
        x0      : tebakan awal (default: semua nol)
        tol     : toleransi error relatif (%) untuk konvergensi
        max_iter: jumlah iterasi maksimum
        lam     : faktor relaxation (1.0 = tanpa relaxation)
        verbose : tampilkan detail tiap iterasi
    
    Return:
        x     : vektor solusi
        iters : jumlah iterasi yang dilakukan
        errors: list error relatif tiap iterasi
    """
    n = len(b)
    x = np.zeros(n) if x0 is None else x0.copy().astype(float)
    errors = []
    
    # Cek diagonal dominance
    for i in range(n):
        row_sum = sum(abs(A[i, j]) for j in range(n) if j != i)
        if abs(A[i, i]) < row_sum:
            if verbose:
                print(f"  [!] Baris {i+1}: tidak diagonally dominant "
                      f"(|{A[i,i]}| < {row_sum:.2f}), konvergensi tidak dijamin!")
    
    if verbose:
        header = f"{'Iter':>5}  " + "  ".join([f"{'x'+str(i+1):>12}" for i in range(n)])
        header += f"  {'Max Error %':>12}"
        print(header)
        print("-" * (len(header) + 5))
        init_row = f"{'0':>5}  " + "  ".join([f"{xi:>12.6f}" for xi in x])
        print(init_row + "  (initial)")
    
    for iteration in range(1, max_iter + 1):
        x_old = x.copy()
        
        for i in range(n):
            # Hitung nilai baru x[i]
            sigma = sum(A[i, j] * x[j] for j in range(n) if j != i)
            x_new_i = (b[i] - sigma) / A[i, i]
            
            # Terapkan relaxation
            x[i] = lam * x_new_i + (1 - lam) * x_old[i]
        
        # Hitung error relatif
        err = []
        for i in range(n):
            if abs(x[i]) > 1e-12:
                err.append(abs((x[i] - x_old[i]) / x[i]) * 100)
            else:
                err.append(0)
        max_err = max(err)
        errors.append(max_err)
        
        if verbose:
            row = f"{iteration:>5}  " + "  ".join([f"{xi:>12.6f}" for xi in x])
            row += f"  {max_err:>12.4f}%"
            print(row)
        
        # Cek konvergensi
        if max_err < tol:
            if verbose:
                print(f"\n  [✓] Konvergen dalam {iteration} iterasi!")
            return x, iteration, errors
    
    if verbose:
        print(f"\n  [!] Tidak konvergen dalam {max_iter} iterasi.")
    return x, max_iter, errors

# ─── Program utama ────────────────────────────────────────
print("=" * 60)
print("SOAL 11.26 - Metode Gauss-Seidel")
print("=" * 60)

# ─── Example 11.3 ────────────────────────────────────────
print("""
Example 11.3 (dari buku):
  3x₁  - 0.1x₂ - 0.2x₃ =  7.85
  0.1x₁ + 7x₂  - 0.3x₃ = -19.3
  0.3x₁ - 0.2x₂ + 10x₃ =  71.4

Solusi eksak: x₁=3, x₂=-2.5, x₃=7
""")

A = np.array([
    [ 3.0, -0.1, -0.2],
    [ 0.1,  7.0, -0.3],
    [ 0.3, -0.2, 10.0],
])
b = np.array([7.85, -19.3, 71.4])
x_exact = np.array([3.0, -2.5, 7.0])

# Cek diagonal dominance
print("CEK DIAGONAL DOMINANCE:")
for i in range(3):
    diag = abs(A[i, i])
    off = sum(abs(A[i, j]) for j in range(3) if j != i)
    status = "✓" if diag > off else "✗"
    print(f"  Baris {i+1}: |{A[i,i]}| = {diag} > {off:.1f} {status}")

# ─── (a) Tanpa relaxation ─────────────────────────────────
print("\n" + "=" * 60)
print("(a) TANPA RELAXATION (λ = 1.0), toleransi εₛ = 5%")
print("=" * 60)

x_sol, n_iter, errs = gauss_seidel(A, b, tol=5.0, lam=1.0)

print("\nHasil:")
for i in range(3):
    err_true = abs(x_sol[i] - x_exact[i]) / abs(x_exact[i]) * 100
    print(f"  x{i+1} = {x_sol[i]:.6f}  (eksak: {x_exact[i]}, error: {err_true:.4f}%)")

# ─── Soal 11.9: Konsentrasi reaktor ──────────────────────
print("\n" + "=" * 60)
print("SOAL 11.9 - Konsentrasi Reaktor (uji tambahan)")
print("=" * 60)
print("""
  15c₁ -  3c₂ -   c₃ = 3800
  -3c₁ + 18c₂ -  6c₃ = 1200
  -4c₁ -   c₂ + 12c₃ = 2350
""")

A9 = np.array([
    [15, -3,  -1],
    [-3, 18,  -6],
    [-4, -1,  12],
], dtype=float)
b9 = np.array([3800, 1200, 2350], dtype=float)

x9, n9, e9 = gauss_seidel(A9, b9, tol=5.0, lam=1.0)
print(f"\nHasil:")
for i in range(3):
    print(f"  c{i+1} = {x9[i]:.4f} g/m³")

# Verifikasi
print("\nVerifikasi [A]{x} = {b}:")
Ax9 = np.dot(A9, x9)
for i in range(3):
    ok = '✓' if abs(Ax9[i] - b9[i]) / abs(b9[i]) * 100 < 5 else '~'
    print(f"  Baris {i+1}: {Ax9[i]:.2f} ≈ {b9[i]:.2f} {ok}")

print("\n" + "=" * 60)
print("RINGKASAN:")
print("=" * 60)
print("""
  Gauss-Seidel menggunakan nilai TERBARU yang tersedia
  (tidak seperti Jacobi yang menggunakan semua nilai lama).
  
  Relaxation λ > 1: over-relaxation (percepat konvergensi)
  Relaxation λ < 1: under-relaxation (stabilkan divergen)
  Relaxation λ = 1: Gauss-Seidel standard
  
  Syarat konvergensi: Matriks diagonally dominant
""")