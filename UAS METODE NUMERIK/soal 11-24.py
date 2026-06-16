"""
=============================================================
SOAL 11.24 - Program Algoritma Thomas (Tridiagonal)
=============================================================

PERMASALAHAN:
Buat program user-friendly untuk menyelesaikan sistem tridiagonal
menggunakan Algoritma Thomas (TDMA - TriDiagonal Matrix Algorithm).

Uji program dengan menyelesaikan kembali hasil Example 11.1:
  [ 2.04  -1              ] [T₁]   [40.8]
  [-1    2.04  -1         ] [T₂]   [ 0.8]
  [      -1   2.04  -1   ] [T₃] = [ 0.8]
  [           -1   2.04  ] [T₄]   [200.8]

ALGORITMA THOMAS:
Untuk sistem tridiagonal:
  aᵢxᵢ₋₁ + bᵢxᵢ + cᵢxᵢ₊₁ = dᵢ

Forward sweep (dekomposisi):
  Untuk i = 2..n:
    w = aᵢ / b'ᵢ₋₁
    b'ᵢ = bᵢ - w·cᵢ₋₁
    d'ᵢ = dᵢ - w·d'ᵢ₋₁

Back substitution:
  xₙ = d'ₙ / b'ₙ
  Untuk i = n-1..1:
    xᵢ = (d'ᵢ - cᵢ·xᵢ₊₁) / b'ᵢ
=============================================================
"""

import numpy as np

# ─── Implementasi Algoritma Thomas ────────────────────────
def thomas_algorithm(a, b, c, d):
    """
    Menyelesaikan sistem tridiagonal dengan Algoritma Thomas.
    
    Parameter:
        a : sub-diagonal (a[0] tidak digunakan, a[1..n-1])
        b : diagonal utama (b[0..n-1])
        c : super-diagonal (c[0..n-2], c[n-1] tidak digunakan)
        d : vektor RHS (d[0..n-1])
    
    Return:
        x : vektor solusi
    
    Semua input berukuran n.
    """
    n = len(d)
    
    # Salin agar tidak memodifikasi input
    b_prime = b.copy().astype(float)
    d_prime = d.copy().astype(float)
    x = np.zeros(n)
    
    print("\n  FORWARD SWEEP:")
    print(f"  {'Langkah':>8}  {'b_prime':>12}  {'d_prime':>12}  {'faktor w':>12}")
    print("  " + "-" * 50)
    print(f"  {'i=1':>8}  {b_prime[0]:>12.6f}  {d_prime[0]:>12.6f}  {'(init)':>12}")
    
    # Forward sweep
    for i in range(1, n):
        w = a[i] / b_prime[i-1]
        b_prime[i] = b[i] - w * c[i-1]
        d_prime[i] = d[i] - w * d_prime[i-1]
        print(f"  {'i='+str(i+1):>8}  {b_prime[i]:>12.6f}  {d_prime[i]:>12.6f}  {w:>12.6f}")
    
    # Back substitution
    x[n-1] = d_prime[n-1] / b_prime[n-1]
    
    print("\n  BACK SUBSTITUTION:")
    print(f"  {'Langkah':>8}  {'x':>12}")
    print("  " + "-" * 24)
    print(f"  {'i='+str(n):>8}  {x[n-1]:>12.6f}")
    
    for i in range(n-2, -1, -1):
        x[i] = (d_prime[i] - c[i] * x[i+1]) / b_prime[i]
        print(f"  {'i='+str(i+1):>8}  {x[i]:>12.6f}")
    
    return x

# ─── Program User-Friendly ───────────────────────────────
def solve_tridiagonal_interactive():
    """
    Interface user-friendly untuk menyelesaikan sistem tridiagonal.
    """
    print("=" * 60)
    print("PROGRAM PENYELESAI SISTEM TRIDIAGONAL")
    print("Algoritma Thomas (TDMA)")
    print("=" * 60)
    print("""
Sistem yang akan diselesaikan berbentuk:
  b₁x₁ + c₁x₂                          = d₁
  a₂x₁ + b₂x₂ + c₂x₃                  = d₂
        a₃x₂ + b₃x₃ + c₃x₄            = d₃
              ...
              aₙxₙ₋₁ + bₙxₙ            = dₙ
""")

# ─── UJI dengan Example 11.1 ──────────────────────────────
print("=" * 60)
print("UJI PROGRAM: Example 11.1 dari buku")
print("=" * 60)
print("""
Sistem:
  [ 2.04  -1.00              ] [T₁]   [ 40.8]
  [-1.00  2.04  -1.00       ] [T₂]   [  0.8]
  [      -1.00  2.04  -1.00 ] [T₃] = [  0.8]
  [            -1.00  2.04  ] [T₄]   [200.8]
""")

n = 4
# Sub-diagonal a (a[0] tidak dipakai)
a = np.array([0,    -1.00, -1.00, -1.00])
# Diagonal utama b
b = np.array([2.04,  2.04,  2.04,  2.04])
# Super-diagonal c (c[n-1] tidak dipakai)
c = np.array([-1.00, -1.00, -1.00, 0])
# RHS
d = np.array([40.8, 0.8, 0.8, 200.8])

solve_tridiagonal_interactive()

x = thomas_algorithm(a, b, c, d)

print("\n" + "=" * 60)
print("HASIL:")
print("=" * 60)
for i, xi in enumerate(x):
    print(f"  T{i+1} = {xi:.6f}")

# ─── Verifikasi ───────────────────────────────────────────
print("\nVERIFIKASI [A]{x} = {b}:")
A_full = np.array([
    [2.04, -1.00,  0.00,  0.00],
    [-1.00, 2.04, -1.00,  0.00],
    [0.00, -1.00,  2.04, -1.00],
    [0.00,  0.00, -1.00,  2.04],
])
Ax = np.dot(A_full, x)
for i in range(n):
    ok = '✓' if abs(Ax[i] - d[i]) < 1e-6 else '✗'
    print(f"  Baris {i+1}: {Ax[i]:.6f} = {d[i]:.6f} {ok}")

# ─── Uji tambahan: soal 11.1 ──────────────────────────────
print("\n" + "=" * 60)
print("UJI TAMBAHAN: Soal 11.1 (tridiagonal 3x3)")
print("=" * 60)
print("""
  [ 0.8  -0.4        ] [x₁]   [ 41 ]
  [-0.4   0.8  -0.4  ] [x₂] = [ 25 ]
  [      -0.4   0.8  ] [x₃]   [105 ]
""")

a2 = np.array([0,    -0.4, -0.4])
b2 = np.array([0.8,   0.8,  0.8])
c2 = np.array([-0.4, -0.4,  0])
d2 = np.array([41,   25,   105])

x2 = thomas_algorithm(a2, b2, c2, d2)
print("\nHasil:")
for i, xi in enumerate(x2):
    print(f"  x{i+1} = {xi:.6f}")