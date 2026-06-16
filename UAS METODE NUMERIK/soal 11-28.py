"""
=============================================================
SOAL 11.28 - Program Solver Sistem Pentadiagonal
=============================================================

PERMASALAHAN:
Sistem pentadiagonal (bandwidth 5) berbentuk:
  fᵢxᵢ + gᵢxᵢ₊₁ + hᵢxᵢ₊₂ + eᵢxᵢ₋₁ + dᵢxᵢ₋₂ = rᵢ

Struktur matriks:
  | f₁  g₁  h₁              |
  | e₂  f₂  g₂  h₂          |
  | d₃  e₃  f₃  g₃  h₃      |
  |     d₄  e₄  f₄  g₄  h₄  |
  |         ...              |

Uji dengan sistem:
  |  8  -2  -1   0   0 | [x₁]   [ 5]
  | -2   9  -4  -1   0 | [x₂]   [ 2]
  | -1  -3   7  -1  -2 | [x₃] = [ 0]
  |  0  -4  -2  12  -5 | [x₄]   [ 1]
  |  0   0  -7  -3  15 | [x₅]   [ 5]

METODE:
Eliminasi Gauss tanpa pivoting (memanfaatkan struktur sparse).
Setiap baris hanya mempengaruhi maksimal 2 baris di bawahnya.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# ─── Implementasi Solver Pentadiagonal ───────────────────
def pentadiagonal_solver(d, e, f, g, h, r, verbose=True):
    """
    Menyelesaikan sistem pentadiagonal tanpa pivoting.
    
    Parameter (semuanya array ukuran n):
        d : diagonal ke-2 bawah  (d[0], d[1] tidak digunakan)
        e : diagonal ke-1 bawah  (e[0] tidak digunakan)
        f : diagonal utama
        g : diagonal ke-1 atas   (g[n-1] tidak digunakan)
        h : diagonal ke-2 atas   (h[n-1], h[n-2] tidak digunakan)
        r : vektor RHS
    
    Return:
        x : vektor solusi
    """
    n = len(r)
    
    # Salin semua array agar tidak memodifikasi input
    d_ = d.copy().astype(float)
    e_ = e.copy().astype(float)
    f_ = f.copy().astype(float)
    g_ = g.copy().astype(float)
    h_ = h.copy().astype(float)
    r_ = r.copy().astype(float)
    
    if verbose:
        print("\n  FORWARD ELIMINATION:")
        print("  " + "-" * 50)
    
    # Forward elimination
    for k in range(n):
        pivot = f_[k]
        if abs(pivot) < 1e-12:
            raise ValueError(f"Pivot nol pada baris {k+1}! Butuh pivoting.")
        
        # Eliminasi baris k+1 (menggunakan e[k+1])
        if k + 1 < n:
            factor1 = e_[k+1] / pivot
            e_[k+1] = 0
            f_[k+1]  -= factor1 * g_[k]
            g_[k+1]  -= factor1 * h_[k]   if k+1 < n else 0
            r_[k+1]  -= factor1 * r_[k]
            if verbose:
                print(f"  Eliminasi baris {k+2} (via baris {k+1}): faktor = {factor1:.6f}")
        
        # Eliminasi baris k+2 (menggunakan d[k+2])
        if k + 2 < n:
            factor2 = d_[k+2] / pivot
            d_[k+2] = 0
            e_[k+2]  -= factor2 * g_[k]
            f_[k+2]  -= factor2 * h_[k]
            r_[k+2]  -= factor2 * r_[k]
            if verbose:
                print(f"  Eliminasi baris {k+3} (via baris {k+1}): faktor = {factor2:.6f}")
    
    # Back substitution
    x = np.zeros(n)
    x[n-1] = r_[n-1] / f_[n-1]
    
    if n >= 2:
        x[n-2] = (r_[n-2] - g_[n-2] * x[n-1]) / f_[n-2]
    
    for i in range(n-3, -1, -1):
        x[i] = (r_[i] - g_[i] * x[i+1] - h_[i] * x[i+2]) / f_[i]
    
    return x

# ─── Bangun matriks penuh untuk verifikasi ────────────────
def build_full_matrix(d, e, f, g, h):
    n = len(f)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i] = f[i]
        if i > 0:    A[i, i-1] = e[i]
        if i > 1:    A[i, i-2] = d[i]
        if i < n-1:  A[i, i+1] = g[i]
        if i < n-2:  A[i, i+2] = h[i]
    return A

# ─── Program utama ────────────────────────────────────────
print("=" * 60)
print("SOAL 11.28 - Solver Sistem Pentadiagonal")
print("=" * 60)

print("""
Matriks pentadiagonal yang diselesaikan:
  |  8  -2  -1   0   0 | [x₁]   [ 5]
  | -2   9  -4  -1   0 | [x₂]   [ 2]
  | -1  -3   7  -1  -2 | [x₃] = [ 0]
  |  0  -4  -2  12  -5 | [x₄]   [ 1]
  |  0   0  -7  -3  15 | [x₅]   [ 5]
""")

n = 5

# Diagonal ke-2 bawah (d[0], d[1] tidak digunakan)
d = np.array([0,  0, -1,  0,  0], dtype=float)
# Diagonal ke-1 bawah (e[0] tidak digunakan)
e = np.array([0, -2, -3, -4,  0], dtype=float)
# Diagonal utama
f = np.array([8,  9,  7, 12, 15], dtype=float)
# Diagonal ke-1 atas (g[n-1] tidak digunakan)
g = np.array([-2, -4, -1, -3,  0], dtype=float)
# Diagonal ke-2 atas (h[n-2], h[n-1] tidak digunakan)
h = np.array([-1, -1, -2,  0,  0], dtype=float)
# RHS
r = np.array([5,  2,  0,  1,  5], dtype=float)

# Bangun matriks penuh untuk verifikasi
A_full = build_full_matrix(d, e, f, g, h)
print("Matriks A (full):")
print(np.array2string(A_full.astype(int), precision=0))

# ─── Selesaikan dengan pentadiagonal solver ───────────────
print("\n" + "=" * 60)
print("PENYELESAIAN dengan Pentadiagonal Solver:")
print("=" * 60)

x = pentadiagonal_solver(d, e, f, g, h, r)

print("\nHasil:")
for i, xi in enumerate(x):
    print(f"  x{i+1} = {xi:.8f}")

# ─── Verifikasi dengan numpy ──────────────────────────────
x_np = np.linalg.solve(A_full, r)
print("\nVerifikasi dengan numpy.linalg.solve:")
for i, xi in enumerate(x_np):
    print(f"  x{i+1} = {xi:.8f}")

print("\nKonfirmasi Ax = r:")
Ax = np.dot(A_full, x)
for i in range(n):
    ok = '✓' if abs(Ax[i] - r[i]) < 1e-6 else '✗'
    print(f"  Baris {i+1}: {Ax[i]:.4f} = {r[i]:.4f} {ok}")

err = np.linalg.norm(x - x_np)
print(f"\nError terhadap numpy: {err:.2e}")

# ─── Visualisasi struktur matriks ────────────────────────
print("\nMembuat visualisasi struktur matriks...")

fig, ax = plt.subplots(figsize=(6, 6))
mask = A_full != 0
colors = np.where(mask, np.abs(A_full), np.nan)

im = ax.imshow(A_full, cmap='coolwarm', aspect='equal')
for i in range(n):
    for j in range(n):
        val = int(A_full[i, j])
        ax.text(j, i, str(val), ha='center', va='center',
                fontsize=12, fontweight='bold',
                color='white' if abs(A_full[i,j]) > 5 else 'black')

plt.colorbar(im, ax=ax)
ax.set_title('Struktur Matriks Pentadiagonal\n(bandwidth = 5)')
ax.set_xlabel('Kolom j')
ax.set_ylabel('Baris i')
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([f'x{i+1}' for i in range(n)])
ax.set_yticklabels([f'r{i+1}' for i in range(n)])
plt.tight_layout()
plt.savefig('plot_11_28.png', dpi=120)
plt.show()
print("Plot disimpan: plot_11_28.png")

print("\n" + "=" * 60)
print("RINGKASAN:")
print("=" * 60)
print("""
  Sistem pentadiagonal memiliki bandwidth = 5
  (maksimal 2 elemen di kiri dan kanan diagonal utama).
  
  Keuntungan solver pentadiagonal:
  - Hanya menyimpan 5 diagonal (bukan matriks n×n penuh)
  - Forward elimination hanya perlu 2 operasi per baris
  - Kompleksitas O(n) bukan O(n³)
  
  Perbedaan dengan tridiagonal (3-band):
  - Tiap baris eliminasi mempengaruhi 2 baris di bawahnya
  - Back substitution butuh 2 elemen sebelumnya (xᵢ₊₁ dan xᵢ₊₂)
""")