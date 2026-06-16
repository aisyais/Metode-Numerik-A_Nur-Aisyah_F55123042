"""
=============================================================
SOAL 11.25 - Program Dekomposisi Cholesky
=============================================================

PERMASALAHAN:
Buat program untuk Dekomposisi Cholesky pada matriks simetris
positif definit.

Uji dengan Example 11.2 dari buku:
  [A] = | 6   15   55  |
        | 15  55   225 |
        | 55  225  979 |

DEKOMPOSISI CHOLESKY:
Untuk matriks A simetris positif definit:
  [A] = [L][L]ᵀ

Dimana L adalah matriks lower triangular:
  L[j,j] = sqrt(A[j,j] - Σₖ L[j,k]²)     (diagonal)
  L[i,j] = (A[i,j] - Σₖ L[i,k]·L[j,k]) / L[j,j]  (off-diagonal)

Setelah L diperoleh, selesaikan [A]{x} = {b} dengan:
  [L]{y} = {b}   (forward substitution)
  [L]ᵀ{x} = {y}  (back substitution)
=============================================================
"""

import numpy as np

# ─── Implementasi Cholesky ────────────────────────────────
def cholesky_decomposition(A, verbose=True):
    """
    Dekomposisi Cholesky: [A] = [L][L]ᵀ
    
    Parameter:
        A : matriks simetris positif definit (n×n)
        verbose: tampilkan detail perhitungan
    
    Return:
        L : matriks lower triangular
    """
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)
    
    if verbose:
        print("\n  Proses Dekomposisi Cholesky:")
        print("  " + "-" * 50)
    
    for j in range(n):
        # Elemen diagonal
        sum_sq = sum(L[j, k]**2 for k in range(j))
        val = A[j, j] - sum_sq
        if val < 0:
            raise ValueError(f"Matriks tidak positif definit! A[{j},{j}]-Σ = {val:.4f}")
        L[j, j] = np.sqrt(val)
        
        if verbose:
            print(f"  L[{j+1},{j+1}] = sqrt({A[j,j]:.4f} - {sum_sq:.4f}) = {L[j,j]:.6f}")
        
        # Elemen sub-diagonal di kolom j
        for i in range(j+1, n):
            sum_prod = sum(L[i, k] * L[j, k] for k in range(j))
            L[i, j] = (A[i, j] - sum_prod) / L[j, j]
            if verbose:
                print(f"  L[{i+1},{j+1}] = ({A[i,j]:.4f} - {sum_prod:.4f}) / {L[j,j]:.6f} = {L[i,j]:.6f}")
    
    return L

def forward_substitution(L, b):
    """Selesaikan [L]{y} = {b} dengan forward substitution."""
    n = len(b)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]
    return y

def back_substitution(U, y):
    """Selesaikan [U]{x} = {y} dengan back substitution. (U = Lᵀ)"""
    n = len(y)
    x = np.zeros(n)
    for i in range(n-1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
    return x

def cholesky_solve(A, b, verbose=True):
    """
    Selesaikan [A]{x} = {b} menggunakan dekomposisi Cholesky.
    """
    L = cholesky_decomposition(A, verbose)
    LT = L.T
    
    if verbose:
        print("\n  Matriks L:")
        print(np.array2string(L, precision=6))
        print("\n  Matriks Lᵀ:")
        print(np.array2string(LT, precision=6))
        print("\n  Verifikasi L·Lᵀ = A:")
        print(np.array2string(np.dot(L, LT), precision=4))
    
    # Forward substitution: [L]{y} = {b}
    y = forward_substitution(L, b)
    if verbose:
        print(f"\n  Forward sub → y = {y}")
    
    # Back substitution: [Lᵀ]{x} = {y}
    x = back_substitution(LT, y)
    if verbose:
        print(f"  Back sub → x = {x}")
    
    return x, L

# ─── Uji dengan Example 11.2 ──────────────────────────────
print("=" * 60)
print("SOAL 11.25 - Dekomposisi Cholesky")
print("=" * 60)

print("""
MATRIKS A (Example 11.2):
  | 6    15    55  |
  | 15   55   225  |
  | 55  225   979  |
""")

A = np.array([
    [  6,   15,   55],
    [ 15,   55,  225],
    [ 55,  225,  979],
], dtype=float)

b = np.array([152.6, 585.6, 2488.8])

print(f"Vektor b = {b}")

x, L = cholesky_solve(A, b)

# ─── Verifikasi ───────────────────────────────────────────
print("\n" + "=" * 60)
print("VERIFIKASI:")
print("=" * 60)

Ax = np.dot(A, x)
print("\nAx vs b:")
for i in range(len(b)):
    ok = '✓' if abs(Ax[i] - b[i]) < 1e-4 else '✗'
    print(f"  Baris {i+1}: {Ax[i]:.4f} = {b[i]:.4f} {ok}")

print("\nVerifikasi L·Lᵀ = A:")
LLT = np.dot(L, L.T)
match = np.allclose(LLT, A, atol=1e-8)
print(np.array2string(LLT, precision=4))
print(f"L·Lᵀ = A? {'YA ✓' if match else 'TIDAK ✗'}")

# ─── Soal 11.5 ───────────────────────────────────────────
print("\n" + "=" * 60)
print("SOAL 11.5: Matriks lain (untuk latihan)")
print("=" * 60)

A5 = np.array([
    [  6,   15,   55],
    [ 15,   55,  225],
    [ 55,  225,  979],
], dtype=float)

b5 = np.array([152.6, 585.6, 2488.8])
x5, L5 = cholesky_solve(A5, b5, verbose=False)

print(f"\nHasil (a₀, a₁, a₂):")
for i, xi in enumerate(x5):
    print(f"  a{i} = {xi:.6f}")

# ─── Bandingkan dengan numpy ──────────────────────────────
print("\n" + "=" * 60)
print("PERBANDINGAN DENGAN numpy.linalg:")
print("=" * 60)

x_np = np.linalg.solve(A, b)
print(f"\nnumpy.linalg.solve: {x_np}")
print(f"Cholesky manual   : {x}")
print(f"Selisih: {np.linalg.norm(x - x_np):.2e}")

print("\n" + "=" * 60)
print("RINGKASAN:")
print("=" * 60)
print(f"  Metode Cholesky cocok untuk matriks SIMETRIS POSITIF DEFINIT")
print(f"  Lebih efisien dari LU dekomposisi biasa (faktor ~2)")
print(f"  Hanya butuh setengah penyimpanan (lower triangular)")