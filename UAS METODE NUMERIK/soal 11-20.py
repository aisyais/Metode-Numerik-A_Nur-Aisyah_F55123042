"""
=============================================================
SOAL 11.20 - Matriks Vandermonde 6x6 & Condition Number
=============================================================

PERMASALAHAN:
Ulangi soal 11.19, tetapi untuk matriks Vandermonde 6x6:
  x₁=4, x₂=2, x₃=7, x₄=10, x₅=3, x₆=5

Matriks Vandermonde:
  V[i,j] = xᵢ^(j-1), untuk i,j = 1..n

  V = | 1  x₁  x₁²  x₁³  x₁⁴  x₁⁵ |
      | 1  x₂  x₂²  x₂³  x₂⁴  x₂⁵ |
      | ...                          |
      | 1  x₆  x₆²  x₆³  x₆⁴  x₆⁵ |

Pertanyaan:
(a) Hitung spectral condition number
(b) Selesaikan Vx = b dimana b[i] = sum baris i (x_exact = 1)
(c) Bandingkan error dengan prediksi condition number

KONSEP:
- Vandermonde muncul dalam interpolasi polinomial
- Sering ill-conditioned untuk n besar atau titik x tersebar luas
=============================================================
"""

import numpy as np

print("=" * 60)
print("SOAL 11.20 - Matriks Vandermonde 6x6")
print("=" * 60)

# ─── Titik-titik x ────────────────────────────────────────
x_points = np.array([4, 2, 7, 10, 3, 5], dtype=float)
n = len(x_points)

print(f"\nTitik-titik x: {x_points}")
print(f"Dimensi: {n}x{n}")

# ─── Bangun matriks Vandermonde ───────────────────────────
print("\n1. Membangun Matriks Vandermonde")
print("   V[i,j] = x_i^(j-1), i,j mulai dari 1")

V = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        V[i, j] = x_points[i] ** j  # kolom 0..n-1 → x^0 s/d x^(n-1)

print("\n   Matriks V:")
print(np.array2string(V, precision=2, suppress_small=True))

# ─── Spectral Condition Number ────────────────────────────
print("\n2. Menghitung Spectral Condition Number")

sigma = np.linalg.svd(V, compute_uv=False)
kappa = sigma[0] / sigma[-1]

print(f"\n   σ_max = {sigma[0]:.6e}")
print(f"   σ_min = {sigma[-1]:.6e}")
print(f"   κ (spectral) = {kappa:.6e}")

digits_lost = np.log10(kappa)
print(f"   Log₁₀(κ) ≈ {digits_lost:.2f}")
print(f"   → Sekitar {digits_lost:.1f} digit presisi yang hilang")

# ─── Bangun vektor b ──────────────────────────────────────
print("\n3. Membangun vektor b (b[i] = sum baris i → x_exact = 1)")

b = V.sum(axis=1)
x_exact = np.ones(n)

print(f"   b = {np.array2string(b, precision=4)}")

# ─── Selesaikan sistem ────────────────────────────────────
print("\n4. Menyelesaikan Vx = b")

x_computed = np.linalg.solve(V, b)

print("\n   Perbandingan solusi:")
print(f"   {'i':>3}  {'x_computed':>15}  {'x_exact':>10}  {'abs_error':>12}")
print("   " + "-" * 45)
for i in range(n):
    err = abs(x_computed[i] - x_exact[i])
    print(f"   {i+1:>3}  {x_computed[i]:>15.10f}  {x_exact[i]:>10.1f}  {err:>12.4e}")

# ─── Analisis error ───────────────────────────────────────
rel_err = np.linalg.norm(x_computed - x_exact) / np.linalg.norm(x_exact)

print("\n5. Analisis Error")
print(f"   Error relatif aktual : {rel_err:.4e}")
print(f"   Log₁₀(error aktual)  : {np.log10(rel_err):.2f}")
print(f"   Condition number κ   : {kappa:.4e}")
print(f"   Log₁₀(κ)             : {np.log10(kappa):.2f}")

# ─── Perbandingan dengan Hilbert (11.19) ──────────────────
print("\n6. Perbandingan dengan Soal 11.19 (Hilbert 10x10)")
print(f"   Hilbert 10x10 : κ ≈ 10^13, sangat ill-conditioned")
print(f"   Vandermonde 6x6: κ ≈ {kappa:.2e}")
if kappa < 1e10:
    print("   → Vandermonde 6x6 LEBIH BAIK kondisinya dari Hilbert 10x10")
else:
    print("   → Vandermonde 6x6 juga ill-conditioned")

print("\n" + "=" * 60)
print("RINGKASAN:")
print("=" * 60)
print(f"  Matriks         : Vandermonde {n}x{n}")
print(f"  Titik x         : {x_points}")
print(f"  Condition number: κ = {kappa:.4e}")
print(f"  Digit hilang    : ≈ {digits_lost:.1f} digit")
print(f"  Error relatif   : {rel_err:.4e}")