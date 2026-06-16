"""
=============================================================
SOAL 11.19 - Hilbert Matrix 10x10 & Condition Number
=============================================================

PERMASALAHAN:
Tentukan spectral condition number untuk matriks Hilbert 10x10.
Elemen Hilbert: H[i,j] = 1/(i+j-1)  (indeks mulai dari 1)

Pertanyaan:
(a) Berapa digit presisi yang hilang akibat ill-conditioning?
(b) Selesaikan sistem Hx = b, dimana b[i] = jumlah elemen baris i
    (sehingga semua x seharusnya = 1). Bandingkan error aktual
    dengan prediksi dari condition number.

KONSEP:
- Condition number κ = ||A|| · ||A⁻¹||
- Spectral condition number menggunakan norma-2 (nilai singular)
- Log₁₀(κ) ≈ jumlah digit presisi yang hilang
- Error bound: ||Δx||/||x|| ≤ κ · ||Δb||/||b||
=============================================================
"""

import numpy as np
from scipy.linalg import hilbert

print("=" * 60)
print("SOAL 11.19 - Matriks Hilbert 10x10 & Condition Number")
print("=" * 60)

n = 10  # dimensi Hilbert matrix

# ─── Buat Hilbert matrix secara manual ───────────────────
print(f"\n1. Membuat Hilbert Matrix {n}x{n}")
print("   H[i,j] = 1/(i+j-1), i,j = 1..10")

H = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        H[i, j] = 1.0 / (i + j + 1)  # i,j mulai 0 → i+j+1

print("\n   5 elemen pertama (baris 1-5, kolom 1-5):")
print(np.array2string(H[:5, :5], precision=6, suppress_small=True))
print("   ...")

# ─── Spectral Condition Number ────────────────────────────
print("\n2. Menghitung Spectral Condition Number (norma-2)")

# Menggunakan nilai singular (SVD)
sigma = np.linalg.svd(H, compute_uv=False)
kappa_spectral = sigma[0] / sigma[-1]

# Menggunakan numpy langsung
kappa_np = np.linalg.cond(H, 2)

print(f"\n   Nilai singular maksimum σ_max = {sigma[0]:.6e}")
print(f"   Nilai singular minimum σ_min = {sigma[-1]:.6e}")
print(f"   Spectral condition number κ = σ_max/σ_min = {kappa_spectral:.6e}")
print(f"   (Konfirmasi numpy) κ = {kappa_np:.6e}")

digits_lost = np.log10(kappa_spectral)
print(f"\n   Log₁₀(κ) ≈ {digits_lost:.1f}")
print(f"   → Sekitar {digits_lost:.0f} digit presisi HILANG akibat ill-conditioning!")
print(f"   → Dengan double precision (~15-16 digit), tersisa ~{15-digits_lost:.0f} digit akurat")

# ─── Bangun vektor b: b[i] = sum of row i ─────────────────
print("\n3. Membangun vektor b (b[i] = jumlah elemen baris ke-i)")
print("   (Sehingga solusi eksak x = [1, 1, ..., 1])")

b = H.sum(axis=1)
x_exact = np.ones(n)

print(f"\n   b = {np.array2string(b, precision=6)}")

# ─── Selesaikan sistem Hx = b ─────────────────────────────
print("\n4. Menyelesaikan sistem Hx = b dengan numpy.linalg.solve")

x_computed = np.linalg.solve(H, b)

print("\n   Solusi yang dihitung vs solusi eksak (x=1):")
print(f"   {'i':>3}  {'x_computed':>15}  {'x_exact':>10}  {'error':>12}")
print("   " + "-" * 45)
for i in range(n):
    err = abs(x_computed[i] - x_exact[i])
    print(f"   {i+1:>3}  {x_computed[i]:>15.8f}  {x_exact[i]:>10.1f}  {err:>12.4e}")

# ─── Analisis error ───────────────────────────────────────
print("\n5. Analisis Error")
rel_error = np.linalg.norm(x_computed - x_exact) / np.linalg.norm(x_exact)
print(f"\n   Error relatif aktual  : {rel_error:.4e}")
print(f"   Condition number κ    : {kappa_spectral:.4e}")
print(f"   Log₁₀(error aktual)   : {np.log10(rel_error):.1f}")
print(f"\n   Interpretasi:")
print(f"   Error aktual sangat besar karena matriks Hilbert sangat ill-conditioned.")
print(f"   Meskipun b tepat, pembulatan floating-point menyebabkan galat besar pada x.")

print("\n" + "=" * 60)
print("RINGKASAN:")
print("=" * 60)
print(f"  Spectral condition number κ ≈ {kappa_spectral:.2e}")
print(f"  Digit presisi yang hilang   ≈ {digits_lost:.0f} digit")
print(f"  Error relatif solusi        ≈ {rel_error:.2e}")
print(f"  Matriks Hilbert 10x10 sangat ILL-CONDITIONED!")