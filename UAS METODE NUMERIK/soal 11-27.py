"""
=============================================================
SOAL 11.27 - Persamaan Diferensial → Sistem Persamaan Linear
=============================================================

PERMASALAHAN:
Persamaan diferensial dari mass balance zat kimia di kanal 1D:
  
  0 = D·(d²c/dx²) - U·(dc/dx) - k·c

Parameter:
  D = 2   (koefisien difusi)
  U = 1   (kecepatan aliran)
  k = 0.2 (laju peluruhan first-order)
  c(0) = 80  (konsentrasi batas kiri)
  c(10) = 20 (konsentrasi batas kanan)
  Δx = 2    (langkah spasial, dari x=0 ke x=10)

DISKRITISASI BEDA HINGGA:
Menggunakan aproksimasi beda terpusat:
  d²c/dx² ≈ (cᵢ₊₁ - 2cᵢ + cᵢ₋₁) / Δx²
  dc/dx   ≈ (cᵢ₊₁ - cᵢ₋₁) / (2Δx)

Substitusi ke PD:
  D(cᵢ₊₁ - 2cᵢ + cᵢ₋₁)/Δx² - U(cᵢ₊₁ - cᵢ₋₁)/(2Δx) - k·cᵢ = 0

Kelompokkan berdasarkan cᵢ₋₁, cᵢ, cᵢ₊₁:
  (D/Δx² + U/2Δx)·cᵢ₋₁ + (-2D/Δx² - k)·cᵢ + (D/Δx² - U/2Δx)·cᵢ₊₁ = 0
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("SOAL 11.27 - PD ke Sistem Persamaan Linear")
print("=" * 60)

# ─── Parameter ────────────────────────────────────────────
D  = 2.0    # koefisien difusi
U  = 1.0    # kecepatan aliran
k  = 0.2    # laju peluruhan
c0 = 80.0   # c(0)  = 80
cL = 20.0   # c(10) = 20
x0 = 0.0    # batas kiri
xL = 10.0   # batas kanan
dx = 2.0    # langkah spasial

print(f"\nParameter:")
print(f"  D = {D}, U = {U}, k = {k}")
print(f"  c(0) = {c0}, c(10) = {cL}")
print(f"  Δx = {dx}")

# ─── Titik-titik grid ────────────────────────────────────
x_nodes = np.arange(x0, xL + dx/2, dx)
N = len(x_nodes)
print(f"\nTitik-titik grid x = {x_nodes}")
print(f"Jumlah titik total: {N} (termasuk batas)")
print(f"Titik interior (unknown): {N-2}")

# ─── Koefisien diskritisasi ───────────────────────────────
# Faktor
A_coeff = D / dx**2 + U / (2*dx)   # koefisien cᵢ₋₁
B_coeff = -2*D / dx**2 - k          # koefisien cᵢ (diagonal)
C_coeff = D / dx**2 - U / (2*dx)   # koefisien cᵢ₊₁

print(f"\nKoefisien diskritisasi beda hingga:")
print(f"  a (koef cᵢ₋₁) = D/Δx² + U/(2Δx) = {D/dx**2:.4f} + {U/(2*dx):.4f} = {A_coeff:.4f}")
print(f"  b (koef cᵢ  ) = -2D/Δx² - k      = {-2*D/dx**2:.4f} - {k:.4f} = {B_coeff:.4f}")
print(f"  c (koef cᵢ₊₁) = D/Δx² - U/(2Δx) = {D/dx**2:.4f} - {U/(2*dx):.4f} = {C_coeff:.4f}")

# ─── Bangun sistem [A]{c} = {rhs} ─────────────────────────
n_int = N - 2  # jumlah titik interior
A_mat = np.zeros((n_int, n_int))
rhs   = np.zeros(n_int)

print(f"\nMembangun matriks untuk {n_int} titik interior (i=1..{n_int}):")

for i in range(n_int):
    A_mat[i, i] = B_coeff  # diagonal
    
    if i > 0:
        A_mat[i, i-1] = A_coeff  # sub-diagonal
    else:
        rhs[i] -= A_coeff * c0   # titik batas kiri
    
    if i < n_int - 1:
        A_mat[i, i+1] = C_coeff  # super-diagonal
    else:
        rhs[i] -= C_coeff * cL   # titik batas kanan

print("\nMatriks A:")
print(np.array2string(A_mat, precision=4))
print(f"\nVektor RHS = {rhs}")

# ─── Selesaikan sistem ────────────────────────────────────
c_interior = np.linalg.solve(A_mat, rhs)

# Gabungkan dengan batas
c_all = np.concatenate([[c0], c_interior, [cL]])

print("\nSolusi numerik:")
print(f"{'x':>6}  {'c(x)':>12}")
print("-" * 20)
for xi, ci in zip(x_nodes, c_all):
    print(f"{xi:>6.1f}  {ci:>12.6f}")

# ─── Solusi analitik (untuk verifikasi) ───────────────────
# Solusi analitik: c = A·e^(r1·x) + B·e^(r2·x)
# r = (U ± sqrt(U² + 4kD)) / (2D)
discriminant = U**2 + 4*k*D
r1 = (U + np.sqrt(discriminant)) / (2*D)
r2 = (U - np.sqrt(discriminant)) / (2*D)

# Syarat batas: A·e^(r1·0) + B·e^(r2·0) = c0
#               A·e^(r1·10) + B·e^(r2·10) = cL
A_mat2 = np.array([[1, 1],
                    [np.exp(r1*xL), np.exp(r2*xL)]])
b_vec2 = np.array([c0, cL])
coefs = np.linalg.solve(A_mat2, b_vec2)
A_an, B_an = coefs

x_fine = np.linspace(x0, xL, 200)
c_analytic = A_an * np.exp(r1 * x_fine) + B_an * np.exp(r2 * x_fine)
c_analytic_nodes = A_an * np.exp(r1 * x_nodes) + B_an * np.exp(r2 * x_nodes)

# ─── Plot ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_fine, c_analytic, 'b-', linewidth=2, label='Solusi Analitik')
ax.plot(x_nodes, c_all, 'ro--', markersize=8, linewidth=1.5, label=f'Solusi Numerik (Δx={dx})')
ax.set_xlabel('Jarak x (m)')
ax.set_ylabel('Konsentrasi c (g/m³)')
ax.set_title('Soal 11.27 - Distribusi Konsentrasi di Kanal 1D\n'
             f'D={D}, U={U}, k={k}')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('plot_11_27.png', dpi=120)
plt.show()
print("\nPlot disimpan: plot_11_27.png")

# ─── Perbandingan ─────────────────────────────────────────
print("\nPerbandingan solusi numerik vs analitik:")
print(f"{'x':>6}  {'Numerik':>12}  {'Analitik':>12}  {'Error%':>10}")
print("-" * 45)
for xi, cn, ca in zip(x_nodes, c_all, c_analytic_nodes):
    err = abs(cn - ca) / abs(ca) * 100 if abs(ca) > 1e-10 else 0
    print(f"{xi:>6.1f}  {cn:>12.6f}  {ca:>12.6f}  {err:>10.4f}%")