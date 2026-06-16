"""
=============================================================
SOAL 11.17 - Sistem Persamaan Nonlinear Simultan
=============================================================

PERMASALAHAN:
Diberikan sepasang persamaan nonlinear simultan:
    f(x, y) = 4 - y - 2x²
    g(x, y) = 8 - y² - 4x

Cari semua pasangan nilai (x, y) yang memenuhi kedua persamaan.

METODE:
Kita gunakan metode Newton-Raphson 2D (Newton untuk sistem nonlinear).
Metode ini membutuhkan Jacobian dari sistem:

    J = | ∂f/∂x  ∂f/∂y |   =   | -4x   -1  |
        | ∂g/∂x  ∂g/∂y |       | -4   -2y  |

Update rule:
    [x_new]   [x]         [f(x,y)]
    [y_new] = [y] - J⁻¹ · [g(x,y)]

Karena ada dua solusi, kita coba berbagai initial guess.
=============================================================
"""

import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ─── Definisi fungsi ───────────────────────────────────────
def f(x, y):
    return 4 - y - 2*x**2

def g(x, y):
    return 8 - y**2 - 4*x

def system(vars):
    x, y = vars
    return [f(x, y), g(x, y)]

# ─── Jacobian analitik ────────────────────────────────────
def jacobian(x, y):
    return np.array([
        [-4*x, -1],
        [-4,   -2*y]
    ])

# ─── Newton-Raphson 2D manual ─────────────────────────────
def newton_raphson_2d(x0, y0, tol=1e-10, max_iter=100):
    """
    Metode Newton-Raphson untuk sistem 2 persamaan nonlinear.
    
    Args:
        x0, y0 : tebakan awal
        tol     : toleransi konvergensi
        max_iter: iterasi maksimum
    
    Returns:
        (x, y, iterations) : solusi dan jumlah iterasi
    """
    x, y = x0, y0
    history = [(x, y)]
    
    for i in range(max_iter):
        F = np.array([f(x, y), g(x, y)])
        J = jacobian(x, y)
        
        # Cek apakah Jacobian singular
        if abs(np.linalg.det(J)) < 1e-12:
            print(f"  [!] Jacobian singular pada iterasi {i}, skip.")
            return None, None, i
        
        # Selesaikan J * delta = -F
        delta = np.linalg.solve(J, -F)
        x, y = x + delta[0], y + delta[1]
        history.append((x, y))
        
        # Cek konvergensi
        if np.linalg.norm(delta) < tol:
            return x, y, i+1
    
    return x, y, max_iter

# ─── Cari semua solusi dengan berbagai initial guess ──────
print("=" * 60)
print("SOAL 11.17 - Sistem Persamaan Nonlinear")
print("=" * 60)
print("\nPersamaan:")
print("  f(x, y) = 4 - y - 2x²  = 0")
print("  g(x, y) = 8 - y² - 4x  = 0")
print()

solutions = []
tol_unique = 1e-4

# Coba grid initial guess: x dari -6 ke 6, y dari -6 ke 6
x_range = np.arange(-6, 7, 2)
y_range = np.arange(-6, 7, 2)

print("Mencari solusi dari berbagai titik awal (grid -6 s/d 6)...")
print("-" * 60)

for x0 in x_range:
    for y0 in y_range:
        try:
            sol = fsolve(system, [x0, y0], full_output=True)
            x_sol, y_sol = sol[0]
            info = sol[1]
            
            # Verifikasi apakah benar-benar solusi
            residual = np.linalg.norm(system([x_sol, y_sol]))
            if residual < 1e-8:
                # Cek apakah sudah ada di list solusi
                is_new = True
                for existing in solutions:
                    if abs(x_sol - existing[0]) < tol_unique and abs(y_sol - existing[1]) < tol_unique:
                        is_new = False
                        break
                if is_new:
                    solutions.append((x_sol, y_sol))
        except:
            pass

# ─── Tampilkan hasil ──────────────────────────────────────
print(f"\nDitemukan {len(solutions)} solusi:\n")
for i, (xs, ys) in enumerate(solutions, 1):
    print(f"  Solusi {i}: x = {xs:.6f}, y = {ys:.6f}")
    print(f"    Verifikasi f(x,y) = {f(xs,ys):.2e}")
    print(f"    Verifikasi g(x,y) = {g(xs,ys):.2e}")
    print()

# ─── Newton-Raphson manual untuk dua solusi ───────────────
print("=" * 60)
print("DETAIL NEWTON-RAPHSON untuk masing-masing solusi:")
print("=" * 60)

guess_per_solution = [(-1, 2), (2, -2)]  # tebakan menuju tiap solusi

for i, (x0, y0) in enumerate(guess_per_solution, 1):
    print(f"\nInitial guess: x₀={x0}, y₀={y0}")
    x_sol, y_sol, iters = newton_raphson_2d(x0, y0)
    if x_sol is not None:
        print(f"  Solusi ditemukan dalam {iters} iterasi:")
        print(f"  x = {x_sol:.8f}")
        print(f"  y = {y_sol:.8f}")
        print(f"  f(x,y) = {f(x_sol, y_sol):.2e}")
        print(f"  g(x,y) = {g(x_sol, y_sol):.2e}")

# ─── Visualisasi kurva ────────────────────────────────────
print("\nMembuat plot visualisasi...")

x_plot = np.linspace(-4, 4, 400)
y_plot = np.linspace(-4, 4, 400)
X, Y = np.meshgrid(x_plot, y_plot)

F_grid = f(X, Y)
G_grid = g(X, Y)

fig, ax = plt.subplots(figsize=(8, 6))
ax.contour(X, Y, F_grid, levels=[0], colors='blue', linewidths=2, label='f(x,y)=0')
ax.contour(X, Y, G_grid, levels=[0], colors='red',  linewidths=2, label='g(x,y)=0')

for i, (xs, ys) in enumerate(solutions, 1):
    ax.plot(xs, ys, 'ko', markersize=10, zorder=5)
    ax.annotate(f'Solusi {i}\n({xs:.3f}, {ys:.3f})',
                xy=(xs, ys), xytext=(xs+0.3, ys+0.3),
                fontsize=9, arrowprops=dict(arrowstyle='->', color='black'))

# Legend manual
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], color='blue', lw=2, label='f(x,y) = 4 - y - 2x² = 0'),
    Line2D([0], [0], color='red',  lw=2, label='g(x,y) = 8 - y² - 4x = 0'),
    Line2D([0], [0], marker='o', color='black', lw=0, markersize=8, label='Solusi')
]
ax.legend(handles=legend_elements, loc='upper right')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Soal 11.17 - Persamaan Nonlinear Simultan\nf(x,y) = 0 dan g(x,y) = 0')
ax.grid(True, alpha=0.3)
ax.axhline(0, color='k', linewidth=0.5)
ax.axvline(0, color='k', linewidth=0.5)
plt.tight_layout()
plt.savefig('plot_11_17.png', dpi=120)
plt.show()
print("Plot disimpan: plot_11_17.png")