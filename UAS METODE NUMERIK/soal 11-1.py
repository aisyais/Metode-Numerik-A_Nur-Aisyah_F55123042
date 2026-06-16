"""
SOAL 11.1
=========
Selesaikan sistem tridiagonal berikut menggunakan:
(a) Thomas Algorithm (seperti Example 11.1)
(b) Gauss-Seidel (seperti Example 11.3)

Sistem persamaan:
[ 0.8  -0.4   0  ] [x1]   [41 ]
[-0.4   0.8  -0.4] [x2] = [25 ]
[ 0    -0.4   0.8] [x3]   [105]

Metode:
- Thomas Algorithm: metode efisien untuk matriks tridiagonal.
  Terdiri dari 2 langkah:
    1. Forward sweep (eliminasi ke bawah)
    2. Back substitution (substitusi mundur)
- Gauss-Seidel: metode iteratif, update setiap variabel satu per satu
  menggunakan nilai terbaru hingga konvergen.
"""

import numpy as np

# ── Data soal ──────────────────────────────────────────────────────────────────
# Matriks koefisien tridiagonal
A = np.array([
    [0.8, -0.4,  0.0],
    [-0.4, 0.8, -0.4],
    [0.0, -0.4,  0.8]
], dtype=float)

b = np.array([41.0, 25.0, 105.0])
n = len(b)

# ── BAGIAN (a): Thomas Algorithm ───────────────────────────────────────────────
print("=" * 60)
print("BAGIAN (a): Thomas Algorithm")
print("=" * 60)

# Ekstrak diagonal bawah (e), utama (f), atas (g), dan RHS (r)
e = np.array([0.0, A[1,0], A[2,1]])   # sub-diagonal
f = np.array([A[0,0], A[1,1], A[2,2]], dtype=float)  # diagonal utama
g = np.array([A[0,1], A[1,2], 0.0])   # super-diagonal
r = b.copy().astype(float)             # right-hand side

print("\nLangkah 1 — Forward Sweep (eliminasi ke bawah):")
print(f"  f' = [f1, ...]  |  r' = [r1, ...]")

for k in range(1, n):
    faktor = e[k] / f[k-1]
    f[k]   = f[k] - faktor * g[k-1]
    r[k]   = r[k] - faktor * r[k-1]
    print(f"  k={k}: faktor={faktor:.4f} → f[{k}]={f[k]:.4f}, r[{k}]={r[k]:.4f}")

print("\nLangkah 2 — Back Substitution (substitusi mundur):")
x = np.zeros(n)
x[-1] = r[-1] / f[-1]
print(f"  x[{n-1}] = {r[-1]:.4f} / {f[-1]:.4f} = {x[-1]:.4f}")

for k in range(n-2, -1, -1):
    x[k] = (r[k] - g[k] * x[k+1]) / f[k]
    print(f"  x[{k}] = ({r[k]:.4f} - {g[k]:.4f}*{x[k+1]:.4f}) / {f[k]:.4f} = {x[k]:.4f}")

print(f"\nHasil Thomas Algorithm:")
for i, xi in enumerate(x):
    print(f"  x{i+1} = {xi:.4f}")

# Verifikasi
print(f"\nVerifikasi Ax = b:")
print(f"  A @ x = {A @ x}")
print(f"  b     = {b}")

# ── BAGIAN (b): Gauss-Seidel ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("BAGIAN (b): Gauss-Seidel (toleransi es = 5%)")
print("=" * 60)

# Periksa diagonal dominan
print("\nCek diagonal dominan:")
for i in range(n):
    off = sum(abs(A[i,j]) for j in range(n) if j != i)
    status = "OK" if abs(A[i,i]) >= off else "TIDAK"
    print(f"  Baris {i}: |{A[i,i]}| >= {off} → {status}")

x_gs = np.zeros(n)
es = 5.0        # toleransi persen
max_iter = 100

print(f"\n{'Iter':>4} | {'x1':>10} | {'x2':>10} | {'x3':>10} | {'ea_max':>10}")
print("-" * 55)

for it in range(1, max_iter + 1):
    x_old = x_gs.copy()

    x_gs[0] = (b[0] - A[0,1]*x_gs[1] - A[0,2]*x_gs[2]) / A[0,0]
    x_gs[1] = (b[1] - A[1,0]*x_gs[0] - A[1,2]*x_gs[2]) / A[1,1]
    x_gs[2] = (b[2] - A[2,0]*x_gs[0] - A[2,1]*x_gs[1]) / A[2,2]

    # Hitung error relatif tiap variabel
    ea = []
    for i in range(n):
        if x_gs[i] != 0:
            ea.append(abs((x_gs[i] - x_old[i]) / x_gs[i]) * 100)
        else:
            ea.append(0.0)
    ea_max = max(ea)

    print(f"  {it:>3} | {x_gs[0]:>10.5f} | {x_gs[1]:>10.5f} | {x_gs[2]:>10.5f} | {ea_max:>9.4f}%")

    if ea_max < es and it > 1:
        print(f"\n  Konvergen pada iterasi ke-{it}!")
        break

print(f"\nHasil Gauss-Seidel:")
for i, xi in enumerate(x_gs):
    print(f"  x{i+1} = {xi:.4f}")