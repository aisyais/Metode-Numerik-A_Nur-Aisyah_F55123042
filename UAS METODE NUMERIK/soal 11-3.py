"""
SOAL 11.3
=========
Selesaikan sistem tridiagonal berikut (Crank-Nicolson untuk PDE)
menggunakan Thomas Algorithm:

[ 2.01475   -0.020875    0          0        ] [T1]   [4.175 ]
[-0.020875   2.01475   -0.020875    0        ] [T2] = [0     ]
[ 0         -0.020875   2.01475   -0.020875  ] [T3]   [0     ]
[ 0          0         -0.020875   2.01475   ] [T4]   [2.0875]

Metode: Thomas Algorithm
- Khusus untuk matriks tridiagonal (band width = 1).
- Kompleksitas O(n) jauh lebih efisien dari Gauss eliminasi O(n³).
- Langkah:
    1. Forward sweep: eliminasi sub-diagonal satu per satu
    2. Back substitution: hitung x dari bawah ke atas
"""

import numpy as np

# ── Data soal ──────────────────────────────────────────────────────────────────
n = 4
# sub-diagonal (e[0] tidak digunakan)
e = np.array([0.0, -0.020875, -0.020875, -0.020875])
# diagonal utama
f = np.array([2.01475, 2.01475, 2.01475, 2.01475], dtype=float)
# super-diagonal (g[n-1] tidak digunakan)
g = np.array([-0.020875, -0.020875, -0.020875, 0.0])
# right-hand side
r = np.array([4.175, 0.0, 0.0, 2.0875], dtype=float)

print("=" * 60)
print("Thomas Algorithm — Sistem Crank-Nicolson (4x4)")
print("=" * 60)
print("\nKoefisien awal:")
print(f"  e (sub-diag)   = {e}")
print(f"  f (diag utama) = {f}")
print(f"  g (super-diag) = {g}")
print(f"  r (RHS)        = {r}")

# ── Forward Sweep ──────────────────────────────────────────────────────────────
print("\n--- Forward Sweep ---")
for k in range(1, n):
    faktor = e[k] / f[k-1]
    f[k]   = f[k] - faktor * g[k-1]
    r[k]   = r[k] - faktor * r[k-1]
    print(f"  k={k}: faktor = {e[k]:.6f}/{f[k-1]:.6f} = {faktor:.6f}")
    print(f"         f[{k}] = {f[k]:.6f}, r[{k}] = {r[k]:.6f}")

# ── Back Substitution ──────────────────────────────────────────────────────────
print("\n--- Back Substitution ---")
T = np.zeros(n)
T[-1] = r[-1] / f[-1]
print(f"  T[{n-1}] = {r[-1]:.6f} / {f[-1]:.6f} = {T[-1]:.6f}")

for k in range(n-2, -1, -1):
    T[k] = (r[k] - g[k] * T[k+1]) / f[k]
    print(f"  T[{k}] = ({r[k]:.6f} - {g[k]:.6f}*{T[k+1]:.6f}) / {f[k]:.6f} = {T[k]:.6f}")

# ── Hasil ──────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("Hasil:")
for i, ti in enumerate(T):
    print(f"  T{i+1} = {ti:.6f}")

# Verifikasi dengan numpy
A = np.array([
    [ 2.01475,  -0.020875,  0.0,        0.0      ],
    [-0.020875,  2.01475,  -0.020875,   0.0      ],
    [ 0.0,      -0.020875,  2.01475,   -0.020875 ],
    [ 0.0,       0.0,      -0.020875,   2.01475  ]
])
b = np.array([4.175, 0.0, 0.0, 2.0875])

print("\nVerifikasi A @ T:")
print(f"  A @ T = {A @ T}")
print(f"  b     = {b}")
print(f"  Error = {np.max(np.abs(A @ T - b)):.2e}")