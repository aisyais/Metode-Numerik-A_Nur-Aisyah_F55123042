"""
=============================================================
SOAL 11.23 - Jumlah Operasi: Thomas Algorithm vs Gauss Elimination
=============================================================

PERMASALAHAN:
Tentukan jumlah operasi (perkalian/pembagian dan penjumlahan/pengurangan)
yang dibutuhkan untuk:
  (a) Algoritma Thomas (untuk sistem tridiagonal)
  (b) Eliminasi Gauss (tanpa partial pivoting)

Buat plot jumlah operasi vs n (n = 2 sampai 20).

ANALISIS TEORETIS:

Eliminasi Gauss (nxn penuh):
  - Operasi total ≈ n³/3 + n² - n/3  (pembagian+perkalian-pengurangan)
  - Orde O(n³)

Algoritma Thomas (tridiagonal nxn):
  Forward sweep: 
    - (n-1) pembagian + (n-1) perkalian + (n-1) pengurangan = 3(n-1) ops
  Back substitution:
    - (n-1) perkalian + (n-1) pengurangan + (n) pembagian = 3n-2 ops
  Total ≈ 8n - 7 operasi → O(n) !!!

Keunggulan Thomas sangat signifikan untuk n besar.
=============================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("SOAL 11.23 - Analisis Jumlah Operasi")
print("=" * 60)

# ─── Formula jumlah operasi ───────────────────────────────

def ops_gauss(n):
    """
    Jumlah operasi Eliminasi Gauss (tanpa pivoting):
    Perkalian/Pembagian: n(n-1)(2n-1)/6 + n(n-1)/2
    Pengurangan: n(n-1)(2n-1)/6 + n(n-1)/2 - n + 1
    Total ≈ n³/3 + n² - n/3
    """
    mul_div = (n * (n - 1) * (2*n - 1)) // 6 + (n * (n-1)) // 2
    add_sub = (n * (n - 1) * (2*n - 1)) // 6 + (n * (n-1)) // 2 - n + 1
    return mul_div + add_sub

def ops_thomas(n):
    """
    Jumlah operasi Algoritma Thomas (tridiagonal):
    Forward sweep (decomposisi):
      - Untuk k = 2..n: 1 pembagian, 1 perkalian, 1 pengurangan = 3(n-1)
    Forward sweep (substitusi RHS):
      - Untuk k = 2..n: 1 perkalian, 1 pengurangan = 2(n-1)
    Back substitution:
      - Untuk k = n-1..1: 1 perkalian, 1 pengurangan, 1 pembagian = 3(n-1) + 1 pembagian pertama
    
    Total ≈ 8(n-1) + 1 ≈ 8n - 7
    """
    forward_decomp = 3 * (n - 1)    # dekomposisi elemen diagonal
    forward_rhs    = 2 * (n - 1)    # update RHS
    back_sub       = 3 * (n - 1) + 1  # back substitution
    return forward_decomp + forward_rhs + back_sub

# ─── Tampilkan tabel ──────────────────────────────────────
print("\nTabel Jumlah Operasi:")
print(f"{'n':>5} {'Thomas (8n-7)':>15} {'Gauss (~n³/3)':>15} {'Rasio Gauss/Thomas':>20}")
print("-" * 58)

n_values = list(range(2, 21))
ops_t_list = []
ops_g_list = []

for n in n_values:
    ot = ops_thomas(n)
    og = ops_gauss(n)
    ops_t_list.append(ot)
    ops_g_list.append(og)
    ratio = og / ot if ot > 0 else 0
    print(f"{n:>5} {ot:>15} {og:>15} {ratio:>20.1f}x")

# ─── Plot ─────────────────────────────────────────────────
print("\nMembuat plot...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Linear scale
ax1.plot(n_values, ops_t_list, 'bo-', linewidth=2, markersize=6, label='Thomas Algorithm O(n)')
ax1.plot(n_values, ops_g_list, 'rs-', linewidth=2, markersize=6, label='Gauss Elimination O(n³)')
ax1.set_xlabel('Ukuran sistem n')
ax1.set_ylabel('Jumlah Operasi')
ax1.set_title('Perbandingan Operasi: Thomas vs Gauss\n(Skala Linear)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Log scale
ax2.semilogy(n_values, ops_t_list, 'bo-', linewidth=2, markersize=6, label='Thomas Algorithm O(n)')
ax2.semilogy(n_values, ops_g_list, 'rs-', linewidth=2, markersize=6, label='Gauss Elimination O(n³)')
ax2.set_xlabel('Ukuran sistem n')
ax2.set_ylabel('Jumlah Operasi (log scale)')
ax2.set_title('Perbandingan Operasi: Thomas vs Gauss\n(Skala Log)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('plot_11_23.png', dpi=120)
plt.show()
print("Plot disimpan: plot_11_23.png")

# ─── Perbandingan analitis ────────────────────────────────
print("\n" + "=" * 60)
print("ANALISIS TEORETIS:")
print("=" * 60)
print("""
Algoritma Thomas (Tridiagonal):
  - Forward decomp  : 3(n-1) operasi
  - Forward RHS sub : 2(n-1) operasi  
  - Back substitution: 3(n-1) + 1 operasi
  - TOTAL ≈ 8n - 7  →  O(n)

Eliminasi Gauss (Penuh):
  - TOTAL ≈ 2n³/3 + 3n²/2 - 7n/6  →  O(n³)

Kesimpulan:
  Untuk n=20: Thomas ≈ 153 ops vs Gauss ≈ 5.800 ops → 38x lebih cepat
  Keuntungan Thomas meningkat KUBIK seiring bertambahnya n!
  Inilah mengapa algoritma khusus untuk matriks sparse sangat penting.
""")