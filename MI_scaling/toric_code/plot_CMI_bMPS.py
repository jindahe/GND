"""
CMI(p) curves for r=6,9,12,15 using bMPS contraction.

Each (r, p) point saved to CSV immediately.  Restartable: skips already-computed points.
L = 2 + 4*r (minimum unclipped lattice).
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time
from CMI_calculation import calculate_CMI_bMPS, define_geometry_geom1

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
r_values    = [6, 9, 12, 15]
num_samples = 500
max_bond    = 16

p_values = np.array([
    0.02, 0.04, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095,
    0.10, 0.105, 0.11, 0.115, 0.12, 0.125,
    0.13, 0.14, 0.15, 0.17, 0.20, 0.25, 0.30, 0.40, 0.50
])

RESULTS_CSV = "CMI_bMPS_results.csv"
PLOT_FILE   = "CMI_bMPS_vs_p.png"

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def load_existing():
    results = {}
    if not os.path.exists(RESULTS_CSV):
        return results
    with open(RESULTS_CSV, newline='') as f:
        for row in csv.DictReader(f):
            key = (int(row['r']), float(row['p']))
            results[key] = float(row['cmi']) if row['cmi'] != 'None' else None
    print(f"Loaded {len(results)} cached results from {RESULTS_CSV}")
    return results

def save_result(r, p, cmi):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['r', 'p', 'cmi'])
        if not exists:
            w.writeheader()
        w.writerow({'r': r, 'p': f'{p:.6f}', 'cmi': cmi})

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep():
    existing = load_existing()
    total = len(r_values) * len(p_values)
    done  = sum(1 for k in existing if k[0] in r_values and round(k[1], 6) in [round(p,6) for p in p_values])

    for r in r_values:
        L = 2 + 4 * r
        A, B, C = define_geometry_geom1(L, r)
        print(f"\n{'='*60}")
        print(f"r={r}  L={L}  |A|={len(A)} |B|={len(B)} |C|={len(C)}")
        print(f"{'='*60}")

        for p in p_values:
            key = (r, round(p, 6))
            if key in existing:
                done += 1
                print(f"  r={r} p={p:.4f} → cached: {existing[key]:.5f}")
                continue

            t0 = time.perf_counter()
            cmi = calculate_CMI_bMPS(L=L, p=p, r=r, num_samples=num_samples,
                                     max_bond=max_bond, verbose=False)
            elapsed = time.perf_counter() - t0
            done += 1

            save_result(r, p, cmi)
            existing[key] = cmi
            print(f"  r={r} p={p:.4f} → CMI={cmi:.5f}  ({elapsed:.1f}s)  [{done}/{total}]")

    return existing

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, r in enumerate(r_values):
        ps, cmis = [], []
        for p in p_values:
            key = (r, round(p, 6))
            cmi = results.get(key)
            if cmi is not None:
                ps.append(p)
                cmis.append(cmi)
        if not ps:
            continue
        ax.plot(ps, cmis, 'o-', color=colors[idx % len(colors)],
                label=f'$r={r}$', lw=1.8, ms=5)

    ax.axvline(0.11, color='gray', lw=1, ls='--', label='$p_c=0.11$')
    ax.axhline(0, color='black', lw=0.5)
    ax.set_xlabel('$p$', fontsize=13)
    ax.set_ylabel('$I(A:C|B)$ [nats]', fontsize=13)
    ax.set_title(f'CMI via bMPS  (r=6,9,12,15, {num_samples} samples/pt, $\\chi$={max_bond})',
                 fontsize=11)
    ax.legend(fontsize=11)
    ax.set_xlim(0, 0.52)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=150)
    print(f"\nSaved: {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    results = run_sweep()
    make_plot(results)
