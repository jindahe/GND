"""
Generate CMI vs p curves for L=10, r=1,2,3 and compare with paper Fig. 3(b).

Runs calculate_CMI for each (r, p) pair, saves results to CSV, then plots.
Expected qualitative features matching the paper:
  - Peak near p_c ≈ 0.11
  - Peak height decreasing with r
  - CMI → 0 for both p→0 and p→0.5
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time
from CMI_calculation import calculate_CMI, define_geometry_geom1

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
L           = 10
r_values    = [1, 2, 3]
num_samples = 500

# p grid: denser near p_c ≈ 0.11
p_values = np.array([
    0.02, 0.04, 0.06, 0.07, 0.08, 0.085, 0.09, 0.095,
    0.10, 0.105, 0.11, 0.115, 0.12, 0.125,
    0.13, 0.14, 0.15, 0.17, 0.20, 0.25, 0.30, 0.40, 0.50
])

RESULTS_CSV = "CMI_results_L10.csv"
PLOT_FILE   = "CMI_vs_p_L10.png"

# ---------------------------------------------------------------------------
# Run or load
# ---------------------------------------------------------------------------

def load_existing_results():
    """Load previously computed results from CSV."""
    results = {}   # (r, p) → cmi
    if not os.path.exists(RESULTS_CSV):
        return results
    with open(RESULTS_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row['r']), float(row['p']))
            results[key] = float(row['cmi']) if row['cmi'] != 'None' else None
    print(f"Loaded {len(results)} existing results from {RESULTS_CSV}")
    return results

def save_result(r, p, cmi):
    exists = os.path.exists(RESULTS_CSV)
    with open(RESULTS_CSV, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['r', 'p', 'cmi'])
        if not exists:
            writer.writeheader()
        writer.writerow({'r': r, 'p': f'{p:.6f}', 'cmi': cmi})

def run_sweep():
    existing = load_existing_results()
    total = len(r_values) * len(p_values)
    done = 0

    for r in r_values:
        # Check geometry is valid (C non-empty)
        A, B, C = define_geometry_geom1(L, r)
        if len(C) == 0:
            print(f"r={r}: C is empty — skipping")
            continue

        print(f"\n{'='*55}")
        print(f"r={r}  |A|={len(A)} |B|={len(B)} |C|={len(C)}")
        print(f"{'='*55}")

        for p in p_values:
            key = (r, round(p, 6))
            if key in existing:
                done += 1
                print(f"  r={r} p={p:.4f} → cached: {existing[key]:.5f}")
                continue

            t0 = time.perf_counter()
            cmi = calculate_CMI(L=L, p=p, r=r, num_samples=num_samples, verbose=False)
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
        A, B, C = define_geometry_geom1(L, r)
        if len(C) == 0:
            continue

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
    ax.set_title(f'Conditional Mutual Information  (L={L}, {num_samples} samples/pt)', fontsize=11)
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
