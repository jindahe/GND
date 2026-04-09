"""
Generate CMI vs p curves for configurable geom1 sweeps.

Supports larger r in two ways:
  - automatically enlarges L when the fixed-L geometry no longer fits
  - can switch to bMPS contraction for larger r

Examples:
  python -m src.scripts.plot_cmi_vs_p
  python -m src.scripts.plot_cmi_vs_p --r-values 1,2,3,4,5,6
  python -m src.scripts.plot_cmi_vs_p --r-values 4,6 --method auto --num-samples 50 --no-show
"""

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.cmi_calculation import (
    calculate_CMI,
    calculate_CMI_bMPS,
    define_geometry_geom1,
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_L = 10
DEFAULT_R_VALUES = [1, 2, 3]
DEFAULT_NUM_SAMPLES = 500
DEFAULT_METHOD = "auto"
DEFAULT_MAX_EXACT_R = 4
DEFAULT_MAX_BOND = 64
DEFAULT_RESULTS_CSV = str((Path(__file__).resolve().parents[2] / "outputs" / "CMI_vs_p_results.csv"))
DEFAULT_PLOT_FILE = str((Path(__file__).resolve().parents[2] / "outputs" / "CMI_vs_p.png"))
DEFAULT_P_VALUES = np.linspace(0, 0.5, 100)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def _parse_int_list(text):
    return [int(item.strip()) for item in text.split(',') if item.strip()]


def _parse_float_list(text):
    return np.array([float(item.strip()) for item in text.split(',') if item.strip()])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Sweep CMI(p) for geom1 with exact or bMPS contraction."
    )
    parser.add_argument("--L", type=int, default=DEFAULT_L,
                        help="Base lattice size. Default: 10.")
    parser.add_argument("--r-values", type=str,
                        default=",".join(str(r) for r in DEFAULT_R_VALUES),
                        help="Comma-separated r list. Default: 1,2,3.")
    parser.add_argument("--p-values", type=str, default=None,
                        help="Optional comma-separated p list.")
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES,
                        help="Monte Carlo samples per point.")
    parser.add_argument("--method", choices=["auto", "exact", "bmps"],
                        default=DEFAULT_METHOD,
                        help="Contraction backend. auto: exact for small r, bMPS for large r.")
    parser.add_argument("--max-exact-r", type=int, default=DEFAULT_MAX_EXACT_R,
                        help="In auto mode, use exact TN up to this r.")
    parser.add_argument("--max-bond", type=int, default=DEFAULT_MAX_BOND,
                        help="bMPS maximum bond dimension.")
    parser.add_argument("--fixed-L", action="store_true",
                        help="Disable automatic L expansion for large r.")
    parser.add_argument("--results-csv", type=str, default=DEFAULT_RESULTS_CSV,
                        help="Output CSV path.")
    parser.add_argument("--plot-file", type=str, default=DEFAULT_PLOT_FILE,
                        help="Output plot path.")
    parser.add_argument("--no-show", action="store_true",
                        help="Save figure without opening an interactive window.")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Sweep configuration
# ---------------------------------------------------------------------------

def minimum_geom1_L(r):
    """Smallest L that fits geom1 without clipping A/B/C."""
    return 2 + 4 * r


def choose_solver(r, method, max_exact_r):
    if method != "auto":
        return method
    return "exact" if r <= max_exact_r else "bmps"


def resolve_geometry(L_base, r, fixed_L):
    """
    Return (L, A, B, C), auto-expanding L only when fixed-L geom1 makes C empty.
    """
    L = L_base
    A, B, C = define_geometry_geom1(L, r)

    if not fixed_L and len(C) == 0:
        L = max(L_base, minimum_geom1_L(r))
        A, B, C = define_geometry_geom1(L, r)

    return L, A, B, C


def result_key(solver, L, r, p, max_bond):
    bond_key = max_bond if solver == "bmps" else None
    return solver, int(L), int(r), round(float(p), 6), bond_key


def load_existing_results(csv_path, default_L):
    """Load cached results; supports both new and legacy CSV schema."""
    results = {}
    if not os.path.exists(csv_path):
        return results

    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            solver = row.get('solver', 'exact')
            L = int(row.get('L', default_L))
            r = int(row['r'])
            p = float(row['p'])
            max_bond_raw = row.get('max_bond', '')
            max_bond = int(max_bond_raw) if max_bond_raw not in ('', None) else None
            cmi_raw = row['cmi']
            cmi = float(cmi_raw) if cmi_raw != 'None' else None
            results[result_key(solver, L, r, p, max_bond)] = cmi

    print(f"Loaded {len(results)} existing results from {csv_path}")
    return results


def save_result(csv_path, solver, L, r, p, cmi, max_bond):
    exists = os.path.exists(csv_path)
    with open(csv_path, 'a', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['solver', 'L', 'r', 'p', 'max_bond', 'cmi'],
        )
        if not exists:
            writer.writeheader()
        writer.writerow({
            'solver': solver,
            'L': L,
            'r': r,
            'p': f'{p:.6f}',
            'max_bond': '' if solver != "bmps" else max_bond,
            'cmi': cmi,
        })


def evaluate_point(L, p, r, num_samples, solver, max_bond):
    if solver == "exact":
        return calculate_CMI(L=L, p=p, r=r, num_samples=num_samples, verbose=False)
    return calculate_CMI_bMPS(
        L=L,
        p=p,
        r=r,
        num_samples=num_samples,
        max_bond=max_bond,
        verbose=False,
    )


def build_curve_configs(args, p_values):
    configs = {}
    for r in _parse_int_list(args.r_values):
        L, A, B, C = resolve_geometry(args.L, r, args.fixed_L)
        if len(C) == 0:
            print(f"r={r}: C is empty even after geometry resolution — skipping")
            continue
        configs[r] = {
            'L': L,
            'A': A,
            'B': B,
            'C': C,
            'solver': choose_solver(r, args.method, args.max_exact_r),
            'max_bond': args.max_bond,
        }

    total = len(configs) * len(p_values)
    return configs, total


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

def run_sweep(args, p_values):
    existing = load_existing_results(args.results_csv, args.L)
    configs, total = build_curve_configs(args, p_values)
    done = 0

    for r in sorted(configs):
        cfg = configs[r]
        print(f"\n{'='*70}")
        print(
            f"r={r}  solver={cfg['solver']}  L={cfg['L']}  "
            f"|A|={len(cfg['A'])} |B|={len(cfg['B'])} |C|={len(cfg['C'])}"
        )
        if cfg['solver'] == "bmps":
            print(f"max_bond={cfg['max_bond']}")
        print(f"{'='*70}")

        for p in p_values:
            key = result_key(cfg['solver'], cfg['L'], r, p, cfg['max_bond'])
            if key in existing:
                done += 1
                cached = existing[key]
                cached_text = 'None' if cached is None else f"{cached:.5f}"
                print(f"  r={r} p={p:.4f} → cached: {cached_text}")
                continue

            t0 = time.perf_counter()
            cmi = evaluate_point(
                L=cfg['L'],
                p=p,
                r=r,
                num_samples=args.num_samples,
                solver=cfg['solver'],
                max_bond=cfg['max_bond'],
            )
            elapsed = time.perf_counter() - t0
            done += 1

            save_result(
                args.results_csv,
                solver=cfg['solver'],
                L=cfg['L'],
                r=r,
                p=p,
                cmi=cmi,
                max_bond=cfg['max_bond'],
            )
            existing[key] = cmi
            cmi_text = 'None' if cmi is None else f"{cmi:.5f}"
            print(f"  r={r} p={p:.4f} → CMI={cmi_text}  ({elapsed:.1f}s)  [{done}/{total}]")

    return existing, configs


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def make_plot(results, configs, p_values, args):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    fig, ax = plt.subplots(figsize=(7, 5))

    for idx, r in enumerate(sorted(configs)):
        cfg = configs[r]
        ps, cmis = [], []
        for p in p_values:
            key = result_key(cfg['solver'], cfg['L'], r, p, cfg['max_bond'])
            cmi = results.get(key)
            if cmi is not None:
                ps.append(p)
                cmis.append(cmi)

        if not ps:
            continue

        label = f'$r={r}$'
        extras = []
        if cfg['solver'] == "bmps":
            extras.append(f'bMPS, $\\chi$={cfg["max_bond"]}')
        if cfg['L'] != args.L:
            extras.append(f'L={cfg["L"]}')
        if extras:
            label += " [" + ", ".join(extras) + "]"

        ax.plot(
            ps,
            cmis,
            'o-',
            color=colors[idx % len(colors)],
            label=label,
            lw=1.8,
            ms=5,
        )

    ax.axvline(0.11, color='gray', lw=1, ls='--', label='$p_c=0.11$')
    ax.axhline(0, color='black', lw=0.5)

    ax.set_xlabel('$p$', fontsize=13)
    ax.set_ylabel('$I(A:C|B)$ [nats]', fontsize=13)
    ax.set_title(
        f'Conditional Mutual Information ({args.num_samples} samples/pt)',
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.set_xlim(0, 0.52)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(args.plot_file, dpi=150)
    print(f"\nSaved: {args.plot_file}")
    if args.no_show:
        plt.close(fig)
    else:
        plt.show()


def main():
    args = parse_args()
    p_values = DEFAULT_P_VALUES if args.p_values is None else _parse_float_list(args.p_values)
    results, configs = run_sweep(args, p_values)
    make_plot(results, configs, p_values, args)


if __name__ == "__main__":
    main()
