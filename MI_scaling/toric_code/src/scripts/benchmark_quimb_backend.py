"""
Benchmark script for quimb backend vs exact TN and custom bMPS.

Runs the verification matrix from PLAN.md §P8.3:
  A. Region-level accuracy   (L=10 r=2, L=14 r=3)
  B. CMI-level regression    (same parameter sets)
  C. Large-r practicality    (r=4,5,6, small sample, varying max_bond)

Outputs:
  outputs/quimb_region_benchmark.csv
  outputs/quimb_cmi_benchmark.csv
  outputs/quimb_bond_scan.csv

Usage:
  python -m src.scripts.benchmark_quimb_backend [--quick]
  --quick : smaller samples/chi for a fast test run
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(exist_ok=True)

OUT_REGION = OUT_DIR / "quimb_region_benchmark.csv"
OUT_CMI    = OUT_DIR / "quimb_cmi_benchmark.csv"
OUT_BOND   = OUT_DIR / "quimb_bond_scan.csv"


# ---------------------------------------------------------------------------
# Region-level accuracy benchmark (P8.3 A)
# ---------------------------------------------------------------------------

def run_region_benchmark(cases, seed=42, verbose=True):
    """
    Compare six region contractions: exact vs custom bMPS vs quimb bMPS.

    cases : list of (L, p, r, max_bond) tuples
    Returns list of result dicts.
    """
    from src.core.cmi_calculation import _contract, define_geometry_geom1
    from src.core.bmps_contraction import bMPS_contract_region
    from src.core.quimb_contraction import quimb_contract_region

    results = []

    for L, p, r, chi in cases:
        np.random.seed(seed)
        A, B, C = define_geometry_geom1(L, r)
        AB = A | B
        BC = B | C
        ABC = A | B | C

        h = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
        v = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])
        m = (h[:L] + h[1:] + v[:, :L] + v[:, 1:]) % 2

        contractions = [
            ('Pr(m_ABC)',      ABC, None),
            ('Pr(m_AB)',       AB,  None),
            ('Pr(m_B)',        B,   None),
            ('Pr_signed(m_B)', B,   A),
            ('Pr(m_BC)',       BC,  None),
            ('Pr_signed(m_BC)',BC,  A),
        ]

        if verbose:
            print(f"\n=== Region benchmark: L={L} p={p} r={r} chi={chi} ===")

        for label, rQ, rA in contractions:
            exact = _contract(m, rQ, region_A_signed=rA, p=p)

            t0 = time.perf_counter()
            custom = bMPS_contract_region(m, rQ, rA, p, max_bond=chi)
            t_custom = time.perf_counter() - t0

            t0 = time.perf_counter()
            qb = quimb_contract_region(m, rQ, rA, p, max_bond=chi)
            t_quimb = time.perf_counter() - t0

            denom = abs(exact) + 1e-300
            re_custom = abs(exact - custom) / denom
            re_quimb  = abs(exact - qb) / denom

            row = {
                "L": L, "p": p, "r": r, "max_bond": chi,
                "observable": label,
                "exact": exact,
                "custom_bmps": custom,
                "quimb_bmps": qb,
                "rel_err_custom": re_custom,
                "rel_err_quimb": re_quimb,
                "t_custom_s": t_custom,
                "t_quimb_s": t_quimb,
            }
            results.append(row)

            if verbose:
                better = "✓" if re_quimb < re_custom * 0.99 else (
                    "=" if abs(re_quimb - re_custom) / max(re_custom, 1e-15) < 0.02 else "✗")
                print(f"  {label:<26} exact={exact:.4e}  "
                      f"custom_err={re_custom:.2e}  quimb_err={re_quimb:.2e}  "
                      f"t_c={t_custom:.3f}s  t_q={t_quimb:.3f}s  {better}")

    return results


# ---------------------------------------------------------------------------
# CMI-level regression (P8.3 B)
# ---------------------------------------------------------------------------

def run_cmi_benchmark(cases, verbose=True):
    """
    Compare CMI from exact / custom bMPS / quimb bMPS.

    cases : list of (L, p, r, num_samples, max_bond, seed) tuples
    Returns list of result dicts.
    """
    from src.core.cmi_calculation import calculate_CMI, calculate_CMI_bMPS
    from src.core.quimb_contraction import calculate_CMI_quimb

    results = []

    for L, p, r, N, chi, seed in cases:
        if verbose:
            print(f"\n=== CMI benchmark: L={L} p={p} r={r} N={N} chi={chi} ===")

        row = {"L": L, "p": p, "r": r, "num_samples": N, "max_bond": chi, "seed": seed}

        for name, fn, kw in [
            ("exact",  calculate_CMI,
             dict(L=L, p=p, r=r, num_samples=N, seed=seed, verbose=False)),
            ("custom_bmps", calculate_CMI_bMPS,
             dict(L=L, p=p, r=r, num_samples=N, max_bond=chi, seed=seed, verbose=False)),
            ("quimb_bmps", calculate_CMI_quimb,
             dict(L=L, p=p, r=r, num_samples=N, max_bond=chi, seed=seed, verbose=False)),
        ]:
            t0 = time.perf_counter()
            cmi = fn(**kw)
            elapsed = time.perf_counter() - t0

            row[f"cmi_{name}"] = cmi
            row[f"t_{name}_s"] = elapsed

            if verbose:
                print(f"  {name:<15} CMI={cmi:.6f} nats  ({elapsed:.1f}s, "
                      f"{1e3 * elapsed / N:.1f} ms/sample)")

        row["diff_custom_vs_exact"] = (row["cmi_custom_bmps"] - row["cmi_exact"]
                                       if row["cmi_exact"] is not None else float("nan"))
        row["diff_quimb_vs_exact"]  = (row["cmi_quimb_bmps"] - row["cmi_exact"]
                                       if row["cmi_exact"] is not None else float("nan"))
        results.append(row)

    return results


# ---------------------------------------------------------------------------
# Large-r bond scan (P8.3 C)
# ---------------------------------------------------------------------------

def run_bond_scan(cases, verbose=True):
    """
    Compare custom bMPS and quimb at different chi for large r.

    cases : list of (L, p, r, num_samples, chi_list, seed)
    Returns list of result dicts.
    """
    from src.core.cmi_calculation import calculate_CMI_bMPS
    from src.core.quimb_contraction import calculate_CMI_quimb

    results = []

    for L, p, r, N, chi_list, seed in cases:
        if verbose:
            print(f"\n=== Bond scan: L={L} p={p} r={r} N={N} ===")

        for chi in chi_list:
            for name, fn in [("custom_bmps", calculate_CMI_bMPS),
                              ("quimb_bmps",  calculate_CMI_quimb)]:
                t0 = time.perf_counter()
                cmi = fn(L=L, p=p, r=r, num_samples=N,
                         max_bond=chi, seed=seed, verbose=False)
                elapsed = time.perf_counter() - t0

                row = {
                    "backend": name, "L": L, "p": p, "r": r,
                    "num_samples": N, "max_bond": chi, "seed": seed,
                    "cmi": cmi, "elapsed_s": elapsed,
                    "ms_per_sample": 1e3 * elapsed / N,
                }
                results.append(row)

                if verbose:
                    print(f"  chi={chi:3d}  {name:<14} CMI={cmi:.6f}  "
                          f"({elapsed:.0f}s, {1e3*elapsed/N:.0f} ms/sample)")

    return results


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Use minimal samples/chi for a fast test run")
    args = parser.parse_args()

    SEED = 42

    if args.quick:
        # Fast mode: small N, fewer chi values
        region_cases = [(10, 0.10, 2, 16), (14, 0.11, 3, 16)]
        cmi_cases = [
            (10, 0.10, 2, 50, 16, SEED),
            (14, 0.11, 3, 50, 16, SEED),
        ]
        bond_cases = [
            (18, 0.11, 4, 20, [16, 32], SEED),
        ]
    else:
        # Full benchmark per P8.3 verification matrix
        region_cases = [
            (10, 0.10, 2, 16),
            (14, 0.11, 3, 16),
        ]
        cmi_cases = [
            (10, 0.10, 2, 200, 16, SEED),
            (10, 0.10, 2, 200, 16, SEED + 1),
            (14, 0.11, 3, 200, 16, SEED),
            (14, 0.11, 3, 200, 16, SEED + 1),
        ]
        bond_cases = [
            (18, 0.11, 4, 50, [16, 32],     SEED),
            (22, 0.11, 5, 20, [16, 32, 64], SEED),
            (26, 0.11, 6, 10, [16, 32, 64], SEED),
        ]

    print("=" * 60)
    print("quimb backend benchmark")
    print("=" * 60)

    # A: Region-level accuracy
    print("\n[A] Region-level accuracy")
    region_rows = run_region_benchmark(region_cases, seed=SEED)
    write_csv(OUT_REGION, region_rows)

    # B: CMI regression
    print("\n[B] CMI-level regression")
    cmi_rows = run_cmi_benchmark(cmi_cases)
    write_csv(OUT_CMI, cmi_rows)

    # C: Large-r bond scan
    print("\n[C] Large-r practicality scan")
    bond_rows = run_bond_scan(bond_cases)
    write_csv(OUT_BOND, bond_rows)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if region_rows:
        n_better = sum(1 for r in region_rows if r["rel_err_quimb"] < r["rel_err_custom"] * 0.99)
        n_total  = len(region_rows)
        mean_speedup = (sum(r["t_custom_s"] for r in region_rows) /
                        max(sum(r["t_quimb_s"] for r in region_rows), 1e-9))
        print(f"Region accuracy: quimb better in {n_better}/{n_total} cases")
        print(f"Speed: custom is {1/mean_speedup:.1f}x faster than quimb")

    if cmi_rows:
        for row in cmi_rows:
            ex = row.get("cmi_exact")
            cb = row.get("cmi_custom_bmps")
            qb = row.get("cmi_quimb_bmps")
            if None not in (ex, cb, qb):
                dc = abs(cb - ex)
                dq = abs(qb - ex)
                better = "quimb" if dq < dc else "custom"
                print(f"  L={row['L']} p={row['p']} r={row['r']} chi={row['max_bond']} "
                      f"seed={row['seed']}: "
                      f"exact={ex:.5f}  custom={cb:.5f}  quimb={qb:.5f}  "
                      f"closer: {better}")

    if bond_rows:
        print("\nLarge-r bond scan (CMI values):")
        from itertools import groupby
        by_case = {}
        for row in bond_rows:
            key = (row["L"], row["p"], row["r"])
            by_case.setdefault(key, []).append(row)
        for key, rows in sorted(by_case.items()):
            L, p, r = key
            print(f"  L={L} p={p} r={r}:")
            for row in rows:
                print(f"    chi={row['max_bond']:3d}  {row['backend']:<14} "
                      f"CMI={row['cmi']:.5f}  {row['ms_per_sample']:.0f} ms/sample")

    print("\nDecision criteria (P8.6):")
    print("  Conclusion A (upgrade to 2nd backend): quimb better at r>=5 with <=3x slowdown")
    print("  Conclusion B (validation oracle only): small system stable but large is slow")
    print("  Conclusion C (terminate): small system already unstable")


if __name__ == "__main__":
    main()
