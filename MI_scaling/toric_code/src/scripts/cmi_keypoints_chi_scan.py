"""
P5-A1: keypoint chi convergence scan.

Points:
- p in {0.11, 0.15}
- r in {5, 6}

Grid:
- chi in {16, 32, 64}
- num_samples = 500

Outputs:
- outputs/CMI_keypoints_chi_scan.csv
- outputs/CMI_keypoints_chi_scan.png
"""

import csv
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.cmi_calculation import calculate_CMI_bMPS

ROOT = Path(__file__).resolve().parents[2]
OUT_CSV = ROOT / "outputs" / "CMI_keypoints_chi_scan.csv"
OUT_PNG = ROOT / "outputs" / "CMI_keypoints_chi_scan.png"

P_VALUES = [0.11, 0.15]
R_VALUES = [5, 6]
CHI_VALUES = [16, 32, 64]
NUM_SAMPLES = 500
BASE_SEED = 2026


def stable_seed(p, r, chi):
    return int(BASE_SEED + 10000 * p + 100 * r + chi)


def load_cache():
    out = {}
    if not OUT_CSV.exists():
        return out
    with open(OUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["p"]), int(row["r"]), int(row["max_bond"]))
            out[key] = row
    return out


def append_row(row):
    exists = OUT_CSV.exists()
    fields = ["p", "r", "L", "max_bond", "num_samples", "seed", "cmi", "elapsed_sec"]
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def load_points():
    rows = []
    if not OUT_CSV.exists():
        return rows
    with open(OUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def plot(rows):
    by_point = {}
    for row in rows:
        p = float(row["p"])
        r = int(row["r"])
        chi = int(row["max_bond"])
        cmi = float(row["cmi"])
        by_point.setdefault((p, r), {})[chi] = cmi

    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    styles = {
        (0.11, 5): ("#d62728", "o"),
        (0.11, 6): ("#d62728", "s"),
        (0.15, 5): ("#2ca02c", "o"),
        (0.15, 6): ("#2ca02c", "s"),
    }

    for key in sorted(by_point.keys()):
        p, r = key
        color, marker = styles[key]
        xs = []
        ys = []
        for chi in CHI_VALUES:
            if chi in by_point[key]:
                xs.append(chi)
                ys.append(by_point[key][chi])
        if xs:
            ax.plot(xs, ys, color=color, marker=marker, lw=1.8, ms=6, label=f"p={p:.2f}, r={r}")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("max_bond (chi)")
    ax.set_ylabel("CMI [nats]")
    ax.set_title("Keypoints chi scan")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved plot: {OUT_PNG}", flush=True)


def main():
    cache = load_cache()
    total = len(P_VALUES) * len(R_VALUES) * len(CHI_VALUES)
    done = len(cache)

    if done:
        print(f"Loaded {done}/{total} cached chi-scan points from {OUT_CSV}", flush=True)

    for p in P_VALUES:
        for r in R_VALUES:
            L = 2 + 4 * r
            for chi in CHI_VALUES:
                key = (p, r, chi)
                if key in cache:
                    print(f"[cached] p={p:.2f}, r={r}, chi={chi}, cmi={float(cache[key]['cmi']):.6f}", flush=True)
                    continue

                seed = stable_seed(p, r, chi)
                print(f"[run] p={p:.2f}, r={r}, L={L}, chi={chi}, N={NUM_SAMPLES}, seed={seed}", flush=True)
                t0 = time.perf_counter()
                cmi = calculate_CMI_bMPS(
                    L=L,
                    p=p,
                    r=r,
                    num_samples=NUM_SAMPLES,
                    max_bond=chi,
                    seed=seed,
                    verbose=False,
                )
                elapsed = time.perf_counter() - t0

                row = {
                    "p": f"{p:.6f}",
                    "r": r,
                    "L": L,
                    "max_bond": chi,
                    "num_samples": NUM_SAMPLES,
                    "seed": seed,
                    "cmi": f"{cmi:.16g}",
                    "elapsed_sec": f"{elapsed:.3f}",
                }
                append_row(row)
                cache[key] = row
                print(f"  -> cmi={cmi:.6f}, elapsed={elapsed:.1f}s", flush=True)

    rows = load_points()
    plot(rows)

    print("\nchi-scan table:", flush=True)
    for p in P_VALUES:
        for r in R_VALUES:
            vals = []
            for chi in CHI_VALUES:
                row = cache[(p, r, chi)]
                vals.append((chi, float(row["cmi"])))
            serial = ", ".join([f"chi={chi}:{val:.6f}" for chi, val in vals])
            span = max(v for _, v in vals) - min(v for _, v in vals)
            print(f"  p={p:.2f}, r={r}: {serial}; span={span:.6g}", flush=True)


if __name__ == "__main__":
    main()
