"""
Refine 4 critical large-r points on top of fast2h results.

Base input:
- CMI_vs_r_three_p_fast2h_raw.csv (1 seed, 5000 samples/point, chi16 for r>=4)

Refined points (recomputed):
- (p=0.11, r=5), (p=0.11, r=6)
- (p=0.15, r=5), (p=0.15, r=6)
using bMPS chi=32 with num_samples=500.

Outputs:
- CMI_vs_r_three_p_fast2h_refined_raw.csv
- CMI_vs_r_three_p_fast2h_refined_summary.csv
- CMI_vs_r_p005_011_015_fast2h_refined.png
"""

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from CMI_calculation import calculate_CMI_bMPS

BASE_RAW = Path("CMI_vs_r_three_p_fast2h_raw.csv")
OUT_RAW = Path("CMI_vs_r_three_p_fast2h_refined_raw.csv")
OUT_SUMMARY = Path("CMI_vs_r_three_p_fast2h_refined_summary.csv")
OUT_PNG = Path("CMI_vs_r_p005_011_015_fast2h_refined.png")

P_VALUES = [0.05, 0.11, 0.15]
R_VALUES = [1, 2, 3, 4, 5, 6]

TARGETS = [(0.11, 5), (0.11, 6), (0.15, 5), (0.15, 6)]
REFINE_NUM_SAMPLES = 500
REFINE_CHI = 32
BASE_SEED = 2026


def make_seed(p, r):
    return int(BASE_SEED + 10000 * p + 100 * r)


def load_raw(path):
    rows = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    return rows


def save_raw(path, rows):
    fields = [
        "p", "r", "seed", "cmi", "method", "L",
        "num_samples", "max_bond", "elapsed_sec",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_map(rows):
    out = {}
    for row in rows:
        key = (float(row["p"]), int(row["r"]))
        out[key] = row
    return out


def to_float(points, p, r):
    return float(points[(p, r)]["cmi"])


def save_summary(points):
    fields = ["p", "r", "count", "cmi_mean", "cmi_std", "cmi_sem"]
    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p in P_VALUES:
            for r in R_VALUES:
                cmi = float(points[(p, r)]["cmi"])
                w.writerow({
                    "p": f"{p:.6f}",
                    "r": r,
                    "count": 1,
                    "cmi_mean": f"{cmi:.16g}",
                    "cmi_std": "0.0",
                    "cmi_sem": "0.0",
                })


def fit_eq12(points):
    fits = {}

    for p in [0.05, 0.15]:
        xs, ys = [], []
        for r in R_VALUES:
            y = float(points[(p, r)]["cmi"])
            if y > 0:
                xs.append(float(r))
                ys.append(float(y))
        if len(xs) >= 3:
            x = np.array(xs)
            ly = np.log(np.array(ys))
            slope, intercept = np.polyfit(x, ly, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fits[(p, "exp")] = (float(slope), float(r2), len(xs))
        else:
            fits[(p, "exp")] = (float("nan"), float("nan"), len(xs))

    p = 0.11
    xs, ys = [], []
    for r in R_VALUES:
        y = float(points[(p, r)]["cmi"])
        if y > 0:
            xs.append(float(r))
            ys.append(float(y))
    if len(xs) >= 3:
        lx = np.log(np.array(xs))
        ly = np.log(np.array(ys))
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        alpha = -float(slope)
        fits[(p, "power")] = (alpha, float(r2), len(xs))
    else:
        fits[(p, "power")] = (float("nan"), float("nan"), len(xs))

    return fits


def make_plot(points):
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    for p in P_VALUES:
        xs, ys = [], []
        for r in R_VALUES:
            xs.append(r)
            ys.append(float(points[(p, r)]["cmi"]))
        ax.plot(xs, ys, color=colors[p], marker=markers[p], lw=1.8, ms=6, label=f"p={p:.2f}")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("CMI = I(A:C|B) [nats]", fontsize=12)
    ax.set_title("CMI vs r (fast2h + chi32 refine on key points)", fontsize=12)
    ax.set_xticks(R_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved plot: {OUT_PNG}")


def main():
    if not BASE_RAW.exists():
        raise FileNotFoundError(f"Base raw file not found: {BASE_RAW}")

    base_rows = load_raw(BASE_RAW)
    points = build_map(base_rows)

    expected = len(P_VALUES) * len(R_VALUES)
    if len(points) != expected:
        raise RuntimeError(f"Need {expected} base points, got {len(points)}")

    for p, r in TARGETS:
        L = 2 + 4 * r
        seed = make_seed(p, r)
        print(f"Refining p={p:.2f}, r={r}, L={L}, chi={REFINE_CHI}, N={REFINE_NUM_SAMPLES} ...")
        cmi = calculate_CMI_bMPS(
            L=L,
            p=p,
            r=r,
            num_samples=REFINE_NUM_SAMPLES,
            max_bond=REFINE_CHI,
            seed=seed,
            verbose=False,
        )
        row = points[(p, r)]
        row["cmi"] = f"{cmi:.16g}"
        row["method"] = "bmps"
        row["L"] = str(L)
        row["num_samples"] = str(REFINE_NUM_SAMPLES)
        row["max_bond"] = str(REFINE_CHI)
        row["elapsed_sec"] = ""
        print(f"  -> refined CMI={cmi:.6f}")

    out_rows = []
    for p in P_VALUES:
        for r in R_VALUES:
            out_rows.append(points[(p, r)])

    save_raw(OUT_RAW, out_rows)
    save_summary(points)
    make_plot(points)

    print("\nRefined table:")
    for p in P_VALUES:
        for r in R_VALUES:
            print(f"  p={p:.2f}, r={r}: {float(points[(p, r)]['cmi']):.6f}")

    fits = fit_eq12(points)
    print("\nEq.(12) fit diagnostics (refined):")
    b, r2, n = fits[(0.05, "exp")]
    print(f"  p=0.05 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}")
    b, r2, n = fits[(0.15, "exp")]
    print(f"  p=0.15 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}")
    alpha, r2, n = fits[(0.11, "power")]
    print(f"  p=0.11 power-law: alpha={alpha:.4f}, R^2={r2:.3f}, points={n}")


if __name__ == "__main__":
    main()
