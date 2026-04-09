"""
Refine 4 critical large-r points on top of fast2h results.

Base input:
- outputs/CMI_vs_r_three_p_fast2h_raw.csv

Refined points (recomputed):
- (p=0.11, r=5), (p=0.11, r=6)
- (p=0.15, r=5), (p=0.15, r=6)
using bMPS chi=32 with num_samples=500.

Outputs:
- outputs/CMI_vs_r_three_p_fast2h_refined_raw.csv
- outputs/CMI_vs_r_three_p_fast2h_refined_summary.csv
- outputs/CMI_vs_r_p005_011_015_fast2h_refined.png
"""

import argparse
import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.cmi_calculation import calculate_CMI_bMPS

ROOT = Path(__file__).resolve().parents[2]
BASE_RAW = ROOT / "outputs" / "CMI_vs_r_three_p_fast2h_raw.csv"
OUT_RAW = ROOT / "outputs" / "CMI_vs_r_three_p_fast2h_refined_raw.csv"
OUT_SUMMARY = ROOT / "outputs" / "CMI_vs_r_three_p_fast2h_refined_summary.csv"
OUT_PNG = ROOT / "outputs" / "CMI_vs_r_p005_011_015_fast2h_refined.png"

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
            slope, _intercept = np.polyfit(x, ly, 1)
            pred = slope * x + _intercept
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            fits[(p, "exp")] = (float(slope), float(r2), len(xs), float(_intercept))
        else:
            fits[(p, "exp")] = (float("nan"), float("nan"), len(xs), float("nan"))

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
        slope, _intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + _intercept
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        alpha = -float(slope)
        fits[(p, "power")] = (alpha, float(r2), len(xs), float(_intercept))
    else:
        fits[(p, "power")] = (float("nan"), float("nan"), len(xs), float("nan"))

    return fits


def make_plot(points, fits):
    fig, ax = plt.subplots(figsize=(7.8, 5.3))
    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    pos_xs_by_p = {}
    for p in P_VALUES:
        xs, ys = [], []
        for r in R_VALUES:
            y = float(points[(p, r)]["cmi"])
            if y <= 0:
                continue
            xs.append(r)
            ys.append(y)
        pos_xs_by_p[p] = xs
        if xs:
            ax.plot(
                xs,
                ys,
                color=colors[p],
                marker=markers[p],
                linestyle="none",
                ms=6,
                label=f"p={p:.2f} data",
            )

    for p in [0.05, 0.15]:
        b, _r2, n, intercept = fits[(p, "exp")]
        xs = pos_xs_by_p.get(p, [])
        if np.isfinite(b) and np.isfinite(intercept) and n >= 3 and len(xs) >= 2:
            x_fit = np.linspace(min(xs), max(xs), 200)
            y_fit = np.exp(intercept + b * x_fit)
            ax.plot(x_fit, y_fit, color=colors[p], lw=1.8, linestyle="--", label=f"p={p:.2f} fit")

    alpha, _r2, n, log_pref = fits[(0.11, "power")]
    xs = pos_xs_by_p.get(0.11, [])
    if np.isfinite(alpha) and np.isfinite(log_pref) and n >= 3 and len(xs) >= 2:
        x_fit = np.linspace(min(xs), max(xs), 200)
        y_fit = np.exp(log_pref) * np.power(x_fit, -alpha)
        ax.plot(x_fit, y_fit, color=colors[0.11], lw=1.8, linestyle="--", label="p=0.11 fit")

    ax.set_yscale("log")
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("CMI = I(A:C|B) [nats, log; CMI>0 only]", fontsize=12)
    ax.set_title("CMI vs r (refined, log y-axis, positive-only)", fontsize=12)
    ax.set_xticks(R_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()

    def _fmt(v):
        return "nan" if not np.isfinite(v) else f"{v:.3f}"

    b005, r2_005, _, _ = fits[(0.05, "exp")]
    b015, r2_015, _, _ = fits[(0.15, "exp")]
    alpha, r2_011, _, _ = fits[(0.11, "power")]
    fit_text = "\n".join(
        [
            "Eq.(12) fits:",
            f"p=0.05: b={_fmt(b005)}, R^2={_fmt(r2_005)}",
            f"p=0.15: b={_fmt(b015)}, R^2={_fmt(r2_015)}",
            f"p=0.11: alpha={_fmt(alpha)}, R^2={_fmt(r2_011)}",
        ]
    )
    ax.text(
        0.02,
        0.98,
        fit_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved plot: {OUT_PNG}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Refine critical points and plot")
    parser.add_argument("--plot-only", action="store_true", help="Only regenerate plot from existing refined raw CSV")
    args = parser.parse_args()

    expected = len(P_VALUES) * len(R_VALUES)

    if args.plot_only:
        if not OUT_RAW.exists():
            raise FileNotFoundError(f"Refined raw file not found for --plot-only: {OUT_RAW}")
        points = build_map(load_raw(OUT_RAW))
        if len(points) != expected:
            raise RuntimeError(f"Need {expected} refined points in {OUT_RAW}, got {len(points)}")
    else:
        if not BASE_RAW.exists():
            raise FileNotFoundError(f"Base raw file not found: {BASE_RAW}")

        base_rows = load_raw(BASE_RAW)
        points = build_map(base_rows)

        if len(points) != expected:
            raise RuntimeError(f"Need {expected} base points, got {len(points)}")

        for p, r in TARGETS:
            L = 2 + 4 * r
            seed = make_seed(p, r)
            print(
                f"Refining p={p:.2f}, r={r}, L={L}, chi={REFINE_CHI}, N={REFINE_NUM_SAMPLES}, seed={seed} ...",
                flush=True,
            )
            t0 = time.perf_counter()
            cmi = calculate_CMI_bMPS(
                L=L,
                p=p,
                r=r,
                num_samples=REFINE_NUM_SAMPLES,
                max_bond=REFINE_CHI,
                seed=seed,
                verbose=False,
            )
            elapsed = time.perf_counter() - t0

            row = points[(p, r)]
            row["seed"] = str(seed)
            row["cmi"] = f"{cmi:.16g}"
            row["method"] = "bmps"
            row["L"] = str(L)
            row["num_samples"] = str(REFINE_NUM_SAMPLES)
            row["max_bond"] = str(REFINE_CHI)
            row["elapsed_sec"] = f"{elapsed:.3f}"
            print(f"  -> refined CMI={cmi:.6f}, elapsed={elapsed:.1f}s", flush=True)

        out_rows = []
        for p in P_VALUES:
            for r in R_VALUES:
                out_rows.append(points[(p, r)])
        save_raw(OUT_RAW, out_rows)

    save_summary(points)
    fits = fit_eq12(points)
    make_plot(points, fits)

    print("\nRefined table:", flush=True)
    for p in P_VALUES:
        for r in R_VALUES:
            print(f"  p={p:.2f}, r={r}: {float(points[(p, r)]['cmi']):.6f}", flush=True)

    print("\nEq.(12) fit diagnostics (refined):", flush=True)
    b, r2, n, _ = fits[(0.05, "exp")]
    print(f"  p=0.05 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}", flush=True)
    b, r2, n, _ = fits[(0.15, "exp")]
    print(f"  p=0.15 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}", flush=True)
    alpha, r2, n, _ = fits[(0.11, "power")]
    print(f"  p=0.11 power-law: alpha={alpha:.4f}, R^2={r2:.3f}, points={n}", flush=True)


if __name__ == "__main__":
    main()
