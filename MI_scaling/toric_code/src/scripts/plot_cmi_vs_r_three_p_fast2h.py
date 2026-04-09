"""
Fast ~2h strategy for CMI vs r with single-seed, 5000 samples per point.

Configuration requested:
- p in {0.05, 0.11, 0.15}
- r in {1,2,3,4,5,6}
- single seed per point
- num_samples = 5000 for every point
- r<=3 exact TN
- r>=4 bMPS with chi=16
- multiprocessing parallel execution

Outputs:
- CMI_vs_r_three_p_fast2h_raw.csv
- CMI_vs_r_three_p_fast2h_summary.csv
- CMI_vs_r_p005_011_015_fast2h.png
"""

import csv
import math
import os
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np

from src.core.cmi_calculation import calculate_CMI, calculate_CMI_bMPS


R_VALUES = [1, 2, 3, 4, 5, 6]
P_VALUES = [0.05, 0.11, 0.15]
BASE_SEED = 2026
NUM_SAMPLES = 5000
WORKERS = min(4, os.cpu_count() or 1)

RAW_CSV = str((Path(__file__).resolve().parents[2] / "outputs" / "CMI_vs_r_three_p_fast2h_raw.csv"))
SUMMARY_CSV = str((Path(__file__).resolve().parents[2] / "outputs" / "CMI_vs_r_three_p_fast2h_summary.csv"))
OUT_PNG = str((Path(__file__).resolve().parents[2] / "outputs" / "CMI_vs_r_p005_011_015_fast2h.png"))


def make_seed(p, r):
    return int(BASE_SEED + 10000 * p + 100 * r)


def point_config(p, r):
    L = 2 + 4 * r
    if r <= 3:
        return {
            "method": "exact",
            "L": L,
            "max_bond": "",
        }
    return {
        "method": "bmps",
        "L": L,
        "max_bond": 16,
    }


def run_one_point(task):
    p, r = task
    cfg = point_config(p, r)
    seed = make_seed(p, r)

    t0 = time.perf_counter()
    if cfg["method"] == "exact":
        cmi = calculate_CMI(
            L=cfg["L"],
            p=p,
            r=r,
            num_samples=NUM_SAMPLES,
            seed=seed,
            verbose=False,
        )
    else:
        cmi = calculate_CMI_bMPS(
            L=cfg["L"],
            p=p,
            r=r,
            num_samples=NUM_SAMPLES,
            max_bond=cfg["max_bond"],
            seed=seed,
            verbose=False,
        )

    elapsed = time.perf_counter() - t0
    return {
        "p": p,
        "r": r,
        "seed": seed,
        "cmi": cmi,
        "method": cfg["method"],
        "L": cfg["L"],
        "num_samples": NUM_SAMPLES,
        "max_bond": cfg["max_bond"],
        "elapsed_sec": elapsed,
    }


def load_existing_raw():
    out = {}
    if not os.path.exists(RAW_CSV):
        return out

    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["p"]), int(row["r"]))
            out[key] = {
                "p": float(row["p"]),
                "r": int(row["r"]),
                "seed": int(row["seed"]),
                "cmi": float(row["cmi"]),
                "method": row["method"],
                "L": int(row["L"]),
                "num_samples": int(row["num_samples"]),
                "max_bond": row["max_bond"],
                "elapsed_sec": float(row["elapsed_sec"]),
            }
    return out


def save_raw(points):
    fields = [
        "p", "r", "seed", "cmi", "method", "L",
        "num_samples", "max_bond", "elapsed_sec",
    ]
    with open(RAW_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for p, r in sorted(points.keys()):
            row = points[(p, r)]
            w.writerow({
                "p": f"{row['p']:.6f}",
                "r": row["r"],
                "seed": row["seed"],
                "cmi": f"{row['cmi']:.16g}",
                "method": row["method"],
                "L": row["L"],
                "num_samples": row["num_samples"],
                "max_bond": row["max_bond"],
                "elapsed_sec": f"{row['elapsed_sec']:.3f}",
            })


def save_summary(points):
    fields = ["p", "r", "count", "cmi_mean", "cmi_std", "cmi_sem"]
    with open(SUMMARY_CSV, "w", newline="") as f:
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

    # p != p_c: log(CMI)=a+br
    for p in [0.05, 0.15]:
        xs = []
        ys = []
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

    # p = p_c: log(CMI)=c-alpha*log(r)
    p = 0.11
    xs = []
    ys = []
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
        xs = []
        ys = []
        for r in R_VALUES:
            xs.append(r)
            ys.append(float(points[(p, r)]["cmi"]))
        ax.plot(xs, ys, color=colors[p], marker=markers[p], lw=1.8, ms=6, label=f"p={p:.2f}")

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("CMI = I(A:C|B) [nats]", fontsize=12)
    ax.set_title("CMI vs r (fast2h, 1 seed, 5000 samples/point)", fontsize=12)
    ax.set_xticks(R_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved plot: {OUT_PNG}", flush=True)


def main():
    print(f"Workers={WORKERS}, num_samples per point={NUM_SAMPLES}", flush=True)

    points = load_existing_raw()
    tasks = []
    for p in P_VALUES:
        for r in R_VALUES:
            if (p, r) not in points:
                tasks.append((p, r))

    total = len(P_VALUES) * len(R_VALUES)
    done = len(points)
    if done:
        print(f"Loaded {done}/{total} cached points from {RAW_CSV}", flush=True)

    if tasks:
        print(f"Running {len(tasks)} missing points in parallel...", flush=True)
        with ProcessPoolExecutor(max_workers=WORKERS) as ex:
            future_map = {ex.submit(run_one_point, t): t for t in tasks}
            for fut in as_completed(future_map):
                p, r = future_map[fut]
                row = fut.result()
                points[(p, r)] = row
                done += 1
                print(
                    f"[done {done}/{total}] p={p:.2f}, r={r}, CMI={row['cmi']:.6f}, "
                    f"{row['method']}, L={row['L']}, N={row['num_samples']}, "
                    f"chi={row['max_bond']}, {row['elapsed_sec']:.1f}s",
                    flush=True,
                )
                save_raw(points)

    if len(points) != total:
        raise RuntimeError(f"Expected {total} points, got {len(points)}")

    save_raw(points)
    save_summary(points)
    make_plot(points)

    print("\nFinal table:", flush=True)
    for p in P_VALUES:
        for r in R_VALUES:
            row = points[(p, r)]
            print(
                f"  p={p:.2f}, r={r}: CMI={row['cmi']:.6f} "
                f"({row['method']}, L={row['L']}, N={row['num_samples']}, chi={row['max_bond']})",
                flush=True,
            )

    fits = fit_eq12(points)
    print("\nEq.(12) fit diagnostics:", flush=True)
    b, r2, n = fits[(0.05, "exp")]
    print(f"  p=0.05 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}", flush=True)
    b, r2, n = fits[(0.15, "exp")]
    print(f"  p=0.15 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}", flush=True)
    alpha, r2, n = fits[(0.11, "power")]
    print(f"  p=0.11 power-law: alpha={alpha:.4f}, R^2={r2:.3f}, points={n}", flush=True)


if __name__ == "__main__":
    main()
