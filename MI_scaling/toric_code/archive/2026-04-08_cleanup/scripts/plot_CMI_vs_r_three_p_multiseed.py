"""
Stage-1 stratified, multi-seed CMI-vs-r run for p in {0.05, 0.11, 0.15}.

Goal:
- Improve robustness over single-seed quick run.
- Keep runtime practical on a single machine.

Outputs:
- CMI_vs_r_three_p_multiseed_raw.csv      (all per-seed points)
- CMI_vs_r_three_p_multiseed_summary.csv  (mean/std/sem per (p,r))
- CMI_vs_r_p005_011_015_multiseed.png     (error-bar figure)
"""

import csv
import math
import os
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from CMI_calculation import calculate_CMI, calculate_CMI_bMPS


R_VALUES = [1, 2, 3, 4, 5, 6]
P_VALUES = [0.05, 0.11, 0.15]
SEEDS = [2026, 2027]

RAW_CSV = "CMI_vs_r_three_p_multiseed_raw.csv"
SUMMARY_CSV = "CMI_vs_r_three_p_multiseed_summary.csv"
OUT_PNG = "CMI_vs_r_p005_011_015_multiseed.png"


def point_config(r, p):
    """Practical stratified budget for stage-1."""
    L = 2 + 4 * r

    if r <= 3:
        return {
            "method": "exact",
            "L": L,
            "num_samples": 250,
            "max_bond": "",
        }

    if r == 4:
        return {
            "method": "exact",
            "L": L,
            "num_samples": 50,
            "max_bond": "",
        }

    # r = 5,6 -> bMPS
    if abs(p - 0.11) < 1e-12:
        return {
            "method": "bmps",
            "L": L,
            "num_samples": 40,
            "max_bond": 32,
        }

    return {
        "method": "bmps",
        "L": L,
        "num_samples": 180,
        "max_bond": 16,
    }


def stable_seed(base_seed, p, r):
    return int(base_seed + 10000 * p + 100 * r)


def load_raw_cache():
    cache = {}
    if not os.path.exists(RAW_CSV):
        return cache

    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (int(row["seed"]), float(row["p"]), int(row["r"]))
            cache[key] = row
    return cache


def append_raw_row(row):
    exists = os.path.exists(RAW_CSV)
    with open(RAW_CSV, "a", newline="") as f:
        fields = [
            "seed", "p", "r", "cmi",
            "method", "L", "num_samples", "max_bond", "elapsed_sec",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def run_all_points():
    cache = load_raw_cache()
    values = defaultdict(list)  # (p,r) -> list[cmi]

    total = len(SEEDS) * len(P_VALUES) * len(R_VALUES)
    done = 0

    for base_seed in SEEDS:
        for p in P_VALUES:
            for r in R_VALUES:
                cfg = point_config(r, p)
                seed = stable_seed(base_seed, p, r)
                cache_key = (seed, p, r)

                if cache_key in cache:
                    cmi = float(cache[cache_key]["cmi"])
                    values[(p, r)].append(cmi)
                    done += 1
                    print(
                        f"[cached {done}/{total}] seed={seed}, p={p:.2f}, r={r}, "
                        f"CMI={cmi:.6f}"
                    )
                    continue

                t0 = time.perf_counter()
                if cfg["method"] == "exact":
                    cmi = calculate_CMI(
                        L=cfg["L"],
                        p=p,
                        r=r,
                        num_samples=cfg["num_samples"],
                        seed=seed,
                        verbose=False,
                    )
                else:
                    cmi = calculate_CMI_bMPS(
                        L=cfg["L"],
                        p=p,
                        r=r,
                        num_samples=cfg["num_samples"],
                        max_bond=cfg["max_bond"],
                        seed=seed,
                        verbose=False,
                    )
                elapsed = time.perf_counter() - t0

                values[(p, r)].append(cmi)
                done += 1

                row = {
                    "seed": seed,
                    "p": f"{p:.6f}",
                    "r": r,
                    "cmi": f"{cmi:.16g}" if cmi is not None else "None",
                    "method": cfg["method"],
                    "L": cfg["L"],
                    "num_samples": cfg["num_samples"],
                    "max_bond": cfg["max_bond"],
                    "elapsed_sec": f"{elapsed:.3f}",
                }
                append_raw_row(row)
                print(
                    f"[new {done}/{total}] seed={seed}, p={p:.2f}, r={r}, CMI={cmi:.6f}, "
                    f"{cfg['method']}, L={cfg['L']}, N={cfg['num_samples']}, "
                    f"chi={cfg['max_bond']}, {elapsed:.1f}s"
                )

    return values


def summarize(values):
    rows = []
    for p in P_VALUES:
        for r in R_VALUES:
            arr = np.array(values.get((p, r), []), dtype=float)
            if arr.size == 0:
                rows.append({
                    "p": f"{p:.6f}",
                    "r": r,
                    "count": 0,
                    "cmi_mean": "nan",
                    "cmi_std": "nan",
                    "cmi_sem": "nan",
                })
                continue

            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
            rows.append({
                "p": f"{p:.6f}",
                "r": r,
                "count": int(arr.size),
                "cmi_mean": f"{mean:.16g}",
                "cmi_std": f"{std:.16g}",
                "cmi_sem": f"{sem:.16g}",
            })
    return rows


def save_summary(rows):
    with open(SUMMARY_CSV, "w", newline="") as f:
        fields = ["p", "r", "count", "cmi_mean", "cmi_std", "cmi_sem"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def load_summary_map(rows):
    out = {}
    for row in rows:
        p = float(row["p"])
        r = int(row["r"])
        out[(p, r)] = {
            "count": int(row["count"]),
            "mean": float(row["cmi_mean"]) if row["cmi_mean"] != "nan" else float("nan"),
            "std": float(row["cmi_std"]) if row["cmi_std"] != "nan" else float("nan"),
            "sem": float(row["cmi_sem"]) if row["cmi_sem"] != "nan" else float("nan"),
        }
    return out


def fit_eq12(summary_map):
    result = {}

    # Off-critical: log(CMI) ~ a + b r
    for p in [0.05, 0.15]:
        pts = []
        for r in R_VALUES:
            y = summary_map[(p, r)]["mean"]
            if np.isfinite(y) and y > 0:
                pts.append((r, y))
        if len(pts) >= 3:
            x = np.array([t[0] for t in pts], dtype=float)
            ly = np.log(np.array([t[1] for t in pts], dtype=float))
            slope, intercept = np.polyfit(x, ly, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            result[(p, "exp")] = (float(slope), float(r2), len(pts))
        else:
            result[(p, "exp")] = (float("nan"), float("nan"), len(pts))

    # Critical: log(CMI) ~ c - alpha log(r)
    p = 0.11
    pts = []
    for r in R_VALUES:
        y = summary_map[(p, r)]["mean"]
        if np.isfinite(y) and y > 0:
            pts.append((r, y))
    if len(pts) >= 3:
        lx = np.log(np.array([t[0] for t in pts], dtype=float))
        ly = np.log(np.array([t[1] for t in pts], dtype=float))
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        alpha = -float(slope)
        result[(p, "power")] = (alpha, float(r2), len(pts))
    else:
        result[(p, "power")] = (float("nan"), float("nan"), len(pts))

    return result


def plot_summary(summary_map):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))

    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    for p in P_VALUES:
        xs, ys, es = [], [], []
        for r in R_VALUES:
            record = summary_map[(p, r)]
            if not np.isfinite(record["mean"]):
                continue
            xs.append(r)
            ys.append(record["mean"])
            es.append(record["sem"] if np.isfinite(record["sem"]) else 0.0)

        ax.errorbar(
            xs, ys, yerr=es,
            color=colors[p], marker=markers[p], lw=1.8, ms=6,
            capsize=3, elinewidth=1.0, label=f"p={p:.2f}",
        )

    ax.axhline(0.0, color="black", lw=0.8)
    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("CMI = I(A:C|B) [nats]", fontsize=12)
    ax.set_xticks(R_VALUES)
    ax.set_title("CMI vs r with SEM (multi-seed stratified run)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved plot: {OUT_PNG}")


def main():
    values = run_all_points()
    rows = summarize(values)
    save_summary(rows)

    summary_map = load_summary_map(rows)
    plot_summary(summary_map)

    print("\nSummary table (mean ± sem):")
    for p in P_VALUES:
        for r in R_VALUES:
            rec = summary_map[(p, r)]
            print(
                f"  p={p:.2f}, r={r}: {rec['mean']:.6f} ± {rec['sem']:.6f} "
                f"(n={rec['count']})"
            )

    fits = fit_eq12(summary_map)
    print("\nEq.(12) fit diagnostics on means:")
    b, r2, n = fits[(0.05, "exp")]
    print(f"  p=0.05 exponential: log(CMI)=a+br, b={b:.4f}, R^2={r2:.3f}, points={n}")
    b, r2, n = fits[(0.15, "exp")]
    print(f"  p=0.15 exponential: log(CMI)=a+br, b={b:.4f}, R^2={r2:.3f}, points={n}")
    alpha, r2, n = fits[(0.11, "power")]
    print(f"  p=0.11 power-law: CMI~r^(-alpha), alpha={alpha:.4f}, R^2={r2:.3f}, points={n}")


if __name__ == "__main__":
    main()
