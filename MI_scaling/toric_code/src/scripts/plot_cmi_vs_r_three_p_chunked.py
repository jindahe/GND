"""
P5-B1 / P6: chunk-parallel production run for CMI vs r.

Range:
- p in {0.05, 0.11, 0.15}
- r in {1, 2, 3, 4, 5, 6}

Defaults:
- total samples per (p, r): 5000
- chunk size: 250
- method: exact for r<=3, bMPS(chi=16) for r>=4

Outputs:
- outputs/CMI_chunk_raw.csv
- outputs/CMI_chunk_summary.csv
- outputs/CMI_vs_r_p005_011_015_chunked.png
"""

import argparse
import csv
import math
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.core.cmi_calculation import calculate_CMI, calculate_CMI_bMPS

ROOT = Path(__file__).resolve().parents[2]
OUT_RAW = ROOT / "outputs" / "CMI_chunk_raw.csv"
OUT_SUMMARY = ROOT / "outputs" / "CMI_chunk_summary.csv"
OUT_PNG = ROOT / "outputs" / "CMI_vs_r_p005_011_015_chunked.png"

P_VALUES = [0.05, 0.11, 0.15]
R_VALUES = [1, 2, 3, 4, 5, 6]


def stable_seed(base_seed, p, r, chunk_id):
    return int(base_seed + 10000 * p + 100 * r + 1000000 * chunk_id)


def point_config(r, max_bond):
    L = 2 + 4 * r
    if r <= 3:
        return {"method": "exact", "L": L, "max_bond": ""}
    return {"method": "bmps", "L": L, "max_bond": int(max_bond)}


def run_one_chunk(task):
    p, r, chunk_id, chunk_samples, base_seed, max_bond = task
    cfg = point_config(r, max_bond)
    seed = stable_seed(base_seed, p, r, chunk_id)

    t0 = time.perf_counter()
    if cfg["method"] == "exact":
        cmi = calculate_CMI(
            L=cfg["L"],
            p=p,
            r=r,
            num_samples=chunk_samples,
            seed=seed,
            verbose=False,
        )
    else:
        cmi = calculate_CMI_bMPS(
            L=cfg["L"],
            p=p,
            r=r,
            num_samples=chunk_samples,
            max_bond=cfg["max_bond"],
            seed=seed,
            verbose=False,
        )
    elapsed = time.perf_counter() - t0

    return {
        "p": p,
        "r": r,
        "L": cfg["L"],
        "chunk_id": chunk_id,
        "chunk_samples": chunk_samples,
        "seed": seed,
        "method": cfg["method"],
        "max_bond": cfg["max_bond"],
        "cmi": cmi,
        "elapsed_sec": elapsed,
    }


def load_raw_cache():
    cache = {}
    if not OUT_RAW.exists():
        return cache

    with open(OUT_RAW, newline="") as f:
        for row in csv.DictReader(f):
            key = (float(row["p"]), int(row["r"]), int(row["chunk_id"]))
            cache[key] = {
                "p": float(row["p"]),
                "r": int(row["r"]),
                "L": int(row["L"]),
                "chunk_id": int(row["chunk_id"]),
                "chunk_samples": int(row["chunk_samples"]),
                "seed": int(row["seed"]),
                "method": row["method"],
                "max_bond": row["max_bond"],
                "cmi": float(row["cmi"]),
                "elapsed_sec": float(row["elapsed_sec"]),
            }
    return cache


def append_raw(row):
    exists = OUT_RAW.exists()
    fields = [
        "p", "r", "L", "chunk_id", "chunk_samples", "seed",
        "method", "max_bond", "cmi", "elapsed_sec",
    ]
    with open(OUT_RAW, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow({
            "p": f"{row['p']:.6f}",
            "r": row["r"],
            "L": row["L"],
            "chunk_id": row["chunk_id"],
            "chunk_samples": row["chunk_samples"],
            "seed": row["seed"],
            "method": row["method"],
            "max_bond": row["max_bond"],
            "cmi": f"{row['cmi']:.16g}",
            "elapsed_sec": f"{row['elapsed_sec']:.3f}",
        })


def summarize(cache):
    rows = []
    by_point = defaultdict(list)
    for (_p, _r, _cid), row in cache.items():
        by_point[(row["p"], row["r"])].append(row)

    for p in P_VALUES:
        for r in R_VALUES:
            pts = by_point.get((p, r), [])
            if not pts:
                rows.append({
                    "p": f"{p:.6f}",
                    "r": r,
                    "count_chunks": 0,
                    "total_samples": 0,
                    "cmi_mean": "nan",
                    "cmi_std": "nan",
                    "cmi_sem": "nan",
                })
                continue

            vals = np.array([float(t["cmi"]) for t in pts], dtype=float)
            weights = np.array([int(t["chunk_samples"]) for t in pts], dtype=float)
            total_samples = int(np.sum(weights))
            wmean = float(np.sum(weights * vals) / np.sum(weights))

            std = float(np.std(vals, ddof=1)) if vals.size > 1 else 0.0
            sem = std / math.sqrt(vals.size) if vals.size > 1 else 0.0

            rows.append({
                "p": f"{p:.6f}",
                "r": r,
                "count_chunks": int(vals.size),
                "total_samples": total_samples,
                "cmi_mean": f"{wmean:.16g}",
                "cmi_std": f"{std:.16g}",
                "cmi_sem": f"{sem:.16g}",
            })

    return rows


def save_summary(rows):
    fields = ["p", "r", "count_chunks", "total_samples", "cmi_mean", "cmi_std", "cmi_sem"]
    with open(OUT_SUMMARY, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def summary_map(rows):
    out = {}
    for row in rows:
        p = float(row["p"])
        r = int(row["r"])
        out[(p, r)] = {
            "count_chunks": int(row["count_chunks"]),
            "total_samples": int(row["total_samples"]),
            "mean": float(row["cmi_mean"]) if row["cmi_mean"] != "nan" else float("nan"),
            "sem": float(row["cmi_sem"]) if row["cmi_sem"] != "nan" else float("nan"),
        }
    return out


def fit_eq12(sm):
    result = {}

    for p in [0.05, 0.15]:
        xs, ys = [], []
        for r in R_VALUES:
            y = sm[(p, r)]["mean"]
            if np.isfinite(y) and y > 0:
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
            result[(p, "exp")] = (float(slope), float(r2), len(xs), float(intercept))
        else:
            result[(p, "exp")] = (float("nan"), float("nan"), len(xs), float("nan"))

    p = 0.11
    xs, ys = [], []
    for r in R_VALUES:
        y = sm[(p, r)]["mean"]
        if np.isfinite(y) and y > 0:
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
        result[(p, "power")] = (alpha, float(r2), len(xs), float(intercept))
    else:
        result[(p, "power")] = (float("nan"), float("nan"), len(xs), float("nan"))

    return result


def make_plot(sm, fits):
    fig, ax = plt.subplots(figsize=(7.8, 5.4))
    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    pos_xs_by_p = {}
    for p in P_VALUES:
        xs, ys, es = [], [], []
        for r in R_VALUES:
            rec = sm[(p, r)]
            y = rec["mean"]
            if (not np.isfinite(y)) or (y <= 0):
                continue
            err = rec["sem"] if np.isfinite(rec["sem"]) else 0.0
            err = min(err, y * 0.95)
            xs.append(r)
            ys.append(y)
            es.append(err)

        pos_xs_by_p[p] = xs
        if xs:
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                fmt=markers[p],
                linestyle="none",
                color=colors[p],
                markersize=6,
                capsize=3,
                elinewidth=1.0,
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
    ax.set_xlabel("r")
    ax.set_ylabel("CMI = I(A:C|B) [nats, log; CMI>0 only]")
    ax.set_xticks(R_VALUES)
    ax.set_title("CMI vs r (chunked production, log y-axis, positive-only)")
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


def build_tasks(total_samples, chunk_size, base_seed, max_bond):
    tasks = []
    for p in P_VALUES:
        for r in R_VALUES:
            remaining = int(total_samples)
            chunk_id = 0
            while remaining > 0:
                this_chunk = min(int(chunk_size), remaining)
                tasks.append((p, r, chunk_id, this_chunk, int(base_seed), int(max_bond)))
                remaining -= this_chunk
                chunk_id += 1
    return tasks


def main():
    parser = argparse.ArgumentParser(description="Chunked CMI-vs-r run")
    parser.add_argument("--workers", type=int, default=min(96, os.cpu_count() or 1))
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--chunk", type=int, default=250)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--max-bond", type=int, default=16)
    args = parser.parse_args()

    if args.chunk <= 0 or args.samples <= 0:
        raise ValueError("--samples and --chunk must be positive")

    all_tasks = build_tasks(args.samples, args.chunk, args.base_seed, args.max_bond)
    cache = load_raw_cache()

    tasks = []
    for (p, r, chunk_id, chunk_samples, base_seed, max_bond) in all_tasks:
        key = (p, r, chunk_id)
        if key not in cache:
            tasks.append((p, r, chunk_id, chunk_samples, base_seed, max_bond))

    print(
        f"workers={args.workers}, points={len(P_VALUES)*len(R_VALUES)}, "
        f"chunks/point={math.ceil(args.samples/args.chunk)}, total_chunks={len(all_tasks)}, pending={len(tasks)}",
        flush=True,
    )

    done = len(cache)
    total = len(all_tasks)

    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(run_one_chunk, t): t for t in tasks}
            for fut in as_completed(futures):
                row = fut.result()
                key = (row["p"], row["r"], row["chunk_id"])
                cache[key] = row
                append_raw(row)
                done += 1
                print(
                    f"[done {done}/{total}] p={row['p']:.2f}, r={row['r']}, chunk={row['chunk_id']}, "
                    f"N={row['chunk_samples']}, cmi={row['cmi']:.6f}, {row['method']}, "
                    f"chi={row['max_bond']}, {row['elapsed_sec']:.1f}s",
                    flush=True,
                )

    rows = summarize(cache)
    save_summary(rows)
    sm = summary_map(rows)
    fits = fit_eq12(sm)
    make_plot(sm, fits)

    print("\nSummary table (mean ± sem):", flush=True)
    for p in P_VALUES:
        for r in R_VALUES:
            rec = sm[(p, r)]
            print(
                f"  p={p:.2f}, r={r}: {rec['mean']:.6f} ± {rec['sem']:.6f} "
                f"(chunks={rec['count_chunks']}, N={rec['total_samples']})",
                flush=True,
            )

    print("\nEq.(12) fit diagnostics (chunked):", flush=True)
    b, r2, n, _ = fits[(0.05, "exp")]
    print(f"  p=0.05 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}", flush=True)
    b, r2, n, _ = fits[(0.15, "exp")]
    print(f"  p=0.15 exponential: b={b:.4f}, R^2={r2:.3f}, points={n}", flush=True)
    alpha, r2, n, _ = fits[(0.11, "power")]
    print(f"  p=0.11 power-law: alpha={alpha:.4f}, R^2={r2:.3f}, points={n}", flush=True)


if __name__ == "__main__":
    main()
