"""
P5-A2: keypoint multi-seed statistics.

Points:
- p in {0.11, 0.15}
- r in {5, 6}

Config:
- max_bond = 32 (default)
- num_samples = 2000 (default)
- seeds = 3 (base seeds 2026, 2027, 2028)

Outputs:
- outputs/CMI_keypoints_multiseed.csv
- outputs/CMI_keypoints_multiseed_summary.csv
"""

import argparse
import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Avoid BLAS thread oversubscription when using multiprocessing.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

from src.core.cmi_calculation import calculate_CMI_bMPS

ROOT = Path(__file__).resolve().parents[2]
RAW_CSV = ROOT / "outputs" / "CMI_keypoints_multiseed.csv"
SUMMARY_CSV = ROOT / "outputs" / "CMI_keypoints_multiseed_summary.csv"

P_VALUES = [0.11, 0.15]
R_VALUES = [5, 6]


def stable_seed(base_seed, p, r):
    return int(base_seed + 10000 * p + 100 * r)


def load_cache():
    out = {}
    if not RAW_CSV.exists():
        return out
    with open(RAW_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (
                float(row["p"]),
                int(row["r"]),
                int(row["max_bond"]),
                int(row["num_samples"]),
                int(row["seed"]),
            )
            out[key] = row
    return out


def append_raw(row):
    exists = RAW_CSV.exists()
    fields = [
        "p",
        "r",
        "L",
        "max_bond",
        "num_samples",
        "seed",
        "cmi",
        "elapsed_sec",
    ]
    with open(RAW_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            w.writeheader()
        w.writerow(row)


def save_summary(rows):
    fields = [
        "p",
        "r",
        "max_bond",
        "num_samples",
        "count",
        "cmi_mean",
        "cmi_std",
        "cmi_sem",
    ]
    with open(SUMMARY_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def build_summary(cache, max_bond, num_samples):
    out = []
    for p in P_VALUES:
        for r in R_VALUES:
            vals = []
            for (pp, rr, chi, ns, _seed), row in cache.items():
                if pp == p and rr == r and chi == max_bond and ns == num_samples:
                    vals.append(float(row["cmi"]))
            arr = np.array(vals, dtype=float)
            if arr.size == 0:
                out.append(
                    {
                        "p": f"{p:.6f}",
                        "r": r,
                        "max_bond": max_bond,
                        "num_samples": num_samples,
                        "count": 0,
                        "cmi_mean": "nan",
                        "cmi_std": "nan",
                        "cmi_sem": "nan",
                    }
                )
                continue

            mean = float(np.mean(arr))
            std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
            sem = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
            out.append(
                {
                    "p": f"{p:.6f}",
                    "r": r,
                    "max_bond": max_bond,
                    "num_samples": num_samples,
                    "count": int(arr.size),
                    "cmi_mean": f"{mean:.16g}",
                    "cmi_std": f"{std:.16g}",
                    "cmi_sem": f"{sem:.16g}",
                }
            )
    return out


def run_one(task):
    p, r, max_bond, num_samples, seed = task
    L = 2 + 4 * r
    t0 = time.perf_counter()
    cmi = calculate_CMI_bMPS(
        L=L,
        p=p,
        r=r,
        num_samples=num_samples,
        max_bond=max_bond,
        seed=seed,
        verbose=False,
    )
    elapsed = time.perf_counter() - t0
    return {
        "p": p,
        "r": r,
        "L": L,
        "max_bond": max_bond,
        "num_samples": num_samples,
        "seed": seed,
        "cmi": cmi,
        "elapsed_sec": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Run keypoint multi-seed stats")
    parser.add_argument("--max-bond", type=int, default=32)
    parser.add_argument("--num-samples", type=int, default=2000)
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()

    cache = load_cache()
    base_seeds = [args.base_seed + i for i in range(args.num_seeds)]

    all_tasks = []
    for p in P_VALUES:
        for r in R_VALUES:
            for bs in base_seeds:
                seed = stable_seed(bs, p, r)
                all_tasks.append((p, r, args.max_bond, args.num_samples, seed))

    total = len(all_tasks)
    done = 0
    pending = []

    for p, r, max_bond, num_samples, seed in all_tasks:
        key = (p, r, max_bond, num_samples, seed)
        if key in cache:
            done += 1
            print(
                f"[cached {done}/{total}] p={p:.2f}, r={r}, seed={seed}, "
                f"cmi={float(cache[key]['cmi']):.6f}",
                flush=True,
            )
        else:
            pending.append((p, r, max_bond, num_samples, seed))

    print(
        f"workers={args.workers}, total={total}, cached={done}, pending={len(pending)}",
        flush=True,
    )

    if pending:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            fut_map = {ex.submit(run_one, task): task for task in pending}
            for fut in as_completed(fut_map):
                row = fut.result()
                out_row = {
                    "p": f"{row['p']:.6f}",
                    "r": row["r"],
                    "L": row["L"],
                    "max_bond": row["max_bond"],
                    "num_samples": row["num_samples"],
                    "seed": row["seed"],
                    "cmi": f"{row['cmi']:.16g}",
                    "elapsed_sec": f"{row['elapsed_sec']:.3f}",
                }
                append_raw(out_row)

                key = (
                    float(out_row["p"]),
                    int(out_row["r"]),
                    int(out_row["max_bond"]),
                    int(out_row["num_samples"]),
                    int(out_row["seed"]),
                )
                cache[key] = out_row
                done += 1
                print(
                    f"[done {done}/{total}] p={row['p']:.2f}, r={row['r']}, seed={row['seed']}, "
                    f"cmi={row['cmi']:.6f}, elapsed={row['elapsed_sec']:.1f}s",
                    flush=True,
                )

    summary_rows = build_summary(cache, args.max_bond, args.num_samples)
    save_summary(summary_rows)

    print("\nkeypoint multiseed summary (mean ± sem):", flush=True)
    for row in summary_rows:
        print(
            f"  p={float(row['p']):.2f}, r={int(row['r'])}: "
            f"{row['cmi_mean']} ± {row['cmi_sem']} (n={row['count']})",
            flush=True,
        )


if __name__ == "__main__":
    main()
