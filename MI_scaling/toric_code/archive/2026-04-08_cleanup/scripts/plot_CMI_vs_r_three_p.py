"""
Compute and plot CMI vs r (r=1..6) for p in {0.05, 0.11, 0.15}.

Strategy based on PROBLEM.md:
  - r <= 4: exact TN (calculate_CMI)
  - r >= 5: bMPS (calculate_CMI_bMPS)
      * near critical p=0.11 use larger chi
      * away from critical point use chi=16 for speed

Outputs:
  - CMI_vs_r_three_p.csv
  - CMI_vs_r_p005_011_015.png
"""

import csv
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from CMI_calculation import calculate_CMI, calculate_CMI_bMPS


R_VALUES = [1, 2, 3, 4, 5, 6]
P_VALUES = [0.05, 0.11, 0.15]

OUT_CSV = "CMI_vs_r_three_p.csv"
OUT_PNG = "CMI_vs_r_p005_011_015.png"


def point_config(r, p):
    """Return configuration dict for one (r, p) point."""
    L = 2 + 4 * r

    if r <= 4:
        num_samples = 200 if r <= 3 else 40
        return {
            "method": "exact",
            "L": L,
            "num_samples": num_samples,
            "max_bond": "",
        }

    # r = 5,6 use bMPS
    if abs(p - 0.11) < 1e-12:
        return {
            "method": "bmps",
            "L": L,
            "num_samples": 30,
            "max_bond": 32,
        }

    return {
        "method": "bmps",
        "L": L,
        "num_samples": 120,
        "max_bond": 16,
    }


def point_seed(r, p):
    # Stable deterministic seed per point.
    return int(10000 * p) + 100 * r + 17


def load_existing():
    data = {}
    if not os.path.exists(OUT_CSV):
        return data

    with open(OUT_CSV, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (int(row["r"]), float(row["p"]))
            data[key] = row
    return data


def append_row(row):
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="") as f:
        fieldnames = [
            "r", "p", "cmi",
            "method", "L", "num_samples", "max_bond", "seed", "elapsed_sec",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def compute_all():
    cached = load_existing()
    results = {}

    total = len(R_VALUES) * len(P_VALUES)
    done = 0

    for p in P_VALUES:
        for r in R_VALUES:
            key = (r, p)
            cfg = point_config(r, p)
            seed = point_seed(r, p)

            if key in cached:
                row = cached[key]
                cmi = float(row["cmi"])
                results[key] = cmi
                done += 1
                print(
                    f"[cached {done}/{total}] p={p:.2f}, r={r}, "
                    f"CMI={cmi:.6f} ({row['method']}, L={row['L']}, "
                    f"N={row['num_samples']}, chi={row['max_bond']})"
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
            done += 1
            results[key] = cmi

            row = {
                "r": r,
                "p": f"{p:.6f}",
                "cmi": f"{cmi:.16g}" if cmi is not None else "None",
                "method": cfg["method"],
                "L": cfg["L"],
                "num_samples": cfg["num_samples"],
                "max_bond": cfg["max_bond"],
                "seed": seed,
                "elapsed_sec": f"{elapsed:.3f}",
            }
            append_row(row)
            print(
                f"[new {done}/{total}] p={p:.2f}, r={r}, CMI={cmi:.6f}, "
                f"{cfg['method']}, L={cfg['L']}, N={cfg['num_samples']}, "
                f"chi={cfg['max_bond']}, {elapsed:.1f}s"
            )

    return results


def fit_eq12_trends(results):
    """
    Simple fits for Eq.(12):
      p != p_c: log(CMI) ~ a + b r  (exponential in r)
      p = p_c : log(CMI) ~ c - alpha log(r) (power-law in r)
    """
    summary = []

    # Off-critical exponential checks
    for p in [0.05, 0.15]:
        pts = [(r, results[(r, p)]) for r in R_VALUES if (r, p) in results]
        pts = [(r, y) for r, y in pts if y is not None and y > 0]
        if len(pts) >= 3:
            x = np.array([r for r, _ in pts], dtype=float)
            ly = np.log(np.array([y for _, y in pts], dtype=float))
            slope, intercept = np.polyfit(x, ly, 1)
            pred = slope * x + intercept
            ss_res = float(np.sum((ly - pred) ** 2))
            ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
            summary.append((p, "exp", slope, r2))
        else:
            summary.append((p, "exp", float("nan"), float("nan")))

    # Critical power-law check
    p = 0.11
    pts = [(r, results[(r, p)]) for r in R_VALUES if (r, p) in results]
    pts = [(r, y) for r, y in pts if y is not None and y > 0]
    if len(pts) >= 3:
        lx = np.log(np.array([r for r, _ in pts], dtype=float))
        ly = np.log(np.array([y for _, y in pts], dtype=float))
        slope, intercept = np.polyfit(lx, ly, 1)
        pred = slope * lx + intercept
        ss_res = float(np.sum((ly - pred) ** 2))
        ss_tot = float(np.sum((ly - np.mean(ly)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        alpha_est = -slope
        summary.append((p, "power", alpha_est, r2))
    else:
        summary.append((p, "power", float("nan"), float("nan")))

    return summary


def make_plot(results):
    fig, ax = plt.subplots(figsize=(7.5, 5.3))
    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    for p in P_VALUES:
        xs = []
        ys = []
        for r in R_VALUES:
            y = results.get((r, p))
            if y is None:
                continue
            xs.append(r)
            ys.append(y)

        ax.plot(
            xs,
            ys,
            color=colors[p],
            marker=markers[p],
            lw=1.8,
            ms=6,
            label=f"p={p:.2f}",
        )

    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("CMI = I(A:C|B) [nats]", fontsize=12)
    ax.set_title("CMI vs r (geom1, p=0.05/0.11/0.15)", fontsize=12)
    ax.set_xticks(R_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=180)
    print(f"Saved plot: {OUT_PNG}")


def main():
    results = compute_all()
    make_plot(results)
    summary = fit_eq12_trends(results)
    print("\nEq.(12) quick-fit summary:")
    for p, model, param, r2 in summary:
        if model == "exp":
            print(f"  p={p:.2f}: log(CMI)=a+br, b={param:.4f}, R^2={r2:.3f}")
        else:
            print(f"  p={p:.2f}: CMI~r^(-alpha), alpha={param:.4f}, R^2={r2:.3f}")


if __name__ == "__main__":
    main()
