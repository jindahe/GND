import csv
import math
import argparse
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

DEFAULT_SUMMARY = OUTPUTS / "CMI_final_r1to6_best_summary_2026-04-16.csv"
DEFAULT_OUT_FITS = OUTPUTS / "CMI_p10_refit_p011_p015_window_comparison.csv"

P_VALUES = [0.11, 0.15]
WINDOWS = [
    ("r1to6", [1, 2, 3, 4, 5, 6]),
    ("r2to6", [2, 3, 4, 5, 6]),
    ("r3to6", [3, 4, 5, 6]),
    ("r1to5", [1, 2, 3, 4, 5]),
]


def read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_points(summary_path):
    points = {}
    for row in read_csv(summary_path):
        p = float(row["p"])
        r = int(row["r"])
        if p not in P_VALUES:
            continue
        points[(p, r)] = {
            "mean": float(row["cmi_mean"]),
            "sem": float(row["cmi_sem"]),
            "total_samples": int(row["total_samples"]),
            "max_bond": row["max_bond"],
            "source": row["source"],
        }
    return points


def fit_line(xs, ys, sigmas=None):
    x = np.array(xs, dtype=float)
    y = np.array(ys, dtype=float)
    weights = None if sigmas is None else 1.0 / np.array(sigmas, dtype=float)
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(intercept), r2


def fit_window(points, p, window_name, r_values, weighted):
    xs = []
    ys = []
    sigmas = []
    kept = []
    for r in r_values:
        rec = points[(p, r)]
        mean = rec["mean"]
        sem = rec["sem"]
        if not (np.isfinite(mean) and mean > 0):
            continue
        if p == 0.11:
            xs.append(math.log(float(r)))
            model = "power"
        else:
            xs.append(float(r))
            model = "exp"
        ys.append(math.log(mean))
        sigmas.append(max(sem / mean, 1e-12))
        kept.append(r)

    if len(xs) < 3:
        return None

    slope, intercept, r2 = fit_line(xs, ys, sigmas=sigmas if weighted else None)
    if p == 0.11:
        parameter_name = "alpha"
        parameter = -slope
    else:
        parameter_name = "b"
        parameter = slope

    return {
        "p": f"{p:.6f}",
        "model": model,
        "window": window_name,
        "fit_mode": "weighted" if weighted else "unweighted",
        "kept_r": ",".join(str(r) for r in kept),
        "points": len(kept),
        "parameter_name": parameter_name,
        "parameter": f"{parameter:.16g}",
        "intercept": f"{intercept:.16g}",
        "r2": f"{r2:.16g}",
    }


def main():
    parser = argparse.ArgumentParser(description="Refit p=0.11 and p=0.15 over multiple windows")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FITS)
    args = parser.parse_args()

    points = load_points(args.summary)
    rows = []
    for p in P_VALUES:
        for window_name, r_values in WINDOWS:
            for weighted in [False, True]:
                row = fit_window(points, p, window_name, r_values, weighted)
                if row is not None:
                    rows.append(row)

    fields = [
        "p",
        "model",
        "window",
        "fit_mode",
        "kept_r",
        "points",
        "parameter_name",
        "parameter",
        "intercept",
        "r2",
    ]
    with open(args.out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {args.out}")
    for row in rows:
        print(
            f"p={float(row['p']):.2f} {row['window']} {row['fit_mode']}: "
            f"{row['parameter_name']}={float(row['parameter']):.4f}, "
            f"R^2={float(row['r2']):.4f}, r={row['kept_r']}"
        )


if __name__ == "__main__":
    main()
