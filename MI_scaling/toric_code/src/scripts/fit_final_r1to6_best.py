import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

FINAL_SUMMARY = OUTPUTS / "CMI_final_r1to6_best_summary_2026-04-16.csv"
RAW_R1TO5 = OUTPUTS / "CMI_stageP9_r1to5_chi64_raw.csv"
RAW_R6 = OUTPUTS / "CMI_stageP9_r6_chi96_raw.csv"
RAW_R6_P015 = OUTPUTS / "CMI_verify_p015_r6_chi256_N5000_raw.csv"

OUT_BEST_PNG = OUTPUTS / "CMI_final_r1to6_best_refit_2026-04-16.png"
OUT_COMPARE_PNG = OUTPUTS / "CMI_final_r1to6_fit_comparison_2026-04-16.png"
OUT_FIT_CSV = OUTPUTS / "CMI_final_r1to6_fit_comparison_2026-04-16.csv"
OUT_POINT_CSV = OUTPUTS / "CMI_final_r1to6_point_comparison_2026-04-16.csv"

P_VALUES = [0.05, 0.11, 0.15]
R_VALUES = [1, 2, 3, 4, 5, 6]


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def _sem(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(np.array(values, dtype=float), ddof=1) / math.sqrt(len(values)))


def load_best_summary():
    out = {}
    for row in _read_csv(FINAL_SUMMARY):
        key = (float(row["p"]), int(row["r"]))
        out[key] = {
            "p": float(row["p"]),
            "r": int(row["r"]),
            "mean": float(row["cmi_mean"]),
            "sem": float(row["cmi_sem"]),
            "total_samples": int(row["total_samples"]),
            "count_chunks": int(row["count_chunks"]),
            "max_bond": row["max_bond"],
            "source": row["source"],
        }
    return out


def _summarize_rows(rows):
    vals = [float(row["cmi"]) for row in rows]
    sample = rows[0]
    return {
        "p": float(sample["p"]),
        "r": int(sample["r"]),
        "mean": float(np.mean(vals)),
        "sem": _sem(vals),
        "total_samples": int(sum(int(row["chunk_samples"]) for row in rows)),
        "count_chunks": len(rows),
        "max_bond": sample["max_bond"],
        "source": str(sample["_source"]),
    }


def load_old_baseline():
    grouped = defaultdict(list)

    for path in [RAW_R1TO5, RAW_R6, RAW_R6_P015]:
        for row in _read_csv(path):
            row["_source"] = path.relative_to(ROOT)
            key = (float(row["p"]), int(row["r"]))
            chunk_id = int(row["chunk_id"])

            if path == RAW_R1TO5:
                if chunk_id < 20:
                    grouped[key].append(row)
            elif path == RAW_R6:
                if key == (0.15, 6):
                    continue
                if chunk_id < 20:
                    grouped[key].append(row)
            elif path == RAW_R6_P015:
                grouped[key].append(row)

    out = {}
    for p in P_VALUES:
        for r in R_VALUES:
            key = (p, r)
            rows = grouped[key]
            out[key] = _summarize_rows(rows)
    return out


def fit_eq12(points):
    fits = {}

    for p in [0.05, 0.15]:
        xs, ys = [], []
        for r in R_VALUES:
            y = points[(p, r)]["mean"]
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
            fits[(p, "exp")] = (float(slope), float(r2), len(xs), float(intercept))
        else:
            fits[(p, "exp")] = (float("nan"), float("nan"), len(xs), float("nan"))

    xs, ys = [], []
    for r in R_VALUES:
        y = points[(0.11, r)]["mean"]
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
        fits[(0.11, "power")] = (-float(slope), float(r2), len(xs), float(intercept))
    else:
        fits[(0.11, "power")] = (float("nan"), float("nan"), len(xs), float("nan"))

    return fits


def _fit_lines(points, fits, p):
    xs = [float(r) for r in R_VALUES if points[(p, r)]["mean"] > 0]
    if len(xs) < 2:
        return None, None
    x_fit = np.linspace(min(xs), max(xs), 200)
    if p in [0.05, 0.15]:
        slope, _, _, intercept = fits[(p, "exp")]
        if not np.isfinite(slope):
            return None, None
        y_fit = np.exp(intercept + slope * x_fit)
        return x_fit, y_fit
    alpha, _, _, log_pref = fits[(0.11, "power")]
    if not np.isfinite(alpha):
        return None, None
    y_fit = np.exp(log_pref) * np.power(x_fit, -alpha)
    return x_fit, y_fit


def _fmt(value):
    return "nan" if not np.isfinite(value) else f"{value:.4f}"


def save_fit_csv(old_fits, new_fits):
    fields = ["dataset", "p", "model", "parameter", "r2", "points"]
    rows = []
    for tag, fits in [("old_baseline", old_fits), ("today_best", new_fits)]:
        for p in [0.05, 0.15]:
            param, r2, n, _ = fits[(p, "exp")]
            rows.append({
                "dataset": tag,
                "p": f"{p:.2f}",
                "model": "exp",
                "parameter": f"{param:.16g}",
                "r2": f"{r2:.16g}",
                "points": n,
            })
        param, r2, n, _ = fits[(0.11, "power")]
        rows.append({
            "dataset": tag,
            "p": "0.11",
            "model": "power",
            "parameter": f"{param:.16g}",
            "r2": f"{r2:.16g}",
            "points": n,
        })

    with open(OUT_FIT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save_point_csv(old_points, new_points):
    fields = [
        "p", "r",
        "old_mean", "old_sem", "old_total_samples",
        "new_mean", "new_sem", "new_total_samples",
        "delta_mean", "ratio_new_to_old",
    ]
    rows = []
    for p in P_VALUES:
        for r in R_VALUES:
            old = old_points[(p, r)]
            new = new_points[(p, r)]
            ratio = float("nan")
            if old["mean"] != 0:
                ratio = new["mean"] / old["mean"]
            rows.append({
                "p": f"{p:.6f}",
                "r": r,
                "old_mean": f"{old['mean']:.16g}",
                "old_sem": f"{old['sem']:.16g}",
                "old_total_samples": old["total_samples"],
                "new_mean": f"{new['mean']:.16g}",
                "new_sem": f"{new['sem']:.16g}",
                "new_total_samples": new["total_samples"],
                "delta_mean": f"{(new['mean'] - old['mean']):.16g}",
                "ratio_new_to_old": f"{ratio:.16g}",
            })

    with open(OUT_POINT_CSV, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def make_best_plot(points, fits):
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    for p in P_VALUES:
        xs, ys, es = [], [], []
        for r in R_VALUES:
            rec = points[(p, r)]
            xs.append(r)
            ys.append(rec["mean"])
            es.append(min(rec["sem"], rec["mean"] * 0.95))
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
        x_fit, y_fit = _fit_lines(points, fits, p)
        if x_fit is not None:
            ax.plot(x_fit, y_fit, color=colors[p], lw=1.8, linestyle="--", label=f"p={p:.2f} fit")

    ax.set_yscale("log")
    ax.set_xlabel("r")
    ax.set_ylabel("CMI = I(A:C|B) [nats]")
    ax.set_title("Eq.(12) refit from final r=1..6 best dataset")
    ax.set_xticks(R_VALUES)
    ax.grid(True, alpha=0.3)
    ax.legend()

    text = "\n".join([
        "Today best fits:",
        f"p=0.05 exp: b={_fmt(fits[(0.05, 'exp')][0])}, R^2={_fmt(fits[(0.05, 'exp')][1])}",
        f"p=0.15 exp: b={_fmt(fits[(0.15, 'exp')][0])}, R^2={_fmt(fits[(0.15, 'exp')][1])}",
        f"p=0.11 power: alpha={_fmt(fits[(0.11, 'power')][0])}, R^2={_fmt(fits[(0.11, 'power')][1])}",
    ])
    ax.text(
        0.02,
        0.98,
        text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )

    plt.tight_layout()
    plt.savefig(OUT_BEST_PNG, dpi=180)
    plt.close(fig)


def make_compare_plot(old_points, new_points, old_fits, new_fits):
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2), sharey=True)
    colors = {0.05: "#1f77b4", 0.11: "#d62728", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "s", 0.15: "D"}

    for ax, title, points, fits in [
        (axes[0], "Old baseline (pre-topup)", old_points, old_fits),
        (axes[1], "Today best (post-topup)", new_points, new_fits),
    ]:
        for p in P_VALUES:
            xs, ys, es = [], [], []
            for r in R_VALUES:
                rec = points[(p, r)]
                xs.append(r)
                ys.append(rec["mean"])
                es.append(min(rec["sem"], rec["mean"] * 0.95))
            ax.errorbar(
                xs,
                ys,
                yerr=es,
                fmt=markers[p],
                linestyle="none",
                color=colors[p],
                markersize=5,
                capsize=2.5,
                elinewidth=0.9,
                label=f"p={p:.2f}",
            )
            x_fit, y_fit = _fit_lines(points, fits, p)
            if x_fit is not None:
                ax.plot(x_fit, y_fit, color=colors[p], lw=1.6, linestyle="--")

        ax.set_yscale("log")
        ax.set_xticks(R_VALUES)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("r")

    axes[0].set_ylabel("CMI = I(A:C|B) [nats]")
    axes[0].legend(loc="lower left", fontsize=9)

    compare_text = "\n".join([
        "Fit parameter shift:",
        f"p=0.05: Δb={new_fits[(0.05, 'exp')][0] - old_fits[(0.05, 'exp')][0]:+.4f}",
        f"p=0.15: Δb={new_fits[(0.15, 'exp')][0] - old_fits[(0.15, 'exp')][0]:+.4f}",
        f"p=0.11: Δalpha={new_fits[(0.11, 'power')][0] - old_fits[(0.11, 'power')][0]:+.4f}",
    ])
    fig.text(
        0.5,
        0.01,
        compare_text,
        ha="center",
        va="bottom",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "gray"},
    )

    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig(OUT_COMPARE_PNG, dpi=180)
    plt.close(fig)


def main():
    best_points = load_best_summary()
    old_points = load_old_baseline()
    best_fits = fit_eq12(best_points)
    old_fits = fit_eq12(old_points)

    save_fit_csv(old_fits, best_fits)
    save_point_csv(old_points, best_points)
    make_best_plot(best_points, best_fits)
    make_compare_plot(old_points, best_points, old_fits, best_fits)

    print(f"Saved: {OUT_BEST_PNG.relative_to(ROOT)}")
    print(f"Saved: {OUT_COMPARE_PNG.relative_to(ROOT)}")
    print(f"Saved: {OUT_FIT_CSV.relative_to(ROOT)}")
    print(f"Saved: {OUT_POINT_CSV.relative_to(ROOT)}")
    print()
    print("Old baseline fits:")
    print(f"  p=0.05 exponential: b={old_fits[(0.05, 'exp')][0]:.4f}, R^2={old_fits[(0.05, 'exp')][1]:.3f}")
    print(f"  p=0.15 exponential: b={old_fits[(0.15, 'exp')][0]:.4f}, R^2={old_fits[(0.15, 'exp')][1]:.3f}")
    print(f"  p=0.11 power-law: alpha={old_fits[(0.11, 'power')][0]:.4f}, R^2={old_fits[(0.11, 'power')][1]:.3f}")
    print()
    print("Today best fits:")
    print(f"  p=0.05 exponential: b={best_fits[(0.05, 'exp')][0]:.4f}, R^2={best_fits[(0.05, 'exp')][1]:.3f}")
    print(f"  p=0.15 exponential: b={best_fits[(0.15, 'exp')][0]:.4f}, R^2={best_fits[(0.15, 'exp')][1]:.3f}")
    print(f"  p=0.11 power-law: alpha={best_fits[(0.11, 'power')][0]:.4f}, R^2={best_fits[(0.11, 'power')][1]:.3f}")


if __name__ == "__main__":
    main()
