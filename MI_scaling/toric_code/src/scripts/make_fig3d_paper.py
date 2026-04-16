import csv
import math
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

DEFAULT_IN_TABLE = OUTPUTS / "CMI_paper_repro_main_table_v1.csv"
DEFAULT_OUT_FIT_CSV = OUTPUTS / "FIG3d_repro_fit_summary_v1.csv"
DEFAULT_OUT_PNG = OUTPUTS / "FIG3d_repro_paper_v1.png"
DEFAULT_OUT_PDF = OUTPUTS / "FIG3d_repro_paper_v1.pdf"

P_VALUES = [0.05, 0.11, 0.15]
FIT_CONFIG = {
    0.05: {"model": "exp", "window": [1, 2, 3, 4, 5, 6], "label": "exp, r=1..6", "sigma_mode": "mc"},
    0.11: {"model": "power", "window": [2, 3, 4, 5, 6], "label": "power, r=2..6", "sigma_mode": "eff"},
    0.15: {"model": "exp", "window": [1, 2, 3, 4, 5, 6], "label": "exp, r=1..6", "sigma_mode": "mc"},
}


def read_points(in_table):
    points = {}
    with open(in_table, newline="") as handle:
        for row in csv.DictReader(handle):
            p = float(row["p"])
            r = int(row["r"])
            points[(p, r)] = {
                "mean": float(row["cmi_mean"]),
                "sem": float(row["cmi_sem"]),
                "sigma_eff": float(row["sigma_eff"]),
                "max_bond": int(row["max_bond"]),
                "L": int(row["L"]),
                "fit_note": row["fit_note"],
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
    return float(slope), float(intercept), float(r2)


def fit_curve(points, p):
    cfg = FIT_CONFIG[p]
    xs = []
    ys = []
    sigmas = []
    kept_r = []
    for r in cfg["window"]:
        rec = points[(p, r)]
        mean = rec["mean"]
        sigma = rec["sem"] if cfg["sigma_mode"] == "mc" else rec["sigma_eff"]
        if mean <= 0:
            continue
        if cfg["model"] == "power":
            xs.append(math.log(float(r)))
        else:
            xs.append(float(r))
        ys.append(math.log(mean))
        sigmas.append(max(sigma / mean, 1e-12))
        kept_r.append(r)

    slope, intercept, r2 = fit_line(xs, ys, sigmas=sigmas)
    if cfg["model"] == "power":
        param_name = "alpha"
        parameter = -slope
    else:
        param_name = "b"
        parameter = slope

    return {
        "p": p,
        "model": cfg["model"],
        "window": ",".join(str(r) for r in kept_r),
        "parameter_name": param_name,
        "parameter": parameter,
        "intercept": intercept,
        "r2": r2,
        "label": cfg["label"],
    }


def save_fit_summary(fits, out_fit_csv):
    rows = []
    for fit in fits.values():
        rows.append(
            {
                "p": f"{fit['p']:.6f}",
                "model": fit["model"],
                "window": fit["window"],
                "parameter_name": fit["parameter_name"],
                "parameter": f"{fit['parameter']:.16g}",
                "intercept": f"{fit['intercept']:.16g}",
                "r2": f"{fit['r2']:.16g}",
                "label": fit["label"],
            }
        )

    with open(out_fit_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(points, fits, out_png, out_pdf):
    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    colors = {0.05: "#1f77b4", 0.11: "#ff7f0e", 0.15: "#2ca02c"}
    markers = {0.05: "o", 0.11: "o", 0.15: "o"}

    for p in P_VALUES:
        xs = []
        ys = []
        es = []
        for r in [1, 2, 3, 4, 5, 6]:
            rec = points[(p, r)]
            if rec["mean"] <= 0:
                continue
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
            markersize=6.5,
            capsize=2.5,
            elinewidth=0.9,
            label=fr"$p={p:.2f}$",
        )

        fit = fits[p]
        x_fit = np.linspace(min(FIT_CONFIG[p]["window"]), max(FIT_CONFIG[p]["window"]), 200)
        if fit["model"] == "power":
            y_fit = np.exp(fit["intercept"]) * np.power(x_fit, -fit["parameter"])
        else:
            y_fit = np.exp(fit["intercept"] + fit["parameter"] * x_fit)
        ax.plot(x_fit, y_fit, color="#8f8f8f", lw=1.9, linestyle="--")

    ax.set_yscale("log")
    ax.set_xlabel(r"$r$", fontsize=12)
    ax.set_ylabel(r"$I(A:C|B)$", fontsize=12)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.grid(False)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)

    text = "\n".join(
        [
            fr"$p=0.05$: $b={fits[0.05]['parameter']:.3f}$, $R^2={fits[0.05]['r2']:.3f}$",
            fr"$p=0.11$: $\alpha={fits[0.11]['parameter']:.3f}$, $R^2={fits[0.11]['r2']:.3f}$",
            fr"$p=0.15$: $b={fits[0.15]['parameter']:.3f}$, $R^2={fits[0.15]['r2']:.3f}$",
        ]
    )
    ax.text(
        0.98,
        0.98,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.92, "edgecolor": "#aaaaaa"},
    )

    plt.tight_layout()
    plt.savefig(out_png, dpi=240)
    plt.savefig(out_pdf)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(description="Make Fig.3(d) paper-style reproduction plot.")
    parser.add_argument("--in-table", type=Path, default=DEFAULT_IN_TABLE)
    parser.add_argument("--out-fit-csv", type=Path, default=DEFAULT_OUT_FIT_CSV)
    parser.add_argument("--out-png", type=Path, default=DEFAULT_OUT_PNG)
    parser.add_argument("--out-pdf", type=Path, default=DEFAULT_OUT_PDF)
    return parser.parse_args()


def main():
    args = parse_args()
    points = read_points(args.in_table)
    fits = {p: fit_curve(points, p) for p in P_VALUES}
    save_fit_summary(fits, args.out_fit_csv)
    make_plot(points, fits, args.out_png, args.out_pdf)
    print(f"Saved fit summary: {args.out_fit_csv}")
    print(f"Saved figure: {args.out_png}")
    print(f"Saved figure: {args.out_pdf}")


if __name__ == "__main__":
    main()
