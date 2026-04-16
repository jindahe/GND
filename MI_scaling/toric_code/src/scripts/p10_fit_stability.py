import csv
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = ROOT / "outputs"

FINAL_SUMMARY = OUTPUTS / "CMI_final_r1to6_best_summary_2026-04-16.csv"
OUT_DETAIL = OUTPUTS / "CMI_final_r1to6_fit_stability_2026-04-16.csv"
OUT_SUMMARY = OUTPUTS / "CMI_final_r1to6_fit_stability_summary_2026-04-16.csv"

P_VALUES = [0.05, 0.11, 0.15]
R_VALUES = [1, 2, 3, 4, 5, 6]


def _read_csv(path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_points():
    points = {}
    for row in _read_csv(FINAL_SUMMARY):
        key = (float(row["p"]), int(row["r"]))
        points[key] = {
            "p": float(row["p"]),
            "r": int(row["r"]),
            "mean": float(row["cmi_mean"]),
            "sem": float(row["cmi_sem"]),
            "total_samples": int(row["total_samples"]),
            "max_bond": row["max_bond"],
            "status": row["status"],
            "source": row["source"],
        }
    return points


def _fit_log_model(x_values, y_values, sigma_values=None):
    x_arr = np.array(x_values, dtype=float)
    y_arr = np.array(y_values, dtype=float)

    if sigma_values is None:
        coeffs = np.polyfit(x_arr, y_arr, 1)
    else:
        sigma_arr = np.array(sigma_values, dtype=float)
        coeffs = np.polyfit(x_arr, y_arr, 1, w=1.0 / sigma_arr)

    slope = float(coeffs[0])
    intercept = float(coeffs[1])
    pred = slope * x_arr + intercept

    ss_res = float(np.sum((y_arr - pred) ** 2))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    if sigma_values is None:
        weighted_r2 = float("nan")
    else:
        w = 1.0 / np.square(np.array(sigma_values, dtype=float))
        y_bar = float(np.sum(w * y_arr) / np.sum(w))
        ss_res_w = float(np.sum(w * np.square(y_arr - pred)))
        ss_tot_w = float(np.sum(w * np.square(y_arr - y_bar)))
        weighted_r2 = 1.0 - ss_res_w / ss_tot_w if ss_tot_w > 0 else float("nan")

    return slope, intercept, r2, weighted_r2


def fit_dataset(points, p_value, weighted=False, leave_out_r=None):
    rows = []
    for r in R_VALUES:
        if leave_out_r is not None and r == leave_out_r:
            continue
        rec = points[(p_value, r)]
        if not (np.isfinite(rec["mean"]) and rec["mean"] > 0):
            continue
        if p_value == 0.11:
            x = math.log(float(r))
            model = "power"
        else:
            x = float(r)
            model = "exp"
        y = math.log(rec["mean"])
        sigma_log = rec["sem"] / rec["mean"] if rec["sem"] > 0 else float("nan")
        rows.append(
            {
                "r": r,
                "x": x,
                "y": y,
                "sigma_log": sigma_log,
                "model": model,
            }
        )

    if len(rows) < 3:
        return None

    x_values = [row["x"] for row in rows]
    y_values = [row["y"] for row in rows]
    sigma_values = None
    fit_mode = "unweighted"
    if weighted:
        sigma_values = [max(row["sigma_log"], 1e-12) for row in rows]
        fit_mode = "weighted"

    slope, intercept, r2, weighted_r2 = _fit_log_model(
        x_values=x_values,
        y_values=y_values,
        sigma_values=sigma_values,
    )

    if p_value == 0.11:
        parameter = -slope
        parameter_name = "alpha"
    else:
        parameter = slope
        parameter_name = "b"

    kept_r = ",".join(str(row["r"]) for row in rows)
    return {
        "p": p_value,
        "model": rows[0]["model"],
        "fit_mode": fit_mode,
        "leave_out_r": "" if leave_out_r is None else leave_out_r,
        "kept_r": kept_r,
        "points": len(rows),
        "parameter_name": parameter_name,
        "parameter": parameter,
        "intercept": intercept,
        "r2": r2,
        "weighted_r2": weighted_r2,
    }


def build_detail_rows(points):
    rows = []
    for p_value in P_VALUES:
        for weighted in [False, True]:
            rows.append(fit_dataset(points, p_value, weighted=weighted, leave_out_r=None))
            for leave_out_r in R_VALUES:
                rows.append(fit_dataset(points, p_value, weighted=weighted, leave_out_r=leave_out_r))
    return [row for row in rows if row is not None]


def _finite_values(rows, key):
    values = [float(row[key]) for row in rows if np.isfinite(float(row[key]))]
    return values


def build_summary_rows(detail_rows):
    summary_rows = []
    for p_value in P_VALUES:
        for fit_mode in ["unweighted", "weighted"]:
            group = [row for row in detail_rows if row["p"] == p_value and row["fit_mode"] == fit_mode]
            full = [row for row in group if row["leave_out_r"] == ""]
            loo = [row for row in group if row["leave_out_r"] != ""]
            if not full:
                continue

            parameter_values = _finite_values(loo, "parameter")
            r2_values = _finite_values(loo, "r2")
            wr2_values = _finite_values(loo, "weighted_r2")
            summary_rows.append(
                {
                    "p": f"{p_value:.6f}",
                    "model": full[0]["model"],
                    "fit_mode": fit_mode,
                    "parameter_name": full[0]["parameter_name"],
                    "full_parameter": f"{float(full[0]['parameter']):.16g}",
                    "full_r2": f"{float(full[0]['r2']):.16g}",
                    "full_weighted_r2": f"{float(full[0]['weighted_r2']):.16g}",
                    "loo_parameter_min": f"{min(parameter_values):.16g}",
                    "loo_parameter_max": f"{max(parameter_values):.16g}",
                    "loo_r2_min": f"{min(r2_values):.16g}",
                    "loo_r2_max": f"{max(r2_values):.16g}",
                    "loo_weighted_r2_min": f"{min(wr2_values):.16g}" if wr2_values else "nan",
                    "loo_weighted_r2_max": f"{max(wr2_values):.16g}" if wr2_values else "nan",
                    "loo_points": len(loo),
                }
            )
    return summary_rows


def save_csv(path, fieldnames, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows):
    for row in summary_rows:
        p_value = float(row["p"])
        fit_mode = row["fit_mode"]
        parameter_name = row["parameter_name"]
        print(
            f"p={p_value:.2f}, {fit_mode}: "
            f"{parameter_name}={float(row['full_parameter']):.4f}, "
            f"R^2={float(row['full_r2']):.4f}, "
            f"LOO {parameter_name} range=[{float(row['loo_parameter_min']):.4f}, {float(row['loo_parameter_max']):.4f}], "
            f"LOO R^2 range=[{float(row['loo_r2_min']):.4f}, {float(row['loo_r2_max']):.4f}]"
        )


def main():
    points = load_points()
    detail_rows = build_detail_rows(points)
    summary_rows = build_summary_rows(detail_rows)

    save_csv(
        OUT_DETAIL,
        [
            "p",
            "model",
            "fit_mode",
            "leave_out_r",
            "kept_r",
            "points",
            "parameter_name",
            "parameter",
            "intercept",
            "r2",
            "weighted_r2",
        ],
        detail_rows,
    )
    save_csv(
        OUT_SUMMARY,
        [
            "p",
            "model",
            "fit_mode",
            "parameter_name",
            "full_parameter",
            "full_r2",
            "full_weighted_r2",
            "loo_parameter_min",
            "loo_parameter_max",
            "loo_r2_min",
            "loo_r2_max",
            "loo_weighted_r2_min",
            "loo_weighted_r2_max",
            "loo_points",
        ],
        summary_rows,
    )
    print(f"Saved detail: {OUT_DETAIL}")
    print(f"Saved summary: {OUT_SUMMARY}")
    print_summary(summary_rows)


if __name__ == "__main__":
    main()
