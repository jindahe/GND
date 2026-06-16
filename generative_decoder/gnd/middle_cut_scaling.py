import argparse
import csv
import json
import math
from pathlib import Path
from types import SimpleNamespace

from module import Errormodel, Loading_code, read_code

from .datasets import make_layout
from .evaluate_cut_mi import bootstrap_mi, estimate_plugin_mi
from .exact_mi import enumerate_distribution, exact_cut_mi, error_configs
from .partitions import build_cut
from .records import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_ints(text):
    values = []
    for item in text.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a middle-cut I(beta:gamma) scaling record and check the "
            "logical-sector entropy upper bound."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--result", action="append", default=[], help="Existing gnd cut-MI JSON record.")
    source.add_argument("--d-values", type=parse_ints, help="Comma-separated code distances/L values.")
    parser.add_argument("--n-values", type=parse_ints, default=None, help="Optional comma-separated n values.")
    parser.add_argument("--c-type", default="tor")
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e-model", default="dep", choices=["dep", "x", "z"])
    parser.add_argument("--er", type=float, default=None, help="Physical error rate for direct backends.")
    parser.add_argument("--backend", choices=["sample", "exact"], default="sample")
    parser.add_argument("--samples", type=int, default=10000, help="Samples per L for --backend sample.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-exact-errors", type=int, default=20_000_000)
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--bound-tolerance", type=float, default=1e-10)
    parser.add_argument("--output-json", default="net/gnd/scaling/middle_cut_scaling.json")
    parser.add_argument("--output-csv", default="net/gnd/scaling/middle_cut_scaling.csv")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_json(path):
    with resolve_path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_n(c_type, d):
    if c_type == "tor":
        return 2 * d * d
    if c_type == "sur":
        return d * d + (d - 1) * (d - 1)
    if c_type == "rsur":
        return d * d
    if c_type == "rep":
        return d
    raise ValueError(f"--n-values is required for c_type={c_type}")


def code_n_values(args):
    if args.n_values is not None:
        if len(args.n_values) != len(args.d_values):
            raise ValueError("--n-values must have the same length as --d-values")
        return args.n_values
    return [infer_n(args.c_type, d_value) for d_value in args.d_values]


def middle_result(payload):
    for item in payload.get("results", []):
        if item.get("cut", {}).get("name") == "middle":
            return item
    raise ValueError("Result payload does not contain a middle cut")


def beta_dim_from_result(result, payload):
    cut = result["cut"]
    if "len_A" in cut:
        return int(cut["len_A"])
    if cut.get("name") == "middle":
        return len(cut["a_indices"])
    layout = payload.get("layout", {})
    beta = layout.get("beta")
    if beta:
        return int(beta["stop"]) - int(beta["start"])
    code = payload.get("code", {})
    if "k" in code:
        return 2 * int(code["k"])
    raise ValueError("Could not infer beta dimension for upper-bound check")


def logical_entropy_bound(beta_dim):
    return beta_dim * math.log(2.0)


def row_from_payload(path, payload):
    result = middle_result(payload)
    code = payload.get("code", {})
    l_value = code.get("d")
    if l_value is None:
        raise ValueError(f"{path} does not record code.d")
    beta_dim = beta_dim_from_result(result, payload)
    bootstrap = result.get("bootstrap") or {}
    return {
        "L": int(l_value),
        "n": code.get("n"),
        "k": code.get("k"),
        "c_type": code.get("c_type"),
        "e_model": code.get("e_model"),
        "er": code.get("er"),
        "source_type": payload.get("source_type"),
        "backend": result.get("estimator"),
        "source_path": str(resolve_path(path)),
        "n_samples": result.get("n_samples"),
        "mi": float(result["mi"]),
        "H_A": result["entropy"]["H_A"],
        "H_B": result["entropy"]["H_B"],
        "H_AB": result["entropy"]["H_AB"],
        "bootstrap_std": bootstrap.get("std"),
        "beta_dim": beta_dim,
        "logical_entropy_bound_nats": logical_entropy_bound(beta_dim),
    }


def sample_middle_row(args, d_value, n_value, index):
    info = read_code(d=d_value, k=args.k, n=n_value, seed=args.seed, c_type=args.c_type)
    code = Loading_code(info)
    error_model = Errormodel(args.er, e_model=args.e_model)
    errors = error_model.generate_error(code.n, m=args.samples, seed=args.sample_seed + index)
    samples = error_configs(errors, code, args.k)
    layout = make_layout(code, args.k, target="beta_gamma")
    cut = build_cut(layout, "middle")
    result = estimate_plugin_mi(samples, cut)
    bootstrap = bootstrap_mi(samples, cut, args.bootstrap_samples, args.bootstrap_seed + index)
    if bootstrap is not None:
        result["bootstrap"] = bootstrap
    return row_from_direct_result(args, d_value, n_value, "true_error_model_samples", result)


def exact_middle_row(args, d_value, n_value):
    info = read_code(d=d_value, k=args.k, n=n_value, seed=args.seed, c_type=args.c_type)
    code = Loading_code(info)
    exact_args = SimpleNamespace(
        er=args.er,
        e_model=args.e_model,
        k=args.k,
        max_exact_errors=args.max_exact_errors,
        chunk_size=args.chunk_size,
    )
    layout, joint_distribution, total_errors = enumerate_distribution(code, exact_args)
    result = exact_cut_mi(joint_distribution, build_cut(layout, "middle"))
    row = row_from_direct_result(args, d_value, n_value, "exact_error_enumeration", result)
    row["total_errors_enumerated"] = total_errors
    row["joint_support_size"] = len(joint_distribution)
    return row


def row_from_direct_result(args, d_value, n_value, source_type, result):
    beta_dim = beta_dim_from_result(result, {"code": {"k": args.k}})
    bootstrap = result.get("bootstrap") or {}
    return {
        "L": int(d_value),
        "n": int(n_value),
        "k": int(args.k),
        "c_type": args.c_type,
        "e_model": args.e_model,
        "er": args.er,
        "source_type": source_type,
        "backend": result.get("estimator"),
        "source_path": None,
        "n_samples": result.get("n_samples"),
        "mi": float(result["mi"]),
        "H_A": result["entropy"]["H_A"],
        "H_B": result["entropy"]["H_B"],
        "H_AB": result["entropy"]["H_AB"],
        "bootstrap_std": bootstrap.get("std"),
        "beta_dim": beta_dim,
        "logical_entropy_bound_nats": logical_entropy_bound(beta_dim),
    }


def mean(values):
    return sum(values) / len(values)


def fit_constant(xs, ys):
    estimate = mean(ys)
    residuals = [y - estimate for y in ys]
    return {
        "model": "constant",
        "equation": "I(L) = c",
        "parameters": {"c": estimate},
        "rmse": rmse(residuals),
        "max_abs_residual": max(abs(item) for item in residuals),
    }


def fit_affine(xs, ys, transform_name, transform, expression):
    transformed = [transform(x) for x in xs]
    n_items = len(transformed)
    sum_x = sum(transformed)
    sum_y = sum(ys)
    sum_xx = sum(x * x for x in transformed)
    sum_xy = sum(x * y for x, y in zip(transformed, ys))
    denominator = n_items * sum_xx - sum_x * sum_x
    if abs(denominator) < 1e-15:
        return None
    slope = (n_items * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n_items
    predictions = [intercept + slope * x for x in transformed]
    residuals = [y - y_hat for y, y_hat in zip(ys, predictions)]
    return {
        "model": f"affine_{transform_name}",
        "equation": f"I(L) = a * {expression} + b",
        "parameters": {"a": slope, "b": intercept},
        "rmse": rmse(residuals),
        "max_abs_residual": max(abs(item) for item in residuals),
    }


def fit_power_law(xs, ys):
    if any(y <= 0 for y in ys):
        return None
    log_fit = fit_affine(xs, [math.log(y) for y in ys], "log", math.log, "log(L)")
    if log_fit is None:
        return None
    coefficient = math.exp(log_fit["parameters"]["b"])
    exponent = log_fit["parameters"]["a"]
    predictions = [coefficient * (x ** exponent) for x in xs]
    residuals = [y - y_hat for y, y_hat in zip(ys, predictions)]
    return {
        "model": "power_law",
        "equation": "I(L) = a * L^p",
        "parameters": {"a": coefficient, "p": exponent},
        "rmse": rmse(residuals),
        "max_abs_residual": max(abs(item) for item in residuals),
    }


def rmse(residuals):
    return math.sqrt(sum(item * item for item in residuals) / len(residuals))


def scaling_fits(rows):
    rows = sorted(rows, key=lambda item: item["L"])
    xs = [float(row["L"]) for row in rows]
    ys = [float(row["mi"]) for row in rows]
    fits = [fit_constant(xs, ys)]
    if len(rows) >= 2:
        for fit in (
            fit_affine(xs, ys, "L", lambda item: item, "L"),
            fit_affine(xs, ys, "log", math.log, "log(L)"),
            fit_power_law(xs, ys),
        ):
            if fit is not None:
                fits.append(fit)
    return fits


def upper_bound_summary(rows, tolerance):
    max_row = max(rows, key=lambda item: item["mi"])
    min_margin = min(row["logical_entropy_bound_nats"] - row["mi"] for row in rows)
    all_within = all(row["mi"] <= row["logical_entropy_bound_nats"] + tolerance for row in rows)
    beta_dims = sorted({row["beta_dim"] for row in rows})
    return {
        "bound_name": "logical_sector_entropy",
        "statement": "I(beta:gamma) <= H(beta) <= beta_dim * ln(2)",
        "beta_dims": beta_dims,
        "all_points_within_bound": all_within,
        "tolerance": tolerance,
        "observed_max_mi_nats": max_row["mi"],
        "observed_max_L": max_row["L"],
        "minimum_margin_nats": min_margin,
        "bounded_scaling_class": "O(1)" if all_within and len(beta_dims) == 1 else None,
    }


def build_rows(args):
    if args.result:
        return [row_from_payload(path, load_json(path)) for path in args.result]

    if args.er is None:
        raise ValueError("--er is required when using --d-values")
    if args.backend == "sample" and args.samples <= 0:
        raise ValueError("--samples must be positive")

    rows = []
    for index, (d_value, n_value) in enumerate(zip(args.d_values, code_n_values(args))):
        if args.backend == "sample":
            rows.append(sample_middle_row(args, d_value, n_value, index))
        else:
            rows.append(exact_middle_row(args, d_value, n_value))
    return rows


def write_csv(path, rows):
    columns = [
        "L",
        "n",
        "k",
        "c_type",
        "e_model",
        "er",
        "source_type",
        "backend",
        "n_samples",
        "mi",
        "H_A",
        "H_B",
        "H_AB",
        "bootstrap_std",
        "beta_dim",
        "logical_entropy_bound_nats",
        "within_logical_bound",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows([{key: row.get(key) for key in columns} for row in rows])


def main():
    args = parse_args()
    rows = sorted(build_rows(args), key=lambda item: item["L"])
    for row in rows:
        row["within_logical_bound"] = row["mi"] <= row["logical_entropy_bound_nats"] + args.bound_tolerance

    fits = scaling_fits(rows)
    bound = upper_bound_summary(rows, args.bound_tolerance)
    payload = {
        "record_type": "gnd_middle_cut_scaling",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "cut": "middle",
        "quantity": "I(beta:gamma)",
        "log_unit": "nats",
        "rows": rows,
        "fits": fits,
        "upper_bound": bound,
        "interpretation": (
            "The middle cut is bounded by the entropy of the logical-sector "
            "variables. For toric code with k=2, beta_dim=4 and the universal "
            "bound is 4 ln(2) nats, independent of L."
        ),
    }

    output_json = resolve_path(args.output_json)
    output_csv = resolve_path(args.output_csv)
    write_json_record(output_json, payload)
    write_csv(output_csv, rows)

    print("middle cut scaling rows:")
    for row in rows:
        print(
            f"L={row['L']} mi={row['mi']:.8f} "
            f"bound={row['logical_entropy_bound_nats']:.8f} "
            f"within_bound={row['within_logical_bound']}"
        )
    print(
        "upper bound: "
        f"{bound['statement']}; all_points_within_bound={bound['all_points_within_bound']}"
    )
    print(f"saved: {output_json}")
    print(f"saved: {output_csv}")


if __name__ == "__main__":
    main()
