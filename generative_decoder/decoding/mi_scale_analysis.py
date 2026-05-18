import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

from record_utils import utc_timestamp


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-path",
        action="append",
        default=[],
        help="Path to one mi_bipartite JSON result. Repeatable.",
    )
    parser.add_argument(
        "--result-glob",
        type=str,
        default="",
        help="Optional glob for mi_bipartite JSON results, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="net/mi_scaling/mi_vs_L.json",
        help="Summary JSON output path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        default="net/mi_scaling/mi_vs_L.csv",
        help="Summary CSV output path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--output-plot",
        type=str,
        default="net/mi_scaling/mi_vs_L.png",
        help="Plot output path, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Bipartite MI vs L",
        help="Plot title.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def collect_result_paths(args):
    paths = []
    for item in args.result_path:
        paths.append(resolve_path(item))

    if args.result_glob:
        pattern = resolve_path(args.result_glob)
        paths.extend(sorted(pattern.parent.glob(pattern.name)))

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_paths.append(path)

    if not unique_paths:
        raise ValueError("No result paths were provided. Use --result-path or --result-glob.")
    return unique_paths


def parse_code_meta(result):
    code = result.get("code", {})
    if code:
        return code

    checkpoint = result.get("checkpoint_ab", "")
    match = re.search(r"_n(?P<n>\d+)_d(?P<d>\d+)_k(?P<k>\d+)_seed(?P<seed>\d+)_er(?P<er>[0-9.]+)_(?P<e_model>[A-Za-z0-9]+)_", checkpoint)
    if not match:
        raise ValueError(f"Could not infer code metadata from result: {checkpoint}")

    return {
        "c_type": "tor",
        "n": int(match.group("n")),
        "d": int(match.group("d")),
        "k": int(match.group("k")),
        "seed": int(match.group("seed")),
        "e_model": match.group("e_model"),
        "er": float(match.group("er")),
    }


def load_point(path):
    with path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    code = parse_code_meta(result)
    partition = result["partition"]
    bootstrap = result.get("bootstrap", {})

    len_a = partition["len_A"]
    len_b = partition["len_B"]

    return {
        "path": str(path),
        "L": code["d"],
        "n": code["n"],
        "k": code["k"],
        "seed": code["seed"],
        "er": code["er"],
        "e_model": code["e_model"],
        "n_type": result["n_type"],
        "axis": partition["axis"],
        "cut": partition["cut"],
        "len_A": len_a,
        "len_B": len_b,
        "balanced_cut": len_a == len_b,
        "H_AB": result["entropy"]["H_AB"],
        "H_A": result["entropy"]["H_A"],
        "H_B": result["entropy"]["H_B"],
        "MI": result["mi"],
        "bootstrap_std": bootstrap.get("std"),
        "bootstrap_ci95_low": bootstrap.get("ci95_low"),
        "bootstrap_ci95_high": bootstrap.get("ci95_high"),
    }


def save_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "L",
        "n",
        "k",
        "seed",
        "er",
        "e_model",
        "n_type",
        "axis",
        "cut",
        "len_A",
        "len_B",
        "balanced_cut",
        "H_AB",
        "H_A",
        "H_B",
        "MI",
        "bootstrap_std",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "record_type": "mi_scale_summary",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "points": rows,
        "all_balanced": all(row["balanced_cut"] for row in rows),
        "even_L_only": all(row["L"] % 2 == 0 for row in rows),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


def save_plot(rows, path, title):
    path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["L"] for row in rows]
    ys = [row["MI"] for row in rows]
    yerr = []
    has_error_bars = all(row["bootstrap_std"] is not None for row in rows)
    if has_error_bars:
        yerr = [row["bootstrap_std"] for row in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    if has_error_bars:
        ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linewidth=1.5)
    else:
        ax.plot(xs, ys, marker="o", linewidth=1.5)

    for row in rows:
        if not row["balanced_cut"]:
            ax.annotate("unbalanced", (row["L"], row["MI"]), xytext=(4, 6), textcoords="offset points", fontsize=8)

    ax.set_xlabel("L")
    ax.set_ylabel("I_q(A;B)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    args = parse_args()
    paths = collect_result_paths(args)
    rows = [load_point(path) for path in paths]
    rows.sort(key=lambda row: (row["L"], row["seed"], row["path"]))

    output_json = resolve_path(args.output_json)
    output_csv = resolve_path(args.output_csv)
    output_plot = resolve_path(args.output_plot)

    save_json(rows, output_json)
    save_csv(rows, output_csv)
    save_plot(rows, output_plot, args.title)

    print(f"points: {len(rows)}")
    for row in rows:
        balance = "balanced" if row["balanced_cut"] else "unbalanced"
        print(
            f"L={row['L']} MI={row['MI']:.6f} "
            f"|A|={row['len_A']} |B|={row['len_B']} {balance}"
        )
    print(f"saved json: {output_json}")
    print(f"saved csv: {output_csv}")
    print(f"saved plot: {output_plot}")


if __name__ == "__main__":
    main()
