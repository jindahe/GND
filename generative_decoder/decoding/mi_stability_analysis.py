import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from record_utils import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-sizes", type=int, nargs="+", required=True, help="Monte Carlo sample sizes to test.")
    parser.add_argument("--eval-seeds", type=int, nargs="+", required=True, help="Evaluation seeds to repeat per sample size.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter used for subprocess calls.")
    parser.add_argument("--c-type", type=str, default="tor", help="Code type.")
    parser.add_argument("--n", type=int, required=True, help="Number of physical qubits.")
    parser.add_argument("--d", type=int, required=True, help="Code distance / toric linear size.")
    parser.add_argument("--k", type=int, default=2, help="Number of logical qubits.")
    parser.add_argument("--seed", type=int, default=0, help="Code seed.")
    parser.add_argument("--e-model", type=str, default="dep", help="Physical error model.")
    parser.add_argument("--er", type=float, default=0.05, help="Physical error rate.")
    parser.add_argument("--n-type", type=str, default="made", choices=["made", "nade", "trade"], help="Autoregressive model family.")
    parser.add_argument("--device", type=str, default="cpu", help="Device used by MI evaluation.")
    parser.add_argument("--partition-axis", type=str, default="x", choices=["x", "y"], help="Spatial cut axis.")
    parser.add_argument("--cut", type=int, default=None, help="Optional explicit cut position.")
    parser.add_argument("--save-dir", type=str, default="net/mi_scaling/models", help="Checkpoint directory passed through to mi_bipartite.py.")
    parser.add_argument("--ab-checkpoint", type=str, default="", help="Optional explicit AB checkpoint path.")
    parser.add_argument("--ba-checkpoint", type=str, default="", help="Optional explicit BA checkpoint path.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for MI estimation.")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Bootstrap resamples per run.")
    parser.add_argument("--bootstrap-seed-base", type=int, default=1000, help="Base seed for bootstrap RNG; eval seed is added to this value.")
    parser.add_argument("--result-dir", type=str, default="net/mi_stability/results", help="Directory for per-run MI JSON outputs.")
    parser.add_argument("--summary-dir", type=str, default="net/mi_stability", help="Directory for grouped summary outputs.")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing per-run MI JSON outputs when present.")
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_env():
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gnd-mi-stability")
    return env


def run_step(cmd, env):
    print("exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def result_path(result_dir, args, sample_size, eval_seed):
    cut_suffix = args.cut if args.cut is not None else "mid"
    filename = (
        f"{args.n_type}_{args.c_type}_n{args.n}_d{args.d}_k{args.k}_seed{args.seed}"
        f"_er{args.er}_{args.e_model}_{args.partition_axis}{cut_suffix}"
        f"_samples{sample_size}_eseed{eval_seed}.json"
    )
    return result_dir / filename


def maybe_run_single(args, sample_size, eval_seed, result_dir, env):
    output_path = result_path(result_dir, args, sample_size, eval_seed)
    if output_path.exists() and args.skip_existing:
        print(f"reuse result: {output_path}", flush=True)
        return output_path

    bootstrap_seed = args.bootstrap_seed_base + eval_seed
    cmd = [
        args.python,
        "decoding/mi_bipartite.py",
        "-c_type",
        args.c_type,
        "-n",
        str(args.n),
        "-d",
        str(args.d),
        "-k",
        str(args.k),
        "-seed",
        str(args.seed),
        "-e_model",
        args.e_model,
        "-er",
        str(args.er),
        "-n_type",
        args.n_type,
        "-device",
        args.device,
        "-partition_axis",
        args.partition_axis,
        "-save_dir",
        str(resolve_path(args.save_dir)),
        "-mi_samples",
        str(sample_size),
        "-eval_seed",
        str(eval_seed),
        "-chunk_size",
        str(args.chunk_size),
        "-bootstrap_samples",
        str(args.bootstrap_samples),
        "-bootstrap_seed",
        str(bootstrap_seed),
        "-mi_output_path",
        str(output_path),
    ]
    if args.cut is not None:
        cmd.extend(["-cut", str(args.cut)])
    if args.ab_checkpoint:
        cmd.extend(["-ab_checkpoint", str(resolve_path(args.ab_checkpoint))])
    if args.ba_checkpoint:
        cmd.extend(["-ba_checkpoint", str(resolve_path(args.ba_checkpoint))])
    run_step(cmd, env)
    return output_path


def load_point(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    bootstrap = data.get("bootstrap", {})
    return {
        "path": str(path),
        "mi_samples": data["mi_samples"],
        "eval_seed": data.get("eval_seed"),
        "bootstrap_samples": data.get("bootstrap_samples"),
        "bootstrap_seed": data.get("bootstrap_seed"),
        "mi": data["mi"],
        "H_AB": data["entropy"]["H_AB"],
        "H_A": data["entropy"]["H_A"],
        "H_B": data["entropy"]["H_B"],
        "bootstrap_std": bootstrap.get("std"),
        "bootstrap_ci95_low": bootstrap.get("ci95_low"),
        "bootstrap_ci95_high": bootstrap.get("ci95_high"),
    }


def summarize_points(rows):
    grouped = []
    sample_sizes = sorted({row["mi_samples"] for row in rows})
    for sample_size in sample_sizes:
        subset = [row for row in rows if row["mi_samples"] == sample_size]
        mi_values = [row["mi"] for row in subset]
        grouped.append(
            {
                "mi_samples": sample_size,
                "n_repeats": len(subset),
                "eval_seeds": [row["eval_seed"] for row in subset],
                "mi_mean": statistics.fmean(mi_values),
                "mi_std_across_seeds": statistics.stdev(mi_values) if len(mi_values) > 1 else 0.0,
                "mi_min": min(mi_values),
                "mi_max": max(mi_values),
                "mean_bootstrap_std": statistics.fmean(
                    [row["bootstrap_std"] for row in subset if row["bootstrap_std"] is not None]
                )
                if any(row["bootstrap_std"] is not None for row in subset)
                else None,
            }
        )
    return grouped


def save_csv(rows, grouped, summary_dir):
    raw_csv = summary_dir / "mi_stability_raw.csv"
    grouped_csv = summary_dir / "mi_stability_grouped.csv"

    with raw_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mi_samples",
                "eval_seed",
                "bootstrap_samples",
                "bootstrap_seed",
                "mi",
                "H_AB",
                "H_A",
                "H_B",
                "bootstrap_std",
                "bootstrap_ci95_low",
                "bootstrap_ci95_high",
                "path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    with grouped_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "mi_samples",
                "n_repeats",
                "eval_seeds",
                "mi_mean",
                "mi_std_across_seeds",
                "mi_min",
                "mi_max",
                "mean_bootstrap_std",
            ],
        )
        writer.writeheader()
        writer.writerows(grouped)


def save_plot(grouped, summary_dir):
    plot_path = summary_dir / "mi_stability.png"
    xs = [row["mi_samples"] for row in grouped]
    ys = [row["mi_mean"] for row in grouped]
    yerr = [row["mi_std_across_seeds"] for row in grouped]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, ys, yerr=yerr, marker="o", capsize=4, linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Monte Carlo sample size")
    ax.set_ylabel("I_q(A;B)")
    ax.set_title("MI Stability vs Sample Size")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)
    return plot_path


def main():
    args = parse_args()
    result_dir = resolve_path(args.result_dir)
    summary_dir = resolve_path(args.summary_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    env = build_env()
    rows = []
    for sample_size in args.sample_sizes:
        for eval_seed in args.eval_seeds:
            path = maybe_run_single(args, sample_size, eval_seed, result_dir, env)
            rows.append(load_point(path))

    rows.sort(key=lambda row: (row["mi_samples"], row["eval_seed"]))
    grouped = summarize_points(rows)

    save_csv(rows, grouped, summary_dir)
    plot_path = save_plot(grouped, summary_dir)

    summary_payload = {
        "record_type": "mi_stability_summary",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "script": "decoding/mi_stability_analysis.py",
        "config": {
            "sample_sizes": args.sample_sizes,
            "eval_seeds": args.eval_seeds,
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
            "n_type": args.n_type,
            "device": args.device,
            "partition_axis": args.partition_axis,
            "cut": args.cut,
            "chunk_size": args.chunk_size,
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_seed_base": args.bootstrap_seed_base,
            "save_dir": str(resolve_path(args.save_dir)),
        },
        "raw_points": rows,
        "grouped_summary": grouped,
        "artifacts": {
            "result_dir": str(result_dir),
            "raw_csv": str(summary_dir / "mi_stability_raw.csv"),
            "grouped_csv": str(summary_dir / "mi_stability_grouped.csv"),
            "plot": str(plot_path),
        },
    }
    summary_path = summary_dir / "mi_stability_summary.json"
    write_json_record(summary_path, summary_payload)

    print(f"points: {len(rows)}")
    for row in grouped:
        print(
            f"samples={row['mi_samples']} "
            f"mean={row['mi_mean']:.6f} "
            f"std_across_seeds={row['mi_std_across_seeds']:.6f} "
            f"repeats={row['n_repeats']}"
        )
    print(f"saved summary: {summary_path}")
    print(f"saved plot: {plot_path}")


if __name__ == "__main__":
    main()
