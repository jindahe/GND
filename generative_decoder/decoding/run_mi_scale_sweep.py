import argparse
import os
import subprocess
import sys
from pathlib import Path

from record_utils import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--l-values",
        type=int,
        nargs="+",
        required=True,
        help="Toric-code linear sizes L to evaluate.",
    )
    parser.add_argument(
        "--allow-unbalanced",
        action="store_true",
        help="Allow odd L values even though the default cut L//2 is not a symmetric bipartition.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used for subprocess steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device passed to training and MI evaluation.",
    )
    parser.add_argument(
        "--n-type",
        type=str,
        default="made",
        choices=["made", "nade", "trade"],
        help="Autoregressive model family.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Code seed.")
    parser.add_argument("--error-seed", type=int, default=51697, help="Error-sampling seed.")
    parser.add_argument("--split-seed", type=int, default=0, help="Dataset split seed.")
    parser.add_argument("--train-seed", type=int, default=0, help="Training RNG seed.")
    parser.add_argument("--bootstrap-seed", type=int, default=0, help="Bootstrap seed.")
    parser.add_argument("--er", type=float, default=0.05, help="Physical error rate.")
    parser.add_argument("--e-model", type=str, default="dep", help="Physical error model.")
    parser.add_argument("--partition-axis", type=str, default="x", choices=["x", "y"], help="Spatial cut axis.")
    parser.add_argument("--cut", type=int, default=None, help="Optional explicit cut position.")
    parser.add_argument("--n-train", type=int, default=10000, help="Training samples per L and order.")
    parser.add_argument("--n-val", type=int, default=2000, help="Validation samples per L and order.")
    parser.add_argument("--n-test", type=int, default=2000, help="Test samples per L and order.")
    parser.add_argument("--mi-samples", type=int, default=10000, help="Monte Carlo samples for MI estimation.")
    parser.add_argument("--bootstrap-samples", type=int, default=200, help="Bootstrap resamples.")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for MI estimation.")
    parser.add_argument("--epoch", type=int, default=100, help="Training epochs.")
    parser.add_argument("--batch", type=int, default=256, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=0.001, help="Training learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Adam/AdamW weight decay.")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5, help="Validation-plateau LR decay factor.")
    parser.add_argument(
        "--lr-decay-patience",
        type=int,
        default=5,
        help="Validation-plateau patience before LR decay.",
    )
    parser.add_argument("--min-lr", type=float, default=0.0002, help="Minimum learning rate.")
    parser.add_argument("--log-every", type=int, default=10, help="Training log cadence.")
    parser.add_argument(
        "--early-stop-patience",
        type=int,
        default=20,
        help="Stop after this many unimproved validation epochs; set <=0 to disable.",
    )
    parser.add_argument(
        "--early-stop-min-delta",
        type=float,
        default=0.0,
        help="Minimum validation-NLL improvement required to reset early stopping.",
    )
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"], help="Training dtype.")
    parser.add_argument("--depth", type=int, default=0, help="MADE depth.")
    parser.add_argument("--width", type=int, default=32, help="MADE width.")
    parser.add_argument(
        "--made-activation",
        type=str,
        default="tanh",
        choices=["tanh", "relu", "sigmoid"],
        help="MADE activation.",
    )
    parser.add_argument("--made-residual", action="store_true", help="Enable MADE residual wrapper.")
    parser.add_argument(
        "--made-max-params",
        type=int,
        default=0,
        help="Optional soft cap for MADE parameter count; width is shrunk automatically when positive.",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="NADE hidden dimension.")
    parser.add_argument("--d-model", type=int, default=256, help="TraDE model width.")
    parser.add_argument("--n-heads", type=int, default=4, help="TraDE attention heads.")
    parser.add_argument("--d-ff", type=int, default=256, help="TraDE feed-forward width.")
    parser.add_argument("--n-layers", type=int, default=1, help="TraDE layer count.")
    parser.add_argument(
        "--code-dir",
        type=str,
        default="code",
        help="Directory containing saved code instances, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default="net/mi_scaling/datasets",
        help="Directory for syndrome datasets, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="net/mi_scaling/models",
        help="Directory for trained syndrome models, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="net/mi_scaling/results",
        help="Directory for per-L MI JSON outputs, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--summary-dir",
        type=str,
        default="net/mi_scaling",
        help="Directory for aggregated P8 outputs, relative to repo root unless absolute.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Reuse existing code, dataset, model, and MI files when present.",
    )
    return parser.parse_args()


def resolve_path(path_str):
    path = Path(path_str)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def build_env():
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-gnd-mi-scale")
    return env


def training_seed_suffix(train_seed):
    return f"_tseed{train_seed}" if train_seed != 0 else ""


def run_step(cmd, env):
    print("exec:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def code_path(code_dir, l_value, seed):
    n = 2 * l_value * l_value
    return code_dir / f"tor_n{n}_d{l_value}_k2_seed{seed}"


def dataset_path(dataset_dir, l_value, seed, er, e_model, order, axis, cut):
    n = 2 * l_value * l_value
    cut_suffix = cut if cut is not None else "mid"
    return dataset_dir / f"tor_n{n}_d{l_value}_k2_seed{seed}_er{er}_{e_model}_{order}_{axis}{cut_suffix}.pt"


def model_path(model_dir, n_type, l_value, seed, er, e_model, order, axis, cut, train_seed):
    n = 2 * l_value * l_value
    cut_suffix = cut if cut is not None else "mid"
    train_suffix = training_seed_suffix(train_seed)
    return model_dir / f"{n_type}_tor_n{n}_d{l_value}_k2_seed{seed}_er{er}_{e_model}{train_suffix}_{order}_{axis}{cut_suffix}.pt"


def result_path(result_dir, n_type, l_value, seed, er, e_model, axis, cut, train_seed):
    n = 2 * l_value * l_value
    cut_suffix = cut if cut is not None else "mid"
    train_suffix = training_seed_suffix(train_seed)
    return result_dir / f"{n_type}_tor_n{n}_d{l_value}_k2_seed{seed}_er{er}_{e_model}{train_suffix}_{axis}{cut_suffix}.json"


def validate_l_values(l_values, allow_unbalanced):
    odd_values = [value for value in l_values if value % 2 == 1]
    if odd_values and not allow_unbalanced:
        raise ValueError(
            "Odd L values produce an unbalanced default bipartition under cut=L//2. "
            f"Refusing to continue for L={odd_values}. Pass --allow-unbalanced to override."
        )


def maybe_generate_code(args, l_value, code_dir, env):
    path = code_path(code_dir, l_value, args.seed)
    if path.exists() and args.skip_existing:
        print(f"reuse code: {path}", flush=True)
        return

    n = 2 * l_value * l_value
    cmd = [
        args.python,
        "decoding/code_generator.py",
        "-c_type",
        "tor",
        "-n",
        str(n),
        "-d",
        str(l_value),
        "-k",
        "2",
        "-seed",
        str(args.seed),
    ]
    run_step(cmd, env)


def maybe_generate_dataset(args, l_value, order, dataset_dir, env):
    path = dataset_path(dataset_dir, l_value, args.seed, args.er, args.e_model, order, args.partition_axis, args.cut)
    if path.exists() and args.skip_existing:
        print(f"reuse dataset: {path}", flush=True)
        return

    n = 2 * l_value * l_value
    cmd = [
        args.python,
        "decoding/syndrome_dataset.py",
        "-c_type",
        "tor",
        "-n",
        str(n),
        "-d",
        str(l_value),
        "-k",
        "2",
        "-seed",
        str(args.seed),
        "-e_model",
        args.e_model,
        "-er",
        str(args.er),
        "-error_seed",
        str(args.error_seed),
        "-n_train",
        str(args.n_train),
        "-n_val",
        str(args.n_val),
        "-n_test",
        str(args.n_test),
        "-partition_axis",
        args.partition_axis,
        "-partition_order",
        order,
        "-dataset_dir",
        str(dataset_dir),
        "-split_seed",
        str(args.split_seed),
    ]
    if args.cut is not None:
        cmd.extend(["-cut", str(args.cut)])
    run_step(cmd, env)


def training_common_args(args, l_value, order, dataset_dir, model_dir):
    n = 2 * l_value * l_value
    cmd = [
        args.python,
        "decoding/train_mi_syndrome.py",
        "-c_type",
        "tor",
        "-n",
        str(n),
        "-d",
        str(l_value),
        "-k",
        "2",
        "-seed",
        str(args.seed),
        "-e_model",
        args.e_model,
        "-er",
        str(args.er),
        "-train_seed",
        str(args.train_seed),
        "-n_type",
        args.n_type,
        "-device",
        args.device,
        "-dtype",
        args.dtype,
        "-epoch",
        str(args.epoch),
        "-batch",
        str(args.batch),
        "-lr",
        str(args.lr),
        "-weight_decay",
        str(args.weight_decay),
        "-lr_decay_factor",
        str(args.lr_decay_factor),
        "-lr_decay_patience",
        str(args.lr_decay_patience),
        "-min_lr",
        str(args.min_lr),
        "-log_every",
        str(args.log_every),
        "-early_stop_patience",
        str(args.early_stop_patience),
        "-early_stop_min_delta",
        str(args.early_stop_min_delta),
        "-partition_axis",
        args.partition_axis,
        "-partition_order",
        order,
        "-dataset_dir",
        str(dataset_dir),
        "-save",
        "True",
        "-save_dir",
        str(model_dir),
    ]
    if args.cut is not None:
        cmd.extend(["-cut", str(args.cut)])
    if args.n_type == "made":
        cmd.extend(
            [
                "-depth",
                str(args.depth),
                "-width",
                str(args.width),
                "-made_activation",
                args.made_activation,
                "-made_residual",
                str(args.made_residual),
                "-made_max_params",
                str(args.made_max_params),
            ]
        )
    elif args.n_type == "nade":
        cmd.extend(["-hidden_dim", str(args.hidden_dim)])
    elif args.n_type == "trade":
        cmd.extend(
            [
                "-d_model",
                str(args.d_model),
                "-n_heads",
                str(args.n_heads),
                "-d_ff",
                str(args.d_ff),
                "-n_layers",
                str(args.n_layers),
            ]
        )
    return cmd


def maybe_train_model(args, l_value, order, dataset_dir, model_dir, env):
    path = model_path(
        model_dir,
        args.n_type,
        l_value,
        args.seed,
        args.er,
        args.e_model,
        order,
        args.partition_axis,
        args.cut,
        args.train_seed,
    )
    if path.exists() and args.skip_existing:
        print(f"reuse model: {path}", flush=True)
        return

    cmd = training_common_args(args, l_value, order, dataset_dir, model_dir)
    run_step(cmd, env)


def maybe_evaluate_mi(args, l_value, model_dir, result_dir, env):
    path = result_path(
        result_dir,
        args.n_type,
        l_value,
        args.seed,
        args.er,
        args.e_model,
        args.partition_axis,
        args.cut,
        args.train_seed,
    )
    if path.exists() and args.skip_existing:
        print(f"reuse result: {path}", flush=True)
        return path

    n = 2 * l_value * l_value
    cmd = [
        args.python,
        "decoding/mi_bipartite.py",
        "-c_type",
        "tor",
        "-n",
        str(n),
        "-d",
        str(l_value),
        "-k",
        "2",
        "-seed",
        str(args.seed),
        "-e_model",
        args.e_model,
        "-er",
        str(args.er),
        "-train_seed",
        str(args.train_seed),
        "-n_type",
        args.n_type,
        "-device",
        args.device,
        "-partition_axis",
        args.partition_axis,
        "-save_dir",
        str(model_dir),
        "-mi_samples",
        str(args.mi_samples),
        "-chunk_size",
        str(args.chunk_size),
        "-bootstrap_samples",
        str(args.bootstrap_samples),
        "-bootstrap_seed",
        str(args.bootstrap_seed),
        "-mi_output_path",
        str(path),
    ]
    if args.cut is not None:
        cmd.extend(["-cut", str(args.cut)])
    run_step(cmd, env)
    return path


def summarize(args, result_paths, summary_dir, env):
    json_path = summary_dir / "mi_vs_L.json"
    csv_path = summary_dir / "mi_vs_L.csv"
    plot_path = summary_dir / "mi_vs_L.png"

    cmd = [
        args.python,
        "decoding/mi_scale_analysis.py",
        "--output-json",
        str(json_path),
        "--output-csv",
        str(csv_path),
        "--output-plot",
        str(plot_path),
        "--title",
        f"{args.n_type.upper()} Bipartite MI vs L",
    ]
    for path in result_paths:
        cmd.extend(["--result-path", str(path)])
    run_step(cmd, env)


def write_sweep_manifest(args, result_paths, summary_dir):
    manifest_path = summary_dir / "sweep_manifest.json"
    manifest = {
        "record_type": "mi_scale_sweep",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "script": "decoding/run_mi_scale_sweep.py",
        "config": {
            "l_values": args.l_values,
            "allow_unbalanced": args.allow_unbalanced,
            "device": args.device,
            "n_type": args.n_type,
            "seed": args.seed,
            "error_seed": args.error_seed,
            "split_seed": args.split_seed,
            "train_seed": args.train_seed,
            "bootstrap_seed": args.bootstrap_seed,
            "er": args.er,
            "e_model": args.e_model,
            "partition_axis": args.partition_axis,
            "cut": args.cut,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "n_test": args.n_test,
            "mi_samples": args.mi_samples,
            "bootstrap_samples": args.bootstrap_samples,
            "chunk_size": args.chunk_size,
            "epoch": args.epoch,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "lr_decay_factor": args.lr_decay_factor,
            "lr_decay_patience": args.lr_decay_patience,
            "min_lr": args.min_lr,
            "log_every": args.log_every,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "dtype": args.dtype,
            "depth": args.depth,
            "width": args.width,
            "made_activation": args.made_activation,
            "made_residual": args.made_residual,
            "made_max_params": args.made_max_params,
            "hidden_dim": args.hidden_dim,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "skip_existing": args.skip_existing,
        },
        "artifacts": {
            "result_paths": [str(path) for path in result_paths],
            "summary_json": str(summary_dir / "mi_vs_L.json"),
            "summary_csv": str(summary_dir / "mi_vs_L.csv"),
            "summary_plot": str(summary_dir / "mi_vs_L.png"),
        },
    }
    write_json_record(manifest_path, manifest)
    print(f"manifest: {manifest_path}", flush=True)


def main():
    args = parse_args()
    validate_l_values(args.l_values, args.allow_unbalanced)

    code_dir = resolve_path(args.code_dir)
    dataset_dir = resolve_path(args.dataset_dir)
    model_dir = resolve_path(args.model_dir)
    result_dir = resolve_path(args.result_dir)
    summary_dir = resolve_path(args.summary_dir)

    dataset_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    canonical_code_dir = PROJECT_ROOT / "code"
    if code_dir.resolve() != canonical_code_dir.resolve():
        raise ValueError(
            "code_generator.py currently reads and writes toric code instances only under "
            f"{canonical_code_dir}. Received --code-dir={code_dir}."
        )

    env = build_env()
    result_paths = []

    for l_value in args.l_values:
        print(f"=== L={l_value} ===", flush=True)
        maybe_generate_code(args, l_value, code_dir, env)
        for order in ["AB", "BA"]:
            maybe_generate_dataset(args, l_value, order, dataset_dir, env)
            maybe_train_model(args, l_value, order, dataset_dir, model_dir, env)
        result_paths.append(maybe_evaluate_mi(args, l_value, model_dir, result_dir, env))

    summarize(args, result_paths, summary_dir, env)
    write_sweep_manifest(args, result_paths, summary_dir)


if __name__ == "__main__":
    main()
