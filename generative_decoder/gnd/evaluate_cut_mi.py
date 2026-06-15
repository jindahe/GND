import argparse
from pathlib import Path

import torch

from .models import build_model, get_device, parse_dtype, sample_model
from .partitions import all_outline_cuts, build_cut
from .records import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate outline cut MI from true samples or a GND model.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-path")
    source.add_argument("--checkpoint")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--cut", choices=["all", "middle", "quarter", "three_quarter"], default="all")
    parser.add_argument("--samples", type=int, default=10000, help="Model samples or maximum dataset samples.")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def rows_to_integer_ids(samples):
    samples = samples.to(device="cpu", dtype=torch.int64)
    _, inverse = torch.unique(samples, dim=0, return_inverse=True)
    return inverse


def entropy_from_ids(ids):
    counts = torch.bincount(ids)
    probabilities = counts[counts > 0].to(torch.float64) / ids.numel()
    return -(probabilities * probabilities.log()).sum().item(), int(probabilities.numel())


def estimate_plugin_mi(samples, cut):
    a = samples[:, cut["a_indices"]]
    b = samples[:, cut["b_indices"]]
    ab = torch.cat([a, b], dim=1)
    a_ids = rows_to_integer_ids(a)
    b_ids = rows_to_integer_ids(b)
    ab_ids = rows_to_integer_ids(ab)
    h_a, unique_a = entropy_from_ids(a_ids)
    h_b, unique_b = entropy_from_ids(b_ids)
    h_ab, unique_ab = entropy_from_ids(ab_ids)
    return {
        "cut": cut,
        "estimator": "empirical_discrete_plugin",
        "log_unit": "nats",
        "n_samples": samples.size(0),
        "entropy": {"H_A": h_a, "H_B": h_b, "H_AB": h_ab},
        "unique_states": {"A": unique_a, "B": unique_b, "AB": unique_ab},
        "mi": h_a + h_b - h_ab,
    }


def bootstrap_mi(samples, cut, n_bootstrap, seed):
    if n_bootstrap <= 0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    estimates = []
    for _ in range(n_bootstrap):
        indices = torch.randint(samples.size(0), (samples.size(0),), generator=generator)
        estimates.append(estimate_plugin_mi(samples[indices], cut)["mi"])
    values = torch.tensor(estimates, dtype=torch.float64)
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=True).item() if n_bootstrap > 1 else 0.0,
        "ci95_low": torch.quantile(values, 0.025).item(),
        "ci95_high": torch.quantile(values, 0.975).item(),
    }


def load_dataset_samples(args):
    path = resolve_path(args.dataset_path)
    data = torch.load(path, map_location="cpu")
    samples = data[args.split]
    if args.samples > 0:
        samples = samples[: args.samples]
    return path, data["layout"], data["meta"], data["target"], samples, None


def load_checkpoint_samples(args):
    path = resolve_path(args.checkpoint)
    checkpoint = torch.load(path, map_location="cpu")
    config = checkpoint["model_config"]
    device = get_device(args.device)
    dtype = parse_dtype(config["dtype"])
    model = build_model(config=config, n_bits=config["n_bits"], device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.manual_seed(args.sample_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.sample_seed)
    samples = sample_model(model, config["n_type"], args.samples).cpu()
    capacity = {
        "parameter_count": config.get("parameter_count"),
        "n_bits": config["n_bits"],
        "made_width": config.get("width") if config["n_type"] == "made" else None,
        "nade_hidden_dim": config.get("hidden_dim") if config["n_type"] == "nade" else None,
        "trade_d_model": config.get("d_model") if config["n_type"] == "trade" else None,
    }
    return path, checkpoint["layout"], checkpoint["dataset_meta"], checkpoint["dataset_target"], samples, capacity


def main():
    args = parse_args()
    if args.samples <= 0:
        raise ValueError("--samples must be positive")

    if args.dataset_path:
        source_path, layout, meta, target, samples, capacity = load_dataset_samples(args)
        source_type = "true_error_model_samples"
    else:
        source_path, layout, meta, target, samples, capacity = load_checkpoint_samples(args)
        source_type = "gnd_model_samples"

    cuts = all_outline_cuts(layout) if args.cut == "all" else [build_cut(layout, args.cut)]
    results = []
    for index, cut in enumerate(cuts):
        result = estimate_plugin_mi(samples, cut)
        bootstrap = bootstrap_mi(samples, cut, args.bootstrap_samples, args.bootstrap_seed + index)
        if bootstrap is not None:
            result["bootstrap"] = bootstrap
        results.append(result)
        print(f"{cut['description']} = {result['mi']:.8f} nats")

    payload = {
        "record_type": "gnd_cut_mi",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "source_type": source_type,
        "source_path": str(source_path),
        "target": target,
        "code": meta,
        "model_capacity": capacity,
        "sample_seed": args.sample_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "note": (
            "CMI in outline.md is interpreted as ordinary bipartite mutual information. "
            "The empirical discrete plug-in estimator is biased in high-dimensional sparse regimes."
        ),
        "results": results,
    }

    if args.output_path:
        output_path = resolve_path(args.output_path)
        write_json_record(output_path, payload)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
