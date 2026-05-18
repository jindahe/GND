import json
import sys
from pathlib import Path

import torch

from args import args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module import MADE, NADE, TraDE_binary  # noqa: E402
from record_utils import utc_timestamp  # noqa: E402


def parse_dtype(name):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def get_device():
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {args.device} was requested but torch.cuda.is_available() is False in this environment"
        )
    return device


def resolve_checkpoint_path(order):
    explicit = args.ab_checkpoint if order == "AB" else args.ba_checkpoint
    if explicit:
        path = Path(explicit)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    checkpoint_dir = Path(args.save_dir) if args.save_dir else PROJECT_ROOT / "net" / "syndrome_models"
    if not checkpoint_dir.is_absolute():
        checkpoint_dir = PROJECT_ROOT / checkpoint_dir

    suffix = f"{order}_{args.partition_axis}{args.cut if args.cut is not None else 'mid'}"
    filename = (
        f"{args.n_type}_{args.c_type}_n{args.n}_d{args.d}_k{args.k}_seed{args.seed}"
        f"_er{args.er}_{args.e_model}_{suffix}.pt"
    )
    return checkpoint_dir / filename


def resolve_output_path():
    if not args.mi_output_path:
        return None
    path = Path(args.mi_output_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_model(model_config, n_type, device):
    dtype = parse_dtype(model_config["dtype"])
    n_bits = model_config["n_bits"]

    if n_type == "made":
        model = MADE(
            n=n_bits,
            depth=model_config["depth"],
            width=model_config.get("effective_width", model_config["width"]),
            activator=model_config.get("made_activation", "tanh"),
            residual=model_config.get("made_residual", False),
        )
    elif n_type == "nade":
        model = NADE(
            n=n_bits,
            hidden_dim=model_config["hidden_dim"],
            device=device,
            dtype=dtype,
        )
    elif n_type == "trade":
        model = TraDE_binary(
            n=n_bits,
            d_model=model_config["d_model"],
            n_heads=model_config["n_heads"],
            d_ff=model_config["d_ff"],
            n_layers=model_config["n_layers"],
            device=str(device),
            dropout=0,
        )
    else:
        raise ValueError(f"Unsupported model type: {n_type}")

    return model.to(device).to(dtype)


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu")
    model_config = checkpoint["model_config"]
    n_type = model_config["n_type"]
    model = build_model(model_config=model_config, n_type=n_type, device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint, model


def validate_partition(ab_checkpoint, ba_checkpoint):
    partition_ab = ab_checkpoint["partition"]
    partition_ba = ba_checkpoint["partition"]
    if partition_ab is None or partition_ba is None:
        raise ValueError("Both checkpoints must contain toric partition metadata")

    keys = ["axis", "cut", "idx_A", "idx_B", "order_AB", "order_BA"]
    for key in keys:
        if partition_ab[key] != partition_ba[key]:
            raise ValueError(f"AB/BA checkpoints disagree on partition field: {key}")

    config_ab = ab_checkpoint["model_config"]
    config_ba = ba_checkpoint["model_config"]
    if config_ab["n_bits"] != config_ba["n_bits"]:
        raise ValueError("AB/BA checkpoints disagree on syndrome dimensionality")
    if config_ab["n_type"] != config_ba["n_type"]:
        raise ValueError("AB/BA checkpoints disagree on model type")

    return partition_ab


def sample_from_model(model, n_type, batch_size):
    if n_type == "made":
        return model.sample(batch_size)
    if n_type in {"nade", "trade"}:
        return model.sample(batch_size)
    raise ValueError(f"Unsupported model type: {n_type}")


def token_log_prob(model, n_type, samples):
    if n_type == "made":
        return model.token_log_prob(samples)
    if n_type in {"nade", "trade"}:
        return model.token_log_prob(samples)
    raise ValueError(f"Unsupported model type: {n_type}")


@torch.no_grad()
def estimate_nll_terms(model, n_type, n_samples, chunk_size, prefix_len):
    joint_terms = []
    prefix_terms = []

    remaining = n_samples
    while remaining > 0:
        batch_size = min(chunk_size, remaining)
        samples = sample_from_model(model, n_type, batch_size)
        token_terms = -token_log_prob(model, n_type, samples)
        joint_terms.append(token_terms.sum(dim=1).cpu())
        prefix_terms.append(token_terms[:, :prefix_len].sum(dim=1).cpu())
        remaining -= batch_size

    return torch.cat(joint_terms), torch.cat(prefix_terms)


def bootstrap_mi(ab_joint, a_prefix, b_prefix, n_bootstrap, seed):
    if n_bootstrap <= 0:
        return None

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    n_ab = ab_joint.numel()
    n_b = b_prefix.numel()
    estimates = torch.empty(n_bootstrap, dtype=torch.float64)

    for i in range(n_bootstrap):
        ab_idx = torch.randint(n_ab, (n_ab,), generator=generator)
        b_idx = torch.randint(n_b, (n_b,), generator=generator)
        estimates[i] = a_prefix[ab_idx].mean() + b_prefix[b_idx].mean() - ab_joint[ab_idx].mean()

    return {
        "mean": estimates.mean().item(),
        "std": estimates.std(unbiased=True).item() if n_bootstrap > 1 else 0.0,
        "ci95_low": torch.quantile(estimates, 0.025).item(),
        "ci95_high": torch.quantile(estimates, 0.975).item(),
    }


def main():
    if args.c_type != "tor":
        raise ValueError("mi_bipartite.py currently supports only toric syndrome partitions")
    if args.mi_samples <= 0:
        raise ValueError("mi_samples must be positive")

    device = get_device()
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    ab_path = resolve_checkpoint_path("AB")
    ba_path = resolve_checkpoint_path("BA")
    ab_checkpoint, ab_model = load_checkpoint(ab_path, device=device)
    ba_checkpoint, ba_model = load_checkpoint(ba_path, device=device)
    partition = validate_partition(ab_checkpoint, ba_checkpoint)

    n_type = ab_checkpoint["model_config"]["n_type"]
    len_a = len(partition["idx_A"])
    len_b = len(partition["idx_B"])

    ab_joint, a_prefix = estimate_nll_terms(
        model=ab_model,
        n_type=n_type,
        n_samples=args.mi_samples,
        chunk_size=args.chunk_size,
        prefix_len=len_a,
    )
    _, b_prefix = estimate_nll_terms(
        model=ba_model,
        n_type=n_type,
        n_samples=args.mi_samples,
        chunk_size=args.chunk_size,
        prefix_len=len_b,
    )

    h_ab = ab_joint.mean().item()
    h_a = a_prefix.mean().item()
    h_b = b_prefix.mean().item()
    mi = h_a + h_b - h_ab

    bootstrap = bootstrap_mi(
        ab_joint=ab_joint,
        a_prefix=a_prefix,
        b_prefix=b_prefix,
        n_bootstrap=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )

    result = {
        "record_type": "bipartite_mi",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "observable": "syndrome",
        "estimator": "q-model Monte Carlo",
        "code": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
        },
        "checkpoint_ab": str(ab_path),
        "checkpoint_ba": str(ba_path),
        "device": str(device),
        "n_type": n_type,
        "mi_samples": args.mi_samples,
        "eval_seed": args.eval_seed,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_seed": args.bootstrap_seed,
        "partition": {
            "axis": partition["axis"],
            "cut": partition["cut"],
            "len_A": len_a,
            "len_B": len_b,
        },
        "entropy": {
            "H_AB": h_ab,
            "H_A": h_a,
            "H_B": h_b,
        },
        "mi": mi,
    }
    if bootstrap is not None:
        result["bootstrap"] = bootstrap

    print(f"checkpoint_ab: {ab_path}")
    print(f"checkpoint_ba: {ba_path}")
    print(f"device: {device}")
    print(f"partition: axis={partition['axis']} cut={partition['cut']} |A|={len_a} |B|={len_b}")
    print(f"H_q(A,B) = {h_ab:.6f}")
    print(f"H_q(A)   = {h_a:.6f}")
    print(f"H_q(B)   = {h_b:.6f}")
    print(f"I_q(A;B) = {mi:.6f}")
    if bootstrap is not None:
        print(
            "bootstrap: "
            f"mean={bootstrap['mean']:.6f} std={bootstrap['std']:.6f} "
            f"ci95=[{bootstrap['ci95_low']:.6f}, {bootstrap['ci95_high']:.6f}]"
        )

    output_path = resolve_output_path()
    if output_path is not None:
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
