import argparse
import json
import math
import time
from pathlib import Path

import torch

from module import Errormodel, Loading_code, read_code

from .beta_distribution import beta_entropy
from .exact_mi import enumerate_distribution, error_configs
from .partition_backends.binary_dense_elimination import BinaryDenseVariableEliminationSectorPartition
from .partition_backends.brute_force import BruteForceSectorPartition
from .partition_backends.elimination import VariableEliminationSectorPartition
from .partition_backends.toric_row_transfer import ToricRowTransferSectorPartition
from .records import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Structured true-distribution estimate for GND middle-cut I(beta:gamma)."
    )
    parser.add_argument("--c-type", default="tor")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e-model", default="dep", choices=["dep", "x", "z"])
    parser.add_argument("--er", type=float, required=True)
    parser.add_argument(
        "--backend",
        choices=[
            "brute_force",
            "elimination",
            "binary_dense_elimination",
            "toric_row_transfer",
        ],
        default="brute_force",
    )
    parser.add_argument("--gamma-samples", type=int, default=0)
    parser.add_argument("--gamma-mode", choices=["sample", "exhaustive"], default="sample")
    parser.add_argument("--sample-seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--max-exact-errors", type=int, default=20_000_000)
    parser.add_argument("--chunk-size", type=int, default=65_536)
    parser.add_argument("--elimination-max-intermediate-states", type=int, default=5_000_000)
    parser.add_argument("--toric-transfer-max-states", type=int, default=5_000_000)
    parser.add_argument("--toric-transfer-max-boundary-states", type=int, default=4096)
    parser.add_argument("--toric-transfer-max-dense-character-boundary-states", type=int, default=512)
    parser.add_argument("--output-path", default="")
    parser.add_argument("--sector-records-path", default="")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def sampled_gammas(code, k, error_model, n_samples, seed):
    errors = error_model.generate_error(code.n, m=n_samples, seed=seed)
    configs = error_configs(errors, code, k)
    return configs[:, : -2 * int(k)].to(torch.int64)


def exhaustive_gamma_records(code, args):
    layout, joint, total_errors = enumerate_distribution(code, args)
    del layout
    p_gamma = {}
    for key, probability in joint.items():
        gamma = key[: -2 * int(args.k)]
        p_gamma[gamma] = p_gamma.get(gamma, 0.0) + probability
    return [
        {
            "sample_index": index,
            "gamma": torch.tensor(gamma, dtype=torch.int64),
            "weight": probability,
        }
        for index, (gamma, probability) in enumerate(sorted(p_gamma.items()))
    ], total_errors


def mean(values):
    return sum(values) / len(values)


def sample_std(values):
    if len(values) < 2:
        return 0.0
    average = mean(values)
    return math.sqrt(sum((item - average) ** 2 for item in values) / (len(values) - 1))


def bootstrap_mean(values, n_bootstrap, seed):
    if n_bootstrap <= 0:
        return None
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    tensor = torch.tensor(values, dtype=torch.float64)
    estimates = []
    for _ in range(n_bootstrap):
        indices = torch.randint(tensor.numel(), (tensor.numel(),), generator=generator)
        estimates.append(tensor[indices].mean().item())
    estimates = torch.tensor(estimates, dtype=torch.float64)
    return {
        "mean": estimates.mean().item(),
        "std": estimates.std(unbiased=True).item() if n_bootstrap > 1 else 0.0,
        "ci95_low": torch.quantile(estimates, 0.025).item(),
        "ci95_high": torch.quantile(estimates, 0.975).item(),
    }


def build_backend(args, code):
    if args.backend == "brute_force":
        return BruteForceSectorPartition(
            code=code,
            k=args.k,
            er=args.er,
            e_model=args.e_model,
            max_exact_errors=args.max_exact_errors,
            chunk_size=args.chunk_size,
        )
    if args.backend == "elimination":
        return VariableEliminationSectorPartition(
            code=code,
            k=args.k,
            er=args.er,
            e_model=args.e_model,
            max_intermediate_states=args.elimination_max_intermediate_states,
        )
    if args.backend == "binary_dense_elimination":
        return BinaryDenseVariableEliminationSectorPartition(
            code=code,
            k=args.k,
            er=args.er,
            e_model=args.e_model,
            max_intermediate_states=args.elimination_max_intermediate_states,
        )
    if args.backend == "toric_row_transfer":
        return ToricRowTransferSectorPartition(
            code=code,
            k=args.k,
            er=args.er,
            e_model=args.e_model,
            d=args.d,
            max_states=args.toric_transfer_max_states,
            max_boundary_states=args.toric_transfer_max_boundary_states,
            max_dense_character_boundary_states=args.toric_transfer_max_dense_character_boundary_states,
        )
    raise ValueError(f"Unsupported backend: {args.backend}")


def main():
    started_at = time.time()
    args = parse_args()
    if args.gamma_mode == "sample" and args.gamma_samples <= 0:
        raise ValueError("--gamma-samples must be positive")

    code = Loading_code(read_code(d=args.d, k=args.k, n=args.n, seed=args.seed, c_type=args.c_type))
    error_model = Errormodel(args.er, e_model=args.e_model)
    h_beta, beta_distribution = beta_entropy(code, args.k, error_model)
    backend = build_backend(args, code)
    exact_gamma_support_size = None
    total_errors_enumerated = None
    if args.gamma_mode == "sample":
        gamma_records = [
            {"sample_index": index, "gamma": gamma, "weight": None}
            for index, gamma in enumerate(sampled_gammas(code, args.k, error_model, args.gamma_samples, args.sample_seed))
        ]
    else:
        gamma_records, total_errors_enumerated = exhaustive_gamma_records(code, args)
        exact_gamma_support_size = len(gamma_records)

    sector_records = []
    entropies = []
    weighted_h_cond = 0.0
    for record in gamma_records:
        index = record["sample_index"]
        gamma = record["gamma"]
        gamma_started_at = time.time()
        weights = backend.sector_weights(gamma)
        entropies.append(weights.entropy)
        if record["weight"] is not None:
            weighted_h_cond += record["weight"] * weights.entropy
        sector_records.append(
            {
                "sample_index": index,
                "gamma": [int(item) for item in gamma.tolist()],
                "gamma_probability": record["weight"],
                "log_z": weights.log_z,
                "posterior": weights.posterior,
                "entropy": weights.entropy,
                "elapsed_seconds": time.time() - gamma_started_at,
                "diagnostics": weights.diagnostics,
            }
        )

    if args.gamma_mode == "sample":
        h_cond = mean(entropies)
        h_cond_std = sample_std(entropies)
        h_cond_se = h_cond_std / math.sqrt(len(entropies))
    else:
        h_cond = weighted_h_cond
        h_cond_std = 0.0
        h_cond_se = 0.0
    mi = h_beta - h_cond
    bootstrap = bootstrap_mean(entropies, args.bootstrap_samples, args.bootstrap_seed) if args.gamma_mode == "sample" else None
    mi_bootstrap = None
    if bootstrap is not None:
        mi_bootstrap = {
            "mean": h_beta - bootstrap["mean"],
            "std": bootstrap["std"],
            "ci95_low": h_beta - bootstrap["ci95_high"],
            "ci95_high": h_beta - bootstrap["ci95_low"],
        }

    payload = {
        "record_type": "gnd_structured_true_middle_mi",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "quantity": "I_true(beta:gamma)",
        "cut": "middle",
        "log_unit": "nats",
        "source_type": "structured_true_distribution",
        "backend": args.backend,
        "code": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
        },
        "gamma_mode": args.gamma_mode,
        "gamma_samples": args.gamma_samples,
        "sample_seed": args.sample_seed,
        "exact_gamma_support_size": exact_gamma_support_size,
        "total_errors_enumerated": total_errors_enumerated,
        "H_beta": h_beta,
        "beta_distribution": {"".join(map(str, key)): value for key, value in beta_distribution.items()},
        "H_beta_given_gamma_mean": h_cond,
        "H_beta_given_gamma_sample_std": h_cond_std,
        "H_beta_given_gamma_standard_error": h_cond_se,
        "mi": mi,
        "bootstrap": mi_bootstrap,
        "elapsed_seconds": time.time() - started_at,
        "sector_records_path": args.sector_records_path or None,
        "note": (
            "This estimator samples gamma from the true error model but computes "
            "p(beta|gamma) through sector partition functions, avoiding high-dimensional "
            "plug-in entropy for gamma."
        ),
    }

    print(f"H_true(beta) = {h_beta:.12f} nats")
    print(f"mean H_true(beta|gamma) = {h_cond:.12f} nats")
    print(f"I_true(beta:gamma) = {mi:.12f} nats")
    print(f"MC stderr = {h_cond_se:.12f} nats")

    if args.sector_records_path:
        path = resolve_path(args.sector_records_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in sector_records:
                json.dump(record, handle)
                handle.write("\n")
        print(f"saved sector records: {path}")

    if args.output_path:
        output_path = resolve_path(args.output_path)
        write_json_record(output_path, payload)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
