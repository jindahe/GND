import argparse
import itertools
import math
from pathlib import Path

import torch

from gnd_decoder.core import Errormodel, Loading_code, mod2, read_code

from .datasets import make_layout
from .partitions import all_outline_cuts, build_cut
from .records import utc_timestamp, write_json_record


from gnd_decoder.paths import PROJECT_ROOT, ARTIFACTS_DIR, resolve_path, resolve_output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Exact GND cut MI by exhaustive Pauli-error enumeration.")
    parser.add_argument("--c-type", default="sur")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e-model", default="dep", choices=["dep", "x", "z"])
    parser.add_argument("--er", type=float, required=True)
    parser.add_argument("--cut", choices=["all", "middle", "quarter", "three_quarter"], default="all")
    parser.add_argument(
        "--max-exact-errors",
        type=int,
        default=20_000_000,
        help="Refuse enumeration when 4^n exceeds this many errors.",
    )
    parser.add_argument("--chunk-size", type=int, default=65536)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()



def single_pauli_probabilities(error_model):
    probabilities = error_model.single_p
    return [float(item) for item in probabilities]


def mixed_radix_index(bits, indices):
    value = 0
    for index in indices:
        value = (value << 1) | int(bits[index].item())
    return value


def entropy_from_probs(probs):
    return -sum(prob * math.log(prob) for prob in probs if prob > 0.0)


def exact_cut_mi(joint_distribution, cut):
    p_a = {}
    p_b = {}
    for key, probability in joint_distribution.items():
        if probability == 0.0:
            continue
        bits = key
        a_key = tuple(bits[index] for index in cut["a_indices"])
        b_key = tuple(bits[index] for index in cut["b_indices"])
        p_a[a_key] = p_a.get(a_key, 0.0) + probability
        p_b[b_key] = p_b.get(b_key, 0.0) + probability

    h_a = entropy_from_probs(p_a.values())
    h_b = entropy_from_probs(p_b.values())
    h_ab = entropy_from_probs(joint_distribution.values())
    return {
        "cut": cut,
        "estimator": "exact_error_enumeration",
        "log_unit": "nats",
        "entropy": {"H_A": h_a, "H_B": h_b, "H_AB": h_ab},
        "support_size": {"A": len(p_a), "B": len(p_b), "AB": len(joint_distribution)},
        "mi": h_a + h_b - h_ab,
    }


def logical_indices(k):
    indices = []
    for index in range(k):
        indices.extend([2 * index + 1, 2 * index])
    return indices


def error_configs(errors, code, k):
    helper = mod2(device="cpu", dtype=torch.float32)
    gamma = helper.commute(errors, code.g_stabilizer)
    beta = helper.commute(errors, code.logical_opt)
    if gamma.dim() == 1:
        gamma = gamma.unsqueeze(0)
    if beta.dim() == 1:
        beta = beta.unsqueeze(0)
    beta = beta[:, logical_indices(k)]
    return torch.cat([gamma, beta], dim=1).to(dtype=torch.int64)


def chunked(iterator, chunk_size):
    while True:
        chunk = list(itertools.islice(iterator, chunk_size))
        if not chunk:
            return
        yield chunk


def enumerate_distribution(code, args):
    total_errors = 4 ** code.n
    if total_errors > args.max_exact_errors:
        raise ValueError(
            f"Refusing exact enumeration: 4^{code.n}={total_errors} exceeds "
            f"--max-exact-errors={args.max_exact_errors}"
        )

    error_model = Errormodel(args.er, e_model=args.e_model)
    single_probs = single_pauli_probabilities(error_model)
    layout = make_layout(code, args.k, target="beta_gamma")
    joint = {}

    iterator = itertools.product(range(4), repeat=code.n)
    for pauli_chunk in chunked(iterator, args.chunk_size):
        errors = torch.tensor(pauli_chunk, dtype=torch.float32)
        configs = error_configs(errors, code, args.k)
        for paulis, config in zip(pauli_chunk, configs):
            probability = 1.0
            for pauli in paulis:
                probability *= single_probs[pauli]
            if probability == 0.0:
                continue
            key = tuple(int(item) for item in config.tolist())
            joint[key] = joint.get(key, 0.0) + probability

    norm = sum(joint.values())
    if norm <= 0.0:
        raise ValueError("Exact distribution has zero total probability")
    for key in list(joint):
        joint[key] /= norm

    return layout, joint, total_errors


def main():
    args = parse_args()
    info = read_code(d=args.d, k=args.k, n=args.n, seed=args.seed, c_type=args.c_type)
    code = Loading_code(info)
    layout, joint_distribution, total_errors = enumerate_distribution(code, args)
    cuts = all_outline_cuts(layout) if args.cut == "all" else [build_cut(layout, args.cut)]
    results = [exact_cut_mi(joint_distribution, cut) for cut in cuts]

    for result in results:
        print(f"{result['cut']['description']} = {result['mi']:.12f} nats")

    payload = {
        "record_type": "gnd_exact_cut_mi",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "source_type": "exact_error_enumeration",
        "code": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
        },
        "target": "beta_gamma",
        "layout": layout,
        "total_errors_enumerated": total_errors,
        "joint_support_size": len(joint_distribution),
        "results": results,
    }
    if args.output_path:
        output_path = resolve_path(args.output_path)
        write_json_record(output_path, payload)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
