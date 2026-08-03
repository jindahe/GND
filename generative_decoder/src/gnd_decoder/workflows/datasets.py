import argparse
import sys
from pathlib import Path

import torch

from gnd_decoder.core import Errormodel, Loading_code, mod2, read_code, split_samples

from .records import utc_timestamp


from gnd_decoder.paths import PROJECT_ROOT, ARTIFACTS_DIR, resolve_path, resolve_output_path

def parse_args():
    parser = argparse.ArgumentParser(description="Generate GND beta/gamma datasets from an error model.")
    parser.add_argument("--c-type", default="sur")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e-model", default="dep")
    parser.add_argument("--er", type=float, required=True)
    parser.add_argument("--error-seed", type=int, default=51697)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=10000)
    parser.add_argument("--n-val", type=int, default=2000)
    parser.add_argument("--n-test", type=int, default=2000)
    parser.add_argument("--target", choices=["beta_gamma", "full_config"], default="beta_gamma")
    parser.add_argument("--shuffle", action="store_true", default=True)
    parser.add_argument("--no-shuffle", action="store_false", dest="shuffle")
    parser.add_argument("--output-dir", default="artifacts/gnd/datasets")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def canonical_pure_errors(code, device="cpu", dtype=torch.float32):
    helper = mod2(device=device, dtype=dtype)
    pure = code.pure_es.clone()
    for i in range(code.m):
        conf = helper.commute(pure[i], pure)
        idx = conf.nonzero().squeeze().cpu()
        if idx.numel() == 0:
            continue
        stabilizers = code.g_stabilizer[idx]
        pure[i] = helper.opts_prod(torch.vstack([pure[i], stabilizers]))
    return pure


def make_layout(code, k, target):
    gamma_start = 0
    gamma_stop = code.m
    beta_start = gamma_stop
    beta_stop = beta_start + 2 * k
    layout = {
        "gamma": {"start": gamma_start, "stop": gamma_stop},
        "beta": {"start": beta_start, "stop": beta_stop},
    }
    if target == "full_config":
        layout["alpha"] = {"start": beta_stop, "stop": beta_stop + code.m}
    return layout


def select_target(configs, code, k, target):
    if target == "beta_gamma":
        return configs[:, : code.m + 2 * k]
    if target == "full_config":
        return configs
    raise ValueError(f"Unsupported target: {target}")


def default_output_path(args):
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"{args.target}_{args.c_type}_n{args.n}_d{args.d}_k{args.k}_seed{args.seed}"
        f"_er{args.er}_{args.e_model}_ntrain{args.n_train}.pt"
    )
    return output_dir / name


def main():
    args = parse_args()
    total = args.n_train + args.n_val + args.n_test
    if total <= 0:
        raise ValueError("Dataset must contain at least one sample")

    info = read_code(d=args.d, k=args.k, n=args.n, seed=args.seed, c_type=args.c_type)
    code = Loading_code(info)
    error_model = Errormodel(args.er, e_model=args.e_model)
    errors = error_model.generate_error(code.n, m=total, seed=args.error_seed)
    pure = canonical_pure_errors(code, dtype=torch.float32)
    configs = error_model.configs(
        sta=code.g_stabilizer,
        log=code.logical_opt,
        pe=pure,
        opts=errors,
    ).to(dtype=torch.float32)
    samples = select_target(configs=configs, code=code, k=args.k, target=args.target)
    train, val, test = split_samples(
        samples=samples,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        shuffle=args.shuffle,
        seed=args.split_seed,
    )

    output_path = Path(args.output_path) if args.output_path else default_output_path(args)
    if not output_path.is_absolute():
        output_path = resolve_output_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "record_type": "gnd_dataset",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "target": args.target,
        "meta": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
            "error_seed": args.error_seed,
            "split_seed": args.split_seed,
            "shuffle": args.shuffle,
            "sample_counts": {"train": args.n_train, "val": args.n_val, "test": args.n_test},
            "syndrome_dim": code.m,
            "beta_dim": 2 * args.k,
            "alpha_dim": code.m if args.target == "full_config" else 0,
        },
        "layout": make_layout(code, args.k, args.target),
        "train": train,
        "val": val,
        "test": test,
    }
    torch.save(payload, output_path)
    print(output_path)
    print("train:", tuple(train.shape))
    print("val:", tuple(val.shape))
    print("test:", tuple(test.shape))


if __name__ == "__main__":
    main()
