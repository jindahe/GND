import sys
from pathlib import Path

import torch

from .args import args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "syndrome_only_mi"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module import (  # noqa: E402
    Errormodel,
    Loading_code,
    read_code,
    reorder_bits,
    sample_syndromes,
    split_samples,
    toric_bipartition,
)


def resolve_output_path():
    if args.dataset_dir:
        output_dir = Path(args.dataset_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = ARCHIVE_ROOT / "net" / "syndrome_data"

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.partition_order}_{args.partition_axis}{args.cut if args.cut is not None else 'mid'}"
    filename = (
        f"{args.c_type}_n{args.n}_d{args.d}_k{args.k}_seed{args.seed}"
        f"_er{args.er}_{args.e_model}_{suffix}.pt"
    )
    return output_dir / filename


def main():
    total_samples = args.n_train + args.n_val + args.n_test
    if total_samples <= 0:
        raise ValueError("The dataset must contain at least one sample")

    info = read_code(d=args.d, k=args.k, n=args.n, seed=args.seed, c_type=args.c_type)
    code = Loading_code(info)
    error_model = Errormodel(args.er, e_model=args.e_model)
    syndromes = sample_syndromes(
        code=code,
        error_model=error_model,
        n_samples=total_samples,
        seed=args.error_seed,
        device="cpu",
        dtype=torch.float32,
    )

    partition = None
    applied_order = list(range(code.m))
    if args.c_type == "tor":
        partition = toric_bipartition(args.d, axis=args.partition_axis, cut=args.cut)
        if args.partition_order == "AB":
            applied_order = partition["order_AB"]
        elif args.partition_order == "BA":
            applied_order = partition["order_BA"]
        syndromes = reorder_bits(syndromes, applied_order)
    elif args.partition_order != "none":
        raise ValueError("partition_order is only supported for toric codes in the current syndrome-only pipeline")

    train, val, test = split_samples(
        samples=syndromes,
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        shuffle=args.shuffle,
        seed=args.split_seed,
    )

    output = {
        "meta": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
            "error_seed": args.error_seed,
            "shuffle": args.shuffle,
            "split_seed": args.split_seed,
            "partition_axis": args.partition_axis,
            "cut": args.cut,
            "partition_order": args.partition_order,
            "sample_counts": {
                "train": args.n_train,
                "val": args.n_val,
                "test": args.n_test,
            },
            "syndrome_dim": code.m,
        },
        "partition": partition,
        "applied_order": applied_order,
        "train": train,
        "val": val,
        "test": test,
    }

    path = resolve_output_path()
    torch.save(output, path)

    print(path)
    print("train:", tuple(train.shape))
    print("val:", tuple(val.shape))
    print("test:", tuple(test.shape))
    if partition is not None:
        print("cut:", partition["cut"])
        print("A/B:", len(partition["idx_A"]), len(partition["idx_B"]))
        print("order:", args.partition_order)


if __name__ == "__main__":
    main()
