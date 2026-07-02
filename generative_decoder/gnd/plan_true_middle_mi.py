import argparse
import json
from pathlib import Path

from module import Loading_code, read_code

from .partition_backends.toric_plan import binary_dense_plan_diagnostics
from .records import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Dry-run structured true-MI backend planning diagnostics.")
    parser.add_argument("--c-type", default="tor")
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--d", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--e-model", default="dep", choices=["dep", "x", "z"])
    parser.add_argument("--er", type=float, required=True)
    parser.add_argument("--planner", choices=["binary_dense"], default="binary_dense")
    parser.add_argument("--max-safe-width", type=int, default=30)
    parser.add_argument("--max-planner-physical-bits", type=int, default=512)
    parser.add_argument("--output-path", default="")
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main():
    args = parse_args()
    code = Loading_code(read_code(d=args.d, k=args.k, n=args.n, seed=args.seed, c_type=args.c_type))
    if args.planner != "binary_dense":
        raise ValueError(f"Unsupported planner: {args.planner}")

    diagnostics = binary_dense_plan_diagnostics(
        code=code,
        k=args.k,
        er=args.er,
        e_model=args.e_model,
        target_l=args.d,
        max_safe_width=args.max_safe_width,
        max_planner_physical_bits=args.max_planner_physical_bits,
    ).to_dict()
    payload = {
        "record_type": "gnd_structured_true_middle_mi_plan",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "planner": args.planner,
        "code": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
        },
        "diagnostics": diagnostics,
    }

    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output_path:
        write_json_record(resolve_path(args.output_path), payload)


if __name__ == "__main__":
    main()
