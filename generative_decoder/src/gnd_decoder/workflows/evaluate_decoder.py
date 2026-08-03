import argparse
from pathlib import Path

import torch

from gnd_decoder.core import Errormodel, Loading_code, mod2, read_code

from .models import build_model, generate_beta_from_syndrome, get_device, parse_dtype
from .records import utc_timestamp, write_json_record


from gnd_decoder.paths import PROJECT_ROOT, ARTIFACTS_DIR, resolve_path, resolve_output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained GND checkpoint as a decoder.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--er", type=float, default=None)
    parser.add_argument("--e-model", default=None)
    parser.add_argument("--error-seed", type=int, default=51697)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-path", default="")
    return parser.parse_args()



def load_model(checkpoint, device):
    config = checkpoint["model_config"]
    dtype = parse_dtype(config["dtype"])
    model = build_model(config=config, n_bits=config["n_bits"], device=device, dtype=dtype)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, dtype


def main():
    args = parse_args()
    device = get_device(args.device)
    checkpoint_path = resolve_path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model, dtype = load_model(checkpoint, device)
    meta = checkpoint["dataset_meta"]
    er = meta["er"] if args.er is None else args.er
    e_model = meta["e_model"] if args.e_model is None else args.e_model

    info = read_code(d=meta["d"], k=meta["k"], n=meta["n"], seed=meta["seed"], c_type=meta["c_type"])
    code = Loading_code(info)
    helper = mod2(device=device, dtype=dtype)
    error_model = Errormodel(er, e_model=e_model)
    errors = error_model.generate_error(code.n, m=args.trials, seed=args.error_seed)
    syndrome = helper.commute(errors, code.g_stabilizer).to(device=device, dtype=dtype)
    pure_error = error_model.pure(code.pure_es, syndrome, device=device, dtype=dtype)

    beta_hat = generate_beta_from_syndrome(
        model=model,
        n_type=checkpoint["model_config"]["n_type"],
        syndrome=syndrome,
        dtype=dtype,
        k=meta["k"],
    )
    correction_logical = helper.confs_to_opt(confs=beta_hat, gs=code.logical_opt)
    recover = helper.opt_prod(pure_error, correction_logical)
    check = helper.opt_prod(recover, errors)
    commute = helper.commute(check, code.logical_opt)
    fail = torch.count_nonzero(commute.sum(1))
    logical_error_rate = (fail / args.trials).item()

    result = {
        "record_type": "gnd_decoder_evaluation",
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "checkpoint": str(checkpoint_path),
        "code": meta,
        "trials": args.trials,
        "er": er,
        "e_model": e_model,
        "error_seed": args.error_seed,
        "device": str(device),
        "logical_failures": int(fail.item()),
        "logical_error_rate": logical_error_rate,
    }
    print(f"logical_failures={result['logical_failures']} trials={args.trials}")
    print(f"logical_error_rate={logical_error_rate:.8f}")

    if args.output_path:
        output_path = resolve_path(args.output_path)
        write_json_record(output_path, result)
        print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
