import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .models import build_model, count_parameters, get_device, model_log_prob, parse_dtype
from .records import utc_timestamp, write_json_record


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a GND autoregressive density model.")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--save-dir", default="net/gnd/models")
    parser.add_argument("--record-dir", default="")
    parser.add_argument("--train-seed", type=int, default=0)
    parser.add_argument("--n-type", choices=["made", "nade", "trade"], default="made")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=0.0)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--depth", type=int, default=0)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--made-activation", choices=["tanh", "relu", "sigmoid"], default="tanh")
    parser.add_argument("--made-residual", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--d-ff", type=int, default=256)
    parser.add_argument("--n-layers", type=int, default=1)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def evaluate_nll(model, n_type, loader, device, dtype):
    model.eval()
    total_nll = 0.0
    total_examples = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=dtype)
            nll = -model_log_prob(model, n_type, batch)
            total_nll += nll.sum().item()
            total_examples += batch.size(0)
    return total_nll / total_examples


def checkpoint_name(data, args):
    meta = data["meta"]
    return (
        f"{args.n_type}_{data['target']}_{meta['c_type']}_n{meta['n']}_d{meta['d']}_k{meta['k']}"
        f"_seed{meta['seed']}_er{meta['er']}_{meta['e_model']}_tseed{args.train_seed}.pt"
    )


def model_config(args, n_bits):
    return {
        "n_type": args.n_type,
        "n_bits": n_bits,
        "depth": args.depth,
        "width": args.width,
        "made_activation": args.made_activation,
        "made_residual": args.made_residual,
        "hidden_dim": args.hidden_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "d_ff": args.d_ff,
        "n_layers": args.n_layers,
        "dtype": args.dtype,
    }


def main():
    args = parse_args()
    started_at = utc_timestamp()
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)

    dtype = parse_dtype(args.dtype)
    device = get_device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.train_seed)

    dataset_path = resolve_path(args.dataset_path)
    data = torch.load(dataset_path, map_location="cpu")
    train = data["train"].to(dtype=dtype)
    val = data["val"].to(dtype=dtype)
    test = data["test"].to(dtype=dtype)
    n_bits = train.size(1)

    train_loader = DataLoader(TensorDataset(train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(val), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(TensorDataset(test), batch_size=args.batch, shuffle=False)

    config = model_config(args, n_bits)
    model = build_model(config=config, n_bits=n_bits, device=device, dtype=dtype)
    parameter_count = count_parameters(model)
    optimizer_cls = torch.optim.AdamW if args.weight_decay > 0 else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_nll": [],
        "val_nll": [],
        "test_nll": None,
        "best_val_nll": None,
        "best_epoch": None,
        "epochs_trained": 0,
    }
    best_state = None
    epochs_without_improvement = 0

    print(f"dataset: {dataset_path}")
    print(f"device: {device}")
    print(f"model: {args.n_type} n_bits={n_bits} params={parameter_count}")

    for epoch in range(args.epoch):
        model.train()
        total_nll = 0.0
        total_examples = 0
        for (batch,) in train_loader:
            batch = batch.to(device=device, dtype=dtype)
            nll = -model_log_prob(model, args.n_type, batch)
            loss = nll.mean()
            optimizer.zero_grad()
            loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            optimizer.step()
            total_nll += nll.sum().item()
            total_examples += batch.size(0)

        train_nll = total_nll / total_examples
        val_nll = evaluate_nll(model, args.n_type, val_loader, device, dtype)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["epochs_trained"] = epoch + 1

        improved = history["best_val_nll"] is None or val_nll < history["best_val_nll"] - args.early_stop_min_delta
        if improved:
            history["best_val_nll"] = val_nll
            history["best_epoch"] = epoch + 1
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 0 or (epoch + 1) % args.log_every == 0 or epoch + 1 == args.epoch:
            print(f"epoch={epoch + 1} train_nll={train_nll:.6f} val_nll={val_nll:.6f}")

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(f"early_stop: epoch={epoch + 1} best_epoch={history['best_epoch']}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
        history["test_nll"] = evaluate_nll(model, args.n_type, test_loader, device, dtype)

    save_dir = resolve_path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = save_dir / checkpoint_name(data, args)
    record_dir = resolve_path(args.record_dir) if args.record_dir else save_dir / "records"
    record_path = record_dir / f"{checkpoint_path.stem}.json"

    config["parameter_count"] = parameter_count
    checkpoint = {
        "record_type": "gnd_checkpoint",
        "schema_version": 1,
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": config,
        "dataset_meta": data["meta"],
        "dataset_target": data["target"],
        "layout": data["layout"],
        "dataset_path": str(dataset_path),
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    print(f"saved: {checkpoint_path}")

    record = {
        "record_type": "gnd_training",
        "schema_version": 1,
        "started_at_utc": started_at,
        "finished_at_utc": utc_timestamp(),
        "script": "gnd/train.py",
        "dataset_path": str(dataset_path),
        "model_config": config,
        "training_config": {
            "train_seed": args.train_seed,
            "device": str(device),
            "epoch": args.epoch,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "grad_clip_norm": args.grad_clip_norm,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
        "metrics": history,
        "artifacts": {"checkpoint_path": str(checkpoint_path)},
    }
    write_json_record(record_path, record)
    print(f"record: {record_path}")


if __name__ == "__main__":
    main()
