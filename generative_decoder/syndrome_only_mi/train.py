import random
import sys
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .args import args

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCHIVE_ROOT = PROJECT_ROOT / "syndrome_only_mi"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from module import MADE, NADE, TraDE_binary  # noqa: E402
from .record_utils import utc_timestamp, write_json_record  # noqa: E402


def get_dtype():
    if args.dtype == "float32":
        return torch.float32
    if args.dtype == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {args.dtype}")


def get_device():
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {args.device} was requested but torch.cuda.is_available() is False in this environment"
        )
    return device


def training_seed_suffix():
    return f"_tseed{args.train_seed}" if args.train_seed != 0 else ""


def resolve_dataset_path():
    if args.dataset_path:
        path = Path(args.dataset_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path

    dataset_dir = Path(args.dataset_dir) if args.dataset_dir else ARCHIVE_ROOT / "net" / "syndrome_data"
    if not dataset_dir.is_absolute():
        dataset_dir = PROJECT_ROOT / dataset_dir

    suffix = f"{args.partition_order}_{args.partition_axis}{args.cut if args.cut is not None else 'mid'}"
    filename = (
        f"{args.c_type}_n{args.n}_d{args.d}_k{args.k}_seed{args.seed}"
        f"_er{args.er}_{args.e_model}_{suffix}.pt"
    )
    return dataset_dir / filename


def resolve_checkpoint_path():
    if args.save_dir:
        output_dir = Path(args.save_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = ARCHIVE_ROOT / "net" / "syndrome_models"

    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"{args.partition_order}_{args.partition_axis}{args.cut if args.cut is not None else 'mid'}"
    filename = (
        f"{args.n_type}_{args.c_type}_n{args.n}_d{args.d}_k{args.k}_seed{args.seed}"
        f"_er{args.er}_{args.e_model}{training_seed_suffix()}_{suffix}.pt"
    )
    return output_dir / filename


def resolve_record_path(checkpoint_path):
    if args.record_dir:
        output_dir = Path(args.record_dir)
        if not output_dir.is_absolute():
            output_dir = PROJECT_ROOT / output_dir
    else:
        output_dir = checkpoint_path.parent / "records"

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{checkpoint_path.stem}.json"


def build_checkpoint_payload(model, build_meta, n_bits, model_param_count, dataset_path, data, history):
    return {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "model_config": {
            "n_type": args.n_type,
            "n_bits": n_bits,
            "depth": args.depth,
            "width": args.width,
            "effective_width": build_meta.get("effective_width"),
            "made_activation": args.made_activation,
            "made_residual": args.made_residual,
            "parameter_count": model_param_count,
            "hidden_dim": args.hidden_dim,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "dtype": args.dtype,
        },
        "training_config": {
            "device": str(torch.device(args.device)),
            "train_seed": args.train_seed,
            "epoch": args.epoch,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW" if args.weight_decay > 0 else "Adam",
            "grad_clip_norm": args.grad_clip_norm,
            "warmup_steps": args.warmup_steps,
            "divergence_nll_threshold": args.divergence_nll_threshold,
            "max_train_steps": args.max_train_steps,
            "lr_decay_factor": args.lr_decay_factor,
            "lr_decay_patience": args.lr_decay_patience,
            "min_lr": args.min_lr,
            "log_every": args.log_every,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
        "dataset_path": str(dataset_path),
        "dataset_meta": data["meta"],
        "partition": data["partition"],
        "applied_order": data["applied_order"],
        "history": history,
    }


def sequence_summary(values):
    if values is None:
        return None
    encoded = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return {
        "length": len(values),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def partition_summary(partition):
    if partition is None:
        return None
    return {
        "axis": partition["axis"],
        "cut": partition["cut"],
        "len_A": len(partition["idx_A"]),
        "len_B": len(partition["idx_B"]),
        "n_coords": len(partition.get("coords", [])),
        "idx_A": sequence_summary(partition["idx_A"]),
        "idx_B": sequence_summary(partition["idx_B"]),
        "order_AB": sequence_summary(partition["order_AB"]),
        "order_BA": sequence_summary(partition["order_BA"]),
    }


def build_dataset_record(dataset_path, data, train, val, test):
    return {
        "path": str(dataset_path),
        "meta": data["meta"],
        "partition_summary": partition_summary(data["partition"]),
        "applied_order_summary": sequence_summary(data["applied_order"]),
        "shape": {
            "train": list(train.shape),
            "val": list(val.shape),
            "test": list(test.shape),
        },
    }


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def resolve_made_width(n_bits):
    requested_width = args.width
    if args.n_type != "made" or requested_width <= 0 or args.made_max_params <= 0:
        return requested_width

    effective_width = requested_width
    while effective_width > 1:
        probe = MADE(
            n=n_bits,
            depth=args.depth,
            width=effective_width,
            activator=args.made_activation,
            residual=args.made_residual,
        )
        if count_parameters(probe) <= args.made_max_params:
            break
        effective_width -= 1

    if effective_width != requested_width:
        print(
            f"adjust made width: requested={requested_width} effective={effective_width} "
            f"max_params={args.made_max_params}"
        )
    return effective_width


def build_model(n_bits, device, dtype):
    if args.n_type == "made":
        width = resolve_made_width(n_bits=n_bits)
        model = MADE(
            n=n_bits,
            depth=args.depth,
            width=width,
            activator=args.made_activation,
            residual=args.made_residual,
        ).to(device).to(dtype)
        return model, {"effective_width": width}
    if args.n_type == "nade":
        model = NADE(n=n_bits, hidden_dim=args.hidden_dim, device=device, dtype=dtype).to(device).to(dtype)
        return model, {}
    if args.n_type == "trade":
        kwargs_dict = {
            "n": n_bits,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "device": str(device),
            "dropout": 0,
        }
        model = TraDE_binary(**kwargs_dict).to(device).to(dtype)
        return model, {}
    raise ValueError(f"Unsupported model type: {args.n_type}")


def model_log_prob(model, batch):
    if args.n_type == "made":
        return model.log_prob(batch * 2 - 1)
    if args.n_type == "nade":
        return model.forward(batch)
    if args.n_type == "trade":
        return model.log_prob(batch)
    raise ValueError(f"Unsupported model type: {args.n_type}")


def evaluate_nll(model, loader, device, dtype):
    model.eval()
    total_nll = 0.0
    total_examples = 0
    with torch.no_grad():
        for (batch,) in loader:
            batch = batch.to(device=device, dtype=dtype)
            nll = -model_log_prob(model, batch)
            total_nll += nll.sum().item()
            total_examples += batch.size(0)
    return total_nll / total_examples


def current_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def set_optimizer_lr(optimizer, lr):
    for group in optimizer.param_groups:
        group["lr"] = lr


def apply_warmup_lr(optimizer, global_step, warmup_steps=None, base_lr=None):
    warmup_steps = args.warmup_steps if warmup_steps is None else warmup_steps
    base_lr = args.lr if base_lr is None else base_lr
    if warmup_steps <= 0:
        return current_lr(optimizer)
    if global_step > warmup_steps:
        return current_lr(optimizer)
    scale = float(global_step) / float(warmup_steps)
    lr = base_lr * scale
    set_optimizer_lr(optimizer, lr)
    return lr


def compute_grad_norm(parameters):
    grads = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not grads:
        return 0.0
    norms = [torch.linalg.vector_norm(grad, ord=2) for grad in grads]
    return torch.linalg.vector_norm(torch.stack(norms), ord=2).item()


def check_nll_divergence(train_nll, val_nll):
    metrics = {"train_nll": train_nll, "val_nll": val_nll}
    for metric_name, metric_value in metrics.items():
        if not np.isfinite(metric_value):
            return metric_name, metric_value, "non_finite"
        if args.divergence_nll_threshold > 0 and metric_value > args.divergence_nll_threshold:
            return metric_name, metric_value, "threshold"
    return None, None, None


def main():
    started_at = utc_timestamp()
    dtype = get_dtype()
    device = get_device()
    random.seed(args.train_seed)
    np.random.seed(args.train_seed)
    torch.manual_seed(args.train_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.train_seed)
    dataset_path = resolve_dataset_path()
    data = torch.load(dataset_path)

    train = data["train"].to(dtype=dtype)
    val = data["val"].to(dtype=dtype)
    test = data["test"].to(dtype=dtype)
    n_bits = train.size(1)

    train_loader = DataLoader(TensorDataset(train), batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(val), batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(TensorDataset(test), batch_size=args.batch, shuffle=False)

    model, build_meta = build_model(n_bits=n_bits, device=device, dtype=dtype)
    optimizer_cls = torch.optim.AdamW if args.weight_decay > 0 else torch.optim.Adam
    optimizer = optimizer_cls(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.warmup_steps > 0:
        set_optimizer_lr(optimizer, 0.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.lr_decay_factor,
        patience=args.lr_decay_patience,
        min_lr=args.min_lr,
    )
    model_param_count = count_parameters(model)

    history = {
        "train_nll": [],
        "val_nll": [],
        "test_nll": None,
        "best_val_nll": None,
        "best_epoch": None,
        "epochs_trained": 0,
        "optimizer_steps": 0,
        "lr": [],
        "grad_norm_mean": [],
        "grad_norm_max": [],
    }
    divergence_state = {
        "objective_training_failure": False,
        "reason": None,
        "metric": None,
        "value": None,
        "threshold": args.divergence_nll_threshold if args.divergence_nll_threshold > 0 else None,
        "epoch": None,
        "step": None,
    }
    best_state_dict = None
    epochs_without_improvement = 0
    global_step = 0
    stop_training = False
    print(f"dataset: {dataset_path}")
    print(f"device: {device}")
    print(f"train_seed: {args.train_seed}")
    print(f"train/val/test: {tuple(train.shape)} {tuple(val.shape)} {tuple(test.shape)}")
    print(
        f"optimizer: {optimizer_cls.__name__} lr={args.lr} weight_decay={args.weight_decay} "
        f"grad_clip_norm={args.grad_clip_norm} warmup_steps={args.warmup_steps} "
        f"divergence_nll_threshold={args.divergence_nll_threshold} max_train_steps={args.max_train_steps}"
    )
    if args.n_type == "made":
        print(
            f"made_config: depth={args.depth} requested_width={args.width} "
            f"effective_width={build_meta['effective_width']} activation={args.made_activation} "
            f"residual={args.made_residual} params={model_param_count}"
        )

    for epoch in range(args.epoch):
        model.train()
        epoch_nll = 0.0
        seen = 0
        epoch_grad_norms = []

        for (batch,) in train_loader:
            if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                stop_training = True
                break

            batch = batch.to(device=device, dtype=dtype)
            nll = -model_log_prob(model, batch)
            loss = nll.mean()

            if not torch.isfinite(loss):
                divergence_state.update(
                    {
                        "objective_training_failure": True,
                        "reason": "non_finite_loss",
                        "metric": "loss",
                        "value": float(loss.detach().cpu().item()),
                        "epoch": epoch + 1,
                        "step": global_step + 1,
                    }
                )
                stop_training = True
                break

            global_step += 1
            apply_warmup_lr(optimizer, global_step)
            optimizer.zero_grad()
            loss.backward()
            parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
            if args.grad_clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip_norm).item()
            else:
                grad_norm = compute_grad_norm(parameters)
            if not np.isfinite(grad_norm):
                divergence_state.update(
                    {
                        "objective_training_failure": True,
                        "reason": "non_finite_grad_norm",
                        "metric": "grad_norm",
                        "value": grad_norm,
                        "epoch": epoch + 1,
                        "step": global_step,
                    }
                )
                stop_training = True
                break
            optimizer.step()

            epoch_nll += nll.sum().item()
            seen += batch.size(0)
            epoch_grad_norms.append(grad_norm)

        if seen == 0:
            break

        train_nll = epoch_nll / seen
        val_nll = evaluate_nll(model, val_loader, device=device, dtype=dtype)
        if args.warmup_steps <= 0 or global_step >= args.warmup_steps:
            scheduler.step(val_nll)
        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["epochs_trained"] = epoch + 1
        history["optimizer_steps"] = global_step
        history["lr"].append(current_lr(optimizer))
        history["grad_norm_mean"].append(float(np.mean(epoch_grad_norms)) if epoch_grad_norms else None)
        history["grad_norm_max"].append(float(np.max(epoch_grad_norms)) if epoch_grad_norms else None)

        metric_name, metric_value, reason = check_nll_divergence(train_nll, val_nll)
        if reason is not None:
            divergence_state.update(
                {
                    "objective_training_failure": True,
                    "reason": reason,
                    "metric": metric_name,
                    "value": metric_value,
                    "epoch": epoch + 1,
                    "step": global_step,
                }
            )
            stop_training = True

        improved = history["best_val_nll"] is None or val_nll < (
            history["best_val_nll"] - args.early_stop_min_delta
        )
        if improved:
            history["best_val_nll"] = val_nll
            history["best_epoch"] = epoch + 1
            best_state_dict = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if (epoch + 1) % args.log_every == 0 or epoch == 0 or epoch + 1 == args.epoch:
            print(
                f"epoch={epoch + 1} "
                f"train_nll={train_nll:.6f} "
                f"val_nll={val_nll:.6f} "
                f"lr={current_lr(optimizer):.6g} "
                f"grad_norm_mean={history['grad_norm_mean'][-1]:.6f} "
                f"grad_norm_max={history['grad_norm_max'][-1]:.6f} "
                f"steps={global_step}"
            )

        if stop_training:
            if args.max_train_steps > 0 and global_step >= args.max_train_steps and not divergence_state[
                "objective_training_failure"
            ]:
                print(f"max_train_steps: step={global_step}")
            elif divergence_state["objective_training_failure"]:
                print(
                    f"objective_training_failure: reason={divergence_state['reason']} "
                    f"metric={divergence_state['metric']} value={divergence_state['value']} "
                    f"epoch={divergence_state['epoch']} step={divergence_state['step']}"
                )
            break

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            print(
                f"early_stop: epoch={epoch + 1} "
                f"best_epoch={history['best_epoch']} "
                f"patience={args.early_stop_patience}"
            )
            break

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    if best_state_dict is not None:
        history["test_nll"] = evaluate_nll(model, test_loader, device=device, dtype=dtype)
    print(f"best_epoch={history['best_epoch']} best_val_nll={history['best_val_nll']}")
    print(f"test_nll={history['test_nll']}")

    checkpoint_path = resolve_checkpoint_path()
    record_path = resolve_record_path(checkpoint_path)

    if args.save:
        checkpoint = build_checkpoint_payload(
            model=model,
            build_meta=build_meta,
            n_bits=n_bits,
            model_param_count=model_param_count,
            dataset_path=dataset_path,
            data=data,
            history=history,
        )
        torch.save(checkpoint, checkpoint_path)
        print(f"saved: {checkpoint_path}")

    record = {
        "record_type": "syndrome_training",
        "schema_version": 2,
        "started_at_utc": started_at,
        "finished_at_utc": utc_timestamp(),
        "script": "syndrome_only_mi/train.py",
        "code": {
            "c_type": args.c_type,
            "n": args.n,
            "d": args.d,
            "k": args.k,
            "seed": args.seed,
            "e_model": args.e_model,
            "er": args.er,
        },
        "model_config": {
            "n_type": args.n_type,
            "n_bits": n_bits,
            "depth": args.depth,
            "width": args.width,
            "effective_width": build_meta.get("effective_width"),
            "made_activation": args.made_activation,
            "made_residual": args.made_residual,
            "parameter_count": model_param_count,
            "hidden_dim": args.hidden_dim,
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "d_ff": args.d_ff,
            "n_layers": args.n_layers,
            "dtype": args.dtype,
        },
        "training_config": {
            "device": str(device),
            "train_seed": args.train_seed,
            "epoch": args.epoch,
            "batch": args.batch,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": "AdamW" if args.weight_decay > 0 else "Adam",
            "grad_clip_norm": args.grad_clip_norm,
            "warmup_steps": args.warmup_steps,
            "divergence_nll_threshold": args.divergence_nll_threshold,
            "max_train_steps": args.max_train_steps,
            "lr_decay_factor": args.lr_decay_factor,
            "lr_decay_patience": args.lr_decay_patience,
            "min_lr": args.min_lr,
            "log_every": args.log_every,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
        },
        "dataset": build_dataset_record(dataset_path, data, train, val, test),
        "metrics": {
            "best_epoch": history["best_epoch"],
            "best_val_nll": history["best_val_nll"],
            "test_nll": history["test_nll"],
            "epochs_trained": history["epochs_trained"],
            "train_nll_history": history["train_nll"],
            "val_nll_history": history["val_nll"],
            "lr_history": history["lr"],
            "grad_norm_mean_history": history["grad_norm_mean"],
            "grad_norm_max_history": history["grad_norm_max"],
            "optimizer_steps": history["optimizer_steps"],
        },
        "divergence": divergence_state,
        "artifacts": {
            "checkpoint_path": str(checkpoint_path) if args.save else None,
        },
    }
    write_json_record(record_path, record)
    print(f"record: {record_path}")


if __name__ == "__main__":
    main()
