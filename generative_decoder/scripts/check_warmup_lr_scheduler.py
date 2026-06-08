#!/usr/bin/env python3
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECODING_DIR = PROJECT_ROOT / "decoding"
if str(DECODING_DIR) not in sys.path:
    sys.path.insert(0, str(DECODING_DIR))

from train_mi_syndrome import apply_warmup_lr, current_lr, set_optimizer_lr  # noqa: E402


def assert_close(actual, expected):
    if abs(actual - expected) > 1e-12:
        raise AssertionError(f"expected lr={expected}, got lr={actual}")


def main():
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=0,
        min_lr=0.001,
    )

    set_optimizer_lr(optimizer, 0.0)
    assert_close(apply_warmup_lr(optimizer, 1, warmup_steps=2, base_lr=0.1), 0.05)
    assert_close(apply_warmup_lr(optimizer, 2, warmup_steps=2, base_lr=0.1), 0.1)

    scheduler.step(1.0)
    scheduler.step(1.1)
    assert_close(current_lr(optimizer), 0.05)

    observed = apply_warmup_lr(optimizer, 3, warmup_steps=2, base_lr=0.1)
    assert_close(observed, 0.05)
    assert_close(current_lr(optimizer), 0.05)
    print("WARMUP_LR_SCHEDULER_CHECK_PASSED")


if __name__ == "__main__":
    main()
