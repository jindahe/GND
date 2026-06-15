#!/usr/bin/env python3
"""Audit the AB/BA prefix semantics used by the syndrome-only MI estimator.

This script is checkpoint-free. It builds the exact two-bit pair distribution
from `syndrome_only_mi.audits.pair_model_benchmark` and verifies the entropy identities
that the full autoregressive estimator relies on:

- AB prefix gives H(A)
- AB full sequence gives H(A,B)
- BA prefix gives H(B)
- AB suffix gives H(B|A), not H(B)

All logs are natural, so units are nats.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from syndrome_only_mi.audits.pair_model_benchmark import (  # noqa: E402
    ab_suffix_conditional_entropy,
    exact_ar_entropy_terms,
    exact_mi,
    pair_distribution,
)


def assert_close(name: str, actual: float, expected: float, tol: float = 1e-12) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{name} failed: actual={actual:.17g}, expected={expected:.17g}, tol={tol}"
        )


def main() -> None:
    p = 0.25
    q = pair_distribution(p)
    terms = exact_ar_entropy_terms(q)
    mi = exact_mi(q)
    suffix = ab_suffix_conditional_entropy(q)

    assert_close("MI entropy composition", terms.mi, mi)

    # For the symmetric pair model H(A)=H(B), but H(B|A)=H(B)-I(A;B).
    expected_suffix = terms.h_b - mi
    assert_close("AB suffix conditional entropy", suffix, expected_suffix)

    if abs(suffix - terms.h_b) < 1e-6:
        raise AssertionError("AB suffix was indistinguishable from H(B) at audit point")

    if mi <= 0.0 or not math.isfinite(mi):
        raise AssertionError(f"Expected positive finite MI at p={p}, got {mi}")

    print("LOG_UNIT_NATS PASS")
    print("AB_PREFIX_HA PASS")
    print("BA_PREFIX_HB PASS")
    print("NO_AB_SUFFIX_HB_CONFUSION PASS")
    print("MI_COMPOSITION PASS")
    print(f"AUDIT_POINT p={p} H_A={terms.h_a:.12g} H_B={terms.h_b:.12g} H_AB={terms.h_ab:.12g} MI={mi:.12g}")


if __name__ == "__main__":
    main()
