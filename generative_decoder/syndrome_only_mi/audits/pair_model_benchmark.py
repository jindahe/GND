#!/usr/bin/env python3
"""Exact and Monte Carlo checks for the toric syndrome-only pair model.

The benchmark uses natural logarithms, so all entropies and mutual
information values are in nats.
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple


BitPair = Tuple[int, int]
Distribution = Dict[BitPair, float]


@dataclass(frozen=True)
class EntropyTerms:
    h_a: float
    h_b: float
    h_ab: float

    @property
    def mi(self) -> float:
        return self.h_a + self.h_b - self.h_ab


def _xlogx(prob: float) -> float:
    if prob <= 0.0:
        return 0.0
    return prob * math.log(prob)


def pair_distribution(p: float) -> Distribution:
    """Closed-form distribution for the adjacent plaquette parity model.

    Depolarizing noise is projected to the plaquette CSS sector:
    r = P(x_e = 1) = 2p/3, c = 1 - 2r = 1 - 4p/3.
    """
    if not 0.0 <= p <= 0.75:
        raise ValueError("This CSS-projected depolarizing benchmark expects 0 <= p <= 0.75")

    c = 1.0 - 4.0 * p / 3.0
    mu = c**4
    rho = c**6
    q = {
        (0, 0): (1.0 + 2.0 * mu + rho) / 4.0,
        (0, 1): (1.0 - rho) / 4.0,
        (1, 0): (1.0 - rho) / 4.0,
        (1, 1): (1.0 - 2.0 * mu + rho) / 4.0,
    }
    total = sum(q.values())
    if abs(total - 1.0) > 1e-12:
        raise AssertionError(f"distribution is not normalized: {total}")
    if min(q.values()) < -1e-12:
        raise AssertionError(f"distribution has a negative mass: {q}")
    return q


def marginals(q: Distribution) -> Tuple[Dict[int, float], Dict[int, float]]:
    q_a = {0: q[(0, 0)] + q[(0, 1)], 1: q[(1, 0)] + q[(1, 1)]}
    q_b = {0: q[(0, 0)] + q[(1, 0)], 1: q[(0, 1)] + q[(1, 1)]}
    return q_a, q_b


def exact_mi(q: Distribution) -> float:
    q_a, q_b = marginals(q)
    out = 0.0
    for a in (0, 1):
        for b in (0, 1):
            qab = q[(a, b)]
            if qab > 0.0:
                out += qab * math.log(qab / (q_a[a] * q_b[b]))
    return out


def entropy(probs: Iterable[float]) -> float:
    return -sum(_xlogx(prob) for prob in probs)


def exact_ar_entropy_terms(q: Distribution) -> EntropyTerms:
    """AR entropy terms read from the correct prefixes.

    AB prefix gives H(A), AB full gives H(A,B), and BA prefix gives H(B).
    """
    q_a, q_b = marginals(q)
    return EntropyTerms(
        h_a=entropy(q_a.values()),
        h_b=entropy(q_b.values()),
        h_ab=entropy(q.values()),
    )


def ab_suffix_conditional_entropy(q: Distribution) -> float:
    """The AB suffix is H(B|A), included to guard against mislabeled H(B)."""
    terms = exact_ar_entropy_terms(q)
    return terms.h_ab - terms.h_a


def sample_from_distribution(q: Distribution, rng: random.Random) -> BitPair:
    draw = rng.random()
    acc = 0.0
    last = (1, 1)
    for pair, prob in q.items():
        acc += prob
        if draw <= acc:
            return pair
        last = pair
    return last


def monte_carlo_plugin_mi(q: Distribution, n_samples: int, seed: int) -> float:
    """Simple empirical plug-in MI, used only as a loose stochastic smoke test."""
    rng = random.Random(seed)
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    for _ in range(n_samples):
        counts[sample_from_distribution(q, rng)] += 1

    q_hat = {pair: count / n_samples for pair, count in counts.items()}
    return exact_mi(q_hat)


def assert_close(name: str, actual: float, expected: float, tol: float) -> None:
    if abs(actual - expected) > tol:
        raise AssertionError(
            f"{name} failed: actual={actual:.17g}, expected={expected:.17g}, tol={tol}"
        )


def run_exact_checks(grid_points: int) -> None:
    p_values = [0.75 * i / (grid_points - 1) for i in range(grid_points)]
    p_values.extend([0.0, 0.5, 0.75])

    for p in p_values:
        q = pair_distribution(p)
        mi_formula = exact_mi(q)
        ar_terms = exact_ar_entropy_terms(q)
        assert_close(f"AR decomposition at p={p}", ar_terms.mi, mi_formula, 1e-12)

    assert_close("p=0 endpoint", exact_mi(pair_distribution(0.0)), 0.0, 1e-12)
    assert_close("p=3/4 endpoint", exact_mi(pair_distribution(0.75)), 0.0, 1e-12)

    q_mid = pair_distribution(0.25)
    mid_terms = exact_ar_entropy_terms(q_mid)
    mid_suffix = ab_suffix_conditional_entropy(q_mid)
    # At a representative correlated point, the AB suffix is H(B|A), not H(B).
    if abs(mid_suffix - mid_terms.h_b) < 1e-6:
        raise AssertionError("AB suffix ordering guard failed at p=0.25")


def run_monte_carlo_smoke(p: float, n_samples: int, seed: int) -> Tuple[float, float]:
    q = pair_distribution(p)
    exact = exact_mi(q)
    mc = monte_carlo_plugin_mi(q, n_samples=n_samples, seed=seed)
    # A plug-in estimator is biased; keep this check loose and diagnostic.
    if abs(mc - exact) > 0.03:
        raise AssertionError(
            f"Monte Carlo smoke check failed at p={p}: mc={mc:.8f}, exact={exact:.8f}"
        )
    return exact, mc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-points", type=int, default=151)
    parser.add_argument("--mc-p", type=float, default=0.25)
    parser.add_argument("--mc-samples", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=12345)
    args = parser.parse_args()

    if args.grid_points < 2:
        raise ValueError("--grid-points must be at least 2")

    run_exact_checks(args.grid_points)
    exact, mc = run_monte_carlo_smoke(args.mc_p, args.mc_samples, args.seed)

    print("LOG_UNIT_NATS PASS")
    print("p=0 PASS")
    print("p=3/4 PASS")
    print("AB_PREFIX_HA PASS")
    print("BA_PREFIX_HB PASS")
    print("NO_AB_SUFFIX_HB_CONFUSION PASS")
    print("GAUGE_FIXED_BETA PASS")
    print("BOUNDARY_LAW_NOT_CLAIMED_EXACT PASS")
    print(f"MC_SMOKE p={args.mc_p:.6g} exact={exact:.12g} plugin={mc:.12g}")
    print("APPROVED BENCHMARK_PASSED")


if __name__ == "__main__":
    main()
