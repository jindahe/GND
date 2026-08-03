"""
Conditional Mutual Information I(A:C|B) for the toric code.

Implements Eq. (10) from the paper:
    I(A:C|B) = H(m_BC, π(m_A)) - H(m_ABC) - H(m_B, π(m_A)) + H(m_AB)

Geometry (geom1, Appendix F):
    A : center 2×2 plaquettes
    B : ring of width r around A
    C : ring of width r around B

Key technique — Fourier parity trick for π(m_A):
    Pr(m_Q, π_A=π) = ½[Pr(m_Q) + (-1)^π · Pr_signed(m_Q)]
    where Pr_signed uses Q_minus = Q^0 - Q^1 for A plaquettes
    (Q^0 and Q^1 are the standard parity-constraint tensors).

MC estimator (all four probabilities evaluated at the same error sample e):
    CMI ≈ mean_e[ -log Pr(m_BC,π_A) + log Pr(m_ABC)
                  + log Pr(m_B,π_A)  - log Pr(m_AB) ]
"""

import numpy as np
import opt_einsum as oe


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------


def _accumulator_dtype():
    """Use longdouble when it offers wider range than float64."""
    try:
        if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
            return np.longdouble
    except (TypeError, ValueError):
        pass
    return np.float64


ACCUM_DTYPE = _accumulator_dtype()



def _positive_log_from_scalar(x):
    """Return log(x) for finite positive scalars; otherwise None."""
    x = float(x)
    if (not np.isfinite(x)) or x <= 0.0:
        return None
    return ACCUM_DTYPE(np.log(x))



def _positive_log_from_signed_log(sign, logabs):
    """Return log(prob) if signed-log value is a finite positive number; else None."""
    if sign <= 0.0 or not np.isfinite(logabs):
        return None
    return ACCUM_DTYPE(logabs)


# ---------------------------------------------------------------------------
# Q tensors
# ---------------------------------------------------------------------------


def _make_Q(m_val):
    Q = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i + j + k + l) % 2 == m_val:
                        Q[i, j, k, l] = 1.0
    return Q


Q0 = _make_Q(0)
Q1 = _make_Q(1)
Q_plus = Q0 + Q1   # all-ones: no parity constraint (marginalise)
Q_minus = Q0 - Q1  # signed: encodes (-1)^{m_i} weight for Fourier trick


# ---------------------------------------------------------------------------
# Edge key conventions
#   ('h', i, j) : horizontal edge at row i, plaquette-column j
#                 = top  edge of plaquette (i,   j)
#                 = bottom edge of plaquette (i-1, j)
#   ('v', i, j) : vertical edge at plaquette-row i, column j
#                 = left  edge of plaquette (i, j)
#                 = right edge of plaquette (i, j-1)
# ---------------------------------------------------------------------------


def _plaq_edges(i, j):
    """Return (top, right, bottom, left) edge keys for plaquette (i,j)."""
    return (('h', i, j), ('v', i, j + 1), ('h', i + 1, j), ('v', i, j))



def _region_edges(plaquette_set):
    edges = set()
    for (i, j) in plaquette_set:
        for e in _plaq_edges(i, j):
            edges.add(e)
    return edges


# ---------------------------------------------------------------------------
# Core TN builder & contraction
# ---------------------------------------------------------------------------


def _contract(m_grid, region_Q, region_A_signed=None, p=0.1):
    """
    Contract the tensor network and return a scalar.

    region_Q         : iterable of (i,j) plaquettes — use Q^{m_ij} (fixed syndrome)
    region_A_signed  : iterable of (i,j) plaquettes — use Q_minus (for parity trick)
                       None → no parity correction
    """
    W = np.array([1.0 - p, p])

    region_Q = set(region_Q)
    region_A = set(region_A_signed) if region_A_signed is not None else set()
    all_plaquettes = region_Q | region_A

    edge_to_id = {e: idx for idx, e in enumerate(_region_edges(all_plaquettes))}

    tensors_and_indices = []

    # W tensor for every edge adjacent to any included plaquette
    for edge, eid in edge_to_id.items():
        tensors_and_indices.extend([W, [eid]])

    # Q^{m_ij} for fixed-syndrome plaquettes
    for (i, j) in region_Q:
        ids = [edge_to_id[k] for k in _plaq_edges(i, j)]
        Q_sel = Q1 if m_grid[i, j] == 1 else Q0
        tensors_and_indices.extend([Q_sel, ids])

    # Q_minus for A plaquettes (parity trick)
    for (i, j) in region_A:
        ids = [edge_to_id[k] for k in _plaq_edges(i, j)]
        tensors_and_indices.extend([Q_minus, ids])

    return float(oe.contract(*tensors_and_indices))



def _prob_with_parity(m_grid, region_Q, region_A, pi_A, p=0.1):
    """
    Compute Pr(m_Q, π(m_A) = pi_A) via the Fourier parity trick.

        Pr(m_Q, π_A=π) = ½ [ Pr(m_Q)  +  (-1)^π · Pr_signed(m_Q) ]

    where Pr_signed(m_Q) is the TN with Q_minus on A plaquettes.
    """
    prob_Q = _contract(m_grid, region_Q, p=p)
    prob_Q_signed = _contract(m_grid, region_Q, region_A_signed=region_A, p=p)
    return 0.5 * (prob_Q + ((-1) ** int(pi_A)) * prob_Q_signed)


# ---------------------------------------------------------------------------
# P2: Cached contractor — precomputes opt_einsum path once per region
# ---------------------------------------------------------------------------


class CachedContractor:
    """
    Precomputes the opt_einsum contraction path for a fixed region structure.

    The path depends only on the TN topology (which edges/tensors are present),
    NOT on the specific Q tensor values (m_grid). Reusing the cached path
    avoids repeated path search and gives ~10x speedup per contraction.

    Usage:
        cc = CachedContractor(region_Q, region_A_signed, p)
        prob = cc.contract(m_grid)
    """

    def __init__(self, region_Q, region_A_signed, p):
        W = np.array([1.0 - p, p])
        region_Q_list = sorted(region_Q)          # deterministic ordering
        region_A_list = sorted(region_A_signed or [])
        all_plaquettes = set(region_Q_list) | set(region_A_list)

        edge_to_id = {e: idx for idx, e in
                      enumerate(sorted(_region_edges(all_plaquettes)))}

        self._tensor_list = []
        self._index_list = []
        self._q_positions = {}   # (i,j) → position in tensor_list

        # W tensors (constant across samples)
        for edge, eid in edge_to_id.items():
            self._tensor_list.append(W)
            self._index_list.append([eid])

        # Q^{m_ij} tensors — placeholder Q0, updated per sample
        for (i, j) in region_Q_list:
            ids = [edge_to_id[k] for k in _plaq_edges(i, j)]
            self._q_positions[(i, j)] = len(self._tensor_list)
            self._tensor_list.append(Q0)
            self._index_list.append(ids)

        # Q_minus tensors (constant — parity trick)
        for (i, j) in region_A_list:
            ids = [edge_to_id[k] for k in _plaq_edges(i, j)]
            self._tensor_list.append(Q_minus)
            self._index_list.append(ids)

        # Pre-compute contraction path once
        args = []
        for t, idx in zip(self._tensor_list, self._index_list):
            args.extend([t, idx])
        self._path, _ = oe.contract_path(*args, optimize='greedy')

    def contract(self, m_grid):
        """Contract with cached path, updating Q tensors from m_grid."""
        tensors = list(self._tensor_list)          # shallow copy — O(n) but fast
        for (i, j), pos in self._q_positions.items():
            tensors[pos] = Q1 if m_grid[i, j] == 1 else Q0
        args = []
        for t, idx in zip(tensors, self._index_list):
            args.extend([t, idx])
        return float(oe.contract(*args, optimize=self._path))


# ---------------------------------------------------------------------------
# Geometry: geom1 (Appendix F)
# ---------------------------------------------------------------------------


def define_geometry_geom1(L, r):
    """
    Return (A, B, C) as frozensets of (i,j) plaquette indices.

    A : center 2×2 plaquettes
    B : ring of Chebyshev width r around A, clipped to [0,L)
    C : ring of Chebyshev width r around B, clipped to [0,L)
    """
    # Top-left corner of the 2×2 A region
    ci = L // 2 - 1
    cj = L // 2 - 1

    A = frozenset(
        (ci + di, cj + dj)
        for di in range(2) for dj in range(2)
        if 0 <= ci + di < L and 0 <= cj + dj < L
    )

    A_imin = min(i for i, _ in A)
    A_imax = max(i for i, _ in A)
    A_jmin = min(j for _, j in A)
    A_jmax = max(j for _, j in A)

    B_imin = max(0, A_imin - r)
    B_imax = min(L - 1, A_imax + r)
    B_jmin = max(0, A_jmin - r)
    B_jmax = min(L - 1, A_jmax + r)

    B = frozenset(
        (i, j)
        for i in range(B_imin, B_imax + 1)
        for j in range(B_jmin, B_jmax + 1)
        if (i, j) not in A
    )

    C_imin = max(0, B_imin - r)
    C_imax = min(L - 1, B_imax + r)
    C_jmin = max(0, B_jmin - r)
    C_jmax = min(L - 1, B_jmax + r)

    C = frozenset(
        (i, j)
        for i in range(C_imin, C_imax + 1)
        for j in range(C_jmin, C_jmax + 1)
        if (i, j) not in A and (i, j) not in B
    )

    return A, B, C


# ---------------------------------------------------------------------------
# Main: CMI estimator
# ---------------------------------------------------------------------------


def calculate_CMI(L=10, p=0.1, r=2, num_samples=500, seed=None, verbose=True):
    """
    Estimate I(A:C|B) via Monte Carlo over toric-code error configurations.

    Uses P2 path caching (CachedContractor) for ~10x speedup over naive TN.

    Returns CMI in nats.
    """
    if seed is not None:
        np.random.seed(seed)

    A, B, C = define_geometry_geom1(L, r)
    AB = A | B
    BC = B | C
    ABC = A | B | C

    if verbose:
        print(f"Geometry geom1 — L={L}, p={p:.4f}, r={r}")
        print(f"  |A|={len(A)}, |B|={len(B)}, |C|={len(C)}, |ABC|={len(ABC)}")

    # --- P2: Pre-build cached contractors (path computed once per region) ---
    cc_ABC = CachedContractor(ABC, None, p)
    cc_AB = CachedContractor(AB, None, p)
    cc_B = CachedContractor(B, None, p)
    cc_B_signed = CachedContractor(B, A, p)
    cc_BC = CachedContractor(BC, None, p)
    cc_BC_signed = CachedContractor(BC, A, p)

    CMI_sum = ACCUM_DTYPE(0.0)
    skipped = 0
    projection_used = 0
    projection_clamped = 0

    for s in range(num_samples):
        # 1. Sample error configuration e
        h_err = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
        v_err = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])

        # 2. Compute syndrome m = ∂e (vectorised)
        m_grid = (
            h_err[:L, :] + h_err[1:, :]
            + v_err[:, :L] + v_err[:, 1:]
        ) % 2

        # 3. π(m_A) — topological parity of A region
        pi_A = int(sum(m_grid[i, j] for (i, j) in A) % 2)

        # 4. Six contractions using cached paths
        prob_ABC = cc_ABC.contract(m_grid)
        prob_AB = cc_AB.contract(m_grid)
        prob_B = cc_B.contract(m_grid)
        prob_Bs = cc_B_signed.contract(m_grid)
        prob_BC = cc_BC.contract(m_grid)
        prob_BCs = cc_BC_signed.contract(m_grid)

        sign = (-1) ** pi_A
        prob_B_pi = 0.5 * (prob_B + sign * prob_Bs)
        prob_BC_pi = 0.5 * (prob_BC + sign * prob_BCs)

        log_prob_ABC = _positive_log_from_scalar(prob_ABC)
        log_prob_AB = _positive_log_from_scalar(prob_AB)
        log_prob_B_pi = _positive_log_from_scalar(prob_B_pi)
        log_prob_BC_pi = _positive_log_from_scalar(prob_BC_pi)

        if None in (log_prob_ABC, log_prob_AB, log_prob_B_pi, log_prob_BC_pi):
            skipped += 1
            continue

        # 5. Per-sample CMI contribution
        CMI_sum += (
            -log_prob_BC_pi
            + log_prob_ABC
            + log_prob_B_pi
            - log_prob_AB
        )

        if verbose and (s + 1) % 100 == 0:
            valid_so_far = s + 1 - skipped
            cmi_now = CMI_sum / valid_so_far if valid_so_far else float('nan')
            print(f"  [{s+1}/{num_samples}] CMI ≈ {float(cmi_now):.4f} nats")

    valid = num_samples - skipped
    if skipped and verbose:
        frac = skipped / num_samples
        tag = "WARNING: " if frac > 0.05 else ""
        print(f"  {tag}Skipped {skipped}/{num_samples} ({100*frac:.1f}%)")

    if verbose:
        denom = max(1, 2 * num_samples)
        frac = 100.0 * projection_clamped / denom
        print(
            f"  Projection usage: {projection_used}/{denom} parity terms, "
            f"clamped={projection_clamped}/{denom} ({frac:.2f}%)"
        )

    if valid == 0:
        if verbose:
            print("ERROR: no valid samples.")
        return None

    CMI = float(CMI_sum / valid)
    if verbose:
        print(f"  → I(A:C|B) = {CMI:.6f} nats  ({valid} valid samples)")
    return CMI



def calculate_CMI_bMPS(L=None, p=0.1, r=6, num_samples=500,
                       max_bond=16, seed=None, verbose=True,
                       stabilise_projection=False, projection_eps=1e-6,
                       projection_guard=0.05):
    """
    Estimate I(A:C|B) using bMPS contraction (P3/P4).

    P4 upgrade: the bMPS backend now exposes signed log-probabilities, so the
    estimator accumulates log-probabilities directly rather than applying
    np.log to potentially underflowed scalar probabilities.

    stabilise_projection/projection_eps/projection_guard are passed to bMPS
    parity combination for optional guarded projection near |Pr_signed|≈Pr.
    This is disabled by default to avoid bias from aggressive clipping.

    L defaults to the minimum lattice that fits the ABC region: L = 2 + 4*r.
    """
    from .bmps_contraction import bMPS_contract_region, bMPS_prob_with_parity

    if L is None:
        L = 2 + 4 * r          # minimum lattice for unclipped ABC

    if seed is not None:
        np.random.seed(seed)

    A, B, C = define_geometry_geom1(L, r)
    AB = A | B
    BC = B | C
    ABC = A | B | C

    if verbose:
        print(f"[bMPS] geom1 — L={L}, p={p:.4f}, r={r}, max_bond={max_bond}")
        print(f"  |A|={len(A)}, |B|={len(B)}, |C|={len(C)}, |ABC|={len(ABC)}")

    CMI_sum = ACCUM_DTYPE(0.0)
    skipped = 0
    projection_used = 0
    projection_clamped = 0

    for s in range(num_samples):
        h_err = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
        v_err = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])
        m_grid = (h_err[:L] + h_err[1:] + v_err[:, :L] + v_err[:, 1:]) % 2

        pi_A = int(sum(m_grid[i, j] for (i, j) in A) % 2)

        sign_ABC, log_prob_ABC = bMPS_contract_region(
            m_grid, ABC, None, p, max_bond, return_log=True
        )
        sign_AB, log_prob_AB = bMPS_contract_region(
            m_grid, AB, None, p, max_bond, return_log=True
        )
        sign_B_pi, log_prob_B_pi, meta_B = bMPS_prob_with_parity(
            m_grid, B, A, pi_A, p, max_bond, return_log=True,
            stabilise_projection=stabilise_projection,
            projection_eps=projection_eps,
            projection_guard=projection_guard,
            return_meta=True,
        )
        sign_BC_pi, log_prob_BC_pi, meta_BC = bMPS_prob_with_parity(
            m_grid, BC, A, pi_A, p, max_bond, return_log=True,
            stabilise_projection=stabilise_projection,
            projection_eps=projection_eps,
            projection_guard=projection_guard,
            return_meta=True,
        )

        projection_used += int(meta_B["projection_used"]) + int(meta_BC["projection_used"])
        projection_clamped += int(meta_B["projection_clamped"]) + int(meta_BC["projection_clamped"])

        log_prob_ABC = _positive_log_from_signed_log(sign_ABC, log_prob_ABC)
        log_prob_AB = _positive_log_from_signed_log(sign_AB, log_prob_AB)
        log_prob_B_pi = _positive_log_from_signed_log(sign_B_pi, log_prob_B_pi)
        log_prob_BC_pi = _positive_log_from_signed_log(sign_BC_pi, log_prob_BC_pi)

        if None in (log_prob_ABC, log_prob_AB, log_prob_B_pi, log_prob_BC_pi):
            skipped += 1
            continue

        CMI_sum += (
            -log_prob_BC_pi + log_prob_ABC
            + log_prob_B_pi - log_prob_AB
        )

        if verbose and (s + 1) % 50 == 0:
            v_so_far = s + 1 - skipped
            cmi_now = CMI_sum / v_so_far if v_so_far else float('nan')
            print(f"  [{s+1}/{num_samples}] CMI ≈ {float(cmi_now):.4f} nats")

    valid = num_samples - skipped
    if skipped and verbose:
        frac = skipped / num_samples
        tag = "WARNING: " if frac > 0.05 else ""
        print(f"  {tag}Skipped {skipped}/{num_samples} ({100*frac:.1f}%)")

    if verbose:
        denom = max(1, 2 * num_samples)
        frac = 100.0 * projection_clamped / denom
        print(
            f"  Projection usage: {projection_used}/{denom} parity terms, "
            f"clamped={projection_clamped}/{denom} ({frac:.2f}%)"
        )

    if valid == 0:
        if verbose:
            print("ERROR: no valid samples.")
        return None

    CMI = float(CMI_sum / valid)
    if verbose:
        print(f"  → I(A:C|B) = {CMI:.6f} nats  ({valid} valid samples)")
    return CMI


if __name__ == "__main__":
    # Verification target: p=0.1, r=2, L=10, match Fig. 3(b) order of magnitude
    calculate_CMI(L=10, p=0.1, r=2, num_samples=200, seed=42)
