"""
quimb-based bMPS backend for 2D tensor-network contraction.

Key improvement over bmps_contraction.py:
  The single left→right SVD sweep in _compress_mps is replaced by
  quimb's MatrixProductState, which canonicalises the MPS (QR left-sweep)
  before truncating (SVD right-sweep).  This two-pass scheme distributes
  truncation error evenly across bonds and gives significantly better
  accuracy at the same bond dimension χ.

All physics (site-tensor construction, Fourier parity trick, MC estimator)
is *unchanged* – only the compression subroutine is swapped.

Drop-in replacements:
    bMPS_contract_region  →  quimb_contract_region
    bMPS_prob_with_parity →  quimb_prob_with_parity
    calculate_CMI_bMPS    →  calculate_CMI_quimb
"""

import numpy as np
import quimb.tensor as qtn

from .cmi_calculation import (
    ACCUM_DTYPE,
    _positive_log_from_signed_log,
    define_geometry_geom1,
)
from .bmps_contraction import (
    build_site_grid,
    _apply_mpo,
    _apply_last_row_log,
    _close_mps_log,
    _compress_mps,           # fallback
    _signed_logabs_from_scalar,
    _signed_logsumexp,
    _signed_ratio_over_positive_base,
    _scalar_from_signed_log,
    LOG_TWO,
)


def _mps_normalise(mps):
    """Normalise each MPS tensor by its max element; return cumulative log scale."""
    total_log = 0.0
    for j in range(len(mps)):
        n = np.max(np.abs(mps[j]))
        if n > 0:
            total_log += np.log(n)
            mps[j] = mps[j] / n
    return mps, total_log


# ---------------------------------------------------------------------------
# MPS format conversion helpers
# ---------------------------------------------------------------------------
#
# Our convention:  each site k has shape (χ_l, d=2, χ_r)
#   site 0   : χ_l = 1  (left boundary)
#   site W-1 : χ_r = 1  (right boundary)
#
# quimb's MatrixProductState with shape='lpr' expects:
#   site 0   : (d, χ_r)       – no left bond axis
#   site 1…W-2: (χ_l, d, χ_r)
#   site W-1 : (χ_l, d)       – no right bond axis


def _mps_to_quimb(mps):
    """
    Convert our numpy-list MPS to a quimb MatrixProductState.

    Returns the qtn.MatrixProductState object, or None on failure.
    """
    W = len(mps)
    arrays = []
    for k, A in enumerate(mps):
        if k == 0:
            arrays.append(A[0, :, :])   # (d, χ_r)
        elif k == W - 1:
            arrays.append(A[:, :, 0])   # (χ_l, d)
        else:
            arrays.append(A.copy())     # (χ_l, d, χ_r)
    try:
        return qtn.MatrixProductState(arrays, shape='lpr')
    except Exception:
        return None


def _quimb_to_mps(qmps, W):
    """
    Extract our numpy-list MPS from a quimb MatrixProductState.

    Uses explicit index-name lookup to be robust against any internal
    reordering that quimb performs during canonicalisation / compression.
    Returns None on failure.
    """
    try:
        result = []
        for k in range(W):
            t = qmps[k]
            p = qmps.site_ind(k)

            if k == 0:
                r = qmps.bond(k, k + 1)
                data = t.transpose(p, r).data          # (d, χ_r)
                result.append(data[np.newaxis, :, :])  # (1, d, χ_r)

            elif k == W - 1:
                l = qmps.bond(k - 1, k)
                data = t.transpose(l, p).data          # (χ_l, d)
                result.append(data[:, :, np.newaxis])  # (χ_l, d, 1)

            else:
                l = qmps.bond(k - 1, k)
                r = qmps.bond(k, k + 1)
                data = t.transpose(l, p, r).data       # (χ_l, d, χ_r)
                result.append(data)

        return result

    except Exception:
        return None


# ---------------------------------------------------------------------------
# Core: quimb-enhanced MPS compression
# ---------------------------------------------------------------------------


def _compress_mps_quimb(mps, max_bond, cutoff=1e-14):
    """
    Two-pass MPS compression via quimb.

    Algorithm
    ---------
    1. Convert to quimb MatrixProductState.
    2. Left-canonicalise (QR sweep, no truncation): puts the MPS into
       a form where truncation errors will be evenly distributed.
    3. Right-canonicalise with SVD truncation to max_bond.
    4. Convert back to our numpy format.

    Falls back to the single-pass custom _compress_mps on any failure.
    """
    W = len(mps)
    if W == 0:
        return mps

    # Skip if already within bond limit
    if all(A.shape[2] <= max_bond and A.shape[0] <= max_bond for A in mps):
        return mps

    qmps = _mps_to_quimb(mps)
    if qmps is None:
        return _compress_mps(mps, max_bond)

    try:
        # Step 2: right-canonicalise via QR (R→L, no truncation).
        # Puts the MPS into right-canonical form (orthocenter at site 0),
        # ensuring each bond's singular values are normalised.
        qmps.right_canonize(normalize=False)

        # Step 3: left→right SVD sweep with truncation.
        # Starting from right-canonical form, each L→R SVD truncation is
        # locally optimal (minimises truncation error for that bond).
        qmps.compress(max_bond=max_bond, cutoff=cutoff, normalize=False)

    except Exception:
        # Fallback: plain compress() without prior canonicalisation
        try:
            qmps = _mps_to_quimb(mps)
            if qmps is None:
                raise RuntimeError
            qmps.compress(max_bond=max_bond, cutoff=cutoff, normalize=False)
        except Exception:
            return _compress_mps(mps, max_bond)

    result = _quimb_to_mps(qmps, W)
    if result is None:
        return _compress_mps(mps, max_bond)
    return result


# ---------------------------------------------------------------------------
# Main contraction: row-by-row sweep with quimb compression
# ---------------------------------------------------------------------------


def quimb_bMPS_contract(site_grid, max_bond=16, return_log=False,
                         compress_cutoff=1e-14):
    """
    Contract a rectangular 2D TN via boundary MPS with quimb compression.

    Identical logic to bmps_contraction.bMPS_contract, except that
    _compress_mps is replaced by _compress_mps_quimb (two-pass canonical
    compression).

    Parameters
    ----------
    site_grid : list[list[ndarray]]
        H × W grid of site tensors, each shape (2, 2, 2, 2).
    max_bond : int
        MPS bond-dimension cap.
    return_log : bool
        If True, return (sign, log|result|) instead of a scalar.
    compress_cutoff : float
        SVD singular-value cutoff passed to quimb.
    """
    H = len(site_grid)
    W = len(site_grid[0])

    if H == 0 or W == 0:
        return (1.0, 0.0) if return_log else 1.0

    log_scale = 0.0

    # --- Row 0: build initial MPS (sum over top boundary axis) ---
    mps = []
    for j in range(W):
        A = site_grid[0][j].sum(axis=0).transpose(2, 1, 0)  # (left, bottom, right)
        mps.append(A)
    mps[0] = mps[0].sum(axis=0, keepdims=True)   # (1, bottom, right)
    mps[-1] = mps[-1].sum(axis=2, keepdims=True)  # (left, bottom, 1)

    if H == 1:
        sign, logabs = _close_mps_log(mps)
    else:
        # --- Middle rows: apply MPO + quimb compress ---
        for i in range(1, H - 1):
            mps = _apply_mpo(mps, site_grid[i])
            mps, lf = _mps_normalise(mps)
            log_scale += lf
            mps = _compress_mps_quimb(mps, max_bond, cutoff=compress_cutoff)

        # --- Last row: sum over bottom boundary, contract to scalar ---
        sign, logabs = _apply_last_row_log(mps, site_grid[-1])

    if sign == 0.0 or not np.isfinite(logabs):
        return (0.0, float('-inf')) if return_log else 0.0

    total_logabs = float(logabs + log_scale)
    if return_log:
        return sign, total_logabs
    return _scalar_from_signed_log(sign, total_logabs)


# ---------------------------------------------------------------------------
# High-level interface (drop-in for bMPS_contract_region / bMPS_prob_with_parity)
# ---------------------------------------------------------------------------


def quimb_contract_region(m_grid, region_Q, region_A_signed=None,
                           p=0.1, max_bond=16, return_log=False,
                           compress_cutoff=1e-14):
    """
    Drop-in replacement for bMPS_contract_region using quimb compression.
    """
    region_Q = set(region_Q)
    region_A = set(region_A_signed) if region_A_signed else set()
    all_plaq = region_Q | region_A

    if not all_plaq:
        return (1.0, 0.0) if return_log else 1.0

    R_min = min(i for i, _ in all_plaq)
    R_max = max(i for i, _ in all_plaq)
    C_min = min(j for _, j in all_plaq)
    C_max = max(j for _, j in all_plaq)

    grid = build_site_grid(R_min, R_max, C_min, C_max,
                           m_grid, region_Q, region_A, p)
    return quimb_bMPS_contract(grid, max_bond=max_bond, return_log=return_log,
                                compress_cutoff=compress_cutoff)


def quimb_prob_with_parity(m_grid, region_Q, region_A, pi_A,
                            p=0.1, max_bond=16,
                            return_log=False,
                            stabilise_projection=False,
                            projection_eps=1e-6,
                            projection_guard=0.05,
                            return_meta=False,
                            compress_cutoff=1e-14):
    """
    Drop-in replacement for bMPS_prob_with_parity using quimb compression.

    Pr(m_Q, π(m_A) = pi_A) = ½ [ Pr(m_Q) + (-1)^{pi_A} · Pr_signed(m_Q) ]
    """
    sign_Q, log_Q = quimb_contract_region(
        m_grid, region_Q, None, p, max_bond, return_log=True,
        compress_cutoff=compress_cutoff,
    )
    sign_s, log_s = quimb_contract_region(
        m_grid, region_Q, region_A, p, max_bond, return_log=True,
        compress_cutoff=compress_cutoff,
    )

    sign_s *= (-1) ** int(pi_A)

    meta = {
        "projection_used": False,
        "projection_clamped": False,
        "raw_ratio": float("nan"),
        "clipped_ratio": float("nan"),
    }

    can_project = False
    ratio = float("nan")
    if stabilise_projection and sign_Q > 0.0 and np.isfinite(log_Q):
        ratio = _signed_ratio_over_positive_base(sign_s, log_s, log_Q)
        guard = float(projection_guard)
        can_project = np.isfinite(ratio) and (-1.0 - guard <= ratio <= 1.0 + guard)

    if can_project:
        lo = -1.0 + float(projection_eps)
        hi = 1.0 - float(projection_eps)
        clipped = float(np.clip(ratio, lo, hi))
        meta["projection_used"] = True
        meta["projection_clamped"] = bool(clipped != ratio)
        meta["raw_ratio"] = float(ratio)
        meta["clipped_ratio"] = float(clipped)
        sign_prob = 1.0
        log_prob = float(log_Q + np.log1p(clipped) - LOG_TWO)
    else:
        sign_prob, log_prob = _signed_logsumexp(log_Q, sign_Q, log_s, sign_s)
        if sign_prob != 0.0 and np.isfinite(log_prob):
            log_prob -= LOG_TWO

    if return_meta:
        if return_log:
            return sign_prob, log_prob, meta
        return _scalar_from_signed_log(sign_prob, log_prob), meta

    if return_log:
        return sign_prob, log_prob
    return _scalar_from_signed_log(sign_prob, log_prob)


# ---------------------------------------------------------------------------
# CMI estimator using quimb backend
# ---------------------------------------------------------------------------


def calculate_CMI_quimb(L=None, p=0.1, r=6, num_samples=500,
                         max_bond=16, seed=None, verbose=True,
                         stabilise_projection=False,
                         projection_eps=1e-6,
                         projection_guard=0.05,
                         compress_cutoff=1e-14):
    """
    Estimate I(A:C|B) via Monte Carlo using the quimb bMPS backend.

    Interface identical to calculate_CMI_bMPS; only the compression
    subroutine is replaced.
    """
    if L is None:
        L = 2 + 4 * r

    if seed is not None:
        np.random.seed(seed)

    A, B, C = define_geometry_geom1(L, r)
    AB = A | B
    BC = B | C
    ABC = A | B | C

    if verbose:
        print(f"[quimb] geom1 — L={L}, p={p:.4f}, r={r}, max_bond={max_bond}")
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

        sign_ABC, log_ABC = quimb_contract_region(
            m_grid, ABC, None, p, max_bond, return_log=True,
            compress_cutoff=compress_cutoff,
        )
        sign_AB, log_AB = quimb_contract_region(
            m_grid, AB, None, p, max_bond, return_log=True,
            compress_cutoff=compress_cutoff,
        )
        sign_Bpi, log_Bpi, meta_B = quimb_prob_with_parity(
            m_grid, B, A, pi_A, p, max_bond, return_log=True,
            stabilise_projection=stabilise_projection,
            projection_eps=projection_eps,
            projection_guard=projection_guard,
            return_meta=True,
            compress_cutoff=compress_cutoff,
        )
        sign_BCpi, log_BCpi, meta_BC = quimb_prob_with_parity(
            m_grid, BC, A, pi_A, p, max_bond, return_log=True,
            stabilise_projection=stabilise_projection,
            projection_eps=projection_eps,
            projection_guard=projection_guard,
            return_meta=True,
            compress_cutoff=compress_cutoff,
        )

        projection_used += int(meta_B["projection_used"]) + int(meta_BC["projection_used"])
        projection_clamped += int(meta_B["projection_clamped"]) + int(meta_BC["projection_clamped"])

        lp_ABC = _positive_log_from_signed_log(sign_ABC, log_ABC)
        lp_AB = _positive_log_from_signed_log(sign_AB, log_AB)
        lp_Bpi = _positive_log_from_signed_log(sign_Bpi, log_Bpi)
        lp_BCpi = _positive_log_from_signed_log(sign_BCpi, log_BCpi)

        if None in (lp_ABC, lp_AB, lp_Bpi, lp_BCpi):
            skipped += 1
            continue

        CMI_sum += -lp_BCpi + lp_ABC + lp_Bpi - lp_AB

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


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from .cmi_calculation import _contract, define_geometry_geom1
    from .bmps_contraction import bMPS_contract_region, bMPS_prob_with_parity

    np.random.seed(42)
    L, p, r = 10, 0.1, 2
    A, B, C = define_geometry_geom1(L, r)
    AB = A | B
    BC = B | C
    ABC = A | B | C

    h = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
    v = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])
    m = (h[:L] + h[1:] + v[:, :L] + v[:, 1:]) % 2
    pi_A = int(sum(m[i, j] for i, j in A) % 2)

    print("Correctness check — exact vs custom bMPS vs quimb (max_bond=16):")
    for label, rQ, rA in [
        ('Pr(m_ABC)',     ABC, None),
        ('Pr(m_AB)',      AB,  None),
        ('Pr(m_B)',       B,   None),
        ('Pr_signed(m_B)',B,   A),
        ('Pr(m_BC)',      BC,  None),
        ('Pr_signed(m_BC)',BC, A),
    ]:
        exact  = _contract(m, rQ, region_A_signed=rA, p=p)
        custom = bMPS_contract_region(m, rQ, rA, p, max_bond=16)
        qb     = quimb_contract_region(m, rQ, rA, p, max_bond=16)
        rel_c  = abs(exact - custom) / (abs(exact) + 1e-30)
        rel_q  = abs(exact - qb)    / (abs(exact) + 1e-30)
        print(f"  {label:<26} exact={exact:.5e}  "
              f"custom_err={rel_c:.2e}  quimb_err={rel_q:.2e}")

    print("\nCMI comparison (L=10, p=0.1, r=2, N=200, seed=42):")
    from .cmi_calculation import calculate_CMI
    from .bmps_contraction import bMPS_contract_region as _unused
    from .cmi_calculation import calculate_CMI_bMPS

    t0 = time.perf_counter()
    cmi_exact = calculate_CMI(L=L, p=p, r=r, num_samples=200, seed=42, verbose=False)
    t_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    cmi_bmps = calculate_CMI_bMPS(L=L, p=p, r=r, num_samples=200,
                                   max_bond=16, seed=42, verbose=False)
    t_bmps = time.perf_counter() - t0

    t0 = time.perf_counter()
    cmi_quimb = calculate_CMI_quimb(L=L, p=p, r=r, num_samples=200,
                                     max_bond=16, seed=42, verbose=False)
    t_quimb = time.perf_counter() - t0

    print(f"  exact  CMI = {cmi_exact:.6f}  ({t_exact:.2f}s)")
    print(f"  custom CMI = {cmi_bmps:.6f}  ({t_bmps:.2f}s)")
    print(f"  quimb  CMI = {cmi_quimb:.6f}  ({t_quimb:.2f}s)")
