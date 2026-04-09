"""
Boundary MPS (bMPS) contraction for 2D tensor networks on rectangular grids.

Algorithm (row-by-row):
1. Build initial MPS from row 0 (sum over top boundary with W absorbed).
2. For each middle row: apply as MPO → bond dims double → compress to max_bond via SVD.
3. Apply last row, sum over bottom boundary, contract residual to scalar.

Site tensor convention (each plaquette):
    T[top, right, bottom, left]  — axes 0,1,2,3
    W is absorbed:
        shared edge  →  sqrt(W[e])  multiplied in from each side
        boundary edge → W[e]  multiplied in from the one touching side

All 6 CMI contractions become rectangular bMPS calls:
    Pr(m_ABC)      : (2+4r)×(2+4r) box, Q^{m} everywhere
    Pr(m_AB)       : (2+2r)×(2+2r) box, Q^{m} everywhere
    Pr(m_B)        : (2+2r)×(2+2r) box, B→Q^{m}, A→Q_plus
    Pr_signed(m_B) : (2+2r)×(2+2r) box, B→Q^{m}, A→Q_minus
    Pr(m_BC)       : (2+4r)×(2+4r) box, BC→Q^{m}, A→Q_plus
    Pr_signed(m_BC): (2+4r)×(2+4r) box, BC→Q^{m}, A→Q_minus
"""

import numpy as np
from .cmi_calculation import Q0, Q1, Q_minus, Q_plus


LOG_TWO = float(np.log(2.0))
LOG_MAX_FLOAT64 = float(np.log(np.finfo(np.float64).max))


# ---------------------------------------------------------------------------
# Signed log helpers
# ---------------------------------------------------------------------------


def _signed_logabs_from_scalar(x):
    """Return (sign, log|x|) for a scalar, using -inf for zero."""
    x = float(x)
    if (not np.isfinite(x)) or x == 0.0:
        return 0.0, float('-inf')
    return float(np.sign(x)), float(np.log(abs(x)))



def _signed_logsumexp(logabs_a, sign_a, logabs_b, sign_b):
    """
    Stable signed sum in log space.

    Returns sign, logabs for:
        sign_a * exp(logabs_a) + sign_b * exp(logabs_b)
    """
    if sign_a == 0.0 or not np.isfinite(logabs_a):
        return sign_b, logabs_b
    if sign_b == 0.0 or not np.isfinite(logabs_b):
        return sign_a, logabs_a

    if logabs_b > logabs_a:
        logabs_a, logabs_b = logabs_b, logabs_a
        sign_a, sign_b = sign_b, sign_a

    delta = logabs_b - logabs_a  # <= 0

    if sign_a == sign_b:
        return sign_a, float(logabs_a + np.log1p(np.exp(delta)))

    if delta == 0.0:
        return 0.0, float('-inf')

    diff = -np.expm1(delta)  # 1 - exp(delta), stable for delta near 0-
    if diff <= 0.0:
        return 0.0, float('-inf')

    return sign_a, float(logabs_a + np.log(diff))



def _scalar_from_signed_log(sign, logabs):
    """Convert signed log representation back to float, saturating on overflow."""
    if sign == 0.0 or not np.isfinite(logabs):
        return 0.0
    if logabs > LOG_MAX_FLOAT64:
        return float(sign) * float('inf')
    return float(sign) * float(np.exp(logabs))


# ---------------------------------------------------------------------------
# Site tensor construction
# ---------------------------------------------------------------------------


def make_site_tensor(i, j, m_grid, plaq_type, R_min, R_max, C_min, C_max, p):
    """
    Build T[top, right, bottom, left] for plaquette (i,j) with W absorbed.

    plaq_type : 'Q'     → Q^{m_grid[i,j]}
                'plus'  → Q_plus  (no parity constraint, marginalises out m_i)
                'minus' → Q_minus (signed Fourier parity trick)
    Shared edges absorb sqrt(W); boundary edges absorb full W.
    """
    W = np.array([1.0 - p, p], dtype=np.float64)
    sqW = np.sqrt(W)

    if plaq_type == 'Q':
        Q = Q1 if m_grid[i, j] == 1 else Q0
    elif plaq_type == 'plus':
        Q = Q_plus
    else:
        Q = Q_minus

    # Per-direction weight: W for boundary, sqrt(W) for shared
    w_top = W if i == R_min else sqW
    w_bottom = W if i == R_max else sqW
    w_left = W if j == C_min else sqW
    w_right = W if j == C_max else sqW

    # Broadcast: Q[top, right, bottom, left]
    T = (
        Q
        * w_top[:, None, None, None]
        * w_right[None, :, None, None]
        * w_bottom[None, None, :, None]
        * w_left[None, None, None, :]
    )
    return T  # shape (2, 2, 2, 2)



def build_site_grid(R_min, R_max, C_min, C_max,
                    m_grid, region_Q, region_A_signed, p):
    """
    Build H×W array of site tensors for a rectangular bounding box.

    region_Q         : set of (i,j) — use Q^{m_ij}
    region_A_signed  : set of (i,j) — use Q_minus  (None → no parity correction)
    All other plaquettes in the box use Q_plus (identity / full marginalisation).
    """
    region_A = set(region_A_signed) if region_A_signed else set()
    rows = range(R_min, R_max + 1)
    cols = range(C_min, C_max + 1)
    grid = []
    for i in rows:
        row_tensors = []
        for j in cols:
            if (i, j) in region_Q:
                ptype = 'Q'
            elif (i, j) in region_A:
                ptype = 'minus'
            else:
                ptype = 'plus'
            row_tensors.append(
                make_site_tensor(i, j, m_grid, ptype, R_min, R_max, C_min, C_max, p)
            )
        grid.append(row_tensors)
    return grid


# ---------------------------------------------------------------------------
# bMPS contraction
# ---------------------------------------------------------------------------


def bMPS_contract(site_grid, max_bond=16, return_log=False):
    """
    Contract a rectangular 2D TN via boundary MPS.

    site_grid   : list[list[ndarray]] of shape (H, W), each tensor (2,2,2,2)
    max_bond    : MPS bond dimension cap (truncation threshold)
    return_log  : if True, return (sign, logabs) instead of scalar

    Signed log output is useful for preventing underflow in downstream log-prob
    estimators, especially for large regions and parity-signed contractions.
    """
    H = len(site_grid)
    W = len(site_grid[0])

    if H == 0 or W == 0:
        return (1.0, 0.0) if return_log else 1.0

    log_scale = 0.0  # accumulated log of normalisations

    def _mps_normalise(mps):
        """Normalise each tensor, return cumulative log of removed scales."""
        total_log = 0.0
        for j in range(len(mps)):
            n = np.max(np.abs(mps[j]))
            if n > 0:
                total_log += np.log(n)
                mps[j] = mps[j] / n
        return mps, total_log

    # --- Row 0: build initial MPS ---
    # Sum over top (boundary) axis → shape (right, bottom, left) → transpose (left, bottom, right)
    mps = []
    for j in range(W):
        A = site_grid[0][j].sum(axis=0).transpose(2, 1, 0)  # (left, bottom, right)
        mps.append(A)

    # Sum out left boundary of site 0 and right boundary of site W-1
    mps[0] = mps[0].sum(axis=0, keepdims=True)   # (1, bottom, right)
    mps[-1] = mps[-1].sum(axis=2, keepdims=True)  # (left, bottom, 1)

    if H == 1:
        sign, logabs = _close_mps_log(mps)
    else:
        # --- Middle rows ---
        for i in range(1, H - 1):
            mps = _apply_mpo(mps, site_grid[i])
            mps, lf = _mps_normalise(mps)   # normalise BEFORE SVD to prevent divergence
            log_scale += lf
            mps = _compress_mps(mps, max_bond)

        # --- Last row: sum over bottom boundary, contract to scalar in signed-log form ---
        sign, logabs = _apply_last_row_log(mps, site_grid[-1])

    if sign == 0.0 or not np.isfinite(logabs):
        return (0.0, float('-inf')) if return_log else 0.0

    total_logabs = float(logabs + log_scale)
    if return_log:
        return sign, total_logabs
    return _scalar_from_signed_log(sign, total_logabs)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_mpo(mps, row_tensors):
    """
    Apply one row as an MPO. Bond dimensions grow ×2 (before compression).
    MPO convention: T[top, right, bottom, left] → M[d_in=top, d_out=bottom, m_l=left, m_r=right]
    """
    W_sites = len(mps)
    new_mps = []

    for j in range(W_sites):
        A = mps[j]               # (chi_l, d_in, chi_r)
        T = row_tensors[j]       # (top, right, bottom, left)
        M = T.transpose(0, 2, 3, 1)   # (d_in, d_out, m_l, m_r)

        chi_l, _, chi_r = A.shape
        # Contract: result[chi_l, m_l, d_out, chi_r, m_r]
        C = np.einsum('aib,iomr->amobr', A, M)   # (chi_l, m_l, d_out, chi_r, m_r)

        if j == 0:
            # Left boundary: sum over m_l (axis 1)
            C = C.sum(axis=1, keepdims=True)           # (chi_l, 1, d_out, chi_r, m_r)
            new_mps.append(C.reshape(chi_l, 2, chi_r * 2))
        elif j == W_sites - 1:
            # Right boundary: sum over m_r (axis 4)
            C = C.sum(axis=4, keepdims=True)            # (chi_l, m_l, d_out, chi_r, 1)
            new_mps.append(C.reshape(chi_l * 2, 2, chi_r))
        else:
            new_mps.append(C.reshape(chi_l * 2, 2, chi_r * 2))

    return new_mps



def _svd(A_mat):
    """Robust SVD: falls back to scipy gesvd if numpy gesdd fails."""
    try:
        return np.linalg.svd(A_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        import scipy.linalg
        return scipy.linalg.svd(A_mat, full_matrices=False, lapack_driver='gesvd')



def _compress_mps(mps, max_bond):
    """Left-to-right SVD sweep, truncating bonds to max_bond."""
    W = len(mps)
    for j in range(W - 1):
        A = mps[j]                         # (chi_l, d, chi_r)
        chi_l, d, chi_r = A.shape
        A_mat = A.reshape(chi_l * d, chi_r)
        U, S, Vh = _svd(A_mat)
        k = min(max_bond, len(S))
        U = U[:, :k]
        S = S[:k]
        Vh = Vh[:k, :]
        mps[j] = U.reshape(chi_l, d, k)
        SV = (S[:, None] * Vh)             # (k, chi_r_old)
        A_next = mps[j + 1]                # (chi_r_old, d_next, chi_r_next)
        mps[j + 1] = np.tensordot(SV, A_next, axes=([1], [0]))  # (k, d_next, chi_r_next)
    return mps



def _apply_last_row_log(mps, row_tensors):
    """
    Apply last row (sum over bottom boundary) and contract residual MPS to a
    signed log-value. Uses running normalisation to prevent under/overflow.
    """
    W_sites = len(mps)
    v = np.ones(1)
    log_scale = 0.0

    for j in range(W_sites):
        A = mps[j]          # (chi_l, d_in, chi_r)
        T = row_tensors[j]  # (top, right, bottom, left)
        M_last = T.sum(axis=2).transpose(0, 2, 1)   # (d_in, m_l, m_r)

        chi_l, _, chi_r = A.shape
        B = np.einsum('aib,imr->ambr', A, M_last)   # (chi_l, m_l, chi_r, m_r)

        if j == 0:
            B = B.sum(axis=1)                        # (chi_l=1, chi_r, m_r)
            B_mat = B.reshape(chi_l, chi_r * 2)
        elif j == W_sites - 1:
            B = B.sum(axis=3)                        # (chi_l, m_l, chi_r=1)
            B_mat = B.reshape(chi_l * 2, chi_r)
        else:
            B_mat = B.reshape(chi_l * 2, chi_r * 2)

        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            v = v @ B_mat

        norm = np.max(np.abs(v))
        if norm > 0:
            log_scale += np.log(norm)
            v /= norm
        else:
            return 0.0, float('-inf')

    sign, logabs = _signed_logabs_from_scalar(v.squeeze())
    if sign == 0.0:
        return 0.0, float('-inf')
    return sign, float(log_scale + logabs)



def _close_mps_log(mps):
    """Contract a single-row MPS to a signed log-value."""
    v = np.ones(1)
    log_scale = 0.0

    for A in mps:
        A_summed = A.sum(axis=1)           # (chi_l, chi_r)
        v = v @ A_summed.reshape(A.shape[0], A.shape[2])

        norm = np.max(np.abs(v))
        if norm > 0:
            log_scale += np.log(norm)
            v /= norm
        else:
            return 0.0, float('-inf')

    sign, logabs = _signed_logabs_from_scalar(v.squeeze())
    if sign == 0.0:
        return 0.0, float('-inf')
    return sign, float(log_scale + logabs)


# ---------------------------------------------------------------------------
# High-level interface matching _contract / CachedContractor API
# ---------------------------------------------------------------------------


def bMPS_contract_region(m_grid, region_Q, region_A_signed=None, p=0.1,
                         max_bond=16, return_log=False):
    """
    Drop-in replacement for cmi_calculation._contract using bMPS.

    Automatically determines the bounding box from region_Q ∪ region_A_signed.
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
    return bMPS_contract(grid, max_bond=max_bond, return_log=return_log)



def bMPS_prob_with_parity(m_grid, region_Q, region_A, pi_A, p=0.1,
                          max_bond=16, return_log=False):
    """
    Pr(m_Q, π(m_A) = pi_A) via Fourier parity trick using bMPS.

    If return_log=True, return the probability in signed log form.
    """
    sign_Q, log_Q = bMPS_contract_region(
        m_grid, region_Q, None, p, max_bond, return_log=True
    )
    sign_signed, log_signed = bMPS_contract_region(
        m_grid, region_Q, region_A, p, max_bond, return_log=True
    )

    sign_signed *= (-1) ** int(pi_A)
    sign_prob, log_prob = _signed_logsumexp(log_Q, sign_Q, log_signed, sign_signed)

    if sign_prob != 0.0 and np.isfinite(log_prob):
        log_prob -= LOG_TWO

    if return_log:
        return sign_prob, log_prob
    return _scalar_from_signed_log(sign_prob, log_prob)


# ---------------------------------------------------------------------------
# Self-test: compare bMPS vs exact for small L
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import time
    from .cmi_calculation import _contract, define_geometry_geom1

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

    print('Correctness check — exact vs bMPS (max_bond=16):')
    for label, region_Q, region_A_signed in [
        ('Pr(m_ABC)', ABC, None),
        ('Pr(m_AB)', AB, None),
        ('Pr(m_B)', B, None),
        ('Pr_signed(m_B)', B, A),
        ('Pr(m_BC)', BC, None),
        ('Pr_signed(m_BC)', BC, A),
    ]:
        exact = _contract(m, region_Q, region_A_signed=region_A_signed, p=p)
        approx = bMPS_contract_region(m, region_Q, region_A_signed, p, max_bond=16)
        rel = abs(exact - approx) / (abs(exact) + 1e-30)
        print(f'  {label:<25} exact={exact:.6e}  bMPS={approx:.6e}  rel_err={rel:.2e}')

    sign_prob, log_prob = bMPS_prob_with_parity(
        m, B, A, pi_A, p=p, max_bond=16, return_log=True
    )
    print(
        f"\nParity probability in signed-log form: sign={sign_prob:+.0f}, "
        f"logabs={log_prob:.6f}"
    )

    # Speed comparison: 50 samples
    N = 50
    print(f'\nSpeed (N={N} samples, L={L}, r={r}):')

    t0 = time.perf_counter()
    for _ in range(N):
        h = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
        v = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])
        m = (h[:L] + h[1:] + v[:, :L] + v[:, 1:]) % 2
        _contract(m, ABC, p=p)
    t_exact = time.perf_counter() - t0

    t0 = time.perf_counter()
    for _ in range(N):
        h = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
        v = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])
        m = (h[:L] + h[1:] + v[:, :L] + v[:, 1:]) % 2
        bMPS_contract_region(m, ABC, None, p)
        bMPS_contract_region(m, AB, None, p)
        bMPS_prob_with_parity(m, B, A, pi_A, p)
        bMPS_prob_with_parity(m, BC, A, pi_A, p)
    t_bmps = time.perf_counter() - t0

    print(f'  exact TN total: {t_exact:.3f}s  ({1e3 * t_exact / N:.1f} ms/sample)')
    print(f'  bMPS total:     {t_bmps:.3f}s  ({1e3 * t_bmps / N:.1f} ms/sample)')
