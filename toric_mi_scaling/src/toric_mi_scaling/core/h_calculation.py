import numpy as np
import opt_einsum as oe



def _accumulator_dtype():
    """Use longdouble when it offers wider range than float64."""
    try:
        if np.finfo(np.longdouble).max > np.finfo(np.float64).max:
            return np.longdouble
    except (TypeError, ValueError):
        pass
    return np.float64


ACCUM_DTYPE = _accumulator_dtype()



def create_Q_tensor(m_val, dtype=np.float64):
    """
    生成方形张量 Q (公式 E5)。
    m_val = 0 (黄色 Q^0), m_val = 1 (绿色 Q^1)
    """
    Q = np.zeros((2, 2, 2, 2), dtype=dtype)
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    if (i + j + k + l) % 2 == m_val:
                        Q[i, j, k, l] = 1.0
    return Q



def _positive_log_from_scalar(x):
    """Return log(x) for finite positive scalars; otherwise None."""
    x = float(x)
    if (not np.isfinite(x)) or x <= 0.0:
        return None
    return ACCUM_DTYPE(np.log(x))



def calculate_H_m(L=3, p=0.1, num_samples=100):
    """
    计算 L x L 晶格上的 H(m)
    L: 晶格大小
    p: 物理错误率
    num_samples: 蒙特卡洛采样次数
    """
    # 尽量用更高精度的张量和累加器，降低极端参数下的下溢风险
    Q0 = create_Q_tensor(0, dtype=ACCUM_DTYPE)
    Q1 = create_Q_tensor(1, dtype=ACCUM_DTYPE)
    W = np.array([1 - p, p], dtype=ACCUM_DTYPE)

    H_sum = ACCUM_DTYPE(0.0)
    skipped = 0

    for _ in range(num_samples):
        # --- 步骤 1: 采样错误 e ---
        h_errors = np.random.choice([0, 1], size=(L + 1, L), p=[1 - p, p])
        v_errors = np.random.choice([0, 1], size=(L, L + 1), p=[1 - p, p])

        # --- 步骤 2: 计算构型 m = ∂e ---
        m_grid = (h_errors[:L, :] + h_errors[1:, :] + v_errors[:, :L] + v_errors[:, 1:]) % 2

        # --- 步骤 3 & 4: 构建并收缩张量网络 ---
        tensors_and_indices = []

        # 为了构建网络，给每一条边分配一个唯一的整数 ID
        edge_id_counter = 0
        h_edge_ids = np.zeros((L + 1, L), dtype=int)
        v_edge_ids = np.zeros((L, L + 1), dtype=int)

        for i in range(L + 1):
            for j in range(L):
                h_edge_ids[i, j] = edge_id_counter
                tensors_and_indices.extend([W, [edge_id_counter]])
                edge_id_counter += 1

        for i in range(L):
            for j in range(L + 1):
                v_edge_ids[i, j] = edge_id_counter
                tensors_and_indices.extend([W, [edge_id_counter]])
                edge_id_counter += 1

        # 将每个 plaquette 的 Q 张量加入网络
        for i in range(L):
            for j in range(L):
                top_id = h_edge_ids[i, j]
                bottom_id = h_edge_ids[i + 1, j]
                left_id = v_edge_ids[i, j]
                right_id = v_edge_ids[i, j + 1]

                Q_selected = Q1 if m_grid[i, j] == 1 else Q0
                tensors_and_indices.extend([
                    Q_selected,
                    [top_id, right_id, bottom_id, left_id],
                ])

        prob_m = oe.contract(*tensors_and_indices)

        # --- 步骤 5: 计算对数并累加求期望 ---
        log_prob_m = _positive_log_from_scalar(prob_m)
        if log_prob_m is None:
            skipped += 1
            continue
        H_sum -= log_prob_m

    valid = num_samples - skipped
    if skipped:
        frac = skipped / num_samples
        tag = "WARNING: " if frac > 0.05 else ""
        print(f"{tag}Skipped {skipped}/{num_samples} samples ({100 * frac:.1f}%)")

    if valid == 0:
        print(f"Lattice: {L}x{L}, p: {p}, Samples: {num_samples}")
        print("Calculated H(m) failed: no valid samples.")
        return None

    H_m_approx = float(H_sum / valid)
    print(f"Lattice: {L}x{L}, p: {p}, Samples: {num_samples}")
    print(f"Calculated H(m) ≈ {H_m_approx:.6f} ({valid} valid samples)")

    return H_m_approx


if __name__ == "__main__":
    calculate_H_m(L=3, p=0.1, num_samples=10000)
