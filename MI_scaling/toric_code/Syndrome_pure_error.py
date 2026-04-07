import numpy as np
import itertools
import matplotlib.pyplot as plt
from time import time

def surface_code_MI_fast(L, p_err):
    """
    精确计算 L×L 表面码互信息 I(A;B|S=0)。

    加速原理:
        原始枚举 2^(L²) 个配置再过滤，代价极高。
        L×L 表面码有 (L-1)² 个独立 Z型 plaquette 约束，
        有效配置空间维数 = L² - (L-1)² = 2L-1。

        选第0行 (L个) + 第0列去掉角点 (L-1个) = 2L-1 个自由比特，
        其余比特由约束逐一传播确定:
            e[r+1,c+1] = e[r,c] XOR e[r+1,c] XOR e[r,c+1]

        只需枚举 2^(2L-1) 个有效配置，无需任何过滤。

    可处理规模: L ≤ 12 (N_valid = 2^23 ≈ 8M，内存 ~1GB)
    """
    n = L * L
    num_free = 2 * L - 1

    def idx(i, j):
        return i * L + j

    # 自由比特: 第0行 + 第0列(不含角点)
    free_idx = [idx(0, j) for j in range(L)] + [idx(i, 0) for i in range(1, L)]

    # A/B 分区: 左⌊L/2⌋列 vs 右⌈L/2⌉列
    split = L // 2
    A_idx = [idx(i, j) for i in range(L) for j in range(split)]
    B_idx = [idx(i, j) for i in range(L) for j in range(split, L)]

    # -------------------------------------------------------
    # 生成所有 2^(2L-1) 个有效配置 (不过滤，直接构造)
    # -------------------------------------------------------
    N_valid = 1 << num_free  # 2^(2L-1)

    free_vals = np.array(list(itertools.product([0, 1], repeat=num_free)), dtype=np.int8)
    # shape: (N_valid, 2L-1)

    e = np.zeros((N_valid, n), dtype=np.int8)
    for k, fi in enumerate(free_idx):
        e[:, fi] = free_vals[:, k]

    # 约束传播: 按行列顺序确定所有内部比特
    for r in range(L - 1):
        for c in range(L - 1):
            e[:, idx(r+1, c+1)] = (
                e[:, idx(r, c)] ^ e[:, idx(r+1, c)] ^ e[:, idx(r, c+1)]
            )

    # -------------------------------------------------------
    # 计算归一化概率 (log空间避免数值下溢)
    # -------------------------------------------------------
    num_err = e.sum(axis=1).astype(np.float64)
    log_p = num_err * np.log(p_err) + (n - num_err) * np.log(1 - p_err)
    log_p -= log_p.max()
    probs = np.exp(log_p)
    probs /= probs.sum()

    # -------------------------------------------------------
    # 计算边缘熵 (整数哈希向量化聚合)
    # -------------------------------------------------------
    def marginal_entropy(part_idx):
        k = len(part_idx)
        part = e[:, part_idx].astype(np.int64)
        if k <= 62:
            powers = np.int64(1) << np.arange(k, dtype=np.int64)
            keys = part @ powers
            unique_keys, inv = np.unique(keys, return_inverse=True)
            P = np.zeros(len(unique_keys))
            np.add.at(P, inv, probs)
        else:  # k > 62 时退回字节哈希 (大L情况)
            P_dict = {}
            for i in range(N_valid):
                key = part[i].tobytes()
                P_dict[key] = P_dict.get(key, 0.0) + probs[i]
            P = np.array(list(P_dict.values()))
        P = P[P > 1e-15]
        return float(-np.sum(P * np.log(P)))

    S_A  = marginal_entropy(A_idx)
    S_B  = marginal_entropy(B_idx)
    mask = probs > 1e-15
    S_AB = float(-np.sum(probs[mask] * np.log(probs[mask])))

    return S_A + S_B - S_AB


# ====================
# 运行
# ====================
if __name__ == "__main__":
    p = 0.1
    L_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    print(f"2D Surface Code MI (bit flip, p={p})")
    print(f"{'L':>4} {'n=L²':>6} {'N_valid=2^(2L-1)':>18} {'I(A;B|S=0)':>14}  time(s)")
    print("-" * 62)

    MI_results, n_list = [], []
    for L in L_list:
        n = L * L
        N_valid = 1 << (2 * L - 1)
        t0 = time()
        mi = surface_code_MI_fast(L, p)
        dt = time() - t0
        MI_results.append(mi)
        n_list.append(n)
        print(f"{L:>4} {n:>6} {N_valid:>18} {mi:>14.6f}  {dt:.3f}s")

    plt.figure(figsize=(6, 4))
    plt.plot(n_list, MI_results, 'o-', lw=2, ms=8)
    plt.xlabel("Total qubits $n = L^2$")
    plt.ylabel("$I(A;B|S=0)$ [nats]")
    plt.title(f"2D Surface Code MI (p={p})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("surface_code_MI_2D.png", dpi=150)
    plt.show()
    print("Saved: surface_code_MI_2D.png")
