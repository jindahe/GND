import numpy as np
import opt_einsum as oe

def create_Q_tensor(m_val):
    """
    生成方形张量 Q (公式 E5)。
    m_val = 0 (黄色 Q^0), m_val = 1 (绿色 Q^1)
    """
    Q = np.zeros((2, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for l in range(2):
                    # 宇称约束：四条边之和 mod 2 == m_val
                    if (i + j + k + l) % 2 == m_val:
                        Q[i, j, k, l] = 1.0
    return Q

def calculate_H_m(L=3, p=0.1, num_samples=100):
    """
    计算 L x L 晶格上的 H(m)
    L: 晶格大小
    p: 物理错误率
    num_samples: 蒙特卡洛采样次数
    """
    # 提前定义好 Q^0 和 Q^1 张量
    Q0 = create_Q_tensor(0)
    Q1 = create_Q_tensor(1)
    
    # 定义边上的概率权重张量 (简化公式 E4 中的圆圈张量)
    W = np.array([1 - p, p])
    
    H_sum = 0.0

    for sample in range(num_samples):
        # --- 步骤 1: 采样错误 e ---
        # 对于 L x L 的 plaquettes，有 L*(L+1) 条水平边和 L*(L+1) 条垂直边
        num_h_edges = L * (L + 1)
        num_v_edges = L * (L + 1)
        
        h_errors = np.random.choice([0, 1], size=(L + 1, L), p=[1-p, p])
        v_errors = np.random.choice([0, 1], size=(L, L + 1), p=[1-p, p])

        # --- 步骤 2: 计算构型 m = ∂e ---
        m_grid = np.zeros((L, L), dtype=int)
        for i in range(L):
            for j in range(L):
                top = h_errors[i, j]
                bottom = h_errors[i + 1, j]
                left = v_errors[i, j]
                right = v_errors[i, j + 1]
                m_grid[i, j] = (top + bottom + left + right) % 2

        # --- 步骤 3 & 4: 构建并收缩张量网络 ---
        tensors_and_indices = []
        
        # 为了构建网络，给每一条边分配一个唯一的整数 ID
        edge_id_counter = 0
        h_edge_ids = np.zeros((L + 1, L), dtype=int)
        v_edge_ids = np.zeros((L, L + 1), dtype=int)
        
        for i in range(L + 1):
            for j in range(L):
                h_edge_ids[i, j] = edge_id_counter
                tensors_and_indices.extend([W, [edge_id_counter]]) # 加入权重 W
                edge_id_counter += 1
                
        for i in range(L):
            for j in range(L + 1):
                v_edge_ids[i, j] = edge_id_counter
                tensors_and_indices.extend([W, [edge_id_counter]]) # 加入权重 W
                edge_id_counter += 1

        # 将每个 plaquette 的 Q 张量加入网络
        for i in range(L):
            for j in range(L):
                top_id = h_edge_ids[i, j]
                bottom_id = h_edge_ids[i + 1, j]
                left_id = v_edge_ids[i, j]
                right_id = v_edge_ids[i, j + 1]
                
                # 依据 m 选取 Q0 还是 Q1
                Q_selected = Q1 if m_grid[i, j] == 1 else Q0
                # 添加到张量收缩列表中 (顺序：上, 右, 下, 左)
                tensors_and_indices.extend([Q_selected, [top_id, right_id, bottom_id, left_id]])

        # 使用 opt_einsum 执行精确收缩，得到概率 Pr(m)
        # opt_einsum 能够自动寻找最优收缩路径
        prob_m = oe.contract(*tensors_and_indices)
        
        # --- 步骤 5: 计算对数并累加求期望 ---
        if prob_m > 0:
            H_sum -= np.log(prob_m)
            
    # 计算均值
    H_m_approx = H_sum / num_samples
    print(f"Lattice: {L}x{L}, p: {p}, Samples: {num_samples}")
    print(f"Calculated H(m) ≈ {H_m_approx:.6f}")
    
    return H_m_approx

# 执行计算
if __name__ == "__main__":
    calculate_H_m(L=3, p=0.1, num_samples=10000)