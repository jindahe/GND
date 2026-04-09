# 当前问题与解决方案

## 问题一：bMPS 精度在大 r 下不足

### 现象
- chi=16 在 r=6（26×26 区域）上的 Pr(m_ABC) 误差约 35%，Pr(m_BC) 误差约 50%
- 导致 CMI 估计器出现大幅负值（如 −0.08 nats），物理上不合理
- 误差随 p 非均匀分布，在 p≈p_c 附近（系统相变点）最大

### 根本原因
bMPS 行方向截断误差与区域尺寸的关系：

| r | 区域尺寸 | ABC 误差（chi=16） |
|---|---|---|
| 2 | 10×10 | <0.1%（近似精确） |
| 3 | 14×14 | ~0.5%（精确 TN 可验证） |
| 5 | 22×22 | ~10%（BC_signed） |
| 6 | 26×26 | ~35-55% |

相变点附近 2D TN 的行方向纠缠熵增大，需更大 chi 才能捕获。

### 可行方案

**方案 A（已实现）：精确 TN 用于 r≤4**
- r=3: L=14, 0.1s/sample, ~0.3h 全扫 ✅
- r=4: L=18, 0.5s/sample, ~1.6h 全扫 ✅
- r=5: L=22, 21s/sample, ~67h → 不可行 ❌

**方案 B：bMPS 提高 chi**
| chi | r=5 速度 | r=6 速度 | 精度（r=5, BC_signed） |
|---|---|---|---|
| 16 | 0.17s | 0.17s | 25% 误差 |
| 64 | 1.1s | 1.8s | 2.5% 误差 |
| 128 | ~4s | ~7s | 待测 |

chi=64 在 r=5 上误差降至 ~2.5%，但：
- CMI 绝对误差约 0.02–0.03 nats（相对误差 10–60%）
- 全扫（23p×500s）耗时：r=5 约 3.6h，r=6 约 5.8h

**方案 C：列方向收缩（行方向纠缠熵可能更低）**
- 未实现，理论上可能改善精度
- 需重写 bmps_contraction.py 的收缩顺序

**方案 D：变分 MPS 优化（DMRG 风格）**
- 替代 SVD 截断的非变分方案
- 实现复杂度高，但精度可能更好

### 当前状态
- 已用精确 TN 完成 L=10, r=1,2,3 的 p-sweep（CMI_results_L10.csv）
- bMPS chi=16 完成 r=6 的 p-sweep，但结果不可信（CMI_bMPS_results.csv）
- **待做**：用精确 TN 跑 r=3,4（L=14,18），用 bMPS chi=64 跑 r=5,6

---

## 问题二：SVD 不收敛（Apple Accelerate BLAS）

### 现象
- `np.linalg.svd` 在特定矩阵尺寸下抛出 `LinAlgError: SVD did not converge`
- 出现于 chi≥128、较大区域（r≥5）
- 频率低但会导致整个计算崩溃

### 根本原因
numpy 默认使用 Apple Accelerate 的 `dgesdd`（分治算法），对特定矩阵模式不稳定。

### 已实现的修复
在 `bmps_contraction._svd()` 中加入 fallback：
```python
def _svd(A_mat):
    try:
        return np.linalg.svd(A_mat, full_matrices=False)
    except np.linalg.LinAlgError:
        import scipy.linalg
        return scipy.linalg.svd(A_mat, full_matrices=False, lapack_driver='gesvd')
```
scipy 的 `gesvd`（Jacobi 算法）更稳定，已验证 chi=128 下 r=5 无崩溃。

---

## 问题三：MC 估计器方差大

### 现象
- 1000 个样本下，CMI 估计的相对标准差约 20%（r=2, p=0.1）
- 单点 CMI 值（r=1,2,3）在不同 seed 下跳动 0.006–0.007 nats

### 根本原因
CMI 估计量 $X = -\log\Pr(m_{BC},\pi_A) + \log\Pr(m_{ABC}) + \log\Pr(m_B,\pi_A) - \log\Pr(m_{AB})$ 的方差由各对数项的协方差决定。接近 $p_c$ 时概率比值变化剧烈，方差增大。

### 当前策略
- 500 samples/点（~14% 相对误差/σ）
- 用于定性验证（峰位、单调性），不做精确数值拟合

### 改进方向
- 控制变量法（control variates）降低方差
- 重要性采样
- 增大 num_samples（代价线性增长）

---

## 整体进度

| 任务 | 状态 | 文件 |
|---|---|---|
| P1: CMI 公式 + geom1 几何 | ✅ | cmi_calculation.py |
| P2: opt_einsum 路径缓存 | ✅ (2x 加速) | cmi_calculation.py |
| P3: bMPS 逐行收缩 | ✅ 实现完成，精度受限 | bmps_contraction.py |
| L=10 r=1,2,3 p-sweep | ✅ | CMI_results_L10.csv |
| r=6 chi=16 sweep | ⚠️ 结果不可信 | CMI_bMPS_results.csv |
| r=3,4 精确 TN sweep | 🔲 待运行 | — |
| r=5,6 chi=64 sweep | 🔲 待运行 | — |
