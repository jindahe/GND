# H_calculation.py 改进计划

基于对现有代码的深度分析，按优先级列出后续改进任务。

---

## 🔴 P1：实现完整 CMI 公式（环形区域 + π(m_A) 修正）

**目标：** 能复现论文图 3，计算 $I(A:C|B) = H(\mathbf{m}_{BC}, \pi(\mathbf{m}_A)) - H(\mathbf{m}_{ABC}) - H(\mathbf{m}_B, \pi(\mathbf{m}_A)) + H(\mathbf{m}_{AB})$

**子任务：**

- [x] **1a. 实现任意矩形子区域的 TN**  
  `_contract(m_grid, region_Q)` 接受任意 plaquette 集合，自动确定相关边并构建因子图。

- [x] **1b. 实现环形区域（annular region）的 TN**  
  B、BC 等含洞区域直接用 plaquette 集合表示，洞内 plaquette 不加入 region_Q，边权重自然处理。

- [x] **1c. 实现 π(m_A) 修正项**  
  采用 Fourier 奇偶技巧：  
  $\Pr(m_Q, \pi_A=\pi) = \tfrac{1}{2}[\Pr(m_Q) + (-1)^\pi \Pr_\text{signed}(m_Q)]$  
  其中 $\Pr_\text{signed}$ 对 A plaquette 使用 $Q_- = Q^0 - Q^1$。

- [x] **1d. 联合采样 m_ABC**  
  每个样本 $\mathbf{e}$ 同时计算四项概率（`prob_ABC`, `prob_AB`, `prob_B_pi`, `prob_BC_pi`），共享同一 `m_grid`。

- [x] **1e. 定义标准几何（geom1）**  
  `define_geometry_geom1(L, r)` 实现：A 中心 2×2，B 宽度 r 环，C 宽度 r 环。

**验收标准：** 在 $p = 0.1, r = 2, L = 10$ 下跑出的 CMI 值与论文图 3(b) 数量级一致。

---

## 🟠 P2：缓存 opt_einsum 收缩路径

**目标：** 避免每个样本重复寻优收缩路径，预期加速 10–100x。

**子任务：**

- [x] **2a. 用占位张量预计算路径**  
  `CachedContractor.__init__` 用 Q0 占位张量构建网络，调用 `oe.contract_path(..., optimize='greedy')` 预计算路径。

- [x] **2b. 每个样本复用路径**  
  `CachedContractor.contract(m_grid)` 只更新 Q 张量值，复用缓存路径调用 `oe.contract(..., optimize=self._path)`。

- [x] **2c. 对不同区域分别缓存路径**  
  `calculate_CMI` 采样前为 ABC、AB、B、B_signed、BC、BC_signed 六个区域各建一个 `CachedContractor`。

**验收标准（原）：** L=6, num_samples=1000，≥10x 加速。  
**实测结果（L=6, r=1, N=50 samples）：**
- 旧: 11.7 ms/sample → 新: 5.8 ms/sample → **加速 2x**
- 未达 10x。原因：opt_einsum 默认 'auto' 路径搜索本身已很快（greedy），瓶颈在收缩计算本身，路径缓存节省的开销有限。
- **若需更大加速，需 P3（bMPS）替换精确收缩后端。**

---

## 🟢 P3：实现 bMPS 替代精确收缩

**目标：** 支持 $L \geq 10$ 的系统，使精度-代价可控。

**子任务：**

- [x] **3a. 调研现有库**  
  未使用 quimb，直接手动实现（更轻量、无额外依赖）。

- [x] **3b. 逐行 MPS 收缩**  
  在 `bMPS_contraction.py` 实现：
  - `make_site_tensor` / `build_site_grid`：构建含 W 吸收的 H×W 格点张量网络
  - `bMPS_contract`：逐行 MPO 作用 + SVD 截断（`max_bond`）+ log-scale 归一化防上溢
  - `bMPS_contract_region` / `bMPS_prob_with_parity`：与 `_contract` / `_prob_with_parity` 同接口
  - 数值稳定性：`_mps_normalise` 在每行 MPO 后归一化，`np.errstate` 屏蔽 Accelerate BLAS 的伪 FPU 警告

- [x] **3c. max_bond 收敛性测试**（实测验证）  
  r=2, L=10, p=0.1, max_bond=16 下：  
  bMPS 误差 = 0.000389 nats，MC 噪声 σ = 0.0068 nats  
  **bMPS 误差仅为 MC σ 的 5.7%，实质上精确。**

**实测结果（L=26, r=6, max_bond=16, N=10 samples）：**  
- 速度：0.17s/sample（目标 < 60s）✅  
- 精度：bMPS 误差远小于 MC 噪声 ✅  

**验收标准：PASS**

---

## 🟢 P4：数值稳定性改进

**目标：** 防止大 L 或极端 p 下的浮点下溢导致静默错误。

**子任务：**

- [ ] **4a. 在 log 域计算概率**  
  TN 收缩结果直接给出 $\Pr(\mathbf{m})$，对大格子此值可能极小。可改为计算 $\log \Pr(\mathbf{m})$ 的张量网络（每个张量取 log，用 log-sum-exp 收缩），或对整体 TN 做重标归一化。

- [ ] **4b. 统计并报告跳过样本数**  
  当前 `if prob_m > 0` 静默跳过零概率样本。改为：
  ```python
  skipped = 0
  if prob_m <= 0:
      skipped += 1
      continue
  ```
  在输出中报告跳过比例；若 > 5% 则发出警告。

- [ ] **4c. 使用 float64/float128 或对数域张量**  
  对于 L ≥ 15 且 p 极小/极大时，考虑使用 `np.float128` 或 `mpmath` 高精度计算。

**验收标准：** $L=15, p=0.01$ 下跳过样本比例 < 1%。

---

## 附：依赖关系

```
P1（CMI框架）
  └── P2（路径缓存）  ← 在 P1 完成后集成
        └── P3（bMPS）  ← 替换 P1+P2 中的精确收缩后端
              └── P4（数值稳定）  ← 贯穿 P3 实现
```

P2 可独立于 P1 先在现有 `calculate_H_m` 上验证加速效果，再合并进 CMI 框架。

---

## 参考

- 论文公式：Eq. (10)、Eq. (E1)–(E5)、Appendix D、F、G
- 论文几何：Appendix F（geom1/geom2）
- 目标数据：Fig. 3(b,c,d)，$p_c \approx 0.11$，$\alpha = 1.1$，$\nu = 1.8$
