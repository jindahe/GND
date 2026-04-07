# H_calculation.py 改进计划

基于对现有代码的深度分析，按优先级列出后续改进任务。

---

## 🔴 P1：实现完整 CMI 公式（环形区域 + π(m_A) 修正）

**目标：** 能复现论文图 3，计算 $I(A:C|B) = H(\mathbf{m}_{BC}, \pi(\mathbf{m}_A)) - H(\mathbf{m}_{ABC}) - H(\mathbf{m}_B, \pi(\mathbf{m}_A)) + H(\mathbf{m}_{AB})$

**子任务：**

- [ ] **1a. 实现任意矩形子区域的 TN**  
  当前代码只处理全 $L \times L$ 格子。需要支持传入任意边集合（定义一个区域 $Q$），构建对应的因子图。

- [ ] **1b. 实现环形区域（annular region）的 TN**  
  区域 B（A 外围宽 $r$ 的环）和 BC（B+C 的联合）都是含洞区域。需要：
  - 确定环形区域包含哪些 plaquette 和边
  - 将洞内 plaquette 的 Q 张量排除，但保留洞边界上的边权重

- [ ] **1c. 实现 π(m_A) 修正项**  
  对含洞区域（B、BC），Eq. (9) 修正为 $H(\mathbf{m}_Q, \pi(\mathbf{m}_A))$。需要：
  - 在蒙特卡洛采样时，同时记录 A 内的 anyon 配置
  - 计算 $\pi(\mathbf{m}_A) = \sum_{i \in A} m_i \bmod 2$
  - 将 $\pi(\mathbf{m}_A)$ 作为额外维度加入 TN（扩展最外层 Q 张量的约束）

- [ ] **1d. 联合采样 m_ABC**  
  四个熵项需在同一个样本上计算（共用同一个错误配置 $\mathbf{e}$），然后分别边缘化到各子区域。

- [ ] **1e. 定义标准几何（geom1）**  
  实现论文 Appendix F 的分区：
  - A：中心固定 $2 \times 2$ 区域
  - B：围绕 A 的宽度 $r$ 环形
  - C：围绕 B 的宽度 $r$ 环形（geom1：$r_C = r_B = r$）

**验收标准：** 在 $p = 0.1, r = 2, L = 10$ 下跑出的 CMI 值与论文图 3(b) 数量级一致。

---

## 🟠 P2：缓存 opt_einsum 收缩路径

**目标：** 避免每个样本重复寻优收缩路径，预期加速 10–100x。

**子任务：**

- [ ] **2a. 用占位张量预计算路径**  
  收缩路径只依赖于网络结构（L 和区域形状），不依赖于具体 m_grid 值。做法：
  ```python
  # 用随机占位张量构建一次网络，只计算路径
  path, info = oe.contract_path(*tensors_and_indices, optimize='optimal')
  ```

- [ ] **2b. 每个样本复用路径**  
  每次循环只替换 Q 张量（Q0/Q1 按新的 m_grid 选取），传入预计算的 `path`：
  ```python
  prob_m = oe.contract(*tensors_and_indices, optimize=path)
  ```

- [ ] **2c. 对不同区域分别缓存路径**  
  P1 完成后，AB、B、BC、ABC 四个区域各有固定结构，分别缓存一条路径。

**验收标准：** L=6 下 num_samples=1000 的运行时间比原始版本缩短 10x 以上。

---

## 🟡 P3：实现 bMPS 替代精确收缩

**目标：** 支持 $L \geq 10$ 的系统，使精度-代价可控。

**子任务：**

- [ ] **3a. 调研现有库**  
  检查 `quimb` 是否支持 bMPS 收缩（`quimb.tensor` 中的 `TensorNetwork2D` 和 `contract_boundary`）。若支持，直接调用；否则手动实现。

- [ ] **3b. 逐行 MPS 收缩**  
  将 2D TN 按行展开为 MPS，逐行向上收缩：
  - 每行的张量乘积形成一个 MPO
  - 当前 MPS 与 MPO 相乘后做 SVD 截断（保留 `max_bond` 奇异值）
  - 重复直至所有行收缩完毕

- [ ] **3c. max_bond 收敛性测试**  
  对 $L=10, p=0.1$，扫描 `max_bond` ∈ {4, 8, 16, 32, 64}，验证 $H(m)$ 随 `max_bond` 收敛。

**验收标准：** $L=20$ 下 bMPS（`max_bond=16`）能在合理时间内（< 1分钟/样本）给出与精确收缩一致（相对误差 < 1%）的结果。

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
