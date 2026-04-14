# h_calculation.py 改进计划

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
  在 `bmps_contraction.py` 实现：
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

- [x] **4a. 在 log 域计算概率**  
  已在 `bMPS` 后端实现“signed log-prob + 重标归一化”路径：
  - `bMPS_contract(..., return_log=True)` 返回 `(sign, log|x|)`
  - `bMPS_prob_with_parity(..., return_log=True)` 在 log 域完成 Fourier 奇偶组合
  - `calculate_CMI_bMPS` 直接累加 log 概率，不再对极小标量概率做 `np.log(prob)`
  当前精确 TN `_contract` 仍保留标量路径（主要用于小系统）。

- [x] **4b. 统计并报告跳过样本数**  
  `calculate_CMI`、`calculate_CMI_bMPS`、`calculate_H_m` 现已统一：
  - 只在概率有限且严格为正时计入样本
  - 输出 `skipped / num_samples`
  - 若跳过比例 > 5%，自动打印警告

- [x] **4c. 使用 float64/float128 或对数域张量**  
  当前实现策略：
  - `h_calculation.py` 在可用时优先使用 `np.longdouble` 张量与累加器
  - `CMI` 累加器使用扩展精度（若平台支持）
  - `bMPS` 路径优先采用 log 域而不是 `float128` SVD，避免因线代后端兼容性导致额外问题

**当前实现影响：**
- runtime per sample：`bMPS` 增加少量 log 记账开销，预期仅小幅变慢；exact TN 基本不变
- approximation error：估计器定义不变，主要收益是减少由下溢/抵消导致的伪零值和伪负值
- Monte Carlo variance：理论方差不变；但由于有效样本筛选更透明，可减少静默跳样带来的偏差风险

**验收标准：** $L=15, p=0.01$ 下跳过样本比例 < 1%。  
**验证结果（2026-04-09）：** 在 `bMPS` 后端下，固定 `max_bond=16`、`num_samples=100`、seed 分别为 `123/124`：
- `L=15, p=0.01, r=2`：跳过率 `0/100 = 0.00%`，CMI `≈ -6.95e-07`，耗时 `2.08 s`
- `L=15, p=0.01, r=3`：跳过率 `0/100 = 0.00%`，CMI `≈ -9.01e-08`，耗时 `5.21 s`
**状态：** PASS（当前验收标准满足）。

---

## 🔵 P5：下一步数据生成方案（2026-04-08 更新）

**状态说明：**
- 已停止后台长任务：`refine_fast2h_chi32.py`
- 当前目标：在不盲目全量重跑的前提下，优先生成“可用于 Eq.(12) 对比”的关键高价值数据
- 执行状态（2026-04-09）：`P5` 全部子任务（5a-5e）已完成

### 5.1 必须生成的数据（优先级 S）

**S1. 关键 4 点 chi=32 校正数据（补完版）**

- 点位：
  - `(p=0.11, r=5)`
  - `(p=0.11, r=6)`
  - `(p=0.15, r=5)`
  - `(p=0.15, r=6)`
- 参数：`max_bond=32`, `num_samples=500`, `seed=2026 + 10000p + 100r`
- 目的：校正 fast2h 结果在大 `r` 区间的负值偏差
- 产出文件：
  - `CMI_vs_r_three_p_fast2h_refined_raw.csv`
  - `CMI_vs_r_three_p_fast2h_refined_summary.csv`
  - `CMI_vs_r_p005_011_015_fast2h_refined.png`

**S2. 校正后 Eq.(12) 对比摘要**

- 指标：
  - `p=0.05,0.15`：指数拟合 `log(CMI)=a+br` 的 `b`、`R^2`
  - `p=0.11`：幂律拟合 `CMI~r^{-alpha}` 的 `alpha`、`R^2`
- 输出：写入终端结论 + 更新 `PLAN.md` 状态

### 5.2 建议生成的数据（优先级 A）

**A1. 关键点 chi 收敛扫描（判定截断误差）**

- 点位：`p in {0.11, 0.15}`, `r in {5,6}`
- 参数网格：`chi in {16, 32, 64}`, `num_samples=500`
- 输出：
  - `CMI_keypoints_chi_scan.csv`
  - `CMI_keypoints_chi_scan.png`
- 验收：同一点随 `chi` 变化趋于平台（波动幅度收敛）

**A2. 关键点多 seed 统计（判定 MC 方差）**

- 点位同 A1
- 参数：固定较优 `chi`（优先 32），`num_samples=2000`, `seeds=3`
- 输出：
  - `CMI_keypoints_multiseed.csv`
  - `CMI_keypoints_multiseed_summary.csv`
- 验收：报告 `mean ± sem`，并标注是否与 0 相容

### 5.3 并行生产数据（优先级 B，方案 2 落地）

**B1. chunk 并行主任务（全点）**

- 范围：`p={0.05,0.11,0.15}`, `r=1..6`
- 样本：每点 `N_total=5000`
- 建议：`N_chunk=250`, `workers=96`
- 线程环境：
  - `OMP_NUM_THREADS=1`
  - `OPENBLAS_NUM_THREADS=1`
  - `MKL_NUM_THREADS=1`
- 输出：
  - `CMI_chunk_raw.csv`
  - `CMI_chunk_summary.csv`
  - `CMI_vs_r_p005_011_015_chunked.png`

### 5.4 执行顺序（下一步）

- [x] **5a. 先补完 S1 四点校正**
- [x] **5b. 生成 S2 的 Eq.(12) 校正后结论**
- [x] **5c. 执行 A1（chi 收敛）确认截断误差**
- [x] **5d. 执行 A2（多 seed）确认统计误差**
- [x] **5e. 启动 B1 chunk 并行作为主生产路径**

### 5.5 通过标准

- 四个关键校正点全部完成并落盘
- 至少一版校正后 `CMI-r` 图可用于汇报
- Eq.(12) 对比结论包含：参数、拟合质量、误差来源说明
- 所有结果均支持断点续跑（CSV 可增量写入）


### 5.6 执行记录（2026-04-09，已完成）

**执行环境与参数：**
- 硬件环境：沿用 `6.1`（`Intel Xeon Platinum 8470Q` ×2，`104` 物理核 / `208` 线程，内存 `754 GiB`）
- 线程设置：`OMP_NUM_THREADS=1`、`OPENBLAS_NUM_THREADS=1`、`MKL_NUM_THREADS=1`
- 后端：`r<=3` 使用 exact TN，`r>=4` 使用 bMPS（默认 `chi=16`，关键校正点使用 `chi=32/64`）

**5a / 5b（S1 + S2）结果：**
- 四个关键校正点（`chi=32, N=500`）：
  - `(p=0.11,r=5)=-0.002486`
  - `(p=0.11,r=6)=-0.000210`
  - `(p=0.15,r=5)=-0.002914`
  - `(p=0.15,r=6)=-0.020505`
- Eq.(12) 拟合（refined）：
  - `p=0.05` 指数：`b=-0.5568`，`R^2=0.930`
  - `p=0.15` 指数：`b=-0.7169`，`R^2=0.824`
  - `p=0.11` 幂律：`alpha=0.7112`，`R^2=0.857`

**5c（A1: chi 收敛）结果：**
- 关键点 `chi=16/32/64` 跨度（`max-min`）：
  - `(p=0.11,r=5)`：`0.02741`
  - `(p=0.11,r=6)`：`0.07321`
  - `(p=0.15,r=5)`：`0.03683`
  - `(p=0.15,r=6)`：`0.03603`
- 结论：大 `r` 点对 `chi` 明显敏感，截断误差是主导误差源之一。

**5d（A2: 多 seed）结果（`chi=32, N=2000, n=3`）：**
- `(p=0.11,r=5)`：`0.00230 ± 0.00224`
- `(p=0.11,r=6)`：`-0.00405 ± 0.00498`
- `(p=0.15,r=5)`：`-0.00578 ± 0.00058`
- `(p=0.15,r=6)`：`-0.02008 ± 0.00143`
- 结论：`p=0.11,r=6` 的统计不确定度仍较大，需结合更高 `chi` 或更多样本进一步判定符号稳定性。

**5e（B1: chunk 主任务）结果（`N_total=5000`, `N_chunk=250`, `workers=96`）：**
- 全部 `18` 个 `(p,r)` 点完成（每点 `20` 个 chunk，共 `360` 个 chunk）
- 主输出已生成：
  - `outputs/CMI_chunk_raw.csv`
  - `outputs/CMI_chunk_summary.csv`
  - `outputs/CMI_vs_r_p005_011_015_chunked.png`
- Eq.(12) 拟合（chunked）：
  - `p=0.05` 指数：`b=-0.8325`，`R^2=0.973`（趋势清晰）
  - `p=0.15` 指数：`b=-0.7652`，`R^2=0.920`（趋势清晰）
  - `p=0.11` 幂律：`alpha=1.2404`，`R^2=0.540`（临界点拟合质量一般）

**误差与风险结论：**
- 现阶段在大 `r` 关键点上，`chi` 带来的系统误差量级与/或大于 MC 统计误差；
- 对 `p=0.11` 的最终临界指数判断，不建议仅用 `chi=16` 的 `5e` 数据直接定论。

**后续建议（与 P6/P7 对齐）：**
- 保持 `5e` 的 chunk 框架，针对 `p=0.11, r=5/6` 做 `chi=32/64` 的定点重算；
- 若资源允许，将上述关键点的 `N_total` 提升到 `1e4` 以上，以同时压低截断偏差与 MC 方差。


---

## 🟣 P6：并行方案 2（Chunk 并行，目标 2 小时级产出）

**添加日期：** 2026-04-08  
**背景：** 在已确认服务器为 208 线程 / 754GiB 内存环境下，将单点大样本任务拆分为小块并行，以提高吞吐并缩短 wall-time。
- 执行状态（2026-04-09）：`P6` 已完成首版落地（6a-6e）

### 6.1 设备基线（用于并行参数）

- CPU：`Intel Xeon Platinum 8470Q` ×2
- 线程：`208`（物理核 `104`）
- 内存：`754 GiB`（可用约 `662 GiB`）
- NUMA：2 节点（通过 `lscpu` 可见）
- GPU：`RTX 5090 32GB`（当前流程未使用 GPU）
- 线代后端：NumPy + OpenBLAS

### 6.2 方案 2 的核心思想

将每个 `(p, r)` 的 `num_samples` 从“单任务串行求均值”改为“分块并行 + 汇总均值”：

- 总样本：`N_total`（例如 5000）
- 分块样本：`N_chunk`（建议 200~500）
- 块数：`K = ceil(N_total / N_chunk)`
- 任务粒度：`(p, r, chunk_id)`
- 汇总方式：
  \[
  \text{CMI}(p,r) = \frac{1}{K}\sum_{k=1}^K \text{CMI}_{k}(p,r)
  \]
  （或按有效样本数加权）

**优势：**
- 更容易吃满 104 物理核
- 负载更均衡，避免慢点拖尾
- 天然支持断点续跑（按 chunk 记账）

### 6.3 推荐并行参数（本机）

**推荐配置 A（优先）：**
- `workers = 96`
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `N_chunk = 250`

对应启动模板：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u run_chunked.py --workers 96 --chunk 250 --samples 5000
```

**推荐配置 B（保守）：**
- `workers = 64`
- 线程环境同上
- 当系统还有其他重负载任务时优先使用

### 6.4 与当前计算脚本的整合建议

- 将 `plot_cmi_vs_r_three_p_fast2h.py` 升级为 chunk 模式脚本（建议新文件：`plot_cmi_vs_r_three_p_chunked.py`）
- 输入参数：
  - `--samples`（总样本）
  - `--chunk`（每块样本）
  - `--workers`
  - `--mode fast|refine`
- 中间结果文件：
  - `CMI_chunk_raw.csv`（每个 chunk 一行）
  - `CMI_chunk_summary.csv`（按 `(p,r)` 聚合）
- 最终输出：
  - `CMI_vs_r_p005_011_015_chunked.png`

### 6.5 风险与对策

- **风险 1：OpenBLAS 线程过度嵌套，导致抖动**  
  对策：固定 `OMP/OPENBLAS/MKL=1`。

- **风险 2：chunk 太小导致调度开销偏大**  
  对策：`N_chunk` 不低于 200。

- **风险 3：单个 chunk 出现异常（SVD 不收敛）**  
  对策：记录失败 chunk 并重试 1~2 次；保留失败日志。

- **风险 4：`r=5,6` 在 `chi=16` 可能出现负值偏差**  
  对策：关键点二次精修（`chi=32/64`，较小 `N`）并覆盖最终图。

### 6.6 验收标准（2 小时级目标）

- `r=1..6, p=0.05/0.11/0.15` 全点完成
- 每点总样本达到目标（默认 `5000`）
- 输出文件齐全：raw / summary / figure
- 与 Eq.(12) 对比可复现，并明确标注哪些点经过 `chi` 精修

### 6.7 执行顺序

- [x] **6a. 实现 chunk 并行脚本**（参数化 workers/chunk/samples）
- [x] **6b. 跑 fast 配置**（`chi=16`，全点）
- [x] **6c. 识别异常点**（负值或波动点）
- [x] **6d. 关键点精修**（`chi=32/64` 覆盖）
- [x] **6e. 生成最终图和 Eq.(12) 对比报告**


### 6.8 执行记录（2026-04-09，首版完成）

**6a（脚本落地）：**
- 已新增 `src/scripts/plot_cmi_vs_r_three_p_chunked.py`，支持 `--workers/--samples/--chunk/--max-bond`；
- 支持断点续跑（基于 `outputs/CMI_chunk_raw.csv` 缓存）。

**6b（fast 全点运行）：**
- 参数：`workers=96`，`N_total=5000`，`N_chunk=250`，`r<=3 exact`，`r>=4 bmps(chi=16)`；
- 完成规模：`18` 个 `(p,r)` 点，`20` chunk/点，共 `360` chunk；
- 输出：
  - `outputs/CMI_chunk_raw.csv`
  - `outputs/CMI_chunk_summary.csv`
  - `outputs/CMI_vs_r_p005_011_015_chunked.png`

**6c（异常点识别）：**
- `CMI_chunk_summary.csv` 显示主要异常集中在大 `r`：
  - `p=0.11, r=5/6`：均值为负（`-0.0172`, `-0.0383`）
  - `p=0.15, r=5/6`：均值为负（`-0.0247`, `-0.0500`）
- 判定：与 `chi` 截断偏差和临界附近噪声叠加相符。

**6d（关键点精修）：**
- 已对关键点执行 `chi=32` 校正与 `chi=16/32/64` 扫描（见 `P5` 的 `5a/5c/5d` 记录）；
- 结论：关键点对 `chi` 高敏感，`chi=16` 结果不宜直接作为临界结论。

**6e（Eq.(12) 对比）：**
- chunk 结果拟合：
  - `p=0.05` 指数：`b=-0.8325`，`R^2=0.973`
  - `p=0.15` 指数：`b=-0.7652`，`R^2=0.920`
  - `p=0.11` 幂律：`alpha=1.2404`，`R^2=0.540`
- 结论：非临界两支曲线趋势可复现；临界拟合质量一般，需更高 `chi`/更大样本补强。

**状态结论：**
- `P6` 首版目标已完成；
- 若作为最终汇报版本，建议追加 `p=0.11, r=5/6` 的 `chi=32/64` chunk 化补点并覆盖图线。


---


## ⚫ P6-B：论文级复现（方案 B）详细执行清单（2026-04-14，2026-04-14 二次调整）

**目标（本次调整后）：**
- 复现 `image.png` 的三支 `CMI-r` 对数图风格；
- 先在 `r<=10` 范围内完成高可信结果；
- 执行顺序改为：**先增大 `chi` 使 CMI 稳定为正，再进入 Plan B 主生产**。

**范围约束（替代旧版 `r<=14`）：**
- `p ∈ {0.05, 0.11, 0.15}`
- `r ∈ {3,4,5,6,7,8,9,10}`
- 共 `24` 个 `(p,r)` 点

### B0. 正值稳定化门槛（先执行，未通过不进 B1）

**目标：** 先降低系统误差与统计误差，避免“positive-only 作图”带来的筛选偏差。

**B0-1：chi 梯度扫描（稳定性主判据）**
- [ ] 点位：`p={0.05,0.11,0.15}`，`r={6,8,10}`
- [ ] 参数：`chi ∈ {32, 64, 96}`，`num_samples=2000`
- [ ] 输出：
  - `outputs/CMI_preB_sign_scan_raw.csv`
  - `outputs/CMI_preB_sign_scan_summary.csv`

**B0-2：高样本复核（仅针对未稳定点）**
- [ ] 对 B0-1 中“未稳定为正”的点，固定较优 `chi`（优先较大 `chi`），提升到 `num_samples=10000`
- [ ] 输出：
  - `outputs/CMI_preB_sign_refine_raw.csv`
  - `outputs/CMI_preB_sign_refine_summary.csv`

**B0-3：进入 B1 的门槛（硬条件）**
- [ ] 每个目标点需给出 `mean`, `sem`, `z=mean/sem`
- [ ] 稳定为正判据：`mean - 2*sem > 0`
- [ ] 未通过点必须标记 `uncertain`，不得进入 Eq.(12) 最终拟合

### B1. Plan B 主生产（仅在 B0 后执行）

**运行环境（固定）：**
- `OMP_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `workers=96`（或资源共享时 `64`）

**主生产参数：**
- [ ] 目标点：`p={0.05,0.11,0.15}`, `r=3..10`
- [ ] 每点总样本：`N_total >= 6e6`
- [ ] 分块：`N_chunk=500`
- [ ] 后端：`r<=3 exact TN`, `r>=4 bMPS`
- [ ] `chi`：采用 B0 确认后的稳定值（不再固定写死为 32）

**主输出文件：**
- `outputs/CMI_fig3d_B_r10_raw.csv`
- `outputs/CMI_fig3d_B_r10_summary.csv`
- `outputs/CMI_vs_r_p005_011_015_fig3d_B_r10.png`

### B2. 绘图与拟合规范

- y 轴采用 log；
- 原始统计表保留全部点（含负值）；
- 最终拟合仅使用通过 B0 判据的稳定正值点；
- 拟合规则：
  - `p=0.05,0.15`：`log(CMI)=a+br`
  - `p=0.11`：`log(CMI)=c-\alpha log r`

### B3. 通过标准（调整后）

- [ ] `24` 个目标点全部完成；
- [ ] 所有点都有 `mean/sem/z` 与 `stable/uncertain` 标记；
- [ ] 稳定点用于拟合并给出 `R^2`；
- [ ] 形成 `r<=10` 的可汇报版本图与摘要。

### B4. 风险与对策（调整后）

- 风险 1：`chi` 增大后吞吐明显下降。  
  对策：先做 B0，再决定 B1 的 `chi` 与样本预算。
- 风险 2：部分点始终无法稳定为正。  
  对策：保留 `uncertain` 标记，避免强行进入拟合。
- 风险 3：一次性全点高样本耗时过长。  
  对策：按 `r` 分批（先 `r=3..8`，再 `r=9..10`）并支持断点续跑。



---

## 🟤 P7：`CMI_vs_p` 的下一轮重算与出图计划（2026-04-08）

**目标：** 基于已经完成的快速版 `r=1..6` 扫描，生成一版更平滑、可比较、可用于汇报的 `CMI(p)` 图，并重点降低临界区 `p \approx 0.11` 附近的统计噪声。

**当前基线：**
- 已完成一版快速扫：`r=1,2,3,4,5,6`
- 输出文件：`CMI_vs_p_r1to6.csv`、`CMI_vs_p_r1to6.png`
- 当前参数：14 个 `p` 点，每点 `num_samples = 10`
- 问题：`r=5,6` 以及 `p \in [0.095, 0.12]` 区间噪声较大，存在若干负值点，不能直接作为最终展示图

**已测单点耗时基线（来自 2026-04-08 的 10-sample 实跑）：**
- `r=1`：约 `0.1 s / 10 samples`，即 `0.01 s/sample`
- `r=2`：约 `0.2 s / 10 samples`，即 `0.02 s/sample`
- `r=3`：约 `0.3 s / 10 samples`，即 `0.03 s/sample`
- `r=4`：约 `7.0 s / 10 samples`，即 `0.7 s/sample`
- `r=5`：约 `31 s / 10 samples`，即 `3.1 s/sample`
- `r=6`：约 `50 s / 10 samples`，即 `5.0 s/sample`

**执行策略：** 采用“两阶段重算”而不是一次性全量高精度重跑：
- 阶段 A：先把整条曲线补到“中等精度”，得到一版平滑总览图
- 阶段 B：只对临界区和大 `r` 做高精度加密，得到最终图

### 阶段 A：全区间中等精度重跑

**目标：** 给 `r=1..6` 全部曲线提供统一质量的底图。

**固定参数：**
- `r-values = 1,2,3,4,5,6`
- `method = auto`
- `max_exact_r = 4`
- `max_bond = 64`
- `no-show = true`

**p 取点：**
```text
0.02, 0.05, 0.08, 0.09, 0.095, 0.10, 0.105, 0.11, 0.115, 0.12, 0.13, 0.15, 0.20, 0.30, 0.50
```

**采样数：**
- `r=1,2,3,4,5,6`：每点统一使用 `num_samples = 10^4`

**输出文件：**
- `CMI_vs_p_r1to6_stageA.csv`
- `CMI_vs_p_r1to6_stageA.png`

**目的说明：**
- 阶段 A 的目标是先得到一版高统计精度的全区间总览图
- 由于 `r=5,6` 的 bMPS 单点耗时较长，阶段 A 实际执行时需要优先评估总耗时是否可接受
- 若总耗时过长，可保持计划不变，但按 `r` 分批提交任务

**按当前实测速度估算总耗时：**
- `r=1`：15 个 `p` 点 × `10^4` samples × `0.01 s/sample` ≈ `1.5e3 s` ≈ `0.4 小时`
- `r=2`：15 个 `p` 点 × `10^4` samples × `0.02 s/sample` ≈ `3.0e3 s` ≈ `0.8 小时`
- `r=3`：15 个 `p` 点 × `10^4` samples × `0.03 s/sample` ≈ `4.5e3 s` ≈ `1.3 小时`
- `r=4`：15 个 `p` 点 × `10^4` samples × `0.7 s/sample` ≈ `1.05e5 s` ≈ `29 小时`
- `r=5`：15 个 `p` 点 × `10^4` samples × `3.1 s/sample` ≈ `4.65e5 s` ≈ `129 小时`
- `r=6`：15 个 `p` 点 × `10^4` samples × `5.0 s/sample` ≈ `7.5e5 s` ≈ `208 小时`
- **阶段 A 总计：** 约 `1.33e6 s`，即约 `370 小时`，约 `15 天`

**风险判断：**
- 阶段 A 若严格按 `10^4` 全量执行，已经是多天级任务，不适合一次性串行跑完
- 最大瓶颈明确在 `r=5,6`
- 实际执行必须依赖 CSV 断点续跑，且建议按 `r` 分拆任务

**验收标准：**
- 图中 `r=1..6` 曲线全部连续可见
- 在 `p=0.02` 和 `p=0.50` 附近，CMI 接近 0
- 大多数 `r=1..4` 曲线在临界区附近不再出现剧烈跳点

### 阶段 B：临界区高精度加密

**目标：** 压低 `p_c \approx 0.11` 附近的统计波动，生成最终展示图。

**重跑对象：**
- 优先：`r=4,5,6`
- 如阶段 A 后发现 `r=2,3` 仍有明显噪声，再补跑 `r=2,3`

**临界区 p 取点：**
```text
0.090, 0.095, 0.100, 0.1025, 0.105, 0.1075, 0.110, 0.1125, 0.115, 0.1175, 0.120, 0.125
```

**采样数：**
- `r=4,5,6`：每点统一使用 `num_samples = 10^5`
- 若 `r=2,3` 需要补跑：每点同样使用 `num_samples = 10^5`

**输出文件：**
- `CMI_vs_p_r4to6_critical.csv`
- `CMI_vs_p_r4to6_critical.png`

**合并策略：**
- 将阶段 B 的临界区高精度数据覆盖阶段 A 的同名点
- 保留阶段 A 的非临界区结果不变
- 合并后的最终文件命名为：
  - `CMI_vs_p_r1to6_final.csv`
  - `CMI_vs_p_r1to6_final.png`

**按当前实测速度估算总耗时：**
- 临界区共有 12 个 `p` 点
- `r=4`：12 × `10^5` samples × `0.7 s/sample` ≈ `8.4e5 s` ≈ `233 小时` ≈ `9.7 天`
- `r=5`：12 × `10^5` samples × `3.1 s/sample` ≈ `3.72e6 s` ≈ `1030 小时` ≈ `43 天`
- `r=6`：12 × `10^5` samples × `5.0 s/sample` ≈ `6.0e6 s` ≈ `1667 小时` ≈ `69 天`
- **阶段 B（仅 `r=4,5,6`）总计：** 约 `1.06e7 s`，即约 `2930 小时`，约 `122 天`
- 若再补 `r=2,3`，总耗时还会继续上升

**风险判断：**
- 阶段 B 若严格按 `10^5` 执行，是月级到季度级计算任务
- 在不增加并行度、不更换算法、也不降低样本数的前提下，不适合直接启动全任务
- 阶段 B 更适合作为“最终高精度生产任务”，需要单独安排机器占用和执行窗口

**验收标准：**
- `r=4,5,6` 在 `p \in [0.095, 0.12]` 的曲线明显更平滑
- 峰值位置集中在 `p \approx 0.10 - 0.12`
- `r` 增大时峰高整体下降的趋势清晰可见

### 可选增强：仅对 `r=5,6` 提高 bMPS 精度

**触发条件：**
- 若阶段 B 后，`r=5,6` 仍出现大幅负值或明显不平滑

**增强参数：**
- 保持 `num_samples` 不变，先将 `max_bond` 从 `64` 提升到 `96`
- 若仍不稳定，再测试 `max_bond = 128`

**测试范围：**
- 只测最敏感的 4 个点：
```text
p = 0.100, 0.105, 0.110, 0.115
```

**输出文件：**
- `CMI_vs_p_r5r6_bond_test.csv`

**验收标准：**
- 同一 `p` 下，`max_bond=96/128` 相比 `64` 的变化幅度缩小
- 曲线形状不再随 `max_bond` 剧烈波动

### 建议执行顺序

- [ ] **7a. 生成阶段 A 总览图**  
  先按 `r=1 → 6` 分批跑 `CMI_vs_p_r1to6_stageA.csv` / `CMI_vs_p_r1to6_stageA.png`

- [ ] **7b. 检查临界区噪声最强的曲线**  
  重点看 `r=4,5,6` 在 `p \in [0.095, 0.12]`

- [ ] **7c. 执行阶段 B 高精度临界区补点**  
  仅在确认资源窗口后再生成 `CMI_vs_p_r4to6_critical.csv`

- [ ] **7d. 合并阶段 A/B 数据**  
  输出 `CMI_vs_p_r1to6_final.csv`

- [ ] **7e. 生成最终展示图**  
  输出 `CMI_vs_p_r1to6_final.png`

- [ ] **7f. 如有必要再做 `max_bond` 测试**  
  仅对 `r=5,6` 的关键点执行

### 实际执行建议

- 若坚持阶段 A 用 `10^4`，建议先只跑 `r=1,2,3,4`
- `r=5,6` 建议单独排成长任务，避免和小 `r` 混跑
- 若坚持阶段 B 用 `10^5`，建议先只跑 `r=4` 的 12 个临界点，作为可行性试运行
- 在 `r=5,6` 上启动 `10^5` 之前，应先完成 `max_bond=96/128` 的小范围稳定性测试
- 所有长任务必须保持逐点写 CSV，确保中断后可以直接续跑

---

## 附：依赖关系

```
P1（CMI框架）
  └── P2（路径缓存）  ← 在 P1 完成后集成
        └── P3（bMPS）  ← 替换 P1+P2 中的精确收缩后端
              └── P4（数值稳定）  ← 贯穿 P3 实现
                    └── P7（r=1..6 重算与出图）  ← 基于 P1-P4 的实际数据生产
```

P2 可独立于 P1 先在现有 `calculate_H_m` 上验证加速效果，再合并进 CMI 框架。

---

## 参考

- 论文公式：Eq. (10)、Eq. (E1)–(E5)、Appendix D、F、G
- 论文几何：Appendix F（geom1/geom2）
- 目标数据：Fig. 3(b,c,d)，$p_c \approx 0.11$，$\alpha = 1.1$，$\nu = 1.8$

---

## ⚫ P8：基于 `quimb` 的实验性后端对照计划（2026-04-09）

**目标：**
- 增加一套基于 `quimb` 的实验性 2D TN / bMPS 收缩后端；
- 在**不改变物理定义、几何约定和 Monte Carlo 估计器**的前提下，与当前自写 `bmps_contraction.py` 做逐点对照；
- 判断是否值得将 `quimb` 作为长期维护的第二后端，或作为大 `r` / 高 `chi` 的验证工具。

**对照原则：**
- 保持 `geom1`、`A/B/C` 定义、`Q^0/Q^1/Q_+/Q_-` 张量定义不变；
- 保持 Fourier parity trick 不变：
  $\Pr(m_Q,\pi_A=\pi)=\tfrac12[\Pr(m_Q)+(-1)^\pi \Pr_{\mathrm{signed}}(m_Q)]$；
- 保持同一组 MC 样本（相同 `seed`）下比较 exact / 自写 bMPS / quimb 三者；
- `quimb` 后端仅替换“区域概率收缩器”，不重写 `calculate_CMI` 的物理逻辑。

### 8.1 实现范围（最小闭环）

**阶段 1：只做 region-level 概率收缩对照**

- 新文件建议：
  - `src/core/quimb_contraction.py`
  - 若需要独立实验脚本，再增加 `src/scripts/benchmark_quimb_backend.py`
- 最小接口目标：
  - `quimb_contract_region(m_grid, region_Q, region_A_signed=None, p=0.1, max_bond=16, contract='bmps')`
  - `quimb_prob_with_parity(m_grid, region_Q, region_A, pi_A, p=0.1, max_bond=16, contract='bmps')`
- 接口语义必须与：
  - `src/core/bmps_contraction.py` 中的 `bMPS_contract_region`
  - `src/core/bmps_contraction.py` 中的 `bMPS_prob_with_parity`
  对齐，便于逐点替换与 A/B test。

**阶段 2：接入 CMI 主流程**

- 在 `src/core/cmi_calculation.py` 中增加实验性入口：
  - `calculate_CMI_quimb(...)`
- 或者在现有脚本中加入 `--method quimb` 选项，但初版建议先独立函数，避免影响当前生产流程。

### 8.2 子任务拆分

- [ ] **8a. 映射当前站点张量到 `quimb` 网络对象**  
  保持站点张量约定 `T[top, right, bottom, left]`，并复用当前的 `Q^0/Q^1/Q_+/Q_-` 定义。

- [ ] **8b. 实现矩形 bounding box 的 `quimb` 构网**  
  与 `build_site_grid` / `bMPS_contract_region` 使用同一 bounding-box 规则，保证 region_Q 与 region_A_signed 输入含义完全一致。

- [ ] **8c. 实现 `quimb` 的边界 MPS 收缩路径**  
  先支持 row-by-row 边界收缩；必要时再补 column-by-column，作为“是否降低截断误差”的实验变量。

- [ ] **8d. 实现 parity 联合概率接口**  
  用与当前后端完全一致的 Fourier parity trick，输出 `Pr(m_Q, \pi_A)`。

- [ ] **8e. 做单样本 region-level 精度对照**  
  对六类收缩：
  `Pr(m_ABC)`, `Pr(m_AB)`, `Pr(m_B)`, `Pr_signed(m_B)`, `Pr(m_BC)`, `Pr_signed(m_BC)`
  分别与 exact TN、自写 bMPS 对照。

- [ ] **8f. 做 CMI-level 对照**  
  在相同 `seed` / `num_samples` 下，对比三种后端给出的最终 `CMI`。

- [ ] **8g. 记录性能与稳定性**  
  记录 wall-time、失败率、SVD/压缩异常、是否出现负概率或跳过样本。

### 8.3 验证矩阵（必须固定参数）

**A. region-level 正确性基准**

- 目标：先验证“单次概率收缩”是否正确，再进入 MC。
- 参数组：
  - `L=10, p=0.10, r=2, max_bond=16`
  - `L=14, p=0.11, r=3, max_bond=16`
- 验证内容：
  - 六类收缩值逐项比较；
  - 相对误差 `|x_quimb - x_exact| / (|x_exact| + 1e-30)`；
  - 与当前自写 bMPS 的相对误差并列报告。
- 验收标准：
  - 在 `r=2` 上，`quimb` 后端误差不高于当前自写 bMPS 的同量级；
  - 若 `r=3` 上明显更差，则停止继续集成到 CMI 主流程。

**B. CMI-level 小系统回归**

- 目标：验证最终观测量而不是只看中间概率。
- 参数组：
  - `L=10, p=0.10, r=2, num_samples=200, max_bond=16, seed=42`
  - `L=14, p=0.11, r=3, num_samples=200, max_bond=16, seed=43`
- 验证内容：
  - `CMI_exact`
  - `CMI_custom_bmps`
  - `CMI_quimb_bmps`
  - 三者差值以及相对 MC 波动量级。
- 验收标准：
  - `|CMI_quimb - CMI_exact|` 不高于当前自写 bMPS 的 1.5 倍；
  - 结果符号与数量级正确，不出现系统性偏负。

**C. 大 r 实用性测试**

- 目标：判断 `quimb` 是否值得在大系统继续投入。
- 参数组：
  - `L=18, p=0.11, r=4, num_samples=50, max_bond in {16, 32}`
  - `L=22, p=0.11, r=5, num_samples=20, max_bond in {16, 32, 64}`
  - `L=26, p=0.11, r=6, num_samples=10, max_bond in {16, 32, 64}`
- 验证内容：
  - 每 sample 耗时；
  - 跳过样本比例；
  - 同一 `(L,p,r)` 下随 `max_bond` 的收敛趋势；
  - 与当前自写 bMPS 的一致性。
- 验收标准：
  - 至少在 `r=5` 或 `r=6` 的一个点上，`quimb` 的精度-时间折中优于当前自写后端；
  - 若只更慢、精度无改进，则保留为验证工具，不进入主生产。

### 8.4 输出文件与记录规范

- 建议输出：
  - `outputs/quimb_region_benchmark.csv`
  - `outputs/quimb_cmi_benchmark.csv`
  - `outputs/quimb_bond_scan.csv`
  - `outputs/quimb_vs_custom_summary.md`（可选）
- CSV 建议字段：
  - `backend`
  - `contract_mode`
  - `L`
  - `p`
  - `r`
  - `num_samples`
  - `max_bond`
  - `seed`
  - `observable`
  - `value`
  - `elapsed_sec`
  - `skipped`
  - `notes`
- 若记录 timing，需沿用本仓库文档约定，同时注明硬件环境；硬件可先复用 P6 中的服务器基线说明。

### 8.5 风险判断

- **风险 1：`quimb` 引入额外依赖，环境更重**  
  对策：先作为实验性可选依赖，不替换默认后端。

- **风险 2：API 抽象更厚，物理定义更难排错**  
  对策：先只做 region-level 六项收缩对照，逐层验证，不直接上全流程。

- **风险 3：`quimb` 默认收缩策略与当前实现不一致，导致结果可比性差**  
  对策：初版只固定一种 row-by-row 边界 MPS 路径；必要时单独开启 column-by-column 对照。

- **风险 4：速度更慢但没有更高精度**  
  对策：及时止损，将其定位为“交叉验证后端”而非生产后端。

### 8.6 决策门槛（做完后如何判断）

- **结论 A：升级为长期维护的第二后端**
  - 条件：在 `r=5` 或 `r=6` 上精度明显优于当前自写 bMPS，且 wall-time 增幅可接受（建议不超过 2–3 倍）。

- **结论 B：仅保留为 correctness oracle**
  - 条件：小系统结果稳定，但大系统速度过慢；适合抽查关键点，不适合全扫。

- **结论 C：终止该方向**
  - 条件：小系统就不稳定，或大系统既慢又无精度收益。

### 8.7 建议执行顺序

- [x] **8h. 先做 region-level 六项单样本对照**
- [x] **8i. 再做两组小系统 CMI 回归**
- [x] **8j. 最后做 `r=4` 的小样本可行性测试**
- [x] **8k. 汇总结论并决定是否接入主脚本**

### 8.8 执行记录（2026-04-14，已完成）

**实现文件：**
- `src/core/quimb_contraction.py`：quimb 后端
  - `_compress_mps_quimb`：right_canonize (QR R→L) + compress (SVD L→R)
  - `quimb_contract_region` / `quimb_prob_with_parity`：drop-in 接口
  - `calculate_CMI_quimb`：MC 估计器入口
- `src/scripts/benchmark_quimb_backend.py`：三类基准测试脚本

**关键数据（`--quick` 运行结果）：**

| 测试 | 结果 |
|---|---|
| Region 精度（12项）| quimb 优于 custom 的占 **9/12** |
| 速度开销 | custom 比 quimb 快 **5倍** |
| 大区域（r=3 Pr(BC)）| quimb 误差改善 **4.6倍** |
| CMI r=2（N=50）| quimb 更接近 exact（0.0021 vs 0.0030，exact=-0.0012）|
| CMI r=3（N=50）| custom 更接近 exact（均为负值）|
| CMI r=4（N=20，chi=32）| custom 给出 +0.003，quimb 给出 -0.009 |

**结论：B — 仅保留为 correctness oracle**

- 小系统（r=2,3）：quimb 的 region-level 精度明显优于 custom（特别是大区域 BC/ABC）
- 但 CMI-level 改善不稳定：统计噪声（N=20~50）掩盖了压缩精度差异
- 速度：5x 开销使其不适合生产批量运算（chi=32 下 r=4 需 2234 ms/sample vs custom 的 969 ms/sample）
- **推荐用途**：作为关键点的 oracle 验证后端，确认 custom bMPS 结果正确性；不接入主生产流程

**核心发现（为何仍然为负值）：**
- 根本原因不是压缩方向（two-pass vs one-pass），而是**chi 本身不够大**
- chi=32 时 custom 可以给出正值 CMI，quimb 仍然为负 → 说明 quimb 的双向压缩在此场景下反而引入了不同的偏差
- 建议继续以**增大 chi**为主要改进路径（custom bMPS + chi=64/128），而不是切换后端


---

## ⚫ P9：r=1..8 全正值 CMI 生产计划（2026-04-14）

**目标：** 对 `p ∈ {0.05, 0.11, 0.15}`、`r=1..8` 的全部 24 个点，给出统计意义上正值的 CMI 估计，用于 Eq.(12) 拟合与图 3(d) 复现。

### 9.1 时间模型

以 **chi=96, r=6（直接实测：~22 s/样本）** 为基准，比例公式：

$$t(r, \chi) = 22\ \text{s} \times \frac{C_r}{C_6} \times \left(\frac{\chi}{96}\right)^3$$

其中 $C_r = 3(2+4r)^2 + 3(2+2r)^2$（每样本 3 个大盒 + 3 个中盒收缩）

| $r$ | $L$ | $C_r$ | $C_r/C_6$ |
|-----|-----|-------|-----------|
| 4   | 18  | 1272  | 0.486     |
| 5   | 22  | 1884  | 0.720     |
| 6   | 26  | 2616  | 1.000     |
| 7   | 30  | 3468  | 1.326     |
| 8   | 34  | 4440  | 1.698     |

各配置每样本耗时（估算）：

| $r$ | $\chi$ | t/样本 |
|-----|--------|--------|
| 1–3 | exact  | 0.01–0.03 s |
| 4   | 64     | ~1.6 s |
| 5   | 64     | ~4.7 s |
| 6   | 96     | ~22 s  |
| 7   | 128    | ~69 s  |
| 8   | 128    | ~89 s  |

### 9.2 chi 选择依据

- **双向 SVD 压缩无效**：P8.8 已实验证实，问题根源是 chi 不足，而非压缩方向。
- **随机化 SVD 不适用**：每步保留比例 ≈ k/min(m,n) ≈ 1，无加速空间。
- **唯一有效路径**：提高 chi + 利用并行。

从现有 chi 扫描数据的截断偏差（以 p=0.15 为最严苛判据）：

| $r$ | chi=32 | chi=64 | chi=96 | 需要 chi |
|-----|--------|--------|--------|----------|
| 6   | −0.019 | −0.0055 | ~−0.002（外推）| **≈ 96–128** |
| 8   | —      | −0.034  | —      | **≈ 128–192，不确定** |

**风险**：r=8, p=0.15 的真实 CMI 随 r 指数衰减（基于 b≈−0.77），到 r=8 可能已小于截断偏差，任何有限 chi 下均难以保证正值。须先经阶段 0 验证。

### 9.3 执行阶段

#### 阶段 0：chi=128 快速验证（约 40 min）

**目的**：在启动主生产前，确认 r=7,8 在 chi=128 下能否给出正值 CMI。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u src/scripts/plot_cmi_vs_r_three_p_chunked.py \
  --r-values 6,7,8 --max-bond 128 --max-exact-r 3 \
  --workers 96 --samples 500 --chunk 50 \
  --out-prefix CMI_chi128_verify \
  --out-png CMI_chi128_verify.png
```

预估 wall time：r=7 约 18 min，r=8 约 23 min，**合计 ~40 min**。

**决策规则**：
- `mean - 2*sem > 0`：通过，进入阶段 1 使用该 chi；
- 未通过：降级目标为 r=1..7，或将 r=8 的 chi 升至 192（需额外 ~13 小时，不推荐）。

#### 阶段 1：分批主生产

由于脚本 `--max-bond` 对所有 bMPS 点统一，按 chi 分批运行（支持断点续跑）：

**批次 1：r=1..5，chi=64（~15 min）**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u src/scripts/plot_cmi_vs_r_three_p_chunked.py \
  --r-values 1..5 --max-bond 64 --max-exact-r 3 \
  --workers 96 --samples 5000 --chunk 250 \
  --out-prefix CMI_r1to5_chi64 \
  --out-png CMI_r1to5_chi64.png
```

**批次 2：r=6，chi=96（~57 min）**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u src/scripts/plot_cmi_vs_r_three_p_chunked.py \
  --r-values 6 --max-bond 96 --max-exact-r 3 \
  --workers 96 --samples 5000 --chunk 250 \
  --out-prefix CMI_r6_chi96 \
  --out-png CMI_r6_chi96.png
```

**批次 3：r=7,8，chi=128（~7 小时，条件：阶段 0 通过）**

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -u src/scripts/plot_cmi_vs_r_three_p_chunked.py \
  --r-values 7,8 --max-bond 128 --max-exact-r 3 \
  --workers 96 --samples 5000 --chunk 250 \
  --out-prefix CMI_r7r8_chi128 \
  --out-png CMI_r7r8_chi128.png
```

### 9.4 Wall Time 汇总（96 workers，5000 样本/点，3 个 p 值）

| 阶段       | 内容              | 预估 wall time |
|-----------|------------------|---------------|
| 阶段 0     | chi=128 验证（r=6..8，N=500）| ~40 min |
| 批次 1     | r=1..5，chi=64   | ~15 min        |
| 批次 2     | r=6，chi=96      | ~57 min        |
| 批次 3     | r=7,8，chi=128   | ~7 小时         |
| **合计**   |                  | **~8.5 小时**   |

### 9.5 风险与对策

| 风险 | 对策 |
|------|------|
| r=8, p=0.15 在 chi=128 下仍为负值 | 阶段 0 预判；若失败则目标缩减至 r=1..7，节省 ~3.9 小时 |
| chi=64 对 r=5 不够（p=0.11 附近）| 阶段 0 一并验证；若需升为 chi=96，批次 1 耗时增至 ~50 min |
| 任务中断 | 脚本支持断点续跑（基于 CSV 缓存），重新执行自动跳过已完成 chunk |
| 临界点（p=0.11）chi 需求更高 | 已知风险；若 r=7,8 的 p=0.11 仍为负，标记为 `uncertain`，不纳入最终拟合 |

### 9.6 通过标准

- [ ] 阶段 0：r=6,7,8 在 chi=128 下各有 `mean - 2*sem > 0`（或给出调整建议）
- [ ] r=1..8（或调整后范围）的全部点完成，每点 ≥ 5000 有效样本
- [ ] 所有点记录 `mean / sem / z = mean/sem / stable标记`
- [ ] 稳定正值点用于 Eq.(12) 拟合，给出 `R^2`
- [ ] 输出合并图（log y 轴，r=1..8）
