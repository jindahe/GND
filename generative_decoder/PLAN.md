这是一套针对计算 **$L \times L$ Toric Code 在 GND（生成式神经解码器）模型下的二分互信息 (Bipartite MI)** 的执行计划。

本计划按“**先做最小正确版本，再逐步扩展**”组织，优先保证物理定义清楚、实现路径可落地、与当前仓库结构兼容。

---

# 🚀 Toric Code 二分互信息计算执行计划

## 一、总目标
通过计算 GND 学到的分布在空间切割下的二分经典互信息，判断模型是否捕捉到了拓扑码中的长程约束。

当前建议将任务拆成两个层级：

1. **最小版本（第一优先级）**：计算 syndrome 变量的二分经典互信息  
   $$I(\gamma_A;\gamma_B)$$
2. **完整版本（第二优先级）**：计算联合分布  
   $$I(A;B),\quad A,B \subset (\beta,\gamma)$$

其中：
* $\gamma$ 表示 syndrome / stabilizer measurement bits；
* $\beta$ 表示 logical-sector bits。

---

## 二、P0 的计算对象与基本立场

P0 的目标必须先明确：

> **P0 主要计算的是模型分布 `q` 的二分互信息，不是直接去计算物理分布 `p` 的二分互信息。**

更具体地说：

1. **主要计算对象是 `q`**  
   我们要利用 GPU 训练和评估自回归模型，得到模型分布
   $$q_\theta(\gamma)$$
   或更严格地说，得到按空间顺序重排后的
   $$q_\theta(\gamma_A,\gamma_B).$$

2. **`p` 只承担数据来源角色**  
   物理误差模型或采样器提供训练样本与验证样本，即先生成误差 $E$，再诱导出 syndrome
   $$\gamma=\mathrm{commute}(E,S).$$
   这些样本来自物理分布 `p`，但 **P0 的主要 GPU 计算负担不在 `p` 上，而在 `q` 的训练、采样、log-prob 评估和熵估计上。**

3. **P0 的直接产物是“模型学到的相关性”**  
   因此，P0 输出的核心量应理解为
   $$I_q(\gamma_A;\gamma_B),$$
   即由模型分布 `q` 定义的 bipartite classical MI。

4. **P0 不是“精确恢复物理分布 `p` 的 MI”**  
   只有当模型拟合足够好、`q \approx p` 时，`I_q` 才能作为物理分布相关性的近似代理。

---

## 三、为什么先做最小版本

先计算 $I(\gamma_A;\gamma_B)$，原因如下：

1. **空间定义清楚**：$\gamma$ 的每一位都对应具体 stabilizer 位置，适合做空间二分。
2. **物理解释直接**：它直接衡量 syndrome 图样跨越切割边界的经典相关性。
3. **实现改动最小**：不必先处理 $\beta$ 的空间归属问题。
4. **与当前仓库兼容**：当前训练主线就是围绕 syndrome 与 logical bits 的自回归建模，可先抽出 syndrome-only 路径。

> 结论：**第一个计划必须是 syndrome-only 的 `q`-based bipartite MI，而不是直接做 $(\beta,\gamma)$ 联合空间切割，更不是一开始就去显式计算 `p` 的 MI。**

---

## 四、核心公式

对任意空间划分 $X=(A,B)$，二分经典互信息定义为

$$I(A;B)=H(A)+H(B)-H(A,B)$$

在 P0 中，取

$$X=\gamma,\qquad A=\gamma_A,\qquad B=\gamma_B$$

因此 P0 的目标量写成

$$I_q(\gamma_A;\gamma_B)=H_q(\gamma_A)+H_q(\gamma_B)-H_q(\gamma_A,\gamma_B)$$

若变量顺序已经重排为 `[A, B]`，则自回归模型可直接给出：

$$\hat H_q(A,B)=-\frac1N\sum_{k=1}^N \log q_\theta(A^{(k)},B^{(k)})$$

$$\hat H_q(A)=-\frac1N\sum_{k=1}^N \sum_{i\in A} \log q_\theta(x_i^{(k)}|x_{<i}^{(k)})$$

而 $H(B)$ 的严格计算，建议通过**反向顺序模型** `[B, A]` 获得：

$$\hat H_q(B)=-\frac1N\sum_{k=1}^N \sum_{i\in B} \log q_\phi(y_i^{(k)}|y_{<i}^{(k)})$$

其中 $y=(B,A)$。

---

## 五、计划 1：最小正确版本 —— 只计算 syndrome 的 `q`-based 二分互信息

### 4.1 目标
实现并验证 GPU 主导的模型量

$$I_q(\gamma_A;\gamma_B)$$

其中切割方式采用 toric code 上的空间 cut，例如

$$x=L/2$$

左半边 stabilizer 属于区域 $A$，右半边 stabilizer 属于区域 $B$。

这里再强调一次：

* **P0 主要使用 GPU 去训练和评估 `q`；**
* **`p` 只负责生成训练/验证样本，不作为主要计算对象；**
* **P0 的结果首先解释为模型分布 `q` 的相关性结构。**

### 4.2 输入输出

**输入：**
* toric code 距离 $L$
* 错误模型参数（用于生成 syndrome 数据）
* 两个顺序的自回归模型配置：`AB` 与 `BA`
* syndrome 坐标映射
* A/B 划分规则
* GPU 计算资源（优先）

**输出：**
* $\hat H_q(\gamma_A,\gamma_B)$
* $\hat H_q(\gamma_A)$
* $\hat H_q(\gamma_B)$
* $\hat I_q(\gamma_A;\gamma_B)$
* 训练日志与收敛曲线
* Bootstrap 误差条（建议）

### 4.3 实现步骤

#### Step 1：建立 toric syndrome 的空间坐标
为每个 syndrome bit 分配二维坐标 $(x,y)$。

需要产出：
* `gamma_coords`
* `idx_A`
* `idx_B`
* `order_AB`
* `order_BA`

#### Step 2：从物理误差模型采样训练数据
从物理误差样本生成 syndrome：

$$\gamma = \mathrm{commute}(E, S)$$

只保留 syndrome 位，不引入 $\beta$。  
这一步的作用是**提供训练/验证数据集**，不是直接去算 `p` 的 MI。

需要产出：
* `syndrome_train`
* `syndrome_val`
* `syndrome_test`

#### Step 3：在 GPU 上训练两个 syndrome-only 自回归模型
训练两个顺序不同但变量内容相同的模型：

1. `AB` 顺序模型：输入顺序为 `[γ_A, γ_B]`
2. `BA` 顺序模型：输入顺序为 `[γ_B, γ_A]`

目的：
* `AB` 模型用于计算 $\hat H_q(\gamma_A,\gamma_B)$ 和 $\hat H_q(\gamma_A)$
* `BA` 模型用于计算 $\hat H_q(\gamma_B)$

要求：
* 训练任务优先放在 GPU 上执行；
* 模型输入长度为 `m`，只包含 syndrome bits；
* 不复用当前解码主线的 `[γ,β]` 输入格式，而是新建 syndrome-only 路径。

#### Step 4：提取逐位 log-prob
模型不能只返回总 log-likelihood，还要返回每个位点的

$$\log q(x_i|x_{<i})$$

需要在 MADE / NADE / TraDE 上增加统一接口，例如：
* `token_log_prob(x)`

#### Step 5：在 GPU 上评估 `q` 并估计熵
从训练好的模型中采样或批量评估 $N$ 个 syndrome 样本，建议：

$$N \ge 10^5$$

计算：
* 全部 token 和 $\rightarrow \hat H_q(\gamma_A,\gamma_B)$
* `A` 前缀 token 和 $\rightarrow \hat H_q(\gamma_A)$
* `B` 前缀 token 和 $\rightarrow \hat H_q(\gamma_B)$

这里的主要计算负担包括：
* 批量前向传播；
* 批量 token 级 log-prob 提取；
* 大样本熵平均；
* bootstrap 重采样。

这些都应优先放在 GPU 上执行。

#### Step 6：汇总互信息
最终计算

$$\hat I_q(\gamma_A;\gamma_B)=\hat H_q(\gamma_A)+\hat H_q(\gamma_B)-\hat H_q(\gamma_A,\gamma_B)$$

### 4.4 P0 的解释边界

P0 算出来的量，首先应解释为：

$$\hat I_q(\gamma_A;\gamma_B)$$

即**模型分布 `q` 的 bipartite classical MI**。

只有在模型拟合质量足够好时，才进一步把它看作对物理分布相关性的近似代理。

### 4.5 验证标准

最小版本完成的标志：

1. 能在 `L=4` toric code 上训练出两个 syndrome-only 模型；
2. 能在 GPU 上稳定输出有限的 $\hat I_q(\gamma_A;\gamma_B)$ 数值；
3. 改变随机种子后结果波动可接受；
4. Bootstrap 误差条不发散；
5. 改变切割方向或 cut 位置时，$\hat I_q$ 有合理变化；
6. 所得结果随样本数增加逐渐收敛。

---

## 六、计划 2：扩展到联合变量 $q_\theta(\beta,\gamma)$

### 5.1 目标
从最小版本扩展到文档原始目标，即计算空间切割下的

$$I(A;B),\qquad A,B \subset (\beta,\gamma)$$

### 5.2 主要困难

1. **$\beta$ 不是天然局域变量**：logical bits 通常是全局自由度。
2. **需要定义 $\beta$ 如何归属 A/B**：否则“空间切割”没有唯一物理解释。
3. **顺序必须完全重排**：不能继续沿用当前仓库默认的 `[γ, β]` 排列。

### 5.3 可选策略

#### 策略 A：把 $\beta$ 当作全局附加变量
把全部 $\beta$ 放在某一侧，计算

$$I((\gamma_A,\beta); \gamma_B)$$

优点：
* 实现简单

缺点：
* 物理解释偏人为

#### 策略 B：为 logical operators 构造几何归属
根据逻辑环路穿越区域的支撑范围，将逻辑自由度分配到 A/B。

优点：
* 更接近空间切割物理图像

缺点：
* 定义复杂，需要额外约定

> 建议：**先完成计划 1，再决定是否进入计划 2。**

---

## 七、计划 3：做尺度行为分析（Scaling Study）

### 6.1 目标
研究

$$I(\gamma_A;\gamma_B) \quad \text{或} \quad I(A;B)$$

随系统尺寸 $L$ 的变化趋势。

### 6.2 建议尺寸
* $L=4$
* $L=6$
* $L=8$
* $L=10$

### 6.3 需要输出
* `MI vs L` 数据表
* `MI vs L` 图像
* 误差条
* 线性拟合结果

### 6.4 物理判据

1. 若
   $$I \propto L$$
   则说明存在边界尺度的相关性；
2. 若拟合
   $$I(L)=aL-b$$
   可进一步分析常数项是否具有拓扑信息意义；
3. 比较不同架构在大 $L$ 下是否出现饱和。

---

## 八、计划 4：做架构比较（MADE / NADE / TraDE）

### 7.1 目标
比较不同自回归结构在捕捉长程相关上的能力差异。

### 7.2 对比对象
* `MADE`
* `NADE`
* `TraDE_binary`

### 7.3 关注指标
* NLL / log-likelihood
* $I(\gamma_A;\gamma_B)$
* 随 $L$ 增长的保持能力
* 训练稳定性
* 推断耗时

### 7.4 预期现象
* Transformer 类模型更可能维持随 $L$ 增长的相关信息；
* 较弱的自回归结构可能在大尺度下表现出信息瓶颈。

---

## 九、建议的代码改动清单

### 8.1 新增模块

1. `module/spatial_partition.py`
   * toric syndrome 坐标
   * A/B 划分
   * 顺序重排索引

2. `module/mi_utils.py`
   * 逐位 log-prob 提取
   * 祖先采样
   * 熵与 MI 估计
   * bootstrap 误差估计

3. `decoding/train_mi_syndrome.py`
   * syndrome-only 训练入口
   * 支持 `AB` / `BA` 两种顺序
   * 优先面向 GPU 训练

4. `decoding/mi_bipartite.py`
   * 最小版本评估入口
   * 先支持 `observable=syndrome`
   * 输出 $\hat I_q(\gamma_A;\gamma_B)$
   * 后续再扩展到 `observable=joint_beta_gamma`

### 8.2 需要补充的模型接口

对 `MADE` / `NADE` / `TraDE_binary` 建议统一增加：

* `token_log_prob(x)`
* `sample(batch_size)`
* `log_prob(x)` 保持现有接口

---

## 十、任务清单（按执行顺序）

| 优先级 | 任务 | 关键动作 | 预期产出 |
| :--- | :--- | :--- | :--- |
| P0 | **最小版本：syndrome-only q-MI** | 建立 toric syndrome 空间切割、在 GPU 上训练 AB/BA 两个顺序模型、计算 $\hat I_q(\gamma_A;\gamma_B)$ | 第一版可运行 q-based MI 数值 |
| P1 | **统一逐位 log-prob 接口** | 给 MADE / NADE / TraDE_binary 增加 token 级概率输出 | 熵计算接口 |
| P2 | **误差分析** | Bootstrap 与样本数收敛检查 | 误差条 |
| P3 | **联合变量扩展** | 讨论并实现 $\beta$ 的空间归属 | $I(A;B)$ for $(\beta,\gamma)$ |
| P4 | **尺度分析** | 改变 $L$ 重复实验 | `MI vs L` |
| P5 | **架构比较** | 在 MADE / NADE / TraDE 上重复 P0-P4 | 架构对比结果 |

---

## 十一、风险提示

1. **不能直接复用当前默认变量顺序**  
   当前仓库默认更接近 `[γ, β]` 的任务组织，不是空间切割顺序。

2. **若不训练 BA 模型，$H(B)$ 只能做近似**  
   这会削弱最终 MI 的严格性。

3. **$\beta$ 的空间划分不是自动成立的**  
   在计划 2 之前，不应把“联合变量 MI”视作已定义清楚。

4. **不要把 `I_q` 和 `I_p` 混为一谈**  
   P0 首先得到的是模型分布 `q` 的相关性，而不是物理分布 `p` 的严格 MI。

5. **样本量不足会导致熵估计偏差**  
   建议至少做 bootstrap 与多随机种子检查。

---

## 十二、下一步建议

最合理的下一步是：

1. 先新增 `module/spatial_partition.py`
2. 先补 `token_log_prob` 接口
3. 先实现 `decoding/train_mi_syndrome.py`
4. 再实现 `decoding/mi_bipartite.py`
5. 先在 `L=4` toric code 上用 GPU 跑通 **计划 1**

> 在 **计划 1** 完成之前，不建议直接跳到联合变量 $(\beta,\gamma)$ 的 MI。

---

## 十三、当前实现进度与剩余任务

**更新日期：2026-05-14**

### 13.1 已完成实现

1. **P0 基础设施已落地**
   * 已修复 `decoding/args.py` 的 help 文本与布尔参数解析问题。
   * 已让 `mwpm.py` 与 `bposd.py` 支持单点 smoke test，不再只适合长 sweep。
   * 已修复 `forward_decoding.py` 在单样本 timing 统计下输出 `nan` 的问题。

2. **P1 空间切割基础已落地**
   * 已在 `module/utils.py` 中加入 toric syndrome 空间坐标与 `AB/BA` 顺序工具。
   * 当前可直接得到 `coords`、`idx_A`、`idx_B`、`order_AB`、`order_BA`。

3. **P2 syndrome-only 数据路径已落地**
   * 已新增 `decoding/syndrome_dataset.py`。
   * 当前可从物理误差直接采样 syndrome，并保存 `train / val / test` 数据集。
   * 当前支持 toric code 的 `AB` 与 `BA` 两种顺序重排。

4. **P3 syndrome-only 训练入口已落地**
   * 已新增 `decoding/train_mi_syndrome.py`。
   * 当前支持 `MADE`、`NADE`、`TraDE_binary` 三种模型。
   * 当前支持读取 P2 数据包、训练、验证、测试并保存 checkpoint。

5. **TraDE 数值稳定性修复已完成**
   * 已修复 `module/TraDE.py` 中 causal attention mask 的实现错误。
   * 该修复后 `TraDE_binary` 的 syndrome-only 训练已能完成最小 CPU 验证。

6. **P4 最小 MI 评估脚本已落地**
   * 已新增 `decoding/mi_bipartite.py`。
   * 当前可加载 `AB/BA` 两个 syndrome-only checkpoint，并估计
     $H_q(A,B)$、$H_q(A)$、$H_q(B)$ 与 $I_q(A;B)$。
   * 当前支持 Monte Carlo 采样估计与结果 JSON 保存。

7. **P5 统一 token 级 log-prob 接口已落地**
   * 已给 `MADE`、`NADE`、`TraDE_binary` 增加 `token_log_prob(...)`。
   * 当前 `MADE` 与 `TraDE_binary` 也提供统一的 `sample(...)` 入口，便于评估脚本复用。

8. **P6 Bootstrap 误差分析最小版本已落地**
   * `decoding/mi_bipartite.py` 当前支持 bootstrap 重采样。
   * 当前可输出 bootstrap 均值、标准差与 95% 置信区间。

9. **P7 GPU 最小闭环已验证**
   * 已通过 `codex exec --sandbox danger-full-access` 的 GPU 包装路径完成
     `train_mi_syndrome.py` 的 `AB/BA` 最小 `cuda:0` 训练。
   * 已在 GPU 上完成 `mi_bipartite.py` 的最小评估闭环。

10. **P8 自动化尺度流水线已落地**
   * 已新增 `decoding/run_mi_scale_sweep.py`。
   * 当前可按多个 `L` 自动执行 code 生成、`AB/BA` 数据集生成、训练、MI 评估与最终聚合。
   * 当前默认拒绝奇数 `L`，除非显式传入 `--allow-unbalanced`。

11. **P6 稳定性分析入口已落地**
   * 已新增 `decoding/mi_stability_analysis.py`。
   * 当前可对同一组 `AB/BA` checkpoint 执行“多样本数 × 多评估种子”的重复评估。
   * 当前可输出 raw CSV、grouped CSV、summary JSON 与收敛图。

12. **P7 MADE 正式 GPU 基线已落地**
   * 已在 `L=4`、toric code、`er=0.05` 上完成 `MADE` 的 `AB/BA` 正式 GPU 训练。
   * 当前正式基线配置为：`depth=0`、`width=64`、`epoch=100`、`batch=512`、`lr=1e-3`、`dtype=float32`、`device=cuda:0`。
   * 当前正式评估使用 `mi_samples=20000`、`bootstrap_samples=200`。
   * 产物已写入 `net/mi_scaling/p7_made_d4/`。

13. **P8 MADE even-L 正式尺度结果已落地**
   * 已在 `L=4,6,8,10,12` 上完成 `MADE` 的正式 GPU 尺度扫描。
   * 当前 `MI vs L` 汇总写入 `net/mi_scaling/p8_made_even/`。
   * 当前结果为：`L=4 -> 0.823137`，`L=6 -> 1.076084`，`L=8 -> 2.772041`，`L=10 -> 0.986748`，`L=12 -> 1.280560`。

14. **P8 MADE 大尺度异常已完成诊断与一轮修复验证**
   * 已确认 `L=10,12` 的异常不主要来自 MI 公式，而是训练策略失效。
   * 原正式配置使用 `StepLR(step_size=2000)`，但 `P8` 训练仅运行 `100` epoch，因此学习率在整轮训练中实际上从未下降。
   * 现已将 `decoding/train_mi_syndrome.py` 改为验证集驱动的 `ReduceLROnPlateau`，并补充 `early stopping`、`effective_width`、`parameter_count` 与调度器配置记录。
   * `MADE` 相关新参数已接入 `decoding/args.py`、`decoding/run_mi_scale_sweep.py` 与 `decoding/mi_bipartite.py`，保证训练、评估、批量 sweep 三条路径一致。
   * 端到端复验结果：在新目录 `net/mi_scaling/p8_made_plateau_l10/` 下，`L=10` 的正式复跑结果从旧的 `MI=0.986748` 提升到 `MI=2.934803`；对应 `AB` 模型最佳验证 NLL 从 `62.221869` 改善到 `61.849250`。
   * 额外针对性验证表明：单纯启用 `made_max_params` 缩宽并不能稳定改善结果，当前更应把 plateau 学习率调度视为默认修复方案。

### 13.2 当前剩余任务

| 状态 | 任务 | 关键动作 | 预期产出 |
| :--- | :--- | :--- | :--- |
| 已完成 | **P4：最小 MI 评估脚本** | `decoding/mi_bipartite.py` 已可加载 `AB/BA` 模型并估计 $H_q(A,B)$、$H_q(A)$、$H_q(B)$ | 第一版 $\hat I_q(\gamma_A;\gamma_B)$ 数值 |
| 已完成 | **P5：统一 token 级 log-prob 接口** | `MADE / NADE / TraDE_binary` 已支持 `token_log_prob(x)` | 熵拆分与区域级 NLL 计算接口 |
| 部分完成 | **P6：Bootstrap 误差分析** | 已支持 bootstrap 重采样，并新增 `decoding/mi_stability_analysis.py`；仍需补正式多样本数与多随机种子实验数据 | 初版误差条与稳定性表 |
| 部分完成 | **P7：GPU 正式训练闭环** | 已完成 `MADE` 在 `L=4` 上的正式 GPU 基线训练与评估；后续仍需把同类配置推广到更多 `L` 或更多架构 | 一组可复现实验 checkpoint |
| 部分完成 | **P8：尺度分析** | 已完成 `MADE` 在 `L=4,6,8,10,12` 上的首轮正式 GPU 尺度结果，并已定位大尺寸异常的训练策略根因；后续需用 plateau 调度重新生成正式 even-`L` 曲线 | 修正后的 `MI vs L` 曲线 |
| 待完成 | **P9：架构比较** | 在 `MADE / NADE / TraDE_binary` 上重复尺度分析 | 架构对比结果 |
| 待完成 | **P10：联合变量扩展** | 定义 $\beta$ 的空间归属并扩展到 $(\beta,\gamma)$ 联合 MI | $I(A;B)$ for joint variables |

### 13.3 下一步推荐顺序

1. 先用当前新默认训练策略重跑 `P8 MADE even-L` 正式结果
   * 推荐固定配置：`width=64`、`lr=1e-3`、`lr_decay_factor=0.5`、`lr_decay_patience=5`、`min_lr=2e-4`、`early_stop_patience=20`、`dtype=float32`、`device=cuda:0`
   * 优先补 `L=8,10,12`，并与旧目录 `net/mi_scaling/p8_made_even/` 做并排对比
   * 新正式结果建议单独写入一个新目录，例如 `net/mi_scaling/p8_made_plateau_even/`
2. 完整回填 `P6` 的正式稳定性数据
   * 对修复后的 `MADE` checkpoint 再做样本数收敛检查与多评估种子重复
   * 重点确认 `MI` 提升不是单次 Monte Carlo 波动
3. 固化 `MADE` 正式推荐训练策略
   * 以修复后的 `P8` 结果为依据，决定 `plateau + early stop` 是否作为仓库默认正式配置
   * 明确 `made_max_params` 仅作为超大系统的可选保护阀，而不是默认收缩策略
4. 再进入 `P9` 架构比较
   * 在 `NADE / TraDE_binary` 上复用同一套记录规范与尺度流水线
   * 对比的不只是 `MI vs L`，还应包括 `best_val_nll`、`epochs_trained`、推理成本
5. 最后进入 `P10` 联合变量扩展

### 13.4 本轮问题处理总结

1. **现象**
   * `P8` 首轮 `MADE` even-`L` 结果在 `L=10,12` 上出现明显非单调，`MI` 相比 `L=8` 异常偏低。

2. **根因判断**
   * `MI` 评估公式本身无误，计算的是模型分布 `q` 上的
     $I_q(A;B)=H_q(A)+H_q(B)-H_q(A,B)$。
   * 异常主要来自训练策略：原 `StepLR(step_size=2000)` 在 `100` epoch 训练中永远不触发，导致大尺寸模型在验证集上早早停滞后持续漂移。

3. **代码修改**
   * `decoding/train_mi_syndrome.py`：
     将学习率调度改为 `ReduceLROnPlateau`，加入 `early stopping`，并把 `effective_width`、`parameter_count`、`epochs_trained` 与调度器参数写入 checkpoint/record。
   * `decoding/args.py`：
     增加 `made_activation`、`made_residual`、`made_max_params`、`weight_decay`、`lr_decay_factor`、`lr_decay_patience`、`min_lr`、`early_stop_patience` 等参数。
   * `decoding/run_mi_scale_sweep.py`：
     把新训练参数接入自动化尺度流水线。
   * `decoding/mi_bipartite.py`：
     评估时按 checkpoint 中的 `effective_width` / activation / residual 重建模型。

4. **验证结果**
   * `L=10` 旧正式结果：`MI=0.986748`。
   * `L=10` 新正式复跑结果：`MI=2.934803`，输出目录为 `net/mi_scaling/p8_made_plateau_l10/`。
   * `L=10 AB` 最佳验证 NLL：`62.221869 -> 61.849250`。
   * 附加试验显示，单纯缩小 `width` 或改用 `relu` 并不能稳定解决问题，因此不应把“缩模型”作为默认修法。

### 13.5 Sandbox GPU 尝试清单

如果目标是让 **当前 sandbox 环境** 也能使用 GPU，建议单独按下面顺序排查：

1. **确认设备节点是否暴露**
   * 检查 sandbox 内是否能看到 `/dev/nvidia0`、`/dev/nvidiactl`、`/dev/nvidia-uvm`、`/dev/nvidia-uvm-tools`、`/dev/nvidia-modeset`。

2. **确认 NVIDIA 用户态库是否暴露**
   * 检查 sandbox 内是否能解析 `libcuda.so.1`、`libnvidia-ml.so.1` 等必要动态库。

3. **确认 PyTorch 与驱动版本匹配**
   * 当前推荐环境仍是 `ai-env-cu128`，避免回到 `cu130` 版本导致 `torch.cuda.is_available()` 为 `False`。

4. **确认 sandbox 启动器是否允许透传 GPU**
   * 当前现象表明 session 在受限 sandbox 中启动，默认没有把 `/dev/nvidia*` 暴露进去。
   * 若运行器层不改，仅修改仓库代码无法让 GPU 在 sandbox 内可用。

5. **做最小 GPU 探针**
   * 先只跑 `torch.cuda.is_available()`、`torch.cuda.device_count()` 和一个最小 CUDA tensor 运算。
   * 在这些探针通过之前，不建议直接启动 P3 正式训练。

### 13.6 Sandbox GPU 成功标准

以下条件都满足，才认为 sandbox GPU 可用于 MI 训练：

1. `torch.cuda.is_available()` 返回 `True`
2. `torch.cuda.device_count()` 至少为 `1`
3. 最小 CUDA tensor 运算成功
4. `decoding/train_mi_syndrome.py` 能以 `-device cuda:0` 完成至少 1 epoch
5. 保存出的 checkpoint 与 CPU 路径结构一致
