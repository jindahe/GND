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
