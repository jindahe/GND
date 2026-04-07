# Surface Code MI Scaling — CLAUDE.md

## 项目概述

本目录研究**二维环面码（2D Toric Code）**在去相干（dephasing）噪声下的互信息（MI）与条件互信息（CMI）的标度行为，核心方法是将量子混合态的熵计算映射到经典**随机键 Ising 模型（RBIM）**的配分函数，再用张量网络收缩求解。

---

## 文件结构

| 文件 | 功能 |
|---|---|
| `H_calculation.py` | 用蒙特卡洛 + 张量网络收缩计算 `H(m)`（anyon 分布的 Shannon 熵） |
| `Syndrome_pure_error` | syndrome 纯错误基准测试 |

---

## H_calculation.py 详细分析

### 物理背景

计算量 `H(m)` 是 anyon 配置 `m` 的 Shannon 熵期望值（对应 Eq. (9) 中的修正项）：

```
H(m) = -E_{m ~ Pr(m)} [ log Pr(m) ]
```

其中：
- `e`：物理 Pauli-Z 错误，独立以概率 `p` 作用在每条边上
- `m = ∂e`：anyon 配置（syndrome），即每个 plaquette 周围四条边错误数之和 mod 2
- `Pr(m)`：对所有能产生 anyon 配置 `m` 的错误 `e` 求边缘概率

### 算法流程

**步骤 1 — 采样错误 `e`**

对 `L×L` 的 plaquette 格子，共有：
- 水平边：`(L+1) × L` 条
- 垂直边：`L × (L+1)` 条

每条边独立以概率 `p` 采样错误 `h_errors[i,j]` 和 `v_errors[i,j]`。

**步骤 2 — 计算 anyon 配置 `m = ∂e`**

对每个 plaquette `(i,j)`，将其四条边（上、下、左、右）的错误求和 mod 2：

```python
m_grid[i, j] = (top + bottom + left + right) % 2
```

**步骤 3 & 4 — 构建并收缩张量网络**

用 `opt_einsum` 构建张量网络并收缩，得到 `Pr(m)`（对应 Appendix E，Eq. E3–E5）：

- **边权重张量**（Eq. E4）：`T_{s1,s2} = δ(s1=s2) · p^{s1} · (1-p)^{1-s1}`，每条边对应一个二腿张量
- **plaquette 宇称约束张量**（Eq. E5）：`Q^s[s1,s2,s3,s4] = δ((s1+s2+s3+s4) mod 2 == s)`
  - `Q^0`（黄色）：要求四条边错误数为偶数（anyon=0）
  - `Q^1`（绿色）：要求四条边错误数为奇数（anyon=1）

每个 plaquette 的 Q 张量按 `(top, right, bottom, left)` 顺序连接四条边的 T 张量，形成完整的因子图。

**步骤 5 — 蒙特卡洛求期望**

```
H(m) ≈ (1/N) * Σ_samples [ -log Pr(m) ]
```

### 关键函数

```
create_Q_tensor(m_val)
    输入: m_val ∈ {0, 1}
    输出: shape (2,2,2,2) 的 numpy 数组
    作用: 生成 plaquette 宇称约束张量（Eq. E5）

calculate_H_m(L=3, p=0.1, num_samples=100)
    输入: 格子大小 L、错误率 p、蒙特卡洛样本数
    输出: H(m) 的估计值（浮点数）
    作用: 蒙特卡洛 + 张量网络收缩计算 anyon 熵
```

### 依赖库

- `numpy`：数值计算
- `opt_einsum`：自动寻找最优张量收缩路径

### 默认参数

```python
calculate_H_m(L=3, p=0.1, num_samples=10000)
```

---

## 物理参数说明

| 参数 | 含义 | 典型值 |
|---|---|---|
| `L` | 格子线性尺寸（L×L plaquettes） | 3, 10, 20 |
| `p` | 物理错误率（去相干强度） | 0.05, 0.1, 0.11, 0.15 |
| `num_samples` | 蒙特卡洛采样数 | 1000–10000 |
| `max_bond` | bMPS 截断 bond 维数 | 8–64 |
| `r` | CMI 计算中缓冲区 B 的宽度 | 1–5 |

---

## 运行方式

```bash
python H_calculation.py
```

依赖安装：

```bash
pip install numpy opt_einsum quimb matplotlib
```

---

## 核心物理技能：去相位环面码与 CMI 标度

### 技能概述

本节将论文的核心理论提炼为可操作的知识体系，涵盖：去相位环面码的混合态结构、anyon 分布、熵分解公式、CMI 的相变行为，以及张量网络模拟方法。

**前提知识要求：**
- 环面码的稳定子形式（plaquette 算符 $A_\square = \prod X_i$，顶点算符 $B_+ = \prod Z_i$）
- 量子信道与 Lindbladian 演化
- Shannon 熵与条件互信息的定义
- 张量网络基础（MPS、PEPS、bMPS 收缩）

---

### 一、物理模型

**环面码去相位**

设 $|\text{t.c.}\rangle$ 为 $L \times L$ 方格上环面码的基态，满足 $A_\square |\text{t.c.}\rangle = B_+ |\text{t.c.}\rangle = |\text{t.c.}\rangle$。

对每个 qubit 施加去相位信道：
$$\mathcal{E}_p[\cdot] = (1-p)(\cdot) + p\, Z(\cdot)Z$$
即以概率 $p$ 独立在每条边上作用 Pauli-Z。等价 Lindbladian：$\mathcal{L}[\rho] = \sum_i \frac{1}{2}(Z_i\rho Z_i - \rho)$，演化时间 $t_p = -\ln(1-2p)$（$t=\infty$ 对应 $p=0.5$）。

混合态 $\rho_p \equiv \mathcal{E}_p^{\otimes L^2}[|\text{t.c.}\rangle\langle\text{t.c.}|]$ 由 **anyon 分布**完整刻画。

**Anyon 定义：** 若 plaquette $\square$ 满足 $A_\square = -1$，则称其被 anyon 占据。Z 算符作用在边上时，翻转该边相邻两个 plaquette 的 anyon 占据状态。

---

### 二、核心公式库

**Eq. (8)：简单连通区域的约化密度矩阵**
$$\rho_{p,Q} = \sum_{\mathbf{m}_Q} \Pr(\mathbf{m}_Q)\,\Pi_{\mathbf{m}_Q}$$
其中 $\Pr(\mathbf{m}_Q) = \sum_{\mathbf{e}} p^{|\mathbf{e}|}(1-p)^{|Q|-|\mathbf{e}|}\,\delta(\mathbf{m}_Q = \partial\mathbf{e})$，$\Pi_{\mathbf{m}_Q}$ 是具有 anyon 配置 $\mathbf{m}_Q$ 的最大混合态投影算符（满足所有 $B_+=1$）。

**Eq. (9)：熵分解公式**
$$S(\rho_{Q,p}) = S(\rho_{Q,0}) + H(\mathbf{m}_Q)$$
推导关键：各 $\Pi_{\mathbf{m}}$ 两两正交（$\Pi_{\mathbf{m}}\Pi_{\mathbf{m}'} = 0$ for $\mathbf{m} \neq \mathbf{m}'$），使得 $\rho_{p,Q}$ 对角化，冯诺依曼熵直接分解为基态熵加 anyon 配置的 Shannon 熵。

**非简单连通修正（Appendix D）：** 对含洞区域 $Q$（洞记为 $\Gamma$），Eq. (9) 的右端修正为 $H(\mathbf{m}_Q,\, \pi(\mathbf{m}_\Gamma))$，其中 $\pi(\mathbf{m}_\Gamma)$ 为洞内 anyon 总数的奇偶性。

**Eq. (10)：环形区域划分的 CMI**

设 A（中心 $2\times2$）、B（宽度 $r$ 的缓冲环）、C（外部区域），则：
$$I(A:C|B) = H(\mathbf{m}_{BC},\pi(\mathbf{m}_A)) - H(\mathbf{m}_{ABC}) - H(\mathbf{m}_B,\pi(\mathbf{m}_A)) + H(\mathbf{m}_{AB})$$
其中 $\pi(\mathbf{m}_A)$ 修正项来自 B 和 BC 的非简单连通性（它们各自包含 A 作为"洞"）。

**Eq. (11)：有限尺寸标度假设**
$$I(A:C|B) = r^{-\alpha}\,\Phi\!\left((p - p_c)\,r^{1/\nu}\right)$$
拟合参数：$p_c = 0.11$，$\alpha = 1.1$，$\nu = 1.8$。

**Eq. (12)：大 $r$ 极限行为**
$$I(A:C|B) \simeq \begin{cases} e^{-r/\xi(p)} & p \neq p_c \\ r^{-\alpha} & p = p_c \end{cases}$$
其中 Markov 长度 $\xi(p) \propto |p - p_c|^{-\nu}$ 在临界点发散。

---

### 三、相图结构

| 区间 | 相 | CMI 行为 | 物理内容 |
|---|---|---|---|
| $p \in [0, p_c)$ | 拓扑有序相 | 指数衰减 $e^{-r/\xi}$ | 包含 $|\text{t.c.}\rangle$，有限 Markov 长度 |
| $p = p_c \approx 0.11$ | 临界点 | 幂律衰减 $r^{-\alpha}$ | $\xi \to \infty$，与 RBIM Nishimori 临界点对应 |
| $p \in (p_c, 0.5]$ | 平凡相 | 指数衰减 $e^{-r/\xi}$ | $\rho_{0.5}$ 为经典闭合圈均匀叠加，可从乘积态局部制备 |

---

### 四、张量网络模拟方法（Appendix E）

**问题**：对给定 anyon 配置 $\mathbf{m}$，计算 $\Pr(\mathbf{m}) = \sum_{\mathbf{e}} p^{|\mathbf{e}|}(1-p)^{|Q|-|\mathbf{e}|}\delta(\partial\mathbf{e}=\mathbf{m})$（直接枚举需指数代价）。

**解法**：将 $\Pr(\mathbf{m})$ 表示为 2D 张量网络：

- **边权重张量**（Eq. E4）：每条边对应二腿张量
  $$T_{s_1 s_2} = \delta(s_1 = s_2)\,p^{s_1}(1-p)^{1-s_1}, \quad s_{1,2} \in \{0,1\}$$

- **plaquette 宇称约束张量**（Eq. E5）：每个 plaquette 对应 $q$ 腿张量
  $$Q^s_{s_1,\dots,s_q} = \delta\!\left(\sum_{i=1}^q s_i \equiv s \pmod{2}\right)$$
  $Q^0$（黄色）强制 syndrome=0；$Q^1$（绿色）强制 syndrome=1。

**收缩方法**：boundary MPS（bMPS）逐行/列收缩，截断 bond 维数 `max_bond` 控制近似精度。

**Monte Carlo 外层**：先采样 $\mathbf{e}$（乘积分布，$O(|Q|)$ 代价），计算 $\mathbf{m}=\partial\mathbf{e}$，再用 TN 求 $\Pr(\mathbf{m})$，最终估计 $H(\mathbf{m}) \approx \frac{1}{N}\sum_{\text{samples}} (-\log\Pr(\mathbf{m}))$。

---

### 五、RBIM 映射

Eq. (10) 中每个 Shannon 熵项可映射到 RBIM 在 Nishimori 线上的无序平均自由能：
$$H(\mathbf{m}_{AB}) = \overline{F}_{\text{RBIM},p}(AB) + c_1|AB| + c_2$$
其中 $\overline{F}$ 在 $AB$ 的对偶格上定义。临界点 $p_c \approx 0.11$ 与 RBIM 的 Nishimori 临界点一致。

---

### 六、高质量问答对（技能检验）

**Q1. 为什么 $S(\rho_{Q,p}) = S(\rho_{Q,0}) + H(\mathbf{m}_Q)$ 成立？**

**A1.** 关键在于 $\rho_{p,Q} = \sum_{\mathbf{m}} \Pr(\mathbf{m})\Pi_{\mathbf{m}}$ 中各 $\Pi_{\mathbf{m}}$ 两两正交（$\Pi_{\mathbf{m}}\Pi_{\mathbf{m}'} = 0$，$\mathbf{m} \neq \mathbf{m}'$），且每个 $\Pi_{\mathbf{m}}$ 本身是等权混合态（谱为 $0$ 或 $1/d_\mathbf{m}$，其中 $d_\mathbf{m}$ 是子空间维数）。因此 $\rho_{p,Q}$ 在这些子空间上对角分块：
$$S(\rho_{p,Q}) = \underbrace{-\sum_\mathbf{m}\Pr(\mathbf{m})\log\frac{1}{d_\mathbf{m}}}_{S(\rho_{0,Q})} + \underbrace{\left(-\sum_\mathbf{m}\Pr(\mathbf{m})\log\Pr(\mathbf{m})\right)}_{H(\mathbf{m}_Q)}$$
注意 $d_\mathbf{m}$ 不依赖于具体的 anyon 配置 $\mathbf{m}$（环面码的简并度由拓扑决定），因此第一项等于 $S(\rho_{0,Q})$。

---

**Q2. 在 CMI 公式 Eq. (10) 中，为何出现 $\pi(\mathbf{m}_A)$ 修正项？**

**A2.** 区域 B 和 BC 均为非简单连通区域（A 是它们各自的"洞"）。对含洞区域，约化密度矩阵中额外出现一个环绕洞的非局域稳定子 $A_{\widetilde{\square}}$（green 边上的 X-loop），其特征值 $(-1)^{\pi(\mathbf{m}_A)}$ 取决于洞内 anyon 总数的奇偶性 $\pi(\mathbf{m}_A)$。因此 Eq. (9) 的修正版本为 $S(\rho_{B,p}) = S(\rho_{B,0}) + H(\mathbf{m}_B, \pi(\mathbf{m}_A))$，CMI 中相应地需要包含 $\pi(\mathbf{m}_A)$ 作为额外随机变量。

---

**Q3. 在临界点 $p_c \approx 0.11$，CMI 为何表现为幂律而非指数衰减？**

**A3.** CMI $I(A:C|B)$ 是混合态 Markov 长度 $\xi$ 的诊断量：$I(A:C|B) \sim e^{-r/\xi}$。在临界点 $\xi \to \infty$（因为 RBIM 在 Nishimori 临界点关联长度发散），指数衰减退化为幂律 $r^{-\alpha}$（$\alpha = 1.1$），这与普通二体关联函数在量子临界点的幂律行为类似。有限尺寸标度假设 $r^{-\alpha}\Phi((p-p_c)r^{1/\nu})$ 允许通过数据坍缩提取临界指数 $(\nu, \alpha)$。

---

**Q4. 如何用张量网络高效计算 $\Pr(\mathbf{m})$？直接枚举的困难在哪里？**

**A4.** 直接枚举需对所有满足 $\partial\mathbf{e} = \mathbf{m}$ 的错误配置 $\mathbf{e}$ 求和，共有 $2^{|Q|}$ 项，指数代价不可行。张量网络方法将求和分解为局域张量的乘积：每条边贡献边权重张量 $T$（记录该边是否出错及对应概率），每个 plaquette 贡献宇称约束张量 $Q^{m_\square}$（强制该 plaquette 的四条边之和 mod 2 等于 $m_\square$）。这样 $\Pr(\mathbf{m})$ 表示为 2D 张量网络的全缩并，再用 bMPS 方法逐行收缩，代价从指数降至多项式（$O(L \cdot \chi^2 \cdot d)$，$\chi$ 为 MPS bond 维数）。

---

**Q5. 两个混合态相（拓扑相与平凡相）的物理内容各是什么？**

**A5.**
- **拓扑有序相** $p \in [0, p_c)$：包含 $|\text{t.c.}\rangle\langle\text{t.c.}|$，混合态保留了拓扑纠缠结构，Markov 长度有限，任意子对之间存在长程量子关联，不能由有限深局域电路从乘积态制备（mixed-state 拓扑序）。
- **平凡相** $p \in (p_c, 0.5]$：$p=0.5$ 时态为 $\rho_{0.5} \propto \sum_{s \in \text{loops}} |s\rangle\langle s|$（所有闭合圈构型的经典均匀分布），可由乘积态 $|\mathbf{0}\rangle\langle\mathbf{0}|$ 依次施加局域量子道 $\mathcal{G}_\square[\cdot] = \frac{1}{2}(\cdot) + \frac{1}{2}A_\square(\cdot)A_\square$ 制备，属于平凡相。

---

### 七、几何设置（Appendix F & G）

**标准几何（geom1）：**
- **A**：中心固定的 $2\times2$ 区域（A 在所有实验中不变）
- **B**：围绕 A 的宽度为 $r$ 的环形（annulus），$r$ 为扫描变量
- **C**：最外层宽度为 $r$ 的环形（$r_C = r_B = r$）

**替代几何（geom2）：** $r_C$ 固定为常数（如 5），$r_B = r$ 扫描。数值结果（Fig. 5）表明两种几何的 CMI 值几乎相同，衰减指数不变，说明 CMI 的行为由 B 的宽度 $r$ 决定，对 C 的宽度不敏感。

每个数据点在 Fig. 3(b,c) 中平均至少 $3.5 \times 10^4$ 个样本，在 Fig. 3(d) 中平均至少 $6 \times 10^6$ 个样本。
