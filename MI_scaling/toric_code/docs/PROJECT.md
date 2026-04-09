# Surface Code MI Scaling — 项目内容

## 项目概述

本目录研究**二维环面码（2D Toric Code）**在去相干（dephasing）噪声下的互信息（MI）与条件互信息（CMI）的标度行为，核心方法是将量子混合态的熵计算映射到经典**随机键 Ising 模型（RBIM）**的配分函数，再用张量网络收缩求解。

---

## 文件结构

| 文件 | 功能 |
|---|---|
| `h_calculation.py` | 用蒙特卡洛 + 张量网络收缩计算 `H(m)`（anyon 分布的 Shannon 熵） |
| `syndrome_pure_error.py` | syndrome 纯错误基准测试 |

---

## h_calculation.py 详细分析

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
python h_calculation.py
```

依赖安装：

```bash
pip install numpy opt_einsum quimb matplotlib
```
