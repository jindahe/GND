# Surface Code MI Scaling — CLAUDE.md

## 项目概述

本目录研究**二维环面码（2D Toric Code）**在去相干（dephasing）噪声下的互信息（MI）与条件互信息（CMI）的标度行为，核心方法是将量子混合态的熵计算映射到经典**随机键 Ising 模型（RBIM）**的配分函数，再用张量网络收缩求解。

---

## 文件结构

| 文件 | 功能 |
|---|---|
| `H_calculation.py` | 用蒙特卡洛 + 精确张量网络收缩计算 `H(m)`（syndrome 熵） |
| `CMI.py` | 用 quimb PEPS 张量网络计算 `I(A:C|B)`（条件互信息）的标度 |
| `MI.py` | 用 n=2 复制技巧（replicated RBIM）+ quimb BMPS 计算 Renyi-2 熵及 CMI proxy |
| `Syndrome_pure_error` | （待确认内容） |

---

## H_calculation.py 详细分析

### 物理背景

计算量 `H(m)` 是 syndrome 测量结果 `m` 的香农熵期望值：

```
H(m) = -E_e[ log Pr(m) ]
```

其中：
- `e`：物理 Pauli-Z 错误，独立以概率 `p` 作用在每条边上
- `m = ∂e`：syndrome，即每个 plaquette 周围四条边错误数之和 mod 2
- `Pr(m)`：对所有能产生 syndrome `m` 的错误 `e` 求边缘概率

### 算法流程

**步骤 1 — 采样错误 `e`**

对 `L×L` 的 plaquette 格子，共有：
- 水平边：`(L+1) × L` 条
- 垂直边：`L × (L+1)` 条

每条边独立以概率 `p` 采样错误 `h_errors[i,j]` 和 `v_errors[i,j]`。

**步骤 2 — 计算 syndrome `m = ∂e`**

对每个 plaquette `(i,j)`，将其四条边（上、下、左、右）的错误求和 mod 2：

```python
m_grid[i, j] = (top + bottom + left + right) % 2
```

**步骤 3 & 4 — 构建并收缩张量网络**

用 `opt_einsum` 构建如下张量网络并精确收缩，得到 `Pr(m)`：

- **边张量 `W = [1-p, p]`**：每条边对应一个权重向量，表示该边错误与否的概率幅
- **plaquette 张量 `Q^m`**（公式 E5）：`Q[i,j,k,l] = 1` 当且仅当 `(i+j+k+l) % 2 == m`，否则为 0
  - `Q^0`（黄色）：要求四条边错误数为偶数（syndrome=0）
  - `Q^1`（绿色）：要求四条边错误数为奇数（syndrome=1）

每个 plaquette 的 Q 张量按 `(top, right, bottom, left)` 顺序连接四条边的 W 张量，形成完整的因子图。

**步骤 5 — 蒙特卡洛求期望**

```
H(m) ≈ (1/N) * Σ_samples [ -log Pr(m) ]
```

### 关键函数

```
create_Q_tensor(m_val)
    输入: m_val ∈ {0, 1}
    输出: shape (2,2,2,2) 的 numpy 数组
    作用: 生成 plaquette 宇称约束张量

calculate_H_m(L=3, p=0.1, num_samples=100)
    输入: 格子大小 L、错误率 p、蒙特卡洛样本数
    输出: H(m) 的估计值（浮点数）
    作用: 蒙特卡洛 + 张量网络收缩计算 syndrome 熵
```

### 依赖库

- `numpy`：数值计算
- `opt_einsum`：自动寻找最优张量收缩路径

### 默认参数

```python
calculate_H_m(L=3, p=0.1, num_samples=10000)
```

---

## CMI.py 详细分析

计算 `I(A:C|B) = H(AB) + H(BC) - H(B) - H(ABC)`，其中：
- **A**：中心 2×2 区域
- **B**：宽度 `r` 的缓冲环（annulus）
- **C**：其余外部区域

构建顶点张量网络（RBIM 配分函数映射），边权重矩阵 `W = diag(1-p, p)`，delta 张量强制 Z2 宇称守恒（XOR=0），用 quimb BMPS 收缩后计算各区域的香农熵。

---

## MI.py 详细分析

使用 **n=2 复制技巧**计算 Renyi-2 熵：
- Nishimori 温度：`β = 0.5 * log((1-p)/p)`
- 构建 4 腿复制 RBIM 张量（bond dim=4），用 quimb PEPS + BMPS 收缩
- 通过引入 Z2 缺陷线（branch cut）区分拓扑扇区，计算 CMI proxy
- 相图：`p < 0.11`（拓扑相，CMI 指数衰减）；`p ≈ 0.11`（临界点，代数衰减）；`p > 0.11`（平凡相）

---

## 物理参数说明

| 参数 | 含义 | 典型值 |
|---|---|---|
| `L` | 格子线性尺寸（L×L plaquettes） | 3, 10, 20 |
| `p` | 物理错误率（去相干强度） | 0.05, 0.1, 0.11, 0.15 |
| `num_samples` | 蒙特卡洛采样数 | 1000–10000 |
| `max_bond` | BMPS 截断 bond 维数 | 8–64 |
| `r` / `r_width` | CMI 计算中缓冲区宽度 | 1–5 |

---

## 运行方式

```bash
# 计算 H(m)
python H_calculation.py

# 计算 CMI 标度（需要 quimb）
python CMI.py

# 计算 Renyi-2 CMI proxy（需要 quimb）
python MI.py
```

依赖安装：

```bash
pip install numpy opt_einsum quimb matplotlib
```
