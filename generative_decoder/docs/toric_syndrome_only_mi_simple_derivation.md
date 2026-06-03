# Toric Code Syndrome-Only 互信息的简单模型推导

本文给出 `toric code syndrome-only MI` 的一条工程友好的推导路径：先说明目标互信息是什么，再解释为什么需要 `AB/BA` 两种自回归排序，最后用一个可手算的局域 toy model 检查公式，并说明它如何外推为完整 Toric Code 的边界律。

本文默认使用自然对数 `log = ln`，所以熵和互信息的单位均为 **nats**。若要换成 bits，需要整体除以 `log 2`。

---

## 0. 批判者审阅摘要

原始推导的主体方向是正确的，但有几处容易导致误解：

1. **真实分布与模型分布需要隔离。**
   物理噪声诱导的真实 syndrome 分布记为 `P`，自回归模型学到的分布记为 `q`。训练代码实际估计的是模型分布下的互信息，或者更严格地说，是由 `q_AB` 与 `q_BA` 两个排序模型拼接出来的互信息代理量。

2. **`AB/BA` 两模型拼接不是自动严格等于同一个 `I_q(A;B)`。**
   只有当 `q_AB` 和 `q_BA` 都表示同一个底层联合分布 `q(a,b)` 时，
   $$
   H_A + H_B - H_{AB}
   $$
   才严格等于该 `q` 下的互信息。若二者训练误差不同，这个量应记为 `I_comp`，是一个可计算代理量。

3. **`AB` 后缀不能当作 `H(B)`。**
   在 `AB` 排序中，后半段 token 的 NLL 是
   $$
   -\log q_{AB}(b\mid a),
   $$
   其期望是条件熵 `H(B|A)`，不是边缘熵 `H(B)`。

4. **局域 toy model 的几何描述需要更精确。**
   “两个相邻 plaquette 的四边 parity 模型”这个名字容易误导。公式实际描述的是：每个 syndrome check 由 4 条边的 parity 给出，其中左右两个 checks 共享 1 条边，并且各自还有 3 条私有边。因此总共涉及 7 个边变量，而不是总共 4 条边。

5. **边界律不是有限 `L` 精确闭式。**
   $$
   I_L(A;B)=\alpha(p)|\partial A|+\beta(p)+o(1)
   $$
   是有限相关长度区域的大尺寸标度律，不是从 toy model 直接推出的全局闭式。

6. **`beta(p)` 依赖 syndrome 坐标表示。**
   若保留 full checks，全局 parity constraints 可能贡献 1 bit 或 2 bits 的常数相关；若使用 independent syndrome generators，这类冗余约束通常已被 gauge fixed 掉。

下面的修订版按这些问题重新组织推导。

---

## 1. 问题设置

考虑边长为 `L` 的 toric code。经过物理噪声后，系统产生一个 syndrome 构型。只保留 syndrome、不显式记录底层物理错误时，一个样本记为
$$
s=(s_1,s_2,\dots,s_m)\in\{0,1\}^m.
$$

这里 `m` 是所采用 syndrome 坐标中的 bit 数。需要注意：

- 若使用 **full check** 坐标，torus 上 plaquette 和 star checks 各有一条全局 parity 约束。
- 若使用 **independent syndrome generators**，则删掉冗余 checks，只保留线性独立坐标。当前仓库采用的是这种表示。

对空间做一次二分，将 syndrome 写为
$$
s=(a,b),
$$
其中：

- `a` 是区域 `A` 上的 syndrome bits；
- `b` 是区域 `B` 上的 syndrome bits。

后文中的 `A/B` 是 syndrome 坐标的空间二分，不是物理 qubit 集合的二分。

---

## 2. 目标量：真实分布与模型分布

物理噪声诱导一个真实 syndrome 分布，记为
$$
P(a,b).
$$

若直接讨论真实分布的互信息，则定义为
$$
I_P(A;B)
=
\sum_{a,b}P(a,b)\log\frac{P(a,b)}{P_A(a)P_B(b)}.
$$

实际工程中，我们通常不穷举 `P(a,b)`，而是训练自回归模型。模型分布记为 `q`，对应互信息为
$$
I_q(A;B)
=
\sum_{a,b}q(a,b)\log\frac{q(a,b)}{q_A(a)q_B(b)}.
$$

等价地，
$$
I_q(A;B)=H_q(A)+H_q(B)-H_q(A,B),
$$
其中
$$
H_q(A,B)=-\sum_{a,b}q(a,b)\log q(a,b),
$$
$$
H_q(A)=-\sum_a q_A(a)\log q_A(a),
$$
$$
H_q(B)=-\sum_b q_B(b)\log q_B(b).
$$

本文后续推导的核心是：如何利用自回归模型的前缀概率结构，高效估计这些熵项。

---

## 3. 自回归分解

将 syndrome bits 按某个顺序排列成序列
$$
x=(x_1,x_2,\dots,x_m).
$$

一个自回归模型定义
$$
q(x)=\prod_{t=1}^m q(x_t\mid x_{<t}),
$$
其中 `x_{<t}` 表示第 `t` 位之前的全部 token。

取对数：
$$
\log q(x)=\sum_{t=1}^m \log q(x_t\mid x_{<t}),
$$
因此
$$
-\log q(x)=\sum_{t=1}^m -\log q(x_t\mid x_{<t}).
$$

这一步是精确的链式分解。数值误差只来自模型近似和 Monte Carlo 采样。

---

## 4. 为什么需要 `AB` 和 `BA` 两种排序

对同一个空间二分 `A/B`，定义两种 token 顺序。

### 4.1 `AB` 排序

把区域 `A` 的全部 syndrome bits 放在前面，区域 `B` 放在后面：
$$
x^{AB}=(a_1,\dots,a_{|A|},b_1,\dots,b_{|B|}).
$$

对应模型记为
$$
q_{AB}(a,b)
=
\prod_{t=1}^{m}q_{AB}(x_t\mid x_{<t}).
$$

在该排序下，前缀概率正好是 `A` 的边缘概率：
$$
q_{AB}(a)
=
\prod_{t=1}^{|A|}q_{AB}(x_t\mid x_{<t}).
$$

所以 `AB` 排序可以直接给出：

- `H(A)`：来自前 `|A|` 个 token 的 NLL；
- `H(A,B)`：来自全序列 NLL。

### 4.2 `BA` 排序

反过来，将区域 `B` 放在前面：
$$
x^{BA}=(b_1,\dots,b_{|B|},a_1,\dots,a_{|A|}).
$$

对应模型记为
$$
q_{BA}(b,a)
=
\prod_{t=1}^{m}q_{BA}(x_t\mid x_{<t}).
$$

在该排序下，
$$
q_{BA}(b)
=
\prod_{t=1}^{|B|}q_{BA}(x_t\mid x_{<t}),
$$
所以 `BA` 排序可以直接给出 `H(B)`。

### 4.3 关键警告：`AB` 后缀不是 `H(B)`

在 `AB` 排序中，后缀 token 给出的是
$$
q_{AB}(b\mid a),
$$
因此
$$
\sum_{t=|A|+1}^{m}-\log q_{AB}(x_t\mid x_{<t})
=
-\log q_{AB}(b\mid a).
$$

其期望是
$$
H_{q_{AB}}(B\mid A),
$$
不是边缘熵 `H(B)`。

因此，不能用 `AB` 后缀替代 `BA` 前缀。

---

## 5. 三个熵项的估计

### 5.1 联合熵

对 `AB` 模型，
$$
H_{q_{AB}}(A,B)
=
\mathbb E_{(a,b)\sim q_{AB}}
\left[-\log q_{AB}(a,b)\right].
$$

利用自回归分解，
$$
H_{q_{AB}}(A,B)
=
\mathbb E_{(a,b)\sim q_{AB}}
\left[
\sum_{t=1}^{m}
-\log q_{AB}(x_t\mid x_{<t})
\right].
$$

### 5.2 区域 `A` 的边缘熵

因为 `A` 是 `AB` 排序的前缀，
$$
-\log q_{AB}(a)
=
\sum_{t=1}^{|A|}
-\log q_{AB}(x_t\mid x_{<t}).
$$

所以
$$
H_{q_{AB}}(A)
=
\mathbb E_{(a,b)\sim q_{AB}}
\left[
\sum_{t=1}^{|A|}
-\log q_{AB}(x_t\mid x_{<t})
\right].
$$

### 5.3 区域 `B` 的边缘熵

因为 `B` 是 `BA` 排序的前缀，
$$
-\log q_{BA}(b)
=
\sum_{t=1}^{|B|}
-\log q_{BA}(x_t\mid x_{<t}).
$$

所以
$$
H_{q_{BA}}(B)
=
\mathbb E_{(b,a)\sim q_{BA}}
\left[
\sum_{t=1}^{|B|}
-\log q_{BA}(x_t\mid x_{<t})
\right].
$$

---

## 6. 组合互信息：严格等式与代理量

实际代码计算的是
$$
H_{AB}:=H_{q_{AB}}(A,B),
$$
$$
H_A:=H_{q_{AB}}(A),
$$
$$
H_B:=H_{q_{BA}}(B),
$$
并组合
$$
I_{\mathrm{comp}}(A;B)
=
H_A+H_B-H_{AB}.
$$

如果存在同一个联合分布 `q(a,b)`，使得 `q_AB` 和 `q_BA` 只是该分布的两种自回归排序表示，那么
$$
H_A=H_q(A),
\qquad
H_B=H_q(B),
\qquad
H_{AB}=H_q(A,B),
$$
于是
$$
I_{\mathrm{comp}}(A;B)=I_q(A;B).
$$

但在实际训练中，`q_AB` 与 `q_BA` 是两个分别训练的模型。若二者拟合同一个真实 syndrome 分布的误差不同，则
$$
I_{\mathrm{comp}}(A;B)
$$
应被理解为一个 **两模型拼接的互信息代理量**。它在两个模型都训练充分且一致时才近似真实模型互信息或物理互信息。

---

## 7. Monte Carlo 估计器

从 `AB` 模型采样：
$$
(a^{(i)},b^{(i)})\sim q_{AB},
\qquad i=1,\dots,N,
$$
从 `BA` 模型采样：
$$
(b^{(i)},a^{(i)})\sim q_{BA},
\qquad i=1,\dots,N.
$$

构造：
$$
\widehat H_{AB}
=
\frac1N\sum_{i=1}^N
\left[-\log q_{AB}(a^{(i)},b^{(i)})\right],
$$
$$
\widehat H_A
=
\frac1N\sum_{i=1}^N
\left[-\log q_{AB}(a^{(i)})\right],
$$
$$
\widehat H_B
=
\frac1N\sum_{i=1}^N
\left[-\log q_{BA}(b^{(i)})\right].
$$

最终估计：
$$
\widehat I_{\mathrm{comp}}
=
\widehat H_A+\widehat H_B-\widehat H_{AB}.
$$

Monte Carlo 误差可用 bootstrap 估计；训练误差需要通过多个 `train_seed` 评估。

---

## 8. 物理含义

在 syndrome-only 任务中，互信息衡量的是：

- 模型认为区域 `A` 的 syndrome 与区域 `B` 的 syndrome 之间共享多少信息；
- 或者知道 `A` 之后，对 `B` 的不确定性减少多少。

若 `I(A;B)` 大，说明模型捕捉到较强的跨 cut syndrome 相关；若较小，则说明模型认为两边接近独立。

需要强调：该量是 syndrome 分布上的互信息，而不是底层物理 error chain 的互信息。

---

## 9. 可手算的 depolarizing toy model

为了给自回归互信息估计提供解析 benchmark，考虑 cut 两侧相邻的两个 plaquette checks。

depolarizing 噪声下，每条物理边的 Pauli 错误为
$$
(x_e,z_e)\in\mathbb F_2^2,
$$
且
$$
\mathbb P(0,0)=1-p,
\qquad
\mathbb P(1,0)=\mathbb P(0,1)=\mathbb P(1,1)=\frac p3.
$$

plaquette syndrome 只看 `x_e` 分量，所以
$$
r:=\mathbb P(x_e=1)=\frac{2p}{3},
\qquad
c:=1-2r=1-\frac{4p}{3}.
$$

### 9.1 最简共享边模型

若只保留一条共享边，忽略其它边，则
$$
S_A=x_e,
\qquad
S_B=x_e.
$$

于是二者完全相同：
$$
I(S_A;S_B)=H_2(r),
$$
其中
$$
H_2(r)=-r\log r-(1-r)\log(1-r).
$$

代入 depolarizing 噪声：
$$
I(S_A;S_B)
=
H_2\!\left(\frac{2p}{3}\right).
$$

这是一个上界式直觉模型，因为真实 plaquette syndrome 是周围多条边的 parity，而不是单条边。

### 9.2 相邻 plaquette 的局域 parity 模型

更接近真实几何的局域模型是：左右两个相邻 plaquette checks 共享一条边，同时每个 check 还各自有三条私有边。

注意：该模型总共涉及 7 个边变量。所谓“每个 plaquette 的四边 parity”指的是每个 check 由 4 条边参与 parity，而不是整个模型只有 4 条边。

记共享边错误为 `x_0`，左右私有边 parity 分别为 `U,V`：
$$
S_A=x_0\oplus U,
\qquad
S_B=x_0\oplus V.
$$

其中：

- `x_0` 是 Bernoulli$(r)$；
- `U` 是三条独立 Bernoulli$(r)$ 变量的 parity；
- `V` 是另外三条独立 Bernoulli$(r)$ 变量的 parity。

使用 bias 表示：
$$
\mathbb E[(-1)^{x_e}]=c.
$$

则
$$
\mu:=\mathbb E[(-1)^{S_A}]
=
\mathbb E[(-1)^{S_B}]
=
c^4,
$$
因为 `S_A` 或 `S_B` 各自包含 4 条边的 parity。

同时
$$
\rho:=\mathbb E[(-1)^{S_A\oplus S_B}]
=
c^6.
$$

这里共享边 `x_0` 在 `S_A xor S_B` 中抵消，只剩左右共 6 条私有边。

二元 Fourier 反演给出联合分布：
$$
q_{ab}
=
\frac14
\left[
1+\mu\bigl((-1)^a+(-1)^b\bigr)
+\rho(-1)^{a+b}
\right].
$$

显式写为：
$$
q_{00}=\frac{1+2\mu+\rho}{4},
$$
$$
q_{01}=q_{10}=\frac{1-\rho}{4},
$$
$$
q_{11}=\frac{1-2\mu+\rho}{4}.
$$

边缘分布为：
$$
q_A(0)=q_B(0)=\frac{1+\mu}{2},
\qquad
q_A(1)=q_B(1)=\frac{1-\mu}{2}.
$$

因此
$$
I_{\mathrm{pair}}(p)
=
\sum_{a,b\in\{0,1\}}
q_{ab}\log\frac{q_{ab}}{q_A(a)q_B(b)}.
$$

其中
$$
\mu=\left(1-\frac{4p}{3}\right)^4,
\qquad
\rho=\left(1-\frac{4p}{3}\right)^6.
$$

展开为：
$$
I_{\mathrm{pair}}(p)
=
q_{00}\log\frac{q_{00}}{\left(\frac{1+\mu}{2}\right)^2}
+2q_{01}\log\frac{q_{01}}
{\left(\frac{1+\mu}{2}\right)\left(\frac{1-\mu}{2}\right)}
+q_{11}\log\frac{q_{11}}{\left(\frac{1-\mu}{2}\right)^2}.
$$

重要极限：

- `p=0` 时 syndrome 确定，互信息为 0。
- depolarizing 单 CSS 投影的无偏点是 `p=3/4`，此时 `r=1/2`，`c=0`，所以 `mu=rho=0`，联合分布均匀独立，互信息为 0。

### 9.3 与自回归估计的对应

若自回归模型只学习这两个 syndrome bits，且模型分布等于上述 `q_ab`，则
$$
H_A=H_2\!\left(\frac{1-\mu}{2}\right),
\qquad
H_B=H_2\!\left(\frac{1-\mu}{2}\right),
$$
$$
H_{AB}=-\sum_{a,b}q_{ab}\log q_{ab}.
$$

因此
$$
H_A+H_B-H_{AB}=I_{\mathrm{pair}}(p).
$$

这个 toy model 的作用是作为解析 benchmark，而不是完整 Toric Code 的全局互信息公式。

### 9.4 同时观察 plaquette 与 star 的最简共享边模型

若左右两侧都直接观察同一条边的完整 Pauli 变量 `(x_e,z_e)`，则
$$
(S_A^p,S_A^s)=(x_e,z_e),
\qquad
(S_B^p,S_B^s)=(x_e,z_e).
$$

互信息等于单条 Pauli 噪声变量的熵：
$$
I=H(x_e,z_e)
=
-(1-p)\log(1-p)-p\log\frac p3.
$$

该式保留了 depolarizing 噪声中 `X/Y/Z` 造成的 `x,z` 相关。若错误地把 `x` 和 `z` 当作独立 Bernoulli 变量，会丢掉这部分相关。

---

## 10. 从 toy model 到完整 Toric Code

完整系统中，`A/B` 两侧 syndrome 不是单个 check，而是许多局域 checks 的集合。推广 toy model 的正确方式不是把 `I_pair(p)` 沿 cut 机械相加，而是识别共同噪声源。

把物理错误变量分为：
$$
E_A,\qquad E_B,\qquad E_\partial.
$$

其中：

- `E_A` 只影响 `A` 内部 syndrome；
- `E_B` 只影响 `B` 内部 syndrome；
- `E_\partial` 同时影响 cut 两侧 syndrome。

独立物理噪声给出
$$
E_A\perp E_B\perp E_\partial.
$$

syndrome 是局域函数：
$$
S_A=f_A(E_A,E_\partial),
\qquad
S_B=f_B(E_B,E_\partial).
$$

因此有条件独立关系：
$$
S_A\perp S_B\mid E_\partial.
$$

也就是 Markov 结构：
$$
S_A\leftarrow E_\partial\rightarrow S_B.
$$

数据处理不等式给出严格上界：
$$
I(S_A;S_B)\le H(E_\partial).
$$

单 CSS depolarizing 投影下，
$$
H(E_\partial)=|\partial E|H_2\!\left(\frac{2p}{3}\right).
$$

完整 Pauli depolarizing 边变量下，
$$
H(E_\partial)=|\partial E|
\left[
-(1-p)\log(1-p)-p\log\frac p3
\right].
$$

该上界通常不是等号，因为区域内部 parity 噪声会降低边界错误变量在 syndrome 中的可辨识度。

---

## 11. 边界律与有限相关长度

若噪声率处在有限相关长度区域，远离 cut 的 syndrome 对跨区互信息的贡献随距离指数衰减。互信息主要来自 cut 附近宽度 `O(xi(p))` 的边界带。

因此大尺寸下自然得到边界律：
$$
I_L(A;B)
=
\alpha(p)|\partial A|+\beta(p)+o(1).
$$

对 torus 上标准半系统切割，边界有两条长度为 `L` 的界面，因此
$$
|\partial A|=2L,
$$
并得到
$$
I_L(A;B)
=
2\alpha(p)L+\beta(p)+o(1).
$$

这里：

- `alpha(p)` 是单位边界长度的有效 syndrome 互信息密度；
- `beta(p)` 是次领先常数项，依赖几何、坐标表示和拟合窗口；
- `I_pair(p)` 只能看作 `alpha(p)` 的局域 sanity check，通常不等于真实 `alpha(p)`。

若使用 full check 坐标而不是 independent generators，torus 上 plaquette 与 star 的全局 parity constraints 可能贡献常数互信息。单 CSS 扇区可能是 1 bit，plaquette + star 两个扇区可能是 2 bits。当前仓库使用 independent syndrome generators，因此这类 full-check 冗余常数通常已被 gauge fixed 掉，不能自动加进 `beta(p)`。

---

## 12. 数值验证方式

对多个 `L` 训练或采样同一类模型，得到
$$
\widehat I(L)=\widehat H_A+\widehat H_B-\widehat H_{AB}.
$$

再拟合
$$
\widehat I(L)=aL+b.
$$

若采用 torus 左右半切割且有两条 cut，则
$$
a\approx 2\alpha(p),
\qquad
b\approx \beta(p).
$$

数值报告必须同时给出：

- `H_A, H_B, H_AB, I_hat`；
- Monte Carlo bootstrap 误差；
- 多个 `train_seed` 的均值与方差；
- syndrome 坐标表示是 full checks 还是 independent generators；
- 对数单位是 nats 还是 bits。

---

## 13. 总结

这条 syndrome-only 互信息计算路径可以概括为：

1. 将 syndrome 按空间切成 `A/B`。
2. 构造 `AB` 和 `BA` 两种自回归排序。
3. 用 `AB` 前缀得到 `H(A)`，用 `AB` 全序列得到 `H(A,B)`。
4. 用 `BA` 前缀得到 `H(B)`。
5. 组合
   $$
   I_{\mathrm{comp}}=H_A+H_B-H_{AB}.
   $$
6. 若 `q_AB` 与 `q_BA` 都逼近同一个底层分布，则该量近似真实 syndrome 互信息。
7. 用 pair model 闭式公式检查局域解析基准。
8. 用 Markov 结构和有限相关长度解释完整系统的边界律。

修订后的关键逻辑是：**自回归部分给出可计算的模型互信息估计器；toy model 给出解析 benchmark；完整 Toric Code 的边界律来自共同边界噪声源、局域性和有限相关长度，而不是来自 pair model 的机械相乘。**
