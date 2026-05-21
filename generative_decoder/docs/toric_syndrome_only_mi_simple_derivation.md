# Toric Code Syndrome-Only 互信息的简单模型推导


## 1. 问题背景

考虑边长为 `L` 的 toric code。经过物理噪声后，系统产生一个 syndrome 构型。若只保留 syndrome，而不显式跟踪底层物理误差，那么一个样本就可以记为
$$
s=(s_1,s_2,\dots,s_m)\in\{0,1\}^m,
$$
其中 `m` 为 syndrome bit 的总数。

在 toric code 中，这些 syndrome bits 可以按其空间位置分成两个区域。为了讨论 bipartite mutual information，我们取一个空间切割，将 syndrome 写成
$$
s=(a,b),
$$
其中

- `a` 表示区域 `A` 上的 syndrome bits
- `b` 表示区域 `B` 上的 syndrome bits

这里 `A` 与 `B` 是对整个 syndrome 自由度的一次二分，通常可理解为沿某个方向做左右切分或上下切分。

---

## 2. 目标量：syndrome-only 的互信息

设 `q(a,b)` 是某个模型在 syndrome 空间上定义的联合分布。我们关心的量是模型分布下的互信息
$$
I_q(A;B)
=
\sum_{a,b} q(a,b)\log\frac{q(a,b)}{q(a)q(b)}.
$$

它也可以写成熵的组合：
$$
I_q(A;B)=H_q(A)+H_q(B)-H_q(A,B),
$$
其中
$$
H_q(A,B)=-\sum_{a,b}q(a,b)\log q(a,b),
$$
$$
H_q(A)=-\sum_a q(a)\log q(a),
$$
$$
H_q(B)=-\sum_b q(b)\log q(b).
$$

因此，核心问题是如何有效得到这三个熵项。

---

## 3. 自回归模型的引入

在 `syndrome-only` 设定中，我们并不直接穷举全部 syndrome 构型，而是用一个自回归概率模型来拟合 syndrome 分布。

先将全部 syndrome bits 排列成一个序列
$$
x=(x_1,x_2,\dots,x_m).
$$

一个自回归模型将联合分布写为
$$
q(x)=\prod_{t=1}^m q(x_t\mid x_{<t}),
$$
其中 `x_{<t}` 表示第 `t` 位之前的全部变量。

取对数得到
$$
\log q(x)=\sum_{t=1}^m \log q(x_t\mid x_{<t}),
$$
于是
$$
-\log q(x)=\sum_{t=1}^m -\log q(x_t\mid x_{<t}).
$$

这说明：一个样本的联合负对数概率，可以精确分解成逐位条件负对数概率之和。

---

## 4. 针对 bipartition 的两种排序

为了从自回归模型中提取区域 `A` 和 `B` 的边缘熵，我们需要利用“前缀概率”这一结构。

为此，对同一个二分 `A/B`，考虑两种变量排序：

### 4.1 `AB` 排序

把区域 `A` 的全部变量放在前面，区域 `B` 的全部变量放在后面：
$$
x^{AB}=(a_1,\dots,a_{|A|},b_1,\dots,b_{|B|}).
$$

对应的自回归模型记为
$$
q_{AB}(a,b)
=
\prod_{t=1}^{m} q_{AB}(x_t\mid x_{<t}).
$$

### 4.2 `BA` 排序

反过来，把区域 `B` 放在前面，区域 `A` 放在后面：
$$
x^{BA}=(b_1,\dots,b_{|B|},a_1,\dots,a_{|A|}).
$$

对应模型记为
$$
q_{BA}(b,a)
=
\prod_{t=1}^{m} q_{BA}(x_t\mid x_{<t}).
$$

这两种排序的目的不是改变物理问题，而是让 `A` 或 `B` 分别成为自回归前缀，从而使边缘概率可以直接读取。

---

## 5. 联合熵的推导

对 `AB` 模型而言，联合熵定义为
$$
H_{q_{AB}}(A,B)
=
-\sum_{a,b} q_{AB}(a,b)\log q_{AB}(a,b).
$$

写成期望形式：
$$
H_{q_{AB}}(A,B)
=
\mathbb{E}_{(a,b)\sim q_{AB}}
\bigl[-\log q_{AB}(a,b)\bigr].
$$

再利用自回归分解：
$$
-\log q_{AB}(a,b)
=
\sum_{t=1}^{m} -\log q_{AB}(x_t\mid x_{<t}),
$$
因此
$$
H_{q_{AB}}(A,B)
=
\mathbb{E}_{(a,b)\sim q_{AB}}
\left[
\sum_{t=1}^{m} -\log q_{AB}(x_t\mid x_{<t})
\right].
$$

这表明联合熵就是逐位条件 NLL 总和的平均值。

---

## 6. 区域 `A` 的边缘熵

现在考虑 `AB` 排序。在这种排序下，前 `|A|` 个变量恰好就是区域 `A` 的全部变量，因此
$$
q_{AB}(a)
=
\prod_{t=1}^{|A|} q_{AB}(x_t\mid x_{<t}).
$$

两边取负对数，得到
$$
-\log q_{AB}(a)
=
\sum_{t=1}^{|A|} -\log q_{AB}(x_t\mid x_{<t}).
$$

再对 `q_{AB}` 取期望：
$$
H_{q_{AB}}(A)
=
\mathbb{E}_{(a,b)\sim q_{AB}}
\bigl[-\log q_{AB}(a)\bigr]
$$
$$
=
\mathbb{E}_{(a,b)\sim q_{AB}}
\left[
\sum_{t=1}^{|A|} -\log q_{AB}(x_t\mid x_{<t})
\right].
$$

因此，在 `AB` 排序中，`A` 的边缘熵就是“前 `|A|` 个 token 的负对数概率之和”的平均值。

---

## 7. 区域 `B` 的边缘熵

若仍停留在 `AB` 排序中，则序列尾部对应的是条件概率
$$
q_{AB}(b\mid a),
$$
因为
$$
q_{AB}(a,b)=q_{AB}(a)\,q_{AB}(b\mid a).
$$

于是尾部 token 的负对数和给出的其实是
$$
-\log q_{AB}(b\mid a),
$$
它对应的期望是条件熵
$$
H_{q_{AB}}(B\mid A),
$$
而不是我们想要的边缘熵 `H_q(B)`。

因此，需要再引入 `BA` 排序。在 `BA` 排序中，区域 `B` 被放到了前缀位置，因此
$$
q_{BA}(b)
=
\prod_{t=1}^{|B|} q_{BA}(x_t\mid x_{<t}),
$$
从而
$$
-\log q_{BA}(b)
=
\sum_{t=1}^{|B|} -\log q_{BA}(x_t\mid x_{<t}).
$$

再取期望即可得到
$$
H_{q_{BA}}(B)
=
\mathbb{E}_{(b,a)\sim q_{BA}}
\bigl[-\log q_{BA}(b)\bigr]
$$
$$
=
\mathbb{E}_{(b,a)\sim q_{BA}}
\left[
\sum_{t=1}^{|B|} -\log q_{BA}(x_t\mid x_{<t})
\right].
$$

---

## 8. 互信息的组合公式

现在三个量都已经可以由模型的逐位条件概率得到：

1. 联合熵
$$
H_{AB}
:=
\mathbb{E}_{(a,b)\sim q_{AB}}
\bigl[-\log q_{AB}(a,b)\bigr]
$$

2. 区域 `A` 的边缘熵
$$
H_A
:=
\mathbb{E}_{(a,b)\sim q_{AB}}
\bigl[-\log q_{AB}(a)\bigr]
$$

3. 区域 `B` 的边缘熵
$$
H_B
:=
\mathbb{E}_{(b,a)\sim q_{BA}}
\bigl[-\log q_{BA}(b)\bigr]
$$

于是定义组合量
$$
I_{\mathrm{comp}}(A;B)=H_A+H_B-H_{AB}.
$$

如果 `q_{AB}` 与 `q_{BA}` 都较好地逼近同一个潜在 syndrome 分布 `q(a,b)`，则有近似关系
$$
I_{\mathrm{comp}}(A;B)\approx I_q(A;B).
$$

从理论上说，最理想的情况是存在同一个统一的 `q(a,b)`，使得
$$
H_A=H_q(A),\qquad H_B=H_q(B),\qquad H_{AB}=H_q(A,B),
$$
这时就严格得到
$$
I_q(A;B)=H_q(A)+H_q(B)-H_q(A,B).
$$

---

## 9. Monte Carlo 估计

上面的公式已经把互信息写成了概率论上的精确表达式，但具体数值仍需要对模型分布下的期望进行估计。

设从 `AB` 模型独立采样
$$
(a^{(1)},b^{(1)}),\dots,(a^{(N)},b^{(N)})\sim q_{AB},
$$
从 `BA` 模型独立采样
$$
(b^{(1)},a^{(1)}),\dots,(b^{(N)},a^{(N)})\sim q_{BA}.
$$

则可以构造以下估计量：
$$
\widehat{H}_{AB}
=
\frac{1}{N}\sum_{i=1}^{N} \bigl[-\log q_{AB}(a^{(i)},b^{(i)})\bigr],
$$
$$
\widehat{H}_{A}
=
\frac{1}{N}\sum_{i=1}^{N} \bigl[-\log q_{AB}(a^{(i)})\bigr],
$$
$$
\widehat{H}_{B}
=
\frac{1}{N}\sum_{i=1}^{N} \bigl[-\log q_{BA}(b^{(i)})\bigr].
$$

最终的互信息估计为
$$
\widehat{I}
=
\widehat{H}_A+\widehat{H}_B-\widehat{H}_{AB}.
$$

这就是 `syndrome-only` 路径中最自然的模型计算方式：先用自回归模型定义概率分布，再对熵项做采样平均。

---

## 10. 该方法的物理含义

在 toric code 的 `syndrome-only` 任务中，这个量衡量的是：

- 在模型分布下，区域 `A` 的 syndrome 与区域 `B` 的 syndrome 之间共享了多少信息
- 或者说，知道 `A` 的 syndrome 之后，对 `B` 的不确定性减少了多少

若互信息较大，说明模型认为跨越 cut 的 syndrome 相关性较强；若互信息较小，则说明模型认为两边更接近独立。

因此，`I_q(A;B)` 可以视为一个描述 syndrome 空间长程相关性或空间耦合强度的统计量。

---

## 11. 一个可手算的 depolarizing toy model

上面的自回归公式给出的是通用计算方式。为了理解 toric code 中 `I(A;B)` 的来源，可以看一个最小的局域模型：取 cut 两侧相邻的两个 plaquette check，它们共享一条物理边。

在 depolarizing 噪声下，每条物理边的 Pauli 错误可写成
$$
(x_e,z_e)\in\mathbb F_2^2,
$$
其中
$$
\mathbb P(0,0)=1-p,\qquad
\mathbb P(1,0)=\mathbb P(0,1)=\mathbb P(1,1)=\frac p3.
$$

对 plaquette syndrome 而言，只需要 $x_e$ 分量。因此单条边上
$$
r:=\mathbb P(x_e=1)=\frac{2p}{3},
$$
并且
$$
c:=1-2r=1-\frac{4p}{3}.
$$

### 11.1 最简共享边模型

如果忽略其它边，只保留穿过 cut 的共享边，则左右两个 syndrome bit 都等于同一个随机变量：
$$
S_A=x_e,\qquad S_B=x_e.
$$

因此二者完全相同，互信息就是单个 Bernoulli 变量的熵：
$$
I(S_A;S_B)=H_2(r),
$$
其中
$$
H_2(r)=-r\log r-(1-r)\log(1-r).
$$

代入 depolarizing 噪声得到
$$
I(S_A;S_B)
=
H_2\!\left(\frac{2p}{3}\right).
$$

这个模型过于理想化，因为真实 toric code 中每个 plaquette syndrome 是四条边错误的奇偶和，而不是只由 cut 上一条边决定。

### 11.2 两个相邻 plaquette 的四边 parity 模型

更合理的局域模型是：左右两个相邻 plaquette 共享一条边，同时各自还有三条不共享的边。记共享边错误为 $x_0$，左右私有边 parity 分别为 $U,V$，则
$$
S_A=x_0\oplus U,\qquad
S_B=x_0\oplus V.
$$

其中 $x_0$ 是 Bernoulli$(r)$，而 $U,V$ 分别是三个独立 Bernoulli$(r)$ 变量的奇偶和。

使用二元变量的 bias 表示，
$$
\mathbb E[(-1)^{x_e}]=c.
$$

于是
$$
\mu:=\mathbb E[(-1)^{S_A}]
=\mathbb E[(-1)^{S_B}]
=c^4,
$$
而
$$
\rho:=\mathbb E[(-1)^{S_A\oplus S_B}]
=c^6.
$$

因此两个 syndrome bit 的联合分布为
$$
q_{ab}
=
\mathbb P(S_A=a,S_B=b)
=
\frac14
\left[
1+\mu\bigl((-1)^a+(-1)^b\bigr)
+\rho(-1)^{a+b}
\right].
$$

也就是
$$
q_{00}=\frac{1+2\mu+\rho}{4},
$$
$$
q_{01}=q_{10}=\frac{1-\rho}{4},
$$
$$
q_{11}=\frac{1-2\mu+\rho}{4}.
$$

边缘分布为
$$
q_A(0)=q_B(0)=\frac{1+\mu}{2},
\qquad
q_A(1)=q_B(1)=\frac{1-\mu}{2}.
$$

所以该局域模型的互信息可以完全写成
$$
I_{\mathrm{pair}}(p)
=
\sum_{a,b\in\{0,1\}}
q_{ab}\log\frac{q_{ab}}{q_A(a)q_B(b)},
$$
其中
$$
\mu=\left(1-\frac{4p}{3}\right)^4,
\qquad
\rho=\left(1-\frac{4p}{3}\right)^6.
$$

展开后就是
$$
I_{\mathrm{pair}}(p)
=
q_{00}\log\frac{q_{00}}{\left(\frac{1+\mu}{2}\right)^2}
+2q_{01}\log\frac{q_{01}}{\left(\frac{1+\mu}{2}\right)\left(\frac{1-\mu}{2}\right)}
+q_{11}\log\frac{q_{11}}{\left(\frac{1-\mu}{2}\right)^2}.
$$

这里默认对数为自然对数，单位是 nats；若要以 bits 为单位，需要再除以 $\log 2$。

### 11.3 与自回归模型估计的关系

如果自回归模型只学习这一对 syndrome bits，那么理想情况下它给出的三个熵项应满足
$$
H_A=H_2\!\left(\frac{1-\mu}{2}\right),
\qquad
H_B=H_2\!\left(\frac{1-\mu}{2}\right),
$$
$$
H_{AB}=-\sum_{a,b}q_{ab}\log q_{ab},
$$
从而
$$
H_A+H_B-H_{AB}=I_{\mathrm{pair}}(p).
$$

这个 toy model 的作用是给 syndrome-only 自回归互信息一个解析 benchmark：模型采样估计出来的 `AB/BA` 熵组合，应该在这个受控二变量问题上回到上面的闭式公式。

若同时取 plaquette 与 star 两类 syndrome，并在最简共享边模型中让左右两侧都观察同一条边的 $(x_e,z_e)$，则
$$
(S_A^p,S_A^s)=(x_e,z_e),
\qquad
(S_B^p,S_B^s)=(x_e,z_e),
$$
互信息就是单条边 Pauli 噪声的熵：
$$
I=H(x_e,z_e)
=
-(1-p)\log(1-p)-p\log\frac p3.
$$

这个公式保留了 depolarizing 噪声中 $X/Y/Z$ 三类错误对 plaquette 与 star syndrome 的相关性；若把 $x$ 与 $z$ 扇区错误地当成独立 Bernoulli 噪声，就会丢掉这部分跨扇区相关。

### 11.4 从 toy model 到完整 toric code

完整 toric code 中，区域 `A` 与 `B` 的 syndrome 不是由一对 checks 给出，而是由许多局域 checks 组成。推广 toy model 的正确方式不是把单个 pair 公式机械相乘，而是把 cut 附近的物理边作为共同噪声源。

设物理错误变量分成三类：
$$
E_A,\qquad E_B,\qquad E_\partial,
$$
其中 $E_A$ 只影响区域 `A` 内部 checks，$E_B$ 只影响区域 `B` 内部 checks，$E_\partial$ 是同时影响 cut 两侧 syndrome 的边界错误变量。因为物理噪声独立，
$$
E_A\perp E_B\perp E_\partial.
$$

syndrome 可以写成
$$
S_A=f_A(E_A,E_\partial),
\qquad
S_B=f_B(E_B,E_\partial).
$$

于是有严格的条件独立关系
$$
S_A\perp S_B\mid E_\partial.
$$

这给出一个有用的上界：
$$
I(S_A;S_B)\le H(E_\partial).
$$

若只看单 CSS 扇区，cut 上每条边的相关错误是 Bernoulli$(2p/3)$，所以
$$
H(E_\partial)=|\partial E|\,H_2\!\left(\frac{2p}{3}\right).
$$

若同时保留 plaquette 与 star syndrome，则每条边的边界错误是 Pauli 变量 $(x,z)$，所以
$$
H(E_\partial)=|\partial E|
\left[
-(1-p)\log(1-p)-p\log\frac p3
\right].
$$

这个上界是严格的，但通常不是等号；真实 syndrome 还混入了区域内部 parity 噪声，所以每条边界边贡献的有效互信息会小于这个共享错误熵。

### 11.5 与 $L$ 的标度律

对 torus 上左右二分的标准切割，边界有两条长度为 $L$ 的界面，因此
$$
|\partial E|\propto 2L.
$$

如果噪声率处在有限相关长度区域，远离 cut 的 syndrome 对跨区互信息贡献指数衰减，互信息主要由 cut 附近宽度 $O(\xi(p))$ 的边界带贡献。于是完整 toric code 的 syndrome-only 互信息具有边界律
$$
I_L(A;B)
=
\alpha(p)|\partial A|+\beta(p)+o(1).
$$

对标准半系统切割，
$$
I_L(A;B)
=
2\alpha(p)L+\beta(p)+o(1).
$$

这里 $\alpha(p)$ 是单位边界长度的有效 syndrome 相关密度。toy model 中的 $I_{\mathrm{pair}}(p)$ 可以看作 $\alpha(p)$ 的最小局域近似，但一般不等于真实 $\alpha(p)$，因为完整 toric code 的边界 checks 共享更多局域约束，并且存在全局 parity 约束。

对自回归模型来说，最直接的验证方式是对多个 $L$ 训练或采样同一类模型，然后拟合
$$
\widehat I(L)=aL+b.
$$

若采用左右二分且有两条 cut，则
$$
a\approx 2\alpha(p),
\qquad
b\approx \beta(p).
$$

如果使用全部 checks 坐标而不是独立 syndrome 生成元，$\beta(p)$ 还可能包含全局 parity 约束带来的常数项；若使用独立生成元表示，这个常数项通常被规范固定掉。

---

## 12. 总结

对 toric code 的 `syndrome-only` 互信息计算，可以归纳为以下逻辑：

1. 将 syndrome 按空间切分为 `A` 与 `B`
2. 用自回归模型表示 syndrome 联合分布
3. 通过 `AB` 排序，把 `A` 变成前缀，从而得到 `H(A)` 和 `H(A,B)`
4. 通过 `BA` 排序，把 `B` 变成前缀，从而得到 `H(B)`
5. 最终用
$$
I(A;B)=H(A)+H(B)-H(A,B)
$$
组合出 bipartite mutual information

因此，这个方法的关键不在于对全部 syndrome 配置做显式求和，而在于利用自回归模型的前缀概率结构，把边缘熵和联合熵都化成可计算的负对数概率平均值。
