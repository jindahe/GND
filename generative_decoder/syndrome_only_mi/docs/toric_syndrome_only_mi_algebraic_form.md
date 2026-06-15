# Toric Code 的代数结构下 Syndrome-Only 互信息的解析形式

本文讨论一个更偏理论的问题：在 toric code 的 `syndrome-only` 设定下，能否利用其特殊的代数结构，把 bipartite mutual information
$$
I(A;B)
$$
进一步化简，或者写成关于系统尺度 $L$ 的解析表达式。

结论先行：

1. 可以把 syndrome 分布写成由 toric code 边界算子控制的精确代数表达式。
2. 可以把该分布改写成“固定边界下的闭合回路求和”或 Fourier-Walsh 形式。
3. 一般情形下，互信息不能继续简化成一个简单的闭式 $f(L)$。
4. 在大尺寸极限下，最自然的结果是边界律：
$$
I(A;B)=\alpha(p)\,|\partial A|+\beta(p)+o(1),
$$
对 torus 上左右二分的半系统切割，通常就是
$$
I(L)\approx 2\alpha(p)L+\beta(p).
$$

---

## 1. 设定

考虑边长为 $L$ 的 toric code。只看 syndrome，而不显式记录底层误差构型。

设 syndrome 自由度共有 $m$ 个，记为
$$
s=(s_1,\dots,s_m)\in\mathbb F_2^m.
$$

对空间切割 $A/B$，将 syndrome 写为
$$
s=(s_A,s_B).
$$

我们关心模型或真实分布下的互信息
$$
I(A;B)=H(A)+H(B)-H(A,B).
$$

本文只讨论：toric code 的几何与线性代数结构能否让这个量进一步化简。

---

## 2. 单 CSS 扇区中的 syndrome 作为边界

先看最干净的情形：只考虑一个二元扇区，例如只考虑会产生 plaquette syndrome 的那部分误差。

设
$$
e\in\mathbb F_2^{n}
$$
是边上的二元误差变量，$n=2L^2$。对应 syndrome 满足
$$
s=He \pmod 2,
$$
其中 $H$ 是该扇区的 parity-check 矩阵。

对 toric code 来说，这个矩阵本质上就是一个边界算子：

- 若讨论 plaquette syndrome，则 $H$ 可视作“边到面”的边界映射
- 若讨论 star syndrome，则 $H$ 可视作“边到点”的边界映射

因此，syndrome 的本质就是 error chain 的边界。

---

## 3. syndrome 分布的精确求和表达式

设每条边独立翻转，概率为 $p$。则任意误差链 $e$ 的概率为
$$
\mathbb P(e)=p^{|e|}(1-p)^{n-|e|},
$$
其中 $|e|$ 表示 $e$ 中非零边的条数。

于是 syndrome 分布精确地写成
$$
q(s)
=
\sum_{e:\,He=s}
p^{|e|}(1-p)^{n-|e|}.
$$

这已经是一个完全解析的表达式：它表示对所有边界等于 $s$ 的 error chains 的概率求和。

从 toric code 的角度看，它是“固定边界”的链和。

---

## 4. 闭合回路空间上的求和

选取一个参考误差链 $e_0(s)$，满足
$$
He_0(s)=s.
$$

那么任何满足相同 syndrome 的误差链都可以唯一写成
$$
e=e_0(s)\oplus c,
\qquad c\in\ker H.
$$

这里 $\ker H$ 表示所有闭合回路组成的线性空间，因为
$$
Hc=0.
$$

于是 syndrome 分布可改写为
$$
q(s)
=
\sum_{c\in\ker H}
p^{|e_0(s)\oplus c|}(1-p)^{n-|e_0(s)\oplus c|}.
$$

若记
$$
t=\frac{p}{1-p},
$$
则上式也可写成
$$
q(s)
=
(1-p)^n
\sum_{c\in\ker H}
t^{|e_0(s)\oplus c|}.
$$

这个表达式最直接体现了 toric code 的代数结构：

- $e_0(s)$ 固定 syndrome 的边界类
- $\ker H$ 给出所有闭合回路自由度
- 固定 syndrome 下的所有误差只是在同一边界类内加上任意闭回路

因此，从解析角度讲，syndrome 分布已经被完全化简为一个“缺陷背景下的 loop-gas 配分函数”。

---

## 5. 用 Fourier-Walsh 变换写出联合分布

对二元线性系统，一个非常有用的解析形式是 Fourier-Walsh 展开。

对任意 $u\in\mathbb F_2^m$，定义字符
$$
\chi_u(s)=(-1)^{u\cdot s}.
$$

则 syndrome 分布可写为
$$
q(s)
=
2^{-m}\sum_{u\in\mathbb F_2^m}\chi_u(s)\,\widehat q(u),
$$
其中 Fourier 系数为
$$
\widehat q(u)=\mathbb E\big[(-1)^{u\cdot s}\big].
$$

因为 $s=He$，所以
$$
u\cdot s = u\cdot He = (H^Tu)\cdot e.
$$

又由于各条边误差独立，
$$
\widehat q(u)
=
\prod_{j=1}^{n}\mathbb E\big[(-1)^{(H^Tu)_j e_j}\big].
$$

对 Bernoulli 误差，
$$
\mathbb E[(-1)^{\xi e_j}]
=
\begin{cases}
1,&\xi=0,\\[4pt]
1-2p,&\xi=1.
\end{cases}
$$

因此
$$
\widehat q(u)
=(1-2p)^{\mathrm{wt}(H^Tu)},
$$
这里 $wt$ 表示 Hamming weight。

最终得到精确表达式
$$
q(s)
=
2^{-m}\sum_{u\in\mathbb F_2^m}
(-1)^{u\cdot s}
(1-2p)^{\mathrm{wt}(H^Tu)}.
$$

这个式子非常重要，因为它把 syndrome 分布的全部信息压缩进了 $H^T u$ 的 weight 结构中。

---

## 6. 切割后的联合分布与边缘分布

现在把 syndrome 分成两块：
$$
s=(s_A,s_B),
\qquad
u=(u_A,u_B),
$$
并把 $H$ 按行分块写成
$$
H=
\begin{pmatrix}
H_A\\
H_B
\end{pmatrix}.
$$

于是联合分布为
$$
q(s_A,s_B)
=
2^{-m}
\sum_{u_A\in\mathbb F_2^{m_A}}
\sum_{u_B\in\mathbb F_2^{m_B}}
(-1)^{u_A\cdot s_A + u_B\cdot s_B}
(1-2p)^{\mathrm{wt}(H_A^T u_A + H_B^T u_B)}.
$$

边缘分布则是
$$
q_A(s_A)
=
2^{-m_A}
\sum_{u_A\in\mathbb F_2^{m_A}}
(-1)^{u_A\cdot s_A}
(1-2p)^{\mathrm{wt}(H_A^T u_A)},
$$
$$
q_B(s_B)
=
2^{-m_B}
\sum_{u_B\in\mathbb F_2^{m_B}}
(-1)^{u_B\cdot s_B}
(1-2p)^{\mathrm{wt}(H_B^T u_B)}.
$$

从而互信息可以形式上写成
$$
I(A;B)
=
\sum_{s_A,s_B}
q(s_A,s_B)
\log\frac{q(s_A,s_B)}{q_A(s_A)\,q_B(s_B)}.
$$

这已经是一个严格解析的表达式，但一般并不能再化成简单初等函数。

---

## 7. 为什么通常得不到简单的闭式 $I(L)$

虽然前面的表达式都是精确的，但要继续把互信息化到 $L$ 的简单函数形式时，会遇到两个障碍。

### 7.1 熵是对整个分布取对数后的全局量

互信息包含熵项，而熵需要对整个分布求和再取对数。即便 $q(s)$ 本身已有 Fourier 或 loop-gas 形式，
$$
H(A,B)=-\sum_{s_A,s_B}q(s_A,s_B)\log q(s_A,s_B),
$$
仍然会把简单的线性结构变成复杂的非线性组合。

### 7.2 toric code 的相关性不是单一模态控制

对一般 $p$，影响 $q(s)$ 的不只是少数几个全局拓扑模，而是大量局域闭回路的统计权重。因此互信息通常不是某个有限参数模型的一步代入结果，而是一个真正的多体统计量。

因此，一般不能期待存在一个对所有 $L$ 与所有 $p$ 都成立的简洁闭式
$$
I(L)=f(L).
$$

---

## 8. 特殊极限下的简化

虽然一般没有简单闭式，但在若干特殊极限中，可以得到更明确的结果。

### 8.1 无噪声极限 $p=0$

此时只有零误差链，因此 syndrome 恒为零：
$$
q(s)=\delta_{s,0}.
$$

于是
$$
H(A)=H(B)=H(A,B)=0,
$$
从而
$$
I(A;B)=0.
$$

### 8.2 单扇区的完全无偏点 $p=\frac12$

这时
$$
1-2p=0.
$$

Fourier 展开中除零模外，其余所有项都消失，因此在独立 syndrome 生成元坐标下，
$$
q(s)=2^{-m}.
$$

于是 $A$ 与 $B$ 独立均匀，
$$
I(A;B)=0.
$$

这里要注意：这个结论依赖于所使用的是独立生成元坐标。如果保留全部 checks 而不去掉全局冗余约束，则还会残留全局奇偶约束带来的常数相关。

### 8.3 全部 checks 表示下的全局约束常数项

若不是使用独立生成元，而是保留全部 plaquette 与全部 star checks，则存在两条全局约束
$$
\bigoplus_{\text{all plaquettes}} s_p = 0,
$$
$$
\bigoplus_{\text{all stars}} s_s = 0.
$$

若 $A$ 与 $B$ 各自都包含两类 checks 的一部分，则在“均匀允许 syndrome”的极限中，这两条全局约束各贡献 1 bit 的跨区相关，因此总互信息出现一个常数项
$$
I(A;B)=2\ \text{bits}.
$$

若只看单 CSS 扇区，则对应为
$$
I(A;B)=1\ \text{bit}.
$$

但这不是当前独立生成元表示下的结果，而是“全部 checks 坐标”下的结果。

---

## 9. 大尺寸极限下最自然的 $L$ 标度

虽然一般得不到精确闭式 $I(L)$，但从局域统计相关性的角度，最自然的结论是边界律：
$$
I(A;B)=\alpha(p)\,|\partial A|+\beta(p)+o(1).
$$

其物理含义是：

- 主要贡献来自穿过 cut 的局域相关
- 这些贡献按边界长度累积
- 还可能带有次领先常数项

对 torus 上把系统左右平分的标准切割，边界通常由两条长度约为 `L` 的界面组成，所以
$$
|\partial A|\approx 2L.
$$

于是最自然的尺度形式就是
$$
I(L)\approx 2\alpha(p)L+\beta(p).
$$

这里

- $\alpha(p)$ 依赖噪声率与所考察的 syndrome 统计结构
- $\beta(p)$ 是次领先常数项

如果采用的是独立 syndrome 生成元表示，那么由全局冗余约束引起的“纯拓扑常数项”通常已被规范固定掉，因此 $\beta(p)$ 未必具有简单的拓扑解释。

---

## 10. depolarizing 噪声下的推广

对 toric code 的 depolarizing 噪声，每条边的误差可写为两个二元变量
$$
(x_e,z_e)\in\mathbb F_2^2.
$$

plaquette syndrome 由 $x$ 分量决定，star syndrome 由 $z$ 分量决定：
$$
s_p = H_p x,
\qquad
s_s = H_s z.
$$

因此总 syndrome 分布的 Fourier 系数不再只是一个 $(1-2p)^{\mathrm{wt}(\cdot)}$，而是变为每条边上的单边特征函数乘积：
$$
q(s_p,s_s)
=
2^{-m}
\sum_{u,v}
(-1)^{u\cdot s_p + v\cdot s_s}
\prod_{e=1}^{n}
\phi\!\big((H_p^T u)_e,(H_s^T v)_e\big),
$$
其中
$$
\phi(\alpha,\beta)
=
\sum_{x,z\in\mathbb F_2}
\pi(x,z)(-1)^{\alpha x+\beta z},
$$
而 $\pi(x,z)$ 是单条边上的 Pauli 噪声分布。

对 depolarizing 噪声，
$$
\pi(0,0)=1-p,
\qquad
\pi(1,0)=\pi(0,1)=\pi(1,1)=\frac{p}{3}.
$$

于是
$$
\phi(0,0)=1,
$$
$$
\phi(1,0)=\phi(0,1)=1-\frac{4p}{3},
$$
$$
\phi(1,1)=1-\frac{2p}{3}.
$$

这就给出了 depolarizing 情况下的精确解析表达式。它依然是完全解析的，但通常仍不足以把互信息写成简单闭式。

---

## 11. depolarizing 噪声下的局域互信息闭式例子

上一节给出的是完整 toric code 的 Fourier-Walsh 表达式。为了得到一个可以完全手算的互信息公式，可以把问题缩小到 cut 附近的一对相邻 checks。

### 11.1 单边共享的最小模型

取 cut 两侧相邻的两个 plaquette check，只保留它们共享的那条物理边。对 depolarizing 噪声，
$$
\mathbb P(I)=1-p,
\qquad
\mathbb P(X)=\mathbb P(Y)=\mathbb P(Z)=\frac p3.
$$

若 plaquette syndrome 由 $x$ 分量触发，则
$$
r=\mathbb P(x=1)=\mathbb P(X)+\mathbb P(Y)=\frac{2p}{3}.
$$

在最小共享边模型中，
$$
S_A=x,
\qquad
S_B=x.
$$

因此
$$
I(S_A;S_B)=H(S_A)=H_2\!\left(\frac{2p}{3}\right),
$$
其中
$$
H_2(r)=-r\log r-(1-r)\log(1-r).
$$

这个结果说明：只要一条跨 cut 物理边同时影响左右两侧 syndrome，左右 syndrome 就会共享信息。这个模型给出的是局域相关的上界式直觉，因为它忽略了其它边带来的 parity 噪声。

如果同时保留 plaquette 与 star 两类 syndrome，并让左右两侧都观察同一条边的两种 syndrome 分量，则
$$
(S_A^p,S_A^s)=(x,z),
\qquad
(S_B^p,S_B^s)=(x,z).
$$

这时互信息等于单边 Pauli 随机变量的熵：
$$
I(A;B)=H(x,z)
=
-(1-p)\log(1-p)-p\log\frac p3.
$$

它不同于两个独立 Bernoulli 扇区熵的简单相加，因为 depolarizing 噪声中 $x$ 与 $z$ 并不独立：
$$
\mathbb P(x=1,z=1)=\mathbb P(Y)=\frac p3,
$$
而
$$
\mathbb P(x=1)\mathbb P(z=1)=\frac{4p^2}{9}.
$$

### 11.2 相邻 plaquette 的四边 parity 模型

更接近 toric code 几何的局域模型是保留两个相邻 plaquette 的全部四条边。它们共享一条边 $x_0$，左右两侧各有三条私有边。令
$$
S_A=x_0\oplus U,
\qquad
S_B=x_0\oplus V,
$$
其中 $U,V$ 分别是三条私有边 $x$ 分量的 parity。

仍记
$$
r=\frac{2p}{3},
\qquad
c=1-2r=1-\frac{4p}{3}.
$$

因为 parity 的 Fourier bias 会相乘，
$$
\mu:=\mathbb E[(-1)^{S_A}]
=\mathbb E[(-1)^{S_B}]
=c^4,
$$
并且
$$
\rho:=\mathbb E[(-1)^{S_A+S_B}]
=c^6.
$$

其中 $\rho$ 中的指数是普通整数写法；在 $\mathbb F_2$ 意义下就是 $S_A\oplus S_B$。共享边在 $S_A\oplus S_B$ 中抵消，所以只剩六条私有边贡献 bias。

由二元 Fourier 反演，
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

显式写为
$$
q_{00}=\frac{1+2\mu+\rho}{4},
\qquad
q_{01}=q_{10}=\frac{1-\rho}{4},
\qquad
q_{11}=\frac{1-2\mu+\rho}{4}.
$$

边缘分布为
$$
q_A(0)=q_B(0)=\frac{1+\mu}{2},
\qquad
q_A(1)=q_B(1)=\frac{1-\mu}{2}.
$$

因此互信息为
$$
I_{\mathrm{pair}}(p)
=
q_{00}\log\frac{q_{00}}{\left(\frac{1+\mu}{2}\right)^2}
+2q_{01}\log\frac{q_{01}}{\left(\frac{1+\mu}{2}\right)\left(\frac{1-\mu}{2}\right)}
+q_{11}\log\frac{q_{11}}{\left(\frac{1-\mu}{2}\right)^2},
$$
其中
$$
\mu=\left(1-\frac{4p}{3}\right)^4,
\qquad
\rho=\left(1-\frac{4p}{3}\right)^6.
$$

这个式子是一个真正的闭式解析结果。它不是完整 toric code 的全局互信息，但它捕捉了边界律系数 $\alpha(p)$ 的最局域来源：每一段 cut 附近都有一批共享物理边或共享局域约束，它们给出类似的局域互信息贡献。

### 11.3 推广到完整 toric code 的严格结构

完整 toric code 中，toy model 的一个 pair 要替换成沿 cut 排列的一条边界带。关键结构是：区域内部错误只影响本区域 syndrome，跨 cut 的物理错误同时影响两侧 syndrome。

把物理错误变量分成
$$
E_A,\qquad E_B,\qquad E_\partial,
$$
其中 $E_\partial$ 表示所有同时进入 $A$、$B$ 两侧 checks 的边界错误。对独立同分布噪声，
$$
E_A\perp E_B\perp E_\partial.
$$

syndrome 是局域线性映射，因此存在函数
$$
S_A=f_A(E_A,E_\partial),
\qquad
S_B=f_B(E_B,E_\partial).
$$

由此得到严格的 Markov 结构
$$
S_A\longleftarrow E_\partial\longrightarrow S_B,
$$
更准确地说，
$$
S_A\perp S_B\mid E_\partial.
$$

这个条件独立关系给出严格上界
$$
I(S_A;S_B)
\le
I(S_A;S_B,E_\partial)
=
I(S_A;E_\partial)+I(S_A;S_B\mid E_\partial)
=
I(S_A;E_\partial)
\le
H(E_\partial).
$$

也可以直接用数据处理不等式理解：两侧 syndrome 之间能共享的信息必须经过共同噪声源 $E_\partial$。

对单 CSS 扇区，边界错误是 Bernoulli$(r)$，其中
$$
r=\frac{2p}{3},
$$
所以
$$
H(E_\partial)=|\partial E|\,H_2\!\left(\frac{2p}{3}\right).
$$

对同时保留 plaquette 与 star syndrome 的 depolarizing 噪声，每条边界边携带一个 Pauli 变量 $(x,z)$，因此
$$
H(E_\partial)
=
|\partial E|
\left[
-(1-p)\log(1-p)-p\log\frac p3
\right].
$$

这些是上界而不是一般等式。等号只在最小共享边模型这类退化情形中成立；完整 toric code 中，区域内部 parity 噪声会降低两侧 syndrome 对共享边界错误的可辨识度。

### 11.4 与 $L$ 的标度律

若噪声率处在有限相关长度区域，跨区相关只来自 cut 附近宽度 $O(\xi(p))$ 的边界带。因此互信息具有边界律
$$
I_L(A;B)
=
\alpha(p)|\partial A|+\beta(p)+o(1).
$$

对 torus 上左右半系统切割，边界由两条长度为 $L$ 的界面组成，因此
$$
|\partial A|=2L
$$
在通常的 lattice 单位下成立，于是
$$
I_L(A;B)
=
2\alpha(p)L+\beta(p)+o(1).
$$

这里 $\alpha(p)$ 是单位 cut 长度的有效 syndrome 互信息密度。toy model 的
$$
I_{\mathrm{pair}}(p)
$$
是 $\alpha(p)$ 的一个局域 sanity check，而不是完整系数的严格值。它验证了三个事实：

1. 当 $p=0$ 时，syndrome 确定，局域互信息为 $0$。
2. 当单 CSS 扇区达到 $r=1/2$，即 $p=3/4$ 时，局域 syndrome 均匀且独立，局域互信息为 $0$。
3. 在中间噪声率，cut 附近共享错误产生正的局域互信息贡献。

常数项 $\beta(p)$ 依赖 syndrome 坐标选择。若保留全部 checks，torus 上 plaquette 与 star 两个全局 parity 约束可以给出 $O(1)$ 的常数互信息；若使用独立 syndrome 生成元，这类常数项通常被规范固定掉。

---

## 12. 总结

在 toric code 的代数结构下，`syndrome-only` 互信息可以精确追溯到以下几个等价层次：

1. 固定 syndrome 的误差链求和
$$
q(s)=\sum_{e:\,He=s} w(e)
$$

2. 固定边界类上的闭回路求和
$$
q(s)=\sum_{c\in\ker H} w(e_0(s)\oplus c)
$$

3. Fourier-Walsh 展开
$$
q(s)
=
2^{-m}\sum_{u}
(-1)^{u\cdot s}
(1-2p)^{\mathrm{wt}(H^Tu)}
$$

这些都属于真正的解析表达式，并且清楚体现了 toric code 的边界算子与闭回路结构。

但是，互信息作为熵的非线性组合，一般不能再进一步简化为一个对所有 $L$ 与所有噪声率都成立的简单闭式函数。

对大尺寸系统，最稳妥也最自然的结果是边界律
$$
I(A;B)=\alpha(p)\,|\partial A|+\beta(p)+o(1),
$$
也就是在标准半系统切割下
$$
I(L)\approx 2\alpha(p)L+\beta(p).
$$

因此，toric code 的特殊代数结构确实给出了 syndrome-only 互信息的精确解析表示，但它通常带来的是“可解析表述”和“可识别的尺度律”，而不是一个简单的闭式 $f(L)$。
