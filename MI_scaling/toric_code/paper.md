
*Example:* Dephased toric code—Given that finite $\xi$ implies continuity of a phase, it is natural to expect that a phase transition occurs when $\xi$ diverges. We demonstrate this with a concrete example: toric code topological order subject to dephasing noise.

Let $|\text{t.c.}\rangle$ be a ground state of the toric code Hamiltonian $H_{\text{t.c.}} = -\sum_{\square} A_{\square} - \sum_{+} B_{+}$, where qubits reside on edges of an $L \times L$ square lattice, and $\square (+)$ are plaquettes (vertices). The two terms are $A_{\square} \equiv \prod_{i \in \square} X_i$ and $B_{+} \equiv \prod_{i \in +} Z_i$ and all mutually commute. Thus the ground state satisfies all terms, i.e., $A_{\square}|\text{t.c.}\rangle = B_{+}|\text{t.c.}\rangle = |\text{t.c.}\rangle$.

> **FIG. 3.** CMI of dephased toric code state. (a) Partition with $A$ fixed and varying $r$ (width of $B$, $C$). (b) $I(A:C|B)$ peaks around $p_c \approx 0.11$ and peak size decays with $r$. (c) Finite-size collapse with the scaling ansatz Eq. (11), with $(p_c, \nu, \alpha) = (0.11, 1.8, 1.1)$. (d) Above ($p = 0.15$) or below ($p = 0.05$) the critical point, CMI decays exponentially with $r$, in contrast to power-law decay at the critical point $p_c \approx 0.11$.

After applying dephasing noise $\mathcal{E}_p[\cdot] = (1-p)(\cdot) + pZ(\cdot)Z$ on each qubit, we obtain a mixed-state $\rho_p \equiv \mathcal{E}_p^{\otimes L^2}[|\text{t.c.}\rangle\langle\text{t.c.}|]$. Physically this corresponds to applying a Pauli-Z gate independently on each qubit with probability $p$. The channel can be realized by evolving the Lindbladian $\mathcal{L}[\rho] = \sum_i \frac{1}{2}(Z_i \rho Z_i - \rho)$ for time $t_p = -\ln(1-2p)$, with $t = \infty$ corresponding to $p = 0.5$.

The mixed state is concisely described by an anyon distribution. We say a plaquette $\square$ is occupied by an anyon if $A_{\square} = -1$ (before decoherence, the ground state has no anyons). Once a $Z$ operator acts on an edge, it flips the anyon occupancy on the two adjacent plaquettes.

Thus the reduced density matrix for a simply connected region $Q$ is

$$\rho_{p,Q} = \sum_{\mathbf{m}_Q} \text{Pr}(\mathbf{m}_Q)\Pi_{\mathbf{m}_Q}, \quad (8)$$

where $\text{Pr}(\mathbf{m}_Q) \equiv \sum_{\mathbf{e}} p^{|\mathbf{e}|} (1-p)^{|Q|-|\mathbf{e}|} \delta(\mathbf{m}_Q = \partial \mathbf{e})$, and the binary vector $\mathbf{m}_Q$ indicates the anyon configuration of plaquettes within $Q$. $\Pi_{\mathbf{m}_Q}$ is the maximally mixed state that has anyon configuration $\mathbf{m}_Q$ and satisfies $B_+ = 1$ for all vertices.

One can show that region $Q$'s von Neumann entropy is

$$S(\rho_{Q,p}) = S(\rho_{Q,0}) + H(\mathbf{m}_Q), \quad (9)$$

$H(\mathbf{m}) \equiv -\sum_{\mathbf{m}} \text{Pr}(\mathbf{m}) \log \text{Pr}(\mathbf{m})$ is the Shannon entropy of the anyon distribution $\text{Pr}(\mathbf{m})$. If $Q$ is not simply connected and contains a hole denoted $\Gamma$, the rhs of Eq. (9) is replaced by $H(\mathbf{m}_Q, \pi(\mathbf{m}_\Gamma))$, with $\pi(\mathbf{m}_\Gamma)$ being the parity of anyon number within $\Gamma$ (see SM for details). Thus for an annulus-shaped $A$, $B$, $C$ partition [Fig. 3(a)],

$$I(A:C|B) = H(\mathbf{m}_{BC}, \pi(\mathbf{m}_A)) - H(\mathbf{m}_{ABC}) - H(\mathbf{m}_B, \pi(\mathbf{m}_A)) + H(\mathbf{m}_{AB}). \quad (10)$$

We simulate $I(A:C|B)$ in this geometry for various system sizes using tensor network techniques (see SM for details).

We first focus on the parameter regime around the presumed critical point $p_c \approx 0.11$. Our results (Fig. 3) show that for any system size $r$, CMI has a peak around $p_c$ with height decreasing with $r$. Furthermore, data from different system sizes collapse with the scaling ansatz:

$$I(A:C|B) = r^{-\alpha} \Phi((p-p_c)r^{1/\nu}) \quad (11)$$

by choosing $p_c = 0.11$, $\alpha = 1.1$, and $\nu = 1.8$. In particular, when $p = p_c$, CMI decays as a power law with $r$, in contrast to mutual information and conventional correlation functions.

Next, to verify that the state is FML away from the critical point $p_c$, we pick two representative points below and above the threshold: $p = 0.05 < p_c$ and $p = 0.15 > p_c$. As shown in Fig. 3(d), at both points CMI decays exponentially. These observations imply that in the large $r$ limit

$$I(A:C|B) \simeq \begin{cases} e^{-r/\xi(p)} & p \neq p_c \\ r^{-\alpha} & p = p_c, \end{cases} \quad (12)$$

where $\xi(p)$ diverges near the critical point as $\xi \propto |p - p_c|^{-\nu}$.

Because the Markov length remains finite in $p \in [0, p_c)$ and $p \in (p_c, 0.5]$, these intervals constitute two mixed-state phases. The former is a topologically ordered phase containing $|\text{t.c.}\rangle$, and the latter is a trivial phase containing $\rho_{0.5} \propto \sum_{s \in \text{loops}} |s\rangle\langle s|$, i.e., a classical uniform distribution of all closed-loop spin configurations. This state can be obtained from a product state $|\mathbf{0}\rangle\langle\mathbf{0}|$ by applying $\mathcal{G}_\square[\cdot] \equiv \frac{1}{2}(\cdot) + \frac{1}{2}A_\square(\cdot)A_\square$ on each plaquette $\square$ of the lattice; thus it belongs to a trivial phase.

One can understand the simulated CMI's behavior theoretically by mapping it to the RBIM along the Nishimori line. Each term in Eq. (10) can be mapped to a free energy in RBIM (see SM for detailed mapping and RBIM definition). For instance, $H(\mathbf{m}_{AB}) = \overline{F}_{\text{RBIM}, p}(AB) + c_1|AB| + c_2$, where $\overline{F}_{\text{RBIM}, p}(AB)$ is the disorder-averaged free energy of the RBIM defined on region $AB$'s dual lattice, and $c_{1,2}$ are constants. For non-simply connected regions $B$ and $BC$, the central hole $A$ is treated as a single dual lattice site in the corresponding RBIM. Since $A$ is $O(1)$ sized, from a coarse-grained point of view the CMI is

$$I(A:C|B) = \overline{F}_{\text{def}}(4r) - \overline{F}_{\text{def}}(2r), \quad (13)$$

where $\overline{F}_{\text{def}}(x)$ is the free energy cost of introducing a point defect at the center of RBIM on an $x \times x$ lattice. The two terms come from $H(\pi(\mathbf{m}_A), \mathbf{m}_{BC}) - H(\mathbf{m}_{ABC})$ and $H(\pi(\mathbf{m}_A), \mathbf{m}_B) - H(\mathbf{m}_{AB})$, respectively. The RBIM has ferromagnetic and paramagnetic phases, separated by a critical point presumably described by a conformal field theory, where $\overline{F}_{\text{def}}(r)$ has a subleading piece decaying as a power law with $r$. Thus we expect the scaling form Eq. (11) to originate from the RBIM critical point. As the correlation length is the only length scale near a critical point, we expect that the Markov length $\xi$ can be identified with the RBIM's correlation length. We note however that our fitted exponent $\nu \approx 1.8$ deviates from $\nu_{\text{RBIM}} \approx 1.5$ reported in [57], and we leave this discrepancy, likely due to finite-size effects, for future work.

***

### 附录部分提取

### Appendix D: Derivation of Eq.(9)

Suppose $Q$ is a non-simply-connected region of the toric code ground states surrounding a hole.

> **Figure 4.** Illustration of a non-simply connected region $Q$. Only qubits (edges) that belong to $Q$ is drawn. Supports of the two non-local operators $A_{\widetilde{\square}}$ and $B_{\widetilde{+}}$ surrounding the hole are denoted with green and blue edges, respectively.

Before the dephasing, the reduced density operator on $Q$ is:

$$\rho_{0,Q} = \text{tr}\,(|\text{t.c.}\rangle\langle\text{t.c.}|) = \left(\frac{1+A_{\widetilde{\square}}}{2}\right)\left(\frac{1+B_{\widetilde{+}}}{2}\right) \prod_{\square \text{ within } Q} \left(\frac{1+A_\square}{2}\right) \prod_{+ \text{ within } Q} \left(\frac{1+B_+}{2}\right) \quad \text{(D1)}$$

where $A_{\widetilde{\square}}$ is a $X$-loop operator acting on green edges (see Fig.4), and $B_{\widetilde{+}}$ is a $Z$-loop operator acting on blue edges. The two terms show up because $Q$ is not simply-connected. We notice that each factor in the expression is a projector operator and all factors commute with each other.

Once a $Z$ operator acts on an edge, anyon occupancies of the two adjacent plaquettes will be flipped. Thus if we use the binary vector $\mathbf{e}$ to indicate the set of edges that are acted by $Z$, plaquettes with a net anyon (indicated with a binary vector $\mathbf{m}$) are those intersects odd number of times with $\mathbf{e}$. We denote this relation as $\mathbf{m} = \partial \mathbf{e}$. The dephased state is the weighted mixture of states result from all the possible $\mathbf{e}$s:

$$\rho_{p,Q} = \frac{1}{2^{z_Q}} \sum_{\mathbf{m}} \text{Pr}(\mathbf{m}) \left(\frac{1+B_{\widetilde{+}}}{2}\right) \left(\frac{1+(-1)^{m_{\widetilde{\square}}}A_{\widetilde{\square}}}{2}\right) \prod_{\square \text{ within } Q} \left(\frac{1+(-1)^{m_\square}A_\square}{2}\right) \prod_{+ \text{ within } Q} \left(\frac{1+B_+}{2}\right)$$
$$\equiv \frac{1}{2^{z_Q}} \sum_{\mathbf{m}} \text{Pr}(\mathbf{m})\Pi_{\mathbf{m}} \quad \text{(D2)}$$

where $\text{Pr}(\mathbf{m}) = \sum_{\mathbf{e}} p^{|\mathbf{e}|}(1-p)^{|Q|-|\mathbf{e}|} \delta(\partial \mathbf{e} = \mathbf{m})$. Noticing that each $\Pi_{\mathbf{m}}$ is a projector and $\Pi_{\mathbf{m}}\Pi_{\mathbf{m}'} = 0$ when $\mathbf{m} \neq \mathbf{m}'$, we obtain the expression for $\rho_{p,Q}$' s von Neumann entropy:

$$S(\rho_{p,Q}) = -\text{tr}(\rho_{p,Q} \log \rho_{p,Q}) = S(\rho_{0,Q}) + H(\mathbf{m}) \quad \text{(D3)}$$

Which is the Eq.(9) in the maintext. We remark that there is a small difference between the notation here and that adapted in the maintext: In the maintext $\mathbf{m}$ represents only anyon configuration of unit plaquettes within $Q$, and the net anyon number in the big plaquette (*i.e.* the green plaquette above) is denoted with $\pi(\mathbf{m}_\Gamma)$. But here we use $\mathbf{m}$ to denote both.

***

### Appendix E: Tensor network technique for simulating $H(\mathbf{m})$

In order to simulate $H(\mathbf{m})$, we first rewrite it as a sample averaged quantity:

$$H(\mathbf{m}) = - \mathbb{E}_{\mathbf{m}\sim \text{Pr}(\mathbf{m})} [\log \text{Pr}(\mathbf{m})]. \quad \text{(E1)}$$

Anyon configuration $\mathbf{m}$ can be efficiently sampled by first sampling $\mathbf{e}$, which follows a product distribution, then calculating $\mathbf{m}$ as $\mathbf{m} = \partial \mathbf{e}$. However a bruteforce evaluation of $\text{Pr}(\mathbf{m})$ is not easy because it requires enumerating all the $\mathbf{e}$s that can produce $\mathbf{m}$, which leads to exponentially many terms.

To circumvant this difficulty, we represent $\text{Pr}(\mathbf{m})$ as a two dimensional tensor network. For instance, for the following anyon configuration $\mathbf{m}$ (plaquettes holding anyons are shaded):

*(Illustration of a square grid with some shaded plaquettes)* $\quad \text{(E2)}$

its probability can be expressed as:

$$\text{Pr}(\mathbf{m}) = \sum_{\mathbf{e}} p^{|\mathbf{e}|}(1-p)^{|Q|-|\mathbf{e}|} \delta(\partial \mathbf{e} = \mathbf{m}) = [\text{Tensor Network Representation}] \quad \text{(E3)}$$

where each two-leg circle tensor $T_{s_1 s_2}$, $s_{1,2} \in \{0, 1\}$ is used for assigning weights:

$$T_{s_1 s_2} = \delta(s_1 = s_2) p^{s_1}(1-p)^{s_1} \quad \text{(E4)}$$

and each $q$-leg square tensor $Q^s_{s_1, \dots, s_q}$, $s_i \in \{0, 1\}$ is used for imposing parity constraint on each plaquette:

$$Q^s_{s_1, \dots, s_q} = \delta\left(\sum_{i=1}^q s_i = s \bmod 2\right). \quad \text{(E5)}$$

In the tensor network image above, $Q^0$s and $Q^1$s are drawn with yellow and green squares, respectively. Correctness of the tensor network representation can be explicitly checked by expanding all the tensors. The 2D tensor network can be evaluated approximately and efficiently using the boundary matrix product state (bMPS) method.

***

### Appendix F: Details on numerical simulation in Fig.3

For numerical results presented in Fig.3, regions $A$, $B$ and $C$ are taken as follows (the plotted figure correponds to $r = 2$):

*(Illustration of grid regions with nested perimeters: $A$ [inner, orange], $B$ [middle, green], $C$ [outer, blue], showing width $r$)*

Edges belong to different regions are indicated with different colors. When varying $r$, the region $A$ remains unchanged. In Figs.3(b, c), each data point is averaged over at least $3.5 \times 10^4$ samples. In Fig.3(d), each data point is averaged over at least $6 \times 10^6$ samples.

***

### Appendix G: Numerical result for another geometry

In Fig.3(d), we have seen that $I(A:C|B)$ decays exponentially with the width of $B$. But since there the width of $B$ and the width of $C$ are tied (we refer to this as geom1 in this appendix), it is unclear whether the behavior depends on this particular choice of geometry. In order to settle this, in this appendix we perform the same simulation but with $r_C$ fixed (referred to as geom2), and compare it with data from geom1. This also leads to a change of total system, compared to geom1. Numerical result is shown in Fig.5. We observe that changing $r_C$ from $r$ to a constant value hardly changes the value of CMI, and in particular does not change the decaying exponent.

> **Figure 5.** CMI comparison between geom1 ($r_C = r_B$) and geom2 ($r_C = 5$).
> *(Plot displays $I(A:B|C)$ on logarithmic y-axis vs. $r$ on x-axis. Data points are plotted for $p=0.05$ and $p=0.15$ comparing geom1 and geom2, showing overlapping linear decay lines)*