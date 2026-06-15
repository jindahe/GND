First step: use the true probability distribution
$p(\beta,\gamma)=p(\beta_1,\beta_2,\gamma_1,\gamma_2)$ to compute ordinary
bipartite mutual information:

- middle cut MI: $I_p(\beta:\gamma)$, where side A is $\beta$ and side B is
  $\gamma$.
- 1/4 cut MI: $I_p(\beta_1:\beta_2,\gamma)$, where side A is $\beta_1$ and
  side B is the rest.
- 3/4 cut MI: $I_p(\beta,\gamma_1:\gamma_2)$, where side A is
  $(\beta,\gamma_1)$ and side B is $\gamma_2$.

Second step: train different neural architectures to learn
$p(\beta,\gamma)$, compute the same bipartite MI cuts under the learned model,
and plot the minimum model capacity $n_d^{min}(L)$ needed to match the true MI
within a fixed tolerance.
