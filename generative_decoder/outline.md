First Step: use the true probability distribution $p(\beta, \gamma) = p(\beta_1, \beta_2, \gamma_1, \gamma_2)$ to get the CMI:
- middle cut CMI: $I_{p}=I_{p}(\beta:\gamma)$, where A part is $\beta$, B part is $\gamma$
- 1/4 cut CMI: $I_p = I_p(\beta_1: \beta_2, \gamma)$. where A part is $\beta_1$, B part is the rest
- 3/4 cut CMI: $I_p = I_p(\beta, \gamma_1: \gamma_2)$. where A part is $\beta, \gamma_1$, B part is the rest

Second Step: use the different neural network architecture to learn the truth probability distribution and plot the $n_d^{min}(L)$