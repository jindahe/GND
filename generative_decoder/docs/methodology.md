# Methodology

The decoder models `q_theta(beta, gamma)`. The built-in variable cuts are
`I(beta:gamma)`, `I(beta_1:beta_2,gamma)`, and
`I(beta,gamma_1:gamma_2)`. Sampled MI uses a discrete plug-in estimator;
exhaustive enumeration is available only for small codes.
