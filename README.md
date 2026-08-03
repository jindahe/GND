# GND Research Workspace

This repository contains two independent Python projects:

- [`generative_decoder`](generative_decoder/README.md): generative decoding and
  mutual-information workflows over `p(beta, gamma)`.
- [`toric_mi_scaling`](toric_mi_scaling/README.md): tensor-network CMI/MI scaling for
  dephased toric-code states.

Install either project from the repository root:

```bash
python -m pip install -e ./generative_decoder
python -m pip install -e ./toric_mi_scaling
```

Each project owns its code, documentation, checked-in data/results, and ignored
runtime artifacts. See its README for commands and reproducibility details.
