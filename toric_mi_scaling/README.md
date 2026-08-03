# Toric MI Scaling

`toric_mi_scaling` computes toric-code CMI/MI observables with exact
tensor-network and boundary-MPS backends.

## Install

```bash
python -m pip install -e .
python -m pip install -e '.[quimb]'  # optional experimental backend
```

## Commands

```bash
python -m toric_mi_scaling.cli.plot_cmi_vs_p --help
python -m toric_mi_scaling.cli.plot_cmi_bmps --help
python -m toric_mi_scaling.cli.make_fig3d_paper --help
```

New scans write to `artifacts/`. Curated, checked-in data is organized under
`results/publication/`, `results/final/`, `results/diagnostics/`, and
`results/manifests/`.

See [`docs/`](docs/) for methods, reproducibility, and current status.
