# Generative Decoder

`gnd_decoder` trains autoregressive models for the joint stabilizer-code
distribution `p(beta, gamma)`, evaluates decoder performance, and estimates
variable-cut mutual information.

## Install

```bash
python -m pip install -e .
```

## Commands

```bash
python -m gnd_decoder.cli.generate_code --help
python -m gnd_decoder.workflows.datasets --help
python -m gnd_decoder.workflows.train --help
python -m gnd_decoder.workflows.evaluate_decoder --help
python -m gnd_decoder.workflows.evaluate_cut_mi --help
```

Relative CLI paths resolve from this project directory. Use an absolute path to
read or write outside the project. Checked-in code instances are in
`data/code_instances/`; generated data and models belong in `artifacts/`.

See [`docs/`](docs/) for the model definition, workflow details, and current
limitations.
