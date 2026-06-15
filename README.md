# GND Workspace

This workspace contains the current Generative Decoder project, a separate
toric-code MI/CMI scaling study, and the original pre-cleanup GND layout kept
for reference.

Link to the article: https://doi.org/10.48550/arXiv.2503.21374

## Directory Map

- [`generative_decoder/`](/home/jinboyu/GND/generative_decoder): current
  Generative Decoder project. This is the recommended entrypoint for GND
  development and experiments.
- [`MI_scaling/toric_code/`](/home/jinboyu/GND/MI_scaling/toric_code): separate
  toric-code MI/CMI scaling research project, already organized as `src/`,
  `docs/`, `outputs/`, and `archive/`.
- [`legacy/original_gnd/`](/home/jinboyu/GND/legacy/original_gnd): original
  repository layout, including the former top-level `code/`, `module/`, and
  `decoding/` directories.
- [`assets/`](/home/jinboyu/GND/assets): workspace-level figures and static
  assets that are not part of a specific package.

## Recommended GND Entry Point

Use the cleaned project folder:

```bash
cd generative_decoder
```

More detail is in
[`generative_decoder/README.md`](/home/jinboyu/GND/generative_decoder/README.md).

Common commands from that directory:

```bash
python decoding/code_generator.py -c_type sur -n 13 -d 3 -k 1 -seed 0
python decoding/training.py -save True -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device cpu -batch 1000 -epoch 1000 -depth 3 -width 20
python decoding/forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cpu -trials 1000 -er 0.189
```

## Toric-Code MI/CMI Project

Use the dedicated project directory:

```bash
cd MI_scaling/toric_code
```

More detail is in
[`MI_scaling/toric_code/README.md`](/home/jinboyu/GND/MI_scaling/toric_code/README.md).

## Legacy Layout

The old top-level GND structure was moved under
[`legacy/original_gnd/`](/home/jinboyu/GND/legacy/original_gnd) to reduce root
directory noise while preserving the original files. See
[`legacy/original_gnd/README.md`](/home/jinboyu/GND/legacy/original_gnd/README.md)
before running those scripts.
