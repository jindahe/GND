# Original GND Layout

This directory preserves the original pre-cleanup GenerativeDecoder layout that
used to live at the workspace root.

## Contents

- `code/`: pre-generated quantum code instances
- `module/`: code definitions, error models, GF(2) helpers, and neural network
  modules
- `decoding/`: original training, decoding, code-generation, MWPM, BP+OSD, and
  circuit-related scripts

The current maintained project lives in
[`../../generative_decoder/`](/home/jinboyu/GND/generative_decoder). Prefer that
directory for new GND work.

## Running Legacy Scripts

The legacy scripts use imports such as `from module import ...` and expect
`code/`, `module/`, and `decoding/` to be siblings. Run them from this directory
so those assumptions stay true:

```bash
cd legacy/original_gnd
python decoding/code_generator.py -c_type sur -n 13 -d 3 -k 1 -seed 0
python decoding/training.py -save True -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device cpu -batch 1000 -epoch 1000 -depth 3 -width 20
python decoding/forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cpu -trials 1000 -er 0.189
```

Generated legacy outputs should stay under `decoding/net/` or
`decoding/lo_rate/`, both ignored by the workspace `.gitignore`.
