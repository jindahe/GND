# Generative Decoder (Clean Layout)

This folder is a cleaned and self-contained version of the repository outside `MI_scaling/`.
It keeps the code-capacity decoding workflow and removes experimental or duplicated files that
were not part of the main training / inference path.

## Kept Scope

- `code/`: pre-generated code instances
- `module/`: core code definitions, GF(2) algebra, error model, MADE / NADE / TraDE
- `decoding/code_generator.py`: generate and save code instances
- `decoding/training.py`: train MADE / NADE / TraDE decoders
- `decoding/forward_decoding.py`: load a trained decoder and evaluate logical error rate
- `decoding/mwpm.py`: MWPM baseline
- `decoding/bposd.py`: BP+OSD baseline

## Removed From This Clean Copy

- duplicated / older model implementations: `module/model.py`, `module/net.py`, `module/graph_generator.py`
- one-off or experimental scripts: `decoding/Block_training.py`, `decoding/Ctrain.py`
- circuit-level and hard-coded simulation scripts: `decoding/cir_*.py`, `decoding/rep_cir.py`
- timing-only helper: `decoding/time.py`
- QCC circuit benchmarking helpers: `module/benchmarkqcc.py`, `module/qcc_circuit.py`

## Typical Usage

Generate a surface code:

```bash
cd generative_decoder/decoding
python code_generator.py -c_type sur -n 13 -d 3 -k 1 -seed 0
```

Train a MADE decoder:

```bash
cd generative_decoder/decoding
python training.py -save True -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device cpu -batch 1000 -epoch 1000 -depth 3 -width 20
```

Run forward decoding with a saved checkpoint:

```bash
cd generative_decoder/decoding
python forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cpu -trials 1000 -er 0.189
```

Check MWPM or BP+OSD baselines:

```bash
cd generative_decoder/decoding
python mwpm.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -trials 1000
python bposd.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -trials 1000
```

## Notes

- Saved checkpoints are written under `generative_decoder/net/code_capacity/`.
- The clean copy intentionally preserves the original tensor / Pauli representation and CLI flags.
- `training.py` and `forward_decoding.py` include minor fixes versus the original scripts:
  stable project-root path handling, correct `trade` / `nade` branching, and simpler checkpoint handling.
- Environment and GPU setup notes are documented in `ENVIRONMENT.md`.
