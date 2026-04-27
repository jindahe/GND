# GenerativeDecoder

Repository for autoregressive quantum error-correction decoders.

The original project structure is still present in the repository, but the recommended entrypoint is
the cleaned project folder [`generative_decoder/`](/home/jinboyu/GND/generative_decoder), which
reorganizes everything outside `MI_scaling/` into a smaller self-contained layout.

Link to the article: https://doi.org/10.48550/arXiv.2503.21374

## Recommended Layout

- [`generative_decoder/code/`](/home/jinboyu/GND/generative_decoder/code): pre-generated code instances
- [`generative_decoder/module/`](/home/jinboyu/GND/generative_decoder/module): core code definitions, error model, GF(2) algebra, MADE / NADE / TraDE
- [`generative_decoder/decoding/`](/home/jinboyu/GND/generative_decoder/decoding): code generation, training, forward decoding, MWPM and BP+OSD baselines

The clean copy removes duplicated or unused files from the main path, including older model variants,
ad hoc experiment scripts, and circuit-specific helpers that were not part of the code-capacity workflow.

## Quick Start

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

Run forward decoding:

```bash
cd generative_decoder/decoding
python forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cpu -trials 1000 -er 0.189
```

Run baselines:

```bash
cd generative_decoder/decoding
python mwpm.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -trials 1000
python bposd.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -trials 1000
```

More detail is in [`generative_decoder/README.md`](/home/jinboyu/GND/generative_decoder/README.md).
