# Generative Decoder Environment Guide

This document records the validated runtime setup for `generative_decoder`,
including CPU usage, GPU usage, and the current Codex sandbox limitation.

## Validated Host Setup

Validated on 2026-04-27 with:

- Host GPU: `NVIDIA H100 PCIe`
- Host driver: `575.57.08`
- Host `nvidia-smi` CUDA version: `12.9`

## Recommended Conda Environments

Two environments were relevant during validation:

- `ai-env`
  - Python: `3.11.15`
  - PyTorch: `2.11.0+cu130`
  - Status: works on CPU, does not initialize GPU on this host
- `ai-env-cu128`
  - Python: `3.11.15`
  - PyTorch: `2.10.0+cu128`
  - Status: validated on both CPU and GPU on this host

The key compatibility point is that this host currently exposes CUDA driver
capability `12.9`, so the `cu130` PyTorch build is too new for GPU use here.
The `cu128` build works with the current driver.

## Recommended Default

Use `ai-env-cu128` for reproducible project work:

```bash
conda activate ai-env-cu128
```

If you prefer not to activate the environment in-shell, use the interpreter
directly:

```bash
/home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python
```

## Quick Verification

### GPU Availability

Run:

```bash
/home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

Expected shape of output on this host:

- `2.10.0+cu128`
- `12.8`
- `True`
- `1`
- `NVIDIA H100 PCIe`

### Minimal CUDA Tensor Check

Run:

```bash
/home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python -c "import torch; x=torch.tensor([1.0,2.0], device='cuda'); y=x.square().sum(); print(x.device); print(y.item())"
```

Expected output:

- `cuda:0`
- `5.0`

## Project Smoke Tests

All commands below were validated from:

```bash
cd /home/jinboyu/GND/generative_decoder/decoding
```

### CPU Smoke Test

Training:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128 /home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python training.py -save True -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device cpu -batch 8 -epoch 1 -trials 8 -depth 1 -width 4
```

Forward decoding:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128 /home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cpu -trials 8 -er 0.189
```

### GPU Smoke Test

Training:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128 /home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python training.py -save True -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -er 0.189 -device cuda:0 -batch 8 -epoch 1 -trials 8 -depth 1 -width 4
```

Forward decoding:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128 /home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python forward_decoding.py -n_type made -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -device cuda:0 -trials 8 -er 0.189
```

MWPM baseline:

```bash
env MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128 /home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python mwpm.py -c_type sur -n 13 -d 3 -k 1 -seed 0 -e_model dep -trials 8
```

## Installing the Validated GPU Environment

If `ai-env-cu128` does not exist on a new machine, create it by cloning
`ai-env` and replacing the PyTorch stack:

```bash
conda create -n ai-env-cu128 --clone ai-env -y
```

Then install the validated CUDA 12.8 build of PyTorch:

```bash
/home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python -m pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
```

If the cloned environment still contains an incompatible CUDA 13 stack, remove
it first:

```bash
/home/jinboyu/miniconda3/envs/ai-env-cu128/bin/python -m pip uninstall -y torch triton cuda-bindings cuda-toolkit nvidia-cublas nvidia-cuda-cupti nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cudnn-cu13 nvidia-cufft nvidia-cufile nvidia-curand nvidia-cusolver nvidia-cusparse nvidia-cusparselt-cu13 nvidia-nccl-cu13 nvidia-nvjitlink nvidia-nvshmem-cu13 nvidia-nvtx
```

## Known Issues

### `ai-env` GPU Failure

`ai-env` currently contains `torch 2.11.0+cu130`. On this host that build
fails to initialize CUDA because the driver capability is below CUDA 13.

Typical symptoms:

- `torch.cuda.is_available()` returns `False`
- warnings mention driver version `12090`
- the GPU is visible to the OS but not usable by PyTorch

### Matplotlib Cache Warning

In restricted environments, imports may warn that
`~/.config/matplotlib` is not writable. This does not block decoding, but it is
cleaner to set:

```bash
export MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128
```

or prefix individual commands with `env MPLCONFIGDIR=/tmp/matplotlib-ai-env-cu128`.

### Codex Sandbox Limitation

Inside the default Codex sandbox, GPU access is not available even when the host
GPU is healthy.

Observed reason during validation:

- the session is started under `bwrap`
- `/dev` is replaced with a minimal virtual device tree
- `/dev/nvidia*` is not present inside the sandbox

Implication:

- GPU commands can work when executed outside the sandbox
- GPU commands cannot work inside the default sandbox without changing the
  Codex runner configuration

To make GPU available inside such a sandbox, the runtime would need to expose
at least:

- `/dev/nvidia0`
- `/dev/nvidiactl`
- `/dev/nvidia-uvm`
- `/dev/nvidia-uvm-tools`
- `/dev/nvidia-modeset`

and the relevant NVIDIA user-space libraries such as:

- `libcuda.so.1`
- `libnvidia-ml.so.1`

## Notes on Project Files

During validation, `forward_decoding.py` was already adjusted to match the
training checkpoint naming convention and to load checkpoints compatibly under
newer PyTorch defaults.
