import torch

from gnd_decoder.core import MADE, NADE, TraDE_binary


def parse_dtype(name):
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {name}")


def get_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device {name} was requested but CUDA is unavailable")
    return device


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())


def build_model(config, n_bits, device, dtype):
    n_type = config["n_type"]
    if n_type == "made":
        model = MADE(
            n=n_bits,
            depth=config.get("depth", 0),
            width=config.get("width", 64),
            activator=config.get("made_activation", "tanh"),
            residual=config.get("made_residual", False),
        )
    elif n_type == "nade":
        model = NADE(
            n=n_bits,
            hidden_dim=config.get("hidden_dim", 512),
            device=device,
            dtype=dtype,
        )
    elif n_type == "trade":
        model = TraDE_binary(
            n=n_bits,
            d_model=config.get("d_model", 256),
            n_heads=config.get("n_heads", 4),
            d_ff=config.get("d_ff", 256),
            n_layers=config.get("n_layers", 1),
            device=str(device),
            dropout=0,
        )
    else:
        raise ValueError(f"Unsupported model type: {n_type}")
    return model.to(device).to(dtype)


def model_log_prob(model, n_type, batch):
    if n_type == "made":
        return model.log_prob(batch * 2 - 1)
    if n_type == "nade":
        return model.forward(batch)
    if n_type == "trade":
        return model.log_prob(batch)
    raise ValueError(f"Unsupported model type: {n_type}")


@torch.no_grad()
def sample_model(model, n_type, batch_size):
    samples = model.sample(batch_size)
    if n_type == "made":
        samples = (samples + 1) / 2
    return samples


@torch.no_grad()
def generate_beta_from_syndrome(model, n_type, syndrome, dtype, k):
    n_s = syndrome.size(0)
    device = syndrome.device
    if n_type == "made":
        condition = syndrome * 2 - 1
        generated = (model.partial_forward(n_s=n_s, condition=condition, device=device, dtype=dtype, k=k) + 1) / 2
    elif n_type in {"nade", "trade"}:
        generated = model.partial_forward(n_s=n_s, condition=syndrome, device=device, dtype=dtype, k=k)
    else:
        raise ValueError(f"Unsupported model type: {n_type}")
    return generated[:, syndrome.size(1):syndrome.size(1) + 2 * k]
