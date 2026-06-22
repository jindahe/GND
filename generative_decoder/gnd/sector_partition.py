import math
from dataclasses import dataclass


@dataclass
class SectorWeights:
    log_z: list
    posterior: list
    entropy: float
    diagnostics: dict


def logsumexp(values):
    finite = [float(item) for item in values if not math.isinf(float(item))]
    if not finite:
        return float("-inf")
    maximum = max(finite)
    return maximum + math.log(sum(math.exp(item - maximum) for item in finite))


def posterior_from_log_weights(log_z):
    total = logsumexp(log_z)
    if math.isinf(total):
        raise ValueError("All sector weights are zero")
    return [0.0 if math.isinf(float(item)) else math.exp(float(item) - total) for item in log_z]


def entropy_from_posterior(posterior):
    return -sum(prob * math.log(prob) for prob in posterior if prob > 0.0)


def sector_weights_from_log_z(log_z, diagnostics=None):
    posterior = posterior_from_log_weights(log_z)
    entropy = entropy_from_posterior(posterior)
    return SectorWeights(
        log_z=[float(item) for item in log_z],
        posterior=posterior,
        entropy=entropy,
        diagnostics=diagnostics or {},
    )
