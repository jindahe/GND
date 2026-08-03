import itertools
import math

import torch

from gnd_decoder.core import mod2

from .exact_mi import logical_indices


def entropy_from_probabilities(probabilities):
    return -sum(prob * math.log(prob) for prob in probabilities if prob > 0.0)


def bits_to_index(bits):
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


def index_to_bits(index, n_bits):
    return tuple((int(index) >> shift) & 1 for shift in range(n_bits - 1, -1, -1))


def single_site_character(single_probabilities, coeff_x, coeff_z):
    value = 0.0
    for pauli, probability in enumerate(single_probabilities):
        x_bit = pauli % 2
        z_bit = pauli // 2
        parity = (int(coeff_x) * x_bit + int(coeff_z) * z_bit) % 2
        value += float(probability) * ((-1.0) ** parity)
    return value


def exact_beta_distribution(code, k, error_model):
    """Return the exact beta marginal for the GND beta convention.

    The beta variables are linear GF(2) parities of independent single-qubit
    Pauli errors. The marginal is computed by Walsh inversion over the 2k
    logical-sector bits, avoiding enumeration of all physical errors.
    """

    n_beta = 2 * int(k)
    helper = mod2(device="cpu", dtype=torch.float64)
    logical = code.logical_opt[logical_indices(k)]
    logical_binary = helper.rep(logical).to(torch.int64)
    n_qubits = int(code.n)
    single_probabilities = [float(item) for item in error_model.single_p]

    characters = {}
    for u_bits in itertools.product((0, 1), repeat=n_beta):
        u = torch.tensor(u_bits, dtype=torch.int64)
        selected = (u[:, None] * logical_binary).sum(dim=0) % 2
        logical_x = selected[:n_qubits]
        logical_z = selected[n_qubits:]
        value = 1.0
        for coeff_x, coeff_z in zip(logical_z, logical_x):
            value *= single_site_character(single_probabilities, coeff_x, coeff_z)
        characters[u_bits] = value

    distribution = {}
    normalizer = float(2**n_beta)
    for beta_bits in itertools.product((0, 1), repeat=n_beta):
        probability = 0.0
        for u_bits, character in characters.items():
            parity = sum(u_bit * beta_bit for u_bit, beta_bit in zip(u_bits, beta_bits)) % 2
            probability += ((-1.0) ** parity) * character
        distribution[beta_bits] = probability / normalizer
    return distribution


def beta_entropy(code, k, error_model):
    distribution = exact_beta_distribution(code, k, error_model)
    return entropy_from_probabilities(distribution.values()), distribution
