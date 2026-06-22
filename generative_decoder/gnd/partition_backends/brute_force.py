import itertools
import math

import torch

from module import Errormodel

from ..beta_distribution import bits_to_index
from ..exact_mi import error_configs
from ..sector_partition import sector_weights_from_log_z


class BruteForceSectorPartition:
    """Exact sector weights by physical-error enumeration.

    This backend is only suitable for tiny codes. It is intended as the
    convention and correctness reference for scalable structured backends.
    """

    def __init__(self, code, k, er, e_model, max_exact_errors=20_000_000, chunk_size=65_536):
        self.code = code
        self.k = int(k)
        self.error_model = Errormodel(er, e_model=e_model)
        self.max_exact_errors = int(max_exact_errors)
        self.chunk_size = int(chunk_size)
        self.total_errors = 4 ** int(code.n)
        if self.total_errors > self.max_exact_errors:
            raise ValueError(
                f"Refusing brute-force sector partition: 4^{code.n}={self.total_errors} "
                f"exceeds max_exact_errors={self.max_exact_errors}"
            )
        self._table = None

    def _build_table(self):
        n_beta = 2 * self.k
        table = {}
        single_probabilities = [float(item) for item in self.error_model.single_p]

        iterator = itertools.product(range(4), repeat=int(self.code.n))
        checked = 0
        while True:
            chunk = list(itertools.islice(iterator, self.chunk_size))
            if not chunk:
                break
            errors = torch.tensor(chunk, dtype=torch.float32)
            configs = error_configs(errors, self.code, self.k)
            gammas = configs[:, :-n_beta]
            betas = configs[:, -n_beta:]
            for paulis, item_gamma, item_beta in zip(chunk, gammas, betas):
                checked += 1
                gamma_key = tuple(int(item) for item in item_gamma.tolist())
                weights = table.setdefault(gamma_key, [0.0 for _ in range(2**n_beta)])
                probability = 1.0
                for pauli in paulis:
                    probability *= single_probabilities[pauli]
                weights[bits_to_index(item_beta.tolist())] += probability
        self._table = {
            key: {
                "weights": weights,
                "p_gamma": sum(weights),
                "matched_errors": sum(1 for item in weights if item > 0.0),
            }
            for key, weights in table.items()
        }
        self._checked_errors = checked

    def sector_weights(self, gamma):
        gamma_key = tuple(int(item) for item in torch.as_tensor(gamma, dtype=torch.int64).flatten().tolist())
        if self._table is None:
            self._build_table()
        record = self._table.get(gamma_key)
        if record is None:
            n_beta = 2 * self.k
            record = {
                "weights": [0.0 for _ in range(2**n_beta)],
                "p_gamma": 0.0,
                "matched_errors": 0,
            }

        log_z = [math.log(item) if item > 0.0 else float("-inf") for item in record["weights"]]
        return sector_weights_from_log_z(
            log_z,
            diagnostics={
                "backend": "brute_force",
                "total_errors_enumerated": self.total_errors,
                "checked_errors": self._checked_errors,
                "nonzero_beta_sectors": record["matched_errors"],
                "p_gamma": record["p_gamma"],
            },
        )
