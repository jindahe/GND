import math
import time
from dataclasses import dataclass

import numpy as np
import torch

from module import Errormodel, mod2

from ..exact_mi import logical_indices
from ..sector_partition import sector_weights_from_log_z


@dataclass
class BinaryDenseFactor:
    scope: tuple
    values: np.ndarray


class BinaryDenseVariableEliminationSectorPartition:
    """Exact binary dense VE over Pauli X/Z bits.

    Physical Pauli errors are represented by two binary variables per qubit
    instead of one 4-state variable. This keeps CSS toric contractions narrower
    and is intended as the exact L=4 calibration backend before implementing a
    compiled transfer/trellis path for L=10/L=20.
    """

    def __init__(
        self,
        code,
        k,
        er,
        e_model,
        max_intermediate_states=100_000_000,
    ):
        self.code = code
        self.k = int(k)
        self.error_model = Errormodel(er, e_model=e_model)
        self.n_beta = 2 * self.k
        self.n_qubits = int(code.n)
        self.n_physical_bits = 2 * self.n_qubits
        self.max_intermediate_states = int(max_intermediate_states)
        self.single_log_prior = np.log(np.array([float(item) for item in self.error_model.single_p], dtype=np.float64))
        self.constraint_rows = self._build_constraint_rows()

    def _operator_binary_row(self, row):
        helper = mod2(device="cpu", dtype=torch.float64)
        binary = helper.rep(row.unsqueeze(0)).squeeze(0).to(torch.int64)
        x_part = binary[: self.n_qubits]
        z_part = binary[self.n_qubits :]
        # Constraint on error bits [x_error, z_error].
        return torch.cat([z_part, x_part]).to(torch.int64)

    def _build_constraint_rows(self):
        rows = [self._operator_binary_row(row) for row in self.code.g_stabilizer]
        rows.extend(self._operator_binary_row(row) for row in self.code.logical_opt[logical_indices(self.k)])
        return rows

    def _weight_factors(self):
        factors = []
        table = np.empty((2, 2), dtype=np.float64)
        table[0, 0] = self.single_log_prior[0]
        table[1, 0] = self.single_log_prior[1]
        table[0, 1] = self.single_log_prior[2]
        table[1, 1] = self.single_log_prior[3]
        for qubit in range(self.n_qubits):
            factors.append(BinaryDenseFactor((qubit, self.n_qubits + qubit), table.copy()))
        return factors

    def _parity_factor(self, scope, target):
        scope = tuple(scope)
        shape = tuple(2 for _ in scope)
        values = np.full(shape, -np.inf, dtype=np.float64)
        for assignment in np.ndindex(shape):
            if sum(assignment) % 2 == int(target):
                values[assignment] = 0.0
        return BinaryDenseFactor(scope, values)

    def _initial_all_sector_factors(self, gamma):
        gamma_bits = [int(item) for item in torch.as_tensor(gamma, dtype=torch.int64).flatten().tolist()]
        if len(gamma_bits) != int(self.code.g_stabilizer.size(0)):
            raise ValueError(f"Expected gamma length {self.code.g_stabilizer.size(0)}, got {len(gamma_bits)}")
        beta_variables = tuple(range(self.n_physical_bits, self.n_physical_bits + self.n_beta))
        targets = gamma_bits + [0 for _ in range(self.n_beta)]
        factors = self._weight_factors()
        for index, (row, target) in enumerate(zip(self.constraint_rows, targets)):
            scope = [bit for bit, value in enumerate(row.tolist()) if int(value)]
            if index >= len(gamma_bits):
                beta_variable = beta_variables[index - len(gamma_bits)]
                scope.append(beta_variable)
            factors.append(self._parity_factor(scope, target))
        return factors, beta_variables

    def _choose_variable(self, active_variables, factor_scopes):
        existing = {frozenset((a, b)) for scope in factor_scopes for a in scope for b in scope if a < b}
        best_key = None
        best_variable = None
        for variable in sorted(active_variables):
            involved = [scope for scope in factor_scopes if variable in scope]
            merged = set()
            for scope in involved:
                merged.update(scope)
            neighbors = sorted(item for item in merged if item != variable)
            fill_edges = 0
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1 :]:
                    if frozenset((left, right)) not in existing:
                        fill_edges += 1
            states = 1 << len(merged)
            key = (fill_edges, len(merged), states, len(involved), variable)
            if best_key is None or key < best_key:
                best_key = key
                best_variable = variable
        return best_variable

    def _align(self, factor, scope):
        if not factor.scope:
            return factor.values
        values = factor.values
        shape = []
        factor_axis = {variable: axis for axis, variable in enumerate(factor.scope)}
        for variable in scope:
            shape.append(2 if variable in factor_axis else 1)
        return values.reshape(shape)

    def _join_factors(self, factors):
        if not factors:
            return BinaryDenseFactor((), np.array(0.0, dtype=np.float64))
        scope = tuple(sorted({variable for factor in factors for variable in factor.scope}))
        states = 1 << len(scope)
        if states > self.max_intermediate_states:
            raise ValueError(
                f"Binary dense elimination intermediate has {states} states, "
                f"exceeding max_intermediate_states={self.max_intermediate_states}"
            )
        result = np.zeros(tuple(2 for _ in scope), dtype=np.float64)
        for factor in factors:
            result = result + self._align(factor, scope)
        return BinaryDenseFactor(scope, result)

    def _sum_out(self, factor, variable):
        axis = factor.scope.index(variable)
        scope = tuple(item for item in factor.scope if item != variable)
        values = np.logaddexp.reduce(factor.values, axis=axis)
        return BinaryDenseFactor(scope, values)

    def _run_elimination(self, factors, variables_to_eliminate):
        started_at = time.time()
        active_variables = set(int(item) for item in variables_to_eliminate)
        order = []
        max_scope_width = max((len(factor.scope) for factor in factors), default=0)
        max_table_size = max((factor.values.size for factor in factors), default=0)
        while active_variables:
            variable = self._choose_variable(active_variables, [factor.scope for factor in factors])
            order.append(variable)
            involved = [factor for factor in factors if variable in factor.scope]
            remaining = [factor for factor in factors if variable not in factor.scope]
            joined = self._join_factors(involved)
            max_scope_width = max(max_scope_width, len(joined.scope))
            max_table_size = max(max_table_size, int(joined.values.size))
            reduced = self._sum_out(joined, variable)
            remaining.append(reduced)
            factors = remaining
            active_variables.remove(variable)
        final_factor = self._join_factors(factors)
        return final_factor, {
            "elimination_order": order,
            "max_scope_width": max_scope_width,
            "max_table_size": max_table_size,
            "elapsed_seconds": time.time() - started_at,
            "exact": True,
            "truncated": False,
        }

    def sector_weights(self, gamma):
        factors, beta_variables = self._initial_all_sector_factors(gamma)
        final_factor, diagnostics = self._run_elimination(factors, variables_to_eliminate=range(self.n_physical_bits))
        log_z = []
        values = final_factor.values
        for sector_index in range(2**self.n_beta):
            beta_bits = tuple((sector_index >> shift) & 1 for shift in range(self.n_beta - 1, -1, -1))
            assignment = tuple(beta_bits[beta_variables.index(variable)] for variable in final_factor.scope)
            log_z.append(float(values[assignment]))
        diagnostics.update(
            {
                "backend": "binary_dense_variable_elimination",
                "approximate": False,
                "mode": "all_sector",
                "binary_variables": self.n_physical_bits,
                "beta_variables": beta_variables,
                "final_scope": final_factor.scope,
            }
        )
        return sector_weights_from_log_z(log_z, diagnostics=diagnostics)
