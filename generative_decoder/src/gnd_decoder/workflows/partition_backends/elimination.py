import math
import time
from dataclasses import dataclass

import torch

from gnd_decoder.core import Errormodel, mod2

from ..beta_distribution import index_to_bits
from ..exact_mi import logical_indices
from ..sector_partition import sector_weights_from_log_z


def logaddexp(left, right):
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def normalize_scope(scope):
    return tuple(sorted(set(int(item) for item in scope)))


@dataclass
class SparseFactor:
    scope: tuple
    table: dict


class VariableEliminationSectorPartition:
    """Exact sector weights by sparse variable elimination.

    Physical Pauli errors are 4-state variables. Syndrome and logical-sector
    constraints are parity factors. This backend is exact for untruncated
    contractions and is intended to calibrate scalable tensor/transfer-matrix
    backends on small toric codes.
    """

    def __init__(
        self,
        code,
        k,
        er,
        e_model,
        max_intermediate_states=5_000_000,
        order_method="min_fill",
    ):
        self.code = code
        self.k = int(k)
        self.error_model = Errormodel(er, e_model=e_model)
        self.n_beta = 2 * self.k
        self.n_variables = int(code.n)
        self.single_log_prior = [math.log(float(item)) for item in self.error_model.single_p]
        self.pauli_bits = [(pauli % 2, pauli // 2) for pauli in range(4)]
        self.max_intermediate_states = int(max_intermediate_states)
        self.order_method = order_method
        self.base_constraints = self._build_base_constraints()

    def _operator_binary_row(self, row):
        helper = mod2(device="cpu", dtype=torch.float64)
        return helper.rep(row.unsqueeze(0)).squeeze(0).to(torch.int64)

    def _constraint_from_operator(self, row, target):
        binary = self._operator_binary_row(row)
        x_coeff = binary[: self.n_variables]
        z_coeff = binary[self.n_variables :]
        variables = []
        coefficients = []
        for variable in range(self.n_variables):
            coeff = (int(x_coeff[variable].item()), int(z_coeff[variable].item()))
            if coeff != (0, 0):
                variables.append(variable)
                coefficients.append(coeff)
        return {
            "variables": tuple(variables),
            "coefficients": tuple(coefficients),
            "target": int(target),
        }

    def _build_base_constraints(self):
        constraints = []
        for row in self.code.g_stabilizer:
            constraints.append(self._constraint_from_operator(row, 0))
        for row in self.code.logical_opt[logical_indices(self.k)]:
            constraints.append(self._constraint_from_operator(row, 0))
        return constraints

    def _constraints_for_targets(self, gamma, beta_bits):
        gamma = [int(item) for item in torch.as_tensor(gamma, dtype=torch.int64).flatten().tolist()]
        if len(gamma) != len(self.code.g_stabilizer):
            raise ValueError(f"Expected gamma length {len(self.code.g_stabilizer)}, got {len(gamma)}")
        targets = gamma + [int(item) for item in beta_bits]
        return [
            {
                "variables": constraint["variables"],
                "coefficients": constraint["coefficients"],
                "target": target,
            }
            for constraint, target in zip(self.base_constraints, targets)
        ]

    def _syndrome_constraints_for_gamma(self, gamma):
        gamma = [int(item) for item in torch.as_tensor(gamma, dtype=torch.int64).flatten().tolist()]
        if len(gamma) != len(self.code.g_stabilizer):
            raise ValueError(f"Expected gamma length {len(self.code.g_stabilizer)}, got {len(gamma)}")
        return [
            {
                "variables": constraint["variables"],
                "coefficients": constraint["coefficients"],
                "target": target,
            }
            for constraint, target in zip(self.base_constraints[: len(gamma)], gamma)
        ]

    def _pauli_parity(self, pauli, coefficient):
        x_bit, z_bit = self.pauli_bits[pauli]
        coeff_x, coeff_z = coefficient
        return (coeff_x * z_bit + coeff_z * x_bit) % 2

    def _unary_factors(self):
        factors = []
        for variable in range(self.n_variables):
            table = {(state,): self.single_log_prior[state] for state in range(4)}
            factors.append(SparseFactor((variable,), table))
        return factors

    def _constraint_factor(self, constraint):
        scope = tuple(constraint["variables"])
        coefficients = tuple(constraint["coefficients"])
        target = int(constraint["target"])
        table = {}

        def visit(position, assignment, parity):
            if position == len(scope):
                if parity == target:
                    table[tuple(assignment)] = 0.0
                return
            coeff = coefficients[position]
            for state in range(4):
                assignment.append(state)
                visit(position + 1, assignment, (parity + self._pauli_parity(state, coeff)) % 2)
                assignment.pop()

        visit(0, [], 0)
        return SparseFactor(scope, table)

    def _logical_output_factor(self, logical_index, beta_variable):
        constraint = self.base_constraints[len(self.code.g_stabilizer) + logical_index]
        scope = tuple(constraint["variables"]) + (beta_variable,)
        coefficients = tuple(constraint["coefficients"])
        table = {}

        def visit(position, assignment, parity):
            if position == len(constraint["variables"]):
                for beta_state in range(2):
                    if parity == beta_state:
                        table[tuple(assignment + [beta_state])] = 0.0
                return
            coeff = coefficients[position]
            for state in range(4):
                assignment.append(state)
                visit(position + 1, assignment, (parity + self._pauli_parity(state, coeff)) % 2)
                assignment.pop()

        visit(0, [], 0)
        return SparseFactor(scope, table)

    def _initial_factors(self, constraints):
        factors = self._unary_factors()
        factors.extend(self._constraint_factor(constraint) for constraint in constraints)
        return factors

    def _initial_all_sector_factors(self, syndrome_constraints, beta_variables):
        factors = self._unary_factors()
        factors.extend(self._constraint_factor(constraint) for constraint in syndrome_constraints)
        factors.extend(
            self._logical_output_factor(logical_index, beta_variable)
            for logical_index, beta_variable in enumerate(beta_variables)
        )
        return factors

    def _choose_variable(self, active_variables, factor_scopes):
        best_variable = None
        best_key = None
        for variable in sorted(active_variables):
            involved = [scope for scope in factor_scopes if variable in scope]
            merged = set()
            for scope in involved:
                merged.update(scope)
            fill_edges = 0
            neighbors = sorted(item for item in merged if item != variable)
            existing = {frozenset((a, b)) for scope in factor_scopes for a in scope for b in scope if a < b}
            for left_index, left in enumerate(neighbors):
                for right in neighbors[left_index + 1 :]:
                    if frozenset((left, right)) not in existing:
                        fill_edges += 1
            key = (fill_edges, len(merged), len(involved), variable)
            if best_key is None or key < best_key:
                best_key = key
                best_variable = variable
        return best_variable

    def _join_factors(self, factors):
        if not factors:
            return SparseFactor((), {(): 0.0})
        result = factors[0]
        for factor in factors[1:]:
            result = self._multiply_two(result, factor)
        return result

    def _multiply_two(self, left, right):
        scope = normalize_scope(left.scope + right.scope)
        left_positions = [scope.index(variable) for variable in left.scope]
        right_positions = [scope.index(variable) for variable in right.scope]
        shared = tuple(variable for variable in left.scope if variable in right.scope)
        left_shared_positions = [left.scope.index(variable) for variable in shared]
        right_shared_positions = [right.scope.index(variable) for variable in shared]

        right_by_shared = {}
        for right_assignment, right_value in right.table.items():
            key = tuple(right_assignment[position] for position in right_shared_positions)
            right_by_shared.setdefault(key, []).append((right_assignment, right_value))

        table = {}
        for left_assignment, left_value in left.table.items():
            key = tuple(left_assignment[position] for position in left_shared_positions)
            for right_assignment, right_value in right_by_shared.get(key, []):
                assignment = [None] * len(scope)
                for position, value in zip(left_positions, left_assignment):
                    assignment[position] = value
                for position, value in zip(right_positions, right_assignment):
                    assignment[position] = value
                table[tuple(assignment)] = left_value + right_value
        return SparseFactor(scope, table)

    def _sum_out(self, factor, variable):
        if variable not in factor.scope:
            return factor
        variable_position = factor.scope.index(variable)
        scope = tuple(item for item in factor.scope if item != variable)
        table = {}
        for assignment, value in factor.table.items():
            reduced = assignment[:variable_position] + assignment[variable_position + 1 :]
            table[reduced] = logaddexp(table.get(reduced, float("-inf")), value)
        return SparseFactor(scope, table)

    def _run_elimination(self, factors, variables_to_eliminate=None):
        started_at = time.time()
        if variables_to_eliminate is None:
            variables_to_eliminate = range(self.n_variables)
        active_variables = set(int(item) for item in variables_to_eliminate)
        order = []
        max_scope_width = max((len(factor.scope) for factor in factors), default=0)
        max_table_size = max((len(factor.table) for factor in factors), default=0)

        while active_variables:
            factor_scopes = [factor.scope for factor in factors]
            variable = self._choose_variable(active_variables, factor_scopes)
            order.append(variable)
            involved = [factor for factor in factors if variable in factor.scope]
            remaining = [factor for factor in factors if variable not in factor.scope]
            joined = self._join_factors(involved)
            max_scope_width = max(max_scope_width, len(joined.scope))
            max_table_size = max(max_table_size, len(joined.table))
            if len(joined.table) > self.max_intermediate_states:
                raise ValueError(
                    f"Elimination intermediate has {len(joined.table)} states, "
                    f"exceeding max_intermediate_states={self.max_intermediate_states}"
                )
            reduced = self._sum_out(joined, variable)
            if reduced.table:
                remaining.append(reduced)
            factors = remaining
            active_variables.remove(variable)

        final_factor = self._join_factors(factors)
        return final_factor, {
            "order_method": self.order_method,
            "elimination_order": order,
            "max_scope_width": max_scope_width,
            "max_table_size": max_table_size,
            "elapsed_seconds": time.time() - started_at,
            "exact": True,
            "truncated": False,
        }

    def sector_weights(self, gamma):
        return self.all_sector_weights(gamma)

    def per_sector_weights(self, gamma):
        log_z = []
        sector_diagnostics = []
        for sector_index in range(2**self.n_beta):
            beta_bits = index_to_bits(sector_index, self.n_beta)
            constraints = self._constraints_for_targets(gamma, beta_bits)
            final_factor, diagnostics = self._run_elimination(self._initial_factors(constraints))
            value = final_factor.table.get((), float("-inf"))
            log_z.append(value)
            diagnostics["sector"] = "".join(map(str, beta_bits))
            sector_diagnostics.append(diagnostics)
        return sector_weights_from_log_z(
            log_z,
            diagnostics={
                "backend": "variable_elimination",
                "approximate": False,
                "exact": True,
                "truncated": False,
                "sectors": sector_diagnostics,
                "max_scope_width": max(item["max_scope_width"] for item in sector_diagnostics),
                "max_table_size": max(item["max_table_size"] for item in sector_diagnostics),
                "elapsed_seconds": sum(item["elapsed_seconds"] for item in sector_diagnostics),
            },
        )

    def all_sector_weights(self, gamma):
        beta_variables = tuple(range(self.n_variables, self.n_variables + self.n_beta))
        factors = self._initial_all_sector_factors(self._syndrome_constraints_for_gamma(gamma), beta_variables)
        final_factor, diagnostics = self._run_elimination(factors, variables_to_eliminate=range(self.n_variables))
        log_z_by_assignment = {assignment: value for assignment, value in final_factor.table.items()}
        log_z = []
        for sector_index in range(2**self.n_beta):
            beta_bits = index_to_bits(sector_index, self.n_beta)
            if final_factor.scope != beta_variables:
                assignment = tuple(beta_bits[final_factor.scope.index(variable)] for variable in beta_variables)
            else:
                assignment = beta_bits
            log_z.append(log_z_by_assignment.get(assignment, float("-inf")))
        diagnostics.update(
            {
                "backend": "variable_elimination",
                "approximate": False,
                "exact": True,
                "truncated": False,
                "mode": "all_sector",
                "beta_variables": beta_variables,
                "final_scope": final_factor.scope,
            }
        )
        return sector_weights_from_log_z(log_z, diagnostics=diagnostics)
