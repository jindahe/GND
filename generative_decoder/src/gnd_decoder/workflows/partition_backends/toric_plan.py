from dataclasses import dataclass

from .binary_dense_elimination import BinaryDenseVariableEliminationSectorPartition


@dataclass
class ContractionPlanDiagnostics:
    backend: str
    exact: bool
    truncated: bool
    n_qubits: int
    n_physical_bits: int
    n_beta: int
    factor_count: int
    max_scope_width: int
    max_table_size: int
    elimination_order_prefix: list
    elimination_order_length: int
    scalable_to_target: bool
    target_l: int | None
    refusal_reason: str | None

    def to_dict(self):
        return {
            "backend": self.backend,
            "exact": self.exact,
            "truncated": self.truncated,
            "n_qubits": self.n_qubits,
            "n_physical_bits": self.n_physical_bits,
            "n_beta": self.n_beta,
            "factor_count": self.factor_count,
            "max_scope_width": self.max_scope_width,
            "max_table_size": self.max_table_size,
            "elimination_order_prefix": self.elimination_order_prefix,
            "elimination_order_length": self.elimination_order_length,
            "scalable_to_target": self.scalable_to_target,
            "target_l": self.target_l,
            "refusal_reason": self.refusal_reason,
        }


def binary_dense_plan_diagnostics(
    code,
    k,
    er,
    e_model,
    target_l=None,
    max_safe_width=30,
    max_planner_physical_bits=512,
):
    """Return dry-run contraction diagnostics for binary dense elimination.

    This planner uses the actual code artifact and the same binary factor graph
    as `BinaryDenseVariableEliminationSectorPartition`, but it only simulates
    min-fill elimination on scopes. It is intentionally cheap and is used to
    prevent accidental dense L=10/L=20 runs.
    """

    backend = BinaryDenseVariableEliminationSectorPartition(code, k, er, e_model)
    if backend.n_physical_bits > int(max_planner_physical_bits):
        return ContractionPlanDiagnostics(
            backend="binary_dense_plan",
            exact=True,
            truncated=False,
            n_qubits=int(code.n),
            n_physical_bits=backend.n_physical_bits,
            n_beta=backend.n_beta,
            factor_count=int(code.n) + int(code.g_stabilizer.size(0)) + backend.n_beta,
            max_scope_width=-1,
            max_table_size=-1,
            elimination_order_prefix=[],
            elimination_order_length=0,
            scalable_to_target=False,
            target_l=target_l,
            refusal_reason=(
                f"binary dense dry-run refused before min-fill: n_physical_bits="
                f"{backend.n_physical_bits} exceeds max_planner_physical_bits="
                f"{max_planner_physical_bits}; use transfer/trellis backend"
            ),
        )
    factors, _beta_variables = backend._initial_all_sector_factors([0 for _ in range(code.g_stabilizer.size(0))])
    factor_scopes = [factor.scope for factor in factors]
    active_variables = set(range(backend.n_physical_bits))
    order = []
    max_scope_width = max((len(scope) for scope in factor_scopes), default=0)
    max_table_size = 1 << max_scope_width

    while active_variables:
        variable = backend._choose_variable(active_variables, factor_scopes)
        order.append(variable)
        involved = [scope for scope in factor_scopes if variable in scope]
        remaining = [scope for scope in factor_scopes if variable not in scope]
        merged = tuple(sorted({item for scope in involved for item in scope}))
        reduced = tuple(item for item in merged if item != variable)
        max_scope_width = max(max_scope_width, len(merged))
        max_table_size = max(max_table_size, 1 << len(merged))
        if reduced:
            remaining.append(reduced)
        factor_scopes = remaining
        active_variables.remove(variable)

    scalable_to_target = max_scope_width <= int(max_safe_width)
    refusal_reason = None
    if not scalable_to_target:
        refusal_reason = (
            f"binary dense plan has max_scope_width={max_scope_width}, "
            f"exceeding max_safe_width={max_safe_width}; use transfer/trellis backend"
        )

    return ContractionPlanDiagnostics(
        backend="binary_dense_plan",
        exact=True,
        truncated=False,
        n_qubits=int(code.n),
        n_physical_bits=backend.n_physical_bits,
        n_beta=backend.n_beta,
        factor_count=len(factors),
        max_scope_width=max_scope_width,
        max_table_size=max_table_size,
        elimination_order_prefix=order[:20],
        elimination_order_length=len(order),
        scalable_to_target=scalable_to_target,
        target_l=target_l,
        refusal_reason=refusal_reason,
    )
