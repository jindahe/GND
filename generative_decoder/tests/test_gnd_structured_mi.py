import math

import torch

from module import Errormodel, Loading_code, read_code
from gnd.beta_distribution import beta_entropy
from gnd.exact_mi import enumerate_distribution, exact_cut_mi
from gnd.partition_backends.binary_dense_elimination import BinaryDenseVariableEliminationSectorPartition
from gnd.partition_backends.brute_force import BruteForceSectorPartition
from gnd.partition_backends.elimination import VariableEliminationSectorPartition
from gnd.partition_backends.toric_plan import binary_dense_plan_diagnostics
from gnd.partition_backends.toric_row_transfer import ToricRowTransferSectorPartition
from gnd.partitions import build_cut


class Args:
    er = 0.05
    e_model = "dep"
    k = 2
    max_exact_errors = 20_000_000
    chunk_size = 65_536


def toric_code(d):
    return Loading_code(read_code(d=d, k=2, n=2 * d * d, seed=0, c_type="tor"))


def test_exact_beta_entropy_l20_regression():
    code = toric_code(20)
    entropy, distribution = beta_entropy(code, 2, Errormodel(0.05, e_model="dep"))
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-12)
    assert all(probability >= -1e-12 for probability in distribution.values())
    assert entropy <= 4 * math.log(2.0)
    assert math.isclose(entropy, 2.6445747158582247, rel_tol=0.0, abs_tol=1e-12)


def test_brute_force_sector_weights_match_exact_l2():
    code = toric_code(2)
    args = Args()
    layout, joint, _ = enumerate_distribution(code, args)
    exact = exact_cut_mi(joint, build_cut(layout, "middle"))
    h_beta, _ = beta_entropy(code, 2, Errormodel(0.05, e_model="dep"))
    backend = BruteForceSectorPartition(code, 2, 0.05, "dep")

    p_gamma = {}
    for key, probability in joint.items():
        gamma = key[:-4]
        p_gamma[gamma] = p_gamma.get(gamma, 0.0) + probability

    h_cond = 0.0
    for gamma_key, probability in p_gamma.items():
        gamma = torch.tensor(gamma_key, dtype=torch.int64)
        weights = backend.sector_weights(gamma)
        h_cond += probability * weights.entropy
        assert math.isclose(sum(weights.posterior), 1.0, rel_tol=0.0, abs_tol=1e-12)
        assert 0.0 <= weights.entropy <= 4 * math.log(2.0)

    structured_mi = h_beta - h_cond
    assert math.isclose(structured_mi, exact["mi"], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(structured_mi, 0.651333947845, rel_tol=0.0, abs_tol=5e-13)


def test_elimination_sector_weights_match_brute_force_l2():
    code = toric_code(2)
    brute_force = BruteForceSectorPartition(code, 2, 0.05, "dep")
    elimination = VariableEliminationSectorPartition(code, 2, 0.05, "dep")
    gammas = [
        torch.zeros(len(code.g_stabilizer), dtype=torch.int64),
        torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
    ]

    for gamma in gammas:
        expected = brute_force.sector_weights(gamma)
        actual = elimination.sector_weights(gamma)
        assert actual.diagnostics["backend"] == "variable_elimination"
        assert actual.diagnostics["exact"] is True
        assert actual.diagnostics["truncated"] is False
        assert math.isclose(sum(actual.posterior), 1.0, rel_tol=0.0, abs_tol=1e-12)
        assert math.isclose(actual.entropy, expected.entropy, rel_tol=0.0, abs_tol=1e-10)
        for left, right in zip(actual.posterior, expected.posterior):
            assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def test_binary_dense_elimination_sector_weights_match_brute_force_l2():
    code = toric_code(2)
    brute_force = BruteForceSectorPartition(code, 2, 0.05, "dep")
    binary_dense = BinaryDenseVariableEliminationSectorPartition(code, 2, 0.05, "dep")
    gammas = [
        torch.zeros(len(code.g_stabilizer), dtype=torch.int64),
        torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
    ]

    for gamma in gammas:
        expected = brute_force.sector_weights(gamma)
        actual = binary_dense.sector_weights(gamma)
        assert actual.diagnostics["backend"] == "binary_dense_variable_elimination"
        assert actual.diagnostics["exact"] is True
        assert actual.diagnostics["truncated"] is False
        assert math.isclose(actual.entropy, expected.entropy, rel_tol=0.0, abs_tol=1e-10)
        for left, right in zip(actual.posterior, expected.posterior):
            assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def test_toric_row_transfer_sector_weights_match_brute_force_l2():
    code = toric_code(2)
    brute_force = BruteForceSectorPartition(code, 2, 0.05, "dep")
    transfer = ToricRowTransferSectorPartition(code, 2, 0.05, "dep", d=2)
    gammas = [
        torch.zeros(len(code.g_stabilizer), dtype=torch.int64),
        torch.tensor([1, 0, 0, 0, 0, 0], dtype=torch.int64),
        torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.int64),
    ]

    for gamma in gammas:
        expected = brute_force.sector_weights(gamma)
        actual = transfer.sector_weights(gamma)
        assert actual.diagnostics["backend"] == "toric_row_transfer"
        assert actual.diagnostics["exact"] is True
        assert actual.diagnostics["truncated"] is False
        assert math.isclose(actual.entropy, expected.entropy, rel_tol=0.0, abs_tol=1e-10)
        for left, right in zip(actual.posterior, expected.posterior):
            assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def test_binary_dense_elimination_l4_exact_smoke():
    code = toric_code(4)
    backend = BinaryDenseVariableEliminationSectorPartition(
        code,
        2,
        0.05,
        "dep",
        max_intermediate_states=100_000_000,
    )
    gamma = torch.zeros(len(code.g_stabilizer), dtype=torch.int64)

    weights = backend.sector_weights(gamma)

    assert weights.diagnostics["backend"] == "binary_dense_variable_elimination"
    assert weights.diagnostics["exact"] is True
    assert weights.diagnostics["truncated"] is False
    assert weights.diagnostics["max_scope_width"] == 26
    assert weights.diagnostics["max_table_size"] == 67_108_864
    assert math.isclose(sum(weights.posterior), 1.0, rel_tol=0.0, abs_tol=1e-12)
    assert 0.0 <= weights.entropy <= 4 * math.log(2.0)
    assert math.isclose(weights.entropy, 2.4038410309570597e-05, rel_tol=0.0, abs_tol=1e-12)


def test_toric_row_transfer_matches_binary_dense_l4_zero_gamma():
    code = toric_code(4)
    transfer = ToricRowTransferSectorPartition(code, 2, 0.05, "dep", d=4)
    binary_dense = BinaryDenseVariableEliminationSectorPartition(
        code,
        2,
        0.05,
        "dep",
        max_intermediate_states=100_000_000,
    )
    gamma = torch.zeros(len(code.g_stabilizer), dtype=torch.int64)

    expected = binary_dense.sector_weights(gamma)
    actual = transfer.sector_weights(gamma)

    assert actual.diagnostics["backend"] == "toric_row_transfer"
    assert actual.diagnostics["transfer_mode"] == "dense_character"
    assert actual.diagnostics["boundary_bits"] == 8
    assert actual.diagnostics["boundary_states"] == 256
    assert actual.diagnostics["max_state_count"] == 4096
    assert actual.diagnostics["exact"] is True
    assert actual.diagnostics["truncated"] is False
    assert math.isclose(actual.entropy, expected.entropy, rel_tol=0.0, abs_tol=1e-10)
    for left, right in zip(actual.posterior, expected.posterior):
        assert math.isclose(left, right, rel_tol=0.0, abs_tol=1e-10)


def test_toric_row_transfer_refuses_l10_until_sparse_transfer_exists():
    code = toric_code(10)
    transfer = ToricRowTransferSectorPartition(code, 2, 0.05, "dep", d=10)
    plan = transfer.transfer_plan_diagnostics()

    assert plan["boundary_bits"] == 20
    assert plan["boundary_states"] == 1_048_576
    assert plan["dense_character_enabled"] is False
    assert plan["exact"] is False
    assert "sparse/compressed transfer" in plan["refusal_reason"]


def test_binary_dense_planner_reports_l4_width_and_l10_refusal():
    l4 = binary_dense_plan_diagnostics(toric_code(4), 2, 0.05, "dep", target_l=4)
    assert l4.max_scope_width == 26
    assert l4.max_table_size == 67_108_864
    assert l4.scalable_to_target is True
    assert l4.refusal_reason is None

    l10 = binary_dense_plan_diagnostics(toric_code(10), 2, 0.05, "dep", target_l=10)
    assert l10.max_scope_width == 72
    assert l10.scalable_to_target is False
    assert "transfer/trellis" in l10.refusal_reason


def test_binary_dense_planner_refuses_l20_before_minfill():
    l20 = binary_dense_plan_diagnostics(
        toric_code(20),
        2,
        0.05,
        "dep",
        target_l=20,
        max_planner_physical_bits=512,
    )
    assert l20.n_physical_bits == 1600
    assert l20.max_scope_width == -1
    assert l20.elimination_order_length == 0
    assert l20.scalable_to_target is False
    assert "refused before min-fill" in l20.refusal_reason


def test_elimination_structured_mi_matches_exact_l2():
    code = toric_code(2)
    args = Args()
    layout, joint, _ = enumerate_distribution(code, args)
    exact = exact_cut_mi(joint, build_cut(layout, "middle"))
    h_beta, _ = beta_entropy(code, 2, Errormodel(0.05, e_model="dep"))
    backend = VariableEliminationSectorPartition(code, 2, 0.05, "dep")

    p_gamma = {}
    for key, probability in joint.items():
        gamma = key[:-4]
        p_gamma[gamma] = p_gamma.get(gamma, 0.0) + probability

    h_cond = 0.0
    for gamma_key, probability in p_gamma.items():
        weights = backend.sector_weights(torch.tensor(gamma_key, dtype=torch.int64))
        h_cond += probability * weights.entropy

    structured_mi = h_beta - h_cond
    assert math.isclose(structured_mi, exact["mi"], rel_tol=0.0, abs_tol=1e-10)
