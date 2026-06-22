import math

import torch

from module import Errormodel, Loading_code, read_code
from gnd.beta_distribution import beta_entropy
from gnd.exact_mi import enumerate_distribution, exact_cut_mi
from gnd.partition_backends.brute_force import BruteForceSectorPartition
from gnd.partition_backends.elimination import VariableEliminationSectorPartition
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
