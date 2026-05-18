from pathlib import Path

import networkx as nx
import numpy as np
import torch

from .mod2 import mod2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CODE_DIR = PROJECT_ROOT / "code"
mod2 = mod2()


def generate_graph(n, m, degree=3, seed=0, G_type="rrg"):
    torch.manual_seed(seed)
    if G_type == "rrg":
        return nx.random_regular_graph(degree, n * m, seed=seed)
    if G_type == "erg":
        return nx.erdos_renyi_graph(n, degree / n, seed=seed)
    if G_type == "2D":
        grid = nx.grid_graph([n, m])
        adjacency = np.array(nx.adjacency_matrix(grid).todense())
        return nx.from_numpy_array(adjacency)
    raise ValueError(f"Unsupported graph type: {G_type}")


def read_code(d, k, n, seed=0, c_type="sur"):
    path = CODE_DIR / f"{c_type}_n{n}_d{d}_k{k}_seed{seed}"
    return torch.load(path)


def toric_syndrome_coords(d):
    coords = []
    # Keep the same independent-generator ordering as Toric.get_generators_of_stabilizers():
    # first all plaquette checks in row-major order, excluding the first cell,
    # then all star checks in row-major order, excluding the last cell.
    for row in range(d):
        for col in range(d):
            if row == 0 and col == 0:
                continue
            coords.append({
                "index": len(coords),
                "kind": "plaquette",
                "row": row,
                "col": col,
                "x": col,
                "y": row,
            })

    for row in range(d):
        for col in range(d):
            if row == d - 1 and col == d - 1:
                continue
            coords.append({
                "index": len(coords),
                "kind": "star",
                "row": row,
                "col": col,
                "x": col,
                "y": row,
            })

    return coords


def toric_bipartition(d, axis="x", cut=None):
    if axis not in {"x", "y"}:
        raise ValueError(f"Unsupported axis: {axis}")

    coords = toric_syndrome_coords(d)
    cut = d // 2 if cut is None else cut

    idx_a = [item["index"] for item in coords if item[axis] < cut]
    idx_b = [item["index"] for item in coords if item[axis] >= cut]
    if not idx_a or not idx_b:
        raise ValueError(f"Cut {cut} along axis {axis} produces an empty partition for d={d}")

    return {
        "coords": coords,
        "axis": axis,
        "cut": cut,
        "idx_A": idx_a,
        "idx_B": idx_b,
        "order_AB": idx_a + idx_b,
        "order_BA": idx_b + idx_a,
    }


def sample_syndromes(code, error_model, n_samples, seed=0, device="cpu", dtype=torch.float32):
    errors = error_model.generate_error(code.n, m=n_samples, seed=seed)
    if errors.dim() == 1:
        errors = errors.unsqueeze(0)

    mod2.to(device=device, dtype=dtype)
    syndrome = mod2.commute(errors, code.g_stabilizer)
    if syndrome.dim() == 1:
        syndrome = syndrome.unsqueeze(0)
    return syndrome.to(device=device, dtype=dtype)


def reorder_bits(samples, order):
    if order is None:
        return samples

    index = torch.tensor(order, dtype=torch.long, device=samples.device)
    return samples.index_select(1, index)


def split_samples(samples, n_train, n_val, n_test, shuffle=True, seed=0):
    total = n_train + n_val + n_test
    if samples.size(0) != total:
        raise ValueError(f"Expected {total} samples, got {samples.size(0)}")

    if shuffle:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        permutation = torch.randperm(total, generator=generator)
        samples = samples[permutation]

    train = samples[:n_train]
    val = samples[n_train:n_train + n_val]
    test = samples[n_train + n_val:]
    return train, val, test


def PCM_to_Stabilizer(pcm):
    n = int(pcm.size(1) / 2)
    stabilizer = torch.zeros_like(pcm)
    stabilizer[:, :n], stabilizer[:, n:] = pcm[:, n:], pcm[:, :n]
    return mod2.xyz(stabilizer)


def PCM(g_stabilizer):
    n = g_stabilizer.size(1)
    binary = mod2.rep(g_stabilizer)
    pcm = torch.zeros_like(binary)
    pcm[:, :n], pcm[:, n:] = binary[:, n:], binary[:, :n]
    return pcm


def Hx_Hz(g_stabilizer):
    hx, hz = [], []
    for row in g_stabilizer:
        if (row % 2).sum() != 0:
            hx.append(row)
        else:
            hz.append(row)
    hx = mod2.rep(torch.vstack(hx))
    hz = mod2.rep(torch.vstack(hz))
    return hx[:, : g_stabilizer.size(1)], hz[:, g_stabilizer.size(1) :]


def error_solver(pcm, b):
    errors = mod2.solve(pcm, b)
    return mod2.xyz(errors)


class Errormodel:
    def __init__(self, e_rate=0.2, e_model="dep"):
        self.e_rate = e_rate
        self.e_model = e_model
        if e_model == "dep":
            self.single_p = np.array([1 - e_rate, e_rate / 3, e_rate / 3, e_rate / 3])
        elif e_model == "x":
            self.single_p = np.array([1 - e_rate, e_rate - 2e-9, 1e-9, 1e-9])
        elif e_model == "z":
            self.single_p = np.array([1 - e_rate, 1e-9, e_rate - 2e-9, 1e-9])
        elif e_model == "dep2":
            self.single_p = [1 - e_rate]
            self.single_p.extend([e_rate / 15] * 15)
        else:
            raise ValueError(f"Unsupported error model: {e_model}")

    def generate_error(self, n, m=1, seed=0):
        if seed is not False:
            np.random.seed(seed)

        if self.e_model != "dep2":
            shape = [n] if m == 1 else [m, n]
            return torch.tensor(np.random.choice([0, 1, 2, 3], shape, p=self.single_p))

        e = np.array(
            [
                [[0, 0], [0, 0]],
                [[0, 0], [0, 1]],
                [[0, 0], [1, 0]],
                [[0, 0], [1, 1]],
                [[0, 1], [0, 0]],
                [[0, 1], [0, 1]],
                [[0, 1], [1, 0]],
                [[0, 1], [1, 1]],
                [[1, 0], [0, 0]],
                [[1, 0], [0, 1]],
                [[1, 0], [1, 0]],
                [[1, 0], [1, 1]],
                [[1, 1], [0, 0]],
                [[1, 1], [0, 1]],
                [[1, 1], [1, 0]],
                [[1, 1], [1, 1]],
            ]
        )
        sampled = np.random.choice(np.arange(16), [m, n], p=self.single_p)
        error = np.zeros([m, n, 2])
        e0 = e[sampled]
        for i in range(n):
            error[:, i, :] = (e0[:, i - 1, 1] + e0[:, i, 0]) % 2
        error = torch.tensor(error).transpose(1, 2).reshape([m, -1])
        return mod2.xyz(error)

    def pure(self, pure_es, syndrome, device="cpu", dtype=torch.float64):
        syndrome = syndrome.to(device=device, dtype=dtype)
        mod2.to(device=device, dtype=dtype)
        return mod2.confs_to_opt(syndrome, pure_es)

    def configs(self, sta, log, pe, opts):
        syndrome = mod2.commute(opts, sta)
        sta_conf = mod2.commute(opts, pe)
        log_conf = mod2.commute(opts, log)

        k = int(log.size(0) / 2)
        logical_indices = []
        for i in range(k):
            logical_indices.extend([2 * i + 1, 2 * i])
        log_conf = log_conf[:, logical_indices]
        return torch.hstack([syndrome, log_conf, sta_conf])
