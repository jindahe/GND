import math
import time

import numpy as np
import torch

from module import Errormodel, mod2

from ..exact_mi import logical_indices
from ..sector_partition import sector_weights_from_log_z


def _logaddexp(left, right):
    if left == float("-inf"):
        return right
    if right == float("-inf"):
        return left
    maximum = max(left, right)
    return maximum + math.log(math.exp(left - maximum) + math.exp(right - maximum))


def _bit(value, index):
    return (int(value) >> int(index)) & 1


def _set_bit(value, index, bit):
    if bit:
        return int(value) | (1 << int(index))
    return int(value) & ~(1 << int(index))


class ToricRowTransferSectorPartition:
    """Exact toric row-transfer backend for fixed-gamma sector weights.

    This backend is the first transfer/trellis implementation behind the
    `sector_weights(gamma)` contract. It scans toric rows, keeps only the
    horizontal edge bit constrained by plaquette checks and the previous-row
    vertical edge bit constrained by star checks. The X/Z orientation is inferred
    from the actual saved code artifact, so both L=2 and L>=4 toric artifacts
    are handled without assuming a fixed plaquette/star Pauli convention.
    """

    def __init__(
        self,
        code,
        k,
        er,
        e_model,
        d=None,
        max_states=5_000_000,
        max_boundary_states=4096,
        max_dense_character_boundary_states=512,
    ):
        self.code = code
        self.k = int(k)
        self.d = int(d or round(math.sqrt(int(code.n) / 2)))
        if int(code.n) != 2 * self.d * self.d:
            raise ValueError(f"Expected toric n=2*d*d, got n={code.n}, d={self.d}")
        if self.k != 2:
            raise ValueError("Toric row transfer currently supports k=2 toric codes")
        self.error_model = Errormodel(er, e_model=e_model)
        self.log_single_p = [math.log(float(item)) for item in self.error_model.single_p]
        self.n_beta = 2 * self.k
        self.max_states = int(max_states)
        self.max_boundary_states = int(max_boundary_states)
        self.max_dense_character_boundary_states = int(max_dense_character_boundary_states)
        self.logical_coefficients = self._logical_coefficients()
        self.logical_masks = self._logical_masks()
        self.stabilizer_map = self._map_stabilizers()
        self.plaquette_pauli, self.star_pauli = self._infer_css_orientation()
        self.plaquette_bit = "x" if self.plaquette_pauli == 2 else "z"
        self.star_bit = "x" if self.star_pauli == 2 else "z"
        self._transition_cache = {}
        self._assignment_log_weight_cache = None
        self._assignment_components_cache = None
        self._row_beta_delta_cache = {}
        self._free_mask_cache = {}

    def _h(self, row, col):
        return int(row) * 2 * self.d + int(col)

    def _v(self, row, col):
        return int(row) * 2 * self.d + self.d + int(col)

    def _plaquette_support(self, row, col):
        return frozenset(
            {
                self._h(row, col),
                self._h((row + 1) % self.d, col),
                self._v(row, col),
                self._v(row, (col + 1) % self.d),
            }
        )

    def _star_support(self, row, col):
        return frozenset(
            {
                self._h(row, col),
                self._h(row, (col - 1) % self.d),
                self._v(row, col),
                self._v((row - 1) % self.d, col),
            }
        )

    def _map_stabilizers(self):
        plaquette_by_support = {
            self._plaquette_support(row, col): (row, col)
            for row in range(self.d)
            for col in range(self.d)
        }
        star_by_support = {
            self._star_support(row, col): (row, col)
            for row in range(self.d)
            for col in range(self.d)
        }
        stabilizers = []
        for index, row in enumerate(self.code.g_stabilizer):
            support_items = [(qubit, int(value.item())) for qubit, value in enumerate(row) if int(value.item())]
            support = frozenset(qubit for qubit, _value in support_items)
            paulis = {value for _qubit, value in support_items}
            if len(paulis) == 1 and support in plaquette_by_support:
                coord = plaquette_by_support[support]
                stabilizers.append(
                    {
                        "index": index,
                        "kind": "plaquette",
                        "row": coord[0],
                        "col": coord[1],
                        "pauli": next(iter(paulis)),
                    }
                )
            elif len(paulis) == 1 and support in star_by_support:
                coord = star_by_support[support]
                stabilizers.append(
                    {
                        "index": index,
                        "kind": "star",
                        "row": coord[0],
                        "col": coord[1],
                        "pauli": next(iter(paulis)),
                    }
                )
            else:
                raise ValueError(
                    f"Could not map toric stabilizer {index}: support={sorted(support)}, paulis={sorted(paulis)}"
                )
        return stabilizers

    def _infer_css_orientation(self):
        plaquette_paulis = {item["pauli"] for item in self.stabilizer_map if item["kind"] == "plaquette"}
        star_paulis = {item["pauli"] for item in self.stabilizer_map if item["kind"] == "star"}
        if len(plaquette_paulis) != 1 or len(star_paulis) != 1:
            raise ValueError("Toric row transfer requires uniform plaquette/star CSS Pauli orientation")
        plaquette_pauli = next(iter(plaquette_paulis))
        star_pauli = next(iter(star_paulis))
        if plaquette_pauli == star_pauli:
            raise ValueError("Toric row transfer requires opposite plaquette/star Pauli orientations")
        return plaquette_pauli, star_pauli

    def _operator_binary_row(self, row):
        helper = mod2(device="cpu", dtype=torch.float64)
        binary = helper.rep(row.unsqueeze(0)).squeeze(0).to(torch.int64)
        x_part = binary[: self.code.n]
        z_part = binary[self.code.n :]
        return torch.cat([z_part, x_part]).to(torch.int64)

    def _logical_coefficients(self):
        return torch.stack(
            [self._operator_binary_row(row) for row in self.code.logical_opt[logical_indices(self.k)]],
            dim=0,
        )

    def _logical_masks(self):
        masks = []
        for row in range(self.d):
            row_masks = []
            for beta_index in range(self.n_beta):
                coeff = self.logical_coefficients[beta_index]
                xh_mask = 0
                zh_mask = 0
                xv_mask = 0
                zv_mask = 0
                for col in range(self.d):
                    hq = self._h(row, col)
                    vq = self._v(row, col)
                    xh_mask |= int(coeff[hq].item()) << col
                    zh_mask |= int(coeff[self.code.n + hq].item()) << col
                    xv_mask |= int(coeff[vq].item()) << col
                    zv_mask |= int(coeff[self.code.n + vq].item()) << col
                row_masks.append((xh_mask, zh_mask, xv_mask, zv_mask))
            masks.append(row_masks)
        return masks

    def _targets_for_gamma(self, gamma):
        gamma_bits = [int(item) for item in torch.as_tensor(gamma, dtype=torch.int64).flatten().tolist()]
        if len(gamma_bits) != int(self.code.g_stabilizer.size(0)):
            raise ValueError(f"Expected gamma length {self.code.g_stabilizer.size(0)}, got {len(gamma_bits)}")
        plaquette_targets = {}
        star_targets = {}
        for item in self.stabilizer_map:
            target = (gamma_bits[item["index"]], item["pauli"])
            key = (item["row"], item["col"])
            if item["kind"] == "plaquette":
                plaquette_targets[key] = target
            else:
                star_targets[key] = target
        return tuple(gamma_bits), plaquette_targets, star_targets

    def _beta_delta(self, row, xh, zh, xv, zv):
        delta = 0
        for beta_index in range(self.n_beta):
            coeff = self.logical_coefficients[beta_index]
            parity = 0
            for col in range(self.d):
                hq = self._h(row, col)
                vq = self._v(row, col)
                parity ^= int(coeff[hq].item()) & _bit(xh, col)
                parity ^= int(coeff[self.code.n + hq].item()) & _bit(zh, col)
                parity ^= int(coeff[vq].item()) & _bit(xv, col)
                parity ^= int(coeff[self.code.n + vq].item()) & _bit(zv, col)
            if parity:
                delta |= 1 << (self.n_beta - beta_index - 1)
        return delta

    def _row_log_weight(self, xh, zh, xv, zv):
        value = 0.0
        for col in range(self.d):
            h_pauli = _bit(xh, col) + 2 * _bit(zh, col)
            v_pauli = _bit(xv, col) + 2 * _bit(zv, col)
            value += self.log_single_p[h_pauli] + self.log_single_p[v_pauli]
        return value

    def _assignment_index(self, xh, zh, xv, zv):
        return int(xh) | (int(zh) << self.d) | (int(xv) << (2 * self.d)) | (int(zv) << (3 * self.d))

    def _assignment_components(self):
        if self._assignment_components_cache is not None:
            return self._assignment_components_cache
        if self.d > 5:
            return None
        indices = np.arange(1 << (4 * self.d), dtype=np.int64)
        mask = (1 << self.d) - 1
        components = (
            indices & mask,
            (indices >> self.d) & mask,
            (indices >> (2 * self.d)) & mask,
            (indices >> (3 * self.d)) & mask,
        )
        self._assignment_components_cache = components
        return components

    def _assignment_log_weights(self):
        if self._assignment_log_weight_cache is not None:
            return self._assignment_log_weight_cache
        if self.d > 5:
            return None
        xh, zh, xv, zv = self._assignment_components()
        table = np.zeros(1 << (4 * self.d), dtype=np.float64)
        for col in range(self.d):
            h_pauli = ((xh >> col) & 1) + 2 * ((zh >> col) & 1)
            v_pauli = ((xv >> col) & 1) + 2 * ((zv >> col) & 1)
            table += np.take(np.asarray(self.log_single_p, dtype=np.float64), h_pauli)
            table += np.take(np.asarray(self.log_single_p, dtype=np.float64), v_pauli)
        self._assignment_log_weight_cache = table
        return table

    def _row_beta_deltas(self, row):
        cached = self._row_beta_delta_cache.get(row)
        if cached is not None:
            return cached
        if self.d > 5:
            return None
        xh, zh, xv, zv = self._assignment_components()
        parity_lookup = np.array([int(value).bit_count() & 1 for value in range(1 << self.d)], dtype=np.uint8)
        table = np.zeros(1 << (4 * self.d), dtype=np.uint16)
        for beta_index, (xh_mask, zh_mask, xv_mask, zv_mask) in enumerate(self.logical_masks[row]):
            parity = (
                parity_lookup[xh & xh_mask]
                ^ parity_lookup[zh & zh_mask]
                ^ parity_lookup[xv & xv_mask]
                ^ parity_lookup[zv & zv_mask]
            )
            table |= parity.astype(np.uint16) << (self.n_beta - beta_index - 1)
        self._row_beta_delta_cache[row] = table
        return table

    def _beta_delta_value(self, row, xh, zh, xv, zv):
        table = self._row_beta_deltas(row)
        if table is None:
            return self._beta_delta(row, xh, zh, xv, zv)
        return int(table[self._assignment_index(xh, zh, xv, zv)])

    def _row_log_weight_value(self, xh, zh, xv, zv):
        table = self._assignment_log_weights()
        if table is None:
            return self._row_log_weight(xh, zh, xv, zv)
        return float(table[self._assignment_index(xh, zh, xv, zv)])

    def _mask_for_bit(self, bit_name, x_mask, z_mask):
        return x_mask if bit_name == "x" else z_mask

    def _expand_next_h_values(self, row, h_mask, v_mask, plaquette_targets):
        free_cols = []
        next_h = 0
        for col in range(self.d):
            target_record = plaquette_targets.get((row, col))
            if target_record is None:
                free_cols.append(col)
                continue
            target, pauli = target_record
            target_bit = "x" if pauli == 2 else "z"
            if target_bit != self.plaquette_bit:
                raise ValueError("Mixed plaquette target bit orientation")
            bit = int(target) ^ _bit(h_mask, col) ^ _bit(v_mask, col) ^ _bit(v_mask, (col + 1) % self.d)
            next_h = _set_bit(next_h, col, bit)
        values = [next_h]
        for col in free_cols:
            values = [item for value in values for item in (value, _set_bit(value, col, 1))]
        return values

    def _star_ok(self, row, h_mask, v_prev_mask, v_mask, star_targets):
        for col in range(self.d):
            target_record = star_targets.get((row, col))
            if target_record is None:
                continue
            target, pauli = target_record
            target_bit = "x" if pauli == 2 else "z"
            if target_bit != self.star_bit:
                raise ValueError("Mixed star target bit orientation")
            parity = _bit(h_mask, col) ^ _bit(h_mask, (col - 1) % self.d) ^ _bit(v_mask, col) ^ _bit(v_prev_mask, col)
            if parity != int(target):
                return False
        return True

    def _expand_star_v_values(self, row, h_mask, v_prev_mask, star_targets):
        free_cols = []
        v_mask = 0
        for col in range(self.d):
            target_record = star_targets.get((row, col))
            if target_record is None:
                free_cols.append(col)
                continue
            target, pauli = target_record
            target_bit = "x" if pauli == 2 else "z"
            if target_bit != self.star_bit:
                raise ValueError("Mixed star target bit orientation")
            bit = int(target) ^ _bit(h_mask, col) ^ _bit(h_mask, (col - 1) % self.d) ^ _bit(v_prev_mask, col)
            v_mask = _set_bit(v_mask, col, bit)
        values = [v_mask]
        for col in free_cols:
            values = [item for value in values for item in (value, _set_bit(value, col, 1))]
        return values

    def _row_transitions(self, gamma_key, row, boundary, plaquette_targets, star_targets):
        cache_key = (gamma_key, row, boundary)
        cached = self._transition_cache.get(cache_key)
        if cached is not None:
            return cached

        mask = (1 << self.d) - 1
        h_boundary = boundary & mask
        v_prev_boundary = boundary >> self.d
        transitions = {}
        for h_other in range(1 << self.d):
            if self.plaquette_bit == "x":
                xh, zh = h_boundary, h_other
            else:
                xh, zh = h_other, h_boundary
            star_h_mask = self._mask_for_bit(self.star_bit, xh, zh)
            plaquette_h_mask = self._mask_for_bit(self.plaquette_bit, xh, zh)
            for v_star in self._expand_star_v_values(row, star_h_mask, v_prev_boundary, star_targets):
                for v_other in range(1 << self.d):
                    if self.star_bit == "x":
                        xv, zv = v_star, v_other
                    else:
                        xv, zv = v_other, v_star
                    star_v_mask = self._mask_for_bit(self.star_bit, xv, zv)
                    plaquette_v_mask = self._mask_for_bit(self.plaquette_bit, xv, zv)
                    next_h_values = self._expand_next_h_values(row, plaquette_h_mask, plaquette_v_mask, plaquette_targets)
                    beta_delta = self._beta_delta_value(row, xh, zh, xv, zv)
                    log_weight = self._row_log_weight_value(xh, zh, xv, zv)
                    for next_h in next_h_values:
                        next_boundary = next_h | (v_star << self.d)
                        key = (next_boundary, beta_delta)
                        transitions[key] = _logaddexp(transitions.get(key, float("-inf")), log_weight)

        result = [(next_boundary, beta_delta, log_weight) for (next_boundary, beta_delta), log_weight in transitions.items()]
        self._transition_cache[cache_key] = result
        return result

    def transfer_plan_diagnostics(self):
        n_boundary_states = 1 << (2 * self.d)
        return {
            "backend": "toric_row_transfer",
            "d": self.d,
            "boundary_bits": 2 * self.d,
            "boundary_states": n_boundary_states,
            "dense_character_enabled": n_boundary_states <= self.max_dense_character_boundary_states,
            "max_dense_character_boundary_states": self.max_dense_character_boundary_states,
            "max_boundary_states": self.max_boundary_states,
            "exact": n_boundary_states <= self.max_dense_character_boundary_states,
            "truncated": False,
            "refusal_reason": None
            if n_boundary_states <= self.max_dense_character_boundary_states
            else (
                f"dense-character transfer has {n_boundary_states} boundary states for d={self.d}, "
                f"exceeding max_dense_character_boundary_states={self.max_dense_character_boundary_states}; "
                "use the next sparse/compressed transfer implementation before L=10/L=20 sampled MI"
            ),
        }

    def _character_signs(self):
        n_sectors = 2**self.n_beta
        signs = np.empty((n_sectors, n_sectors), dtype=np.float64)
        for character in range(n_sectors):
            for beta_delta in range(n_sectors):
                signs[character, beta_delta] = -1.0 if (character & beta_delta).bit_count() % 2 else 1.0
        return signs

    def _free_masks(self, cols):
        key = tuple(cols)
        cached = self._free_mask_cache.get(key)
        if cached is not None:
            return cached
        values = np.array([0], dtype=np.int64)
        for col in key:
            values = np.concatenate([values, values | (1 << int(col))])
        self._free_mask_cache[key] = values
        return values

    def _bit_array(self, values, col):
        return (values >> int(col)) & 1

    def _expand_star_v_arrays(self, row, h_mask, v_prev_mask, star_targets):
        free_cols = []
        v_mask = np.zeros_like(h_mask, dtype=np.int64)
        for col in range(self.d):
            target_record = star_targets.get((row, col))
            if target_record is None:
                free_cols.append(col)
                continue
            target, pauli = target_record
            target_bit = "x" if pauli == 2 else "z"
            if target_bit != self.star_bit:
                raise ValueError("Mixed star target bit orientation")
            bit = (
                int(target)
                ^ self._bit_array(h_mask, col)
                ^ self._bit_array(h_mask, (col - 1) % self.d)
                ^ self._bit_array(v_prev_mask, col)
            )
            v_mask = v_mask | (bit.astype(np.int64) << col)
        return v_mask, self._free_masks(free_cols)

    def _expand_next_h_arrays(self, row, h_mask, v_mask, plaquette_targets):
        free_cols = []
        next_h = np.zeros_like(h_mask, dtype=np.int64)
        for col in range(self.d):
            target_record = plaquette_targets.get((row, col))
            if target_record is None:
                free_cols.append(col)
                continue
            target, pauli = target_record
            target_bit = "x" if pauli == 2 else "z"
            if target_bit != self.plaquette_bit:
                raise ValueError("Mixed plaquette target bit orientation")
            bit = (
                int(target)
                ^ self._bit_array(h_mask, col)
                ^ self._bit_array(v_mask, col)
                ^ self._bit_array(v_mask, (col + 1) % self.d)
            )
            next_h = next_h | (bit.astype(np.int64) << col)
        return next_h, self._free_masks(free_cols)

    def _row_transition_blocks_vectorized(self, row, plaquette_targets, star_targets, n_boundary_states):
        n_sectors = 2**self.n_beta
        mask = (1 << self.d) - 1
        boundaries = np.arange(n_boundary_states, dtype=np.int64)[:, None, None]
        h_others = np.arange(1 << self.d, dtype=np.int64)[None, :, None]
        v_others = np.arange(1 << self.d, dtype=np.int64)[None, None, :]

        boundary = np.broadcast_to(boundaries, (n_boundary_states, 1 << self.d, 1 << self.d)).reshape(-1)
        h_other = np.broadcast_to(h_others, (n_boundary_states, 1 << self.d, 1 << self.d)).reshape(-1)
        v_other = np.broadcast_to(v_others, (n_boundary_states, 1 << self.d, 1 << self.d)).reshape(-1)
        h_boundary = boundary & mask
        v_prev_boundary = boundary >> self.d

        if self.plaquette_bit == "x":
            xh, zh = h_boundary, h_other
        else:
            xh, zh = h_other, h_boundary
        star_h_mask = self._mask_for_bit(self.star_bit, xh, zh)
        plaquette_h_mask = self._mask_for_bit(self.plaquette_bit, xh, zh)
        star_v_base, star_free_masks = self._expand_star_v_arrays(
            row,
            star_h_mask,
            v_prev_boundary,
            star_targets,
        )

        blocks = np.zeros((n_sectors, n_boundary_states, n_boundary_states), dtype=np.float64)
        beta_delta_table = self._row_beta_deltas(row)
        log_weight_table = self._assignment_log_weights()
        raw_transition_count = 0
        for star_free_mask in star_free_masks:
            v_star = star_v_base | int(star_free_mask)
            if self.star_bit == "x":
                xv, zv = v_star, v_other
            else:
                xv, zv = v_other, v_star
            plaquette_v_mask = self._mask_for_bit(self.plaquette_bit, xv, zv)
            next_h_base, next_free_masks = self._expand_next_h_arrays(
                row,
                plaquette_h_mask,
                plaquette_v_mask,
                plaquette_targets,
            )
            assignment_index = (
                xh
                | (zh << self.d)
                | (xv << (2 * self.d))
                | (zv << (3 * self.d))
            ).astype(np.int64)
            beta_delta = beta_delta_table[assignment_index].astype(np.int64)
            weight = np.exp(log_weight_table[assignment_index])
            for next_free_mask in next_free_masks:
                next_h = next_h_base | int(next_free_mask)
                next_boundary = next_h | (v_star << self.d)
                flat_index = (beta_delta * n_boundary_states + boundary) * n_boundary_states + next_boundary
                blocks += np.bincount(
                    flat_index,
                    weights=weight,
                    minlength=n_sectors * n_boundary_states * n_boundary_states,
                ).reshape(n_sectors, n_boundary_states, n_boundary_states)
                raw_transition_count += int(boundary.size)
        return blocks, raw_transition_count

    def _row_transition_blocks(self, row, plaquette_targets, star_targets, n_boundary_states):
        if self.d <= 5:
            return self._row_transition_blocks_vectorized(row, plaquette_targets, star_targets, n_boundary_states)

        n_sectors = 2**self.n_beta
        mask = (1 << self.d) - 1
        blocks = np.zeros((n_sectors, n_boundary_states, n_boundary_states), dtype=np.float64)
        raw_transition_count = 0
        for boundary in range(n_boundary_states):
            h_boundary = boundary & mask
            v_prev_boundary = boundary >> self.d
            for h_other in range(1 << self.d):
                if self.plaquette_bit == "x":
                    xh, zh = h_boundary, h_other
                else:
                    xh, zh = h_other, h_boundary
                star_h_mask = self._mask_for_bit(self.star_bit, xh, zh)
                plaquette_h_mask = self._mask_for_bit(self.plaquette_bit, xh, zh)
                for v_star in self._expand_star_v_values(row, star_h_mask, v_prev_boundary, star_targets):
                    for v_other in range(1 << self.d):
                        if self.star_bit == "x":
                            xv, zv = v_star, v_other
                        else:
                            xv, zv = v_other, v_star
                        plaquette_v_mask = self._mask_for_bit(self.plaquette_bit, xv, zv)
                        next_h_values = self._expand_next_h_values(
                            row,
                            plaquette_h_mask,
                            plaquette_v_mask,
                            plaquette_targets,
                        )
                        beta_delta = self._beta_delta_value(row, xh, zh, xv, zv)
                        weight = math.exp(self._row_log_weight_value(xh, zh, xv, zv))
                        for next_h in next_h_values:
                            next_boundary = next_h | (v_star << self.d)
                            blocks[beta_delta, boundary, next_boundary] += weight
                            raw_transition_count += 1
        return blocks, raw_transition_count

    def _character_transfer_log_z(self, plaquette_targets, star_targets, n_boundary_states):
        n_sectors = 2**self.n_beta
        signs = self._character_signs()
        build_started_at = time.time()
        character_rows = []
        raw_transition_counts = []
        row_block_nonzeros = []
        for row in range(self.d):
            blocks, raw_transition_count = self._row_transition_blocks(
                row,
                plaquette_targets,
                star_targets,
                n_boundary_states,
            )
            character_rows.append(np.tensordot(signs, blocks, axes=(1, 0)))
            raw_transition_counts.append(raw_transition_count)
            row_block_nonzeros.append(int(np.count_nonzero(blocks)))
        build_seconds = time.time() - build_started_at

        product_started_at = time.time()
        character_traces = np.empty(n_sectors, dtype=np.float64)
        for character in range(n_sectors):
            matrix = character_rows[0][character]
            for row in range(1, self.d):
                matrix = matrix @ character_rows[row][character]
            character_traces[character] = np.trace(matrix)
        sector_z = signs.T @ character_traces / float(n_sectors)
        product_seconds = time.time() - product_started_at

        log_z = []
        min_sector_z = float(np.min(sector_z))
        max_sector_z = float(np.max(sector_z))
        for value in sector_z:
            if value <= 0.0:
                if abs(float(value)) <= 1e-14 * max(1.0, max_sector_z):
                    value = 0.0
                else:
                    raise ValueError(f"Character transfer produced a negative sector weight: {value}")
            log_z.append(float("-inf") if value == 0.0 else math.log(float(value)))

        return log_z, {
            "transfer_mode": "dense_character",
            "transition_build_seconds": build_seconds,
            "character_product_seconds": product_seconds,
            "max_raw_transition_count_per_row": max(raw_transition_counts) if raw_transition_counts else 0,
            "max_row_block_nonzeros": max(row_block_nonzeros) if row_block_nonzeros else 0,
            "min_sector_z": min_sector_z,
            "max_sector_z": max_sector_z,
        }

    def sector_weights(self, gamma):
        started_at = time.time()
        gamma_key, plaquette_targets, star_targets = self._targets_for_gamma(gamma)
        n_boundary_states = 1 << (2 * self.d)
        if n_boundary_states > self.max_boundary_states:
            raise ValueError(
                f"Toric row transfer has {n_boundary_states} boundary states for d={self.d}, "
                f"exceeding max_boundary_states={self.max_boundary_states}"
            )
        if n_boundary_states > self.max_dense_character_boundary_states:
            raise ValueError(self.transfer_plan_diagnostics()["refusal_reason"])

        common_diagnostics = {
            "backend": "toric_row_transfer",
            "approximate": False,
            "exact": True,
            "truncated": False,
            "d": self.d,
            "boundary_bits": 2 * self.d,
            "boundary_states": n_boundary_states,
            "stabilizer_map_size": len(self.stabilizer_map),
            "plaquette_pauli": self.plaquette_pauli,
            "star_pauli": self.star_pauli,
            "plaquette_bit": self.plaquette_bit,
            "star_bit": self.star_bit,
        }
        log_z, transfer_diagnostics = self._character_transfer_log_z(
            plaquette_targets,
            star_targets,
            n_boundary_states,
        )
        common_diagnostics.update(transfer_diagnostics)
        common_diagnostics["max_state_count"] = n_boundary_states * (2**self.n_beta)
        common_diagnostics["max_transition_count_per_boundary"] = None
        common_diagnostics["elapsed_seconds"] = time.time() - started_at
        return sector_weights_from_log_z(log_z, diagnostics=common_diagnostics)
