def normalize_layout(layout):
    if "gamma" not in layout or "beta" not in layout:
        raise ValueError("Dataset layout must define gamma and beta slices")
    return layout


def slice_indices(item):
    return list(range(int(item["start"]), int(item["stop"])))


def split_indices(indices, cut=None):
    if len(indices) < 2:
        raise ValueError("Cannot split fewer than two variables")
    cut = len(indices) // 2 if cut is None else int(cut)
    if cut <= 0 or cut >= len(indices):
        raise ValueError(f"Invalid cut {cut} for {len(indices)} variables")
    return indices[:cut], indices[cut:]


def build_cut(layout, cut_name, cut=None):
    layout = normalize_layout(layout)
    gamma = slice_indices(layout["gamma"])
    beta = slice_indices(layout["beta"])

    if cut_name == "middle":
        a_indices = beta
        b_indices = gamma
        description = "I(beta : gamma)"
    elif cut_name == "quarter":
        beta_1, beta_2 = split_indices(beta, cut=cut)
        a_indices = beta_1
        b_indices = beta_2 + gamma
        description = "I(beta_1 : beta_2,gamma)"
    elif cut_name == "three_quarter":
        gamma_1, gamma_2 = split_indices(gamma, cut=cut)
        a_indices = beta + gamma_1
        b_indices = gamma_2
        description = "I(beta,gamma_1 : gamma_2)"
    else:
        raise ValueError(f"Unsupported cut: {cut_name}")

    return {
        "name": cut_name,
        "description": description,
        "a_indices": a_indices,
        "b_indices": b_indices,
        "len_A": len(a_indices),
        "len_B": len(b_indices),
    }


def all_outline_cuts(layout):
    return [build_cut(layout, name) for name in ("middle", "quarter", "three_quarter")]
