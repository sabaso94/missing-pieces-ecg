"""Performance and calibration per subgroup (sex, age), for the fairness analysis in E3."""

import numpy as np
import pandas as pd

from ptbxl_reliable.data import SUPERCLASSES
from ptbxl_reliable.evaluate import evaluate

AGE_EDGES = (40, 60, 80)
UNKNOWN = "unknown"


def age_group(age: pd.Series, edges: tuple[int, ...] = AGE_EDGES) -> pd.Series:
    """Age bins "<40", "40-59", "60-79", ">=80" (for the default edges); missing age -> "unknown".

    Returned as an ordered categorical, so later steps know which bins are neighbours.
    """
    labels = [f"<{edges[0]}"] + [f"{a}-{b - 1}" for a, b in zip(edges, edges[1:])] + [f">={edges[-1]}"]
    codes = np.digitize(age.to_numpy(dtype=float), edges)  # 40 -> bin 1, 80 -> bin 3
    values = np.where(age.isna(), UNKNOWN, np.array(labels, dtype=object)[np.minimum(codes, len(labels) - 1)])
    return pd.Series(pd.Categorical(values, categories=labels + [UNKNOWN], ordered=True), index=age.index)


def merge_small_bins(groups: pd.Series, min_n: int = 150) -> tuple[pd.Series, list[str]]:
    """Merge every bin with fewer than `min_n` records into its neighbour towards the centre.

    Small bins give confidence intervals too wide to say anything. "unknown" is never merged.
    Returns the new groups (ordered categorical) and a description of each merge made.
    """
    order = [c for c in groups.cat.categories if c != UNKNOWN]
    values = groups.astype(object)
    merges = []
    while len(order) > 1:
        counts = values.value_counts()
        small = [i for i, g in enumerate(order) if counts.get(g, 0) < min_n]
        if not small:
            break
        i = small[0]
        j = i + 1 if i < (len(order) - 1) / 2 else i - 1
        a, b = order[min(i, j)], order[max(i, j)]
        merged = f"{a}+{b}"
        merges.append(f"{order[i]} (n={counts.get(order[i], 0)}) merged with {order[j]} (n={counts.get(order[j], 0)})")
        values = values.replace({a: merged, b: merged})
        order[min(i, j) : max(i, j) + 1] = [merged]
    categories = order + ([UNKNOWN] if UNKNOWN in groups.cat.categories else [])
    return pd.Series(pd.Categorical(values, categories=categories, ordered=True), index=groups.index), merges


def subgroup_table(
    y: np.ndarray, probs: np.ndarray, meta: pd.DataFrame, group_col: str, n_boot: int = 1000
) -> pd.DataFrame:
    """One row per group: size, label prevalence and the `evaluate` metrics of ensemble `probs` (M, N, C).

    `meta` is row-aligned with `y` and needs `group_col` and `patient_id`. As everywhere,
    ECE is a point value only; Brier and macro-AUROC carry patient-level CIs.
    """
    groups = meta[group_col]
    names = list(groups.cat.categories) if isinstance(groups.dtype, pd.CategoricalDtype) else sorted(groups.unique())
    rows = []
    for name in names:
        mask = (groups == name).to_numpy()
        if not mask.any():
            continue
        row = {"group": name, "n": int(mask.sum())}
        row.update({f"prevalence_{c}": float(v) for c, v in zip(SUPERCLASSES, y[mask].mean(axis=0))})
        row.update(evaluate(y[mask], probs[:, mask], meta["patient_id"].to_numpy()[mask], n_boot=n_boot))
        rows.append(row)
    return pd.DataFrame(rows)
