import numpy as np
import pandas as pd

from ptbxl_reliable.data import SUPERCLASSES
from ptbxl_reliable.subgroups import age_group, merge_small_bins, subgroup_table


def test_age_group_boundaries():
    groups = age_group(pd.Series([39.9, 40, 59.9, 60, 80, np.nan]))
    assert list(groups) == ["<40", "40-59", "40-59", "60-79", ">=80", "unknown"]


def test_merge_small_bins():
    groups = age_group(pd.Series([30] * 200 + [50] * 200 + [70] * 200 + [85] * 10 + [np.nan] * 3))
    merged, merges = merge_small_bins(groups, min_n=150)
    assert ">=80" not in set(merged)
    assert (merged == "60-79+>=80").sum() == 210
    assert (merged == "unknown").sum() == 3  # never merged
    assert len(merges) == 1 and ">=80" in merges[0]


def test_subgroup_table_rows():
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 2, size=(n, 5))
    meta = pd.DataFrame({"g": ["a"] * 200 + ["b"] * 200, "patient_id": np.arange(n)})
    y[200:, 4] = 0  # group b has no HYP positives
    probs = rng.random((3, n, 5))
    table = subgroup_table(y, probs, meta, "g", n_boot=20)
    assert list(table["group"]) == ["a", "b"]
    assert list(table["n"]) == [200, 200]
    b = table.set_index("group").loc["b"]
    assert b["prevalence_HYP"] == 0
    assert np.isnan(b["auroc_HYP"])
    assert np.isfinite(b["macro_auroc"])
    assert all(f"prevalence_{c}" in table for c in SUPERCLASSES)
