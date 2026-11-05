import numpy as np
import pandas as pd
import pytest

from ptbxl_reliable.data import SUPERCLASSES
from ptbxl_reliable.metrics import macro_auroc
from ptbxl_reliable.subgroups import (
    age_group,
    gap_change_ci,
    gap_ci,
    group_weights,
    merge_small_bins,
    subgroup_table,
)


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


def test_gap_ci_detects_real_gap():
    rng = np.random.default_rng(0)
    n = 600
    y = rng.integers(0, 2, size=(n, 5))
    in_a = np.arange(n) < 300
    p = rng.random((n, 5))
    p[in_a] = 0.8 * y[in_a] + 0.2 * p[in_a]  # group a: near-perfect, group b: random
    point, lo, hi = gap_ci(macro_auroc, y, p, in_a, ~in_a, np.arange(n), n_boot=200)
    assert point == pytest.approx(macro_auroc(y[in_a], p[in_a]) - macro_auroc(y[~in_a], p[~in_a]))
    assert 0 < lo < point < hi


def test_gap_ci_contains_zero_without_gap():
    rng = np.random.default_rng(1)
    n = 600
    y = rng.integers(0, 2, size=(n, 5))
    p = np.clip(0.3 * y + 0.7 * rng.random((n, 5)), 0, 1)  # same model quality in both groups
    in_a = rng.random(n) < 0.5
    _, lo, hi = gap_ci(macro_auroc, y, p, in_a, ~in_a, np.arange(n) // 2, n_boot=200)
    assert lo < 0 < hi


def test_gap_ci_ignores_records_outside_both_groups():
    rng = np.random.default_rng(2)
    n = 900
    y = rng.integers(0, 2, size=(n, 5))
    p = rng.random((n, 5))
    in_a, in_b = np.arange(n) < 300, (np.arange(n) >= 300) & (np.arange(n) < 600)
    p[600:] = y[600:]  # a third group the comparison must not see
    point, _, _ = gap_ci(macro_auroc, y, p, in_a, in_b, np.arange(n), n_boot=10)
    assert point == pytest.approx(macro_auroc(y[in_a], p[in_a]) - macro_auroc(y[in_b], p[in_b]))


def test_gap_change_ci_zero_for_same_predictions_and_detects_fix():
    rng = np.random.default_rng(0)
    n = 800
    y = rng.integers(0, 2, (n, 5)).astype(float)
    in_a, in_b = np.arange(n) < n // 2, np.arange(n) >= n // 2
    pid = np.arange(n) // 2
    p_old = np.where(in_a[:, None], 0.6 * y + 0.4 * rng.random((n, 5)), 0.2 * y + 0.8 * rng.random((n, 5)))
    assert gap_change_ci(macro_auroc, y, p_old, p_old, in_a, in_b, pid, n_boot=50) == (0.0, 0.0, 0.0)
    p_new = p_old.copy()
    p_new[in_b] = 0.6 * y[in_b] + 0.4 * rng.random((in_b.sum(), 5))  # group b now as good as a
    point, lo, hi = gap_change_ci(macro_auroc, y, p_new, p_old, in_a, in_b, pid, n_boot=200)
    assert point < 0 and hi < 0  # the gap a - b shrank


def test_group_weights_balance_groups():
    w = group_weights(pd.Series(["a"] * 1 + ["b"] * 3))
    np.testing.assert_allclose(w, [2, 2 / 3, 2 / 3, 2 / 3])
    assert w.mean() == pytest.approx(1)
