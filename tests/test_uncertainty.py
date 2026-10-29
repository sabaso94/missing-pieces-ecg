import numpy as np
import pytest

from ptbxl_reliable.metrics import macro_auroc
from ptbxl_reliable.uncertainty import (
    ensemble_mean,
    mutual_information,
    predictive_entropy,
    risk_coverage,
)


def test_entropy_extremes():
    assert predictive_entropy(np.full((4, 5), 0.5)) == pytest.approx(np.full(4, np.log(2)))
    confident = np.array([[1e-12, 1 - 1e-12, 1e-12, 1e-12, 1 - 1e-12]])
    assert predictive_entropy(confident)[0] == pytest.approx(0.0, abs=1e-5)


def test_mi_zero_for_identical_members():
    rng = np.random.default_rng(0)
    member = rng.random((10, 5))
    same = np.stack([member] * 5)
    assert mutual_information(same) == pytest.approx(np.zeros(10), abs=1e-9)

    disagree = np.stack([np.full((10, 5), 0.05), np.full((10, 5), 0.95)])
    assert (mutual_information(disagree) > 0.1).all()
    assert ensemble_mean(disagree) == pytest.approx(np.full((10, 5), 0.5))


def test_risk_coverage_full_equals_overall():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (300, 5))
    p = rng.random((300, 5))
    table = risk_coverage(y, p, uncertainty=rng.random(300))
    assert list(table.columns) == ["coverage", "n", "macro_auroc", "brier"]
    assert table["macro_auroc"].iloc[-1] == pytest.approx(macro_auroc(y, p))
    assert table["n"].is_monotonic_increasing and table["n"].iloc[-1] == 300


def test_risk_coverage_keeps_most_certain():
    y = np.tile(np.array([[1], [0]]), (100, 5))
    p = y.astype(float).copy()
    p[100:] = 1.0 - p[100:]  # second half predicted exactly wrong
    uncertainty = np.r_[np.zeros(100), np.ones(100)]  # ... and flagged as most uncertain
    table = risk_coverage(y, p, uncertainty, coverages=np.array([0.5, 1.0]))
    assert table["macro_auroc"].iloc[0] == pytest.approx(1.0)
    assert table["macro_auroc"].iloc[1] < 1.0
