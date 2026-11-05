import numpy as np
import pytest

from ptbxl_reliable.metrics import (
    _patient_resample_indices,
    bootstrap_ci,
    brier,
    ece,
    macro_auroc,
    model_diff_ci,
    per_class_auroc,
)


def _toy(n=200, c=5, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=(n, c))
    return y, rng.random((n, c))


def test_auroc_perfect_and_random():
    y, _ = _toy()
    assert macro_auroc(y, y.astype(float)) == pytest.approx(1.0)
    assert macro_auroc(y, 1.0 - y) == pytest.approx(0.0)


def test_auroc_class_without_positives():
    y, p = _toy()
    y[:, 2] = 0
    aucs = per_class_auroc(y, p)
    assert np.isnan(aucs[2])
    assert macro_auroc(y, p) == pytest.approx(np.mean(aucs[[0, 1, 3, 4]]))


def test_ece_known_value():
    y = np.array([[1], [0]] * 50)
    p = np.full((100, 1), 0.9)
    assert ece(y, p) == pytest.approx(0.4, abs=1e-9)

    rng = np.random.default_rng(1)
    p = rng.random((200_000, 2))
    y = (rng.random((200_000, 2)) < p).astype(int)  # perfectly calibrated by construction
    assert ece(y, p) < 0.01


def test_brier_known_value():
    assert brier(np.array([[1, 0]]), np.array([[0.8, 0.4]])) == pytest.approx(0.1)


def test_bootstrap_patient_level():
    patient_ids = np.array([1] * 10 + [2, 3])
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = _patient_resample_indices(patient_ids, rng)
        n_from_patient_1 = np.sum(patient_ids[idx] == 1)
        assert n_from_patient_1 % 10 == 0  # patient 1's records always travel together

    y, p = _toy(n=12)
    first = bootstrap_ci(macro_auroc, y, p, patient_ids, n_boot=200, seed=0)
    second = bootstrap_ci(macro_auroc, y, p, patient_ids, n_boot=200, seed=0)
    assert first == second
    point, lo, hi = first
    assert point == pytest.approx(macro_auroc(y, p))
    assert lo <= point <= hi


def test_model_diff_ci():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, (400, 5)).astype(float)
    pid = np.arange(400) // 2
    p = rng.random((400, 5))
    assert model_diff_ci(macro_auroc, y, p, p, pid, n_boot=50) == (0.0, 0.0, 0.0)
    good = 0.7 * y + 0.3 * p
    point, lo, hi = model_diff_ci(macro_auroc, y, good, p, pid, n_boot=200)
    assert lo > 0 and lo <= point <= hi
