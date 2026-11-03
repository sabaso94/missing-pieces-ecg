import numpy as np
import pytest

from ptbxl_reliable import labelfree
from ptbxl_reliable.metrics import brier


def _calibrated(n, seed):
    rng = np.random.default_rng(seed)
    p = rng.random((n, 5))
    return (rng.random((n, 5)) < p).astype(float), p


def test_ac_exact_for_calibrated_model():
    y, p = _calibrated(200_000, 0)
    assert labelfree.ac_accuracy(p) == pytest.approx(labelfree.accuracy(y, p), abs=0.005)
    assert labelfree.ac_brier(p) == pytest.approx(brier(y, p), abs=0.005)


def test_atc_and_doc_reproduce_validation_accuracy():
    y, p = _calibrated(5_000, 1)
    t = labelfree.fit_atc(y, p)
    assert labelfree.atc_accuracy(p, t) == pytest.approx(labelfree.accuracy(y, p), abs=0.002)
    assert labelfree.doc_accuracy(p, y, p) == pytest.approx(labelfree.accuracy(y, p))


def test_ac_overestimates_for_overconfident_model():
    y, p = _calibrated(50_000, 2)
    sharp = 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))  # same ranking, overconfident
    assert labelfree.ac_accuracy(sharp) > labelfree.accuracy(y, sharp) + 0.03


def test_estimate_all_keys_and_ci():
    y, p = _calibrated(400, 3)
    row = labelfree.estimate_all(y, p, np.arange(400) // 2, y, p, n_boot=50)
    assert row["n"] == 400 and row["true_acc_lo"] <= row["true_acc"] <= row["true_acc_hi"]
    assert {"ac_acc", "atc_acc", "doc_acc", "ac_brier", "true_brier"} <= set(row)
