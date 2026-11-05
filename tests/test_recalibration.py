import numpy as np

from ptbxl_reliable.metrics import ece
from ptbxl_reliable.recalibration import apply_group_platt, apply_platt, fit_group_platt, fit_platt


def _sharpen(p):
    return 1 / (1 + np.exp(-3 * np.log(p / (1 - p))))


def test_platt_fixes_overconfidence_and_keeps_calibrated():
    rng = np.random.default_rng(0)
    p = rng.random((20_000, 5))
    y = (rng.random(p.shape) < p).astype(float)
    sharp = _sharpen(p)
    assert ece(y, apply_platt(sharp, fit_platt(y, sharp))) < 0.02 < ece(y, sharp)
    np.testing.assert_allclose(fit_platt(y, p), [[1, 0]] * 5, atol=0.1)


def test_platt_identity_for_single_class():
    rng = np.random.default_rng(0)
    p = rng.random((10, 5))
    y = np.tile((np.arange(10) % 2)[:, None], (1, 5)).astype(float)
    y[:, 0] = 0  # class 0 has no positives in this group
    params = fit_platt(y, p)
    np.testing.assert_array_equal(params[0], [1, 0])
    np.testing.assert_allclose(apply_platt(p, params)[:, 0], p[:, 0])


def test_group_platt_fits_each_group():
    rng = np.random.default_rng(1)
    p = rng.random((20_000, 5))
    y = (rng.random(p.shape) < p).astype(float)
    groups = np.where(np.arange(20_000) % 2 == 0, "a", "b")
    q = p.copy()
    q[groups == "a"] = _sharpen(p[groups == "a"])
    fixed = apply_group_platt(q, groups, fit_group_platt(y, q, groups))
    assert ece(y[groups == "a"], fixed[groups == "a"]) < 0.02
