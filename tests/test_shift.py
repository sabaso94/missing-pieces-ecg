import numpy as np

from ptbxl_reliable import shift


def test_mixture_batch_group_share():
    rng = np.random.default_rng(0)
    in_group = np.arange(100) < 10
    idx = shift.mixture_batch(100, in_group, lam=0.5, n=200, rng=rng)
    assert len(idx) == 200 and in_group[idx[:100]].all()


def test_ks_test_detects_shift_and_not_null():
    rng = np.random.default_rng(0)
    ref = rng.random((2000, 5))
    assert shift.ks_test(ref, rng.random((200, 5))) > 0.05
    assert shift.ks_test(ref, rng.random((200, 5)) ** 3) < 0.001


def test_mmd_test_detects_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(500, 8))
    bw = shift.median_bandwidth(ref)
    assert shift.mmd_test(ref, rng.normal(size=(100, 8)) + 0.8, bw, n_perm=200) < 0.01


def test_detectors_false_alarm_rate_near_alpha():
    rng = np.random.default_rng(1)
    ref_p, pool_p = rng.random((1000, 5)), rng.random((1000, 5))
    ref_z, pool_z = rng.normal(size=(1000, 4)), rng.normal(size=(1000, 4))
    table = shift.detection_rates(
        ref_p, ref_z, pool_p, pool_z, np.zeros(1000, bool), lams=[0.0], sizes=[100], n_reps=100, seed=0, device="cpu"
    )
    assert set(table["detector"]) == {"ks_outputs", "mmd_features"}
    assert (table["detection_rate"] <= 0.12).all()
