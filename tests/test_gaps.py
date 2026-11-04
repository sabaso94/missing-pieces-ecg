import numpy as np
import pytest
import torch

from ptbxl_reliable import gaps


def test_gap_mask_counts():
    rng = np.random.default_rng(0)
    assert gaps.gap_mask(1000, 0.0, "one", rng).all()
    one = gaps.gap_mask(1000, 0.3, "one", rng)
    missing = np.flatnonzero(~one)
    assert len(missing) == 300 and (np.diff(missing) == 1).all()
    many = gaps.gap_mask(1000, 0.3, "many", rng)
    assert 300 <= (~many).sum() < 300 + gaps.GAP_LEN


def test_gap_mask_rejects_bad_input():
    rng = np.random.default_rng(0)
    for bad in (-0.1, 1.0):
        with pytest.raises(ValueError):
            gaps.gap_mask(1000, bad, "one", rng)
    with pytest.raises(ValueError):
        gaps.gap_mask(1000, 0.1, "burst", rng)


def test_fixed_gap_masks_deterministic():
    a = gaps.fixed_gap_masks(4, 1000, 0.2, "many", seed=3)
    np.testing.assert_array_equal(a, gaps.fixed_gap_masks(4, 1000, 0.2, "many", seed=3))
    assert a.shape == (4, 1000)


def test_sample_training_gaps_mixes_clean_and_gapped():
    masks = gaps.sample_training_gaps(400, 1000, torch.Generator().manual_seed(0))
    assert masks.shape == (400, 1000) and masks.dtype == torch.bool
    clean = masks.all(dim=1)
    assert 0.4 < clean.float().mean() < 0.6
    assert (~masks).float().mean(dim=1).max() < gaps.MAX_TRAIN_FRACTION + gaps.GAP_LEN / 1000
