import numpy as np
import pytest
import torch

from ptbxl_reliable.leads import LEAD_CONFIGS, apply_mask, config_mask, sample_training_masks


def test_config_masks():
    sums = {name: int(config_mask(name).sum()) for name in ["all12", "limb6", "leadI", "leadII"]}
    assert sums == {"all12": 12, "limb6": 6, "leadI": 1, "leadII": 1}
    assert config_mask("leadI")[0] and config_mask("leadII")[1]
    assert config_mask("limb6")[:6].all() and not config_mask("limb6")[6:].any()
    assert config_mask("all12").dtype == bool


def test_unknown_config_raises():
    with pytest.raises(KeyError):
        config_mask("leadV7")


def test_apply_mask_zero_fills():
    x = torch.randn(3, 12, 50)
    mask = torch.from_numpy(np.stack([config_mask("leadI")] * 3))
    out = apply_mask(x, mask)
    assert torch.equal(out[:, 0], x[:, 0])
    assert torch.count_nonzero(out[:, 1:]) == 0


def test_empty_mask_raises():
    x = torch.randn(2, 12, 50)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[1] = False
    with pytest.raises(ValueError):
        apply_mask(x, mask)


def test_sample_training_masks_distribution():
    g = torch.Generator().manual_seed(0)
    masks = sample_training_masks(10_000, g)
    assert masks.shape == (10_000, 12) and masks.dtype == torch.bool
    allowed = torch.from_numpy(np.stack([config_mask(name) for name in LEAD_CONFIGS]))
    matches = (masks[:, None, :] == allowed[None, :, :]).all(dim=2)
    assert matches.any(dim=1).all()
    share_all12 = masks.all(dim=1).float().mean().item()
    assert abs(share_all12 - 0.5) < 0.03
