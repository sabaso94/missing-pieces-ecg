from pathlib import Path

import numpy as np
import pytest
import torch

from ptbxl_reliable.leads import config_mask
from ptbxl_reliable.model import ECGResNet


def _mask(name, batch):
    return torch.from_numpy(np.stack([config_mask(name)] * batch))


def test_output_shape():
    x = torch.randn(8, 12, 1000)
    assert ECGResNet()(x, _mask("all12", 8)).shape == (8, 5)
    model = ECGResNet(n_tab=8)
    assert model(x, _mask("all12", 8), tab=torch.randn(8, 8)).shape == (8, 5)


def test_mask_is_applied():
    torch.manual_seed(0)
    model = ECGResNet().eval()
    x = torch.randn(4, 12, 1000)
    x_other = x.clone()
    x_other[:, 1:] = torch.randn(4, 11, 1000)  # change every lead except lead I
    with torch.no_grad():
        out = model(x, _mask("leadI", 4))
        out_other = model(x_other, _mask("leadI", 4))
    assert torch.allclose(out, out_other)


def test_tab_required_when_configured():
    with pytest.raises(ValueError):
        ECGResNet(n_tab=4)(torch.randn(2, 12, 1000), _mask("all12", 2))


def test_param_count():
    n_params = sum(p.numel() for p in ECGResNet().parameters())
    assert n_params < 2_000_000


def test_features_then_head_equals_forward():
    torch.manual_seed(0)
    model = ECGResNet().eval()
    x, mask = torch.randn(3, 12, 1000), _mask("limb6", 3)
    with torch.no_grad():
        f = model.features(x, mask)
        assert f.shape == (3, 256)
        assert torch.allclose(model.head(torch.cat([f, mask.float()], dim=1)), model(x, mask))


def test_existing_checkpoint_still_loads():
    path = Path(__file__).resolve().parents[1] / "checkpoints" / "e1_all12" / "seed0.pt"
    if not path.exists():
        pytest.skip("checkpoints are not part of the repo")
    ECGResNet().load_state_dict(torch.load(path, map_location="cpu"), strict=True)
