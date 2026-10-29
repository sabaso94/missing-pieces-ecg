"""Lead configurations used to simulate missing leads, and the masks that remove them."""

import numpy as np
import torch

from ptbxl_reliable.data import LEADS

LEAD_CONFIGS = {
    "all12": LEADS,
    "limb6": ["I", "II", "III", "aVR", "aVL", "aVF"],  # no chest electrodes
    "leadI": ["I"],  # what a smartwatch records (wrist to opposite finger)
    "leadII": ["II"],  # the usual rhythm-monitoring lead
}

# Probability that a training example keeps all 12 leads under lead dropout. The rest is
# split evenly over the reduced configurations, so the model still sees plenty of full ECGs.
P_ALL12 = 0.5


def config_mask(name: str) -> np.ndarray:
    """Boolean (12,) mask, True where the lead is present. Unknown names raise KeyError."""
    present = set(LEAD_CONFIGS[name])
    return np.array([lead in present for lead in LEADS])


def apply_mask(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Zero-fill absent leads. x is (B, 12, T), mask is (B, 12).

    Signals are standardised per lead, so zero is the training mean of that lead.
    """
    if not mask.any(dim=1).all():
        raise ValueError("every record needs at least one lead present")
    return x * mask[:, :, None].to(x.dtype)


def sample_training_masks(batch_size: int, generator: torch.Generator) -> torch.Tensor:
    """Random lead configurations for lead-dropout training, as a (B, 12) bool tensor."""
    reduced = torch.from_numpy(np.stack([config_mask(n) for n in LEAD_CONFIGS if n != "all12"]))
    keep_all = torch.rand(batch_size, generator=generator) < P_ALL12
    choice = torch.randint(len(reduced), (batch_size,), generator=generator)
    masks = reduced[choice]
    masks[keep_all] = True
    return masks
