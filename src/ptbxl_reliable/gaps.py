"""Gaps in time: stretches where a (wearable-style) device recorded nothing, on all leads at once.

A time mask is True where the signal is present. Gaps are zero-filled, which after per-lead
standardisation is the lead's training mean: on its own, a gap looks like a flat line.
"""

import numpy as np
import torch

GAP_LEN = 50  # one short dropout in the "many" pattern: 0.5 s at 100 Hz
MAX_TRAIN_FRACTION = 0.6  # training gaps cover up to this share of the recording
P_NO_GAPS = 0.5  # share of training records that keep the whole signal
PATTERNS = ("one", "many")


def gap_mask(n_samples: int, fraction: float, pattern: str, rng: np.random.Generator) -> np.ndarray:
    """Bool (T,) mask with about `fraction` of the samples missing.

    "one":  a single contiguous gap of round(fraction * T) samples at a random position.
    "many": gaps of GAP_LEN samples at random positions (they may overlap) until at least
            round(fraction * T) samples are missing.
    """
    if not 0 <= fraction < 1:
        raise ValueError(f"fraction must be in [0, 1), got {fraction}")
    if pattern not in PATTERNS:
        raise ValueError(f"unknown gap pattern {pattern!r}, expected one of {PATTERNS}")
    present = np.ones(n_samples, dtype=bool)
    target = int(round(fraction * n_samples))
    if pattern == "one":
        start = rng.integers(0, n_samples - target + 1)
        present[start : start + target] = False
    else:
        while (~present).sum() < target:
            start = rng.integers(0, n_samples - GAP_LEN + 1)
            present[start : start + GAP_LEN] = False
    return present


def fixed_gap_masks(n_records: int, n_samples: int, fraction: float, pattern: str, seed: int) -> np.ndarray:
    """Test-time masks (N, T), fixed by `seed` so that every model is scored on the same gaps."""
    rng = np.random.default_rng(seed)
    return np.stack([gap_mask(n_samples, fraction, pattern, rng) for _ in range(n_records)])


def sample_training_gaps(batch_size: int, n_samples: int, generator: torch.Generator) -> torch.Tensor:
    """Random training masks (B, T): no gaps with probability P_NO_GAPS, otherwise a random pattern
    with a missing fraction drawn uniformly from [0, MAX_TRAIN_FRACTION)."""
    rng = np.random.default_rng(int(torch.randint(2**31 - 1, (1,), generator=generator)))
    masks = np.ones((batch_size, n_samples), dtype=bool)
    for i in range(batch_size):
        if rng.random() >= P_NO_GAPS:
            masks[i] = gap_mask(n_samples, rng.uniform(0, MAX_TRAIN_FRACTION), rng.choice(PATTERNS), rng)
    return torch.from_numpy(masks)
