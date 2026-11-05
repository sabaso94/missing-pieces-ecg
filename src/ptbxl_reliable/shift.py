"""Detecting a change in the patient population from unlabeled data (two-sample tests).

A deployed model sees batches of new ECGs without labels. We compare each batch with a reference
set (the validation fold) in two places: the model's predicted probabilities, and its internal
features. Both are standard "black-box shift detection" choices (Rabanser et al., 2019).
"""

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp

from ptbxl_reliable.metrics import brier


def mixture_batch(pool_size: int, in_group: np.ndarray, lam: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Indices into the pool for one deployment batch of size n.

    The first round(lam * n) are drawn from the subgroup, the rest from the whole pool (which
    contains the subgroup at its natural share). Drawn with replacement: subgroups are small.
    """
    n_group = int(round(lam * n))
    group = np.flatnonzero(in_group)
    from_group = rng.choice(group, size=n_group) if n_group else np.array([], dtype=int)
    return np.concatenate([from_group, rng.integers(0, pool_size, size=n - n_group)])


def ks_test(ref_p: np.ndarray, batch_p: np.ndarray) -> float:
    """Bonferroni-adjusted p-value of one KS test per class on predicted probabilities."""
    n_classes = ref_p.shape[1]
    p_values = [ks_2samp(ref_p[:, c], batch_p[:, c]).pvalue for c in range(n_classes)]
    return float(min(1.0, n_classes * min(p_values)))


def median_bandwidth(z: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    """Median pairwise distance in the reference: the usual default kernel width for MMD."""
    rng = np.random.default_rng(seed)
    if len(z) > max_n:
        z = z[rng.choice(len(z), size=max_n, replace=False)]
    d = torch.cdist(torch.from_numpy(z).double(), torch.from_numpy(z).double())
    return float(d[torch.triu_indices(len(z), len(z), offset=1).unbind()].median())


def mmd_test(
    ref_z: np.ndarray,
    batch_z: np.ndarray,
    bandwidth: float,
    n_perm: int = 200,
    seed: int = 0,
    device: str = "cpu",
) -> float:
    """Permutation p-value of the (biased) squared MMD with a Gaussian kernel (Gretton et al., 2012).

    The kernel matrix is built once. A split of the pooled sample into "reference" and "batch" is a
    weight vector a (1/n_ref on one side, -1/n_batch on the other), and MMD^2 = a^T K a. All
    permutations are stacked as columns of A, so every statistic comes from one matrix product.
    """
    z = torch.from_numpy(np.concatenate([ref_z, batch_z])).to(device, torch.float64)
    n_ref, n = len(ref_z), len(z)
    kernel = torch.exp(-torch.cdist(z, z) ** 2 / (2 * bandwidth**2))

    generator = torch.Generator().manual_seed(seed)
    is_ref = torch.zeros(n_perm + 1, n, dtype=torch.bool)
    is_ref[0, :n_ref] = True  # column 0: the observed split
    for i in range(1, n_perm + 1):
        is_ref[i, torch.randperm(n, generator=generator)[:n_ref]] = True
    weights = torch.where(is_ref, 1.0 / n_ref, -1.0 / (n - n_ref)).T.to(device, torch.float64)

    stats = ((kernel @ weights) * weights).sum(dim=0)
    return float((1 + (stats[1:] >= stats[0]).sum()) / (1 + n_perm))


def detection_rates(
    ref_p: np.ndarray,
    ref_z: np.ndarray,
    pool_p: np.ndarray,
    pool_z: np.ndarray,
    in_group: np.ndarray,
    lams: list[float],
    sizes: list[int],
    n_reps: int,
    seed: int = 0,
    device: str = "cpu",
    alpha: float = 0.05,
    pool_y: np.ndarray | None = None,
    n_perm: int = 200,
) -> pd.DataFrame:
    """Share of simulated deployment batches in which each test raises an alarm (p < alpha).

    Both tests see the same batches. With lam = 0 nothing has changed, so the rate is the false-alarm
    rate. With `pool_y`, also the mean change in Brier score of the batches against the whole pool:
    how much worse the model actually does on the shifted batches.
    """
    rng = np.random.default_rng(seed)
    bandwidth = median_bandwidth(ref_z)
    pool_brier = None if pool_y is None else brier(pool_y, pool_p)
    rows = []
    for lam in lams:
        for size in sizes:
            alarms = {"ks_outputs": 0, "mmd_features": 0}
            brier_changes = []
            for rep in range(n_reps):
                idx = mixture_batch(len(pool_p), in_group, lam, size, rng)
                alarms["ks_outputs"] += ks_test(ref_p, pool_p[idx]) < alpha
                alarms["mmd_features"] += mmd_test(ref_z, pool_z[idx], bandwidth, n_perm, seed + rep, device) < alpha
                if pool_y is not None:
                    brier_changes.append(brier(pool_y[idx], pool_p[idx]) - pool_brier)
            for detector, count in alarms.items():
                row = {"lam": lam, "batch_size": size, "detector": detector, "detection_rate": count / n_reps}
                if pool_y is not None:
                    row["brier_change"] = float(np.mean(brier_changes))
                rows.append({**row, "n_reps": n_reps})
    return pd.DataFrame(rows)
