"""Discrimination and calibration metrics for multi-label predictions, with patient-level bootstrap CIs.

All functions take y (N, C) with 0/1 labels and p (N, C) with predicted probabilities.
"""

from typing import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


def per_class_auroc(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """AUROC per class; NaN where a class has only positives or only negatives (AUROC undefined)."""
    aucs = np.full(y.shape[1], np.nan)
    for c in range(y.shape[1]):
        if 0 < y[:, c].sum() < len(y):
            aucs[c] = roc_auc_score(y[:, c], p[:, c])
    return aucs


def macro_auroc(y: np.ndarray, p: np.ndarray) -> float:
    """Unweighted mean AUROC over the classes where it is defined."""
    aucs = per_class_auroc(y, p)
    return float(np.nanmean(aucs)) if not np.isnan(aucs).all() else float("nan")


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error, computed per class as a binary problem and averaged over classes.

    Within each equal-width probability bin, compare the mean prediction with the observed
    frequency of positives, weighted by the share of samples in the bin.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    errors = []
    for c in range(y.shape[1]):
        # np.digitize against the inner edges gives bins 0..n_bins-1, with p == 1.0 in the last one.
        bins = np.digitize(p[:, c], edges[1:-1])
        error = 0.0
        for b in range(n_bins):
            in_bin = bins == b
            if in_bin.any():
                gap = abs(y[in_bin, c].mean() - p[in_bin, c].mean())
                error += in_bin.mean() * gap
        errors.append(error)
    return float(np.mean(errors))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error between probabilities and labels, over all records and classes."""
    return float(np.mean((p - y) ** 2))


def _patient_resample_indices(patient_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row indices of one bootstrap sample: patients drawn with replacement, all their records kept.

    Several ECGs can come from the same patient and are not independent, so resampling
    records would make the confidence intervals too narrow.
    """
    patients, inverse = np.unique(patient_ids, return_inverse=True)
    # Rows sorted by patient, so patient k's rows are order[starts[k] : starts[k] + counts[k]].
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse)
    starts = np.cumsum(counts) - counts

    drawn = rng.integers(0, len(patients), size=len(patients))
    lengths = counts[drawn]
    # Offset of each output row within its drawn patient's block: 0, 1, ..., length - 1.
    within = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
    return order[np.repeat(starts[drawn], lengths) + within]


def bootstrap_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) of a 95% percentile bootstrap CI, resampling patients."""
    rng = np.random.default_rng(seed)
    point = metric(y, p)
    samples = []
    for _ in range(n_boot):
        idx = _patient_resample_indices(patient_ids, rng)
        samples.append(metric(y[idx], p[idx]))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def model_diff_ci(
    metric: Callable[[np.ndarray, np.ndarray], float],
    y: np.ndarray,
    p_a: np.ndarray,
    p_b: np.ndarray,
    patient_ids: np.ndarray,
    n_boot: int = 1000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """metric(model a) - metric(model b) on the same records, with a paired 95% bootstrap CI.

    Both models are scored on the same patient resample each time, so the variation the two share
    (which records happen to be drawn) cancels out of the difference.
    """
    rng = np.random.default_rng(seed)
    point = metric(y, p_a) - metric(y, p_b)
    samples = []
    for _ in range(n_boot):
        idx = _patient_resample_indices(patient_ids, rng)
        samples.append(metric(y[idx], p_a[idx]) - metric(y[idx], p_b[idx]))
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return float(point), float(lo), float(hi)
