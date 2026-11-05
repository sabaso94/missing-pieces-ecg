"""Estimating a deployed model's performance from its predictions alone, without labels.

`p` are ensemble-mean probabilities (N, C). Each class is a yes/no decision at threshold 0.5, and the
model's confidence in that decision is max(p, 1 - p). All estimators are fitted on the validation fold,
where labels exist, and then applied to unlabeled deployment data.
"""

import numpy as np

from ptbxl_reliable.metrics import bootstrap_ci, brier


def _confidence(p: np.ndarray) -> np.ndarray:
    return np.maximum(p, 1 - p)


def accuracy(y: np.ndarray, p: np.ndarray) -> float:
    """Accuracy of the 0.5-threshold decisions, averaged over classes (the quantity being estimated)."""
    return float(((p >= 0.5) == (y == 1)).mean())


def ac_accuracy(p: np.ndarray) -> float:
    """Average confidence: if the model is calibrated, it is right as often as it is confident."""
    return float(_confidence(p).mean())


def ac_brier(p: np.ndarray) -> float:
    """Expected Brier score of a calibrated model: for y ~ Bernoulli(p), E[(p - y)^2] = p(1 - p)."""
    return float((p * (1 - p)).mean())


def fit_atc(y_val: np.ndarray, p_val: np.ndarray) -> np.ndarray:
    """Average thresholded confidence (Garg et al., 2022): one confidence threshold per class.

    The threshold is chosen so that, on validation data, the share of records more confident than it
    equals the accuracy. On new data, that share is the accuracy estimate. Unlike AC, this only needs
    the confidence ranking to carry over, not the calibration.
    """
    conf = _confidence(p_val)
    acc = ((p_val >= 0.5) == (y_val == 1)).mean(axis=0)
    return np.array([np.quantile(conf[:, c], 1 - acc[c]) for c in range(p_val.shape[1])])


def atc_accuracy(p: np.ndarray, thresholds: np.ndarray) -> float:
    return float((_confidence(p) > thresholds[None, :]).mean())


def doc_accuracy(p: np.ndarray, y_val: np.ndarray, p_val: np.ndarray) -> float:
    """Difference of confidences (Guillory et al., 2021): validation accuracy, shifted by however much the
    average confidence dropped (or rose) on the new data."""
    return accuracy(y_val, p_val) - (_confidence(p_val).mean() - _confidence(p).mean())


def estimate_all(
    y: np.ndarray,
    p: np.ndarray,
    patient_ids: np.ndarray,
    y_val: np.ndarray,
    p_val: np.ndarray,
    n_boot: int = 1000,
) -> dict[str, float]:
    """True accuracy and Brier score (patient-level 95% CIs) next to every label-free estimate.

    `y` is used only for the true values; the estimates see `p` and the validation data.
    """
    row = {"n": len(y)}
    for name, metric in [("acc", accuracy), ("brier", brier)]:
        point, lo, hi = bootstrap_ci(metric, y, p, patient_ids, n_boot=n_boot)
        row.update({f"true_{name}": point, f"true_{name}_lo": lo, f"true_{name}_hi": hi})
    row["ac_acc"] = ac_accuracy(p)
    row["atc_acc"] = atc_accuracy(p, fit_atc(y_val, p_val))
    row["doc_acc"] = doc_accuracy(p, y_val, p_val)
    row["ac_brier"] = ac_brier(p)
    return row
