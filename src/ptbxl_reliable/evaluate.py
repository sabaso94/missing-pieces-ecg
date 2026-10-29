"""Turning ensemble predictions into one row of reported numbers, and storing predictions."""

from pathlib import Path

import numpy as np

from ptbxl_reliable.data import SUPERCLASSES
from ptbxl_reliable.metrics import bootstrap_ci, brier, ece, macro_auroc, per_class_auroc
from ptbxl_reliable.uncertainty import ensemble_mean, mutual_information, predictive_entropy


def evaluate(y: np.ndarray, probs: np.ndarray, patient_ids: np.ndarray, n_boot: int = 1000) -> dict[str, float]:
    """Metrics of an ensemble's predictions `probs` (M, N, C), with patient-level 95% CIs.

    ECE gets no CI. Binned ECE is biased upward, and more so on a bootstrap resample (only
    ~63% unique records), so its percentile interval can sit entirely above the point estimate.
    The Brier score is a proper scoring rule without that problem and carries the
    statistical comparison of calibration; ECE and reliability diagrams describe it.
    """
    p = ensemble_mean(probs)
    result = {}
    for name, metric in [("macro_auroc", macro_auroc), ("brier", brier)]:
        point, lo, hi = bootstrap_ci(metric, y, p, patient_ids, n_boot=n_boot)
        result.update({name: point, f"{name}_lo": lo, f"{name}_hi": hi})
    result["ece"] = ece(y, p)
    for cls, auc in zip(SUPERCLASSES, per_class_auroc(y, p)):
        result[f"auroc_{cls}"] = float(auc)
    result["mean_entropy"] = float(predictive_entropy(p).mean())
    result["mean_mi"] = float(mutual_information(probs).mean())
    return result


def save_preds(path: Path, ecg_ids: np.ndarray, probs: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, ecg_ids=ecg_ids, probs=probs)


def load_preds(path: Path) -> tuple[np.ndarray, np.ndarray]:
    stored = np.load(path)
    return stored["ecg_ids"], stored["probs"]
