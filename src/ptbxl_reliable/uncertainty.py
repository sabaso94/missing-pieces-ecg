"""Uncertainty from a deep ensemble, and selective prediction (risk-coverage).

`probs` arrays are (M, N, C): M ensemble members, N records, C labels.
Each label is a separate yes/no question, so entropies are binary and averaged over labels.
"""

import numpy as np
import pandas as pd

from ptbxl_reliable.metrics import brier, macro_auroc

EPS = 1e-7


def ensemble_mean(probs: np.ndarray) -> np.ndarray:
    """The ensemble's prediction: the average of its members' probabilities, (N, C)."""
    return probs.mean(axis=0)


def _binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def predictive_entropy(p_mean: np.ndarray) -> np.ndarray:
    """Total uncertainty per record (nats): binary entropy of the mean prediction, averaged over labels."""
    return _binary_entropy(p_mean).mean(axis=1)


def mutual_information(probs: np.ndarray) -> np.ndarray:
    """Epistemic uncertainty per record: how much the members disagree.

    Total entropy minus the members' average entropy. It is zero when all members agree,
    even if each of them is unsure (that remaining part is aleatoric uncertainty).
    """
    total = _binary_entropy(ensemble_mean(probs))
    expected = _binary_entropy(probs).mean(axis=0)
    return (total - expected).mean(axis=1)


def risk_coverage(
    y: np.ndarray,
    p_mean: np.ndarray,
    uncertainty: np.ndarray,
    coverages: np.ndarray = np.linspace(0.1, 1.0, 10),
) -> pd.DataFrame:
    """Performance when the model only answers on its most certain records.

    For each coverage level, keep the ceil(coverage * N) records with the lowest uncertainty
    and score them. If uncertainty is useful, performance rises as coverage falls.
    """
    order = np.argsort(uncertainty, kind="stable")
    rows = []
    for coverage in coverages:
        n = int(np.ceil(coverage * len(y)))
        keep = order[:n]
        rows.append(
            {
                "coverage": float(coverage),
                "n": n,
                "macro_auroc": macro_auroc(y[keep], p_mean[keep]),
                "brier": brier(y[keep], p_mean[keep]),
            }
        )
    return pd.DataFrame(rows)
