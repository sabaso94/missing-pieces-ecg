"""Post-hoc recalibration: Platt scaling per class, for everyone or separately per patient group (E8).

Each class gets a logistic regression on the logit of its predicted probability, fitted on the
validation fold. The map is monotone, so it changes calibration but never the ranking (AUROC) within
the records it is applied to.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_platt(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """(C, 2) parameters (a, b): the recalibrated probability is sigmoid(a * logit(p) + b).

    A class with only positives or only negatives in `y` has nothing to fit and keeps the identity (1, 0).
    """
    params = np.tile([1.0, 0.0], (p.shape[1], 1))
    for c in range(p.shape[1]):
        if 0 < y[:, c].sum() < len(y):
            model = LogisticRegression(C=1e6).fit(_logit(p[:, c])[:, None], y[:, c])
            params[c] = model.coef_[0, 0], model.intercept_[0]
    return params


def apply_platt(p: np.ndarray, params: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(params[:, 0] * _logit(p) + params[:, 1])))


def fit_group_platt(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict[str, np.ndarray]:
    """One set of Platt parameters per group (e.g. per sex): needs the group to be known at deployment."""
    return {str(g): fit_platt(y[groups == g], p[groups == g]) for g in np.unique(groups)}


def apply_group_platt(p: np.ndarray, groups: np.ndarray, params: dict[str, np.ndarray]) -> np.ndarray:
    out = np.empty_like(p, dtype=float)
    for g in np.unique(groups):
        out[groups == g] = apply_platt(p[groups == g], params[str(g)])
    return out
