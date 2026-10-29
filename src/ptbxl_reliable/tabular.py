"""Tabular metadata (age, sex, height, weight) as model inputs, under three missing-data strategies.

- "mean_impute": standardised values, missing ones replaced by the training mean (0 after standardising).
- "indicator":   the same, plus one is-missing flag per column, so the model can tell
                 "average height" from "height unknown".
- "dropout":     the same features as "indicator"; the difference is in training, where whole
                 metadata blocks are randomly removed (TrainConfig.tab_dropout).
"""

import numpy as np
import pandas as pd

TAB_COLUMNS = ["age", "sex", "height", "weight"]
STRATEGIES = ["mean_impute", "indicator", "dropout"]


def fit_tab_stats(meta_train: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """Mean and std of the observed values per column. Call this on training records only."""
    return {col: (float(meta_train[col].mean()), float(meta_train[col].std())) for col in TAB_COLUMNS}


def make_tab_features(meta: pd.DataFrame, stats: dict[str, tuple[float, float]], strategy: str) -> np.ndarray:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}, expected one of {STRATEGIES}")
    values = np.stack([(meta[col].to_numpy(dtype=float) - stats[col][0]) / stats[col][1] for col in TAB_COLUMNS], axis=1)
    missing = np.isnan(values)
    values[missing] = 0.0
    if strategy != "mean_impute":
        values = np.concatenate([values, missing.astype(float)], axis=1)
    return values.astype(np.float32)


def remove_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """A copy of `meta` with every tabular field missing (the 'metadata removed' test condition)."""
    removed = meta.copy()
    removed[TAB_COLUMNS] = np.nan
    return removed


def tab_removed_vector(stats: dict[str, tuple[float, float]], strategy: str) -> np.ndarray:
    """The feature vector of a record whose metadata is entirely missing."""
    empty = pd.DataFrame({col: [np.nan] for col in TAB_COLUMNS})
    return make_tab_features(empty, stats, strategy)[0]
