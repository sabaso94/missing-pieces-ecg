from pathlib import Path

import numpy as np
import pytest

from ptbxl_reliable.data import load_metadata
from ptbxl_reliable.tabular import (
    TAB_COLUMNS,
    fit_tab_stats,
    make_tab_features,
    remove_metadata,
    tab_removed_vector,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def meta():
    return load_metadata(FIXTURES)


def test_features_shapes(meta):
    stats = fit_tab_stats(meta)
    widths = {s: make_tab_features(meta, stats, s).shape for s in ["mean_impute", "indicator", "dropout"]}
    assert widths == {"mean_impute": (5, 4), "indicator": (5, 8), "dropout": (5, 8)}
    assert make_tab_features(meta, stats, "indicator").dtype == np.float32


def test_no_nan_and_indicators(meta):
    stats = fit_tab_stats(meta)
    feats = make_tab_features(meta, stats, "indicator")
    assert not np.isnan(feats).any()
    row = meta.index.get_loc(2)  # record 2 has no height and no weight
    height = TAB_COLUMNS.index("height")
    assert feats[row, height] == 0.0  # imputed with the (standardised) training mean
    assert feats[row, 4 + height] == 1.0
    assert feats[meta.index.get_loc(1), 4 + height] == 0.0
    with pytest.raises(ValueError):
        make_tab_features(meta, stats, "median")


def test_stats_train_only(meta):
    train = meta.index[meta["strat_fold"] <= 8]
    stats = fit_tab_stats(meta.loc[train])
    changed = meta.copy()
    changed.loc[~changed.index.isin(train), "weight"] = 500.0
    assert fit_tab_stats(changed.loc[train]) == stats


def test_removed_vector_matches_remove_metadata(meta):
    stats = fit_tab_stats(meta)
    for strategy in ["mean_impute", "indicator", "dropout"]:
        feats = make_tab_features(remove_metadata(meta), stats, strategy)
        expected = tab_removed_vector(stats, strategy)
        np.testing.assert_array_equal(feats, np.tile(expected, (len(meta), 1)))
    assert meta["height"].notna().any()  # remove_metadata must not modify its input
