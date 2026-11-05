from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ptbxl_reliable import data
from ptbxl_reliable.data import (
    SUPERCLASSES,
    PreparedData,
    fit_lead_stats,
    load_metadata,
    load_signals,
    restrict,
    split_ids,
    standardise,
)

FIXTURES = Path(__file__).parent / "fixtures"
REAL_DATA = Path(__file__).parents[1] / "data" / "ptb-xl"


@pytest.fixture
def meta():
    return load_metadata(FIXTURES)


def test_label_mapping(meta):
    # Any listed diagnostic code counts, whatever its likelihood (as in the PTB-XL benchmark).
    assert meta.loc[1, ["NORM", "MI", "STTC", "CD", "HYP"]].tolist() == [1, 1, 0, 0, 0]
    assert meta.loc[4, "HYP"] == 1
    assert meta.loc[5, "STTC"] == 1
    # 'FAKE' has diagnostic == 0, so it must not produce a CD label.
    assert meta.loc[6, ["NORM", "CD"]].tolist() == [1, 0]
    # Record 3 only has a rhythm statement (no superclass) and is dropped.
    assert 3 not in meta.index
    assert list(meta.columns[-5:]) == SUPERCLASSES


def test_age_capped(meta):
    assert meta.loc[1, "age"] == 90
    assert meta.loc[2, "age"] == 45
    assert np.isnan(meta.loc[4, "age"])


def test_split_ids_folds(meta):
    ids = split_ids(meta)
    assert set(ids["train"]) == {1, 5, 6}
    assert set(ids["val"]) == {2}
    assert set(ids["test"]) == {4}
    all_ids = np.concatenate([ids["train"], ids["val"], ids["test"]])
    assert len(all_ids) == len(set(all_ids))


def test_lead_stats_train_only():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=3.0, scale=2.0, size=(50, 12, 100)).astype(np.float32)
    train = np.arange(30)
    mean, std = fit_lead_stats(x[train])
    x_changed = x.copy()
    x_changed[30:] += 100.0  # change only val/test rows
    mean2, std2 = fit_lead_stats(x_changed[train])
    np.testing.assert_array_equal(mean, mean2)
    np.testing.assert_array_equal(std, std2)
    assert mean.shape == (12,) and std.shape == (12,)

    z = standardise(x[train], mean, std)
    np.testing.assert_allclose(z.mean(axis=(0, 2)), 0.0, atol=1e-4)
    np.testing.assert_allclose(z.std(axis=(0, 2)), 1.0, atol=1e-4)


def test_cache_mismatch_rebuilds(meta, tmp_path, monkeypatch):
    calls = []

    def fake_read(path):
        calls.append(path)
        return np.zeros((12, 1000), dtype=np.float32)

    monkeypatch.setattr(data, "_read_record", fake_read)

    x = load_signals(meta, FIXTURES, sampling_rate=100, cache_dir=tmp_path)
    assert x.shape == (len(meta), 12, 1000) and x.dtype == np.float32
    assert len(calls) == len(meta)

    load_signals(meta, FIXTURES, sampling_rate=100, cache_dir=tmp_path)
    assert len(calls) == len(meta)  # same ecg_ids -> served from cache

    load_signals(meta.iloc[:-1], FIXTURES, sampling_rate=100, cache_dir=tmp_path)
    assert len(calls) == 2 * len(meta) - 1  # different ecg_ids -> rebuilt


@pytest.mark.skipif(not (REAL_DATA / "ptbxl_database.csv").exists(), reason="PTB-XL not downloaded")
def test_real_data_no_patient_overlap():
    meta = load_metadata(REAL_DATA)
    ids = split_ids(meta)
    train_patients = set(meta.loc[ids["train"], "patient_id"])
    test_patients = set(meta.loc[ids["test"], "patient_id"])
    val_patients = set(meta.loc[ids["val"], "patient_id"])
    assert not train_patients & test_patients
    assert not train_patients & val_patients
    assert 21_000 <= len(meta) <= 21_799


def test_restrict_restandardises_without_touching_original():
    rng = np.random.default_rng(0)
    raw = rng.normal(3.0, 2.0, size=(40, 12, 50)).astype(np.float32)
    raw[20:] += 5.0  # rows 20+ come from a different "site"
    m, s = fit_lead_stats(raw)
    data = PreparedData(
        x=standardise(raw, m, s),
        y=np.zeros((40, 5), np.float32),
        meta=pd.DataFrame(index=range(40)),
        idx={"train": np.arange(0, 30), "val": np.arange(30, 35), "test": np.arange(35, 40)},
    )
    before = data.x.copy()
    sub = restrict(data, np.arange(0, 20), np.arange(30, 35))
    np.testing.assert_array_equal(data.x, before)
    m2, s2 = fit_lead_stats(raw[:20])
    np.testing.assert_allclose(sub.x, standardise(raw, m2, s2), atol=1e-4)
    np.testing.assert_array_equal(sub.idx["train"], np.arange(0, 20))
    np.testing.assert_array_equal(sub.idx["test"], data.idx["test"])
