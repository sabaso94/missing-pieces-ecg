"""Loading PTB-XL: metadata, superclass labels, official splits and raw signals."""

import ast
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]
LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

# PTB-XL's recommended split: folds 1-8 train, 9 validation, 10 test.
# Folds are stratified and assigned per patient, so no patient crosses splits.
SPLIT_FOLDS = {"train": list(range(1, 9)), "val": [9], "test": [10]}

# Ages above 89 are recorded as 300 for anonymisation; we cap them at 90.
AGE_CAP = 90


def load_metadata(data_dir: Path) -> pd.DataFrame:
    """Read ptbxl_database.csv and attach one 0/1 column per diagnostic superclass.

    Records without any superclass label are dropped, as in the PTB-XL benchmark.
    """
    data_dir = Path(data_dir)
    meta = pd.read_csv(data_dir / "ptbxl_database.csv", index_col="ecg_id")
    statements = pd.read_csv(data_dir / "scp_statements.csv", index_col=0)

    # Only diagnostic statements map to a superclass (form and rhythm statements don't).
    diagnostic = statements[statements["diagnostic"] == 1]
    code_to_class = diagnostic["diagnostic_class"].to_dict()

    scp_codes = meta["scp_codes"].apply(ast.literal_eval)
    for cls in SUPERCLASSES:
        # The benchmark ignores the likelihood value: a code being listed is enough.
        meta[cls] = scp_codes.apply(lambda codes: int(any(code_to_class.get(c) == cls for c in codes)))

    meta = meta[meta[SUPERCLASSES].sum(axis=1) > 0].copy()
    meta["age"] = meta["age"].clip(upper=AGE_CAP)
    meta["patient_id"] = meta["patient_id"].astype(int)

    other = [c for c in meta.columns if c not in SUPERCLASSES]
    return meta[other + SUPERCLASSES]


def split_ids(meta: pd.DataFrame) -> dict[str, np.ndarray]:
    """ecg_ids for the train / val / test splits, from the official strat_fold column."""
    return {
        name: meta.index[meta["strat_fold"].isin(folds)].to_numpy()
        for name, folds in SPLIT_FOLDS.items()
    }


def _read_record(path: Path) -> np.ndarray:
    """Read one WFDB record as a (12, T) float32 array, leads in the order of LEADS."""
    signal, fields = wfdb.rdsamp(str(path))
    # PTB-XL headers write the augmented leads in capitals (AVR, AVL, AVF).
    names = [name.upper() for name in fields["sig_name"]]
    assert names == [lead.upper() for lead in LEADS], f"unexpected lead order in {path}: {names}"
    return signal.T.astype(np.float32)


def load_signals(
    meta: pd.DataFrame,
    data_dir: Path,
    sampling_rate: int = 100,
    cache_dir: Path | None = None,
) -> np.ndarray:
    """All signals for `meta` as float32 (N, 12, T), in `meta.index` order.

    Reading ~21k WFDB files takes a few minutes, so the array is cached together with
    the ecg_ids it was built from. If the ids differ (e.g. the record selection changed),
    the cache is rebuilt rather than silently returning misaligned signals.
    """
    data_dir = Path(data_dir)
    cache_dir = Path(cache_dir) if cache_dir is not None else data_dir.parent / "cache"
    cache_file = cache_dir / f"signals_{sampling_rate}.npz"
    ecg_ids = meta.index.to_numpy()

    if cache_file.exists():
        cached = np.load(cache_file)
        if np.array_equal(cached["ecg_ids"], ecg_ids):
            return cached["x"]

    column = {100: "filename_lr", 500: "filename_hr"}[sampling_rate]
    x = np.stack([_read_record(data_dir / name) for name in meta[column]])

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache_file, x=x, ecg_ids=ecg_ids)
    return x


def fit_lead_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-lead mean and std over records and time. Call this on training data only."""
    return x.mean(axis=(0, 2)), x.std(axis=(0, 2))


def standardise(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((x - mean[None, :, None]) / std[None, :, None]).astype(np.float32)


@dataclass
class PreparedData:
    x: np.ndarray  # standardised signals, (N, 12, T)
    y: np.ndarray  # multi-hot superclass labels, (N, 5)
    meta: pd.DataFrame  # one row per record, same order as x and y
    idx: dict[str, np.ndarray]  # integer row positions for "train" / "val" / "test"


def prepare_data(data_dir: Path, sampling_rate: int = 100) -> PreparedData:
    """Everything the experiments need: standardised signals, labels, metadata and splits."""
    meta = load_metadata(data_dir)
    x = load_signals(meta, data_dir, sampling_rate)
    idx = {name: meta.index.get_indexer(ids) for name, ids in split_ids(meta).items()}

    mean, std = fit_lead_stats(x[idx["train"]])
    return PreparedData(
        x=standardise(x, mean, std),
        y=meta[SUPERCLASSES].to_numpy(dtype=np.float32),
        meta=meta,
        idx=idx,
    )


def restrict(data: PreparedData, train_idx: np.ndarray, val_idx: np.ndarray) -> PreparedData:
    """The same records with a smaller training and validation set, re-standardised on the new training set.

    Standardisation is affine per lead, so re-standardising the already-standardised signals gives the
    same result as standardising the raw signals with the new training statistics. Returns a new object;
    `data.x` is not modified. The test split is kept.
    """
    mean, std = fit_lead_stats(data.x[train_idx])
    idx = {**data.idx, "train": np.asarray(train_idx), "val": np.asarray(val_idx)}
    return PreparedData(x=standardise(data.x, mean, std), y=data.y, meta=data.meta, idx=idx)
