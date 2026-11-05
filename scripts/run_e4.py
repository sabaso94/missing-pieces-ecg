"""E4: can a two-sample test notice that the deployed model's patient population has changed?

Reference = validation fold (what the model was checked on); deployment pool = test fold. Simulated
deployment batches draw a share lam of their records from one subgroup (age >= 80, women, site 0)
and the rest from the whole pool; lam = 0 is "no shift" and measures false alarms. Two detectors:
- ks_outputs:   KS test per class on the E1 ensemble's probabilities (Bonferroni over classes)
- mmd_features: MMD permutation test on the pooled features of E1 member seed 0, PCA to 32 dims
Writes results/e4_detection.csv; the features are stored for later experiments.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA

from ptbxl_reliable.data import SUPERCLASSES, prepare_data
from ptbxl_reliable.evaluate import load_preds
from ptbxl_reliable.model import ECGResNet
from ptbxl_reliable.shift import detection_rates
from ptbxl_reliable.train import extract_features
from ptbxl_reliable.uncertainty import ensemble_mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LAMS = [0.25, 0.5, 1.0]
SIZES = [50, 100, 200, 500, 1000]
N_REPS = 100
N_PERM = 200
PCA_DIMS = 32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    model = ECGResNet()
    model.load_state_dict(torch.load(ROOT / "checkpoints" / "e1_all12" / "seed0.pt", map_location="cpu"))

    probs, feats = {}, {}
    for split in ["val", "test"]:
        idx = data.idx[split]
        ecg_ids, member_probs = load_preds(RESULTS / "preds" / f"e1_{split}.npz")
        if not np.array_equal(ecg_ids, data.meta.index[idx].to_numpy()):
            raise ValueError(f"e1_{split}: stored predictions are not in {split}-split order")
        probs[split] = ensemble_mean(member_probs)
        feats[split] = extract_features(model, data.x[idx], "all12", device=args.device)
        np.savez(RESULTS / "preds" / f"e1_seed0_features_{split}.npz", ecg_ids=ecg_ids, features=feats[split])

    # PCA fitted on the reference only: the deployment data must not shape the test.
    pca = PCA(n_components=PCA_DIMS, random_state=0).fit(feats["val"])
    ref_z, pool_z = pca.transform(feats["val"]), pca.transform(feats["test"])

    test = data.meta.iloc[data.idx["test"]]
    pool_y = test[SUPERCLASSES].to_numpy(dtype=float)
    shifts = {
        "age_80plus": (test["age"] >= 80).to_numpy(),
        "female": (test["sex"] == 1).to_numpy(),
        "site0": (test["site"] == 0).to_numpy(),
    }
    common = dict(sizes=SIZES, n_reps=N_REPS, device=args.device, pool_y=pool_y, n_perm=N_PERM)
    tables = [
        detection_rates(probs["val"], ref_z, probs["test"], pool_z, np.zeros(len(test), bool), [0.0], seed=0, **common)
        .assign(shift="none")
    ]
    for i, (name, in_group) in enumerate(shifts.items(), start=1):
        print(f"{name}: {in_group.sum()} test records")
        table = detection_rates(probs["val"], ref_z, probs["test"], pool_z, in_group, LAMS, seed=i, **common)
        tables.append(table.assign(shift=name))

    result = pd.concat(tables)[["shift", "lam", "batch_size", "detector", "detection_rate", "brier_change", "n_reps"]]
    result.to_csv(RESULTS / "e4_detection.csv", index=False)
    print(result.pivot_table(index=["shift", "lam", "detector"], columns="batch_size", values="detection_rate").round(2))


if __name__ == "__main__":
    main()
