"""E6: what does it cost to deploy at a hospital site the model never saw, and do E4/E5 notice?

For each of the three big sites s (0, 1, 2):
- loso:    an ensemble trained and selected without site s (folds 1-8 / fold 9, site s removed)
- control: an ensemble trained on a random subset of folds 1-8 of the same size as loso's, all sites
           included, selected on the full fold 9. loso - control isolates "never saw the site" from
           "trained on less data".
Both are compared with E1 on the fold-10 records of site s and of all other sites. Each restricted data
set is re-standardised with its own training statistics, as a model trained without site s would be.
Then E4's detectors (reference: loso's fold 9 without s; deployment: fold-10 site s) and E5's estimators
are run on the held-out site.
Writes results/e6_sites.csv, e6_diffs.csv, e6_detection.csv, e6_labelfree.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from ptbxl_reliable.data import prepare_data, restrict
from ptbxl_reliable.evaluate import evaluate, load_preds, save_preds
from ptbxl_reliable.labelfree import estimate_all
from ptbxl_reliable.metrics import brier, macro_auroc, model_diff_ci
from ptbxl_reliable.shift import detection_rates
from ptbxl_reliable.train import TrainConfig, extract_features, predict_ensemble, train_ensemble
from ptbxl_reliable.uncertainty import ensemble_mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SITES = [0, 1, 2]
SIZES = [50, 100, 200, 500, 1000]  # as E4
N_REPS = 100
N_PERM = 200
PCA_DIMS = 32
SITE_COLUMNS = ["macro_auroc", "macro_auroc_lo", "macro_auroc_hi", "brier", "brier_lo", "brier_hi", "ece"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    site = data.meta["site"].to_numpy()  # NaN (no site recorded) never equals s, so counts as "other"
    patients = data.meta["patient_id"].to_numpy()
    train, val, test = data.idx["train"], data.idx["val"], data.idx["test"]

    e1 = {}
    for split in ["val", "test"]:
        ecg_ids, e1[split] = load_preds(RESULTS / "preds" / f"e1_{split}.npz")
        if not np.array_equal(ecg_ids, data.meta.index[data.idx[split]].to_numpy()):
            raise ValueError(f"e1_{split}: stored predictions are not in {split}-split order")

    site_rows, diff_rows, detection_tables, labelfree_rows = [], [], [], []
    for s in SITES:
        loso_train, loso_val = train[site[train] != s], val[site[val] != s]
        rng = np.random.default_rng(s)
        control_train = np.sort(rng.choice(train, size=len(loso_train), replace=False))
        print(f"site {s}: loso trains on {len(loso_train)} records (control: same number, all sites)")

        probs = {"e1": e1["test"]}
        loso_val_probs = loso_data = loso_models = None
        for name, train_idx, val_idx in [("control", control_train, val), ("loso", loso_train, loso_val)]:
            sub = restrict(data, train_idx, val_idx)
            models = train_ensemble(
                TrainConfig(lead_mode="all12", device=args.device), sub,
                ckpt_dir=ROOT / "checkpoints" / f"e6_{name}_site{s}",
            )
            for split, idx in [("val", val_idx), ("test", test)]:
                split_probs = predict_ensemble(models, sub.x[idx], "all12", device=args.device)
                save_preds(RESULTS / "preds" / f"e6_{name}_site{s}_{split}.npz", data.meta.index[idx].to_numpy(),
                           split_probs)
                if split == "test":
                    probs[name] = split_probs
                elif name == "loso":
                    loso_val_probs = split_probs
            if name == "loso":
                loso_data, loso_models = sub, models
            del sub

        at_site = site[test] == s
        for model, model_probs in probs.items():
            for eval_set, keep in [("site", at_site), ("other_sites", ~at_site)]:
                result = evaluate(data.y[test][keep], model_probs[:, keep], patients[test][keep], n_boot=args.n_boot)
                site_rows.append({"site": s, "model": model, "eval_set": eval_set, "n": int(keep.sum()),
                                  **{k: result[k] for k in SITE_COLUMNS}})

        y_site, pid_site = data.y[test][at_site], patients[test][at_site]
        mean = {model: ensemble_mean(model_probs[:, at_site]) for model, model_probs in probs.items()}
        for a, b in [("loso", "control"), ("loso", "e1"), ("control", "e1")]:
            for metric_name, metric in [("macro_auroc", macro_auroc), ("brier", brier)]:
                point, lo, hi = model_diff_ci(metric, y_site, mean[a], mean[b], pid_site, n_boot=args.n_boot)
                diff_rows.append({"site": s, "contrast": f"{a} - {b}", "metric": metric_name,
                                  "diff": point, "lo": lo, "hi": hi})

        # E4 on the held-out site: the reference is what loso was checked on (fold 9 without s).
        ref_p, pool_p = ensemble_mean(loso_val_probs), ensemble_mean(probs["loso"])
        feats = {split: extract_features(loso_models[0], loso_data.x[idx], "all12", device=args.device)
                 for split, idx in [("val", loso_val), ("test", test)]}
        pca = PCA(n_components=PCA_DIMS, random_state=0).fit(feats["val"])
        ref_z, pool_z = pca.transform(feats["val"]), pca.transform(feats["test"])
        common = dict(sizes=SIZES, n_reps=N_REPS, device=args.device, n_perm=N_PERM, seed=s)
        detection_tables += [
            detection_rates(ref_p, ref_z, pool_p[~at_site], pool_z[~at_site], np.zeros((~at_site).sum(), bool),
                            [0.0], **common).assign(site=s, shift="none (other sites)"),
            detection_rates(ref_p, ref_z, pool_p[at_site], pool_z[at_site], np.ones(at_site.sum(), bool),
                            [1.0], **common).assign(site=s, shift="held-out site"),
        ]

        # E5 on the held-out site: estimators fitted on loso's fold 9 without s.
        row = estimate_all(y_site, pool_p[at_site], pid_site, data.y[loso_val], ref_p, n_boot=args.n_boot)
        labelfree_rows.append({"scenario": f"heldout_site{s}", "model": "e6_loso", **row})
        del loso_data, loso_models

    sites = pd.DataFrame(site_rows)
    diffs = pd.DataFrame(diff_rows)
    detection = pd.concat(detection_tables)[["site", "shift", "lam", "batch_size", "detector", "detection_rate",
                                             "n_reps"]]
    labelfree = pd.DataFrame(labelfree_rows)
    sites.to_csv(RESULTS / "e6_sites.csv", index=False)
    diffs.to_csv(RESULTS / "e6_diffs.csv", index=False)
    detection.to_csv(RESULTS / "e6_detection.csv", index=False)
    labelfree.to_csv(RESULTS / "e6_labelfree.csv", index=False)

    print(sites.round(4).to_string(index=False))
    print(diffs.round(4).to_string(index=False))
    print(detection.pivot_table(index=["site", "shift", "detector"], columns="batch_size",
                                values="detection_rate").round(2))
    print(labelfree.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
