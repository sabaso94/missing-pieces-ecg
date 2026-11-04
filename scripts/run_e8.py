"""E8: can the subgroup gaps of E3 be narrowed, and at what cost overall?

Variants of the 12-lead model, all scored on the test fold:
- e1:              the E1 ensemble as it is
- e1_platt_global: Platt scaling per class, fitted on fold 9 for everyone (does calibration alone help?)
- e1_platt_sex:    Platt scaling per class and sex (needs sex at deployment; always recorded in PTB-XL)
- e1_platt_age:    Platt scaling per class and age group
- reweighted:      a new ensemble trained like E1, with each record's loss weighted by the inverse size of its
                   sex x age cell in the training set (every cell carries the same total weight)
A gap counts as narrowed only if the CI of its change against E1 (same patient resamples) excludes zero.
Writes results/e8_groups.csv, e8_gaps.csv, e8_overall.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_reliable.data import prepare_data
from ptbxl_reliable.evaluate import evaluate, load_preds, save_preds
from ptbxl_reliable.metrics import brier, macro_auroc, model_diff_ci
from ptbxl_reliable.recalibration import apply_group_platt, apply_platt, fit_group_platt, fit_platt
from ptbxl_reliable.subgroups import age_group, gap_change_ci, gap_ci, group_weights, subgroup_table
from ptbxl_reliable.train import TrainConfig, predict_ensemble, train_ensemble
from ptbxl_reliable.uncertainty import ensemble_mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEX_LABELS = {0: "male", 1: "female"}
GAPS = [("sex", "male", "female"), ("age", ">=80", "60-79"), ("age", "<40", "60-79")]  # E3's reference groups
METRICS = [("macro_auroc", macro_auroc), ("brier", brier)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    meta = data.meta.copy()
    meta["sex_group"] = pd.Categorical(meta["sex"].map(SEX_LABELS), categories=["male", "female"], ordered=True)
    meta["age_group"] = age_group(meta["age"])  # no unknown ages in PTB-XL after loading

    # Reweighted ensemble: weights from the training rows' sex x age cells; other rows are never trained on.
    train = data.idx["train"]
    cells = meta["sex_group"].astype(str) + " " + meta["age_group"].astype(str)
    weights = np.ones(len(meta), dtype=np.float32)
    weights[train] = group_weights(cells.iloc[train])
    print("training weights per cell:", pd.Series(weights[train], index=cells.iloc[train]).groupby(level=0).first()
          .round(2).to_dict())
    models = train_ensemble(TrainConfig(lead_mode="all12", sample_weight=weights, device=args.device), data,
                            ckpt_dir=ROOT / "checkpoints" / "e8_reweighted")

    probs = {}
    for split in ["val", "test"]:
        idx = data.idx[split]
        ecg_ids, probs["e1", split] = load_preds(RESULTS / "preds" / f"e1_{split}.npz")
        if not np.array_equal(ecg_ids, meta.index[idx].to_numpy()):
            raise ValueError(f"e1_{split}: stored predictions are not in {split}-split order")
        probs["reweighted", split] = predict_ensemble(models, data.x[idx], "all12", device=args.device)
        save_preds(RESULTS / "preds" / f"e8_reweighted_{split}.npz", meta.index[idx].to_numpy(),
                   probs["reweighted", split])

    # Recalibration: fitted on E1's fold-9 ensemble mean, applied to its fold-10 ensemble mean (one "member").
    val, test = data.idx["val"], data.idx["test"]
    p_val, p_test = ensemble_mean(probs["e1", "val"]), ensemble_mean(probs["e1", "test"])
    y_val, y = data.y[val], data.y[test]
    variants = {"e1": probs["e1", "test"]}
    variants["e1_platt_global"] = apply_platt(p_test, fit_platt(y_val, p_val))[None]
    for attribute in ["sex", "age"]:
        g_val = meta[f"{attribute}_group"].astype(str).to_numpy()[val]
        g_test = meta[f"{attribute}_group"].astype(str).to_numpy()[test]
        variants[f"e1_platt_{attribute}"] = apply_group_platt(p_test, g_test, fit_group_platt(y_val, p_val, g_val))[None]
    variants["reweighted"] = probs["reweighted", "test"]

    test_meta = meta.iloc[test]
    patients = test_meta["patient_id"].to_numpy()
    mean = {name: ensemble_mean(p) for name, p in variants.items()}
    group_tables, gap_rows, overall_rows = [], [], []
    for name, p in variants.items():
        for attribute in ["sex", "age"]:
            table = subgroup_table(y, p, test_meta, f"{attribute}_group", n_boot=args.n_boot)
            group_tables.append(table.assign(variant=name, attribute=attribute, n_members=len(p)))

        for attribute, group, versus in GAPS:
            groups = test_meta[f"{attribute}_group"].astype(str).to_numpy()
            in_a, in_b = groups == group, groups == versus
            for metric_name, metric in METRICS:
                gap, gap_lo, gap_hi = gap_ci(metric, y, mean[name], in_a, in_b, patients, n_boot=args.n_boot)
                row = {"variant": name, "attribute": attribute, "group": group, "versus": versus,
                       "metric": metric_name, "gap": gap, "gap_lo": gap_lo, "gap_hi": gap_hi}
                if name != "e1":
                    change, lo, hi = gap_change_ci(metric, y, mean[name], mean["e1"], in_a, in_b, patients,
                                                   n_boot=args.n_boot)
                    row.update({"change": change, "change_lo": lo, "change_hi": hi})
                    # A map that is monotone within each compared group cannot change its AUROC: true for global
                    # Platt, and for per-group Platt when the gap is between groups of the same attribute.
                    monotone = name in ("e1_platt_global", f"e1_platt_{attribute}")
                    if monotone and metric_name == "macro_auroc" and abs(change) > 1e-4:
                        raise AssertionError(f"{name}: recalibration changed the {group} AUROC gap by {change}")
                gap_rows.append(row)

        result = evaluate(y, p, patients, n_boot=args.n_boot)
        row = {"variant": name, "n_members": len(p), **{k: result[k] for k in
               ["macro_auroc", "macro_auroc_lo", "macro_auroc_hi", "brier", "brier_lo", "brier_hi", "ece"]}}
        if name != "e1":
            for metric_name, metric in METRICS:
                diff, lo, hi = model_diff_ci(metric, y, mean[name], mean["e1"], patients, n_boot=args.n_boot)
                row.update({f"{metric_name}_vs_e1": diff, f"{metric_name}_vs_e1_lo": lo, f"{metric_name}_vs_e1_hi": hi})
        overall_rows.append(row)
        print("evaluated", name)

    groups_table = pd.concat(group_tables)
    groups_table = groups_table[["variant", "attribute"] + [c for c in groups_table if c not in ("variant", "attribute")]]
    gaps, overall = pd.DataFrame(gap_rows), pd.DataFrame(overall_rows)
    groups_table.to_csv(RESULTS / "e8_groups.csv", index=False)
    gaps.to_csv(RESULTS / "e8_gaps.csv", index=False)
    overall.to_csv(RESULTS / "e8_overall.csv", index=False)
    print(gaps.round(4).to_string(index=False))
    print(overall.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
