"""E5: how well can a deployed model's accuracy and Brier score be estimated without labels?

Uses saved predictions only (no model is re-run). Scenarios on the test fold:
- e1:          the 12-lead ensemble on subgroups (age, sex, the three big sites) and on everyone
- e2a_naive:   the 12-lead model with leads removed (it becomes confidently wrong, see E2a)
- e2a_dropout: the lead-dropout model with leads removed (it stays calibrated)
The estimators (AC, ATC, DoC) are fitted on each model's fold-9 predictions with all 12 leads: the
deployment shift is the loss of leads, and labels for the shifted data would not exist.
Writes results/e5_labelfree.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_reliable.data import SUPERCLASSES, load_metadata, split_ids
from ptbxl_reliable.evaluate import load_preds
from ptbxl_reliable.labelfree import estimate_all
from ptbxl_reliable.subgroups import age_group
from ptbxl_reliable.uncertainty import ensemble_mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LEAD_CONFIGS = ["all12", "limb6", "leadII", "leadI"]


def _mean_probs(name: str, split: pd.DataFrame) -> np.ndarray:
    ecg_ids, probs = load_preds(RESULTS / "preds" / f"{name}.npz")
    if not np.array_equal(ecg_ids, split.index.to_numpy()):
        raise ValueError(f"{name}: stored predictions are not in split order")
    return ensemble_mean(probs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    meta = load_metadata(ROOT / "data" / "ptb-xl")
    ids = split_ids(meta)
    val, test = meta.loc[ids["val"]], meta.loc[ids["test"]]
    y_val, y = val[SUPERCLASSES].to_numpy(float), test[SUPERCLASSES].to_numpy(float)
    patients = test["patient_id"].to_numpy()

    rows = []

    def add(scenario: str, model: str, keep: np.ndarray, p: np.ndarray, p_val: np.ndarray) -> None:
        row = estimate_all(y[keep], p[keep], patients[keep], y_val, p_val, n_boot=args.n_boot)
        rows.append({"scenario": scenario, "model": model, **row})

    p_val, p = _mean_probs("e1_val", val), _mean_probs("e1_test", test)
    ages = age_group(test["age"]).astype(str).to_numpy()
    subgroups = {"all": np.ones(len(test), bool)}
    subgroups |= {f"age {g}": ages == g for g in ["<40", "40-59", "60-79", ">=80"]}
    subgroups |= {"male": (test["sex"] == 0).to_numpy(), "female": (test["sex"] == 1).to_numpy()}
    subgroups |= {f"site {s}": (test["site"] == s).to_numpy() for s in [0, 1, 2]}
    for scenario, keep in subgroups.items():
        add(scenario, "e1", keep, p, p_val)

    everyone = np.ones(len(test), bool)
    for strategy in ["naive", "dropout"]:
        p_val = _mean_probs(f"e2a_{strategy}_all12_val", val)
        for config in LEAD_CONFIGS:
            add(f"leads {config}", f"e2a_{strategy}", everyone, _mean_probs(f"e2a_{strategy}_{config}_test", test), p_val)

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "e5_labelfree.csv", index=False)
    show = table.copy()
    for method in ["ac", "atc", "doc"]:
        show[f"{method}_err"] = show[f"{method}_acc"] - show["true_acc"]
    show["ac_brier_err"] = show["ac_brier"] - show["true_brier"]
    cols = ["scenario", "model", "n", "true_acc", "ac_err", "atc_err", "doc_err", "true_brier", "ac_brier_err"]
    print(show[cols].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
