"""E3: subgroup performance and calibration by sex and age, on the test split.

Uses the test predictions saved by E1/E2a/E2b, so no model is re-run. Models:
- e1_all12:              ECG-only baseline, 12 leads
- e2a_<selected>_all12:  the missing-lead model chosen on val, 12 leads
- e2a_<selected>_leadI:  the same model on lead I only (does reduced input hurt some groups more?)
- e2b_<selected>_full:   the metadata fusion model chosen on val, metadata as recorded
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_reliable.data import SUPERCLASSES, load_metadata, split_ids
from ptbxl_reliable.evaluate import load_preds
from ptbxl_reliable.subgroups import UNKNOWN, age_group, merge_small_bins, subgroup_table

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEX_LABELS = {0: "male", 1: "female"}  # coding checked in the data step (mean height 173.5 vs 161.0 cm)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    meta = load_metadata(ROOT / "data" / "ptb-xl")
    test = meta.loc[split_ids(meta)["test"]].copy()
    test["sex_group"] = pd.Categorical(test["sex"].map(SEX_LABELS), categories=["male", "female"], ordered=True)
    test["age_group"], merges = merge_small_bins(age_group(test["age"]))

    lead_sel = json.loads((RESULTS / "e2a_selection.json").read_text())["strategy"]
    meta_sel = json.loads((RESULTS / "e2b_selection.json").read_text())["strategy"]
    models = {
        "e1_all12": "e1_test",
        f"e2a_{lead_sel}_all12": f"e2a_{lead_sel}_all12_test",
        f"e2a_{lead_sel}_leadI": f"e2a_{lead_sel}_leadI_test",
        f"e2b_{meta_sel}_full": f"e2b_{meta_sel}_full_test",
    }

    tables = []
    for model, preds_name in models.items():
        ecg_ids, probs = load_preds(RESULTS / "preds" / f"{preds_name}.npz")
        if not np.array_equal(ecg_ids, test.index.to_numpy()):
            raise ValueError(f"{preds_name}: stored predictions are not in test-split order")
        y = test[SUPERCLASSES].to_numpy()
        for attribute in ["sex", "age"]:
            known = (test[f"{attribute}_group"] != UNKNOWN).to_numpy()
            table = subgroup_table(y[known], probs[:, known], test[known], f"{attribute}_group", n_boot=args.n_boot)
            tables.append(table.assign(model=model, attribute=attribute))

    table = pd.concat(tables)
    table = table[["model", "attribute"] + [c for c in table if c not in ("model", "attribute")]]
    table.to_csv(RESULTS / "e3_subgroups.csv", index=False)
    (RESULTS / "e3_age_merges.json").write_text(json.dumps(merges, indent=2))

    cols = ["n", "macro_auroc", "macro_auroc_lo", "macro_auroc_hi", "brier", "ece"]
    print(table.set_index(["model", "attribute", "group"])[cols].round(3).to_string())
    print("age bin merges:", merges or "none")


if __name__ == "__main__":
    main()
