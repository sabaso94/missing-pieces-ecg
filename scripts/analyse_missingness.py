"""How is height/weight missingness distributed, and is it predictable from where and when the ECG was recorded?

If missingness depends on site, device or year, it is not "missing completely at random", and that
changes how the E2b results should be read (a missingness indicator may carry real information).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from ptbxl_reliable.data import SUPERCLASSES, load_metadata, split_ids
from ptbxl_reliable.tabular import TAB_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    meta = load_metadata(ROOT / "data" / "ptb-xl")
    meta["year"] = pd.to_datetime(meta["recording_date"]).dt.year
    meta["site"] = meta["site"].fillna(-1).astype(int).astype(str)
    for col in ["height", "weight"]:
        meta[f"{col}_missing"] = meta[col].isna().astype(int)
    ids = split_ids(meta)
    train = meta.loc[ids["train"]]

    # 1. Missing fraction per column, overall and per split.
    rows = [{"column": c, "split": "all", "missing_fraction": meta[c].isna().mean()} for c in TAB_COLUMNS]
    for split, split_ids_ in ids.items():
        rows += [{"column": c, "split": split, "missing_fraction": meta.loc[split_ids_, c].isna().mean()} for c in TAB_COLUMNS]
    pd.DataFrame(rows).to_csv(RESULTS / "e2b_missingness.csv", index=False)

    # 2. Missing rate by recording site, device and year (all records).
    by_source = []
    for source in ["site", "device", "year"]:
        grouped = meta.groupby(source).agg(
            n=("height_missing", "size"), height_missing=("height_missing", "mean"), weight_missing=("weight_missing", "mean")
        )
        by_source.append(grouped.reset_index().rename(columns={source: "value"}).assign(source=source))
    by_source = pd.concat(by_source)[["source", "value", "n", "height_missing", "weight_missing"]]
    by_source.to_csv(RESULTS / "e2b_missingness_by_source.csv", index=False)

    # 3. Can site + device + year predict missingness? 5-fold CV AUROC on the training split.
    #    0.5 = no better than chance (consistent with random missingness); near 1 = missingness is systematic.
    features = pd.get_dummies(train[["site", "device", "year"]].astype(str))
    rows = []
    for target in ["height_missing", "weight_missing"]:
        auc = cross_val_score(
            LogisticRegression(max_iter=2000), features, train[target], cv=5, scoring="roc_auc"
        ).mean()
        rows.append({"analysis": "cv_auroc_site_device_year", "group": target, "value": auc})

    # 4. Is missingness related to the labels? Label prevalence for records with vs without height.
    for missing, group in train.groupby("height_missing"):
        for cls in SUPERCLASSES:
            rows.append({"analysis": f"prevalence_{cls}", "group": f"height_missing={missing}", "value": group[cls].mean()})
    predictability = pd.DataFrame(rows)
    predictability.to_csv(RESULTS / "e2b_missingness_predictability.csv", index=False)

    print(pd.read_csv(RESULTS / "e2b_missingness.csv").query("split == 'all'").round(3).to_string(index=False))
    print(by_source[by_source.source == "device"].round(3).to_string(index=False))
    print(by_source[by_source.source == "year"].round(3).to_string(index=False))
    top_sites = by_source[by_source.source == "site"].sort_values("n", ascending=False).head(8)
    print(top_sites.round(3).to_string(index=False))
    print(predictability.round(3).to_string(index=False))
    print("records per site with n >= 100:", int((by_source.query("source == 'site'").n >= 100).sum()))
    print("share of records at the largest site:", np.round(top_sites.n.iloc[0] / len(meta), 3))


if __name__ == "__main__":
    main()
