"""E2b: missing metadata. ECG + (age, sex, height, weight) fusion under three missing-data strategies.

Each fusion ensemble, and the ECG-only E1 ensemble for comparison, is evaluated under three conditions:
- full:              the metadata as recorded (height/weight often missing)
- complete_metadata: only records where all four fields are present
- metadata_removed:  every field removed, i.e. the model gets the ECG only

Use --only to train a single ensemble, e.g. `--only indicator --device cuda:1`.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ptbxl_reliable.data import prepare_data
from ptbxl_reliable.evaluate import evaluate, save_preds
from ptbxl_reliable.tabular import (
    STRATEGIES,
    TAB_COLUMNS,
    fit_tab_stats,
    make_tab_features,
    remove_metadata,
    tab_removed_vector,
)
from ptbxl_reliable.train import TrainConfig, predict_ensemble, train_ensemble

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"
TAB_DROPOUT = 0.3  # share of training records whose metadata is removed, for the "dropout" strategy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--only", choices=STRATEGIES, help="train this ensemble and exit")
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    stats = fit_tab_stats(data.meta.iloc[data.idx["train"]])  # training records only

    def fusion_ensemble(strategy: str):
        cfg = TrainConfig(
            tab=make_tab_features(data.meta, stats, strategy),
            tab_dropout=TAB_DROPOUT if strategy == "dropout" else 0.0,
            tab_removed=tab_removed_vector(stats, strategy),
            device=args.device,
        )
        return train_ensemble(cfg, data, ckpt_dir=CKPT / f"e2b_{strategy}")

    if args.only:
        fusion_ensemble(args.only)
        return

    complete = data.meta[TAB_COLUMNS].notna().all(axis=1).to_numpy()
    ecg_only = train_ensemble(TrainConfig(device=args.device), data, ckpt_dir=CKPT / "e1_all12")  # loads E1

    rows = []
    for strategy in ["ecg_only"] + STRATEGIES:
        models = ecg_only if strategy == "ecg_only" else fusion_ensemble(strategy)
        for split in ["val", "test"]:
            idx = data.idx[split]
            for condition in ["full", "complete_metadata", "metadata_removed"]:
                rows_idx = idx[complete[idx]] if condition == "complete_metadata" else idx
                tab = None
                if strategy != "ecg_only":
                    meta = remove_metadata(data.meta) if condition == "metadata_removed" else data.meta
                    tab = make_tab_features(meta.iloc[rows_idx], stats, strategy)
                probs = predict_ensemble(models, data.x[rows_idx], "all12", tab, device=args.device)
                save_preds(
                    RESULTS / "preds" / f"e2b_{strategy}_{condition}_{split}.npz",
                    data.meta.index[rows_idx].to_numpy(),
                    probs,
                )
                result = evaluate(data.y[rows_idx], probs, data.meta["patient_id"].to_numpy()[rows_idx], args.n_boot)
                result["n"] = len(rows_idx)
                rows += [
                    {"strategy": strategy, "condition": condition, "split": split, "metric": k, "value": v}
                    for k, v in result.items()
                ]

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "e2b_metadata.csv", index=False)

    # Model selection for E3 on the validation split only: the best fusion strategy on the full condition.
    val = table.query("split == 'val' and condition == 'full' and metric == 'macro_auroc' and strategy != 'ecg_only'")
    scores = dict(zip(val["strategy"], val["value"]))
    selection = {"strategy": max(scores, key=scores.get), "val_macro_auroc_full": scores}
    (RESULTS / "e2b_selection.json").write_text(json.dumps(selection, indent=2))

    summary = table.query("split == 'test' and metric in ['macro_auroc', 'macro_auroc_lo', 'macro_auroc_hi', 'ece', 'n']")
    print(summary.pivot_table(index=["condition", "strategy"], columns="metric", values="value").round(4).to_string())
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
