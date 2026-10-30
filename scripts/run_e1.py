"""E1: ECG-only baseline. A 5-seed ensemble on all 12 leads, evaluated on validation (fold 9) and test (fold 10)."""

import argparse
from pathlib import Path

import pandas as pd

from ptbxl_reliable.data import prepare_data
from ptbxl_reliable.evaluate import evaluate, save_preds
from ptbxl_reliable.train import TrainConfig, predict_ensemble, train_ensemble

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    models = train_ensemble(
        TrainConfig(lead_mode="all12", device=args.device), data, ckpt_dir=ROOT / "checkpoints" / "e1_all12"
    )

    rows = []
    for split in ["val", "test"]:
        idx = data.idx[split]
        probs = predict_ensemble(models, data.x[idx], "all12", device=args.device)
        save_preds(ROOT / "results" / "preds" / f"e1_{split}.npz", data.meta.index[idx].to_numpy(), probs)
        result = evaluate(
            data.y[idx], probs, data.meta["patient_id"].to_numpy()[idx], n_boot=args.n_boot, per_class_ci=True
        )
        rows += [{"split": split, "metric": k, "value": v} for k, v in result.items()]

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "e1_baseline.csv", index=False)
    print(table.pivot(index="metric", columns="split", values="value").round(4).to_string())


if __name__ == "__main__":
    main()
