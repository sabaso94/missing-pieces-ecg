"""E7: gaps in time, as when a wearable loses contact. Does training with gaps help, and does it help to
tell the model where the gaps are?

Two new lead-dropout ensembles trained with random gaps (E2a's lead dropout plus gaps in time):
- e7_gaps_zero:    gaps are zero-filled, so the model cannot tell a gap from a flat line
- e7_gaps_channel: the same, plus an extra input channel marking where the signal is present
Compared with E2a's lead-dropout ensemble (never saw gaps) and E1 (12 leads, never saw gaps) on the test
fold, with 0-50% of each recording missing as one gap or many short ones. Every model sees the same gaps.
Writes results/e7_gaps.csv and results/e7_diffs.csv.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ptbxl_reliable.data import prepare_data
from ptbxl_reliable.evaluate import evaluate
from ptbxl_reliable.gaps import fixed_gap_masks
from ptbxl_reliable.metrics import brier, macro_auroc, model_diff_ci
from ptbxl_reliable.train import TrainConfig, predict_ensemble, train_ensemble
from ptbxl_reliable.uncertainty import ensemble_mean

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = ROOT / "checkpoints"
FRACTIONS = [0.1, 0.2, 0.3, 0.5]
PATTERNS = ["one", "many"]
COLUMNS = ["macro_auroc", "macro_auroc_lo", "macro_auroc_hi", "brier", "brier_lo", "brier_hi", "ece", "mean_entropy"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    configs = {
        "e1": (TrainConfig(lead_mode="all12", device=args.device), "e1_all12"),
        "e2a_dropout": (TrainConfig(lead_mode="dropout", device=args.device), "e2a_dropout"),
        "e7_gaps_zero": (TrainConfig(lead_mode="dropout", gap_augment=True, device=args.device), "e7_gaps_zero"),
        "e7_gaps_channel": (
            TrainConfig(lead_mode="dropout", gap_augment=True, time_mask_channel=True, device=args.device),
            "e7_gaps_channel",
        ),
    }
    ensembles = {name: train_ensemble(cfg, data, ckpt_dir=CHECKPOINTS / ckpt) for name, (cfg, ckpt) in configs.items()}

    test = data.idx["test"]
    y, patients, n_samples = data.y[test], data.meta["patient_id"].to_numpy()[test], data.x.shape[2]
    # One set of gap positions per (fraction, pattern), shared by every model.
    settings = [(0.0, "none", None)] + [
        (f, pattern, fixed_gap_masks(len(test), n_samples, f, pattern, seed=100 + int(f * 100)))
        for f in FRACTIONS for pattern in PATTERNS
    ]

    rows, probs = [], {}
    for name, models in ensembles.items():
        for leads in ["all12"] if name == "e1" else ["all12", "leadI"]:
            for fraction, pattern, time_mask in settings:
                p = predict_ensemble(models, data.x[test], leads, device=args.device, time_mask=time_mask)
                probs[name, leads, fraction, pattern] = ensemble_mean(p)
                result = evaluate(y, p, patients, n_boot=args.n_boot)
                rows.append({"model": name, "leads": leads, "fraction": fraction, "pattern": pattern,
                             **{k: result[k] for k in COLUMNS}})
        print("evaluated", name)

    diff_rows = []
    for leads in ["all12", "leadI"]:
        for pattern in PATTERNS:
            for a, b in [("e7_gaps_channel", "e7_gaps_zero"), ("e7_gaps_zero", "e2a_dropout")]:
                for metric_name, metric in [("macro_auroc", macro_auroc), ("brier", brier)]:
                    point, lo, hi = model_diff_ci(metric, y, probs[a, leads, 0.3, pattern], probs[b, leads, 0.3, pattern],
                                                  patients, n_boot=args.n_boot)
                    diff_rows.append({"leads": leads, "fraction": 0.3, "pattern": pattern, "contrast": f"{a} - {b}",
                                      "metric": metric_name, "diff": point, "lo": lo, "hi": hi})

    table, diffs = pd.DataFrame(rows), pd.DataFrame(diff_rows)
    table.to_csv(ROOT / "results" / "e7_gaps.csv", index=False)
    diffs.to_csv(ROOT / "results" / "e7_diffs.csv", index=False)
    print(table.pivot_table(index=["leads", "model"], columns=["pattern", "fraction"], values="macro_auroc").round(3))
    print(diffs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
