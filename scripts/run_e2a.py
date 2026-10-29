"""E2a: missing leads. Three strategies, evaluated on four lead configurations.

- naive:      the E1 model (trained on 12 leads), tested with leads removed
- dropout:    one model trained with random lead removal, tested on every configuration
- specialist: a separate model per configuration (reference upper bound; the 12-lead specialist is E1)

Use --only to train a single ensemble, e.g. to run several on different GPUs at once:
    python scripts/run_e2a.py --only specialist:leadI --device cuda:1
"""

import argparse
import json
from pathlib import Path

import pandas as pd

from ptbxl_reliable.data import prepare_data
from ptbxl_reliable.evaluate import evaluate, save_preds
from ptbxl_reliable.leads import LEAD_CONFIGS
from ptbxl_reliable.train import TrainConfig, predict_ensemble, train_ensemble
from ptbxl_reliable.uncertainty import ensemble_mean, mutual_information, predictive_entropy, risk_coverage

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
RESULTS = ROOT / "results"


def ensemble_specs() -> dict[str, tuple[str, Path]]:
    """name -> (lead_mode used in training, checkpoint directory)."""
    specs = {"naive": ("all12", CKPT / "e1_all12"), "dropout": ("dropout", CKPT / "e2a_dropout")}
    for config in LEAD_CONFIGS:
        ckpt = CKPT / "e1_all12" if config == "all12" else CKPT / f"e2a_{config}"
        specs[f"specialist:{config}"] = (config, ckpt)
    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--only", choices=list(ensemble_specs()), help="train this ensemble and exit")
    args = parser.parse_args()

    data = prepare_data(ROOT / "data" / "ptb-xl")
    specs = ensemble_specs()

    def ensemble(name: str):
        lead_mode, ckpt_dir = specs[name]
        return train_ensemble(TrainConfig(lead_mode=lead_mode, device=args.device), data, ckpt_dir=ckpt_dir)

    if args.only:
        ensemble(args.only)
        return

    # Which configurations each strategy is evaluated on.
    plan = {"naive": list(LEAD_CONFIGS), "dropout": list(LEAD_CONFIGS)}
    plan.update({f"specialist:{c}": [c] for c in LEAD_CONFIGS})

    rows, coverage_tables = [], []
    for name, configs in plan.items():
        models = ensemble(name)
        strategy = name.split(":")[0]
        for config in configs:
            for split in ["val", "test"]:
                idx = data.idx[split]
                y, patients = data.y[idx], data.meta["patient_id"].to_numpy()[idx]
                probs = predict_ensemble(models, data.x[idx], config, device=args.device)
                save_preds(RESULTS / "preds" / f"e2a_{strategy}_{config}_{split}.npz", data.meta.index[idx].to_numpy(), probs)
                result = evaluate(y, probs, patients, n_boot=args.n_boot)
                rows += [{"strategy": strategy, "config": config, "split": split, "metric": k, "value": v} for k, v in result.items()]

                if strategy in ("naive", "dropout") and config in ("all12", "leadI") and split == "test":
                    p_mean = ensemble_mean(probs)
                    for unc_name, unc in [("entropy", predictive_entropy(p_mean)), ("mutual_information", mutual_information(probs))]:
                        table = risk_coverage(y, p_mean, unc)
                        coverage_tables.append(table.assign(strategy=strategy, config=config, uncertainty=unc_name))

    table = pd.DataFrame(rows)
    table.to_csv(RESULTS / "e2a_missing_leads.csv", index=False)
    coverage = pd.concat(coverage_tables)
    coverage = coverage[["strategy", "config", "uncertainty", "coverage", "n", "macro_auroc", "brier"]]
    coverage.to_csv(RESULTS / "e2a_risk_coverage.csv", index=False)

    # Model selection for E3 uses the validation split only (never test): best of naive / dropout on lead I.
    val_lead_i = table.query("split == 'val' and config == 'leadI' and metric == 'macro_auroc' and strategy != 'specialist'")
    scores = dict(zip(val_lead_i["strategy"], val_lead_i["value"]))
    selection = {"strategy": max(scores, key=scores.get), "val_macro_auroc_leadI": scores}
    (RESULTS / "e2a_selection.json").write_text(json.dumps(selection, indent=2))

    summary = table.query("split == 'test' and metric in ['macro_auroc', 'ece', 'brier', 'mean_entropy']")
    print(summary.pivot_table(index=["strategy", "config"], columns="metric", values="value").round(4).to_string())
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
