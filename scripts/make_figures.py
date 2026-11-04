"""Figures for the README, built only from results/ (labels for the reliability diagram come from the metadata)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ptbxl_reliable.data import SUPERCLASSES, load_metadata, split_ids
from ptbxl_reliable.evaluate import load_preds
from ptbxl_reliable.leads import LEAD_CONFIGS
from ptbxl_reliable.uncertainty import ensemble_mean

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

# Fixed categorical order (never cycled), plus ink and chrome colours.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
CONFIG_LABELS = {"all12": "12 leads", "limb6": "6 limb leads", "leadII": "lead II", "leadI": "lead I"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2, "legend.frameon": False, "legend.labelcolor": INK_2, "font.size": 9.5,
    "lines.linewidth": 2, "savefig.dpi": 200,
})


def _wide(table: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    """(…, metric, value) long table -> one column per metric."""
    return table.pivot_table(index=index, columns="metric", values="value").reset_index()


def _title(fig, text: str, legend_ax=None) -> None:
    """Figure title above the panel titles; the legend (if any) in one row beside it, clear of the data."""
    fig.suptitle(text, x=0.01, y=1.06, ha="left", color=INK, fontsize=12)
    if legend_ax is not None:
        handles, labels = legend_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.99, 1.0), ncol=len(labels))


def _save(fig, name: str) -> None:
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / name, bbox_inches="tight")
    plt.close(fig)


def e1_reliability() -> None:
    meta = load_metadata(ROOT / "data" / "ptb-xl")
    test = meta.loc[split_ids(meta)["test"]]
    ecg_ids, probs = load_preds(RESULTS / "preds" / "e1_test.npz")
    assert np.array_equal(ecg_ids, test.index.to_numpy())
    y, p = test[SUPERCLASSES].to_numpy(), ensemble_mean(probs)
    edges = np.linspace(0, 1, 16)  # the 15 bins used by metrics.ece

    fig, axes = plt.subplots(1, 5, figsize=(12, 2.8), sharey=True)
    for c, (ax, cls) in enumerate(zip(axes, SUPERCLASSES)):
        bins = np.digitize(p[:, c], edges[1:-1])
        keep = [b for b in range(15) if (bins == b).sum() >= 10]  # skip near-empty bins
        mean_p = [p[bins == b, c].mean() for b in keep]
        freq = [y[bins == b, c].mean() for b in keep]
        ax.plot([0, 1], [0, 1], color=AXIS, linewidth=1, linestyle="--")
        ax.plot(mean_p, freq, color=SERIES[0], marker="o", markersize=4)
        ax.set(title=cls, xlim=(0, 1), ylim=(0, 1), xlabel="predicted probability")
        ax.set_xticks([0, 0.5, 1], ["0", "0.5", "1"])
    axes[0].set_ylabel("observed frequency")
    _title(fig, "E1: ensemble calibration on the test set (dashed = perfect calibration)")
    _save(fig, "e1_reliability.png")


def e2a_leads() -> None:
    table = _wide(pd.read_csv(RESULTS / "e2a_missing_leads.csv").query("split == 'test'"), ["strategy", "config"])
    strategies = [("naive", "trained on 12 leads"), ("dropout", "lead dropout"), ("specialist", "specialist per config")]
    x = np.arange(len(LEAD_CONFIGS))

    fig, (ax_auc, ax_ece) = plt.subplots(1, 2, figsize=(10, 3.4))
    for k, (strategy, label) in enumerate(strategies):
        rows = table[table["strategy"] == strategy].set_index("config").reindex(LEAD_CONFIGS)
        offset = (k - 1) * 0.18
        err = [rows["macro_auroc"] - rows["macro_auroc_lo"], rows["macro_auroc_hi"] - rows["macro_auroc"]]
        ax_auc.errorbar(x + offset, rows["macro_auroc"], yerr=err, fmt="o", color=SERIES[k], markersize=6,
                        elinewidth=1.5, capsize=0, label=label)
        ax_ece.plot(x + offset, rows["ece"], "o", color=SERIES[k], markersize=6, label=label)
    ax_auc.axhline(0.5, color=MUTED, linewidth=1, linestyle=":")
    ax_auc.set(ylabel="macro-AUROC (95% CI)", ylim=(0.4, 1.0), title="Discrimination")
    ax_ece.set(ylabel="ECE (lower is better)", ylim=(0, None), title="Calibration")
    for ax in (ax_auc, ax_ece):
        ax.set_xticks(x, [CONFIG_LABELS[c] for c in LEAD_CONFIGS])
        ax.grid(axis="x", visible=False)
    _title(fig, "E2a: removing leads at test time", legend_ax=ax_auc)
    _save(fig, "e2a_leads.png")


def e2a_risk_coverage() -> None:
    table = pd.read_csv(RESULTS / "e2a_risk_coverage.csv").query("config == 'leadI' and uncertainty == 'entropy'")
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for k, (strategy, label) in enumerate([("naive", "trained on 12 leads"), ("dropout", "lead dropout")]):
        rows = table[table["strategy"] == strategy]
        ax.plot(rows["coverage"], rows["brier"], color=SERIES[k], marker="o", markersize=4)
        last = rows.iloc[-1]
        ax.annotate(label, (last["coverage"], last["brier"]), xytext=(6, 0), textcoords="offset points",
                    va="center", color=INK_2)
    ax.set(xlabel="coverage (share of records kept, least uncertain first)", ylabel="Brier score (lower is better)",
           xlim=(0.05, 1.3), ylim=(0, None))
    ax.set_xticks(np.linspace(0.1, 1, 10))
    ax.set_title("E2a: abstaining on uncertain records, lead I only", loc="left")
    _save(fig, "e2a_risk_coverage.png")


def e2b_metadata() -> None:
    table = _wide(pd.read_csv(RESULTS / "e2b_metadata.csv").query("split == 'test'"), ["strategy", "condition"])
    strategies = [("ecg_only", "ECG only"), ("mean_impute", "+ metadata, mean impute"),
                  ("indicator", "+ metadata, missing indicators"), ("dropout", "+ metadata, modality dropout")]
    conditions = [("full", "metadata as recorded"), ("complete_metadata", "complete metadata only"),
                  ("metadata_removed", "metadata removed")]
    y = np.arange(len(strategies))[::-1]

    fig, axes = plt.subplots(1, 3, figsize=(11, 2.8), sharex=True, sharey=True)
    for ax, (condition, title) in zip(axes, conditions):
        rows = table[table["condition"] == condition].set_index("strategy").reindex([s for s, _ in strategies])
        err = [rows["macro_auroc"] - rows["macro_auroc_lo"], rows["macro_auroc_hi"] - rows["macro_auroc"]]
        ax.errorbar(rows["macro_auroc"], y, xerr=err, fmt="o", color=SERIES[0], markersize=6, elinewidth=1.5)
        n = int(rows["n"].iloc[0])
        ax.set_title(f"{title} (n={n})", loc="left")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("macro-AUROC (95% CI)")
    axes[0].set_yticks(y, [label for _, label in strategies])
    _title(fig, "E2b: ECG + age, sex, height, weight")
    _save(fig, "e2b_metadata.png")


def e3_subgroups() -> None:
    table = pd.read_csv(RESULTS / "e3_subgroups.csv")
    models = [("e1_all12", "12 leads"), ("e2a_dropout_leadI", "lead I only (lead-dropout model)")]
    groups = list(table.query("model == 'e1_all12'")["group"])
    y = np.arange(len(groups))[::-1]

    fig, (ax_auc, ax_ece) = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for k, (model, label) in enumerate(models):
        rows = table[table["model"] == model].set_index("group").reindex(groups)
        offset = 0.12 if k == 0 else -0.12
        err = [rows["macro_auroc"] - rows["macro_auroc_lo"], rows["macro_auroc_hi"] - rows["macro_auroc"]]
        ax_auc.errorbar(rows["macro_auroc"], y + offset, xerr=err, fmt="o", color=SERIES[k], markersize=6,
                        elinewidth=1.5, label=label)
        ax_ece.plot(rows["ece"], y + offset, "o", color=SERIES[k], markersize=6, label=label)
    n = table.query("model == 'e1_all12'").set_index("group")["n"]
    ax_auc.set_yticks(y, [f"{g} (n={n[g]})" for g in groups])
    ax_auc.axhline(len(groups) - 2.5, color=AXIS, linewidth=1)  # sex above, age below
    ax_ece.axhline(len(groups) - 2.5, color=AXIS, linewidth=1)
    ax_auc.set(xlabel="macro-AUROC (95% CI)", title="Discrimination")
    ax_ece.set(xlabel="ECE (lower is better)", xlim=(0, None), title="Calibration")
    for ax in (ax_auc, ax_ece):
        ax.grid(axis="y", visible=False)
    _title(fig, "E3: test performance by sex and age", legend_ax=ax_auc)
    _save(fig, "e3_subgroups.png")


def e4_detection() -> None:
    table = pd.read_csv(RESULTS / "e4_detection.csv")
    shifts = [("age_80plus", "Aged 80+"), ("female", "Women"), ("site0", "Site 0")]
    detectors = [("mmd_features", "-", "features (MMD)"), ("ks_outputs", "--", "outputs (KS)")]
    sizes = sorted(table["batch_size"].unique())
    null = table[table["shift"] == "none"]

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), sharey=True)
    for ax, (shift, title) in zip(axes, shifts):
        rows = table[table["shift"] == shift]
        for k, lam in enumerate(sorted(rows["lam"].unique())):
            for detector, style, det_label in detectors:
                line = rows[(rows["lam"] == lam) & (rows["detector"] == detector)]
                label = f"{lam:.0%} of batch from subgroup" if detector == "mmd_features" else None
                ax.plot(line["batch_size"], line["detection_rate"], style, color=SERIES[k], marker="o",
                        markersize=3.5, label=label)
        false_alarm = null.groupby("batch_size")["detection_rate"].max()
        ax.plot(false_alarm.index, false_alarm.values, ":", color=MUTED, label="no shift (false alarms)")
        harm = rows[(rows["lam"] == 1.0)]["brier_change"].mean()
        ax.set_title(f"{title}\nBrier change at 100%: {harm:+.3f}", loc="left")
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.minorticks_off()
        ax.set(xlabel="ECGs per deployment batch", ylim=(-0.03, 1.03))
    axes[0].set_ylabel("share of batches flagged")
    axes[0].plot([], [], "-", color=INK_2, label="solid: features (MMD)")
    axes[0].plot([], [], "--", color=INK_2, label="dashed: outputs (KS)")
    # Legend below the panels: seven entries do not fit in one row beside the title.
    fig.suptitle("E4: share of deployment batches flagged as shifted (alpha = 0.05)", x=0.01, y=1.08,
                 ha="left", color=INK, fontsize=12)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    _save(fig, "e4_detection.png")


def e5_labelfree() -> None:
    table = pd.read_csv(RESULTS / "e5_labelfree.csv")
    groups = [("e1", "o", "12-lead model, subgroups"), ("e2a_dropout", "s", "lead-dropout model, leads removed"),
              ("e2a_naive", "^", "12-lead model, leads removed")]
    held_out = RESULTS / "e6_labelfree.csv"  # E6 adds the models that never saw the site they are tested on
    if held_out.exists():
        table = pd.concat([table, pd.read_csv(held_out)])
        groups.append(("e6_loso", "D", "site left out of training (E6)"))
    methods = [("ac_acc", "AC"), ("atc_acc", "ATC"), ("doc_acc", "DoC")]

    fig, (ax_all, ax_zoom, ax_brier) = plt.subplots(1, 3, figsize=(12, 3.9))
    for model, marker, model_label in groups:
        rows = table[table["model"] == model]
        for k, (column, method) in enumerate(methods):
            # DoC is hollow: it often lands exactly on AC, which would otherwise be hidden.
            style = dict(color=SERIES[k], markersize=6 if method == "DoC" else 5,
                         markerfacecolor="none" if method == "DoC" else SERIES[k])
            ax_all.plot(rows["true_acc"], rows[column], marker, label=method if model == "e1" else None, **style)
            if model != "e2a_naive":
                ax_zoom.plot(rows["true_acc"], rows[column], marker, **style)
        ax_brier.plot(rows["true_brier"], rows["ac_brier"], marker, color=SERIES[0], markersize=5)
        ax_all.plot([], [], marker, color=MUTED, label=model_label)
    calibrated = table[table["model"] != "e2a_naive"]
    ax_zoom.errorbar(calibrated["true_acc"], calibrated["true_acc"], xerr=[
        calibrated["true_acc"] - calibrated["true_acc_lo"], calibrated["true_acc_hi"] - calibrated["true_acc"]],
        fmt="none", ecolor=AXIS, elinewidth=1)

    for ax, lims in [(ax_all, (0.45, 0.97)), (ax_zoom, (0.82, 0.95)), (ax_brier, (0.04, 0.27))]:
        ax.plot(lims, lims, color=MUTED, linewidth=1, zorder=0)
        ax.set(xlim=lims, ylim=lims)
    ax_all.set(xlabel="true accuracy", ylabel="estimated accuracy", title="Accuracy, all scenarios")
    ax_zoom.set(xlabel="true accuracy (grey: 95% CI)", title="Accuracy, calibrated models only")
    ax_brier.set(xlabel="true Brier score", ylabel="estimated Brier score (AC)", title="Brier score")
    fig.suptitle("E5: performance estimated without labels vs the true value (test fold)", x=0.01, y=1.02,
                 ha="left", color=INK, fontsize=12)
    handles, labels = ax_all.get_legend_handles_labels()
    fig.tight_layout()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4)
    _save(fig, "e5_labelfree.png")


def e6_sites() -> None:
    sites = pd.read_csv(RESULTS / "e6_sites.csv").query("eval_set == 'site'")
    diffs = pd.read_csv(RESULTS / "e6_diffs.csv").query("contrast == 'loso - control'")
    detection = pd.read_csv(RESULTS / "e6_detection.csv")
    models = [("e1", "E1 (all training data)"), ("control", "size-matched control"),
              ("loso", "site left out of training")]
    site_ids = sorted(sites["site"].unique())
    y = np.arange(len(site_ids))[::-1]

    fig, (ax_auc, ax_brier, ax_det) = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"wspace": 0.3})
    for k, (model, label) in enumerate(models):
        rows = sites[sites["model"] == model].set_index("site").reindex(site_ids)
        offset = 0.2 * (1 - k)
        for ax, metric in [(ax_auc, "macro_auroc"), (ax_brier, "brier")]:
            err = [rows[metric] - rows[f"{metric}_lo"], rows[f"{metric}_hi"] - rows[metric]]
            ax.errorbar(rows[metric], y + offset, xerr=err, fmt="o", color=SERIES[k], markersize=5.5,
                        elinewidth=1.5, label=label)
    n = sites.query("model == 'e1'").set_index("site")["n"]
    for ax, metric, title in [(ax_auc, "macro_auroc", "Discrimination on the held-out site"),
                              (ax_brier, "brier", "Calibration on the held-out site")]:
        d = diffs[diffs["metric"] == metric].set_index("site")
        for s, row_y in zip(site_ids, y):
            # Left out minus control, written above each site's three intervals.
            ax.text(0.01, row_y + 0.38, f"left out − control: {d.loc[s, 'diff']:+.3f} "
                    f"[{d.loc[s, 'lo']:+.3f}, {d.loc[s, 'hi']:+.3f}]", transform=ax.get_yaxis_transform(),
                    color=INK_2, fontsize=8.5)
        ax.set_yticks(y, [f"site {s}\n(n={n[s]})" for s in site_ids])
        ax.set_ylim(y.min() - 0.45, y.max() + 0.6)
        ax.grid(axis="y", visible=False)
        ax.set_title(title, loc="left")
    ax_auc.set(xlabel="macro-AUROC (95% CI)")
    ax_brier.set(xlabel="Brier score (95% CI, lower is better)")

    for k, s in enumerate(site_ids):
        rows = detection[(detection["site"] == s) & (detection["shift"] == "held-out site")]
        for detector, style in [("mmd_features", "-"), ("ks_outputs", "--")]:
            line = rows[rows["detector"] == detector]
            ax_det.plot(line["batch_size"], line["detection_rate"], style, marker="o", markersize=3.5,
                        color=[INK, INK_2, MUTED][k], label=f"site {s}" if detector == "mmd_features" else None)
    false_alarm = detection[detection["shift"] != "held-out site"].groupby("batch_size")["detection_rate"].max()
    ax_det.plot(false_alarm.index, false_alarm.values, ":", color=AXIS, label="no shift (other sites)")
    ax_det.plot([], [], "-", color=INK_2, label="solid: features (MMD)")
    ax_det.plot([], [], "--", color=INK_2, label="dashed: outputs (KS)")
    sizes = sorted(detection["batch_size"].unique())
    ax_det.set_xscale("log")
    ax_det.set_xticks(sizes, [str(s) for s in sizes])
    ax_det.minorticks_off()
    ax_det.set(xlabel="ECGs per deployment batch", ylabel="share of batches flagged", ylim=(-0.03, 1.03))
    ax_det.set_title("Is the new site detected? (E4 tests)", loc="left")
    ax_det.legend(loc="center right", fontsize=8)

    fig.suptitle("E6: a hospital site the model never saw (test fold, records of the held-out site)",
                 x=0.01, y=1.04, ha="left", color=INK, fontsize=12)
    handles, labels = ax_auc.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=3)
    _save(fig, "e6_sites.png")


def e7_gaps() -> None:
    table = pd.read_csv(RESULTS / "e7_gaps.csv")
    models = [("e2a_dropout", "lead dropout, no gap training (E2a)"), ("e7_gaps_zero", "gap training, zero-filled"),
              ("e7_gaps_channel", "gap training + presence channel")]
    patterns = [("one", "one gap"), ("many", "many 0.5 s gaps")]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True, sharey="row")
    for row, (leads, leads_label) in enumerate([("all12", "12 leads"), ("leadI", "lead I")]):
        for col, (pattern, pattern_label) in enumerate(patterns):
            ax = axes[row, col]
            # Fraction 0 is evaluated once ("none") and starts both patterns' curves.
            rows = table[(table["leads"] == leads) & table["pattern"].isin(["none", pattern])].sort_values("fraction")
            if leads == "all12":
                e1 = rows[rows["model"] == "e1"]
                ax.plot(e1["fraction"], e1["macro_auroc"], color=MUTED, linewidth=1.5, marker="o", markersize=3,
                        label="E1, 12-lead model (no gap training)")
            for k, (model, label) in enumerate(models):
                line = rows[rows["model"] == model]
                ax.plot(line["fraction"], line["macro_auroc"], color=SERIES[k], marker="o", markersize=3.5,
                        label=label)
            ax.set_title(f"{leads_label}, {pattern_label}", loc="left")
            ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
            if row == 1:
                ax.set_xlabel("share of the recording missing")
            if col == 0:
                ax.set_ylabel("macro-AUROC")
    # No CI bands: the models are scored on the same records, so their separate intervals overlap far more
    # than the paired differences do; those are tested in results/e7_diffs.csv and the README table.
    fig.suptitle("E7: gaps in time on the test fold (all leads drop out together)", x=0.01, y=1.0, ha="left",
                 color=INK, fontsize=12)
    fig.tight_layout()
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
    _save(fig, "e7_gaps.png")


def e8_mitigation() -> None:
    gaps = pd.read_csv(RESULTS / "e8_gaps.csv")
    variants = [("e1", "E1 as is"), ("e1_platt_global", "Platt, global"), ("e1_platt_sex", "Platt, per sex"),
                ("e1_platt_age", "Platt, per age group"), ("reweighted", "reweighted training")]
    panels = [("male", "female", "Men − women"), (">=80", "60-79", "Aged 80+ − aged 60-79")]
    metrics = [("macro_auroc", "macro-AUROC gap"), ("brier", "Brier score gap")]
    y = np.arange(len(variants))[::-1]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, (group, versus, title) in zip(axes, panels):
        for k, (metric, label) in enumerate(metrics):
            rows = gaps[(gaps["group"] == group) & (gaps["versus"] == versus) & (gaps["metric"] == metric)]
            rows = rows.set_index("variant").reindex([v for v, _ in variants])
            err = [rows["gap"] - rows["gap_lo"], rows["gap_hi"] - rows["gap"]]
            ax.errorbar(rows["gap"], y + (0.12 if k == 0 else -0.12), xerr=err, fmt="o", color=SERIES[k],
                        markersize=5.5, elinewidth=1.5, label=label)
        ax.axvline(0, color=MUTED, linewidth=1, zorder=0)
        ax.set_title(title, loc="left")
        ax.set_xlabel("gap between the groups (95% CI)\nAUROC gap > 0 or Brier gap < 0: first group better")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y, [label for _, label in variants])
    _title(fig, "E8: subgroup gaps after recalibration or reweighting (test fold)", legend_ax=axes[0])
    _save(fig, "e8_mitigation.png")


def main() -> None:
    for make in [e1_reliability, e2a_leads, e2a_risk_coverage, e2b_metadata, e3_subgroups, e4_detection,
                 e5_labelfree, e6_sites, e7_gaps, e8_mitigation]:
        make()
        print("wrote", make.__name__)


if __name__ == "__main__":
    main()
