"""Training one model, training a 5-seed ensemble, and predicting probabilities."""

import copy
import random
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from ptbxl_reliable.data import PreparedData
from ptbxl_reliable.gaps import fixed_gap_masks, sample_training_gaps
from ptbxl_reliable.leads import LEAD_CONFIGS, config_mask, sample_training_masks
from ptbxl_reliable.metrics import macro_auroc
from ptbxl_reliable.model import ECGResNet


@dataclass
class TrainConfig:
    # "all12" or one reduced configuration (a specialist), or "dropout" (random configuration per example).
    lead_mode: str = "all12"
    tab: np.ndarray | None = None  # tabular features for every record (N, n_tab), or None for ECG only
    tab_dropout: float = 0.0  # probability of replacing a training record's metadata by `tab_removed`
    tab_removed: np.ndarray | None = None  # feature vector of a record whose metadata is all missing
    gap_augment: bool = False  # train with random gaps in time (E7)
    time_mask_channel: bool = False  # give the model an extra input channel marking where the signal is present
    sample_weight: np.ndarray | None = None  # loss weight for every record (N,), e.g. to balance groups (E8)
    lr: float = 1e-3
    weight_decay: float = 1e-2
    batch_size: int = 128
    max_epochs: int = 50
    patience: int = 10  # stop after this many epochs without a better validation score
    seed: int = 0
    device: str = "cuda"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _val_configs(lead_mode: str) -> list[str]:
    # A lead-dropout model has to work for every configuration, so it is selected on all of them.
    return list(LEAD_CONFIGS) if lead_mode == "dropout" else [lead_mode]


@torch.no_grad()
def predict(
    model: ECGResNet,
    x: np.ndarray,
    mask_name: str,
    tab: np.ndarray | None = None,
    batch_size: int = 512,
    device: str = "cuda",
    time_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Predicted probabilities (N, 5) with the leads of `mask_name` present and, if given, gaps where
    `time_mask` (N, T) is False."""
    model.eval().to(device)
    mask = torch.from_numpy(config_mask(mask_name)).to(device)
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        tb = None if tab is None else torch.from_numpy(tab[start : start + batch_size]).to(device)
        tm = None if time_mask is None else torch.from_numpy(time_mask[start : start + batch_size]).to(device)
        logits = model(xb, mask.expand(len(xb), -1), tb, tm)
        out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def extract_features(
    model: ECGResNet, x: np.ndarray, mask_name: str, batch_size: int = 512, device: str = "cuda"
) -> np.ndarray:
    """Pooled ECG features (N, 256) of one model, with the leads of `mask_name` present."""
    model.eval().to(device)
    mask = torch.from_numpy(config_mask(mask_name)).to(device)
    out = []
    for start in range(0, len(x), batch_size):
        xb = torch.from_numpy(x[start : start + batch_size]).to(device)
        out.append(model.features(xb, mask.expand(len(xb), -1)).cpu().numpy())
    return np.concatenate(out)


def predict_ensemble(
    models: list[ECGResNet],
    x: np.ndarray,
    mask_name: str,
    tab: np.ndarray | None = None,
    device: str = "cuda",
    time_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Every member's probabilities, stacked as (M, N, 5)."""
    return np.stack([predict(model, x, mask_name, tab, device=device, time_mask=time_mask) for model in models])


def train_model(cfg: TrainConfig, data: PreparedData) -> tuple[ECGResNet, pd.DataFrame]:
    """Train one model; return the weights from the epoch with the best validation macro-AUROC."""
    _seed_everything(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    n_tab = 0 if cfg.tab is None else cfg.tab.shape[1]
    model = ECGResNet(n_tab=n_tab, time_mask_channel=cfg.time_mask_channel).to(cfg.device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.max_epochs)
    loss_fn = nn.BCEWithLogitsLoss()  # multi-label: one independent yes/no per superclass
    weighted_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    train_idx, val_idx = data.idx["train"], data.idx["val"]
    fixed_mask = None if cfg.lead_mode == "dropout" else torch.from_numpy(config_mask(cfg.lead_mode))
    n_samples = data.x.shape[2]
    # A gap-trained model has to work with and without gaps, so it is selected on both.
    val_gaps = [None]
    if cfg.gap_augment:
        val_gaps.append(fixed_gap_masks(len(val_idx), n_samples, 0.3, "many", seed=0))

    best_score, best_state, epochs_without_gain, history = -np.inf, None, 0, []
    for epoch in range(cfg.max_epochs):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=generator).numpy()]
        total_loss = 0.0
        for start in range(0, len(order), cfg.batch_size):
            batch = order[start : start + cfg.batch_size]
            xb = torch.from_numpy(data.x[batch])
            yb = torch.from_numpy(data.y[batch])
            if fixed_mask is None:
                mask = sample_training_masks(len(batch), generator)
            else:
                mask = fixed_mask.expand(len(batch), -1)

            tb = None
            if cfg.tab is not None:
                tab = cfg.tab[batch].copy()
                if cfg.tab_dropout > 0:
                    # Modality dropout: pretend some records have no metadata at all.
                    drop = torch.rand(len(batch), generator=generator).numpy() < cfg.tab_dropout
                    tab[drop] = cfg.tab_removed
                tb = torch.from_numpy(tab).to(cfg.device)

            tm = None
            if cfg.gap_augment:
                tm = sample_training_gaps(len(batch), n_samples, generator).to(cfg.device)

            logits = model(xb.to(cfg.device), mask.to(cfg.device), tb, tm)
            if cfg.sample_weight is None:
                loss = loss_fn(logits, yb.to(cfg.device))
            else:
                # Each record's loss (averaged over classes) times its weight, averaged over the batch.
                wb = torch.from_numpy(cfg.sample_weight[batch]).to(cfg.device, logits.dtype)
                loss = (weighted_loss_fn(logits, yb.to(cfg.device)).mean(dim=1) * wb).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        scheduler.step()

        val_tab = None if cfg.tab is None else cfg.tab[val_idx]
        scores = [
            macro_auroc(
                data.y[val_idx],
                predict(model, data.x[val_idx], name, val_tab, device=cfg.device, time_mask=time_mask),
            )
            for name in _val_configs(cfg.lead_mode)
            for time_mask in val_gaps
        ]
        score = float(np.mean(scores))
        history.append({"epoch": epoch, "train_loss": total_loss / len(order), "val_macro_auroc": score})

        if score > best_score:
            best_score, best_state, epochs_without_gain = score, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= cfg.patience:
                break

    model.load_state_dict(best_state)
    return model.eval(), pd.DataFrame(history)


def train_ensemble(
    cfg: TrainConfig,
    data: PreparedData,
    seeds=range(5),
    ckpt_dir: Path = Path("checkpoints"),
) -> list[ECGResNet]:
    """One model per seed (a deep ensemble). Existing checkpoints are loaded instead of retrained."""
    ckpt_dir = Path(ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    n_tab = 0 if cfg.tab is None else cfg.tab.shape[1]
    models = []
    for seed in seeds:
        path = ckpt_dir / f"seed{seed}.pt"
        if path.exists():
            model = ECGResNet(n_tab=n_tab, time_mask_channel=cfg.time_mask_channel)
            model.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            model, history = train_model(replace(cfg, seed=seed), data)
            torch.save(model.state_dict(), path)
            history.to_csv(ckpt_dir / f"seed{seed}_history.csv", index=False)
        models.append(model.eval())
    return models
