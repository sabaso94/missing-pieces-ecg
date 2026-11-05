import numpy as np
import pandas as pd
import pytest

from ptbxl_reliable import gaps, train
from ptbxl_reliable.data import PreparedData
from ptbxl_reliable.evaluate import evaluate, load_preds, save_preds
from ptbxl_reliable.metrics import macro_auroc
from ptbxl_reliable.train import TrainConfig, predict, train_ensemble, train_model


@pytest.fixture
def synthetic():
    """96 records. Label 0 is learnable (sign of lead I's mean); the others are random."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=(96, 12, 1000)).astype(np.float32)
    shift = rng.choice([-1.0, 1.0], size=96).astype(np.float32)
    x[:, 0] += shift[:, None]
    y = rng.integers(0, 2, size=(96, 5)).astype(np.float32)
    y[:, 0] = shift > 0
    meta = pd.DataFrame({"patient_id": np.arange(96) // 2}, index=np.arange(1, 97))
    idx = {"train": np.arange(64), "val": np.arange(64, 80), "test": np.arange(80, 96)}
    return PreparedData(x=x, y=y, meta=meta, idx=idx)


def _cfg(**kwargs):
    return TrainConfig(device="cpu", max_epochs=3, batch_size=32, **kwargs)


def test_train_runs_and_selects_best(synthetic):
    model, history = train_model(_cfg(), synthetic)
    assert len(history) == 3 and "val_macro_auroc" in history.columns
    val = synthetic.idx["val"]
    p = predict(model, synthetic.x[val], "all12", device="cpu")
    assert macro_auroc(synthetic.y[val], p) == pytest.approx(history["val_macro_auroc"].max())


def test_dropout_mode_trains(synthetic):
    _, history = train_model(_cfg(lead_mode="dropout"), synthetic)
    assert len(history) == 3


def test_tab_dropout_trains(synthetic):
    tab = np.random.default_rng(0).normal(size=(96, 8)).astype(np.float32)
    cfg = _cfg(tab=tab, tab_dropout=0.3, tab_removed=np.zeros(8, dtype=np.float32))
    model, _ = train_model(cfg, synthetic)
    p = predict(model, synthetic.x[:5], "all12", tab=tab[:5], device="cpu")
    assert p.shape == (5, 5)


def test_predict_shape_and_range(synthetic):
    model, _ = train_model(_cfg(), synthetic)
    p = predict(model, synthetic.x[:10], "leadI", device="cpu")
    assert p.shape == (10, 5) and (p >= 0).all() and (p <= 1).all()


def test_ensemble_caches(synthetic, tmp_path, monkeypatch):
    models = train_ensemble(_cfg(), synthetic, seeds=[0, 1], ckpt_dir=tmp_path)
    assert len(models) == 2 and (tmp_path / "seed1.pt").exists()

    def fail(*args, **kwargs):
        raise AssertionError("should have loaded the checkpoint instead of training")

    monkeypatch.setattr(train, "train_model", fail)
    reloaded = train_ensemble(_cfg(), synthetic, seeds=[0, 1], ckpt_dir=tmp_path)
    x = synthetic.x[:4]
    np.testing.assert_allclose(
        predict(models[1], x, "all12", device="cpu"), predict(reloaded[1], x, "all12", device="cpu"), rtol=1e-5
    )


def test_evaluate_keys(synthetic, tmp_path):
    rng = np.random.default_rng(0)
    probs = rng.random((3, 96, 5))
    result = evaluate(synthetic.y, probs, synthetic.meta["patient_id"].to_numpy(), n_boot=50)
    expected = {"macro_auroc", "ece", "brier", "mean_entropy", "mean_mi"}
    expected |= {f"{m}_{b}" for m in ["macro_auroc", "brier"] for b in ["lo", "hi"]}
    expected |= {f"auroc_{c}" for c in ["NORM", "MI", "STTC", "CD", "HYP"]}
    assert expected <= set(result)
    for m in ["macro_auroc", "brier"]:
        assert result[f"{m}_lo"] <= result[m] <= result[f"{m}_hi"]
    # Binned ECE is biased upward under bootstrap resampling (duplicated records), so its
    # percentile interval need not contain the point estimate: report ECE without a CI.
    assert "ece_lo" not in result and "ece_hi" not in result

    save_preds(tmp_path / "p.npz", synthetic.meta.index.to_numpy(), probs)
    ids, loaded = load_preds(tmp_path / "p.npz")
    np.testing.assert_array_equal(ids, synthetic.meta.index.to_numpy())
    np.testing.assert_array_equal(loaded, probs)


def test_evaluate_per_class_ci(synthetic):
    rng = np.random.default_rng(0)
    probs = rng.random((3, 96, 5))
    patients = synthetic.meta["patient_id"].to_numpy()
    assert "auroc_NORM_lo" not in evaluate(synthetic.y, probs, patients, n_boot=20)
    result = evaluate(synthetic.y, probs, patients, n_boot=20, per_class_ci=True)
    for c in ["NORM", "MI", "STTC", "CD", "HYP"]:
        assert result[f"auroc_{c}_lo"] <= result[f"auroc_{c}"] <= result[f"auroc_{c}_hi"]


def test_predict_ensemble_stacks_members(synthetic):
    from ptbxl_reliable.train import predict_ensemble

    models = [train_model(_cfg(seed=s), synthetic)[0] for s in (0, 1)]
    probs = predict_ensemble(models, synthetic.x[:6], "limb6", device="cpu")
    assert probs.shape == (2, 6, 5)
    np.testing.assert_allclose(probs[1], predict(models[1], synthetic.x[:6], "limb6", device="cpu"))


def test_gap_augmented_training_runs(synthetic):
    model, history = train_model(_cfg(lead_mode="dropout", gap_augment=True, time_mask_channel=True), synthetic)
    tm = gaps.fixed_gap_masks(5, 1000, 0.3, "one", seed=0)
    assert predict(model, synthetic.x[:5], "leadI", time_mask=tm, device="cpu").shape == (5, 5)


def test_unit_sample_weights_match_unweighted(synthetic):
    _, h0 = train_model(_cfg(), synthetic)
    _, h1 = train_model(_cfg(sample_weight=np.ones(96, np.float32)), synthetic)
    # Same loss up to float32 rounding (per-record then batch mean vs one mean); the first epoch is exact to
    # 1e-5, later epochs drift slightly as rounding differences accumulate in the weights.
    np.testing.assert_allclose(h0["train_loss"].iloc[0], h1["train_loss"].iloc[0], rtol=1e-5)
    np.testing.assert_allclose(h0["train_loss"], h1["train_loss"], rtol=1e-2)


def test_sample_weights_change_training(synthetic):
    weights = np.where(np.arange(96) % 2 == 0, 1.9, 0.1).astype(np.float32)
    _, h0 = train_model(_cfg(), synthetic)
    _, h1 = train_model(_cfg(sample_weight=weights), synthetic)
    assert not np.allclose(h0["train_loss"], h1["train_loss"], rtol=1e-3)
