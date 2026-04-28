"""
training.py — Training pipeline: split → normalise → build → train → predict.

Public API:
    train_pipeline(X_all, Y_all, cfg, ...)  → artifacts dict
"""

import random

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data

from models.mlp import build_model, build_optimizer


# ------------------------------------------------------------------ #
#  Seeding                                                            #
# ------------------------------------------------------------------ #

def set_seeds(seed: int = 42) -> torch.Generator:
    """Set Python, NumPy, and PyTorch seeds; return a torch Generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ------------------------------------------------------------------ #
#  Train / val split                                                  #
# ------------------------------------------------------------------ #

def train_val_split(n: int, cfg: dict):
    """
    Train/val split — sequential (default) or random.

    In **sequential** mode the first (1 − val_split) rows go to train and
    the last val_split rows go to val, with an optional gap (val_padding).

    In **random** mode rows are shuffled (using ``cfg["seed"]``) before
    the split.  ``val_padding`` is ignored (a warning is printed if > 0).

    Returns
    -------
    idx_tr, idx_val : np.ndarray
        Integer index arrays for training and validation rows.
    """
    mode = cfg.get("split_mode", "sequential")
    val_split = float(cfg.get("val_split", 0.2))
    pad = int(cfg.get("val_padding", 0))
    seed = int(cfg.get("seed", 42))

    idx = np.arange(n)

    if mode == "random":
        if pad > 0:
            import warnings
            warnings.warn(
                "val_padding is ignored when split_mode='random'"
            )
        rng = np.random.RandomState(seed)
        rng.shuffle(idx)
        split_idx = int(n * (1.0 - val_split))
        return idx[:split_idx], idx[split_idx:]

    # sequential (default)
    split_idx = int(n * (1.0 - val_split))
    train_end = max(0, split_idx - pad)
    val_start = min(n, split_idx + pad)
    return idx[:train_end], idx[val_start:]


# ------------------------------------------------------------------ #
#  Normalization                                                      #
# ------------------------------------------------------------------ #

def compute_normalization(Xtr: np.ndarray, Ytr: np.ndarray) -> dict:
    """
    Compute mean/std from the *training* set only.

    Returns dict with keys X_mean, X_std, Y_mean, Y_std (each shape (1, D)).
    """
    return {
        "X_mean": Xtr.mean(axis=0, keepdims=True),
        "X_std":  Xtr.std(axis=0, keepdims=True) + 1e-8,
        "Y_mean": Ytr.mean(axis=0, keepdims=True),
        "Y_std":  Ytr.std(axis=0, keepdims=True) + 1e-8,
    }


def apply_normalization(
    X: np.ndarray, mean: np.ndarray, std: np.ndarray,
) -> np.ndarray:
    """Z-score normalise: (X - mean) / std."""
    return (X - mean) / std


# ------------------------------------------------------------------ #
#  Training loop (private)                                            #
# ------------------------------------------------------------------ #

def _run_training_loop(
    model: nn.Module,
    opt: torch.optim.Optimizer,
    Xtr_t: torch.Tensor,
    Ytr_t: torch.Tensor,
    Xval_t: torch.Tensor,
    Yval_t: torch.Tensor,
    cfg: dict,
    generator: torch.Generator,
) -> tuple[list[float], list[float]]:
    """
    Train for ``cfg['epochs']`` epochs.

    Returns per-epoch (train_losses, val_losses) in normalised space.
    """
    loss_fn = nn.MSELoss()

    use_minibatch = cfg.get("batch_size") and cfg["batch_size"] > 0
    if use_minibatch:
        ds_tr = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
        dl_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
        )

    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_minibatch:
            for xb, yb in dl_tr:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
        else:
            pred = model(Xtr_t)
            loss = loss_fn(pred, Ytr_t)
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            train_loss = loss_fn(model(Xtr_t), Ytr_t).item()
            val_loss   = loss_fn(model(Xval_t), Yval_t).item()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg["epochs"]:
            print(f"Epoch {epoch:3d}: train {train_loss:.4f}, val {val_loss:.4f}")

    return train_losses, val_losses


# ------------------------------------------------------------------ #
#  Full pipeline                                                      #
# ------------------------------------------------------------------ #

def train_pipeline(
    X_all: np.ndarray,
    Y_all: np.ndarray,
    cfg: dict,
    *,
    in_dim: int | None = None,
    out_dim: int | None = None,
    t_all: np.ndarray | None = None,
    extra_arrays: dict[str, np.ndarray] | None = None,
    uvdar_baseline: np.ndarray | None = None,
) -> dict:
    """
    Full training pipeline shared by every pose-estimation variant.

    Parameters
    ----------
    X_all : np.ndarray, shape (N, D_in)
    Y_all : np.ndarray, shape (N, D_out)
    cfg : dict
        Full config (layers, activation, optimizer, lr, …).
    t_all : np.ndarray, optional
        Timestamps (length N); filtered alongside X/Y.
    extra_arrays : dict[str, np.ndarray], optional
        Additional arrays that are filtered/split identically.

    Returns
    -------
    dict — model, predictions, losses, normalization stats, masks, etc.
    """
    if extra_arrays is None:
        extra_arrays = {}

    in_dim  = in_dim  or X_all.shape[1]
    out_dim = out_dim or Y_all.shape[1]

    seed = int(cfg.get("seed", 42))
    g = set_seeds(seed)

    # Filter NaN / inf
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all = X_all[ok]
    Y_all = Y_all[ok]
    if t_all is not None:
        t_all = t_all[ok]
    extra_arrays = {k: v[ok] for k, v in extra_arrays.items()}
    if uvdar_baseline is not None:
        uvdar_baseline = uvdar_baseline[ok]

    # Residual learning: train on (Y − UVDAR baseline) instead of Y
    residual = cfg.get("residual_learning", False)
    if residual:
        if uvdar_baseline is None:
            raise ValueError(
                "residual_learning requires a UVDAR baseline "
                "(enable features.uvdar with a position component)"
            )
        Y_target = Y_all - uvdar_baseline
        print("[residual_learning] Training on residual (Y − UVDAR baseline)")
    else:
        Y_target = Y_all

    # Split
    idx_tr, idx_val = train_val_split(len(X_all), cfg)
    Xtr, Ytr = X_all[idx_tr], Y_target[idx_tr]
    Xval, Yval = X_all[idx_val], Y_target[idx_val]

    # Normalise (train stats only)
    norm = compute_normalization(Xtr, Ytr)
    X_mean, X_std = norm["X_mean"], norm["X_std"]
    Y_mean, Y_std = norm["Y_mean"], norm["Y_std"]

    Xtr_n   = apply_normalization(Xtr,   X_mean, X_std)
    Xval_n  = apply_normalization(Xval,  X_mean, X_std)
    X_all_n = apply_normalization(X_all, X_mean, X_std)

    Ytr_n  = apply_normalization(Ytr,  Y_mean, Y_std)
    Yval_n = apply_normalization(Yval, Y_mean, Y_std)

    # Tensors
    Xtr_t  = torch.from_numpy(Xtr_n).float()
    Ytr_t  = torch.from_numpy(Ytr_n).float()
    Xval_t = torch.from_numpy(Xval_n).float()
    Yval_t = torch.from_numpy(Yval_n).float()

    # Build
    model = build_model(cfg, in_dim, out_dim)
    opt   = build_optimizer(cfg, model)

    # Train
    train_losses, val_losses = _run_training_loop(
        model, opt, Xtr_t, Ytr_t, Xval_t, Yval_t, cfg, g,
    )

    # Predict (denormalised)
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_all_n).float()).numpy()
    pred_target = pred_norm * Y_std + Y_mean

    # Residual learning: add UVDAR baseline back for final predictions
    if residual:
        pred_all = pred_target + uvdar_baseline
    else:
        pred_all = pred_target

    # Masks
    train_mask = np.zeros(len(X_all), dtype=bool)
    val_mask   = np.zeros(len(X_all), dtype=bool)
    train_mask[idx_tr] = True
    val_mask[idx_val]  = True

    artifacts = {
        "model":       model,
        "X_all":       X_all,
        "Y_all":       Y_all,
        "X_all_n":     X_all_n,
        "t_all":       t_all,

        "idx_tr":      idx_tr,
        "idx_val":     idx_val,
        "train_mask":  train_mask,
        "val_mask":    val_mask,

        "train_losses":     train_losses,
        "val_losses":       val_losses,
        "final_train_loss": float(train_losses[-1]),
        "final_val_loss":   float(val_losses[-1]),

        "pred_all":    pred_all,
        "norm_stats":  norm,
    }

    for k, v in extra_arrays.items():
        artifacts[k] = v
    if uvdar_baseline is not None:
        artifacts["uvdar_baseline"] = uvdar_baseline

    return artifacts
