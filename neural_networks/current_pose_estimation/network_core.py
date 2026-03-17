"""
network_core.py — Shared building blocks for all pose-estimation networks.

Provides:
    set_seeds         – reproducible RNG seeding
    build_model       – construct nn.Sequential MLP from config
    build_optimizer   – construct optimizer from config
    train_pipeline    – full pipeline: split → normalise → build → train → predict
"""

import random
import numpy as np
import torch
import torch.nn as nn


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
#  Model builder                                                      #
# ------------------------------------------------------------------ #

_ACTIVATIONS = {
    "relu":       nn.ReLU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
    "tanh":       nn.Tanh,
    "gelu":       nn.GELU,
    "elu":        nn.ELU,
}


def build_model(cfg: dict, in_dim: int, out_dim: int) -> nn.Sequential:
    """
    Build a feed-forward MLP from *cfg*.

    Parameters
    ----------
    cfg : dict
        Must contain ``layers`` (list[int]) and ``activation`` (str).
    in_dim : int
        Dimensionality of the input features.
    out_dim : int
        Dimensionality of the output.
    """
    hidden_sizes = cfg["layers"]
    act_name = cfg["activation"].lower()

    if act_name not in _ACTIVATIONS:
        raise ValueError(
            f"Unsupported activation '{cfg['activation']}'. "
            f"Choose from: {list(_ACTIVATIONS)}"
        )

    layers: list[nn.Module] = []
    cur_dim = in_dim

    for h in hidden_sizes:
        layers.append(nn.Linear(cur_dim, h))
        act_fn = _ACTIVATIONS[act_name]
        layers.append(act_fn() if callable(act_fn) else act_fn)
        cur_dim = h

    layers.append(nn.Linear(cur_dim, out_dim))
    return nn.Sequential(*layers)


# ------------------------------------------------------------------ #
#  Optimizer builder                                                  #
# ------------------------------------------------------------------ #

def build_optimizer(cfg: dict, model: nn.Module) -> torch.optim.Optimizer:
    """
    Create an optimizer from *cfg*.

    Parameters
    ----------
    cfg : dict
        Must contain ``optimizer``, ``learning_rate``, and ``weight_decay``.
    model : nn.Module
        The model whose parameters will be optimized.
    """
    opt_name = cfg["optimizer"].lower()
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    if opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)

    raise ValueError(f"Unsupported optimizer '{cfg['optimizer']}'. Choose from: adam, adamw, sgd")


# ------------------------------------------------------------------ #
#  Train / val split                                                  #
# ------------------------------------------------------------------ #

def train_val_split(n: int, cfg: dict):
    """
    Sequential train/val split with an optional gap (val_padding).

    Returns
    -------
    idx_tr, idx_val : np.ndarray
        Integer index arrays for training and validation rows.
    """
    idx = np.arange(n)
    val_split = float(cfg.get("val_split", 0.2))
    split_idx = int(n * (1.0 - val_split))
    pad = int(cfg.get("val_padding", 0))

    train_end = max(0, split_idx - pad)
    val_start = min(n, split_idx + pad)

    return idx[:train_end], idx[val_start:]


# ------------------------------------------------------------------ #
#  Normalization                                                      #
# ------------------------------------------------------------------ #

def compute_normalization(Xtr: np.ndarray, Ytr: np.ndarray):
    """
    Compute mean/std from the *training* set only.

    Returns
    -------
    dict with keys X_mean, X_std, Y_mean, Y_std  (each shape (1, D)).
    """
    return {
        "X_mean": Xtr.mean(axis=0, keepdims=True),
        "X_std":  Xtr.std(axis=0, keepdims=True) + 1e-8,
        "Y_mean": Ytr.mean(axis=0, keepdims=True),
        "Y_std":  Ytr.std(axis=0, keepdims=True) + 1e-8,
    }


def apply_normalization(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Z-score normalise: (X - mean) / std."""
    return (X - mean) / std


# ------------------------------------------------------------------ #
#  Training loop                                                      #
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
):
    """
    Run the training loop for *cfg['epochs']* epochs.

    Returns
    -------
    train_losses, val_losses : list[float]
        Per-epoch loss values (normalised MSE).
    """
    loss_fn = nn.MSELoss()

    use_minibatch = cfg["batch_size"] and cfg["batch_size"] > 0
    if use_minibatch:
        ds_tr = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
        dl_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=cfg["batch_size"], shuffle=True, generator=generator,
        )

    train_losses: list[float] = []
    val_losses:   list[float] = []

    for epoch in range(1, cfg["epochs"] + 1):
        # --- forward / backward ---
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

        # --- evaluate ---
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
#  Full pipeline: split → normalise → build → train → predict        #
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
):
    """
    Generic training pipeline shared by every pose-estimation network.

    Parameters
    ----------
    X_all : np.ndarray, shape (N, D_in)
        Input features (already aligned and filtered).
    Y_all : np.ndarray, shape (N, D_out)
        Target values (already aligned and filtered).
    cfg : dict
        Full training config (layers, activation, optimizer, lr, …).
    in_dim : int, optional
        Override input dimensionality (defaults to X_all.shape[1]).
    out_dim : int, optional
        Override output dimensionality (defaults to Y_all.shape[1]).
    t_all : np.ndarray, optional
        Timestamps array (length N). Filtered in-place alongside X/Y.
    extra_arrays : dict[str, np.ndarray], optional
        Additional arrays (length N each) that should be filtered/split
        the same way (e.g. ``pred_rel_xyz_all``).

    Returns
    -------
    dict  — see inline comments for every key.
    """
    if extra_arrays is None:
        extra_arrays = {}

    # --- Infer dimensions ---
    in_dim  = in_dim  or X_all.shape[1]
    out_dim = out_dim or Y_all.shape[1]

    # --- Seed ---
    seed = int(cfg.get("seed", 42))
    g = set_seeds(seed)

    # --- Filter NaN / inf ---
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all = X_all[ok]
    Y_all = Y_all[ok]
    if t_all is not None:
        t_all = t_all[ok]
    extra_arrays = {k: v[ok] for k, v in extra_arrays.items()}

    # --- Train / val split ---
    idx_tr, idx_val = train_val_split(len(X_all), cfg)
    Xtr, Ytr = X_all[idx_tr], Y_all[idx_tr]
    Xval, Yval = X_all[idx_val], Y_all[idx_val]

    # --- Normalization (training set only) ---
    norm = compute_normalization(Xtr, Ytr)
    X_mean, X_std = norm["X_mean"], norm["X_std"]
    Y_mean, Y_std = norm["Y_mean"], norm["Y_std"]

    Xtr_n   = apply_normalization(Xtr,   X_mean, X_std)
    Xval_n  = apply_normalization(Xval,  X_mean, X_std)
    X_all_n = apply_normalization(X_all, X_mean, X_std)

    Ytr_n  = apply_normalization(Ytr,  Y_mean, Y_std)
    Yval_n = apply_normalization(Yval, Y_mean, Y_std)

    # --- Convert to tensors ---
    Xtr_t  = torch.from_numpy(Xtr_n).float()
    Ytr_t  = torch.from_numpy(Ytr_n).float()
    Xval_t = torch.from_numpy(Xval_n).float()
    Yval_t = torch.from_numpy(Yval_n).float()

    # --- Build model & optimizer ---
    model = build_model(cfg, in_dim, out_dim)
    opt   = build_optimizer(cfg, model)

    # --- Train ---
    train_losses, val_losses = _run_training_loop(
        model, opt, Xtr_t, Ytr_t, Xval_t, Yval_t, cfg, g,
    )

    # --- Predict on full set (denormalized) ---
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_all_n).float()).numpy()
    pred_all = pred_norm * Y_std + Y_mean

    # --- Masks ---
    train_mask = np.zeros(len(X_all), dtype=bool)
    val_mask   = np.zeros(len(X_all), dtype=bool)
    train_mask[idx_tr]  = True
    val_mask[idx_val]   = True

    # --- Assemble artifacts ---
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

        "train_losses":        train_losses,
        "val_losses":          val_losses,
        "final_train_loss":    float(train_losses[-1]),
        "final_val_loss":      float(val_losses[-1]),

        "pred_all":    pred_all,          # denormalized network predictions (N, out_dim)

        "norm_stats": norm,
    }

    # Include any extra arrays that were passed through
    for k, v in extra_arrays.items():
        artifacts[k] = v

    return artifacts
