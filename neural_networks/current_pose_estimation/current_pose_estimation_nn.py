import torch
import os
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split


import helpers


# ---------- core ML logic (no plotting, no saving) ----------

def train_model_core(cfg, run_dir: str):
    """
    Core training: load data, train model, compute predictions.
    NO filesystem / plotting here.

    Returns:
        dict with model, losses, arrays, normalization, etc.
    """
    print("Using config:", cfg)

    # Load data
    est = helpers.load_xyz(os.path.join(run_dir, "estimations.csv"))
    od1 = helpers.load_xyz(os.path.join(run_dir, "odom1.csv"))
    od2 = helpers.load_xyz(os.path.join(run_dir, "odom2.csv"))

    # Align all on UVDAR timeline
    est_xyz_all, od1_xyz_all, od2_xyz_all, t_all = helpers.align_three_streams_on_uvdar(est, od1, od2)

    # Targets & inputs on EXACT same timestamps
    Y_all = od2_xyz_all - od1_xyz_all   # residual (odom2 - odom1)
    X_all = est_xyz_all                 # UVDAR [x, y, z]

    # Filter out any NaNs / infs
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all, est_xyz_all, t_all = X_all[ok], Y_all[ok], est_xyz_all[ok], t_all[ok]

    # Train/val split
    idx = np.arange(len(X_all))
    idx_tr, idx_val, Xtr, Xval, Ytr, Yval = train_test_split(
        idx, X_all, Y_all, test_size=cfg["val_split"], random_state=42
    )

    # --- Normalization (computed ONLY from training set) ---
    X_mean = Xtr.mean(axis=0, keepdims=True)
    X_std  = Xtr.std(axis=0, keepdims=True) + 1e-8
    Y_mean = Ytr.mean(axis=0, keepdims=True)
    Y_std  = Ytr.std(axis=0, keepdims=True) + 1e-8

    Xtr_n = (Xtr - X_mean) / X_std
    Xval_n = (Xval - X_mean) / X_std
    X_all_n = (X_all - X_mean) / X_std

    Ytr_n = (Ytr - Y_mean) / Y_std
    Yval_n = (Yval - Y_mean) / Y_std

    # --- Build model dynamically from config['layers'] ---
    hidden_sizes = cfg["layers"]  # e.g. [256, 256, 128]

    layers = []
    in_dim = 3  # input is [x, y, z]

    act_name = cfg["activation"].lower()

    for h in hidden_sizes:
        layers.append(nn.Linear(in_dim, h))

        if act_name == "relu":
            layers.append(nn.ReLU())
        elif act_name == "leaky_relu":
            layers.append(nn.LeakyReLU(0.01))
        elif act_name == "tanh":
            layers.append(nn.Tanh())
        elif act_name == "gelu":
            layers.append(nn.GELU())
        elif act_name == "elu":
            layers.append(nn.ELU())
        else:
            raise ValueError(f"Unsupported activation: {cfg['activation']}")

        in_dim = h

    # output layer -> 3D residual
    layers.append(nn.Linear(in_dim, 3))

    model = nn.Sequential(*layers)

    # Optimizer
    opt_name = cfg["optimizer"].lower()
    if opt_name == "adamw":
        opt = torch.optim.AdamW(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"]
        )
    elif opt_name == "sgd":
        opt = torch.optim.SGD(
            model.parameters(),
            lr=cfg["learning_rate"],
            momentum=0.9,
            weight_decay=cfg["weight_decay"]
        )
    elif opt_name == "adam":
        opt = torch.optim.Adam(
            model.parameters(),
            lr=cfg["learning_rate"],
            weight_decay=cfg["weight_decay"]
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")

    loss_fn = nn.MSELoss()

    # --- Training ---
    Xtr_t = torch.from_numpy(Xtr_n).float()
    Ytr_t = torch.from_numpy(Ytr_n).float()
    Xval_t = torch.from_numpy(Xval_n).float()
    Yval_t = torch.from_numpy(Yval_n).float()

    use_minibatch = cfg["batch_size"] and cfg["batch_size"] > 0
    if use_minibatch:
        ds_tr = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
        dl_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=cfg["batch_size"], shuffle=True
        )

    final_train_loss = None
    final_val_loss = None

    train_losses = []
    val_losses = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_minibatch:
            ep_loss = 0.0
            for xb, yb in dl_tr:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += loss.item() * xb.size(0)
            train_loss = ep_loss / len(ds_tr)
        else:
            pred = model(Xtr_t)
            loss = loss_fn(pred, Ytr_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss = loss.item()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), Yval_t).item()

        final_train_loss = train_loss
        final_val_loss = val_loss

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg["epochs"]:
            print(f"Epoch {epoch:3d}: train {train_loss:.4f}, val {val_loss:.4f}")

    # Predict on full set (denormalized back to meters)
    model.eval()
    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_all_n).float()).numpy()
    pred_res_all = pred_norm * Y_std + Y_mean

    # Masks (in case you'd want to use them later)
    train_mask = np.zeros(len(X_all), dtype=bool)
    val_mask   = np.zeros(len(X_all), dtype=bool)
    train_mask[idx_tr] = True
    val_mask[idx_val] = True

    return {
        "model": model,                         # Trained PyTorch MLP model (nn.Sequential)

        "X_all": X_all,                         # Shape (N, 3)  Original UVDAR inputs [x,y,z] after filtering (meters)
        "Y_all": Y_all,                         # Shape (N, 3)  True residuals odom2 - odom1 (meters)
        "X_all_n": X_all_n,                     # Shape (N, 3)  Normalized inputs: (X_all - X_mean) / X_std

        "est_xyz_all": est_xyz_all,             # Shape (N, 3)  UVDAR estimates aligned to odometry timeline (meters)
        "t_all": t_all,                         # Shape (N,)    Timestamps after alignment & filtering (float or int)

        "idx_tr": idx_tr,                       # Shape (N_tr,) Integer indices of training samples
        "idx_val": idx_val,                     # Shape (N_val,) Integer indices of validation samples

        "train_mask": train_mask,               # Shape (N,)    Boolean mask: True = training sample
        "val_mask": val_mask,                   # Shape (N,)    Boolean mask: True = validation sample

        "train_losses": train_losses,           # Shape (epochs,)  Per-epoch training MSE (on NORMALIZED targets)
        "val_losses": val_losses,               # Shape (epochs,)  Per-epoch validation MSE (normalized)
        "final_train_loss": float(final_train_loss),  # Scalar      Last-epoch train MSE (normalized)
        "final_val_loss": float(final_val_loss),      # Scalar      Last-epoch val MSE (normalized)

        "pred_res_all": pred_res_all,           # Shape (N, 3)  Predicted residuals (meters), denormalized: Y_pred = pred_norm*Y_std + Y_mean

        "norm_stats": {                         # Normalization statistics (computed *from training set only*)
            "X_mean": X_mean,                   # Shape (1, 3)
            "X_std":  X_std,
            "Y_mean": Y_mean,
            "Y_std":  Y_std,
        }
    }
