#!/usr/bin/env python3
import argparse, os, json, shutil
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------- helpers ----------

def load_config():
    """Load key=value pairs from config.txt next to this script."""
    cfg_path = os.path.join(os.path.dirname(__file__), "config.txt")
    if not os.path.isfile(cfg_path):
        raise FileNotFoundError(f"config.txt not found next to the script: {cfg_path}")
    cfg = {}
    with open(cfg_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip()
    parsed = {
        "learning_rate": float(cfg.get("learning_rate", 1e-3)),
        "epochs": int(cfg.get("epochs", 100)),
        "hidden_layers": int(cfg.get("hidden_layers", 2)),
        "hidden_size": int(cfg.get("hidden_size", 128)),
        "batch_size": int(float(cfg.get("batch_size", 0))),
        "val_split": float(cfg.get("val_split", 0.2)),
    }
    return parsed, cfg_path

def load_xyz(path):
    """Load CSV with time,x,y,z and add integer milliseconds (t_ms) for exact matching."""
    df = pd.read_csv(path)[["time","x","y","z"]].dropna().sort_values("time")
    df["t_ms"] = (df["time"] * 1000).round().astype(np.int64)
    df = df.drop_duplicates(subset="t_ms")
    return df

def interpolate_on_index(target_idx, df):
    """Reindex df (with t_ms,x,y,z) onto target_idx (int ms)."""
    d = (df[["t_ms","x","y","z"]].set_index("t_ms").sort_index())
    out = (d.reindex(target_idx)
             .interpolate(method="index")
             .ffill()
             .bfill())
    out.index.name = "t_ms"
    return out[["x","y","z"]]

def align_three_streams_on_uvdar(est_df, od1_df, od2_df):
    """Build ONE common timeline = unique, sorted UVDAR t_ms."""
    ref_ms = np.unique(est_df["t_ms"].values)
    od1_al = interpolate_on_index(ref_ms, od1_df)
    od2_al = interpolate_on_index(ref_ms, od2_df)
    est_on_ref = (est_df.set_index("t_ms").loc[ref_ms])[["x","y","z"]]
    t_sec = ref_ms.astype(np.float64) / 1000.0
    return est_on_ref.values.astype(np.float32), \
           od1_al.values.astype(np.float32), \
           od2_al.values.astype(np.float32), \
           t_sec

def ensure_fresh_results_dir(name: str) -> str:
    """Create ./results/<name> if it does not exist; error if it does."""
    out = os.path.join(".", "results", name)
    if os.path.exists(out):
        raise FileExistsError(f"Results folder already exists: {out}")
    os.makedirs(out, exist_ok=False)
    return out

# ---------- main ----------

def main():
    cfg, cfg_path = load_config()
    print("Loaded config:", cfg)

    ap = argparse.ArgumentParser(
        description="UVDAR [x,y,z] -> residual (odom2-odom1) with strict time alignment."
    )
    ap.add_argument("run_dir", help="Folder containing estimations.csv, odom1.csv, odom2.csv")
    ap.add_argument("name", help="Name of the results run; creates ./results/<name>")
    args = ap.parse_args()

    # Make results dir (fail if exists)
    results_dir = ensure_fresh_results_dir(args.name)
    print(f"Results will be saved to: {results_dir}")

    # Save a copy of the current config and a resolved JSON snapshot
    shutil.copyfile(cfg_path, os.path.join(results_dir, "config.txt"))

    # Load data
    est = load_xyz(os.path.join(args.run_dir,"estimations.csv"))
    od1 = load_xyz(os.path.join(args.run_dir,"odom1.csv"))
    od2 = load_xyz(os.path.join(args.run_dir,"odom2.csv"))

    # Align all on UVDAR timeline
    est_xyz_all, od1_xyz_all, od2_xyz_all, t_all = align_three_streams_on_uvdar(est, od1, od2)

    # Targets & inputs on EXACT same timestamps
    Y_all = od2_xyz_all - od1_xyz_all
    X_all = est_xyz_all


    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all, est_xyz_all, t_all = X_all[ok], Y_all[ok], est_xyz_all[ok], t_all[ok]

    # Train/val split
    idx = np.arange(len(X_all))
    idx_tr, idx_val, Xtr, Xval, Ytr, Yval = train_test_split(
        idx, X_all, Y_all, test_size=cfg["val_split"], random_state=42
    )

    # --- Build model dynamically from config ---
    layers = [nn.Linear(3, cfg["hidden_size"]), nn.ReLU()]
    for _ in range(cfg["hidden_layers"] - 1):
        layers += [nn.Linear(cfg["hidden_size"], cfg["hidden_size"]), nn.ReLU()]
    layers += [nn.Linear(cfg["hidden_size"], 3)]
    model = nn.Sequential(*layers)

    opt = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    loss_fn = nn.MSELoss()

    # --- Training ---
    Xtr_t = torch.from_numpy(Xtr).float()
    Ytr_t = torch.from_numpy(Ytr).float()
    Xval_t = torch.from_numpy(Xval).float()
    Yval_t = torch.from_numpy(Yval).float()

    use_minibatch = cfg["batch_size"] and cfg["batch_size"] > 0
    if use_minibatch:
        ds_tr = torch.utils.data.TensorDataset(Xtr_t, Ytr_t)
        dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=cfg["batch_size"], shuffle=True)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        if use_minibatch:
            ep_loss = 0.0
            for xb, yb in dl_tr:
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad(); loss.backward(); opt.step()
                ep_loss += loss.item() * xb.size(0)
            train_loss = ep_loss / len(ds_tr)
        else:
            pred = model(Xtr_t)
            loss = loss_fn(pred, Ytr_t)
            opt.zero_grad(); loss.backward(); opt.step()
            train_loss = loss.item()

        if epoch % 10 == 0 or epoch == 1 or epoch == cfg["epochs"]:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xval_t), Yval_t).item()
            print(f"Epoch {epoch:3d}: train {train_loss:.4f}, val {val_loss:.4f}")

    # Save weights & biases
    torch.save(model.state_dict(), os.path.join(results_dir, "model_state_dict.pt"))
    print(f"Model weights saved to: {results_dir}/model_state_dict.pt")

    # Predict on full set
    model.eval()
    with torch.no_grad():
        pred_res_all = model(torch.from_numpy(X_all).float()).numpy()

    # Build masks for coloring
    train_mask = np.zeros(len(X_all), dtype=bool); train_mask[idx_tr] = True
    val_mask   = np.zeros(len(X_all), dtype=bool); val_mask[idx_val] = True

    # Plots
    labels = ["dx", "dy", "dz"]
    for i, lab in enumerate(labels):
        plt.figure(figsize=(10,4))


        plt.plot(t_all, Y_all[:, i],         label="true (odom2-odom1)", linewidth=1.6)
        plt.plot(t_all, est_xyz_all[:, i], label="uvdar (est-odom1)",  linewidth=1.2)
        plt.plot(t_all, pred_res_all[:, i],  label="predicted",          linewidth=1.2)

        plt.xlabel("Time [s]"); plt.ylabel(f"{lab} [m]")
        plt.legend(); plt.grid(True); plt.tight_layout()
        out = os.path.join(results_dir, f"{lab}_pred_vs_true_vs_uvdar.png")
        plt.savefig(out, dpi=1500); plt.close()
        print(f"Saved {out}")

    print(f"✅ All results saved to: {results_dir}")

if __name__ == "__main__":
    main()
