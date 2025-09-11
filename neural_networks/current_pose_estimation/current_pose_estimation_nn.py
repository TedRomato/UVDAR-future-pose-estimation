#!/usr/bin/env python3
import argparse, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# ---------- helpers ----------

def load_xyz(path):
    """
    Load CSV with time,x,y,z. Keep 3-decimal times as integer ms for exact matching.
    """
    df = pd.read_csv(path)[["time","x","y","z"]].dropna().sort_values("time")
    df["t_ms"] = (df["time"] * 1000).round().astype(np.int64)
    df = df.drop_duplicates(subset="t_ms")
    return df

def interpolate_on_index(target_idx, df):
    """
    Reindex df (with t_ms,x,y,z) onto target_idx (int ms).
    Returns a DataFrame indexed by t_ms with columns x,y,z.
    Uses linear interpolation along numeric index, then ffill/bfill for edges.
    """
    d = (df[["t_ms","x","y","z"]]
         .set_index("t_ms")
         .sort_index())
    out = (d.reindex(target_idx)
             .interpolate(method="index")
             .ffill()
             .bfill())
    out.index.name = "t_ms"
    return out[["x","y","z"]]

def align_three_streams_on_uvdar(est_df, od1_df, od2_df):
    """
    Build ONE common timeline = UVDAR t_ms.
    Interpolate odom1 and odom2 onto that exact grid, then return aligned arrays.
    """
    ref_ms = est_df["t_ms"].values  # exact integer ms grid
    # Make sure it's unique & sorted (should already be)
    ref_ms = np.unique(ref_ms)

    od1_al = interpolate_on_index(ref_ms, od1_df)  # index = ref_ms
    od2_al = interpolate_on_index(ref_ms, od2_df)

    # Sanity: both aligned indices must equal ref_ms
    assert np.array_equal(od1_al.index.values, ref_ms), "odometry1 not aligned to ref grid"
    assert np.array_equal(od2_al.index.values, ref_ms), "odometry2 not aligned to ref grid"

    # Align UVDAR positions by selecting the same time rows (mask membership on ref_ms)
    # est_df may have fewer rows than ref_ms if duplicates were removed differently; match by index.
    est_on_ref = (est_df.set_index("t_ms").loc[ref_ms])[["x","y","z"]]

    # Return aligned arrays + the matching float time vector for plots
    t_sec = ref_ms.astype(np.float64) / 1000.0
    return est_on_ref.values.astype(np.float32), \
           od1_al.values.astype(np.float32), \
           od2_al.values.astype(np.float32), \
           t_sec

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="UVDAR [x,y,z] -> residual (odom2-odom1) with strict time alignment.")
    ap.add_argument("run_dir")
    args = ap.parse_args()

    # Load data
    est = load_xyz(os.path.join(args.run_dir,"estimations.csv"))
    od1 = load_xyz(os.path.join(args.run_dir,"odom1.csv"))
    od2 = load_xyz(os.path.join(args.run_dir,"odom2.csv"))

    # Strict alignment on a single timeline (UV-DAR t_ms)
    est_xyz_all, od1_xyz_all, od2_xyz_all, t_all = align_three_streams_on_uvdar(est, od1, od2)

    # Targets and inputs on EXACT same timestamps
    Y_all = od2_xyz_all - od1_xyz_all                     # true residual
    X_all = est_xyz_all                                   # model input

    # Filter any rows with NaNs/Infs (shouldn’t happen after ffill/bfill, but just in case)
    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all, Y_all, t_all = X_all[ok], Y_all[ok], t_all[ok]

    # Train/val split — keep indices to color background on the original time axis
    idx = np.arange(len(X_all))
    idx_tr, idx_val, Xtr, Xval, Ytr, Yval = train_test_split(idx, X_all, Y_all, test_size=0.2, random_state=42)
    Xtr, Ytr = torch.from_numpy(Xtr), torch.from_numpy(Ytr)
    Xval, Yval = torch.from_numpy(Xval), torch.from_numpy(Yval)

    # Tiny model (no class)
    model = nn.Sequential(
        nn.Linear(3, 2048),
        nn.ReLU(),
        nn.ReLU(),
        nn.ReLU(),
        nn.ReLU(),
        nn.Linear(2048, 3)
    )
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train
    EPOCHS = 500
    for epoch in range(1, EPOCHS+1):
        model.train()
        pred = model(Xtr)
        loss = loss_fn(pred, Ytr)
        opt.zero_grad(); loss.backward(); opt.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == EPOCHS:
            model.eval()
            with torch.no_grad():
                val_loss = loss_fn(model(Xval), Yval).item()
            print(f"Epoch {epoch:3d}: train {loss.item():.4f}, val {val_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(args.run_dir, "simple_model.pt"))
    print("Model saved. Generating plots...")

    # Predict on all aligned samples
    model.eval()
    with torch.no_grad():
        pred_res_all = model(torch.from_numpy(X_all)).numpy()

    # Build train/val boolean masks in timeline order
    train_mask = np.zeros(len(X_all), dtype=bool); train_mask[idx_tr] = True
    val_mask   = np.zeros(len(X_all), dtype=bool); val_mask[idx_val] = True

    # Plots: true vs UV-DAR vs predicted
    labels = ["dx", "dy", "dz"]
    for i, lab in enumerate(labels):
        plt.figure(figsize=(10,4))

        # Curves
        plt.plot(t_all, Y_all[:,i],         label="true (odom2-odom1)", linewidth=1.6)
        plt.plot(t_all, est_xyz_all[:,i], label="uvdar (est-odom1)",  linewidth=1.2)
        plt.plot(t_all, pred_res_all[:,i],  label="predicted",          linewidth=1.2)

        plt.xlabel("Time [s]"); plt.ylabel(f"{lab} [m]")
        plt.legend(); plt.grid(True); plt.tight_layout()
        out = os.path.join(args.run_dir, f"{lab}_pred_vs_true_vs_uvdar.png")
        plt.savefig(out, dpi=150); plt.close()
        print(f"Saved {out}")

if __name__ == "__main__":
    main()
