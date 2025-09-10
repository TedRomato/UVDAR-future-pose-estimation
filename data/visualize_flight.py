#!/usr/bin/env python3
import argparse
import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Break UV-DAR line if time gap exceeds this (seconds)
UVDAR_GAP_SEC = 0.25  # adjust if you want stricter/looser breaking

NEEDED = [
    "time","x","y","z",
    "roll_sin","roll_cos","pitch_sin","pitch_cos","yaw_sin","yaw_cos"
]

def load_one(path):
    """Load one CSV; returns DataFrame with: time,t_rel,x,y,z,yaw_deg,pitch_deg,roll_deg."""
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    missing = [c for c in NEEDED if c not in df.columns]
    if missing:
        raise ValueError(f"{os.path.basename(path)} missing columns: {missing}")

    # reconstruct angles (radians) then degrees
    df["yaw"]   = [math.atan2(s, c) for s, c in zip(df["yaw_sin"],   df["yaw_cos"])]
    df["pitch"] = [math.atan2(s, c) for s, c in zip(df["pitch_sin"], df["pitch_cos"])]
    df["roll"]  = [math.atan2(s, c) for s, c in zip(df["roll_sin"],  df["roll_cos"])]
    df["yaw_deg"]   = df["yaw"]   * 180.0 / math.pi
    df["pitch_deg"] = df["pitch"] * 180.0 / math.pi
    df["roll_deg"]  = df["roll"]  * 180.0 / math.pi
    return df[["time","x","y","z","yaw_deg","pitch_deg","roll_deg"]].copy()

def add_t_rel(dfs):
    """Align to common t0; add t_rel column."""
    t0s = [df["time"].iloc[0] for df in dfs if df is not None and len(df) > 0]
    t0 = min(t0s) if t0s else 0.0
    out = []
    for df in dfs:
        if df is None: 
            out.append(None)
            continue
        d = df.copy()
        d["t_rel"] = d["time"].astype(float) - float(t0)
        out.append(d)
    return out

def break_uvdar_gaps(df, gap_sec):
    """
    Insert NaNs after large time gaps to create visible breaks in the line
    (no interpolation). Works in-place-like: returns a new DataFrame with same columns.
    """
    if df is None or len(df) == 0:
        return df
    d = df.copy()
    t = d["t_rel"].to_numpy()
    # indices where gap exceeds threshold
    gaps = np.where(np.diff(t) > gap_sec)[0]
    if gaps.size == 0:
        return d
    # Build a new dataframe with NaN rows inserted after each gap index
    rows = []
    cols = d.columns
    for i in range(len(d)):
        rows.append(d.iloc[i].values)
        if i in gaps:
            nan_row = np.array([np.nan]*len(cols), dtype='float64')
            # keep time axis monotonic with same t_rel so matplotlib breaks the line
            # (time being NaN would also break; but keep t for readable axis)
            nan_row[list(cols).index("t_rel")] = d.iloc[i]["t_rel"]
            rows.append(nan_row)
    d2 = pd.DataFrame(rows, columns=cols)
    return d2

def plot_param(run_dir, param, ylabel, colors, est, od1, od2):
    plt.figure(figsize=(10,5))
    have_any = False
    if est is not None:
        plt.plot(est["t_rel"], est[param], color=colors["uvdar"], label="uvdar", linewidth=1.6)
        have_any = True
    if od1 is not None:
        plt.plot(od1["t_rel"], od1[param], color=colors["odom1"], label="odom1", linewidth=1.2)
        have_any = True
    if od2 is not None:
        plt.plot(od2["t_rel"], od2[param], color=colors["odom2"], label="odom2", linewidth=1.2)
        have_any = True
    if not have_any:
        plt.close()
        print(f"[skip] {param}: no data")
        return
    plt.xlabel("Time [s]")
    plt.ylabel(ylabel)
    # ultra-simple legend: just color = source
    plt.legend(title="", loc="best")
    plt.grid(True)
    plt.tight_layout()
    out = os.path.join(run_dir, f"param_{param}.png")
    plt.savefig(out, dpi=150)
    print(f"[saved] {out}")

def main():
    ap = argparse.ArgumentParser(description="Plot per-parameter graphs with UV-DAR + odom1 + odom2 (no interpolation for UV-DAR).")
    ap.add_argument("run_dir", help="Folder with estimations.csv / odom1.csv / odom2.csv (e.g., helix/csv_data/1)")
    args = ap.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    est_path = os.path.join(run_dir, "estimations.csv")  # UV-DAR
    od1_path = os.path.join(run_dir, "odom1.csv")
    od2_path = os.path.join(run_dir, "odom2.csv")

    est = load_one(est_path) if os.path.exists(est_path) else None
    od1 = load_one(od1_path) if os.path.exists(od1_path) else None
    od2 = load_one(od2_path) if os.path.exists(od2_path) else None

    if est is None and od1 is None and od2 is None:
        print("[error] No CSVs found to plot.")
        return

    # Align times, then break uvdar gaps
    est, od1, od2 = add_t_rel([est, od1, od2])
    if est is not None:
        est = break_uvdar_gaps(est, UVDAR_GAP_SEC)

    # Fixed, simple colors (feel free to change)
    colors = {"uvdar": "tab:orange", "odom1": "tab:blue", "odom2": "tab:green"}

    # Positions
    plot_param(run_dir, "x", "X [m]", colors, est, od1, od2)
    plot_param(run_dir, "y", "Y [m]", colors, est, od1, od2)
    plot_param(run_dir, "z", "Z [m]", colors, est, od1, od2)

    # Attitudes (degrees)
    plot_param(run_dir, "yaw_deg",   "Yaw [deg]",   colors, est, od1, od2)
    plot_param(run_dir, "pitch_deg", "Pitch [deg]", colors, est, od1, od2)
    plot_param(run_dir, "roll_deg",  "Roll [deg]",  colors, est, od1, od2)

    # Optional interactive show
    plt.show()

if __name__ == "__main__":
    main()
