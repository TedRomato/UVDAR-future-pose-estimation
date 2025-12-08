#!/usr/bin/env python3
import argparse
import os
import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

import helpers  # uses load_xyz + align_three_streams_on_uvdar


# ---------- model reconstruction ----------

def build_model_from_config(cfg, in_dim=3, out_dim=3):
    """
    Rebuild the MLP architecture used in training from the config.
    """
    hidden_sizes = cfg["layers"]
    act_name = cfg["activation"].lower()

    layers = []
    cur_dim = in_dim

    for h in hidden_sizes:
        layers.append(nn.Linear(cur_dim, h))

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

        cur_dim = h

    layers.append(nn.Linear(cur_dim, out_dim))  # output 3D residual
    return nn.Sequential(*layers)


# ---------- plotting ----------

def plot_per_axis_residuals(artifacts):
    Y_all        = artifacts["Y_all"]         # (N, 3)
    est_xyz_all  = artifacts["est_xyz_all"]   # (N, 3)
    pred_res_all = artifacts["pred_res_all"]  # (N, 3)
    t_all        = artifacts["t_all"]         # (N,)

    labels = ["dx", "dy", "dz"]

    for i, lab in enumerate(labels):
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(10, 6), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )

        l_true = ax1.plot(
            t_all, Y_all[:, i],
            label="true (odom2 - odom1)", linewidth=1.6,
        )[0]

        l_uvdar = ax1.plot(
            t_all, est_xyz_all[:, i],
            label="uvdar (est - odom1)", linewidth=1.2,
        )[0]

        l_pred = ax1.plot(
            t_all, pred_res_all[:, i],
            label="predicted (NN)", linewidth=1.2,
        )[0]

        ax1.set_ylabel(f"{lab} [m]")
        ax1.legend()
        ax1.grid(True)

        abs_err_uvdar = np.abs(est_xyz_all[:, i] - Y_all[:, i])
        abs_err_pred  = np.abs(pred_res_all[:, i] - Y_all[:, i])

        ax2.plot(
            t_all, abs_err_uvdar,
            label="|UVDAR error|", color=l_uvdar.get_color(),
        )
        ax2.plot(
            t_all, abs_err_pred,
            label="|NN error|", color=l_pred.get_color(),
        )

        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("|error| [m]")
        ax2.legend()
        ax2.grid(True)

        fig.tight_layout()


def plot_error_magnitudes(artifacts):
    Y_all        = artifacts["Y_all"]
    est_xyz_all  = artifacts["est_xyz_all"]
    pred_res_all = artifacts["pred_res_all"]
    t_all        = artifacts["t_all"]

    err_vec_uvdar = est_xyz_all - Y_all
    err_vec_pred  = pred_res_all - Y_all

    err_mag_uvdar = np.linalg.norm(err_vec_uvdar, axis=1)
    err_mag_pred  = np.linalg.norm(err_vec_pred, axis=1)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_all, err_mag_uvdar, label="||UVDAR error||", alpha=0.8)
    ax.plot(t_all, err_mag_pred,  label="||NN error||",    alpha=0.8)

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Error magnitude [m]")
    ax.set_title("3D Error Magnitude Over Time")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()


def visualize_all(artifacts):
    plot_per_axis_residuals(artifacts)
    plot_error_magnitudes(artifacts)
    plt.show()


# ---------- main: load everything from a results_dir ----------

def main():
    ap = argparse.ArgumentParser(
        description="Visualize NN vs UVDAR residuals from a saved results directory."
    )
    ap.add_argument(
        "results_dir",
        help="Directory with config_used.yaml, model_state_dict.pt, normalization.json, dataset_path.txt",
    )
    ap.add_argument(
        "--run-dir",
        "--dataset-path",
        dest="run_dir_override",
        help="Override dataset path instead of using dataset_path.txt from results_dir",
    )
    args = ap.parse_args()

    results_dir = os.path.abspath(args.results_dir)

    # --- load config ---
    cfg_path = os.path.join(results_dir, "config_used.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # --- load norm stats ---
    norm_path = os.path.join(results_dir, "normalization.json")
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization stats not found: {norm_path}")
    with open(norm_path, "r") as f:
        norm_raw = json.load(f)
    norm_stats = {k: np.array(v) for k, v in norm_raw.items()}
    X_mean = norm_stats["X_mean"]
    X_std  = norm_stats["X_std"]
    Y_mean = norm_stats["Y_mean"]
    Y_std  = norm_stats["Y_std"]

    # --- decide dataset path (run_dir) ---
    if args.run_dir_override:
        run_dir = os.path.abspath(args.run_dir_override)
        print(f"Using OVERRIDDEN dataset path: {run_dir}")
    else:
        dataset_path_file = os.path.join(results_dir, "dataset_path.txt")
        if not os.path.exists(dataset_path_file):
            raise FileNotFoundError(f"Dataset path file not found: {dataset_path_file}")
        with open(dataset_path_file, "r") as f:
            run_dir = f.readline().strip()
        print(f"Using dataset path from results_dir: {run_dir}")

    # --- rebuild model & load weights ---
    model = build_model_from_config(cfg)
    weights_path = os.path.join(results_dir, "model_state_dict.pt")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model weights not found: {weights_path}")
    state_dict = torch.load(weights_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # --- reload CSVs and recompute residuals like in training ---
    est = helpers.load_xyz(os.path.join(run_dir, "estimations.csv"))
    od1 = helpers.load_xyz(os.path.join(run_dir, "odom1.csv"))
    od2 = helpers.load_xyz(os.path.join(run_dir, "odom2.csv"))

    est_xyz_all, od1_xyz_all, od2_xyz_all, t_all = helpers.align_three_streams_on_uvdar(
        est, od1, od2
    )

    Y_all = od2_xyz_all - od1_xyz_all  # true residuals [m]
    X_all = est_xyz_all                # UVDAR [x,y,z] [m]

    ok = np.isfinite(X_all).all(axis=1) & np.isfinite(Y_all).all(axis=1)
    X_all = X_all[ok]
    Y_all = Y_all[ok]
    est_xyz_all = est_xyz_all[ok]
    t_all = t_all[ok]

    X_all_n = (X_all - X_mean) / X_std

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(X_all_n).float()).numpy()
    pred_res_all = pred_norm * Y_std + Y_mean

    artifacts = {
        "Y_all": Y_all,
        "est_xyz_all": est_xyz_all,
        "pred_res_all": pred_res_all,
        "t_all": t_all,
    }

    visualize_all(artifacts)


if __name__ == "__main__":
    main()
