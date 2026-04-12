#!/usr/bin/env python3
"""
evaluation/visualize.py — Visualize a single NN result vs ground truth
and (optionally) the old UVDAR baseline.

Usage:
    python -m evaluation.visualize results/my-run_val0.001234
    python -m evaluation.visualize results/my-run --run-dir ../../data/LARGE_DATASET
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt

from evaluation import (
    GT_COLOR, OLD_COLOR,
    GT_LINESTYLE, OLD_LINESTYLE,
    GT_LINEWIDTH, SYS_LINEWIDTH,
    RMSE_TEXT_LOC, RMSE_TEXT_KW,
    AXIS_LABELS,
    shade_train_val, split_masks,
    rmse, improvement_pct, insert_gap_nans,
)
from evaluation.reconstruct import load_run

NEW_COLOR = "#0072B2"
NEW_LINESTYLE = "-"


def visualize_all(artifacts: dict):
    Y_all            = artifacts["Y_all"]
    pred_rel_xyz_all = artifacts.get("pred_rel_xyz_all")
    pred_res_all     = artifacts["pred_res_all"]
    t_all            = artifacts["t_all"]
    val_split        = artifacts["val_split"]
    train_mask, val_mask = split_masks(len(t_all), val_split)

    has_old = pred_rel_xyz_all is not None

    # Insert NaN at large time gaps so lines break instead of bridging
    if has_old:
        t_plot, Y_plot, pred_old_plot, pred_new_plot = insert_gap_nans(
            t_all, Y_all, pred_rel_xyz_all, pred_res_all)
    else:
        t_plot, Y_plot, pred_new_plot = insert_gap_nans(
            t_all, Y_all, pred_res_all)
        pred_old_plot = None

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Per-axis (x, y, z)
    for i, (ax, axis_label) in enumerate(zip(axs[:3], AXIS_LABELS)):
        shade_train_val(ax, t_all, val_split)

        ax.plot(t_plot, Y_plot[:, i],
                label="ground truth", color=GT_COLOR,
                linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)

        mean_gt = np.nanmean(Y_plot[:, i])
        ax.axhline(y=mean_gt, color=GT_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.6,
                   label="GT mean" if i == 0 else None, zorder=1)

        if has_old:
            ax.plot(t_plot, pred_old_plot[:, i],
                    label="old system (UVDAR)", color=OLD_COLOR,
                    linestyle=OLD_LINESTYLE, linewidth=SYS_LINEWIDTH, zorder=2)

        ax.plot(t_plot, pred_new_plot[:, i],
                label="NN prediction", color=NEW_COLOR,
                linestyle=NEW_LINESTYLE, linewidth=SYS_LINEWIDTH, zorder=3)

        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(True)

        new_err = pred_res_all[:, i] - Y_all[:, i]
        rmse_new_tr  = rmse(new_err[train_mask])
        rmse_new_val = rmse(new_err[val_mask])

        if has_old:
            old_err = pred_rel_xyz_all[:, i] - Y_all[:, i]
            rmse_old_tr  = rmse(old_err[train_mask])
            rmse_old_val = rmse(old_err[val_mask])
            imp_tr  = improvement_pct(rmse_old_tr, rmse_new_tr)
            imp_val = improvement_pct(rmse_old_val, rmse_new_val)
            txt = (
                f"Train RMSE old: {rmse_old_tr:.3f} m\n"
                f"Val RMSE old: {rmse_old_val:.3f} m\n"
                f"Train RMSE NN: {rmse_new_tr:.3f} m\n"
                f"Val RMSE NN: {rmse_new_val:.3f} m\n"
                f"Train impr: {imp_tr:.2f}%\n"
                f"Val impr: {imp_val:.2f}%"
            )
        else:
            txt = (
                f"Train RMSE NN: {rmse_new_tr:.3f} m\n"
                f"Val RMSE NN: {rmse_new_val:.3f} m"
            )

        ax.text(RMSE_TEXT_LOC[0], RMSE_TEXT_LOC[1], txt,
                transform=ax.transAxes, **RMSE_TEXT_KW)

        if i == 0:
            title = ("Ground Truth vs UVDAR vs NN" if has_old
                     else "Ground Truth vs NN")
            ax.set_title(title)
            ax.legend()

    # Euclidean error magnitude
    ax_err = axs[3]
    shade_train_val(ax_err, t_all, val_split)

    err_mag_new = np.linalg.norm(pred_res_all - Y_all, axis=1)

    if has_old:
        err_mag_old = np.linalg.norm(pred_rel_xyz_all - Y_all, axis=1)
        t_err_plot, err_old_plot, err_new_plot = insert_gap_nans(
            t_all, err_mag_old, err_mag_new)
        ax_err.plot(t_err_plot, err_old_plot,
                    label="old system error", color=OLD_COLOR,
                    linestyle=OLD_LINESTYLE, linewidth=SYS_LINEWIDTH)
    else:
        t_err_plot, err_new_plot = insert_gap_nans(t_all, err_mag_new)

    ax_err.plot(t_err_plot, err_new_plot,
                label="NN error", color=NEW_COLOR,
                linestyle=NEW_LINESTYLE, linewidth=SYS_LINEWIDTH)

    ax_err.set_ylabel("Error (Euclidean) [m]")
    ax_err.set_title("3D Error Magnitude")
    ax_err.grid(True)
    ax_err.legend()

    rmse_nn_3d_tr  = rmse(err_mag_new[train_mask])
    rmse_nn_3d_val = rmse(err_mag_new[val_mask])

    if has_old:
        rmse_old_3d_tr  = rmse(err_mag_old[train_mask])
        rmse_old_3d_val = rmse(err_mag_old[val_mask])
        imp_3d_tr  = improvement_pct(rmse_old_3d_tr, rmse_nn_3d_tr)
        imp_3d_val = improvement_pct(rmse_old_3d_val, rmse_nn_3d_val)
        txt = (
            f"Train RMSE old: {rmse_old_3d_tr:.3f} m\n"
            f"Val RMSE old: {rmse_old_3d_val:.3f} m\n"
            f"Train RMSE NN: {rmse_nn_3d_tr:.3f} m\n"
            f"Val RMSE NN: {rmse_nn_3d_val:.3f} m\n"
            f"Train impr: {imp_3d_tr:.2f}%\n"
            f"Val impr: {imp_3d_val:.2f}%"
        )
    else:
        txt = (
            f"Train RMSE NN: {rmse_nn_3d_tr:.3f} m\n"
            f"Val RMSE NN: {rmse_nn_3d_val:.3f} m"
        )

    ax_err.text(RMSE_TEXT_LOC[0], RMSE_TEXT_LOC[1], txt,
                transform=ax_err.transAxes, **RMSE_TEXT_KW)

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Visualize NN predictions from a saved results directory.",
    )
    ap.add_argument(
        "results_dir",
        help="Directory with config.yaml, model.pt, normalization.json",
    )
    ap.add_argument(
        "--run-dir", dest="run_dir_override", default=None,
        help="Override dataset path",
    )
    args = ap.parse_args()

    data = load_run(args.results_dir, args.run_dir_override)
    visualize_all(data)


if __name__ == "__main__":
    main()
