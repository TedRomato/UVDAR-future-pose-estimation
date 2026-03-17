#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt

from visualize_results_core import (
    GT_COLOR, OLD_COLOR,
    GT_LINESTYLE, OLD_LINESTYLE,
    GT_LINEWIDTH, SYS_LINEWIDTH,
    RMSE_TEXT_LOC, RMSE_TEXT_KW,
    AXIS_LABELS,
    shade_train_val, split_masks,
    rmse, improvement_pct,
    load_results_dir,
)

NEW_COLOR = "#0072B2"
NEW_LINESTYLE = "-"


# ---------- plotting ----------

def visualize_all(artifacts):
    model_type       = artifacts.get("model_type", "3d")
    Y_all            = artifacts["Y_all"]
    pred_rel_xyz_all = artifacts.get("pred_rel_xyz_all")  # None for blinkers
    pred_res_all     = artifacts["pred_res_all"]
    t_all            = artifacts["t_all"]
    val_split        = artifacts["val_split"]
    train_mask, val_mask = split_masks(len(t_all), val_split)

    has_old = pred_rel_xyz_all is not None

    _MODEL_LABELS = {
        "3d":              "3D Predicted Pose",
        "blinkers":        "Blinkers",
        "blinkers_and_3d": "Blinkers + 3D",
    }
    model_label = _MODEL_LABELS.get(model_type, model_type)

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # --- Per-axis plots (x, y, z) ---
    for i, (ax, axis_label) in enumerate(zip(axs[:3], AXIS_LABELS)):
        shade_train_val(ax, t_all, val_split)

        ax.plot(t_all, Y_all[:, i],
                label="ground truth (true relative pose)",
                color=GT_COLOR, linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)

        if has_old:
            ax.plot(t_all, pred_rel_xyz_all[:, i],
                    label="old system (predicted rel. pose)",
                    color=OLD_COLOR, linestyle=OLD_LINESTYLE, linewidth=SYS_LINEWIDTH, zorder=2)

        ax.plot(t_all, pred_res_all[:, i],
                label="new system (NN)",
                color=NEW_COLOR, linestyle=NEW_LINESTYLE, linewidth=SYS_LINEWIDTH, zorder=3)

        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(True)

        new_err = pred_res_all[:, i] - Y_all[:, i]
        rmse_new_train = rmse(new_err[train_mask])
        rmse_new_val   = rmse(new_err[val_mask])

        if has_old:
            old_err = pred_rel_xyz_all[:, i] - Y_all[:, i]
            rmse_old_train = rmse(old_err[train_mask])
            rmse_old_val   = rmse(old_err[val_mask])
            imp_train = improvement_pct(rmse_old_train, rmse_new_train)
            imp_val   = improvement_pct(rmse_old_val, rmse_new_val)
            rmse_text = (
                f"Training RMSE old: {rmse_old_train:.3f} m\n"
                f"Validation RMSE old: {rmse_old_val:.3f} m\n"
                f"Training RMSE new: {rmse_new_train:.3f} m\n"
                f"Validation RMSE new: {rmse_new_val:.3f} m\n"
                f"Training improvement: {imp_train:.2f}%\n"
                f"Validation improvement: {imp_val:.2f}%"
            )
        else:
            rmse_text = (
                f"Training RMSE NN: {rmse_new_train:.3f} m\n"
                f"Validation RMSE NN: {rmse_new_val:.3f} m"
            )

        ax.text(RMSE_TEXT_LOC[0], RMSE_TEXT_LOC[1], rmse_text,
                transform=ax.transAxes,
                **{k: v for k, v in RMSE_TEXT_KW.items() if k != "transform"})

        if i == 0:
            title = (f"Ground Truth vs UVDAR vs NN ({model_label})"
                     if has_old else
                     f"Ground Truth vs NN ({model_label})")
            ax.set_title(title)
            ax.legend()

    # --- Error magnitude (Euclidean distance) plot ---
    ax_err = axs[3]
    shade_train_val(ax_err, t_all, val_split)

    err_mag_new = np.linalg.norm(pred_res_all - Y_all, axis=1)

    if has_old:
        err_mag_old = np.linalg.norm(pred_rel_xyz_all - Y_all, axis=1)
        ax_err.plot(t_all, err_mag_old,
                    label="old system error", color=OLD_COLOR,
                    linestyle=OLD_LINESTYLE, linewidth=SYS_LINEWIDTH)

    ax_err.plot(t_all, err_mag_new,
                label="new system error", color=NEW_COLOR,
                linestyle=NEW_LINESTYLE, linewidth=SYS_LINEWIDTH)

    ax_err.set_ylabel("Error (Euclidean) [m]")
    ax_err.set_title("3D Error Magnitude")
    ax_err.grid(True)
    ax_err.legend()

    rmse_new_3d_train = rmse(err_mag_new[train_mask])
    rmse_new_3d_val   = rmse(err_mag_new[val_mask])

    if has_old:
        rmse_old_3d_train = rmse(err_mag_old[train_mask])
        rmse_old_3d_val   = rmse(err_mag_old[val_mask])
        imp_3d_train = improvement_pct(rmse_old_3d_train, rmse_new_3d_train)
        imp_3d_val   = improvement_pct(rmse_old_3d_val, rmse_new_3d_val)
        err_text = (
            f"Training RMSE old: {rmse_old_3d_train:.3f} m\n"
            f"Validation RMSE old: {rmse_old_3d_val:.3f} m\n"
            f"Training RMSE new: {rmse_new_3d_train:.3f} m\n"
            f"Validation RMSE new: {rmse_new_3d_val:.3f} m\n"
            f"Training improvement: {imp_3d_train:.2f}%\n"
            f"Validation improvement: {imp_3d_val:.2f}%"
        )
    else:
        err_text = (
            f"Training RMSE NN: {rmse_new_3d_train:.3f} m\n"
            f"Validation RMSE NN: {rmse_new_3d_val:.3f} m"
        )

    ax_err.text(
        RMSE_TEXT_LOC[0], RMSE_TEXT_LOC[1], err_text,
        transform=ax_err.transAxes,
        **{k: v for k, v in RMSE_TEXT_KW.items() if k != "transform"},
    )

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.show()


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Visualize NN vs UVDAR residuals from a saved results directory."
    )
    ap.add_argument(
        "results_dir",
        help="Directory with config_used.yaml, model_state_dict.pt, normalization.json, dataset_path.txt",
    )
    ap.add_argument(
        "--run-dir", "--dataset-path", dest="run_dir_override",
        help="Override dataset path instead of using dataset_path.txt from results_dir.",
    )
    args = ap.parse_args()

    data = load_results_dir(args.results_dir, args.run_dir_override)
    visualize_all(data)


if __name__ == "__main__":
    main()
