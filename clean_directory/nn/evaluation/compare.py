#!/usr/bin/env python3
"""
evaluation/compare.py — Compare up to 5 NN result directories against
ground truth and the old UVDAR baseline on the same dataset.

Usage:
    python -m evaluation.compare results/dir1 results/dir2
    python -m evaluation.compare results/dir1 results/dir2 --run-dir ../../data/LARGE_DATASET
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from plot_style import apply_style  # noqa: E402

apply_style()

from evaluation import (
    GT_COLOR, OLD_COLOR,
    GT_LINESTYLE, OLD_LINESTYLE,
    SYS_LINEWIDTH,
    NN_COLORS, AXIS_LABELS,
    shade_train_val, insert_gap_nans,
    t_to_seconds,
)
from evaluation.reconstruct import load_run, friendly_name

MAX_RESULTS = 5
OUTPUT_BASENAME = "nn_compare"

# Match visualize.py: a thicker GT line so it stays readable behind the
# overlaid prediction traces.
GT_LINEWIDTH = 3


def _attach_save_button(fig, basename: str = OUTPUT_BASENAME):
    """Add a Save button that exports the figure to {pgf,pdf,svg}.
    The button itself is hidden during the export."""
    save_ax = fig.add_axes([0.88, 0.005, 0.10, 0.035])
    button = Button(save_ax, "Save")

    def _save(_event):
        save_ax.set_visible(False)
        for ext in ("pgf", "pdf", "svg"):
            fig.savefig(f"{basename}.{ext}",
                        bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {basename}.{{pgf,pdf,svg}}")

    button.on_clicked(_save)
    return button  # keep reference alive



def compare_all(datasets: list[dict],
                show_split: bool = False,
                show_errors: bool = True,
                alpha: float = 1.0):
    if not datasets:
        print("Nothing to compare.")
        return

    ref = datasets[0]
    Y_ref            = ref["Y_all"]
    pred_rel_xyz_all = ref.get("pred_rel_xyz_all")
    t_ref            = t_to_seconds(ref["t_all"])
    val_split        = ref["val_split"]
    has_old = pred_rel_xyz_all is not None

    # Insert NaN at large time gaps so lines break instead of bridging.
    # GT is excluded — drawn continuously from the raw odometry CSV below.
    if has_old:
        t_ref_plot, pred_old_plot = insert_gap_nans(t_ref, pred_rel_xyz_all)
    else:
        t_ref_plot = t_ref
        pred_old_plot = None

    # Continuous GT from raw odometry (no gaps).
    gt_raw_t_ns = ref.get("gt_raw_t_ns")
    gt_raw_xyz  = ref.get("gt_raw_xyz")
    if (gt_raw_t_ns is not None and len(gt_raw_t_ns) > 0
            and len(ref["t_all"]) > 0):
        t0_ns = float(ref["t_all"][0])
        t_gt_raw = (gt_raw_t_ns.astype(np.float64) - t0_ns) / 1e9
        in_range = (t_gt_raw >= t_ref[0]) & (t_gt_raw <= t_ref[-1])
        t_gt_raw = t_gt_raw[in_range]
        gt_raw_xyz = gt_raw_xyz[in_range]
    else:
        t_gt_raw = None
        gt_raw_xyz = None

    n_rows = 4 if show_errors else 3
    fig, axs = plt.subplots(n_rows, 1,
                            figsize=(12, 3 * n_rows),
                            sharex=True)

    for i, (ax, axis_label) in enumerate(zip(axs[:3], AXIS_LABELS)):
        if show_split:
            shade_train_val(ax, t_ref, val_split)

        if t_gt_raw is not None:
            ax.plot(t_gt_raw, gt_raw_xyz[:, i],
                    label="Ground Truth", color=GT_COLOR,
                    linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)
        else:
            ax.plot(t_ref, Y_ref[:, i],
                    label="Ground Truth", color=GT_COLOR,
                    linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)

        if has_old:
            ax.plot(t_ref_plot, pred_old_plot[:, i],
                    label="UVDAR prediction", color=OLD_COLOR,
                    linestyle="none", marker=".", markersize=2,
                    linewidth=SYS_LINEWIDTH, alpha=alpha, zorder=10)

        for j, ds in enumerate(datasets):
            color = NN_COLORS[j % len(NN_COLORS)]
            t_ds, pred_ds = insert_gap_nans(
                t_to_seconds(ds["t_all"]), ds["pred_res_all"])
            ax.plot(t_ds, pred_ds[:, i],
                    label=ds["label"], color=color,
                    linestyle="-", linewidth=SYS_LINEWIDTH,
                    alpha=alpha, zorder=3 + j)

        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(True)

        if i == 0:
            ax.legend()

    if show_errors:
        ax_err = axs[3]
        if show_split:
            shade_train_val(ax_err, t_ref, val_split)

        if has_old:
            err_old = np.linalg.norm(pred_rel_xyz_all - Y_ref, axis=1)
            t_old_e, err_old_plot = insert_gap_nans(t_ref, err_old)
            ax_err.plot(t_old_e, err_old_plot,
                        label="UVDAR Prediction Error", color=OLD_COLOR,
                        linestyle="none", marker=".", markersize=2,
                        linewidth=SYS_LINEWIDTH, alpha=alpha)

        for j, ds in enumerate(datasets):
            color = NN_COLORS[j % len(NN_COLORS)]
            t_ds = t_to_seconds(ds["t_all"])
            err = np.linalg.norm(ds["pred_res_all"] - ds["Y_all"], axis=1)
            t_ds_e, err_plot = insert_gap_nans(t_ds, err)
            ax_err.plot(t_ds_e, err_plot,
                        label=f"{ds['label']} Error", color=color,
                        linestyle="-", linewidth=SYS_LINEWIDTH, alpha=alpha)

        ax_err.set_ylabel("Error (Euclidean) [m]")
        ax_err.grid(True)
        ax_err.legend()

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    _button = _attach_save_button(fig)  # noqa: F841 (kept alive by ref)
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Compare up to 5 NN result directories on the same dataset.",
    )
    ap.add_argument("results_dirs", nargs="+",
                    help="1-5 result directories to compare.")
    ap.add_argument("--run-dir", dest="run_dir_override", default=None,
                    help="Override dataset path for all runs.")
    ap.add_argument("--show-split", action="store_true",
                    help="Shade train/val split regions (off by default).")
    ap.add_argument("--hide-errors", action="store_true",
                    help="Hide the Euclidean error subplot (shown by default).")
    ap.add_argument("--alpha", type=float, default=1.0, metavar="A",
                    help="Opacity for all prediction traces (0.0–1.0, default 1.0).")
    ap.add_argument("--labels", "--names", dest="labels", default=None,
                    help="Comma-separated display names, one per result dir "
                         "(in the same order). Use '-' to keep the auto label "
                         "for a given slot, e.g. --labels 'Direct,-,Residual'.")
    args = ap.parse_args()


    if len(args.results_dirs) > MAX_RESULTS:
        print(f"Warning: only the first {MAX_RESULTS} directories will be shown.")
        args.results_dirs = args.results_dirs[:MAX_RESULTS]

    custom_labels: list[str] = []
    if args.labels is not None:
        custom_labels = [s.strip() for s in args.labels.split(",")]
        if len(custom_labels) != len(args.results_dirs):
            print(f"Warning: got {len(custom_labels)} --labels for "
                  f"{len(args.results_dirs)} result dirs; extras ignored, "
                  f"missing slots keep auto labels.", file=sys.stderr)

    datasets = []
    for idx, rd in enumerate(args.results_dirs):
        try:
            data = load_run(rd, args.run_dir_override)
            auto = friendly_name(rd)
            override = (custom_labels[idx]
                        if idx < len(custom_labels) and custom_labels[idx]
                           and custom_labels[idx] != "-"
                        else None)
            data["label"] = override if override is not None else auto
            datasets.append(data)
            print(f"Loaded: {rd}  ->  label='{data['label']}'")
        except Exception as e:
            print(f"Skipping {rd}: {e}", file=sys.stderr)

    if not datasets:
        print("No valid result directories loaded.", file=sys.stderr)
        sys.exit(1)

    compare_all(datasets,
                show_split=args.show_split,
                show_errors=not args.hide_errors,
                alpha=args.alpha)


if __name__ == "__main__":
    main()
