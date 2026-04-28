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

import numpy as np
import matplotlib.pyplot as plt

from evaluation import (
    GT_COLOR, OLD_COLOR,
    GT_LINESTYLE, OLD_LINESTYLE,
    GT_LINEWIDTH, SYS_LINEWIDTH,
    NN_COLORS, AXIS_LABELS,
    shade_train_val, insert_gap_nans,
    t_to_seconds,
)
from evaluation.reconstruct import load_run, friendly_name

MAX_RESULTS = 5


def compare_all(datasets: list[dict]):
    if not datasets:
        print("Nothing to compare.")
        return

    ref = datasets[0]
    Y_ref            = ref["Y_all"]
    pred_rel_xyz_all = ref.get("pred_rel_xyz_all")
    t_ref            = t_to_seconds(ref["t_all"])
    val_split        = ref["val_split"]
    has_old = pred_rel_xyz_all is not None

    # Insert NaN at large time gaps so lines break instead of bridging
    if has_old:
        t_ref_plot, Y_ref_plot, pred_old_plot = insert_gap_nans(
            t_ref, Y_ref, pred_rel_xyz_all)
    else:
        t_ref_plot, Y_ref_plot = insert_gap_nans(t_ref, Y_ref)
        pred_old_plot = None

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for i, (ax, axis_label) in enumerate(zip(axs, AXIS_LABELS)):
        shade_train_val(ax, t_ref, val_split)

        ax.plot(t_ref_plot, Y_ref_plot[:, i],
                label="ground truth", color=GT_COLOR,
                linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)

        mean_gt = np.nanmean(Y_ref_plot[:, i])
        ax.axhline(y=mean_gt, color=GT_COLOR, linestyle="--",
                   linewidth=1.0, alpha=0.6,
                   label="GT mean" if i == 0 else None, zorder=1)

        if has_old:
            ax.plot(t_ref_plot, pred_old_plot[:, i],
                    label="UVDAR (input)", color=OLD_COLOR,
                    linestyle=OLD_LINESTYLE, linewidth=SYS_LINEWIDTH, zorder=2)

        for j, ds in enumerate(datasets):
            color = NN_COLORS[j % len(NN_COLORS)]
            # Each dataset may have its own timeline (different feature configs
            # produce different sample counts), so use each dataset's own t_all.
            t_ds, pred_ds = insert_gap_nans(
                t_to_seconds(ds["t_all"]), ds["pred_res_all"])
            ax.plot(t_ds, pred_ds[:, i],
                    label=ds["label"], color=color,
                    linestyle="-", linewidth=SYS_LINEWIDTH, zorder=3 + j)

        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(True)

        if i == 0:
            title = ("Comparison: GT vs UVDAR vs NN" if has_old
                     else "Comparison: GT vs NN")
            ax.set_title(title)
            ax.legend(fontsize=8)

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Compare up to 5 NN result directories on the same dataset.",
    )
    ap.add_argument("results_dirs", nargs="+",
                    help="1-5 result directories to compare.")
    ap.add_argument("--run-dir", dest="run_dir_override", default=None,
                    help="Override dataset path for all runs.")
    args = ap.parse_args()

    if len(args.results_dirs) > MAX_RESULTS:
        print(f"Warning: only the first {MAX_RESULTS} directories will be shown.")
        args.results_dirs = args.results_dirs[:MAX_RESULTS]

    datasets = []
    for rd in args.results_dirs:
        try:
            data = load_run(rd, args.run_dir_override)
            data["label"] = friendly_name(rd)
            datasets.append(data)
            print(f"Loaded: {rd}  ->  label='{data['label']}'")
        except Exception as e:
            print(f"Skipping {rd}: {e}", file=sys.stderr)

    if not datasets:
        print("No valid result directories loaded.", file=sys.stderr)
        sys.exit(1)

    compare_all(datasets)


if __name__ == "__main__":
    main()
