#!/usr/bin/env python3
"""
Compare up to 5 NN result directories against ground truth and the
original UVDAR prediction on the same dataset.

Usage:
    python compare_results.py results/dir1 results/dir2 [results/dir3 ...]
    python compare_results.py results/dir1 results/dir2 --run-dir LARGE_DATASET
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from visualize_results_core import (
    GT_COLOR, OLD_COLOR,
    GT_LINESTYLE, OLD_LINESTYLE,
    GT_LINEWIDTH, SYS_LINEWIDTH,
    NN_COLORS, AXIS_LABELS,
    shade_train_val,
    load_results_dir, friendly_name,
)

MAX_RESULTS = 5


def compare_all(datasets):
    """
    Parameters
    ----------
    datasets : list[dict]
        Each dict is the output of load_results_dir(), plus a 'label' key.
    """
    if not datasets:
        print("Nothing to compare.")
        return

    # Use the first result directory's data as reference for ground truth,
    # time axis, and val_split.
    ref = datasets[0]
    Y_all            = ref["Y_all"]
    pred_rel_xyz_all = ref.get("pred_rel_xyz_all")  # None for blinkers
    t_all            = ref["t_all"]
    val_split        = ref["val_split"]

    has_old = pred_rel_xyz_all is not None

    # Collect model types for the title
    model_types_seen = sorted(set(ds.get("model_type", "3d") for ds in datasets))
    type_tag = ", ".join(model_types_seen)

    fig, axs = plt.subplots(3, 1, figsize=(12, 9), sharex=True)

    for i, (ax, axis_label) in enumerate(zip(axs, AXIS_LABELS)):
        shade_train_val(ax, t_all, val_split)

        # Ground truth
        ax.plot(t_all, Y_all[:, i],
                label="ground truth",
                color=GT_COLOR, linestyle=GT_LINESTYLE,
                linewidth=GT_LINEWIDTH, zorder=1)

        # Original / old-system prediction (only for 3d model type)
        if has_old:
            ax.plot(t_all, pred_rel_xyz_all[:, i],
                    label="UVDAR (input)",
                    color=OLD_COLOR, linestyle=OLD_LINESTYLE,
                    linewidth=SYS_LINEWIDTH, zorder=2)

        # NN predictions (up to 5)
        for j, ds in enumerate(datasets):
            color = NN_COLORS[j % len(NN_COLORS)]
            ax.plot(t_all, ds["pred_res_all"][:, i],
                    label=ds["label"],
                    color=color, linestyle="-",
                    linewidth=SYS_LINEWIDTH, zorder=3 + j)

        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(True)

        if i == 0:
            title = (f"Comparison: Ground Truth vs UVDAR vs NN [{type_tag}]"
                     if has_old else
                     f"Comparison: Ground Truth vs NN [{type_tag}]")
            ax.set_title(title)
            ax.legend(fontsize=8)

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Compare up to 5 NN result directories on the same dataset.",
    )
    ap.add_argument(
        "results_dirs", nargs="+",
        help="1–5 result directories to compare.",
    )
    ap.add_argument(
        "--run-dir", "--dataset-path", dest="run_dir_override", default=None,
        help="Override dataset path for all result directories.",
    )
    args = ap.parse_args()

    if len(args.results_dirs) > MAX_RESULTS:
        print(f"Warning: only the first {MAX_RESULTS} result directories will be shown.")
        args.results_dirs = args.results_dirs[:MAX_RESULTS]

    datasets = []
    for rd in args.results_dirs:
        try:
            data = load_results_dir(rd, args.run_dir_override)
            data["label"] = friendly_name(rd)
            datasets.append(data)
            print(f"Loaded: {rd}  →  label='{data['label']}'")
        except Exception as e:
            print(f"Skipping {rd}: {e}", file=sys.stderr)

    if not datasets:
        print("No valid result directories loaded.", file=sys.stderr)
        sys.exit(1)

    compare_all(datasets)


if __name__ == "__main__":
    main()
