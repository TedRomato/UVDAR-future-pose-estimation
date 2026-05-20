#!/usr/bin/env python3
"""
evaluation/visualize.py — Visualize a single NN result vs ground truth
and (optionally) the old UVDAR baseline.

Usage:
    python -m evaluation.visualize results/my-run_val0.001234
    python -m evaluation.visualize results/my-run --run-dir ../../data/LARGE_DATASET
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
    GT_LINEWIDTH, SYS_LINEWIDTH,
    RMSE_TEXT_LOC, RMSE_TEXT_KW,
    AXIS_LABELS,
    shade_train_val, split_masks,
    rmse, improvement_pct, insert_gap_nans,
    t_to_seconds,
)

from evaluation.reconstruct import load_run

NEW_COLOR = "#0072B2"
NEW_LINESTYLE = "-"
GT_LINEWIDTH = 4
EMA_COLOR = "#56B4E9"
EMA_LINESTYLE = "-"

# Opacity for the raw (un-filtered) NN prediction traces.
RAW_NN_ALPHA = 0.85

OUTPUT_BASENAME = "nn_visualize"


def _attach_save_button(fig, basename: str = OUTPUT_BASENAME):
    """Add a Save button that exports the figure to {pgf,pdf,svg}.
    The button itself is hidden during the export."""
    save_ax = fig.add_axes([0.88, 0.005, 0.10, 0.030])
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


def _ema_filter_time_aware(
    x: np.ndarray,
    t: np.ndarray,
    tau: float,
    reset_gap: float | None = None,
) -> np.ndarray:
    """
    Causal, time-aware exponential moving average.

    The per-sample blend factor is ``alpha = 1 - exp(-dt / tau)``, so the
    filter behaves consistently regardless of sample rate (designed for
    irregular ~30–60 Hz streams).

    Parameters
    ----------
    x : (N, D) or (N,) array
        Signal to filter. NaN rows are skipped (state held).
    t : (N,) array
        Sample timestamps in seconds (monotonic).
    tau : float
        Time constant in seconds. Smaller = more responsive.
    reset_gap : float, optional
        If the gap to the previous valid sample exceeds this many seconds,
        re-seed the state with the new sample (no stale blending).

    Returns
    -------
    Filtered array with the same shape as ``x``. Held-state samples take
    the previous filter value; output is NaN before the first valid sample.
    """
    if tau <= 0.0:
        raise ValueError(f"EMA tau must be > 0, got {tau}")
    out = np.full_like(x, np.nan, dtype=float)
    state = None
    last_t = None
    for i in range(len(x)):
        row = x[i]
        ti = t[i]
        if np.any(~np.isfinite(row)):
            if state is not None:
                out[i] = state
            continue
        if state is None or last_t is None:
            state = row.copy() if hasattr(row, "copy") else row
        else:
            dt = ti - last_t
            if dt <= 0.0:
                pass  # duplicate / out-of-order timestamp: keep state
            elif reset_gap is not None and dt > reset_gap:
                state = row.copy() if hasattr(row, "copy") else row
            else:
                alpha = 1.0 - np.exp(-dt / tau)
                state = alpha * row + (1.0 - alpha) * state
        out[i] = state
        last_t = ti
    return out


def _coverage_by_bins(
    t: np.ndarray,
    valid: np.ndarray,
    bin_seconds: float,
) -> tuple[int, int]:
    """Bin the timeline into ``bin_seconds`` chunks; count bins containing
    at least one valid sample. Returns (n_covered, n_total)."""
    if len(t) == 0:
        return 0, 0
    t0, t1 = float(t[0]), float(t[-1])
    n_total = max(1, int(np.ceil((t1 - t0) / bin_seconds)))
    bin_idx = np.minimum(((t - t0) / bin_seconds).astype(np.int64),
                         n_total - 1)
    covered = np.zeros(n_total, dtype=bool)
    np.logical_or.at(covered, bin_idx[valid], True)
    return int(covered.sum()), int(n_total)


def _rmse_dict(err_axis: dict, err_3d: np.ndarray, mask: np.ndarray) -> dict:
    return {**{ax: rmse(err_axis[ax][mask]) for ax in ("x", "y", "z")},
            "3d": rmse(err_3d[mask])}


def _print_rmse_row(name: str, vals: dict) -> None:
    print(f"  {name:<16} x={vals['x']:7.3f}  y={vals['y']:7.3f}  "
          f"z={vals['z']:7.3f}  3D={vals['3d']:7.3f}  m")


def _print_improvement(name: str, base: dict, new: dict) -> None:
    imp = {k: improvement_pct(base[k], new[k]) for k in ("x", "y", "z", "3d")}
    print(f"  {name:<16} x={imp['x']:+7.2f}%  y={imp['y']:+7.2f}%  "
          f"z={imp['z']:+7.2f}%  3D={imp['3d']:+7.2f}%   "
          f"(on UVDAR-overlap samples)")


def _report_subset(
    label: str,
    mask: np.ndarray,
    err_axis_old: dict, err_3d_old: np.ndarray,
    err_axis_new: dict, err_3d_new: np.ndarray,
    err_axis_ema: dict | None, err_3d_ema: np.ndarray | None,
    overlap: np.ndarray,
    show_raw_nn: bool,
) -> None:
    """Print full-subset RMSE table + improvement vs UVDAR (overlap-only)."""
    n = int(mask.sum())
    n_ov = int((mask & overlap).sum())
    print(f"\n[{label}]   ({n} samples, {n_ov} with UVDAR)")
    print("  RMSE (full subset):")
    _print_rmse_row("UVDAR", _rmse_dict(err_axis_old, err_3d_old, mask))
    if show_raw_nn:
        _print_rmse_row("Network Prediction", _rmse_dict(err_axis_new, err_3d_new, mask))
    if err_axis_ema is not None:
        _print_rmse_row("Filtered Network Prediction",
                        _rmse_dict(err_axis_ema, err_3d_ema, mask))

    ov = mask & overlap
    if ov.sum() == 0:
        print("  (no UVDAR-overlap samples; skipping improvement)")
        return
    print("  Improvement vs UVDAR:")
    base = _rmse_dict(err_axis_old, err_3d_old, ov)
    if show_raw_nn:
        _print_improvement("NN (raw)", base,
                           _rmse_dict(err_axis_new, err_3d_new, ov))
    if err_axis_ema is not None:
        _print_improvement("NN + EMA", base,
                           _rmse_dict(err_axis_ema, err_3d_ema, ov))


def _report_coverage(
    label: str, t: np.ndarray, mask: np.ndarray,
    valid_uvdar: np.ndarray, valid_nn: np.ndarray,
    bin_seconds: float,
    show_raw_nn: bool,
) -> None:
    """Coverage: % of fixed-width time bins with ≥1 valid sample.

    Compares UVDAR vs raw NN only — EMA coverage equals raw NN coverage
    after the first valid sample (state is held through gaps), so it adds
    no information.
    """
    t_sub = t[mask]
    if len(t_sub) == 0:
        return
    uv_sub = valid_uvdar[mask]
    nn_sub = valid_nn[mask]

    cov_uv, n_total = _coverage_by_bins(t_sub, uv_sub, bin_seconds)
    cov_nn, _       = _coverage_by_bins(t_sub, nn_sub, bin_seconds)
    bin_s = bin_seconds

    def pct(c): return 100.0 * c / n_total if n_total else float("nan")

    print(f"\n[{label}]   coverage  bin = {bin_seconds}s, "
          f"total = {n_total} bins ({n_total * bin_s:.1f}s)")
    print(f"  UVDAR    : {cov_uv:6d} / {n_total} bins  "
          f"({cov_uv * bin_s:7.2f}s, {pct(cov_uv):6.2f}%)")
    if show_raw_nn:
        print(f"  NN (raw) : {cov_nn:6d} / {n_total} bins  "
              f"({cov_nn * bin_s:7.2f}s, {pct(cov_nn):6.2f}%)")
        d = cov_nn - cov_uv
        print(f"    NN − UVDAR : {d:+d} bins  "
              f"({d * bin_s:+7.2f}s, {pct(d):+6.2f} pp)")


def visualize_all(
    artifacts: dict,
    show_split: bool = True,
    ema_tau: float | None = None,
    ema_reset_gap: float | None = None,
    hide_raw_nn: bool = False,
    hide_uvdar: bool = False,
    coverage_bin: float = 0.5,
):
    Y_all            = artifacts["Y_all"]
    pred_rel_xyz_all = artifacts["pred_rel_xyz_all"]   # always present (NaN where UVDAR missing)
    pred_res_all     = artifacts["pred_res_all"]
    t_all            = t_to_seconds(artifacts["t_all"])
    val_split        = artifacts["val_split"]
    train_mask, val_mask = split_masks(len(t_all), val_split)

    # Raw GT (full timeline, before feature-validity masking) — used to
    # draw a continuous GT line that stays visible across feature
    # dropouts. Aligned to the same time origin as t_all (first sample).
    gt_raw_t_ns = artifacts.get("gt_raw_t_ns")
    gt_raw_xyz  = artifacts.get("gt_raw_xyz")
    if (gt_raw_t_ns is not None and len(gt_raw_t_ns) > 0
            and len(artifacts["t_all"]) > 0):
        t0_ns = float(artifacts["t_all"][0])
        t_gt_raw = (gt_raw_t_ns.astype(np.float64) - t0_ns) / 1e9
        # Clip to the visible time range so the plot isn't stretched.
        in_range = (t_gt_raw >= t_all[0]) & (t_gt_raw <= t_all[-1])
        t_gt_raw = t_gt_raw[in_range]
        gt_raw_xyz = gt_raw_xyz[in_range]
    else:
        t_gt_raw = None
        gt_raw_xyz = None

    if hide_raw_nn and ema_tau is None:
        raise ValueError("--hide-raw-nn requires --ema-tau to be set")
    show_raw_nn = not hide_raw_nn
    show_uvdar  = not hide_uvdar
    show_ema    = ema_tau is not None
    pred_ema_all = (_ema_filter_time_aware(pred_res_all, t_all, ema_tau,
                                            reset_gap=ema_reset_gap)
                    if show_ema else None)
    if show_ema:
        gap_str = f", reset_gap={ema_reset_gap}s" if ema_reset_gap else ""
        print(f"[visualize] time-aware EMA on NN prediction "
              f"(tau={ema_tau}s{gap_str})")

    n_uvdar = int(np.isfinite(pred_rel_xyz_all).all(axis=1).sum())
    print(f"[visualize] UVDAR baseline available at {n_uvdar}/{len(t_all)} timestamps")

    # Insert NaN at large time gaps so lines break instead of bridging.
    # GT is excluded — we always draw it as a continuous line.
    if show_ema:
        t_plot, pred_old_plot, pred_new_plot, pred_ema_plot = \
            insert_gap_nans(t_all, pred_rel_xyz_all, pred_res_all,
                            pred_ema_all)
    else:
        t_plot, pred_old_plot, pred_new_plot = insert_gap_nans(
            t_all, pred_rel_xyz_all, pred_res_all)
        pred_ema_plot = None

    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

    # Per-axis (x, y, z)
    for i, (ax, axis_label) in enumerate(zip(axs[:3], AXIS_LABELS)):
        if show_split:
            shade_train_val(ax, t_all, val_split)

        # GT is drawn from the raw CSV (full timeline) so it stays
        # visible across feature/UVDAR/NN dropouts. Falls back to the
        # feature-masked Y_all if the raw CSV wasn't available.
        if t_gt_raw is not None:
            ax.plot(t_gt_raw, gt_raw_xyz[:, i],
                    label="Ground Truth", color=GT_COLOR,
                    linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)
        else:
            ax.plot(t_all, Y_all[:, i],
                    label="Ground Truth", color=GT_COLOR,
                    linestyle=GT_LINESTYLE, linewidth=GT_LINEWIDTH, zorder=1)

        mean_gt = np.nanmean(Y_all[:, i])
        # ax.axhline(y=mean_gt, color=GT_COLOR, linestyle="--",
        #            linewidth=1.0, alpha=0.6,
        #            label="GT mean" if i == 0 else None, zorder=1)

        # UVDAR reference: sparse — render as markers so isolated points show.
        if show_uvdar:
            ax.plot(t_plot, pred_old_plot[:, i],
                    label="UVDAR prediction", color=OLD_COLOR,
                    linestyle="none", marker=".", markersize=2,
                    linewidth=SYS_LINEWIDTH, zorder=10)

        if show_raw_nn:
            nn_label = "Network Prediction"
            nn_ls    = "--" if show_ema else NEW_LINESTYLE
            ax.plot(t_plot, pred_new_plot[:, i],
                    label=nn_label, color=NEW_COLOR,
                    linestyle=nn_ls, linewidth=SYS_LINEWIDTH,
                    alpha=RAW_NN_ALPHA, zorder=3)

        if show_ema:
            ax.plot(t_plot, pred_ema_plot[:, i],
                    label=f"Filtered Network Prediction", color=EMA_COLOR,
                    linestyle=EMA_LINESTYLE, linewidth=SYS_LINEWIDTH, zorder=4)

        ax.set_ylabel(f"{axis_label} [m]")
        ax.grid(True)

        if i == 0:
            ax.legend()

    # Euclidean error magnitude
    ax_err = axs[3]
    if show_split:
        shade_train_val(ax_err, t_all, val_split)

    err_mag_new = np.linalg.norm(pred_res_all - Y_all, axis=1)
    err_mag_old = np.linalg.norm(pred_rel_xyz_all - Y_all, axis=1)
    err_mag_ema = (np.linalg.norm(pred_ema_all - Y_all, axis=1)
                   if show_ema else None)
    if show_ema:
        t_err_plot, err_old_plot, err_new_plot, err_ema_plot = \
            insert_gap_nans(t_all, err_mag_old, err_mag_new, err_mag_ema)
    else:
        t_err_plot, err_old_plot, err_new_plot = insert_gap_nans(
            t_all, err_mag_old, err_mag_new)
        err_ema_plot = None

    if show_uvdar:
        ax_err.plot(t_err_plot, err_old_plot,
                    label="UVDAR Prediction Error", color=OLD_COLOR,
                    linestyle="none", marker=".", markersize=2,
                    linewidth=SYS_LINEWIDTH)
    if show_raw_nn:
        nn_err_label = "Network Prediction Error" if show_ema else "Network Prediction Error"
        nn_err_ls    = "-" if show_ema else NEW_LINESTYLE
        ax_err.plot(t_err_plot, err_new_plot,
                    label=nn_err_label, color=NEW_COLOR,
                    linestyle=nn_err_ls, linewidth=2,
                    alpha=RAW_NN_ALPHA)
    if show_ema:
        ax_err.plot(t_err_plot, err_ema_plot,
                    label=f"Filtered Network Prediction Error",
                    color=EMA_COLOR,
                    linestyle=EMA_LINESTYLE, linewidth=SYS_LINEWIDTH)

    ax_err.set_ylabel("Error (Euclidean) [m]")
    ax_err.grid(True)
    ax_err.legend()

    axs[-1].set_xlabel("Time [s]")
    fig.tight_layout()

    # ── Console report: RMSE table + improvements (UVDAR-overlap) ─────
    err_axis_old = {"x": pred_rel_xyz_all[:, 0] - Y_all[:, 0],
                    "y": pred_rel_xyz_all[:, 1] - Y_all[:, 1],
                    "z": pred_rel_xyz_all[:, 2] - Y_all[:, 2]}
    err_axis_new = {"x": pred_res_all[:, 0] - Y_all[:, 0],
                    "y": pred_res_all[:, 1] - Y_all[:, 1],
                    "z": pred_res_all[:, 2] - Y_all[:, 2]}
    err_axis_ema = ({"x": pred_ema_all[:, 0] - Y_all[:, 0],
                     "y": pred_ema_all[:, 1] - Y_all[:, 1],
                     "z": pred_ema_all[:, 2] - Y_all[:, 2]}
                    if show_ema else None)
    err_3d_ema_arg = err_mag_ema if show_ema else None
    overlap = np.isfinite(pred_rel_xyz_all).all(axis=1)

    print("\n========== RMSE & Improvement ==========")
    print("  per-axis RMSE = sqrt(mean( (pred_axis − gt_axis)^2 )) over "
          "finite samples in the subset")
    print("  3D RMSE       = sqrt(mean( ||pred − gt||² )) over finite "
          "samples in the subset")
    print("  Improvement % = (1 − RMSE_model / RMSE_UVDAR) * 100, computed "
          "only on samples where UVDAR is also available")
    if show_split:
        _report_subset("train", train_mask,
                       err_axis_old, err_mag_old,
                       err_axis_new, err_mag_new,
                       err_axis_ema, err_3d_ema_arg,
                       overlap, show_raw_nn)
        _report_subset("val", val_mask,
                       err_axis_old, err_mag_old,
                       err_axis_new, err_mag_new,
                       err_axis_ema, err_3d_ema_arg,
                       overlap, show_raw_nn)
    else:
        all_mask = np.ones(len(t_all), dtype=bool)
        _report_subset("all", all_mask,
                       err_axis_old, err_mag_old,
                       err_axis_new, err_mag_new,
                       err_axis_ema, err_3d_ema_arg,
                       overlap, show_raw_nn)

    # ── Console report: coverage by time-bins ─────────────────────────
    print("\n========== Coverage ==========")
    print(f"  Each bin is {coverage_bin}s wide. A bin counts as covered "
          "if it contains ≥1 sample with finite predictions.")
    valid_uv = np.isfinite(pred_rel_xyz_all).all(axis=1)
    valid_nn = np.isfinite(pred_res_all).all(axis=1)
    if show_split:
        _report_coverage("train", t_all, train_mask,
                         valid_uv, valid_nn,
                         coverage_bin, show_raw_nn)
        _report_coverage("val", t_all, val_mask,
                         valid_uv, valid_nn,
                         coverage_bin, show_raw_nn)
    else:
        _report_coverage("all", t_all, np.ones(len(t_all), dtype=bool),
                         valid_uv, valid_nn,
                         coverage_bin, show_raw_nn)
    print()

    _button = _attach_save_button(fig)  # noqa: F841 (kept alive by ref)
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
    ap.add_argument(
        "--ema-tau", type=float, default=None, metavar="SECONDS",
        help="Apply causal time-aware EMA to NN prediction with time "
             "constant TAU in seconds. Tuned for ~30–60 Hz; try 0.05–0.15.",
    )
    ap.add_argument(
        "--ema-reset-gap", type=float, default=0.5, metavar="SECONDS",
        help="Re-seed the EMA state if the gap to the previous valid "
             "sample exceeds this many seconds (default: 0.5). Use 0 to "
             "disable.",
    )
    ap.add_argument(
        "--hide-raw-nn", action="store_true",
        help="Hide the raw NN prediction trace; only show the EMA-filtered "
             "one. Requires --ema-tau.",
    )
    ap.add_argument(
        "--hide-uvdar", action="store_true",
        help="Hide the UVDAR baseline traces (per-axis points and error).",
    )
    ap.add_argument(
        "--coverage-bin", type=float, default=0.5, metavar="SECONDS",
        help="Bin width (seconds) used for the coverage report. "
             "Default: 0.5.",
    )
    args = ap.parse_args()

    data = load_run(args.results_dir, args.run_dir_override)

    # If the user pointed at a different dataset, the original train/val
    # split has no meaning — hide it and report a single overall RMSE.
    visualize_all(
        data,
        show_split=args.run_dir_override is None,
        ema_tau=args.ema_tau,
        ema_reset_gap=args.ema_reset_gap if args.ema_reset_gap > 0 else None,
        hide_raw_nn=args.hide_raw_nn,
        hide_uvdar=args.hide_uvdar,
        coverage_bin=args.coverage_bin,
    )


if __name__ == "__main__":
    main()
