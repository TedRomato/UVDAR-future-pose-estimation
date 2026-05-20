"""Single-figure illustration of linear-interpolation error on a smooth
nonlinear UAV vertical motion.

Left  panel : coarse temporal sampling -> visible interpolation error.
Right panel : fine   temporal sampling -> almost no interpolation error,
              with a zoomed inset around the same error point.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- the smooth ground-truth motion --------------------------------
def z_true(t):
    """Smooth ascending UAV altitude, starting at 0, leveling off near 1."""
    return -np.exp(-t) + 1.0


# ---- colors ---------------------------------------------------------
TRUE_COLOR     = "tab:blue"
SAMPLE_COLOR   = "tab:orange"
INTERP_COLOR   = "tab:orange"
ERROR_COLOR    = "tab:red"
NEAREST_COLOR  = "tab:purple"


# ---- error highlight time (same in both panels) --------------------
ERROR_T = 2.0


def _draw_error_marker(ax, error_t, z_a, z_b, *, color=ERROR_COLOR,
                       label=None, cap_size=10, x_offset=0.0):
    """Vertical dashed line + horizontal end-caps between two z-values
    at (``error_t + x_offset``)."""
    x = error_t + x_offset
    ax.vlines(x, min(z_a, z_b), max(z_a, z_b),
              color=color, linestyle="--", linewidth=1.8,
              label=label)
    for z in (z_a, z_b):
        ax.plot([x], [z], marker="_", color=color,
                markersize=cap_size, markeredgewidth=1.8)


def _nearest_sample_value(error_t, t_samples, z_samples):
    """Value at the sample closest in time to ``error_t``."""
    idx = int(np.argmin(np.abs(t_samples - error_t)))
    return float(t_samples[idx]), float(z_samples[idx])


def _draw_panel(ax, *, t_grid, t_samples, error_t,
                interp_label, nearest_label):
    z_grid    = z_true(t_grid)
    z_samples = z_true(t_samples)

    ax.plot(t_grid, z_grid, color=TRUE_COLOR, linewidth=2.0,
            label="True trajectory")
    ax.plot(t_samples, z_samples, color=INTERP_COLOR, linewidth=1.4,
            linestyle="-", label="Linear interpolation")
    ax.plot(t_samples, z_samples, marker="o", linestyle="none",
            color=SAMPLE_COLOR, markersize=6.0, markeredgecolor="black",
            markeredgewidth=0.6, label="Odometry samples")

    z_at_true   = z_true(error_t)
    z_at_interp = float(np.interp(error_t, t_samples, z_samples))
    _draw_error_marker(ax, error_t, z_at_true, z_at_interp,
                       color=ERROR_COLOR, label=interp_label,
                       x_offset=-0.04)

    # Nearest-sample (discretization) error: snap to the closest sample.
    t_near, z_near = _nearest_sample_value(error_t, t_samples, z_samples)
    ax.plot([t_near, error_t], [z_near, z_near],
            color=NEAREST_COLOR, linestyle=":", linewidth=1.4)
    _draw_error_marker(ax, error_t, z_at_true, z_near,
                       color=NEAREST_COLOR, label=nearest_label,
                       x_offset=+0.04)

    ax.set_xlabel(r"Time $t$ [s]")
    ax.set_ylabel(r"Vertical position $z$ [m]")
    ax.grid(True, alpha=0.3)


def _add_save_button(fig, basename):
    save_ax = fig.add_axes([0.88, 0.02, 0.10, 0.05])
    button = Button(save_ax, "Save")

    def _save(event):
        save_ax.set_visible(False)
        fig.savefig(f"{basename}.pgf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{basename}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{basename}.svg", bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {basename}.{{pgf,pdf,svg}}")

    button.on_clicked(_save)
    fig._save_button = button
    fig._save_ax = save_ax
    return button


def _add_zoom_inset(fig, ax, t_grid, t_samples, *, dt=0.45,
                    inset_w_frac=0.45, inset_h_frac=0.45):
    """Add a zoom inset that preserves the parent's data aspect ratio."""
    fig.canvas.draw()  # finalise parent axis limits
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    parent_data_aspect = (y1 - y0) / (x1 - x0)

    bbox = ax.get_position()
    inset_w_in = inset_w_frac * bbox.width  * fig.get_size_inches()[0]
    inset_h_in = inset_h_frac * bbox.height * fig.get_size_inches()[1]
    inset_pixel_aspect = inset_h_in / inset_w_in

    dz = dt * parent_data_aspect * inset_pixel_aspect
    z_center = z_true(ERROR_T)

    zoom = inset_axes(ax, width=f"{inset_w_frac*100:.0f}%",
                      height=f"{inset_h_frac*100:.0f}%",
                      loc="center right", borderpad=2.0)
    zoom.plot(t_grid, z_true(t_grid), color=TRUE_COLOR, linewidth=2.0)
    zoom.plot(t_samples, z_true(t_samples), color=INTERP_COLOR, linewidth=1.4)
    zoom.plot(t_samples, z_true(t_samples), marker="o", linestyle="none",
              color=SAMPLE_COLOR, markersize=5.0,
              markeredgecolor="black", markeredgewidth=0.5)

    zoom.set_xlim(ERROR_T - dt, ERROR_T + dt)
    zoom.set_ylim(z_center - dz, z_center + dz)
    zoom.set_xticklabels([])
    zoom.set_yticklabels([])
    zoom.tick_params(length=0)
    zoom.grid(True, alpha=0.3)

    z_at_true_z   = z_true(ERROR_T)
    z_at_interp_z = float(np.interp(ERROR_T, t_samples, z_true(t_samples)))
    _draw_error_marker(zoom, ERROR_T, z_at_true_z, z_at_interp_z,
                       color=ERROR_COLOR, cap_size=14, x_offset=-dt*0.06)

    t_near_z, z_near_z = _nearest_sample_value(
        ERROR_T, t_samples, z_true(t_samples))
    zoom.plot([t_near_z, ERROR_T], [z_near_z, z_near_z],
              color=NEAREST_COLOR, linestyle=":", linewidth=1.4)
    _draw_error_marker(zoom, ERROR_T, z_at_true_z, z_near_z,
                       color=NEAREST_COLOR, cap_size=14,
                       x_offset=+dt*0.06)

    ax.add_patch(plt.Rectangle((ERROR_T - dt, z_center - dz),
                               2.0 * dt, 2.0 * dz,
                               fill=False, edgecolor="0.5", linewidth=0.8))
    return zoom


def draw_coarse(t_grid, t_coarse):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    _draw_panel(
        ax,
        t_grid=t_grid,
        t_samples=t_coarse,
        error_t=ERROR_T,
        interp_label="Interpolation error",
        nearest_label="Discretization error",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    _add_zoom_inset(fig, ax, t_grid, t_coarse)
    _add_save_button(fig, "interpolation_error_coarse")
    return fig


def draw_fine(t_grid, t_fine):
    fig, ax = plt.subplots(figsize=(7.0, 5.2))
    _draw_panel(
        ax,
        t_grid=t_grid,
        t_samples=t_fine,
        error_t=ERROR_T,
        interp_label="Interpolation error",
        nearest_label="Discretization error",
    )
    ax.legend(loc="lower right")
    fig.tight_layout()
    _add_zoom_inset(fig, ax, t_grid, t_fine)
    _add_save_button(fig, "interpolation_error_fine")
    return fig


if __name__ == "__main__":
    t_grid   = np.linspace(0.0, 5.0, 600)
    t_coarse = np.array([0.0, 0.6, 1.4, 2.6, 4.2, 5.0])
    t_fine   = np.linspace(0.0, 5.0, 40)

    draw_coarse(t_grid, t_coarse)
    draw_fine(t_grid, t_fine)
    plt.show()
