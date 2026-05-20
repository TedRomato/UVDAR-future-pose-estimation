"""Numerical SE(3) example: two coordinate frames in 3D.

Frame A is at the world origin with no rotation.
Frame B is a copy of A translated by a vector ``t`` and rotated by a yaw
angle around the world Z axis. The translation vector itself is drawn
between the two origins so the SE(3) action is visible.

Visual style matches ``tf_tree_visualizer.py``: dark background,
red/green/blue axes drawn with cone-tipped arrows, light labels and a
clean XY-plane grid. The only stylistic difference is that this figure
keeps the X and Y tick labels visible (Z is hidden) so the numerical
translation can be read off the grid.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- visual style (matches tf_tree_visualizer) ---------------------
AXIS_COLORS        = ("tab:red", "tab:green", "tab:blue")  # X, Y, Z
TRANSLATION_COLOR  = "#f38ba8"

FIG_FACECOLOR      = "#1e1e2e"
AX_FACECOLOR       = "#1e1e2e"
LABEL_COLOR        = "#cdd6f4"
TEXT_OUTLINE_COLOR = "#000000"
TEXT_OUTLINE_WIDTH = 3.0

GRID_COLOR         = "#3a3a4a"
GRID_LINEWIDTH     = 0.6
GRID_AXIS_COLOR    = "#6b6b80"
GRID_AXIS_LW       = 1.0


# ---- SE(3) helpers --------------------------------------------------
def yaw_matrix(yaw_rad: float) -> np.ndarray:
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = t
    return T


# ---- drawing helpers (mirror tf_tree_visualizer) -------------------
def _outlined(text_obj):
    text_obj.set_path_effects([
        pe.withStroke(linewidth=TEXT_OUTLINE_WIDTH,
                      foreground=TEXT_OUTLINE_COLOR),
        pe.Normal(),
    ])
    return text_obj


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r1 = np.cross(d, ref); r1 /= np.linalg.norm(r1)
    r2 = np.cross(d, r1)
    return r1, r2


def _draw_cone(ax, tip: np.ndarray, direction: np.ndarray,
               cone_length: float, color: str, n_pts: int = 24) -> None:
    d = direction / np.linalg.norm(direction)
    cone_radius = cone_length * 0.4
    base_center = tip - d * cone_length
    r1, r2 = _perp_basis(d)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    t = np.array([0.0, 1.0])
    T, Theta = np.meshgrid(t, theta)
    R = cone_radius * (1 - T)
    X = base_center[0] + T * cone_length * d[0] + R * (np.cos(Theta) * r1[0] + np.sin(Theta) * r2[0])
    Y = base_center[1] + T * cone_length * d[1] + R * (np.cos(Theta) * r1[1] + np.sin(Theta) * r2[1])
    Z = base_center[2] + T * cone_length * d[2] + R * (np.cos(Theta) * r1[2] + np.sin(Theta) * r2[2])
    ax.plot_surface(X, Y, Z, color=color, shade=True, alpha=0.95, linewidth=0)


def _draw_arrow(ax, p_start: np.ndarray, p_end: np.ndarray,
                color: str, diameter: float,
                arrow_size: float | None = None) -> None:
    """Straight shaft (no built-in head) + cone tip — same as tf_tree_visualizer."""
    d = p_end - p_start
    length = float(np.linalg.norm(d))
    if length == 0:
        return
    direction = d / length
    cone_len = arrow_size if arrow_size is not None else length * 0.15
    cone_len = min(cone_len, length * 0.9)
    shaft_end = p_end - direction * cone_len
    shaft_d = shaft_end - p_start
    ax.quiver(p_start[0], p_start[1], p_start[2],
              shaft_d[0], shaft_d[1], shaft_d[2],
              color=color, linewidth=diameter, arrow_length_ratio=0)
    _draw_cone(ax, p_end, direction, cone_len, color)


def _draw_axis_shaft(ax, origin: np.ndarray, direction: np.ndarray,
                     length: float, color: str, diameter: float) -> None:
    d = direction * length
    ax.quiver(origin[0], origin[1], origin[2], d[0], d[1], d[2],
              color=color, linewidth=diameter, arrow_length_ratio=0)


def _draw_frame(ax, T: np.ndarray, *, axis_length: float, axis_diameter: float,
                name: str, label_side: str = "right") -> None:
    o = T[:3, 3]
    R = T[:3, :3]
    for i, color in enumerate(AXIS_COLORS):
        _draw_axis_shaft(ax, o, R[:, i], axis_length, color, axis_diameter)

    # Axis tip labels at the end of each coloured ray.
    for i, axis_name in enumerate(("X", "Y", "Z")):
        tip = o + R[:, i] * axis_length * 1.15
        _outlined(ax.text(tip[0], tip[1], tip[2], axis_name,
                          color=AXIS_COLORS[i], fontsize=10,
                          ha="center", va="center"))

    # Frame name (matches tf_tree_visualizer offset).
    label_text = f"  {name}" if label_side == "right" else f"{name}  "
    ha = "left" if label_side == "right" else "right"
    _outlined(ax.text(o[0], o[1], o[2] + 0.04 * axis_length / 0.15,
                      label_text, color=LABEL_COLOR, fontsize=11,
                      ha=ha, zorder=100))


def _draw_translation(ax, p_from: np.ndarray, p_to: np.ndarray, *,
                      label: str, diameter: float = 2.5,
                      arrow_size: float = 0.18) -> None:
    _draw_arrow(ax, p_from, p_to, color=TRANSLATION_COLOR,
                diameter=diameter, arrow_size=arrow_size)
    mid = p_from + 0.5 * (p_to - p_from)
    _outlined(ax.text(mid[0], mid[1], mid[2] + 0.08, label,
                      color=TRANSLATION_COLOR, fontsize=11,
                      ha="center", va="bottom"))


def _draw_grid(ax, half: float, step: float, z: float = 0.0,
               draw_labels: bool = True) -> None:
    """XY-plane grid centred at origin (matches tf_tree_visualizer)."""
    xs = np.arange(-half, half + step * 0.5, step)
    ys = xs
    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], [z, z],
                color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], [z, z],
                color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)
    ax.plot([xs[0], xs[-1]], [0, 0], [z, z],
            color=GRID_AXIS_COLOR, linewidth=GRID_AXIS_LW, zorder=1)
    ax.plot([0, 0], [ys[0], ys[-1]], [z, z],
            color=GRID_AXIS_COLOR, linewidth=GRID_AXIS_LW, zorder=1)

    if draw_labels:
        # Tick labels anchored to fixed 3D points on the grid plane so they
        # don't drift around when matplotlib re-projects on hover/zoom.
        offset = step * 0.25
        for x in xs:
            if x == 0:
                continue
            _outlined(ax.text(x, -offset, z, f"{x:g}",
                              color=LABEL_COLOR, fontsize=9,
                              ha="center", va="top", zorder=2))
        for y in ys:
            if y == 0:
                continue
            _outlined(ax.text(-offset, y, z, f"{y:g}",
                              color=LABEL_COLOR, fontsize=9,
                              ha="right", va="center", zorder=2))
        # Single "0" at the origin.
        _outlined(ax.text(-offset, -offset, z, "0",
                          color=LABEL_COLOR, fontsize=9,
                          ha="right", va="top", zorder=2))
        # Axis name labels at the far end of each axis.
        _outlined(ax.text(xs[-1] + offset, 0, z, "X [m]",
                          color=LABEL_COLOR, fontsize=10,
                          ha="left", va="center", zorder=2))
        _outlined(ax.text(0, ys[-1] + offset, z, "Y [m]",
                          color=LABEL_COLOR, fontsize=10,
                          ha="center", va="bottom", zorder=2))


def _style_axes(ax, *, lo: np.ndarray, hi: np.ndarray) -> None:
    """tf_tree_visualizer-style 3D axes: panes/spines/ticks all hidden."""
    ax.set_facecolor(AX_FACECOLOR)

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis.pane.set_edgecolor("none")
        axis.line.set_color("none")

    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        span_xy = hi[0] - lo[0]
        span_z  = hi[2] - lo[2]
        ax.set_box_aspect((1.0, 1.0, span_z / span_xy))
    except Exception:
        pass

    # Hide all built-in tick labels — we draw our own as 3D text.
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_zticks([0])
    for axis_name in ("x", "y", "z"):
        ax.tick_params(axis=axis_name, color="none", labelcolor="none", length=0)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")

    ax.grid(False)


def _add_save_button(fig, basename):
    save_ax = fig.add_axes([0.88, 0.02, 0.10, 0.05])
    button = Button(save_ax, "Save")

    def _save(event):
        save_ax.set_visible(False)
        for ext in ("pgf", "pdf", "svg"):
            fig.savefig(f"{basename}.{ext}", bbox_inches="tight",
                        pad_inches=0.02, facecolor=fig.get_facecolor())
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {basename}.{{pgf,pdf,svg}}")

    button.on_clicked(_save)
    fig._save_button = button
    fig._save_ax = save_ax


# ---- main figure ----------------------------------------------------
def draw_se3_example(*, t: np.ndarray, yaw_deg: float,
                     axis_length: float = 0.6,
                     axis_diameter: float = 4.0,
                     a_name: str = "A",
                     b_name: str = "B",
                     translation_label: str | None = None):
    t = np.asarray(t, float).reshape(3)
    yaw = np.deg2rad(yaw_deg)
    R = yaw_matrix(yaw)
    T_AB = se3(R, t)

    fig = plt.figure(figsize=(8, 7))
    fig.patch.set_facecolor(FIG_FACECOLOR)
    ax = fig.add_subplot(111, projection="3d")

    span = max(axis_length * 2.0,
               float(np.max(np.abs(t))) + axis_length * 1.5,
               1.0)
    span = 2
    half = float(np.ceil(span))
    step = 0.5 if half <= 3.0 else 1.0
    ticks = np.arange(-half, half + step * 0.5, step)

    lo = np.array([-half, -half, 0.0])
    hi = np.array([ half,  half,  half])

    _style_axes(ax, lo=lo, hi=hi)
    _draw_grid(ax, half=half, step=step, z=0.0)

    _draw_frame(ax, np.eye(4), axis_length=axis_length,
                axis_diameter=axis_diameter, name=a_name)
    _draw_frame(ax, T_AB, axis_length=axis_length,
                axis_diameter=axis_diameter, name=b_name)

    if translation_label is None:
        translation_label = (rf"$t = ({t[0]:.2f},\ {t[1]:.2f},\ {t[2]:.2f})$"
                             rf"  $\psi = {yaw_deg:.0f}^\circ$")
    _draw_translation(ax, np.zeros(3), t, label=translation_label)

    title = " " or (rf"$T_{{{a_name}{b_name}}} \in \mathrm{{SE}}(3)$:  "
             rf"translate by $t$, rotate by yaw $\psi$")
    _outlined(fig.text(0.5, 0.95, title, color=LABEL_COLOR,
                       ha="center", va="top", fontsize=14))

    ax.view_init(elev=22, azim=-60)
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.08)
    _add_save_button(fig, "se3_example")

    np.set_printoptions(precision=3, suppress=True)
    print("R (yaw {:.1f} deg) =\n{}".format(yaw_deg, R))
    print("t = {}".format(t))
    print("T_{}{} =\n{}".format(a_name, b_name, T_AB))

    return fig


if __name__ == "__main__":
    T_VECTOR = np.array([1.5, 0.8, 0.4])
    YAW_DEG  = 35.0

    draw_se3_example(t=T_VECTOR, yaw_deg=YAW_DEG)
    plt.show()
