"""Step-by-step visualization of how the camera FOV is constructed
from two image-corner observations (top-left and bottom-right) made
from a single observer position.

Algorithm (matches the comments in the panels):
  1. Form unit vectors v_TL, v_BR from the observer towards the two
     observed corner points.
  2. The optical axis is the normalized sum  a = (v_TL + v_BR) / ||.||.
     Because we assume zero roll, the camera "up" is world +Z; the
     mirroring plane is the vertical plane that contains a and +Z.
  3. Reflect v_TL and v_BR across that plane to obtain the other
     two FOV corner directions  v_TR, v_BL.
  4. Place a rectangle at distance d along the optical axis,
     perpendicular to a, whose corners lie on the four direction rays.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- shared style ---------------------------------------------------
CORNER_COLORS = {
    "TL": "tab:red",
    "BR": "tab:blue",
    "TR": "tab:purple",
    "BL": "tab:cyan",
}
AXIS_COLOR  = "tab:orange"
PLANE_COLOR = "cyan"
RECT_COLOR  = "tab:green"


# ---- helpers --------------------------------------------------------
def _normalize(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _mirror_across_plane(v, n):
    """Reflect vector ``v`` across the plane with unit normal ``n``."""
    v = np.asarray(v, float)
    n = np.asarray(n, float)
    return v - 2.0 * float(v @ n) * n


def _draw_camera_marker(ax, p):
    ax.scatter([p[0]], [p[1]], [p[2]],
               color="black", s=70, marker="s", depthshade=False, zorder=6)


def _draw_arrow(ax, start, end, color, lw=2.0, label=None):
    s = np.asarray(start, float)
    e = np.asarray(end,   float)
    d = e - s
    ax.quiver(
        s[0], s[1], s[2],
        d[0], d[1], d[2],
        color=color, linewidth=lw,
        arrow_length_ratio=0.15,
        label=label,
    )


def _draw_point(ax, p, color, label=None, marker="o"):
    ax.scatter([p[0]], [p[1]], [p[2]],
               color=color, s=55, marker=marker, edgecolors="k",
               linewidths=0.6, depthshade=False, label=label)


def _setup_panel(ax, title, lim):
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title, fontsize=10)
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    try:
        ax.set_box_aspect((lim[0][1]-lim[0][0],
                           lim[1][1]-lim[1][0],
                           lim[2][1]-lim[2][0]))
    except Exception:
        pass


def _draw_plane(ax, point, normal, *, u_extent=4.0, v_extent=4.0,
                color=PLANE_COLOR, alpha=0.18):
    """Filled rectangle representing a plane patch through ``point``
    with normal ``normal``. Two in-plane basis vectors are derived
    automatically."""
    n = _normalize(normal)
    # Pick any vector not parallel to n to build u.
    helper = np.array([0.0, 0.0, 1.0]) if abs(n[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = _normalize(np.cross(n, helper))
    v = _normalize(np.cross(n, u))
    p = np.asarray(point, float)
    corners = np.array([
        p + u*u_extent + v*v_extent,
        p - u*u_extent + v*v_extent,
        p - u*u_extent - v*v_extent,
        p + u*u_extent - v*v_extent,
    ])
    poly = Poly3DCollection([corners], facecolor=color, edgecolor=color,
                            alpha=alpha, linewidths=1.0)
    ax.add_collection3d(poly)


def _ray_to_distance_along_axis(direction, axis, distance):
    """Return the point along ``direction`` (from origin) such that
    its projection onto ``axis`` equals ``distance``."""
    d = float(np.asarray(direction) @ np.asarray(axis))
    return np.asarray(direction, float) * (distance / d)


# ---- main visualization --------------------------------------------
def draw_fov_construction(
    *,
    observer=(0.0, 0.0, 0.0),
    top_left=(8.0,  3.0, 4.0),
    bottom_right=(8.0, -4.0, -2.0),
    rect_distance=7.0,
):
    O   = np.asarray(observer,     float)
    TL  = np.asarray(top_left,     float)
    BR  = np.asarray(bottom_right, float)

    # Step 1: unit direction vectors from observer to the two corners.
    v_TL = _normalize(TL - O)
    v_BR = _normalize(BR - O)

    # Step 2: optical axis & mirroring plane (zero roll => contains world +Z).
    axis = _normalize(v_TL + v_BR)
    # Plane normal is horizontal, perpendicular to axis's XY projection,
    # i.e. perpendicular to both axis and world-Z.
    z_world = np.array([0.0, 0.0, 1.0])
    plane_normal = _normalize(np.cross(axis, z_world))

    # Step 3: mirror to obtain the other two FOV corner directions.
    v_TR = _mirror_across_plane(v_TL, plane_normal)
    v_BL = _mirror_across_plane(v_BR, plane_normal)

    # Step 4: rectangle at distance d along the axis, perpendicular to axis.
    rect_TL = O + _ray_to_distance_along_axis(v_TL, axis, rect_distance)
    rect_BR = O + _ray_to_distance_along_axis(v_BR, axis, rect_distance)
    rect_TR = O + _ray_to_distance_along_axis(v_TR, axis, rect_distance)
    rect_BL = O + _ray_to_distance_along_axis(v_BL, axis, rect_distance)
    rect_corners = np.array([rect_TL, rect_TR, rect_BR, rect_BL, rect_TL])

    # ---- common axis limits across all panels (so the eye can compare).
    all_pts = np.vstack([
        O, TL, BR,
        O + v_TL * rect_distance, O + v_BR * rect_distance,
        O + v_TR * rect_distance, O + v_BL * rect_distance,
        rect_corners,
    ])
    pad = 1.0
    lim = [(all_pts[:, i].min() - pad, all_pts[:, i].max() + pad)
           for i in range(3)]

    # ---- figure layout ------------------------------------------------
    fig = plt.figure(figsize=(15, 11))
    fig.suptitle("FOV construction from two image-corner observations",
                 fontsize=13)
    axes = [fig.add_subplot(2, 2, i + 1, projection="3d") for i in range(4)]

    L_TL = "Normalized top left observation"
    L_BR = "Normalized bottom right observation"

    # ---- Panel 1: lines fitted to TL & BR, then normalized -----------
    ax = axes[0]
    _draw_camera_marker(ax, O)
    _draw_point(ax, TL, CORNER_COLORS["TL"], label="Top-left observation")
    _draw_point(ax, BR, CORNER_COLORS["BR"], label="Bottom-right observation")
    # Raw rays from O to the observed points (faint).
    ax.plot([O[0], TL[0]], [O[1], TL[1]], [O[2], TL[2]],
            color=CORNER_COLORS["TL"], lw=0.9, linestyle=":")
    ax.plot([O[0], BR[0]], [O[1], BR[1]], [O[2], BR[2]],
            color=CORNER_COLORS["BR"], lw=0.9, linestyle=":")
    # Unit vectors (the actual outputs of step 1).
    _draw_arrow(ax, O, O + v_TL, CORNER_COLORS["TL"], label=L_TL)
    _draw_arrow(ax, O, O + v_BR, CORNER_COLORS["BR"], label=L_BR)
    ax.scatter([], [], color="black", s=70, marker="s", label="Camera")
    _setup_panel(ax, "1) Fit rays to corner observations and normalize", lim)
    ax.legend(loc="upper left", fontsize=8)

    # ---- Panel 2: optical axis + mirroring plane ---------------------
    ax = axes[1]
    _draw_camera_marker(ax, O)
    _draw_arrow(ax, O, O + v_TL, CORNER_COLORS["TL"], lw=1.4)
    _draw_arrow(ax, O, O + v_BR, CORNER_COLORS["BR"], lw=1.4)
    # Optical axis = normalize(v_TL + v_BR).
    axis_len = 1.5
    _draw_arrow(ax, O, O + axis * axis_len, AXIS_COLOR, lw=2.5,
                label="Optical axis")
    # Mirroring plane through O containing the axis and world +Z.
    extent = 1
    _draw_plane(ax, O + axis * (axis_len * 0.5), plane_normal,
                u_extent=extent, v_extent=extent)
    ax.plot([], [], color=PLANE_COLOR, lw=4, alpha=0.5,
            label="Mirroring plane")
    ax.scatter([], [], color="black", s=70, marker="s", label="Camera")
    _setup_panel(ax, "2) Construct optical axis and mirroring plane", lim)
    ax.legend(loc="upper left", fontsize=8)

    # ---- Panel 3: mirror v_TL and v_BR -> v_TR and v_BL --------------
    ax = axes[2]
    _draw_camera_marker(ax, O)
    _draw_arrow(ax, O, O + axis * axis_len, AXIS_COLOR, lw=1.6)
    _draw_plane(ax, O + axis * (axis_len * 0.5), plane_normal,
                u_extent=extent, v_extent=extent, alpha=0.10)
    _draw_arrow(ax, O, O + v_TL, CORNER_COLORS["TL"], lw=2.0, label="Top left")
    _draw_arrow(ax, O, O + v_BR, CORNER_COLORS["BR"], lw=2.0, label="Bottom right")
    _draw_arrow(ax, O, O + v_TR, CORNER_COLORS["TR"], lw=2.0,
                label="Top right")
    _draw_arrow(ax, O, O + v_BL, CORNER_COLORS["BL"], lw=2.0,
                label="Bottom left")
    # Dotted "mirror lines" connecting each vector tip to its image.
    for a_, b_ in [(v_TL, v_TR), (v_BR, v_BL)]:
        p1 = O + a_; p2 = O + b_
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color="grey", lw=1.5, linestyle="--")
    ax.plot([], [], color=AXIS_COLOR, lw=2.0, label="Optical axis")
    ax.plot([], [], color=PLANE_COLOR, lw=4, alpha=0.5, label="Mirroring plane")
    ax.scatter([], [], color="black", s=70, marker="s", label="Camera")
    _setup_panel(ax, "3) Mirror edge vectors across the plane", lim)
    ax.legend(loc="upper left", fontsize=8)

    # ---- Panel 4: final FOV with rectangle at distance d -------------
    _EDGE_COLOR = "#666666"
    ax = axes[3]
    _draw_camera_marker(ax, O)
    _draw_arrow(ax, O, O + axis * rect_distance, AXIS_COLOR, lw=2.0,
                label=f"Optical axis")
    # FOV pyramid edges: O to each rectangle corner.
    for v_, key in [(v_TL, "TL"), (v_TR, "TR"), (v_BR, "BR"), (v_BL, "BL")]:
        end = O + _ray_to_distance_along_axis(v_, axis, rect_distance)
        ax.plot([O[0], end[0]], [O[1], end[1]], [O[2], end[2]],
                color=_EDGE_COLOR, lw=1.6)
    # Rectangle perpendicular to optical axis (no filled plane).
    ax.plot(rect_corners[:, 0], rect_corners[:, 1], rect_corners[:, 2],
            color=_EDGE_COLOR, lw=2.0,
            label=f"FOV geometry")
    # Corner labels (text only, no point markers).

    ax.plot([], [], color=AXIS_COLOR, lw=2.0, label="Optical axis")
    ax.plot([], [], color=_EDGE_COLOR, lw=1.6, label="FOV edges")
    ax.scatter([], [], color="black", s=70, marker="s", label="Camera")
    _setup_panel(ax, "4) Construct FOV", lim)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


if __name__ == "__main__":
    OBSERVER     = (0.0, 0.0, 5.0)
    TOP_LEFT     = (0.8,  0.6, 5.8)   # close corner
    BOTTOM_RIGHT = (1.2, -0.7, 4.4)   # far corner
    RECT_DIST    = 1.0

    fig = draw_fov_construction(
        observer=OBSERVER,
        top_left=TOP_LEFT,
        bottom_right=BOTTOM_RIGHT,
        rect_distance=RECT_DIST,
    )

    def _save(event):
        save_ax.set_visible(False)
        fig.savefig("fov_construction.pgf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig("fov_construction.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig("fov_construction.svg", bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print("Saved fov_construction.{pgf,pdf,svg}")

    save_ax = fig.add_axes([0.82, 0.02, 0.12, 0.05])
    save_button = Button(save_ax, "Save")
    save_button.on_clicked(_save)

    plt.show()
