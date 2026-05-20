"""Interactive 3D GUI: two UAV coordinate systems + a gold arrow between
them, drawn over a static background image.

The background image is rendered on a separate, non-interactive 2D axes
sitting *behind* a transparent 3D axes.  Rotating / zooming the 3D scene
leaves the background completely untouched.

Usage::

    python3 helper_visualizations/two_uav_arrow_gui.py --image path/to/bg.png

Sliders below the plot let you set X, Y, Z and yaw for each UAV.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Button, Slider
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ---- Style (mirrors tf_tree_visualizer.py) --------------------------------
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")  # X, Y, Z
LINK_COLOR = "gold"
ARROW_COLOR = LINK_COLOR
ARROW_DIAMETER = 2.5
ARROW_HEAD_FRACTION = 0.15

UAV_AXIS_LENGTH = 1.0
UAV_AXIS_DIAMETER = 3.0

FIG_FACECOLOR = "#1e1e2e"
LABEL_COLOR = "#cdd6f4"
TEXT_OUTLINE_COLOR = "#000000"
TEXT_OUTLINE_WIDTH = 3
SLIDER_TRACK_COLOR = "#313244"
SLIDER_HANDLE_COLOR = "#89b4fa"

# 3D world bounds (independent of the background image).
WORLD_HALF = 20.0
WORLD_Z_MAX = 20.0

# Reference grid drawn on the z = 0 plane (translucent, hidden on save).
GRID_COLOR = "#6c7086"
GRID_ALPHA = 0.35
GRID_LINEWIDTH = 0.6
GRID_STEP = 2.0

# Per-slider shift buttons (−/+) pan the slider window by this amount.
SLIDER_SHIFT = 15.0


def _outlined_text(text_obj) -> None:
    text_obj.set_path_effects([
        pe.withStroke(linewidth=TEXT_OUTLINE_WIDTH, foreground=TEXT_OUTLINE_COLOR),
        pe.Normal(),
    ])


@dataclass
class UAVState:
    name: str
    x: float
    y: float
    z: float
    yaw: float


def _yaw_R(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])


# ---- 3D drawing primitives (style matches tf_tree_visualizer.py) ----------

def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r1 = np.cross(d, ref)
    r1 /= np.linalg.norm(r1)
    r2 = np.cross(d, r1)
    return r1, r2


def _draw_cone(ax, tip: np.ndarray, direction: np.ndarray,
               cone_length: float, color: str, n_pts: int = 24):
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
    return ax.plot_surface(X, Y, Z, color=color, shade=True, alpha=0.95, linewidth=0)


def _draw_axis_arrow(ax, origin: np.ndarray, direction: np.ndarray,
                     length: float, color: str, diameter: float):
    d = direction * length
    return ax.quiver(origin[0], origin[1], origin[2],
                     d[0], d[1], d[2],
                     color=color, linewidth=diameter,
                     arrow_length_ratio=0)


def _draw_uav(ax, state: UAVState) -> list:
    R = _yaw_R(state.yaw)
    o = np.array([state.x, state.y, state.z])
    artists = []
    for i, color in enumerate(AXIS_COLORS):
        artists.append(_draw_axis_arrow(ax, o, R[:, i], UAV_AXIS_LENGTH,
                                        color, UAV_AXIS_DIAMETER))
    label = ax.text(o[0], o[1], o[2] + 0.15 * UAV_AXIS_LENGTH,
                    f"  {state.name}", color=LABEL_COLOR, fontsize=10,
                    zorder=100, ha="left")
    _outlined_text(label)
    artists.append(label)
    return artists


def _draw_arrow(ax, p_start: np.ndarray, p_end: np.ndarray,
                cap_fraction: float = ARROW_HEAD_FRACTION) -> list:
    d = p_end - p_start
    length = float(np.linalg.norm(d))
    if length < 1e-9:
        return []
    direction = d / length
    cone_len = max(min(length * cap_fraction, length * 0.9), 1e-6)
    shaft_end = p_end - direction * cone_len
    shaft_d = shaft_end - p_start
    shaft = ax.quiver(p_start[0], p_start[1], p_start[2],
                      shaft_d[0], shaft_d[1], shaft_d[2],
                      color=ARROW_COLOR, linewidth=ARROW_DIAMETER,
                      arrow_length_ratio=0)
    head = _draw_cone(ax, p_end, direction, cone_len, ARROW_COLOR)
    return [shaft, head]


# ============================================================================

class TwoUAVScene:
    def __init__(self, image_path: Path):
        self.img = mpimg.imread(str(image_path))

        self.uavs = [
            UAVState("uav1", x=-2.0, y=0.0, z=1.0, yaw=0.0),
            UAVState("uav2", x= 2.0, y=0.0, z=1.0, yaw=float(np.pi)),
        ]
        self._uav_artists: list[list] = [[], []]
        self._arrow_artists: list = []
        self._grid_artists: list = []
        self._cap_fraction: float = ARROW_HEAD_FRACTION

        self._build_figure()

    def _build_figure(self):
        self.fig = plt.figure(figsize=(10, 9), facecolor=FIG_FACECOLOR)

        # Plot region shared by background + 3D axes.
        plot_rect = [0.05, 0.32, 0.90, 0.64]

        # ---- Static background axes (behind everything, non-interactive) --
        self.bg_ax = self.fig.add_axes(plot_rect, zorder=0)
        self.bg_ax.imshow(self.img, aspect="auto", interpolation="bilinear")
        self.bg_ax.set_xticks([])
        self.bg_ax.set_yticks([])
        for spine in self.bg_ax.spines.values():
            spine.set_visible(False)
        self.bg_ax.set_navigate(False)  # ignore pan/zoom
        self.bg_ax.set_in_layout(False)

        # ---- Transparent 3D axes (UAVs + arrow only) ----------------------
        self.ax = self.fig.add_axes(plot_rect, projection="3d", zorder=1)
        self.ax.patch.set_alpha(0.0)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("none")
        self.ax.grid(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.set_zticks([])
        self.ax.xaxis.line.set_color("none")
        self.ax.yaxis.line.set_color("none")
        self.ax.zaxis.line.set_color("none")

        self.ax.set_xlim(-WORLD_HALF, WORLD_HALF)
        self.ax.set_ylim(-WORLD_HALF, WORLD_HALF)
        self.ax.set_zlim(0.0, WORLD_Z_MAX)
        try:
            self.ax.set_box_aspect((1.0, 1.0, WORLD_Z_MAX / (2 * WORLD_HALF)))
        except Exception:
            pass

        self._draw_grid()

        # ---- Sliders ------------------------------------------------------
        slider_specs = [
            ("uav1 X", -WORLD_HALF, WORLD_HALF,
             lambda i=0: self.uavs[i].x,
             lambda v, i=0: self._set_field(i, "x", v)),
            ("uav1 Y", -WORLD_HALF, WORLD_HALF,
             lambda i=0: self.uavs[i].y,
             lambda v, i=0: self._set_field(i, "y", v)),
            ("uav1 Z", 0.0, WORLD_Z_MAX,
             lambda i=0: self.uavs[i].z,
             lambda v, i=0: self._set_field(i, "z", v)),
            ("uav1 yaw [deg]", -180.0, 180.0,
             lambda i=0: float(np.degrees(self.uavs[i].yaw)),
             lambda v, i=0: self._set_field(i, "yaw", float(np.radians(v)))),
            ("uav2 X", -WORLD_HALF, WORLD_HALF,
             lambda i=1: self.uavs[i].x,
             lambda v, i=1: self._set_field(i, "x", v)),
            ("uav2 Y", -WORLD_HALF, WORLD_HALF,
             lambda i=1: self.uavs[i].y,
             lambda v, i=1: self._set_field(i, "y", v)),
            ("uav2 Z", 0.0, WORLD_Z_MAX,
             lambda i=1: self.uavs[i].z,
             lambda v, i=1: self._set_field(i, "z", v)),
            ("uav2 yaw [deg]", -180.0, 180.0,
             lambda i=1: float(np.degrees(self.uavs[i].yaw)),
             lambda v, i=1: self._set_field(i, "yaw", float(np.radians(v)))),
        ]

        self.sliders: list[Slider] = []
        self._shift_buttons: list[Button] = []  # keep refs alive
        slider_h = 0.025
        slider_gap = 0.008
        bottom_margin = 0.02
        btn_w = 0.022
        btn_gap = 0.004
        col_w = 0.36 - 2 * (btn_w + btn_gap)
        for col in range(2):
            for row in range(4):
                idx = col * 4 + row
                label, vmin, vmax, getter, setter = slider_specs[idx]
                left = 0.09 + col * (0.36 + 0.10)
                bottom = bottom_margin + (3 - row) * (slider_h + slider_gap)
                ax_s = self.fig.add_axes([left, bottom, col_w, slider_h],
                                         facecolor=SLIDER_TRACK_COLOR)
                s = Slider(ax_s, label, vmin, vmax, valinit=getter(),
                           color=SLIDER_HANDLE_COLOR)
                s.label.set_color(LABEL_COLOR)
                s.valtext.set_color(LABEL_COLOR)
                s.on_changed(setter)
                self.sliders.append(s)

                # Skip shift buttons for yaw (cyclic, fixed range).
                if "yaw" in label:
                    continue
                btn_left = left + col_w + btn_gap
                ax_minus = self.fig.add_axes(
                    [btn_left, bottom, btn_w, slider_h],
                    facecolor=SLIDER_TRACK_COLOR)
                ax_plus = self.fig.add_axes(
                    [btn_left + btn_w + btn_gap, bottom, btn_w, slider_h],
                    facecolor=SLIDER_TRACK_COLOR)
                b_minus = Button(ax_minus, "−",
                                 color=SLIDER_TRACK_COLOR,
                                 hovercolor=SLIDER_HANDLE_COLOR)
                b_plus = Button(ax_plus, "+",
                                color=SLIDER_TRACK_COLOR,
                                hovercolor=SLIDER_HANDLE_COLOR)
                b_minus.label.set_color(LABEL_COLOR)
                b_plus.label.set_color(LABEL_COLOR)
                b_minus.on_clicked(
                    lambda _e, s=s: self._shift_slider(s, -SLIDER_SHIFT))
                b_plus.on_clicked(
                    lambda _e, s=s: self._shift_slider(s, +SLIDER_SHIFT))
                self._shift_buttons.extend([b_minus, b_plus])

        # ---- Arrow cap slider (full width, centred) ----------------------
        ax_cap = self.fig.add_axes([0.09, 0.145, 0.72, 0.025],
                                    facecolor=SLIDER_TRACK_COLOR)
        self.cap_slider = Slider(ax_cap, "arrow cap size", 0.01, 0.60,
                                 valinit=ARROW_HEAD_FRACTION,
                                 color=SLIDER_HANDLE_COLOR)
        self.cap_slider.label.set_color(LABEL_COLOR)
        self.cap_slider.valtext.set_color(LABEL_COLOR)
        self.cap_slider.on_changed(self._on_cap_changed)
        self.sliders.append(self.cap_slider)

        # ---- Save ---------------------------------------------------------
        self.save_ax = self.fig.add_axes([0.86, 0.18, 0.10, 0.045])
        self.save_button = Button(self.save_ax, "Save",
                                  color=SLIDER_TRACK_COLOR,
                                  hovercolor=SLIDER_HANDLE_COLOR)
        self.save_button.label.set_color(LABEL_COLOR)
        self.save_button.on_clicked(self._on_save)

        self._redraw()

    # ---- Reference grid ---------------------------------------------------

    def _draw_grid(self):
        ticks = np.arange(-WORLD_HALF, WORLD_HALF + GRID_STEP * 0.5, GRID_STEP)
        for x in ticks:
            line, = self.ax.plot([x, x], [-WORLD_HALF, WORLD_HALF], [0, 0],
                                 color=GRID_COLOR, alpha=GRID_ALPHA,
                                 linewidth=GRID_LINEWIDTH, zorder=0)
            self._grid_artists.append(line)
        for y in ticks:
            line, = self.ax.plot([-WORLD_HALF, WORLD_HALF], [y, y], [0, 0],
                                 color=GRID_COLOR, alpha=GRID_ALPHA,
                                 linewidth=GRID_LINEWIDTH, zorder=0)
            self._grid_artists.append(line)

    def _set_grid_visible(self, visible: bool):
        for a in self._grid_artists:
            a.set_visible(visible)

    def _on_cap_changed(self, value: float):
        self._cap_fraction = float(value)
        self._redraw()

    # ---- Slider window shifting ------------------------------------------

    @staticmethod
    def _shift_slider(slider: Slider, delta: float):
        """Pan a Slider's [valmin, valmax] window by ``delta`` in-place."""
        new_min = slider.valmin + delta
        new_max = slider.valmax + delta
        slider.valmin = new_min
        slider.valmax = new_max
        slider.ax.set_xlim(new_min, new_max)
        # Clamp current value into the new window; on_changed will fire only
        # if the value actually moved, which correctly syncs the UAV pose.
        cur = slider.val
        clamped = min(max(cur, new_min), new_max)
        slider.set_val(clamped)
        slider.ax.figure.canvas.draw_idle()

    # ---- State updates ----------------------------------------------------

    def _set_field(self, idx: int, field: str, value: float):
        setattr(self.uavs[idx], field, float(value))
        self._redraw()

    def _redraw(self):
        for artists in self._uav_artists:
            for a in artists:
                try:
                    a.remove()
                except Exception:
                    pass
        for a in self._arrow_artists:
            try:
                a.remove()
            except Exception:
                pass
        self._uav_artists = [[], []]
        self._arrow_artists = []

        p1 = np.array([self.uavs[0].x, self.uavs[0].y, self.uavs[0].z])
        p2 = np.array([self.uavs[1].x, self.uavs[1].y, self.uavs[1].z])
        self._arrow_artists = _draw_arrow(self.ax, p1, p2,
                                          cap_fraction=self._cap_fraction)
        for i, uav in enumerate(self.uavs):
            self._uav_artists[i] = _draw_uav(self.ax, uav)

        self.fig.canvas.draw_idle()

    # ---- Save -------------------------------------------------------------

    def _on_save(self, event):
        widget_axes = [self.save_ax] + [s.ax for s in self.sliders]
        for wa in widget_axes:
            wa.set_visible(False)
        self._set_grid_visible(False)
        try:
            stem = "two_uav_arrow"
            for ext in ("pgf", "pdf", "svg", "png"):
                self.fig.savefig(f"{stem}.{ext}", bbox_inches="tight",
                                 pad_inches=0.02,
                                 facecolor=self.fig.get_facecolor())
            print(f"Saved {stem}.{{pgf,pdf,svg,png}}")
        finally:
            for wa in widget_axes:
                wa.set_visible(True)
            self._set_grid_visible(True)
            self.fig.canvas.draw_idle()

    def show(self):
        plt.show()


# ============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path,
                   help="background image (png/jpg/...)")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.image.exists():
        raise SystemExit(f"image not found: {args.image}")
    TwoUAVScene(image_path=args.image).show()


if __name__ == "__main__":
    main()
