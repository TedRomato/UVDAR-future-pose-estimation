"""RViz-style visualization of a graph of named transforms.

Transforms are loaded from ``transforms.yaml`` next to this script (same
schema as a ROS ``/tf_static`` dump: each entry has ``frame_id``,
``child_frame_id``, ``pose`` and ``rot``).  The file may describe an
arbitrary directed multigraph (forest, multi-parent, even cycles): a frame
may appear as ``child_frame_id`` of more than one entry.  Relative poses
are computed by BFS from the scene's chosen ``origin`` frame, applying the
inverse transform automatically when an edge is traversed child -> parent.
The whole resolved scene is then placed so the chosen origin sits at
(0, 0, 0).

Per scene you can pick any frame as the rendering origin and choose which
frames to draw, plus per-frame overrides for axis length, axis line width
and incoming-link line width.  Scenes may also draw explicit frame-to-frame
vectors after the normal frame/link/grid rendering.

Run::

    python3 helper_visualizations/tf_tree_visualizer.py --list-scenes
    python3 helper_visualizations/tf_tree_visualizer.py --scene full_tree
    python3 helper_visualizations/tf_tree_visualizer.py --scene uav1_cameras \\
        --axis-length 0.05 --no-links
    python3 helper_visualizations/tf_tree_visualizer.py \\
        --transforms path/to/other.yaml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import yaml
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d projection)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ============================================================================
# Config
# ============================================================================

# Default axis arrow length (metres) and arrow line widths (in points).
# Note: matplotlib's 3D quiver does not support true tube radii, so the
# "diameter" knobs map onto Line2D ``linewidth`` (points).  Visually this is
# the right knob to tweak; numerically it is not a real metric thickness.
AXIS_LENGTH      = 0.15
AXIS_DIAMETER    = 2.0
LINK_DIAMETER    = 1.5
AXIS_COLORS = ("tab:red", "tab:green", "tab:blue")  # X, Y, Z
LINK_COLOR  = "gold"
VECTOR_COLOR = "#f38ba8"
VECTOR_DIAMETER = 2.5
VECTOR_ARROW_SIZE: float | None = None  # cone length in metres; None = 15% of arrow length

# Custom XY-plane grid drawn at z = grid_z.
# GRID_SIZE = None -> size is chosen automatically from the scene extent.
# GRID_STEP = None -> step is chosen automatically from the grid extent.
GRID_COLOR        = "#3a3a4a"   # subtle dark lines on dark background
GRID_LINEWIDTH    = 0.6
GRID_AXIS_COLOR   = "#6b6b80"   # slightly brighter centre lines
GRID_AXIS_LW      = 1.0
GRID_SIZE: float | None = None  # metres, full side length; None = auto
GRID_STEP: float | None = None  # metres; None = auto

FIG_FACECOLOR  = "#1e1e2e"  # dark background
AX_FACECOLOR   = "#1e1e2e"  # same for the 3-D axes pane
LABEL_COLOR    = "#cdd6f4"  # light text for title / frame names
TEXT_OUTLINE_COLOR = "#000000"  # black stroke behind light text
TEXT_OUTLINE_WIDTH = 3          # points


def _outlined_text(text_obj) -> None:
    """Apply a dark stroke outline to a matplotlib Text object in-place."""
    text_obj.set_path_effects([
        pe.withStroke(linewidth=TEXT_OUTLINE_WIDTH, foreground=TEXT_OUTLINE_COLOR),
        pe.Normal(),
    ])


@dataclass
class Transform:
    """A directed edge in the TF graph: T_parent_child.

    Multiple ``Transform`` entries with the same ``child_id`` are allowed
    (a frame may have several parents).  When computing relative poses,
    BFS from the scene origin picks a spanning path; inverses are applied
    automatically when an edge is traversed child -> parent.

    ``parent_id = None`` registers a free-standing frame with no edges.
    ``children_ids`` and ``draw_link_arrow`` are informational/visual only.
    """
    child_id:    str
    parent_id:   str | None
    xyzq: tuple[float, float, float, float, float, float, float]
    children_ids: list[str] = field(default_factory=list)
    draw_link_arrow: bool = False  # show arrowhead on the parent->child link


@dataclass
class FrameStyle:
    """Per-frame visualization overrides for a Scene.

    Any field left as ``None`` falls through to the scene's setting and
    then to the global default.  ``link_diameter`` controls the parent->
    child link incoming to *this* frame.
    """
    axis_length:   float | None = None
    axis_diameter: float | None = None
    link_diameter: float | None = None
    label_side:    str | None = None  # "right" or "left"; None = inherit scene


@dataclass
class VectorToDraw:
    """A visual vector from one resolved frame position to another."""
    start_frame: str
    end_frame: str
    color: str = VECTOR_COLOR
    diameter: float = VECTOR_DIAMETER
    arrow: bool = True
    arrow_size: float | None = None  # cone length in metres; None = global default (15% of length)


@dataclass
class Scene:
    """A render configuration: which root, what to show, link toggle, sizes."""
    name: str
    origin: str
    display_name: str | None = None  # figure title; falls back to auto if None
    visible: list[str] | None = None  # None => all transforms
    draw_links: bool = True
    axis_length:   float | None = None  # None => fall back to global / CLI
    axis_diameter: float | None = None
    link_diameter: float | None = None
    grid_size:     float | None = None  # full side length in metres; None = auto
    grid_step:     float | None = None  # grid square side length in metres; None = auto
    label_side:    str = "right"       # "right" or "left" frame labels
    vectors_to_draw: list[VectorToDraw | tuple[str, str]] = field(default_factory=list)
    # Per-frame overrides keyed by child_id.  Frames not listed use the
    # scene-level / global values.
    frame_styles: dict[str, FrameStyle] = field(default_factory=dict)


# ---- Default YAML source for transforms ----------------------------------
DEFAULT_TRANSFORMS_YAML = Path(__file__).with_name("transforms.yaml")


def load_transforms_yaml(path: Path | str = DEFAULT_TRANSFORMS_YAML
                         ) -> list[Transform]:
    """Load transforms from a YAML file with the schema::

        transforms:
          - frame_id: <parent>
            child_frame_id: <child>
            pose: {x: .., y: .., z: ..}
            rot:  {x: .., y: .., z: .., w: ..}

    Each row is a directed edge.  Duplicate ``child_frame_id`` is allowed
    (multi-parent), and the graph need not be connected.
    """
    with open(path) as f:
        data = yaml.safe_load(f)
    if not data or "transforms" not in data:
        raise ValueError(f"{path}: missing top-level 'transforms' list")

    transforms: list[Transform] = []
    children_by_parent: dict[str, list[str]] = {}
    for row in data["transforms"]:
        parent = row["frame_id"]
        child  = row["child_frame_id"]
        p, r   = row["pose"], row["rot"]
        xyzq = (float(p["x"]), float(p["y"]), float(p["z"]),
                float(r["x"]), float(r["y"]), float(r["z"]), float(r["w"]))
        transforms.append(Transform(child_id=child, parent_id=parent, xyzq=xyzq))
        children_by_parent.setdefault(parent, []).append(child)

    # Fill in children_ids (informational only) for the first edge of each child.
    seen: set[str] = set()
    for tf in transforms:
        if tf.child_id in seen:
            continue
        seen.add(tf.child_id)
        if tf.child_id in children_by_parent:
            tf.children_ids = children_by_parent[tf.child_id]
    return transforms


# Loaded at import time from helper_visualizations/transforms.yaml.
# Override on the CLI with --transforms PATH.
TRANSFORMS: list[Transform] = load_transforms_yaml()


# ---- Edit me: declare your scenes here. -----------------------------------
# Scene origin must be one of the frame names present in TRANSFORMS.
SCENES: list[Scene] = [
    Scene(
        name="cameras_frames",
        origin="uav1/fcu",
        display_name="Side cameras",
        visible=["uav1/fcu", "uav1/uvcam_right", "uav1/uvcam_left"],
        grid_size=1.0,
        grid_step=0.1,
        vectors_to_draw=[],
        frame_styles={
            "uav1/fcu": FrameStyle(axis_length=0.1,axis_diameter=4, link_diameter=1),
            "uav1/uvcam_left": FrameStyle(axis_length=0.065, axis_diameter=2, link_diameter=1),
            "uav1/uvcam_right": FrameStyle(axis_length=0.065, axis_diameter=2, link_diameter=1),
        },
    ),
    Scene(
        name="relative_pose_frames",
        origin="uav/gps_baro_origin",
        display_name="UAV relative pose",
        visible=["uav1/fcu", "uav2/fcu"], #,"uav/gps_baro_origin", "uav1/uvcam_right"],
        grid_size=20.0,
        grid_step=1.0,
        vectors_to_draw=[
            VectorToDraw("uav2/fcu", "uav1/fcu", color="cyan", diameter=1.0, arrow_size=0.3),
        ],
        frame_styles={
            "uav1/fcu": FrameStyle(axis_length=0.75,axis_diameter=4, link_diameter=1),
            "uav2/fcu": FrameStyle(axis_length=0.75, axis_diameter=4, link_diameter=1),
            "uav/gps_baro_origin": FrameStyle(axis_length=1, axis_diameter=4, link_diameter=1),
            "uav1/uvcam_right": FrameStyle(axis_length=0.4, axis_diameter=2, link_diameter=1, label_side="left"),
        },
    ),
    Scene(
        name="uvdar_frame_mapping",
        origin="uav1/local_origin",
        display_name="UVDAR remapping",
        visible=["uav1/fcu", "uav2/fcu", "uav1/local_origin", "uav1/fixed_origin", "uav1/uvcam_right"],
        grid_size=20.0,
        grid_step=1.0,
        vectors_to_draw=[
            VectorToDraw("uav1/local_origin", "uav2/fcu", color="orange", diameter=1.0, arrow_size=0.3),
            VectorToDraw("uav1/uvcam_right", "uav2/fcu", color="cyan", diameter=1.0, arrow_size=0.3),
        ],
        frame_styles={
            "uav1/fcu": FrameStyle(axis_length=0.75,axis_diameter=4, link_diameter=1, label_side="left"),
            "uav2/fcu": FrameStyle(axis_length=0.75, axis_diameter=4, link_diameter=1),
            "uav1/local_origin": FrameStyle(axis_length=0.75,axis_diameter=4, link_diameter=1),
            "uav1/fixed_origin": FrameStyle(axis_length=0.75, axis_diameter=4, link_diameter=1),
            "uav1/uvcam_right": FrameStyle(axis_length=0.4, axis_diameter=2, link_diameter=1, label_side="right"),
        },
    )
]


# ============================================================================
# SE(3) helpers (xyzw quaternion convention)
# ============================================================================

def _quat_to_R(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    n = (qx * qx + qy * qy + qz * qz + qw * qw) ** 0.5
    if n == 0.0:
        raise ValueError("zero-norm quaternion")
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array([
        [1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy)],
        [    2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx)],
        [    2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)],
    ])


def _make_T(xyzq) -> np.ndarray:
    if len(xyzq) != 7:
        raise ValueError(f"expected 7-tuple (x,y,z,qx,qy,qz,qw), got {xyzq!r}")
    x, y, z, qx, qy, qz, qw = xyzq
    T = np.eye(4)
    T[:3, :3] = _quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = (x, y, z)
    return T


# ============================================================================
# Graph resolution (multi-parent BFS)
# ============================================================================

# A directed edge: (parent, child, T_parent_child, draw_link_arrow).
Edge = tuple[str, str, np.ndarray, bool]


def _build_graph(transforms: list[Transform]
                 ) -> tuple[dict[str, list[tuple[str, np.ndarray]]],
                            list[Edge], set[str]]:
    """Build an undirected adjacency with per-edge SE(3) transforms.

    For each YAML row ``T_parent_child`` we add two adjacency entries:
    parent -> child stores ``T``, and child -> parent stores ``inv(T)``.
    BFS can then traverse in either direction and the relative pose at the
    neighbour is simply ``T_origin_neighbour = T_origin_node @ T_node_neighbour``.
    """
    adj: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
    edges: list[Edge] = []
    frames: set[str] = set()
    for tf in transforms:
        frames.add(tf.child_id)
        if tf.parent_id is None:
            continue
        frames.add(tf.parent_id)
        T = _make_T(tf.xyzq)
        T_inv = np.linalg.inv(T)
        adj[tf.parent_id].append((tf.child_id, T))
        adj[tf.child_id].append((tf.parent_id, T_inv))
        edges.append((tf.parent_id, tf.child_id, T, tf.draw_link_arrow))
    return adj, edges, frames


def compute_relative_poses(transforms: list[Transform], origin: str
                            ) -> tuple[dict[str, np.ndarray], list[Edge], set[str]]:
    """BFS from ``origin`` and return T_origin_frame for every reachable frame.

    Frames not connected to the origin are simply absent from the returned
    dict (callers should validate that visible frames are reachable).
    """
    adj, edges, frames = _build_graph(transforms)
    if origin not in frames:
        raise ValueError(f"origin {origin!r} is not present in any transform")

    poses: dict[str, np.ndarray] = {origin: np.eye(4)}
    queue: deque[str] = deque([origin])
    while queue:
        node = queue.popleft()
        T_on = poses[node]
        for nbr, T_node_nbr in adj.get(node, []):
            if nbr in poses:
                continue
            poses[nbr] = T_on @ T_node_nbr
            queue.append(nbr)
    return poses, edges, frames


# ============================================================================
# Rendering
# ============================================================================

def _draw_axis_arrow(ax, origin: np.ndarray, direction: np.ndarray,
                     length: float, color: str, diameter: float) -> None:
    d = direction * length
    ax.quiver(
        origin[0], origin[1], origin[2],
        d[0], d[1], d[2],
        color=color, linewidth=diameter,
        arrow_length_ratio=0,
    )


def _draw_frame(ax, name: str, T: np.ndarray,
        axis_length: float, axis_diameter: float,
        label_side: str) -> None:
    o = T[:3, 3]
    R = T[:3, :3]
    for i, color in enumerate(AXIS_COLORS):
        _draw_axis_arrow(ax, o, R[:, i], axis_length, color, axis_diameter)
    label_text = f"  {name}" if label_side == "right" else f"{name}  "
    ha = "left" if label_side == "right" else "right"
    ax.text(o[0], o[1], o[2] + 0.04 * axis_length / 0.15,
        label_text, fontsize=10, color=LABEL_COLOR, zorder=100, ha=ha)
    _outlined_text(ax.texts[-1])


def _perp_basis(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two unit vectors orthogonal to unit vector ``d``."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    r1 = np.cross(d, ref)
    r1 /= np.linalg.norm(r1)
    r2 = np.cross(d, r1)
    return r1, r2


def _draw_cone(ax, tip: np.ndarray, direction: np.ndarray,
               cone_length: float, color: str, n_pts: int = 24) -> None:
    """Draw a 3D cone whose apex is at ``tip``, base facing away from ``tip``."""
    d = direction / np.linalg.norm(direction)
    cone_radius = cone_length * 0.4  # base radius = 40% of length → nicely pointy
    base_center = tip - d * cone_length
    r1, r2 = _perp_basis(d)
    theta = np.linspace(0, 2 * np.pi, n_pts)
    t = np.array([0.0, 1.0])  # 0 = base, 1 = tip
    T, Theta = np.meshgrid(t, theta)
    R = cone_radius * (1 - T)
    X = base_center[0] + T * cone_length * d[0] + R * (np.cos(Theta) * r1[0] + np.sin(Theta) * r2[0])
    Y = base_center[1] + T * cone_length * d[1] + R * (np.cos(Theta) * r1[1] + np.sin(Theta) * r2[1])
    Z = base_center[2] + T * cone_length * d[2] + R * (np.cos(Theta) * r1[2] + np.sin(Theta) * r2[2])
    ax.plot_surface(X, Y, Z, color=color, shade=True, alpha=0.95, linewidth=0)


def _draw_link(ax, p_parent: np.ndarray, p_child: np.ndarray,
               link_diameter: float, arrow: bool = False,
               arrow_size: float | None = None) -> None:
    d = p_child - p_parent
    length = float(np.linalg.norm(d))
    if length == 0:
        return
    direction = d / length
    if arrow:
        cone_len = arrow_size if arrow_size is not None else length * 0.15
        cone_len = min(cone_len, length * 0.9)
        shaft_end = p_child - direction * cone_len
    else:
        shaft_end = p_child
        cone_len = 0.0
    shaft_d = shaft_end - p_parent
    ax.quiver(
        p_parent[0], p_parent[1], p_parent[2],
        shaft_d[0], shaft_d[1], shaft_d[2],
        color=LINK_COLOR, linewidth=link_diameter,
        arrow_length_ratio=0,
    )
    if arrow and cone_len > 0:
        _draw_cone(ax, p_child, direction, cone_len, LINK_COLOR)


def _draw_vector(ax, p_start: np.ndarray, p_end: np.ndarray,
                 color: str, diameter: float, arrow: bool,
                 arrow_size: float | None = None) -> None:
    d = p_end - p_start
    length = float(np.linalg.norm(d))
    if length == 0:
        return
    direction = d / length
    if arrow:
        cone_len = arrow_size if arrow_size is not None else length * 0.15
        cone_len = min(cone_len, length * 0.9)
        shaft_end = p_end - direction * cone_len
    else:
        shaft_end = p_end
        cone_len = 0.0
    shaft_d = shaft_end - p_start
    ax.quiver(
        p_start[0], p_start[1], p_start[2],
        shaft_d[0], shaft_d[1], shaft_d[2],
        color=color, linewidth=diameter,
        arrow_length_ratio=0,
    )
    if arrow and cone_len > 0:
        _draw_cone(ax, p_end, direction, cone_len, color)


def _normalize_vector(spec: VectorToDraw | tuple[str, str]) -> VectorToDraw:
    if isinstance(spec, VectorToDraw):
        return spec
    if len(spec) != 2:
        raise ValueError(
            "vector tuple must be (start_frame, end_frame), "
            f"got {spec!r}")
    return VectorToDraw(start_frame=spec[0], end_frame=spec[1])


def _compute_limits(points: np.ndarray, axis_length: float
                    ) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (lo3, hi3, half_span) for equal-aspect box."""
    if points.size == 0:
        lo = np.array([-1.0, -1.0, -1.0])
        hi = np.array([ 1.0,  1.0,  1.0])
    else:
        pad = max(axis_length * 1.5, 0.1)
        lo = points.min(axis=0) - pad
        hi = points.max(axis=0) + pad
    span = (hi - lo).max()
    mid  = 0.5 * (hi + lo)
    half = 0.5 * span
    lo3 = mid - half
    hi3 = mid + half
    return lo3, hi3, half


def _set_equal_aspect(ax, lo3: np.ndarray, hi3: np.ndarray) -> None:
    ax.set_xlim(lo3[0], hi3[0])
    ax.set_ylim(lo3[1], hi3[1])
    ax.set_zlim(lo3[2], hi3[2])


def _draw_grid(ax, lo3: np.ndarray, hi3: np.ndarray,
               size: float | None = None,
               step: float | None = None, z: float | None = None) -> None:
    """Draw a symmetric XY-plane grid at height z (default: lo3[2]).

    The grid is square and centred at the XY midpoint of the scene box so
    it looks the same from any view angle.  Replaces the default matplotlib
    3D grid (ax.grid(False) must be called before this).
    """
    grid_z = lo3[2] if z is None else z

    # Compute a symmetric half-extent centred at world (0, 0) so the
    # grid axes always cross at the origin.
    if size is not None and size <= 0:
        raise ValueError(f"grid size must be positive, got {size!r}")
    half = (size * 0.5 if size is not None
            else max(hi3[0] - lo3[0], hi3[1] - lo3[1]) * 0.5)
    xmin, xmax = -half, half
    ymin, ymax = -half, half

    if step is not None and step <= 0:
        raise ValueError(f"grid step must be positive, got {step!r}")

    if step is None:
        # Pick a round step that gives roughly 8-12 lines per axis.
        raw_step = (xmax - xmin) / 10.0
        magnitude = 10 ** np.floor(np.log10(max(raw_step, 1e-9)))
        for s in (magnitude, 2 * magnitude, 5 * magnitude, 10 * magnitude):
            if (xmax - xmin) / s <= 12:
                step = s
                break
        else:
            step = magnitude

    # Round grid bounds outward to the nearest step.
    xs = np.arange(np.floor(xmin / step) * step,
                   np.ceil(xmax / step) * step + step * 0.5, step)
    ys = np.arange(np.floor(ymin / step) * step,
                   np.ceil(ymax / step) * step + step * 0.5, step)

    for x in xs:
        ax.plot([x, x], [ys[0], ys[-1]], [grid_z, grid_z],
                color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)
    for y in ys:
        ax.plot([xs[0], xs[-1]], [y, y], [grid_z, grid_z],
                color=GRID_COLOR, linewidth=GRID_LINEWIDTH, zorder=0)

    # Emphasise X and Y axes on the ground plane.
    ax.plot([xs[0], xs[-1]], [0, 0], [grid_z, grid_z],
            color=GRID_AXIS_COLOR, linewidth=GRID_AXIS_LW, zorder=1)
    ax.plot([0, 0], [ys[0], ys[-1]], [grid_z, grid_z],
            color=GRID_AXIS_COLOR, linewidth=GRID_AXIS_LW, zorder=1)


def render_scene(scene: Scene,
                 transforms: list[Transform],
                 axis_length: float,
                 axis_diameter: float,
                 link_diameter: float,
                 draw_links: bool,
                 grid_size: float | None = None,
                 grid_step: float | None = None) -> None:
    T_rel, edges, frames = compute_relative_poses(transforms, scene.origin)

    visible_names = (sorted(T_rel.keys()) if scene.visible is None
                     else list(dict.fromkeys(scene.visible)))  # dedupe, preserve order
    for n in visible_names:
        if n not in frames:
            raise ValueError(
                f"scene {scene.name!r}: visible frame {n!r} is unknown")
        if n not in T_rel:
            raise ValueError(
                f"scene {scene.name!r}: visible frame {n!r} is not connected "
                f"to origin {scene.origin!r}")

    vectors = [_normalize_vector(spec) for spec in scene.vectors_to_draw]
    for vec in vectors:
        for frame_name in (vec.start_frame, vec.end_frame):
            if frame_name not in frames:
                raise ValueError(
                    f"scene {scene.name!r}: vector frame {frame_name!r} is unknown")
            if frame_name not in T_rel:
                raise ValueError(
                    f"scene {scene.name!r}: vector frame {frame_name!r} is not connected "
                    f"to origin {scene.origin!r}")
        if vec.diameter <= 0:
            raise ValueError(
                f"scene {scene.name!r}: vector diameter must be positive, "
                f"got {vec.diameter!r}")

    # Per-frame style resolution: frame_styles -> scene -> CLI/global default.
    def _style(name: str) -> FrameStyle:
        return scene.frame_styles.get(name, FrameStyle())

    def _len(name: str) -> float:
        s = _style(name).axis_length
        return s if s is not None else axis_length

    def _diam(name: str) -> float:
        s = _style(name).axis_diameter
        return s if s is not None else axis_diameter

    def _link(name: str) -> float:
        s = _style(name).link_diameter
        return s if s is not None else link_diameter

    def _label_side(name: str) -> str:
        s = _style(name).label_side
        side = s if s is not None else scene.label_side
        if side not in ("right", "left"):
            raise ValueError(
                f"scene {scene.name!r}: label_side for frame {name!r} must be "
                f"'right' or 'left', got {side!r}")
        return side

    fig = plt.figure(figsize=(8, 7), facecolor=FIG_FACECOLOR)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(AX_FACECOLOR)

    visible_set = set(visible_names)

    for n in visible_names:
        _draw_frame(ax, n, T_rel[n], _len(n), _diam(n), _label_side(n))

    if draw_links:
        for parent, child, _T, arrow in edges:
            if parent not in visible_set or child not in visible_set:
                continue
            if parent not in T_rel or child not in T_rel:
                continue  # disconnected from origin
            _draw_link(ax,
                       T_rel[parent][:3, 3],
                       T_rel[child][:3, 3],
                       _link(child),
                       arrow=arrow)

    limit_points = [T_rel[n][:3, 3] for n in visible_names]
    for vec in vectors:
        limit_points.append(T_rel[vec.start_frame][:3, 3])
        limit_points.append(T_rel[vec.end_frame][:3, 3])
    pts = np.array(limit_points)
    lo3, hi3, _ = _compute_limits(pts, axis_length)

    # Replace the default 3D grid with a clean XY-plane grid at z = 0.
    ax.grid(False)
    _draw_grid(ax, lo3, hi3, size=grid_size, step=grid_step, z=0.0)

    # Hide all default pane walls, tick lines and the z/x/y axes entirely.
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("none")
    ax.yaxis.pane.set_edgecolor("none")
    ax.zaxis.pane.set_edgecolor("none")
    # Use color="none" instead of set_visible(False): hiding axis lines
    # entirely empties the bbox and crashes matplotlib's title layout.
    # Keeping one ghost tick at 0 also prevents an empty-bbox ValueError.
    ax.xaxis.line.set_color("none")
    ax.yaxis.line.set_color("none")
    ax.zaxis.line.set_color("none")
    ax.set_xticks([0])
    ax.set_yticks([0])
    ax.set_zticks([0])
    ax.tick_params(axis="x", color="none", labelcolor="none", length=0)
    ax.tick_params(axis="y", color="none", labelcolor="none", length=0)
    ax.tick_params(axis="z", color="none", labelcolor="none", length=0)

    _set_equal_aspect(ax, lo3, hi3)

    title = (scene.display_name if scene.display_name is not None
             else f"TF tree — scene: {scene.name} (origin={scene.origin})")
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Draw explicit vectors last so they sit visually on top of the resolved scene.
    for vec in vectors:
        # arrow_size: per-vector → global constant → None (auto 15% of length)
        arrow_size = vec.arrow_size if vec.arrow_size is not None else VECTOR_ARROW_SIZE
        _draw_vector(ax,
                     T_rel[vec.start_frame][:3, 3],
                     T_rel[vec.end_frame][:3, 3],
                     color=vec.color,
                     diameter=vec.diameter,
                     arrow=vec.arrow,
                     arrow_size=arrow_size)

    # tight_layout crashes when the z-axis is fully hidden; use manual margins.
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.08)

    # Save button — same pattern as the other helper visualizations.
    save_stem = scene.name

    def _save(event):
        save_ax.set_visible(False)
        fig.savefig(f"{save_stem}.pgf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{save_stem}.pdf", bbox_inches="tight", pad_inches=0.02)
        fig.savefig(f"{save_stem}.svg", bbox_inches="tight", pad_inches=0.02)
        save_ax.set_visible(True)
        fig.canvas.draw_idle()
        print(f"Saved {save_stem}.{{pgf,pdf,svg}}")

    save_ax = fig.add_axes([0.82, 0.02, 0.12, 0.05])
    save_button = Button(save_ax, "Save")  # noqa: F841  (keeps reference alive)
    save_button.on_clicked(_save)

    plt.show()


# ============================================================================
# CLI
# ============================================================================

def _scene_by_name(name: str) -> Scene:
    for s in SCENES:
        if s.name == name:
            return s
    available = ", ".join(s.name for s in SCENES)
    raise SystemExit(f"unknown scene {name!r}; available: {available}")


def _resolve(cli_value, scene_value, default):
    if cli_value is not None:
        return cli_value
    if scene_value is not None:
        return scene_value
    return default


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scene", default=None,
                    help="Scene name (default: first scene in SCENES).")
    ap.add_argument("--list-scenes", action="store_true",
                    help="List configured scene names and exit.")
    ap.add_argument("--no-links", action="store_true",
                    help="Disable parent->child link arrows for this run.")
    ap.add_argument("--axis-length",   type=float, default=None)
    ap.add_argument("--axis-diameter", type=float, default=None)
    ap.add_argument("--link-diameter", type=float, default=None)
    ap.add_argument("--grid-size", type=float, default=None,
                    help="Override grid side length in metres (default: scene/global auto).")
    ap.add_argument("--grid-step", type=float, default=None,
                    help="Override grid square side length in metres (default: scene/global auto).")
    ap.add_argument("--transforms", type=Path, default=None,
                    help=f"Override the YAML source (default: "
                         f"{DEFAULT_TRANSFORMS_YAML.name}).")
    args = ap.parse_args()

    if args.list_scenes:
        for s in SCENES:
            print(s.name)
        return

    if not SCENES:
        raise SystemExit("no scenes configured in SCENES")

    transforms = (load_transforms_yaml(args.transforms)
                  if args.transforms is not None else TRANSFORMS)

    scenes_to_render = (
        [_scene_by_name(args.scene)] if args.scene else list(SCENES)
    )

    for scene in scenes_to_render:
        axis_length   = _resolve(args.axis_length,   scene.axis_length,   AXIS_LENGTH)
        axis_diameter = _resolve(args.axis_diameter, scene.axis_diameter, AXIS_DIAMETER)
        link_diameter = _resolve(args.link_diameter, scene.link_diameter, LINK_DIAMETER)
        grid_size     = _resolve(args.grid_size,     scene.grid_size,     GRID_SIZE)
        grid_step     = _resolve(args.grid_step,     scene.grid_step,     GRID_STEP)
        draw_links    = scene.draw_links and not args.no_links

        render_scene(scene, transforms,
                     axis_length=axis_length,
                     axis_diameter=axis_diameter,
                     link_diameter=link_diameter,
                     grid_size=grid_size,
                     grid_step=grid_step,
                     draw_links=draw_links)


if __name__ == "__main__":
    main()
