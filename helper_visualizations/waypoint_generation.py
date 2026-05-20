import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_style import apply_style  # noqa: E402

apply_style()


# ============================================================
# Basic helpers
# ============================================================

def normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero-length vector.")
    return v / n


def rotation_matrix(axis, angle_rad):
    axis = normalize(axis)
    x, y, z = axis
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    C = 1 - c

    return np.array([
        [c + x*x*C,     x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s, c + z*z*C]
    ])


def prism_edges():
    return [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]


# Shared colour scheme used by BOTH the 2D side-view and the 3D explainer.
CATEGORY_COLORS = {
    "accepted": "tab:blue",
    "distance": "tab:red",
    "fov":      "grey",
    "ground":   "green",
}
CATEGORY_LABELS = {
    "accepted": "Accepted",
    "distance": "Distance disqualified",
    "fov":      "FOV disqualified",
    "ground":   "Ground disqualified",
}


# ============================================================
# 2D side-view explainer of waypoint sampling
# ============================================================

def _rotate_2d(points, center, angle_rad):
    """Rotate (N,2) points around `center` by `angle_rad` (CCW)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return (pts - center) @ R.T + center


class _ZeroWidthHandler:
    """Legend handler that consumes no handle width, making the label flush left."""
    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        from matplotlib.patches import Rectangle
        handlebox.set_width(0)
        handlebox.set_height(fontsize)
        patch = Rectangle([0, 0], 0, 0, visible=False,
                          transform=handlebox.get_transform())
        handlebox.add_artist(patch)
        return patch


def draw_sideview_2d(
    ax=None,
    *,
    hfov_deg=90.0,
    vfov_deg=60.0,
    min_dist=2.0,
    max_dist=20.0,
    camera_xz=(0.0, 5.0),
    pitch_deg=10.0,
    points=None,
    point_color=None,
    point_categories=None,
    show_ground=True,
):
    """
    2D side-view (world XZ plane) explainer of the waypoint-sampling
    geometry. Y is ignored. Yaw is a 3D-only concept and is not drawn.

    Parameters
    ----------
    ax : matplotlib Axes, optional
        Where to draw. If None, a new equal-aspect figure is created.
    hfov_deg, vfov_deg : float
        Horizontal / vertical full FOV angles. Side view uses vfov.
    min_dist, max_dist : float
        Min / max sampling distance from the camera origin.
    camera_xz : (float, float)
        Camera origin in the XZ plane. Default (0, 5).
    pitch_deg : float
        Pitch of the camera (positive = nose up). The whole rectangle +
        FOV triangle + arcs + axis arrow are rotated around the
        rectangle center by this angle.
    points : array-like (N, 2), optional
        Sample points to scatter, in world XZ.
    point_color : color or list of colors, optional
        Forwarded to `ax.scatter(c=...)`. Used only if `point_categories`
        is not given.
    point_categories : list[str], optional
        Per-point category strings (keys of `CATEGORY_COLORS`). When
        provided, point colours are looked up automatically and override
        `point_color`.
    show_ground : bool
        Draw a green ground line at z=0.
    """
    from matplotlib.patches import Arc

    cam0 = np.asarray(camera_xz, dtype=float)
    half_v = np.deg2rad(vfov_deg) / 2.0
    pitch = np.deg2rad(pitch_deg)

    # Rectangle (side-view cross-section of the bounding prism), built
    # before rotation. Right edge centered on the camera origin, axis
    # along world +X, height from vfov.
    height = 2.0 * max_dist * np.sin(half_v)
    half_h = height / 2.0
    rect_right_x = cam0[0]
    rect_left_x = cam0[0] + max_dist
    rect_top_z = cam0[1] + half_h
    rect_bot_z = cam0[1] - half_h

    rect = np.array([
        [rect_right_x, rect_top_z],
        [rect_left_x,  rect_top_z],
        [rect_left_x,  rect_bot_z],
        [rect_right_x, rect_bot_z],
        [rect_right_x, rect_top_z],
    ])
    rect_center = np.array([(rect_right_x + rect_left_x) / 2.0,
                            (rect_top_z + rect_bot_z) / 2.0])

    # Camera axis arrow tip (world +X, length = max_dist) before rotation.
    axis_tip = cam0 + np.array([max_dist/2, 0.0])

    # FOV triangle: apex at camera origin, far corners = top-far and
    # bottom-far of the rectangle.
    fov_tri = np.array([
        cam0,
        [rect_left_x, rect_top_z],
        [rect_left_x, rect_bot_z],
        cam0,
    ])

    # Rotate everything by `pitch` around the rectangle center.
    cam_r       = _rotate_2d(cam0[None, :], cam0, pitch)[0]
    axis_tip_r  = _rotate_2d(axis_tip[None, :], cam0, pitch)[0]
    rect_r      = _rotate_2d(rect, cam0, pitch)
    fov_tri_r   = _rotate_2d(fov_tri, cam0, pitch)

    # Min / max distance arcs follow the rotation: angle offset = pitch_deg.
    arc_center = tuple(cam_r)
    arc_theta1 = np.rad2deg(-half_v) + pitch_deg
    arc_theta2 = np.rad2deg(+half_v) + pitch_deg

    # ---- draw ----
    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_aspect("equal", adjustable="datalim")

    if show_ground:
        x_lo = min(cam_r[0], rect_r[:, 0].min()) - 2.0
        x_hi = max(cam_r[0], rect_r[:, 0].max()) + 2.0
        ax.plot([x_lo, x_hi], [0.0, 0.0],
                color="green", lw=2.0, label="Ground")

    ax.plot(rect_r[:, 0], rect_r[:, 1],
            color="grey", linestyle="--", lw=1.4, label="Bounding rectangle")

    ax.plot(fov_tri_r[:, 0], fov_tri_r[:, 1],
            color="#666666", lw=1.6, label=f"FOV")

    ax.annotate(
        "", xy=axis_tip_r, xytext=cam_r,
        arrowprops=dict(arrowstyle="->", color="orange", lw=2.0),
    )
    ax.plot([], [], color="orange", lw=2.0, label="Camera optical axis")

    arc_min = Arc(arc_center, 2 * min_dist, 2 * min_dist,
                  angle=0.0, theta1=-90 + pitch_deg, theta2=90+ pitch_deg,
                  color="red", lw=1.2, linestyle="--")
    arc_max = Arc(arc_center, 2 * max_dist, 2 * max_dist,
                  angle=0.0, theta1=arc_theta1, theta2=arc_theta2,
                  color="red", lw=1.2)
    ax.add_patch(arc_min)
    ax.add_patch(arc_max)
    ax.plot([], [], color="red", lw=1.2, linestyle="--",
            label=f"Min distance")
    ax.plot([], [], color="red", lw=1.2,
            label=f"Max distance")

    ax.scatter([cam_r[0]], [cam_r[1]], color="black", s=60, marker="s",
               zorder=5, label="Camera origin")

    if points is not None:
        pts = np.asarray(points, dtype=float).reshape(-1, 2)
        if len(pts) > 0:
            keep_mask = np.ones(len(pts), dtype=bool)
            if point_categories is not None:
                # Hide FOV-rejected points whose XZ position lies *inside*
                # the rotated FOV triangle. Such points are rejected in 3D
                # only by yaw (the dropped Y axis), so showing them inside
                # the triangle in the side view is misleading.
                # Test by rotating each point back by -pitch around the
                # rectangle center, then checking the un-rotated triangle
                # whose apex is at cam0 and which opens along +X.
                pts_unrot = _rotate_2d(pts, cam0, -pitch)
                rel_x = pts_unrot[:, 0] - cam0[0]
                rel_z = pts_unrot[:, 1] - cam0[1]
                ang = np.arctan2(rel_z, rel_x)
                inside_tri = (rel_x > 0.0) & (np.abs(ang) <= half_v)
                for i, c in enumerate(point_categories):
                    if c == "fov" and inside_tri[i]:
                        keep_mask[i] = False
                colors_full = [CATEGORY_COLORS[c] for c in point_categories]
                colors = [c for c, k in zip(colors_full, keep_mask) if k]
            elif point_color is not None:
                if (isinstance(point_color, (list, tuple, np.ndarray))
                        and len(point_color) == len(pts)):
                    colors = list(point_color)
                else:
                    colors = point_color
            else:
                colors = CATEGORY_COLORS["accepted"]
            pts_show = pts[keep_mask]
            if len(pts_show) > 0:
                ax.scatter(pts_show[:, 0], pts_show[:, 1], c=colors, s=35,
                           edgecolors="k", linewidths=0.5, zorder=6)

    # Point-colour legend (always shown so the viewer understands the colour key).
    _point_legend = [
        ("tab:blue",   "Accepted"),
        ("grey",       "FOV disqualified"),
        ("green",      "Ground disqualified"),
        ("tab:orange", "Distance disqualified"),
    ]
    _header_handle = ax.plot([], [], color="none", label="Sampled points legend:")[0]
    for color, label in _point_legend:
        ax.scatter([], [], color=color, s=35, edgecolors="k", linewidths=0.5,
                   label=label)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    ax.set_title(f"Side-view sampling geometry")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize="small",
              handler_map={_header_handle: _ZeroWidthHandler()})
    if own_fig:
        plt.tight_layout()

    return ax


# ============================================================
# 3D step-by-step explainer of waypoint sampling
# ============================================================


def _draw_distance_arcs(ax, center, radius, color="red", lw=0.9, alpha=0.7,
                         n=80):
    """Draw two great-circle arcs of a sphere: one in the XZ plane and
    one in the XY plane (both pass through `center`). This is much less
    visually noisy than a full wireframe sphere.
    """
    t = np.linspace(0.0, 2.0 * np.pi, n)
    cx, cy, cz = center
    # XZ-plane circle (y = cy)
    ax.plot(cx + radius * np.cos(t),
            np.full_like(t, cy),
            cz + radius * np.sin(t),
            color=color, lw=lw, alpha=alpha)
    # XY-plane circle (z = cz)
    ax.plot(cx + radius * np.cos(t),
            cy + radius * np.sin(t),
            np.full_like(t, cz),
            color=color, lw=lw, alpha=alpha)


def _yaw_pitch_matrix(yaw_deg, pitch_deg):
    """Yaw about world Z, then pitch about world Y.

    Pitch sign matches the 2D side-view convention: negative pitch tips
    the camera optical axis (+X) downward (toward -Z). This is the
    opposite of the right-hand rule about +Y, so we negate `pitch_deg`
    when building Ry.
    """
    Rz = rotation_matrix([0, 0, 1], np.deg2rad(yaw_deg))
    Ry = rotation_matrix([0, 1, 0], -np.deg2rad(pitch_deg))
    return Rz @ Ry


def _rotate_about(points, origin, R):
    pts = np.asarray(points, float).reshape(-1, 3)
    o = np.asarray(origin, float)
    return (pts - o) @ R.T + o


def _draw_camera_marker(ax, origin):
    ax.scatter([origin[0]], [origin[1]], [origin[2]],
               color="black", s=60, marker="s", label="Camera origin")


def _draw_axis_arrow(ax, origin, axis, length, color="orange",
                      label="Camera optical axis"):
    a = normalize(axis) * length
    ax.quiver(origin[0], origin[1], origin[2], a[0], a[1], a[2],
              color=color, linewidth=2.0, arrow_length_ratio=0.12)
    ax.plot([], [], [], color=color, lw=2.0, label=label)


def _prism_vertices(origin, length, width, height):
    ox, oy, oz = origin
    return np.array([
        [ox,           oy - width/2, oz - height/2],
        [ox + length,  oy - width/2, oz - height/2],
        [ox + length,  oy + width/2, oz - height/2],
        [ox,           oy + width/2, oz - height/2],
        [ox,           oy - width/2, oz + height/2],
        [ox + length,  oy - width/2, oz + height/2],
        [ox + length,  oy + width/2, oz + height/2],
        [ox,           oy + width/2, oz + height/2],
    ])


def _draw_prism_wire_verts(ax, verts, color="grey", lw=1.0,
                            label="Bounding prism"):
    for i, j in prism_edges():
        ax.plot([verts[i, 0], verts[j, 0]],
                [verts[i, 1], verts[j, 1]],
                [verts[i, 2], verts[j, 2]],
                color=color, lw=lw)
    if label is not None:
        ax.plot([], [], [], color=color, lw=lw, label=label)


def _draw_prism_wire(ax, origin, length, width, height, color="grey", lw=1.0,
                      label="Bounding prism"):
    verts = _prism_vertices(origin, length, width, height)
    _draw_prism_wire_verts(ax, verts, color=color, lw=lw, label=label)
    return verts


def _draw_fov_pyramid_edges(ax, origin, length, width, height,
                             color="blue", lw=1.0, label="FOV pyramid"):
    """Edges-only FOV pyramid (no face fill)."""
    apex = np.asarray(origin, float)
    far_x = apex[0] + length
    base = np.array([
        [far_x, apex[1] - width/2, apex[2] - height/2],
        [far_x, apex[1] + width/2, apex[2] - height/2],
        [far_x, apex[1] + width/2, apex[2] + height/2],
        [far_x, apex[1] - width/2, apex[2] + height/2],
    ])
    _draw_pyramid_edges_verts(ax, apex, base, color=color, lw=lw, label=label)
    return base


def _draw_pyramid_edges_verts(ax, apex, base, color="blue", lw=1.0,
                                label="FOV pyramid"):
    for c in base:
        ax.plot([apex[0], c[0]], [apex[1], c[1]], [apex[2], c[2]],
                color=color, lw=lw)
    for i in range(4):
        a, b = base[i], base[(i + 1) % 4]
        ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                color=color, lw=lw)
    if label is not None:
        ax.plot([], [], [], color=color, lw=lw, label=label)


def _setup_3d_panel(ax, title, all_pts):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(True)
    P = np.vstack(all_pts) if len(all_pts) else np.zeros((1, 3))
    mins = P.min(axis=0); maxs = P.max(axis=0)
    cent = (mins + maxs) * 0.5
    half = float((maxs - mins).max() * 0.55) or 1.0
    ax.set_xlim(cent[0] - half, cent[0] + half)
    ax.set_ylim(cent[1] - half, cent[1] + half)
    ax.set_zlim(cent[2] - half, cent[2] + half)
    ax.set_box_aspect((1, 1, 1))
    ax.set_proj_type("ortho")
    ax.legend(loc="upper left", fontsize="x-small")


def draw_3d_explainer(
    *,
    demo_points,
    hfov_deg=110.0,
    vfov_deg=45.0,
    min_dist=2.0,
    max_dist=20.0,
    yaw_deg=0,
    pitch_deg=-15.0,
    camera_xyz=(0.0, 0.0, 8.0),
):
    """Step-by-step 3D explainer of the waypoint-sampling algorithm.

    Uses the SAME pre-classified demo points as the 2D side-view
    (``draw_sideview_2d``). Each ``demo_points`` entry is a 4-tuple
    ``(x, y, z, category)`` where ``category`` is one of
    ``'accepted' | 'distance' | 'fov' | 'ground'`` — i.e. which filter
    rejects the point (or ``'accepted'`` if it survives all filters).

    The four panels show:
      1. Bounding prism + all candidate points (blue).
      2. Distance filter shells; distance-rejected points get an X here.
      3. FOV pyramid (edges only); FOV-rejected points get an X here.
      4. Pose rotation + ground; ground-rejected points get an X here.
    Each helper shape (prism / spheres / pyramid / ground) appears in
    only one panel — the panel where it is used to filter.
    Rejected points appear ONLY in the panel that filtered them out.
    """
    cam = np.asarray(camera_xyz, float)
    pts  = np.array([(p[0], p[1], p[2]) for p in demo_points], dtype=float)
    cats = np.array([p[3] for p in demo_points], dtype=object)

    R = _yaw_pitch_matrix(yaw_deg, pitch_deg)
    axis_len = max_dist / 2.0
    length = max_dist
    width  = 2.0 * max_dist * np.sin(np.deg2rad(hfov_deg) / 2.0)
    height = 2.0 * max_dist * np.sin(np.deg2rad(vfov_deg) / 2.0)

    # Helper: which points are still "alive" at the start of panel `p`,
    # i.e. NOT rejected by an earlier filter.
    def alive_at(panel):
        rejected_before = set()
        if panel >= 3:
            rejected_before.add("distance")
        if panel >= 4:
            rejected_before.add("fov")
        return np.array([c not in rejected_before for c in cats])

    fig = plt.figure(figsize=(13, 11))

    # ---------------- Panel 1: bounding prism ----------------
    ax1 = fig.add_subplot(2, 2, 1, projection="3d")
    verts1 = _draw_prism_wire(ax1, cam, length, width, height)
    _draw_axis_arrow(ax1, cam, [1, 0, 0], axis_len)
    _draw_camera_marker(ax1, cam)
    ax1.scatter(pts[:, 0], pts[:, 1], pts[:, 2],
                color=CATEGORY_COLORS["accepted"], s=45,
                edgecolors="k", linewidths=0.5, depthshade=False,
                label=f"Candidate points")
    _setup_3d_panel(ax1, "1. Sampling region (bounding prism)",
                     [verts1, pts])

    # ---------------- Panel 2: distance filter ----------------
    ax2 = fig.add_subplot(2, 2, 2, projection="3d")
    _draw_prism_wire(ax2, cam, length, width, height)
    _draw_axis_arrow(ax2, cam, [1, 0, 0], axis_len)
    _draw_camera_marker(ax2, cam)
    _draw_distance_arcs(ax2, cam, min_dist)
    _draw_distance_arcs(ax2, cam, max_dist)
    ax2.plot([], [], [], color="red", lw=0.9, alpha=0.7,
             label=f"Distance shells ({min_dist:g}, {max_dist:g})")
    alive2 = alive_at(2)
    rej2  = alive2 & (cats == "distance")
    surv2 = alive2 & (cats != "distance")
    if surv2.any():
        ax2.scatter(pts[surv2, 0], pts[surv2, 1], pts[surv2, 2],
                    color=CATEGORY_COLORS["accepted"], s=45,
                    edgecolors="k", linewidths=0.5, depthshade=False,
                    label=CATEGORY_LABELS["accepted"])
    if rej2.any():
        ax2.scatter(pts[rej2, 0], pts[rej2, 1], pts[rej2, 2],
                    color=CATEGORY_COLORS["distance"], marker="x",
                    s=80, lw=2.0, depthshade=False,
                    label=CATEGORY_LABELS["distance"])
    _setup_3d_panel(ax2, "2. Distance filter", [pts])

    # ---------------- Panel 3: FOV pyramid filter ----------------
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    _draw_prism_wire(ax3, cam, length, width, height)
    _draw_axis_arrow(ax3, cam, [1, 0, 0], axis_len)
    _draw_camera_marker(ax3, cam)
    base3 = _draw_fov_pyramid_edges(ax3, cam, length, width, height)
    alive3 = alive_at(3)
    rej3  = alive3 & (cats == "fov")
    surv3 = alive3 & (cats != "fov")
    if surv3.any():
        ax3.scatter(pts[surv3, 0], pts[surv3, 1], pts[surv3, 2],
                    color=CATEGORY_COLORS["accepted"], s=45,
                    edgecolors="k", linewidths=0.5, depthshade=False,
                    label=CATEGORY_LABELS["accepted"])
    if rej3.any():
        ax3.scatter(pts[rej3, 0], pts[rej3, 1], pts[rej3, 2],
                    color=CATEGORY_COLORS["fov"], marker="x",
                    s=80, lw=2.0, depthshade=False,
                    label=CATEGORY_LABELS["fov"])
    _setup_3d_panel(ax3, "3. FOV pyramid filter", [pts, base3])

    # ---------------- Panel 4: rotation + ground filter ----------------
    ax4 = fig.add_subplot(2, 2, 4, projection="3d")
    # Rotated bounding prism (helpers rotate together with the camera).
    verts4 = _prism_vertices(cam, length, width, height)
    verts4_rot = _rotate_about(verts4, cam, R)
    _draw_prism_wire_verts(ax4, verts4_rot)
    axis_dir_rot = R @ np.array([1.0, 0.0, 0.0])
    _draw_axis_arrow(ax4, cam, axis_dir_rot, axis_len)
    _draw_camera_marker(ax4, cam)

    alive4 = alive_at(4)
    rej4  = alive4 & (cats == "ground")
    surv4 = alive4 & (cats != "ground")
    pts_surv_rot = _rotate_about(pts[surv4], cam, R) if surv4.any() else np.empty((0, 3))
    pts_rej_rot  = _rotate_about(pts[rej4],  cam, R) if rej4.any()  else np.empty((0, 3))

    # Ground plane (large translucent green square).
    extent_ref = np.vstack([cam[None, :], pts_surv_rot, pts_rej_rot,
                             cam + axis_dir_rot, verts4_rot])
    g_size = float(np.max(np.abs(extent_ref[:, :2])) * 1.5) + max_dist
    gx, gy = np.meshgrid([-g_size, g_size], [-g_size, g_size])
    gz = np.zeros_like(gx)
    # ax4.plot_surface(gx, gy, gz, color=(0.3, 0.8, 0.3, 0.25),
    #                  linewidth=0, zorder=0)
    # ax4.plot([], [], [], color=(0.3, 0.8, 0.3), lw=8, alpha=0.4,
    #          label="Ground (z=0)")

    # Ground projection (z=0) of every prism-edge segment that lies below
    # the ground plane, drawn as a green dashed line.
    proj_drawn = False
    for i, j in prism_edges():
        A = verts4_rot[i]
        B = verts4_rot[j]
        za, zb = A[2], B[2]
        if za <= 0 and zb <= 0:
            P0, P1 = A, B
        elif za > 0 and zb > 0:
            continue
        else:
            t = za / (za - zb)
            cross = A + t * (B - A)
            P0 = A if za <= 0 else cross
            P1 = B if zb <= 0 else cross
        ax4.plot([P0[0], P1[0]], [P0[1], P1[1]], [0.0, 0.0],
                 color="green", lw=1.4, linestyle="--")
        proj_drawn = True
    if proj_drawn:
        ax4.plot([], [], [], color="green", lw=1.4, linestyle="--",
                 label="Underground edges (projected)")

    if len(pts_surv_rot):
        ax4.scatter(pts_surv_rot[:, 0], pts_surv_rot[:, 1], pts_surv_rot[:, 2],
                    color=CATEGORY_COLORS["accepted"], s=45,
                    edgecolors="k", linewidths=0.5, depthshade=False,
                    label=CATEGORY_LABELS["accepted"])
    if len(pts_rej_rot):
        ax4.scatter(pts_rej_rot[:, 0], pts_rej_rot[:, 1], pts_rej_rot[:, 2],
                    color=CATEGORY_COLORS["ground"], marker="x",
                    s=80, lw=2.0, depthshade=False,
                    label=CATEGORY_LABELS["ground"])

    _setup_3d_panel(
        ax4,
        f"4. Rotation + ground filter",
        [extent_ref],
    )

    fig.suptitle(
        f"3D waypoint sampling example",
        fontsize=12,
    )
    fig.tight_layout()
    return fig


# ============================================================
# Example usage
# ============================================================

def assign_category(
    point,
    *,
    hfov_deg,
    vfov_deg,
    min_dist,
    max_dist,
    yaw_deg,
    pitch_deg,
    camera_xyz,
):
    """Run the waypoint-sampling acceptance algorithm on a single point.

    Returns one of ``"accepted" | "distance" | "fov" | "ground"`` —
    the FIRST filter (in algorithm order) that rejects the point, or
    ``"accepted"`` if it survives all of them.

    Filter order (matches the visual explainer):
      1. distance:  ``min_dist <= ||p - cam|| <= max_dist``
      2. fov:       camera-frame yaw/pitch angles within hfov/2, vfov/2
      3. ground:    after applying the pose rotation, world z >= 0
    """
    p   = np.asarray(point, float)
    cam = np.asarray(camera_xyz, float)
    rel = p - cam

    d = float(np.linalg.norm(rel))
    if d < min_dist or d > max_dist:
        return "distance"

    eps = 1e-12
    x = rel[0]
    if x <= eps:
        return "fov"
    yaw_a   = np.arctan2(rel[1], x)
    pitch_a = np.arctan2(rel[2], x)
    if (abs(yaw_a)   > np.deg2rad(hfov_deg) / 2.0
            or abs(pitch_a) > np.deg2rad(vfov_deg) / 2.0):
        return "fov"

    R = _yaw_pitch_matrix(yaw_deg, pitch_deg)
    p_rot = (rel @ R.T) + cam
    if p_rot[2] < 0.0:
        return "ground"

    return "accepted"


def sample_random_points_in_prism(
    n,
    *,
    hfov_deg,
    vfov_deg,
    max_dist,
    camera_xyz,
    seed=0,
):
    """Uniformly sample ``n`` random points inside the bounding prism.

    The prism extends from the camera origin along +X by ``max_dist``,
    with full extents ``2*max_dist*sin(hfov/2)`` in Y and
    ``2*max_dist*sin(vfov/2)`` in Z.
    """
    rng = np.random.default_rng(seed)
    cam = np.asarray(camera_xyz, float)
    width  = 2.0 * max_dist * np.sin(np.deg2rad(hfov_deg) / 2.0)
    height = 2.0 * max_dist * np.sin(np.deg2rad(vfov_deg) / 2.0)
    x = rng.uniform(0.0,         max_dist,    size=n)
    y = rng.uniform(-width/2.0,  width/2.0,   size=n)
    z = rng.uniform(-height/2.0, height/2.0,  size=n)
    return cam[None, :] + np.stack([x, y, z], axis=1)


# ============================================================
# Example usage
# ============================================================

if __name__ == "__main__":
    # ---- Shared scene parameters ----
    HFOV_DEG    = 110.0
    VFOV_DEG    = 45.0
    MIN_DIST    = 3.0
    MAX_DIST    = 20.0
    YAW_DEG     = 0.0
    PITCH_DEG   = -15.0
    CAMERA_XYZ  = (0.0, 0.0, 8.0)
    N_POINTS    = 60
    SEED        = 1

    # Sample uniformly inside the bounding prism, then classify each
    # point by running the actual acceptance algorithm.
    raw_pts = sample_random_points_in_prism(
        N_POINTS,
        hfov_deg=HFOV_DEG,
        vfov_deg=VFOV_DEG,
        max_dist=MAX_DIST,
        camera_xyz=CAMERA_XYZ,
        seed=SEED,
    )
    demo_points = [
        (float(p[0]), float(p[1]), float(p[2]),
         assign_category(
             p,
             hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG,
             min_dist=MIN_DIST, max_dist=MAX_DIST,
             yaw_deg=YAW_DEG,  pitch_deg=PITCH_DEG,
             camera_xyz=CAMERA_XYZ,
         ))
        for p in raw_pts
    ]

    # ------------------------------------------------------------------
    # 2D side-view: use a SEPARATE 2D point set so the per-point reject
    # category is never lost by the y -> 0 projection. We sample inside
    # the *unrotated* side-view rectangle (camera frame, XZ plane),
    # classify each point with y=0 using the same 3D algorithm, then
    # rotate the points by pitch around the rectangle center so they
    # align with the rotated rectangle drawn by `draw_sideview_2d`.
    # ------------------------------------------------------------------
    N_POINTS_2D = 60
    SEED_2D     = 4
    rng_2d = np.random.default_rng(SEED_2D)
    half_v_2d = np.deg2rad(VFOV_DEG) / 2.0
    height_2d = 2.0 * MAX_DIST * np.sin(half_v_2d)
    # Unrotated rectangle: x in [cam_x, cam_x + max_dist],
    # z in [cam_z - height/2, cam_z + height/2].
    cam_x, cam_z = CAMERA_XYZ[0], CAMERA_XYZ[2]
    xs_2d = rng_2d.uniform(cam_x,              cam_x + MAX_DIST,    size=N_POINTS_2D)
    zs_2d = rng_2d.uniform(cam_z - height_2d/2, cam_z + height_2d/2, size=N_POINTS_2D)

    sample_categories_2d = [
        assign_category(
            (x, 0.0, z),
            hfov_deg=HFOV_DEG, vfov_deg=VFOV_DEG,
            min_dist=MIN_DIST, max_dist=MAX_DIST,
            yaw_deg=YAW_DEG,  pitch_deg=PITCH_DEG,
            camera_xyz=CAMERA_XYZ,
        )
        for x, z in zip(xs_2d, zs_2d)
    ]

    # Rotate the 2D points by pitch around the rectangle center so
    # they sit inside the rotated rectangle drawn in the side view.
    rot_center_2d = np.array([cam_x, cam_z])
    pts_2d_unrot   = np.stack([xs_2d, zs_2d], axis=1)
    sample_xz      = _rotate_2d(pts_2d_unrot, rot_center_2d, np.deg2rad(PITCH_DEG))

    # 2D side-view (auto-coloured from categories).
    ax_2d = draw_sideview_2d(
        hfov_deg=HFOV_DEG,
        vfov_deg=VFOV_DEG,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        camera_xz=(cam_x, cam_z),
        pitch_deg=PITCH_DEG,
        points=sample_xz,
        point_categories=sample_categories_2d,
    )
    fig_2d = ax_2d.figure

    def _save_2d(event):
        save_ax_2d.set_visible(False)
        fig_2d.savefig("waypoint_sideview.pgf", bbox_inches="tight", pad_inches=0.02)
        fig_2d.savefig("waypoint_sideview.pdf", bbox_inches="tight", pad_inches=0.02)
        fig_2d.savefig("waypoint_sideview.svg", bbox_inches="tight", pad_inches=0.02)
        save_ax_2d.set_visible(True)
        fig_2d.canvas.draw_idle()
        print("Saved waypoint_sideview.{pgf,pdf,svg}")

    save_ax_2d = fig_2d.add_axes([0.82, 0.02, 0.12, 0.05])
    save_btn_2d = Button(save_ax_2d, "Save")
    save_btn_2d.on_clicked(_save_2d)

    # 3D step-by-step explainer (same points, same colour scheme).
    fig_3d = draw_3d_explainer(
        demo_points=demo_points,
        hfov_deg=HFOV_DEG,
        vfov_deg=VFOV_DEG,
        min_dist=MIN_DIST,
        max_dist=MAX_DIST,
        yaw_deg=YAW_DEG,
        pitch_deg=PITCH_DEG,
        camera_xyz=CAMERA_XYZ,
    )

    def _save_3d(event):
        save_ax_3d.set_visible(False)
        fig_3d.savefig("waypoint_3d_explainer.pgf", bbox_inches="tight", pad_inches=0.02)
        fig_3d.savefig("waypoint_3d_explainer.pdf", bbox_inches="tight", pad_inches=0.02)
        fig_3d.savefig("waypoint_3d_explainer.svg", bbox_inches="tight", pad_inches=0.02)
        save_ax_3d.set_visible(True)
        fig_3d.canvas.draw_idle()
        print("Saved waypoint_3d_explainer.{pgf,pdf,svg}")

    save_ax_3d = fig_3d.add_axes([0.82, 0.02, 0.12, 0.05])
    save_btn_3d = Button(save_ax_3d, "Save")
    save_btn_3d.on_clicked(_save_3d)

    plt.show()