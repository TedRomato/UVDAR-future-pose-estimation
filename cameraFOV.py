from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D plots)

# ----------------------------- helpers -----------------------------

def v3(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(3)

def unit(v: np.ndarray) -> np.ndarray:
    v = v3(v)
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Zero-length vector")
    return v / n

def clamp(x: float, a: float = -1.0, b: float = 1.0) -> float:
    return max(a, min(b, x))

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    u, v = unit(u), unit(v)
    return float(np.arccos(clamp(float(np.dot(u, v)))))

def proj_onto_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    n = unit(n)
    return v3(v) - float(np.dot(v, n)) * n

def signed_angle_in_plane(ref: np.ndarray, vec: np.ndarray, plane_normal: np.ndarray) -> float:
    r = unit(ref)
    w = unit(vec)
    k = unit(plane_normal)
    sin_term = float(np.dot(k, np.cross(r, w)))
    cos_term = float(np.dot(r, w))
    return float(np.arctan2(sin_term, cos_term))

# ----------------------------- config -----------------------------

@dataclass(frozen=True)
class SamplingConfig:
    distance_min: float = 1.0
    distance_max: float = 50.0
    max_trials_per_point: int = 1000
    min_z: float = 1.0     # minimum allowed Z (e.g., drone safety height)
    ground_z: float = 0.0  # ground visualization height

# ----------------------------- main class --------------------------

@dataclass
class PyramidFOV:
    C: np.ndarray
    f: np.ndarray
    mid_top: np.ndarray
    mid_bottom: np.ndarray
    mid_left: np.ndarray
    mid_right: np.ndarray
    k_tb: np.ndarray
    k_lr: np.ndarray
    tb_half_rad: float
    lr_half_rad: float
    edge_tl: np.ndarray
    edge_tr: np.ndarray
    edge_bl: np.ndarray
    edge_br: np.ndarray

    # ---------- constructors ----------

    @staticmethod
    def _find_axis(v_tl, v_tr, v_bl, v_br) -> np.ndarray:
        v_tl, v_tr, v_bl, v_br = map(unit, (v_tl, v_tr, v_bl, v_br))
        return unit(v_tl + v_tr + v_bl + v_br)

    @staticmethod
    def _face_middles(v_tl, v_tr, v_bl, v_br):
        v_tl, v_tr, v_bl, v_br = map(unit, (v_tl, v_tr, v_bl, v_br))
        return (
            unit(v_tl + v_tr),
            unit(v_bl + v_br),
            unit(v_tl + v_bl),
            unit(v_tr + v_br),
        )

    @staticmethod
    def _plane_normals(mid_top, mid_bottom, mid_left, mid_right):
        return unit(np.cross(mid_top, mid_bottom)), unit(np.cross(mid_left, mid_right))

    @staticmethod
    def _half_ranges_from_middles(axis, mid_top, mid_bottom, mid_left, mid_right):
        proj_top    = float(np.dot(axis, mid_top))    * mid_top
        proj_bottom = float(np.dot(axis, mid_bottom)) * mid_bottom
        proj_left   = float(np.dot(axis, mid_left))   * mid_left
        proj_right  = float(np.dot(axis, mid_right))  * mid_right
        tb_full = angle_between(proj_top, proj_bottom)
        lr_full = angle_between(proj_left, proj_right)
        return tb_full * 0.5, lr_full * 0.5

    @classmethod
    def from_2_edge_points(
        cls,
        C,
        P_tl,
        P_br,
        up_hint: np.ndarray = np.array([0.0, 0.0, 1.0]),
    ) -> "PyramidFOV":
        """
        Build a rectangular, no-roll frustum from two *opposite* edge points
        (top-left and bottom-right) measured at arbitrary depths.

        Assumptions:
        - No roll (the TB plane is aligned using up_hint).
        - Rectangular symmetry (HFOV != VFOV allowed).
        - Axis is perpendicular to the base.
        """
        C = v3(C)
        e_tl = unit(v3(P_tl) - C)
        e_br = unit(v3(P_br) - C)

        # 1) Axis (bisector of opposite corners)
        f = unit(e_tl + e_br)

        # 2) No-roll basis from world up hint (project up into plane ⟂ f)
        u = proj_onto_plane(up_hint, f)
        if np.linalg.norm(u) < 1e-12:
            raise ValueError("up_hint is parallel to axis; choose a different up_hint.")
        u = unit(u)
        r = unit(np.cross(u, f))  # right-handed

        # 3) Decompose the TL ray into this basis
        c = float(np.dot(e_tl, f))  # forward component
        a = float(np.dot(e_tl, r))  # LR component (TL typically has a<0)
        b = float(np.dot(e_tl, u))  # TB component (TL typically has b>0)

        # 4) Reconstruct the other two corner rays using rectangular sign pattern:
        # TL: ( +c,  +a,  +b )
        # TR: ( +c, -a,  +b )
        # BR: ( +c, -a,  -b )  <-- given
        # BL: ( +c, +a,  -b )
        e_tr = unit(c * f - a * r + b * u)
        e_bl = unit(c * f + a * r - b * u)

        # 5) Hand off to the standard constructor (rays can be any nonzero length)
        return cls.from_edge_rays(C, e_tl, e_tr, e_bl, e_br)

    @classmethod
    def from_4_edge_points(cls, C, P_tl, P_tr, P_bl, P_br) -> "PyramidFOV":
        C = v3(C)
        v_tl, v_tr, v_bl, v_br = (v3(P_tl)-C, v3(P_tr)-C, v3(P_bl)-C, v3(P_br)-C)
        return cls.from_edge_rays(C, v_tl, v_tr, v_bl, v_br)

    @classmethod
    def from_edge_rays(cls, C, v_tl, v_tr, v_bl, v_br) -> "PyramidFOV":
        C = v3(C)
        e_tl, e_tr, e_bl, e_br = map(unit, (v_tl, v_tr, v_bl, v_br))   # <-- store unit corner rays
        f = cls._find_axis(v_tl, v_tr, v_bl, v_br)
        mid_top, mid_bottom, mid_left, mid_right = cls._face_middles(v_tl, v_tr, v_bl, v_br)
        k_tb, k_lr = cls._plane_normals(mid_top, mid_bottom, mid_left, mid_right)
        tb_half, lr_half = cls._half_ranges_from_middles(f, mid_top, mid_bottom, mid_left, mid_right)
        return cls(C=C, f=f,
                    mid_top=mid_top, mid_bottom=mid_bottom, mid_left=mid_left, mid_right=mid_right,
                    k_tb=k_tb, k_lr=k_lr, tb_half_rad=tb_half, lr_half_rad=lr_half,
                    edge_tl=e_tl, edge_tr=e_tr, edge_bl=e_bl, edge_br=e_br)

    # ---------- angles ----------

    def tb_angle_of(self, direction: np.ndarray) -> float:
        d = v3(direction)
        a_tb = proj_onto_plane(self.f, self.k_tb)
        d_tb = proj_onto_plane(d,     self.k_tb)
        return signed_angle_in_plane(a_tb, d_tb, self.k_tb)

    def lr_angle_of(self, direction: np.ndarray) -> float:
        d = v3(direction)
        a_lr = proj_onto_plane(self.f, self.k_lr)
        d_lr = proj_onto_plane(d,     self.k_lr)
        return signed_angle_in_plane(a_lr, d_lr, self.k_lr)

    # ---------- containment ----------

    def euclidean_distance(self, P: np.ndarray) -> float:
        return float(np.linalg.norm(v3(P) - self.C))

    def contains(self, P: np.ndarray, cfg: SamplingConfig) -> bool:
        d_vec = v3(P) - self.C
        tb = self.tb_angle_of(d_vec)
        lr = self.lr_angle_of(d_vec)
        if abs(tb) > self.tb_half_rad or abs(lr) > self.lr_half_rad:
            return False

        dist = self.euclidean_distance(P)
        if not (cfg.distance_min <= dist <= cfg.distance_max):
            return False

        # safety height
        if P[2] < cfg.min_z:
            return False

        return True

    # ---------- sampling ----------

    def _lateral_dirs(self):
        u_dir = unit(proj_onto_plane(self.mid_top,  self.k_tb))
        r_dir = unit(proj_onto_plane(self.mid_right, self.k_lr))
        u_dir = unit(proj_onto_plane(u_dir, self.f))
        r_dir = unit(proj_onto_plane(r_dir, self.f))
        return r_dir, u_dir

    def _sample_in_bounding_prism(self, cfg: SamplingConfig) -> np.ndarray:
        r_dir, u_dir = self._lateral_dirs()

        # Corner-safe near bound for Euclidean radius:
        s_lr = np.tan(self.lr_half_rad)
        s_tb = np.tan(self.tb_half_rad)
        cos_corner = 1.0 / np.sqrt(1.0 + s_lr**2 + s_tb**2)
        z_min = cos_corner * cfg.distance_min

        z_max = cfg.distance_max
        x_max = s_lr * z_max
        y_max = s_tb * z_max

        x = np.random.uniform(-x_max, +x_max)
        y = np.random.uniform(-y_max, +y_max)
        z = np.random.uniform(z_min,  z_max)

        return self.C + z * self.f + x * r_dir + y * u_dir

    def sample_point(self, cfg: SamplingConfig = SamplingConfig()) -> np.ndarray:
        trials = 0
        while trials < cfg.max_trials_per_point:
            trials += 1
            P = self._sample_in_bounding_prism(cfg)
            if self.contains(P, cfg):
                return P 

        print("Too many trials to generate a point...")
        return []             
            


    def sample_points(self, n_points: int, cfg: SamplingConfig = SamplingConfig()) -> np.ndarray:
        pts = []

        while len(pts) < n_points:

            result = self.sample_point(cfg)
            if len(result):
                pts.append(result)
                

        if len(pts) == 0:
            return np.empty((0, 3))
        return np.vstack(pts)

    # ---------- visualization ----------

    def visualize(
        self,
        pts: np.ndarray | None = None,
        cfg: SamplingConfig | None = None,
        extra_points: np.ndarray | None = None,
        extra_label: str = "Input edge points",
    ):
        """
        Render camera, ground plane, frustum (with ground clipping and projection),
        and optional sampled points and extra points.

        - Edges that would go below the ground are clipped at z=ground_z.
        - The underground parts are replaced by their projection on the ground plane
        (dashed green “shadow”).
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        GZ = 0.0
        if cfg is not None and hasattr(cfg, "ground_z"):
            GZ = float(cfg.ground_z)

        # --- helpers for clipping / projection ---
        def intersect_with_ground(A, B, gz: float):
            """Return intersection point of segment AB with z=gz plane, or None if no intersection within [0,1]."""
            Az, Bz = A[2], B[2]
            denom = (Bz - Az)
            if abs(denom) < 1e-12:
                return None
            t = (gz - Az) / denom
            if 0.0 <= t <= 1.0:
                return A + t * (B - A)
            return None

        def ground_proj(P, gz: float):
            """Vertical projection of P onto ground z=gz."""
            return np.array([P[0], P[1], gz], dtype=float)

        def draw_segment_with_ground(ax, A, B, gz: float, color="gray", lw=1.5,
                                    proj_color=(0.2, 0.7, 0.2), proj_ls="--"):
            """
            Draw a line segment AB clipped against z=gz.
            Any below-ground portion is drawn as a dashed projection on the ground plane.
            """
            A = np.asarray(A, float); B = np.asarray(B, float)
            Az, Bz = A[2], B[2]

            if Az >= gz and Bz >= gz:
                # fully above ground
                ax.plot([A[0], B[0]], [A[1], B[1]], [A[2], B[2]], color=color, lw=lw)
                return [A, B], []  # (drawn_points, projected_points)

            if Az < gz and Bz < gz:
                # fully below ground -> draw ground projection only
                A2, B2 = ground_proj(A, gz), ground_proj(B, gz)
                ax.plot([A2[0], B2[0]], [A2[1], B2[1]], [A2[2], B2[2]],
                        color=proj_color, lw=lw, ls=proj_ls)
                return [], [A2, B2]

            # segment crosses ground once
            I = intersect_with_ground(A, B, gz)
            if I is None:
                # numerically parallel to ground: choose safe fallback
                return [], []

            if Az >= gz and Bz < gz:
                # draw visible A->I, project I->B on ground
                ax.plot([A[0], I[0]], [A[1], I[1]], [A[2], I[2]], color=color, lw=lw)
                B2 = ground_proj(B, gz)
                I2 = ground_proj(I, gz)
                ax.plot([I2[0], B2[0]], [I2[1], B2[1]], [I2[2], B2[2]],
                        color=proj_color, lw=lw, ls=proj_ls)
                return [A, I], [I2, B2]

            else:  # Az < gz and Bz >= gz
                # draw visible I->B, project A->I on ground
                ax.plot([I[0], B[0]], [I[1], B[1]], [I[2], B[2]], color=color, lw=lw)
                A2 = ground_proj(A, gz)
                I2 = ground_proj(I, gz)
                ax.plot([A2[0], I2[0]], [A2[1], I2[1]], [A2[2], I2[2]],
                        color=proj_color, lw=lw, ls=proj_ls)
                return [I, B], [A2, I2]

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        # Orthographic projection (no perspective foreshortening)
        try:
            ax.set_proj_type("ortho")
        except Exception:
            pass  # older matplotlib versions

        # Camera center
        ax.scatter(*self.C, color="black", s=40, label="Camera center")

        # Forward axis
        axis_len = (cfg.distance_max if cfg else 5.0)
        axis_end = self.C + self.f * axis_len
        ax.plot(
            [self.C[0], axis_end[0]],
            [self.C[1], axis_end[1]],
            [self.C[2], axis_end[2]],
            color="red",
            lw=2,
            label="Axis",
        )

        corners = None
        # Frustum edges via single-scalar scaling of unit edge rays (ideal symmetric/no-roll)
        if cfg is not None:
            d = cfg.distance_max
            cos_theta = float(np.dot(self.f, self.edge_tl))
            if abs(cos_theta) < 1e-12:
                cos_theta = 1e-12
            s = d / cos_theta

            corners = [
                self.C + s * self.edge_tl,  # TL
                self.C + s * self.edge_tr,  # TR
                self.C + s * self.edge_br,  # BR
                self.C + s * self.edge_bl,  # BL
            ]

            # (Optional) print numeric diagnostics
            apex_lengths = [np.linalg.norm(c - self.C) for c in corners]
            side_lengths = [np.linalg.norm(corners[i] - corners[(i + 1) % 4]) for i in range(4)]
            print("[viz] apex→corner lengths:", np.round(apex_lengths, 6))
            print("[viz] far-rect side lengths:", np.round(side_lengths, 6))

            # ---- Draw ground plane (semi-transparent green) ----
            size = d * 2  # extent
            xx, yy = np.meshgrid(
                np.linspace(-size, size, 2),
                np.linspace(-size, size, 2)
            )
            zz = np.full_like(xx, GZ)
            ax.plot_surface(xx, yy, zz, color=(0.3, 0.8, 0.3, 0.25), linewidth=0, zorder=0)

            # ---- Draw apex→corner edges with ground clipping/projection ----
            drawn_pts = [self.C.copy()]
            proj_pts = []
            for c in corners:
                drawn, proj = draw_segment_with_ground(ax, self.C, c, GZ, color="gray", lw=1.6,
                                                    proj_color=(0.2, 0.7, 0.2), proj_ls="--")
                drawn_pts.extend(drawn)
                proj_pts.extend(proj)

            # ---- Draw the far rectangle edges with the same clipping/projection ----
            for i in range(4):
                c1, c2 = corners[i], corners[(i + 1) % 4]
                drawn, proj = draw_segment_with_ground(ax, c1, c2, GZ, color="gray", lw=1.4,
                                                    proj_color=(0.2, 0.7, 0.2), proj_ls="--")
                drawn_pts.extend(drawn); proj_pts.extend(proj)

        # Sampled points
        if pts is not None and len(pts) > 0:
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=8, alpha=0.5, color="blue", label="Sampled points")

        # Extra points to show (e.g., input edge points)
        if extra_points is not None and len(extra_points) > 0:
            ep = np.asarray(extra_points, dtype=float).reshape(-1, 3)
            ax.scatter(
                ep[:, 0], ep[:, 1], ep[:, 2],
                s=40, color="limegreen", edgecolors="k", linewidths=0.5,
                label=extra_label,
            )

        # Labels
        ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")

        # ---- Equalize axis limits (so equal lengths look equal on screen) ----
        all_pts = [self.C]
        if corners is not None:
            all_pts.extend(corners)
        if pts is not None and len(pts) > 0:
            all_pts.append(pts)
        if extra_points is not None and len(extra_points) > 0:
            all_pts.append(np.asarray(extra_points, float).reshape(-1, 3))

        # include projections/intersections if we created them
        try:
            if drawn_pts:
                all_pts.append(np.vstack(drawn_pts))
        except NameError:
            pass
        try:
            if proj_pts:
                all_pts.append(np.vstack(proj_pts))
        except NameError:
            pass

        P = np.vstack(all_pts) if len(all_pts) > 1 else np.asarray(self.C, float).reshape(1, 3)
        mins = P.min(axis=0); maxs = P.max(axis=0)
        cent = (mins + maxs) * 0.5
        half = float((maxs - mins).max() * 0.55) or 1.0  # pad a bit, avoid zero
        ax.set_xlim(cent[0] - half, cent[0] + half)
        ax.set_ylim(cent[1] - half, cent[1] + half)
        ax.set_zlim(cent[2] - half, cent[2] + half)
        ax.set_box_aspect([1, 1, 1])

        ax.legend()
        plt.tight_layout()
        plt.show()

# ----------------------------- demo 1 - no rotation-----------------------------
'''
if __name__ == "__main__":
    hfov_deg, vfov_deg = 70.0, 50.0
    ax = np.tan(np.deg2rad(hfov_deg) / 2.0)
    ay = np.tan(np.deg2rad(vfov_deg) / 2.0)

    C = np.array([0.0, 0.0, 0.0])
    v_tl = np.array([-ax,  ay, 1.0])
    v_tr = np.array([+ax,  ay, 1.0])
    v_bl = np.array([-ax, -ay, 1.0])
    v_br = np.array([+ax, -ay, 1.0])

    fov = PyramidFOV.from_edge_rays(C, v_tl, v_tr, v_bl, v_br)
    cfg = SamplingConfig(distance_min=25.0, distance_max=30.0)
    pts = fov.sample_points(1000, cfg)
    print("Sampled:", pts.shape)

    #pts = np.vstack(list(filter(lambda pt: (pt[0] < 0.1 and pt[0] > -0.1) or (pt[1] < 0.1 and pt[1] > -0.1),pts)))

    fov.visualize(pts, cfg)
'''
    
# ----------------------------- demo 2 - yaw + pitch  -----------------------------
'''
if __name__ == "__main__":
    # ---------- General-view demo: yaw + pitch (no roll), using points ----------
    np.random.seed(0)

    # FOV
    hfov_deg, vfov_deg = 70.0, 50.0
    t_lr = np.tan(np.deg2rad(hfov_deg) / 2.0)
    t_tb = np.tan(np.deg2rad(vfov_deg) / 2.0)

    # Camera center
    C = np.array([3,1,-2])

    # World up (to keep "no roll")
    world_up = np.array([0.0, 0.0, 1.0])

    # Start with canonical camera basis (looking +Z)
    f0 = np.array([0.0, 1.0, 0.0])

    # Small helper: Rodrigues rotation
    def rodrigues(v, k, theta):
        v = np.asarray(v, float); k = unit(k)
        c, s = np.cos(theta), np.sin(theta)
        return v*c + np.cross(k, v)*s + k*np.dot(k, v)*(1.0 - c)

    # Yaw (around world up), then pitch (around camera right), no roll
    yaw_deg, pitch_deg = 35.0, -15.0
    f_yaw = unit(rodrigues(f0, world_up, np.deg2rad(yaw_deg)))
    # Right axis after yaw
    r_yaw = unit(np.cross(world_up, f_yaw))
    f = unit(rodrigues(f_yaw, r_yaw, np.deg2rad(pitch_deg)))
    # Up consistent with no roll (span of world_up & f)
    u = unit(np.cross(f, r_yaw))  # this preserves "no roll"
    r = unit(np.cross(u, f))

    # Choose a depth at which to place the four input edge points
    depth_for_points = 7.5
    x_off = t_lr * depth_for_points
    y_off = t_tb * depth_for_points


    # Different depths per corner (measure these by hand in your setup)
    d_tl, d_tr, d_bl, d_br = 6.0, 9.5, 12.0, 7.2

    # IMPORTANT: lateral offsets scale with each depth!
    # That keeps the ray direction identical regardless of how far you go.
    P_tl = C + d_tl * (f + (-t_lr)*r + (+t_tb)*u)
    P_tr = C + d_tr * (f + (+t_lr)*r + (+t_tb)*u)
    P_bl = C + d_bl * (f + (-t_lr)*r + (-t_tb)*u)
    P_br = C + d_br * (f + (+t_lr)*r + (-t_tb)*u)

    # Create the FOV from the points
    # fov = PyramidFOV.from_4_edge_points(C, P_tl, P_tr, P_bl, P_br)
    fov = PyramidFOV.from_2_edge_points(C, P_tr, P_bl)

    # Sample inside with Euclidean radius limits
    cfg = SamplingConfig(distance_min=25.0, distance_max=30.0)
    pts = fov.sample_points(1000, cfg)

    # Render and include the four input edge points
    input_points = np.vstack([P_tl, P_tr, P_bl, P_br])
    fov.visualize(pts, cfg, extra_points=input_points, extra_label="Input edge points")
'''


# ----------------------------- demo 3 - real view -----------------------------


if __name__ == "__main__":
    C = [0.013, -0.031, 1.3248]         # camera
    P_tl = [3.7255, 1.7220, 3.676]      # Point visible on top left edge of FOV
    P_br = [-4.396, -0.2436, -1.014]    # Point visible on bottom right edge of FOV

    fov = PyramidFOV.from_2_edge_points(C, P_tl, P_br)
    cfg = SamplingConfig(distance_min=6.0, distance_max=15.0, min_z=1.0, ground_z=0.0)
    pts = fov.sample_points(800, cfg)

    input_points = np.vstack([P_tl, P_br])
    fov.visualize(pts, cfg, extra_points=input_points, extra_label="Input edge points")


