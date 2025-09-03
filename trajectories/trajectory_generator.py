#!/usr/bin/env python3
import numpy as np

def wrap_angle(a, mode="0_2pi"):
    """Wrap angle array after unwrapping. mode: '0_2pi' or 'pm_pi'."""
    if mode == "0_2pi":
        return np.mod(a, 2*np.pi)
    # [-pi, pi)
    return (a + np.pi) % (2*np.pi) - np.pi

def helix_xyzh(
    radius=6.0,
    z_start=1.0,
    z_end=9.0,
    dz_per_turn=0.15,
    max_step=0.2,
    start_angle=0.0,          # radians; where on the circle to start
    heading_mode="tangent",   # 'tangent' | 'radial_out' | 'radial_in' | 'fixed'
    fixed_heading=0.0,        # used if heading_mode == 'fixed' (radians)
    heading_range="0_2pi"     # '0_2pi' or 'pm_pi'
):
    """
    Returns Nx4 array: [x, y, z, heading]
    - XY is a perfect circle of given radius.
    - z increases linearly from z_start to z_end.
    - Point spacing is chosen so 3D distance between consecutive points <= max_step.
    """
    # Total turns & angle
    total_height = float(z_end - z_start)
    turns = total_height / float(dz_per_turn)
    total_angle = 2.0 * np.pi * turns

    # Arc-length per radian for a circular helix is constant:
    dz_dtheta = dz_per_turn / (2.0 * np.pi)                  # vertical climb per radian
    ds_dtheta = np.sqrt(radius**2 + dz_dtheta**2)            # helix metric

    # Choose number of points so that chord length <= max_step
    total_length = ds_dtheta * total_angle
    n_pts = int(np.ceil(total_length / max_step)) + 1
    n_pts = max(n_pts, 2)

    # Parameter
    theta = start_angle + np.linspace(0.0, total_angle, n_pts, dtype=np.float64)

    # Positions (exact circle in XY, linear in Z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    z = np.linspace(z_start, z_end, n_pts, dtype=np.float64)

    # Heading
    if heading_mode == "tangent":
        # Tangent direction in XY: atan2(dy, dx) with continuous phase
        # Using analytic derivatives keeps it smooth
        dx = -radius * np.sin(theta)
        dy =  radius * np.cos(theta)
        yaw = np.arctan2(dy, dx)
    elif heading_mode == "radial_out":
        # Face away from origin
        yaw = np.arctan2(y, x)
    elif heading_mode == "radial_in":
        # Face toward origin
        yaw = np.arctan2(-y, -x)
    elif heading_mode == "fixed":
        yaw = np.full_like(x, float(fixed_heading))
    else:
        raise ValueError("Unknown heading_mode")

    # Unwrap to make yaw continuous, then wrap to chosen range
    yaw = np.unwrap(yaw)
    yaw = wrap_angle(yaw, heading_range)

    traj = np.column_stack((x, y, z, yaw))

    # --- Diagnostics for peace of mind ---
    steps = np.linalg.norm(np.diff(traj[:, :3], axis=0), axis=1)
    print(f"[helix_xyzh] points: {n_pts}, max_step: {steps.max():.4f} m (target ≤ {max_step})")
    print(f"[helix_xyzh] z monotonic increasing: {bool(np.all(np.diff(z) >= 0))}")

    return traj

if __name__ == "__main__":
    traj = helix_xyzh(
        radius=5.0,
        z_start=0.0,
        z_end=10.0,
        dz_per_turn=1.0,
        max_step=0.4,
        start_angle=0.0,
        heading_mode="tangent",   # try 'radial_out' if you want to look outward
        heading_range="0_2pi"
    )

    # Save in the requested format: x y z heading (spaces, no header)
    np.savetxt("helix_xyzh.txt", traj, fmt="%.6f")

    # Print a small preview
    for row in traj[:10]:
        print(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.2f} {row[3]:.2f}")
    print("Saved helix_xyzh.txt")
