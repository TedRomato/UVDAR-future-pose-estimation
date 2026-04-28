"""Minimal SE(3) helpers for the clean parser.

Quaternions are (qx, qy, qz, qw). Transforms are 4x4 NumPy matrices in
REP-105 convention: a transform with parent=A, child=B maps points from
B to A, i.e. it IS T_A_from_B.
"""

import math
import numpy as np


def quat_to_R(qx, qy, qz, qw):
    n = qx * qx + qy * qy + qz * qz + qw * qw
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (qy * qy + qz * qz),     s * (qx * qy - qz * qw),     s * (qx * qz + qy * qw)],
        [    s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz),     s * (qy * qz - qx * qw)],
        [    s * (qx * qz - qy * qw),     s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy)],
    ])


def R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        return ((R[2, 1] - R[1, 2]) * s, (R[0, 2] - R[2, 0]) * s,
                (R[1, 0] - R[0, 1]) * s, 0.25 / s)
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2])
        return (0.25 * s, (R[0, 1] + R[1, 0]) / s,
                (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s)
    if R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2])
        return ((R[0, 1] + R[1, 0]) / s, 0.25 * s,
                (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s)
    s = 2.0 * math.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1])
    return ((R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
            0.25 * s, (R[1, 0] - R[0, 1]) / s)


def make_T(x, y, z, qx, qy, qz, qw):
    T = np.eye(4)
    T[:3, :3] = quat_to_R(qx, qy, qz, qw)
    T[:3, 3] = (x, y, z)
    return T


def T_to_xyzquat(T):
    x, y, z = T[:3, 3]
    qx, qy, qz, qw = R_to_quat(T[:3, :3])
    return float(x), float(y), float(z), qx, qy, qz, qw


def tf_msg_to_T(tf_msg):
    tr = tf_msg.transform.translation
    r = tf_msg.transform.rotation
    return make_T(tr.x, tr.y, tr.z, r.x, r.y, r.z, r.w)
