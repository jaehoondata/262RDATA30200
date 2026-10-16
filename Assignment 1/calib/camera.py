"""Camera projection, distortion and rotation conversions."""
import numpy as np
from scipy.spatial.transform import Rotation

N_DIST = 5


def as_dist5(dist):
    """Normalize distortion to (k1, k2, p1, p2, k3)."""
    if dist is None:
        return np.zeros(N_DIST)
    d = np.asarray(dist, dtype=np.float64).ravel()
    if d.size > N_DIST:
        if np.any(d[N_DIST:] != 0):
            raise ValueError(f"Only the 5-coefficient model is supported, got {d.size} non-zero-padded coefficients")
        d = d[:N_DIST]
    return np.pad(d, (0, N_DIST - d.size))


def rvec_to_matrix(rvec):
    return Rotation.from_rotvec(np.asarray(rvec, dtype=np.float64).ravel()).as_matrix()


def matrix_to_rvec(R):
    return Rotation.from_matrix(np.asarray(R, dtype=np.float64)).as_rotvec()


def to_camera(object_points, R, t):
    """Transform board points to camera coordinates."""
    X = np.asarray(object_points, dtype=np.float64)
    return X @ np.asarray(R).T + np.asarray(t, dtype=np.float64).ravel()


def distort_normalized(xy, dist):
    """Apply distortion to normalized coordinates."""
    k1, k2, p1, p2, k3 = as_dist5(dist)
    x, y = xy[:, 0], xy[:, 1]
    r2 = x * x + y * y
    radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return np.column_stack([xd, yd])


def project_points(object_points, K, R, t, dist=None):
    """Project board points to pixels with skew and distortion."""
    Xc = to_camera(object_points, R, t)
    xy = Xc[:, :2] / Xc[:, 2:3]
    xy = distort_normalized(xy, dist)
    K = np.asarray(K, dtype=np.float64)
    u = K[0, 0] * xy[:, 0] + K[0, 1] * xy[:, 1] + K[0, 2]
    v = K[1, 1] * xy[:, 1] + K[1, 2]
    return np.column_stack([u, v])


def camera_matrix(fx, fy, cx, cy, skew=0.0):
    return np.array([[fx, skew, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
