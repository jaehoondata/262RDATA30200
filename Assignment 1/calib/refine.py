"""Joint camera and pose refinement with SciPy LM."""
import numpy as np
from scipy.optimize import least_squares

from .camera import camera_matrix, matrix_to_rvec, project_points, rvec_to_matrix, to_camera
from .result import CalibrationResult

MODELS = {
    "LM0": "custom projection with SciPy LM from Z0, zero skew, no distortion",
    "LM2": "custom projection with SciPy LM from Z0, zero skew, k1 k2",
}


class RefinementError(RuntimeError):
    pass


def _unpack(p, n_views, n_dist):
    fx, fy, cx, cy = p[:4]
    dist = np.zeros(5)
    dist[:n_dist] = p[4:4 + n_dist]
    poses = p[4 + n_dist:].reshape(n_views, 6)
    return camera_matrix(fx, fy, cx, cy), dist, poses


def refine(initial, object_points, image_points, condition="LM0"):
    """Refine camera parameters and poses from Z0."""
    if condition not in MODELS:
        raise ValueError(f"Unknown refinement condition {condition!r}")
    if abs(initial.K[0, 1]) > 0:
        raise ValueError("Refinement uses a zero-skew model; start from Z0, not Z_gamma")
    n_dist = 2 if condition == "LM2" else 0
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    n = len(object_points)

    K0 = initial.K
    poses0 = [np.concatenate([matrix_to_rvec(R), t]) for R, t in zip(initial.rotations, initial.translations)]
    p0 = np.concatenate([[K0[0, 0], K0[1, 1], K0[0, 2], K0[1, 2]], np.zeros(n_dist), np.ravel(poses0)])

    def residuals(p):
        K, dist, poses = _unpack(p, n, n_dist)
        out = [project_points(X, K, rvec_to_matrix(pose[:3]), pose[3:], dist) - x
               for X, x, pose in zip(object_points, image_points, poses)]
        return np.concatenate(out).ravel()

    sol = least_squares(residuals, p0, method="lm", x_scale="jac",
                        ftol=1e-12, xtol=1e-12, gtol=1e-12)
    K, dist, poses = _unpack(sol.x, n, n_dist)
    rotations = [rvec_to_matrix(pose[:3]) for pose in poses]
    translations = poses[:, 3:]

    if not sol.success:
        raise RefinementError(f"{condition}: optimizer did not converge ({sol.message})")
    if K[0, 0] <= 0 or K[1, 1] <= 0:
        raise RefinementError(f"{condition}: non-positive focal length {K[0, 0]:.3f}, {K[1, 1]:.3f}")
    min_z = min(to_camera(X, R, t)[:, 2].min() for X, R, t in zip(object_points, rotations, translations))
    if min_z <= 0:
        raise RefinementError(f"{condition}: board point behind the camera after refinement")

    r0 = residuals(p0)
    extra = {
        "initialization": "Z0 (linear zero-skew Zhang), distortion 0",
        "optimizer": "scipy.optimize.least_squares(method='lm')",
        "status": int(sol.status),
        "message": sol.message,
        "n_function_evals": int(sol.nfev),
        "initial_cost": float(0.5 * r0 @ r0),
        "final_cost": float(sol.cost),
        "min_z_camera_mm": float(min_z),
    }
    return CalibrationResult(condition, K, dist, rotations, translations,
                             list(initial.image_ids), MODELS[condition], extra)
