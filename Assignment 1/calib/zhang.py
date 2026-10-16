"""Linear Zhang calibration with optional zero skew."""
import numpy as np

from . import config
from .camera import to_camera
from .result import CalibrationResult


class ZhangError(RuntimeError):
    """Invalid calibration solution."""


# Homography

def normalization_matrix(points):
    """Center points and scale their mean distance to sqrt(2)."""
    pts = np.asarray(points, dtype=np.float64)
    mean = pts.mean(axis=0)
    mean_dist = np.linalg.norm(pts - mean, axis=1).mean()
    if not np.isfinite(mean_dist) or mean_dist <= 0:
        raise ZhangError("Degenerate point set: all points coincide")
    s = np.sqrt(2.0) / mean_dist
    return np.array([[s, 0.0, -s * mean[0]], [0.0, s, -s * mean[1]], [0.0, 0.0, 1.0]])


def _homogeneous(points):
    return np.column_stack([points, np.ones(len(points))])


def estimate_homography(board_xy, image_uv):
    """Estimate the board-to-image homography using normalized DLT."""
    board_xy = np.asarray(board_xy, dtype=np.float64)
    image_uv = np.asarray(image_uv, dtype=np.float64)
    if board_xy.ndim != 2 or board_xy.shape[1] != 2 or board_xy.shape != image_uv.shape:
        raise ValueError(f"Expected matching (N, 2) arrays, got {board_xy.shape} and {image_uv.shape}")
    if len(board_xy) < 4:
        raise ZhangError(f"A homography needs at least 4 points, got {len(board_xy)}")

    T_board = normalization_matrix(board_xy)
    T_image = normalization_matrix(image_uv)
    b = _homogeneous(board_xy) @ T_board.T
    m = _homogeneous(image_uv) @ T_image.T
    b = b[:, :2] / b[:, 2:]
    m = m[:, :2] / m[:, 2:]

    X, Y = b[:, 0], b[:, 1]
    u, v = m[:, 0], m[:, 1]
    zeros, ones = np.zeros_like(X), np.ones_like(X)
    A = np.empty((2 * len(X), 9))
    A[0::2] = np.column_stack([X, Y, ones, zeros, zeros, zeros, -u * X, -u * Y, -u])
    A[1::2] = np.column_stack([zeros, zeros, zeros, X, Y, ones, -v * X, -v * Y, -v])
    _, _, Vt = np.linalg.svd(A)
    H_norm = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T_image) @ H_norm @ T_board
    if abs(H[2, 2]) < 1e-12 * np.linalg.norm(H):
        raise ZhangError("H[2,2] is ~0: the board origin maps to infinity")
    return H / H[2, 2]


# Intrinsics

def _v(H, i, j):
    """Build a Zhang constraint row from homography columns."""
    hi, hj = H[:, i], H[:, j]
    return np.array([
        hi[0] * hj[0],
        hi[0] * hj[1] + hi[1] * hj[0],
        hi[1] * hj[1],
        hi[2] * hj[0] + hi[0] * hj[2],
        hi[2] * hj[1] + hi[1] * hj[2],
        hi[2] * hj[2],
    ])


def build_V(homographies):
    """Stack two intrinsic constraints per view."""
    rows = []
    for H in homographies:
        H = H / np.linalg.norm(H)  # Normalize homography scale.
        rows.append(_v(H, 0, 1))
        rows.append(_v(H, 0, 0) - _v(H, 1, 1))
    return np.array(rows)


def prescale_matrix(image_size):
    """Prescale pixel coordinates while preserving zero skew."""
    w, h = image_size
    s = 2.0 / max(w, h)
    return np.array([[s, 0.0, -s * w / 2.0], [0.0, s, -s * h / 2.0], [0.0, 0.0, 1.0]])


def intrinsics_from_b(b):
    """Recover intrinsics from the constraint solution."""
    b = np.asarray(b, dtype=np.float64)
    if b[0] < 0:  # Choose a positive B11.
        b = -b
    B11, B12, B22, B13, B23, B33 = b
    B = np.array([[B11, B12, B13], [B12, B22, B23], [B13, B23, B33]])
    eig = np.linalg.eigvalsh(B)
    if not np.all(eig > 0):
        raise ZhangError(f"B is not positive definite (eigenvalues {eig}); "
                         "check pose diversity and corner correspondences")

    delta = B11 * B22 - B12 ** 2
    v0 = (B12 * B13 - B11 * B23) / delta
    lam = B33 - (B13 ** 2 + v0 * (B12 * B13 - B11 * B23)) / B11
    if lam / B11 <= 0 or lam * B11 / delta <= 0:
        raise ZhangError(f"Negative square-root argument (lambda={lam:.3e}, B11={B11:.3e}, delta={delta:.3e})")
    alpha = np.sqrt(lam / B11)
    beta = np.sqrt(lam * B11 / delta)
    gamma = -B12 * alpha ** 2 * beta / lam
    u0 = gamma * v0 / beta - B13 * alpha ** 2 / lam
    K = np.array([[alpha, gamma, u0], [0.0, beta, v0], [0.0, 0.0, 1.0]])
    if not np.all(np.isfinite(K)):
        raise ZhangError(f"Non-finite intrinsic matrix: {K}")
    return K


def solve_b(V, zero_skew=False):
    """Solve the intrinsic constraints with optional zero skew."""
    n_unknown = 5 if zero_skew else 6
    Vs = np.delete(V, 1, axis=1) if zero_skew else V
    if Vs.shape[0] < n_unknown - 1:
        raise ZhangError(f"{Vs.shape[0]} equations cannot determine {n_unknown} unknowns up to scale")
    _, sv, Vt = np.linalg.svd(Vs)
    rel = sv / sv[0]
    # Reject degenerate view configurations.
    if rel[n_unknown - 2] < config.ZHANG_RANK_TOL:
        raise ZhangError("V is rank deficient beyond its 1-D null space "
                         f"(relative singular values {rel}); the board poses are degenerate")
    b = Vt[-1]
    if zero_skew:
        b = np.insert(b, 1, 0.0)
    return b, sv


def estimate_intrinsics(homographies, zero_skew=False, image_size=None, prescale=None):
    """Estimate intrinsics from multiple homographies."""
    prescale = config.ZHANG_PRESCALE if prescale is None else prescale
    min_views = config.MIN_VIEWS_ZERO_SKEW if zero_skew else config.MIN_VIEWS_GENERAL
    if len(homographies) < min_views:
        raise ZhangError(f"{'Zero-skew' if zero_skew else 'General'} Zhang needs at least "
                         f"{min_views} views, got {len(homographies)}")
    N = np.eye(3)
    if prescale:
        if image_size is None:
            raise ValueError("image_size is required when prescale=True")
        N = prescale_matrix(image_size)
    V = build_V([N @ H for H in homographies])
    b, sv = solve_b(V, zero_skew=zero_skew)
    K = np.linalg.inv(N) @ intrinsics_from_b(b)
    K /= K[2, 2]
    return K, {"singular_values_V": sv, "prescale": bool(prescale), "b": b}


# Extrinsics

def orthonormalize(Q):
    """Find the nearest rotation matrix."""
    U, _, Vt = np.linalg.svd(Q)
    return U @ np.diag([1.0, 1.0, np.linalg.det(U @ Vt)]) @ Vt


def estimate_extrinsics(K, H):
    """Recover one view's rotation and translation."""
    q = np.linalg.inv(K) @ H
    s = 1.0 / np.linalg.norm(q[:, 0])
    if s * q[2, 2] < 0:  # Keep the board in front of the camera.
        s = -s
    r1, r2, t = s * q[:, 0], s * q[:, 1], s * q[:, 2]
    R = orthonormalize(np.column_stack([r1, r2, np.cross(r1, r2)]))
    return R, t


def check_pose(R, t, object_points):
    """Validate the recovered rotation and board depth."""
    ortho_err = float(np.linalg.norm(R.T @ R - np.eye(3)))
    det = float(np.linalg.det(R))
    min_z = float(to_camera(object_points, R, t)[:, 2].min())
    if not (np.all(np.isfinite(t)) and ortho_err < 1e-9 and abs(det - 1) < 1e-9):
        raise ZhangError(f"Invalid pose: |R^T R - I|={ortho_err:.2e}, det={det:.6f}, t={t}")
    if min_z <= 0:
        raise ZhangError(f"Board point behind the camera (min Z_camera = {min_z:.3f} mm)")
    return {"orthogonality_error": ortho_err, "det_R": det, "min_z_camera_mm": min_z}


# Calibration

def calibrate_zhang(object_points, image_points, image_size, image_ids=None,
                    zero_skew=False, prescale=None):
    """Calibrate planar board points (mm) against image corners (px)."""
    object_points = np.asarray(object_points, dtype=np.float64)
    image_points = np.asarray(image_points, dtype=np.float64)
    if object_points.ndim != 3 or object_points.shape[2] != 3:
        raise ValueError(f"object_points must be (n, M, 3), got {object_points.shape}")
    if image_points.shape != object_points.shape[:2] + (2,):
        raise ValueError(f"image_points must be (n, M, 2) matching object_points, got {image_points.shape}")
    if len(object_points) == 0:
        raise ZhangError("No views given")
    if np.any(np.abs(object_points[:, :, 2]) > 1e-9):
        raise ValueError("Zhang's method assumes a planar board with Z = 0")
    image_ids = list(range(len(object_points))) if image_ids is None else list(image_ids)

    Hs = [estimate_homography(X[:, :2], x) for X, x in zip(object_points, image_points)]
    K, info = estimate_intrinsics(Hs, zero_skew=zero_skew, image_size=image_size, prescale=prescale)

    rotations, translations, pose_checks = [], [], []
    for iid, H, X in zip(image_ids, Hs, object_points):
        R, t = estimate_extrinsics(K, H)
        try:
            pose_checks.append(check_pose(R, t, X))
        except ZhangError as e:
            raise ZhangError(f"Image {iid}: {e}") from e
        rotations.append(R)
        translations.append(t)

    condition = "Z0" if zero_skew else "Z_gamma"
    model = ("linear Zhang, zero skew imposed (B12 = 0), no distortion" if zero_skew
             else "linear Zhang, skew estimated, no distortion")
    info.update({"homographies": Hs, "pose_checks": pose_checks})
    return CalibrationResult(condition, K, None, rotations, np.array(translations), image_ids, model, info)
