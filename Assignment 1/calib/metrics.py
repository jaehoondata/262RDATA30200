"""Reprojection errors and camera parameter comparisons."""
import numpy as np

from .camera import matrix_to_rvec, project_points


def reprojection(result, object_points, image_points):
    """Compute projections and observed-minus-projected residuals."""
    projected, residuals = [], []
    for X, x, R, t in zip(object_points, image_points, result.rotations, result.translations):
        p = project_points(X, result.K, R, t, result.dist)
        projected.append(p)
        residuals.append(np.asarray(x) - p)
    return np.array(projected), np.array(residuals)


def rmse_from_residuals(residuals):
    """Compute per-view and pooled RMSE per corner."""
    sq = np.sum(np.asarray(residuals) ** 2, axis=-1)  # (n, M) squared Euclidean
    per_image = np.sqrt(sq.mean(axis=1))
    overall = float(np.sqrt(sq.sum() / sq.size))
    return per_image, overall


def reprojection_rmse(result, object_points, image_points):
    _, res = reprojection(result, object_points, image_points)
    return rmse_from_residuals(res)


def rotation_angle_deg(R1, R2):
    """Compute the relative rotation angle in degrees."""
    return float(np.degrees(np.linalg.norm(matrix_to_rvec(R1.T @ R2))))


def intrinsics_rows(results, reference="CV5"):
    """Compare intrinsic parameters against the reference."""
    keys = ("fx", "fy", "cx", "cy", "skew")
    ref = results.get(reference)
    rows = []
    for cond, res in results.items():
        row = {"condition": cond, "model": res.model}
        vals = res.intrinsics
        for k in keys:
            row[k] = vals[k]
        for name, value in zip(("k1", "k2", "p1", "p2", "k3"), res.dist):
            row[name] = value
        if ref is not None and cond != reference:
            ref_vals = ref.intrinsics
            for k in keys:
                row[f"{k}_diff_vs_{reference}"] = vals[k] - ref_vals[k]
            for k in ("fx", "fy"):
                row[f"{k}_reldiff_pct_vs_{reference}"] = 100.0 * (vals[k] - ref_vals[k]) / ref_vals[k]
        rows.append(row)
    return rows


def extrinsics_rows(results, reference="CV5"):
    """Compare per-image poses against the reference."""
    ref = results.get(reference)
    rows = []
    for cond, res in results.items():
        for i, (iid, R, t) in enumerate(zip(res.image_ids, res.rotations, res.translations)):
            rv = matrix_to_rvec(R)
            row = {"image_id": iid, "condition": cond,
                   "rvec_x": rv[0], "rvec_y": rv[1], "rvec_z": rv[2],
                   "t_x_mm": t[0], "t_y_mm": t[1], "t_z_mm": t[2]}
            if ref is not None and cond != reference:
                row[f"rot_diff_deg_vs_{reference}"] = rotation_angle_deg(R, ref.rotations[i])
                row[f"t_diff_mm_vs_{reference}"] = float(np.linalg.norm(t - ref.translations[i]))
            rows.append(row)
    return rows
