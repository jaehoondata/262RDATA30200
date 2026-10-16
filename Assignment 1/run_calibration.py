"""Run the calibration conditions on image/ and write outputs/.

    python run_calibration.py                          # Z0 Z_gamma CV5
    python run_calibration.py --conditions Z0 Z_gamma CV5 CV0 LM0 LM2 CV2
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from calib import config
from calib import visualization as viz
from calib.corners import DatasetError, build_dataset, read_image
from calib.metrics import (extrinsics_rows, intrinsics_rows, reprojection,
                           rmse_from_residuals)
from calib.opencv_baseline import calibrate_opencv
from calib.refine import RefinementError, refine
from calib.zhang import ZhangError, calibrate_zhang

# Relative tolerance between our RMSE and the RMS returned by calibrateCamera.
RMS_MATCH_RTOL = 1e-4


def ordered_conditions(requested):
    """Canonical order; LM0 and LM2 need Z0 as their initial value."""
    wanted = set(requested)
    if wanted & {"LM0", "LM2"}:
        wanted.add("Z0")
    return [c for c in config.ALL_CONDITIONS if c in wanted]


def calibrate(cond, ds, results):
    args = (ds.object_points, ds.image_points)
    if cond in ("Z0", "Z_gamma"):
        return calibrate_zhang(*args, ds.image_size, ds.image_ids, zero_skew=(cond == "Z0"))
    if cond in config.OPENCV_FLAGS:
        return calibrate_opencv(*args, ds.image_size, cond, ds.image_ids)
    return refine(results["Z0"], *args, condition=cond)


def write_csv(rows, path):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def example_indices(n, count=3):
    return sorted(set(np.linspace(0, n - 1, min(n, count)).round().astype(int).tolist()))


def run(conditions=config.DEFAULT_CONDITIONS, image_dir=config.IMAGE_DIR, output_dir=config.OUTPUT_DIR):
    output_dir = Path(output_dir)
    fig_dir = output_dir / "figures"
    ds = build_dataset(image_dir, overlay_dir=fig_dir / "corners")
    ds.save(output_dir)
    for w in ds.warnings:
        print(f"WARNING: {w}")

    results, projected, per_image, overall, warnings = {}, {}, {}, {}, list(ds.warnings)
    for cond in ordered_conditions(conditions):
        res = calibrate(cond, ds, results)
        proj, resid = reprojection(res, ds.object_points, ds.image_points)
        per_image[cond], overall[cond] = rmse_from_residuals(resid)
        res.extra["rmse_overall_px"] = overall[cond]
        res.extra["rmse_per_image_px"] = dict(zip(ds.image_ids, per_image[cond].tolist()))
        if "rms_reported_by_opencv" in res.extra:
            reported = res.extra["rms_reported_by_opencv"]
            match = abs(overall[cond] - reported) <= RMS_MATCH_RTOL * max(reported, 1e-12)
            res.extra["rmse_matches_opencv_rms"] = bool(match)
            if not match:
                warnings.append(f"{cond}: own RMSE {overall[cond]:.6f} != OpenCV RMS {reported:.6f}")
        res.save(output_dir / f"calibration_{cond}.json")
        results[cond], projected[cond] = res, proj

    reference = "CV5" if "CV5" in results else None
    write_csv(intrinsics_rows(results, reference), output_dir / "intrinsics_comparison.csv")
    write_csv(extrinsics_rows(results, reference), output_dir / "extrinsics_comparison.csv")
    err_rows = [{"image_id": iid, **{c: per_image[c][i] for c in results}} for i, iid in enumerate(ds.image_ids)]
    err_rows.append({"image_id": "OVERALL", **overall})
    write_csv(err_rows, output_dir / "reprojection_errors.csv")

    viz.plot_rmse_bars(ds.image_ids, per_image, overall, fig_dir / "reprojection_rmse.png")
    overlay_conds = [c for c in ("Z0", "Z_gamma", "CV5") if c in results] or list(results)
    overlay_conds = overlay_conds[:viz.MAX_OVERLAY_CONDITIONS]
    examples = example_indices(len(ds.image_ids))
    for i in examples:
        image = read_image(ds.paths[i])
        stem = Path(ds.image_ids[i]).stem
        viz.plot_reprojection(image, ds.image_points[i], {c: projected[c][i] for c in overlay_conds},
                              fig_dir / f"reprojection_{stem}.png", ds.image_ids[i])
    undistortion_diagnostics = {}
    for cond in ("CV5", "LM2", "CV2"):  # only conditions that actually estimated distortion
        if cond in results:
            i = examples[0]
            model_label = results[cond].model.replace("Z0", viz.CONDITION_LABELS["Z0"])
            undistortion_diagnostics[cond] = viz.plot_undistortion(
                read_image(ds.paths[i]), results[cond].K, results[cond].dist,
                fig_dir / f"undistortion_{cond}_{Path(ds.image_ids[i]).stem}.png",
                f"{viz.CONDITION_LABELS[cond]}: {model_label}", coverage_points=ds.image_points)
    with open(output_dir / "undistortion_diagnostics.json", "w") as f:
        json.dump(undistortion_diagnostics, f, indent=2)
    return {"results": results, "overall_rmse": overall, "warnings": warnings, "dataset": ds}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--conditions", nargs="+", choices=config.ALL_CONDITIONS,
                        default=list(config.DEFAULT_CONDITIONS))
    args = parser.parse_args()
    try:
        out = run(args.conditions)
    except (DatasetError, ZhangError, RefinementError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"{len(out['dataset'].image_ids)} images used. Outputs in {config.OUTPUT_DIR}")
    print(f"{'cond':8} {'fx':>10} {'fy':>10} {'cx':>9} {'cy':>9} {'skew':>8} {'RMSE px':>9}")
    for cond, res in out["results"].items():
        k = res.intrinsics
        print(f"{cond:8} {k['fx']:10.2f} {k['fy']:10.2f} {k['cx']:9.2f} {k['cy']:9.2f} "
              f"{k['skew']:8.3f} {out['overall_rmse'][cond]:9.4f}")
    for w in out["warnings"]:
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
