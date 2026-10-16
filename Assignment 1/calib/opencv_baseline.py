"""OpenCV calibration for CV5, CV0 and CV2."""
import cv2
import numpy as np

from . import config
from .result import CalibrationResult

MODELS = {
    "CV5": "OpenCV calibrateCamera, zero skew, k1 k2 p1 p2 k3",
    "CV0": "OpenCV calibrateCamera, zero skew, distortion fixed to 0",
    "CV2": "OpenCV calibrateCamera, zero skew, k1 k2 only",
}


def calibrate_opencv(object_points, image_points, image_size, condition="CV5", image_ids=None):
    """Estimate camera parameters and poses from shared corners."""
    if condition not in config.OPENCV_FLAGS:
        raise ValueError(f"Unknown OpenCV condition {condition!r}")
    flags = config.OPENCV_FLAGS[condition]
    obj = [np.asarray(X, dtype=np.float32) for X in object_points]
    img = [np.asarray(x, dtype=np.float32).reshape(-1, 1, 2) for x in image_points]
    dist0 = np.zeros((5, 1))
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj, img, tuple(int(v) for v in image_size), None, dist0,
        flags=flags, criteria=config.OPENCV_CRITERIA)

    image_ids = list(range(len(obj))) if image_ids is None else list(image_ids)
    rotations = [cv2.Rodrigues(r)[0] for r in rvecs]
    translations = np.array([t.ravel() for t in tvecs])
    extra = {
        "rms_reported_by_opencv": float(rms),
        "flags": int(flags),
        "criteria": list(config.OPENCV_CRITERIA),
        "image_size_wh": list(image_size),
        "opencv_version": cv2.__version__,
    }
    return CalibrationResult(condition, K, dist.ravel(), rotations, translations,
                             image_ids, MODELS[condition], extra)
