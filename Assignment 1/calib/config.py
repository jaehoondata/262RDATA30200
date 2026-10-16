"""Calibration settings."""
from pathlib import Path

import cv2

# Use CPU processing.
cv2.ocl.setUseOpenCL(False)

ROOT = Path(__file__).resolve().parent.parent
IMAGE_DIR = ROOT / "image"
OUTPUT_DIR = ROOT / "outputs"

# Checkerboard geometry.
PATTERN_SIZE = (13, 9)  # inner corners: columns, rows
SQUARE_SIZE_MM = 20.0

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
# Formats requiring conversion.
UNSUPPORTED_EXTENSIONS = (".heic", ".heif")

# Minimum view counts and assignment requirement.
MIN_VIEWS_GENERAL = 3
MIN_VIEWS_ZERO_SKEW = 2
RECOMMENDED_MIN_VIEWS = 10

# Corner detection and refinement.
DETECTION_MAX_DIM = 2000
CHESSBOARD_FLAGS = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
# Fallback detector flags.
CHESSBOARD_SB_FLAGS = cv2.CALIB_CB_EXHAUSTIVE
# Refinement window relative to corner spacing.
SUBPIX_WINDOW_RATIO = 0.3
SUBPIX_WINDOW_MIN = 3
SUBPIX_WINDOW_MAX = 15
SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-4)

# Coordinate scaling and rank checks.
ZHANG_PRESCALE = True
ZHANG_RANK_TOL = 1e-9

# Distortion models for each OpenCV condition.
OPENCV_FLAGS = {
    "CV5": 0,
    "CV0": (cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3
          | cv2.CALIB_ZERO_TANGENT_DIST),
    "CV2": cv2.CALIB_FIX_K3 | cv2.CALIB_ZERO_TANGENT_DIST,
}
OPENCV_CRITERIA = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 30,
                   2.220446049250313e-16)  # OpenCV defaults

ALL_CONDITIONS = ("Z0", "Z_gamma", "CV5", "CV0", "LM0", "LM2", "CV2")
DEFAULT_CONDITIONS = ("Z0", "Z_gamma", "CV5")
