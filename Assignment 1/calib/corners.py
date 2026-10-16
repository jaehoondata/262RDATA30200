"""Detect checkerboard corners shared by all calibration methods."""
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import config

EXIF_ORIENTATION_TAG = 0x0112


class DatasetError(RuntimeError):
    """Invalid calibration input."""


@dataclass
class CornerDataset:
    image_ids: list
    paths: list
    object_points: np.ndarray   # (n, M, 3) mm
    image_points: np.ndarray    # (n, M, 2) pixels
    image_size: tuple           # (width, height)
    manifest: list              # per-image metadata
    warnings: list = field(default_factory=list)

    def save(self, out_dir):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "manifest.json", "w") as f:
            json.dump({
                "pattern_size_cols_rows": list(config.PATTERN_SIZE),
                "square_size_mm": config.SQUARE_SIZE_MM,
                "image_size_wh": list(self.image_size),
                "read_flags": "IMREAD_IGNORE_ORIENTATION (sensor pixel frame)",
                "n_used": len(self.image_ids),
                "warnings": self.warnings,
                "images": self.manifest,
            }, f, indent=2)
        np.savez(out_dir / "corners.npz", image_ids=np.array(self.image_ids),
                 object_points=self.object_points, image_points=self.image_points,
                 image_size=np.array(self.image_size))


def board_object_points(pattern_size=config.PATTERN_SIZE, square_size=config.SQUARE_SIZE_MM):
    """Create board coordinates in detector order."""
    cols, rows = pattern_size
    grid = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2).astype(np.float64) * square_size
    return np.column_stack([grid, np.zeros(len(grid))])


def list_images(image_dir):
    """List readable and unsupported image files."""
    files = sorted(p for p in Path(image_dir).iterdir() if p.is_file() and not p.name.startswith("."))
    readable = [p for p in files if p.suffix.lower() in config.IMAGE_EXTENSIONS]
    unsupported = [p for p in files if p.suffix.lower() in config.UNSUPPORTED_EXTENSIONS]
    return readable, unsupported


def read_exif_orientation(path):
    """Read EXIF orientation when available."""
    try:
        with Image.open(path) as im:
            return im.getexif().get(EXIF_ORIENTATION_TAG)
    except OSError:
        return None


def read_image(path, gray=False):
    """Load pixels without EXIF rotation."""
    mode = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    return cv2.imread(str(path), mode | cv2.IMREAD_IGNORE_ORIENTATION)


def _min_corner_spacing(corners, pattern_size):
    cols, rows = pattern_size
    grid = corners.reshape(rows, cols, 2)
    dx = np.linalg.norm(np.diff(grid, axis=1), axis=-1)
    dy = np.linalg.norm(np.diff(grid, axis=0), axis=-1)
    return float(min(dx.min(), dy.min()))


def detect_corners(gray, pattern_size=config.PATTERN_SIZE):
    """Detect and refine inner corners."""
    h, w = gray.shape
    scale = min(1.0, config.DETECTION_MAX_DIM / max(h, w))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else gray

    def classic(img):
        return cv2.findChessboardCorners(img, pattern_size, flags=config.CHESSBOARD_FLAGS)

    def sector_based(img):
        return cv2.findChessboardCornersSB(img, pattern_size, flags=config.CHESSBOARD_SB_FLAGS)

    # Try scaled detection before full resolution.
    attempts = [("findChessboardCorners", classic, scale), ("findChessboardCornersSB", sector_based, scale)]
    if scale < 1:
        attempts.append(("findChessboardCornersSB", sector_based, 1.0))
    found = False
    for detector, find, scale in attempts:
        found, corners = find(small if scale < 1 else gray)
        if found:
            break
    info = {"detection_scale": scale if found else None, "detector": detector if found else None}
    if not found:
        return None, info

    corners = corners.reshape(-1, 2).astype(np.float64)
    if scale < 1:  # restore full-resolution pixel centers
        corners = (corners + 0.5) / scale - 0.5
    spacing = _min_corner_spacing(corners, pattern_size)
    half = int(np.clip(round(config.SUBPIX_WINDOW_RATIO * spacing),
                       config.SUBPIX_WINDOW_MIN, config.SUBPIX_WINDOW_MAX))
    refined = cv2.cornerSubPix(gray, corners.astype(np.float32).reshape(-1, 1, 2),
                               (half, half), (-1, -1), config.SUBPIX_CRITERIA)
    info.update({"subpix_half_window": half, "min_corner_spacing_px": spacing})
    return refined.reshape(-1, 2).astype(np.float64), info


def draw_corner_overlay(image, corners, pattern_size=config.PATTERN_SIZE):
    """Mark corners, the board origin and +X direction."""
    out = image.copy()
    cv2.drawChessboardCorners(out, pattern_size, corners.astype(np.float32).reshape(-1, 1, 2), True)
    thick = max(2, int(max(out.shape[:2]) / 400))
    p0 = tuple(int(round(v)) for v in corners[0])
    p1 = tuple(int(round(v)) for v in corners[1])
    cv2.circle(out, p0, 6 * thick, (0, 0, 255), thick)
    cv2.arrowedLine(out, p0, p1, (0, 0, 255), thick, tipLength=0.3)
    cv2.putText(out, "0", (p0[0] + 8 * thick, p0[1] - 8 * thick), cv2.FONT_HERSHEY_SIMPLEX,
                0.6 * thick, (0, 0, 255), thick)
    return out


def build_dataset(image_dir=config.IMAGE_DIR, overlay_dir=None,
                  pattern_size=config.PATTERN_SIZE, square_size=config.SQUARE_SIZE_MM):
    """Build shared checkerboard correspondences."""
    image_dir = Path(image_dir)
    if not image_dir.is_dir():
        raise DatasetError(f"Image folder does not exist: {image_dir}")
    readable, unsupported = list_images(image_dir)
    if unsupported:
        names = ", ".join(p.name for p in unsupported[:5])
        raise DatasetError(f"{len(unsupported)} HEIC/HEIF file(s) cannot be read by OpenCV ({names}...). "
                           "Convert them to JPEG while preserving dimensions and orientation, "
                           "then rerun with only the converted JPEG files in the input folder.")
    if not readable:
        raise DatasetError(f"No images found in {image_dir}. Put the checkerboard photos there "
                           f"({', '.join(config.IMAGE_EXTENSIONS)}).")
    if overlay_dir is not None:
        Path(overlay_dir).mkdir(parents=True, exist_ok=True)

    X = board_object_points(pattern_size, square_size)
    manifest, used, warnings = [], [], []
    for path in readable:
        entry = {"image_id": path.name, "file": str(path), "exif_orientation": read_exif_orientation(path)}
        manifest.append(entry)
        image = read_image(path)
        if image is None:
            entry.update(status="excluded", reason="read failure")
            continue
        entry.update(width=image.shape[1], height=image.shape[0])
        corners, info = detect_corners(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), pattern_size)
        entry.update(info)
        if corners is None:
            entry.update(status="excluded", reason=f"checkerboard {pattern_size} not detected")
            continue
        entry.update(status="used", reason=None)
        used.append((path, corners))
        if overlay_dir is not None:
            cv2.imwrite(str(Path(overlay_dir) / f"{path.stem}_corners.jpg"), draw_corner_overlay(image, corners, pattern_size))

    sizes = Counter((e["width"], e["height"]) for e in manifest if "width" in e)
    if len(sizes) > 1:
        raise DatasetError(f"Images have different sizes {dict(sizes)}; mixing pixel frames would "
                           "break the shared K. Remove or re-export the odd images.")
    orientations = Counter(e["exif_orientation"] for e in manifest)
    if len(orientations) > 1:
        warnings.append(f"Mixed EXIF orientation tags {dict(orientations)}; pixels are read in the "
                        "sensor frame, but check that no image was rotated when exported.")
    if len(used) < config.MIN_VIEWS_GENERAL:
        raise DatasetError(f"Only {len(used)} image(s) with a detected checkerboard; at least "
                           f"{config.MIN_VIEWS_GENERAL} are needed (10+ recommended). See manifest reasons.")
    if len(used) < config.RECOMMENDED_MIN_VIEWS:
        warnings.append(f"Only {len(used)} usable images; the instruction asks for at least "
                        f"{config.RECOMMENDED_MIN_VIEWS} photos.")

    (w, h), = sizes.keys()
    return CornerDataset(
        image_ids=[p.name for p, _ in used],
        paths=[p for p, _ in used],
        object_points=np.array([X] * len(used)),
        image_points=np.array([c for _, c in used]),
        image_size=(w, h),
        manifest=manifest,
        warnings=warnings,
    )
