"""Store and serialize calibration results."""
import json
from dataclasses import dataclass, field

import numpy as np

from .camera import as_dist5, matrix_to_rvec


@dataclass
class CalibrationResult:
    condition: str
    K: np.ndarray                 # (3, 3)
    dist: np.ndarray              # (5,) k1, k2, p1, p2, k3
    rotations: list               # per image (3, 3)
    translations: np.ndarray      # (n_images, 3), mm
    image_ids: list
    model: str                    # model label
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.K = np.asarray(self.K, dtype=np.float64)
        self.dist = as_dist5(self.dist)
        self.rotations = [np.asarray(R, dtype=np.float64) for R in self.rotations]
        self.translations = np.asarray(self.translations, dtype=np.float64).reshape(-1, 3)

    @property
    def intrinsics(self):
        K = self.K
        return {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2], "skew": K[0, 1]}

    def to_dict(self):
        return {
            "condition": self.condition,
            "model": self.model,
            "units": {"K": "pixel", "translation": "mm", "rvec": "rad (Rodrigues)"},
            "pose_convention": "X_camera = R @ X_board + t",
            "K": self.K.tolist(),
            "intrinsics": {k: float(v) for k, v in self.intrinsics.items()},
            "dist_k1_k2_p1_p2_k3": self.dist.tolist(),
            "extrinsics": [
                {"image_id": iid, "R": R.tolist(), "rvec": matrix_to_rvec(R).tolist(), "t": t.tolist()}
                for iid, R, t in zip(self.image_ids, self.rotations, self.translations)
            ],
            "extra": _jsonable(self.extra),
        }

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    return obj
