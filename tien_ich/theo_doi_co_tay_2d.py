"""Filter observed 2D wrist points and project them onto a centered XY plane."""

import math
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


WristPoint = Tuple[float, float, float]


class WristXYTracker:
    def __init__(self, smoothing: float = 0.25, min_visibility: float = 0.7):
        if not math.isfinite(smoothing) or not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be finite and in (0, 1]")
        if not math.isfinite(min_visibility) or not 0.0 <= min_visibility <= 1.0:
            raise ValueError("min_visibility must be finite and in [0, 1]")
        self.smoothing = smoothing
        self.min_visibility = min_visibility
        self._points: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._points.clear()

    def update(self, detections: Mapping[str, Sequence[float]]) -> Dict[str, WristPoint]:
        current: Dict[str, np.ndarray] = {}
        for side in ("LEFT", "RIGHT"):
            detection = detections.get(side)
            if detection is None:
                continue
            point = np.asarray(detection, dtype=np.float64)
            if (
                point.shape != (3,)
                or not np.isfinite(point).all()
                or point[2] < self.min_visibility
            ):
                continue
            previous = self._points.get(side)
            filtered = (
                point.copy()
                if previous is None
                else previous + self.smoothing * (point - previous)
            )
            current[side] = filtered

        self._points = current
        return {
            side: (float(point[0]), float(point[1]), float(point[2]))
            for side, point in current.items()
        }


def screen_pixel_to_centered_mm(
    point: Sequence[float],
    image_size: Tuple[int, int],
    mm_per_pixel: float,
) -> Tuple[float, float]:
    width, height = image_size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if not math.isfinite(mm_per_pixel) or mm_per_pixel <= 0:
        raise ValueError("mm_per_pixel must be finite and positive")
    x, y = map(float, point[:2])
    if not math.isfinite(x) or not math.isfinite(y):
        raise ValueError("point coordinates must be finite")
    return (
        (x - width / 2.0) * mm_per_pixel,
        (height / 2.0 - y) * mm_per_pixel,
    )