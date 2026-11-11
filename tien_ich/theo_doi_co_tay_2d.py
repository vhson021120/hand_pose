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
        self._image_size = None

    def reset(self) -> None:
        self._points.clear()

    def update(self, detections: Mapping[str, Sequence[float]], image_size=None) -> Dict[str, WristPoint]:
        # Pixel histories are meaningful only within the same frame dimensions.
        if image_size is not None:
            size = finite_values(image_size, 2)
            if size is None or min(size) <= 0:
                self.reset()
                self._image_size = None
                return {}
            if self._image_size != size:
                self.reset()
                self._image_size = size
        current: Dict[str, np.ndarray] = {}
        if not isinstance(detections, Mapping):
            self.reset()
            return {}
        for side in ("LEFT", "RIGHT"):
            detection = detections.get(side)
            if detection is None:
                continue
            values = finite_values(detection, 3)
            if values is None:
                continue
            point = np.asarray(values, dtype=np.float64)
            if (
                point.shape != (3,)
                or not np.isfinite(point).all()
                or point[2] < self.min_visibility
                or not 0 <= point[2] <= 1
            ):
                continue
            previous = self._points.get(side)
            with np.errstate(over="ignore", invalid="ignore"):
                filtered = (
                    point.copy()
                    if previous is None
                    else previous + self.smoothing * (point - previous)
                )
            if not np.isfinite(filtered).all():
                continue
            current[side] = filtered

        self._points = current
        return {
            side: (float(point[0]), float(point[1]), float(point[2]))
            for side, point in current.items()
        }


def finite_values(values, length):
    """Return a finite flat numeric tuple, or None for malformed observations."""
    try:
        point = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError, OverflowError):
        return None
    if point.shape != (length,) or not np.isfinite(point).all():
        return None
    return tuple(float(value) for value in point)


def wrist_to_workspace_mm(
    point: Sequence[float],
    roi: Tuple[float, float, float, float],
    work_mm: Tuple[float, float],
) -> Tuple[float, float, bool]:
    """Map an image point to workspace mm (origin at the ROI center, Y up).

    Values are clamped to the workspace edges; ``inside`` tells whether the
    point was really inside the ROI.
    """
    try:
        coordinates = finite_values(point[:2], 2) if len(point) in (2, 3) else None
    except (TypeError, ValueError, IndexError):
        coordinates = None
    bounds = finite_values(roi, 4)
    work = finite_values(work_mm, 2)
    if coordinates is None or bounds is None or work is None:
        return 0.0, 0.0, False
    x, y = coordinates
    left, top, right, bottom = bounds
    if right <= left or bottom <= top or min(work) <= 0:
        return 0.0, 0.0, False
    # Guard arithmetic overflow without changing the mapping for valid inputs.
    if not all(math.isfinite(value) for value in
               (right - left, bottom - top, x - left, y - top)):
        return 0.0, 0.0, False
    u = min(max((x - left) / (right - left), 0.0), 1.0)
    v = min(max((y - top) / (bottom - top), 0.0), 1.0)
    inside = left <= x <= right and top <= y <= bottom
    return (u - 0.5) * work[0], (0.5 - v) * work[1], inside


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
