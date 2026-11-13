"""Filter observed 2D wrist points and project them onto a centered XY plane."""

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

from tien_ich.ket_noi_yumi import utc_timestamp


WRIST_SPECS = (("LEFT", (70, 220, 110)), ("RIGHT", (220, 90, 220)))


WristPoint = Tuple[float, float, float]


@dataclass(frozen=True)
class WristMotionState:
    point_mm: Tuple[float, float]
    delta_mm: Tuple[float, float]
    moved: bool = False
    initialized: bool = False


class WristMotionGate:
    """Hold each accepted wrist until either axis exceeds its mm threshold.

    Compare with the last accepted point, never with the previous noisy frame.
    An accepted movement updates both coordinates and becomes the next anchor.
    """

    def __init__(self, threshold_mm: float = 10.0):
        try:
            threshold_mm = float(threshold_mm)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("motion threshold must be finite and positive") from exc
        if not math.isfinite(threshold_mm) or threshold_mm <= 0:
            raise ValueError("motion threshold must be finite and positive")
        self.threshold_mm = threshold_mm
        self._accepted = {}
        self._context = None

    def reset(self):
        self._accepted.clear()
        self._context = None

    def update(self, points_mm, *, context=None):
        if context != self._context:
            self.reset()
            self._context = context
        current, result = {}, {}
        if not isinstance(points_mm, Mapping):
            self._accepted.clear()
            return result
        for side in ("LEFT", "RIGHT"):
            candidate = finite_values(points_mm.get(side), 2)
            if candidate is None:
                continue
            previous = self._accepted.get(side)
            if previous is None:
                state = WristMotionState(candidate, (0.0, 0.0), initialized=True)
            else:
                delta = (candidate[0] - previous[0], candidate[1] - previous[1])
                if not all(math.isfinite(value) for value in delta):
                    continue
                # Ignore only roundoff at the exact boundary, e.g. the mm
                # conversion of a mathematically exact 10 mm pixel offset.
                moved = any(abs(value) > self.threshold_mm and not math.isclose(
                    abs(value), self.threshold_mm, rel_tol=1e-12, abs_tol=1e-9
                ) for value in delta)
                state = WristMotionState(candidate if moved else previous, delta, moved)
            current[side] = state.point_mm
            result[side] = state
        self._accepted = current
        return result


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




def compute_workspace_roi(image_size, work_mm, fill):
    size = finite_values(image_size, 2)
    work = finite_values(work_mm, 2)
    if size is None or work is None or min(size) <= 0 or min(work) <= 0:
        raise ValueError("image and workspace dimensions must be finite and positive")
    try:
        fill = float(fill)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("workspace fill must be in (0, 1]") from exc
    if not np.isfinite(fill) or not 0 < fill <= 1:
        raise ValueError("workspace fill must be in (0, 1]")
    width, height = size
    work_mm = work
    scale = min(width * fill / work_mm[0], height * fill / work_mm[1])
    roi_w, roi_h = work_mm[0] * scale, work_mm[1] * scale
    if not np.isfinite((roi_w, roi_h)).all() or min(roi_w, roi_h) <= 0:
        raise ValueError("workspace ROI must have finite positive dimensions")
    left, top = (width - roi_w) / 2.0, (height - roi_h) / 2.0
    return left, top, left + roi_w, top + roi_h


def process_wrist_frame(tracker, detections, image_size, work_mm, fill,
                        armed=False, swap_arms=False, frame_id=0,
                        timestamp=None, data_source="CAMERA_UNVERIFIED",
                        motion_gate=None):
    """Production coordinate path, usable offline without a camera or socket.

    A target requires BOTH the raw and EMA wrist to be inside the ROI. Rows
    describe generated targets only; send events are recorded by YumiArmLink.
    With a motion gate, return/display/publish accepted wrists while retaining
    the ungated EMA candidate in the diagnostic columns. The application always
    supplies its gate; omitting it preserves the coordinate-only helper API.
    """
    old_size = tracker._image_size
    wrists = tracker.update(detections, image_size=image_size)
    resized = old_size is not None and old_size != tracker._image_size
    roi_valid = True
    try:
        roi = compute_workspace_roi(image_size, work_mm, fill)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError):
        roi_valid = False
        roi = (0.0, 0.0, 0.0, 0.0)
    stamp = timestamp if timestamp is not None else utc_timestamp()
    rows = []
    for side, _ in WRIST_SPECS:
        raw = finite_values(detections.get(side), 3) if hasattr(detections, "get") else None
        filtered = wrists.get(side)
        raw_inside = wrist_to_workspace_mm(raw, roi, work_mm)[2] if raw else False
        x_mm, y_mm, filtered_inside = (
            wrist_to_workspace_mm(filtered, roi, work_mm) if filtered else (0.0, 0.0, False)
        )
        valid = bool(armed and raw is not None and filtered is not None
                     and raw_inside and filtered_inside)
        arm = "ROB_L" if side == "LEFT" else "ROB_R"
        if swap_arms:
            arm = "ROB_R" if arm == "ROB_L" else "ROB_L"
        rows.append(dict(
            timestamp=stamp, frame_id=frame_id, arm=arm, wrist=side,
            raw_x_pixel=raw[0] if raw else "NOT_AVAILABLE",
            raw_y_pixel=raw[1] if raw else "NOT_AVAILABLE",
            filtered_x_pixel=filtered[0] if filtered else "NOT_AVAILABLE",
            filtered_y_pixel=filtered[1] if filtered else "NOT_AVAILABLE",
            x_workspace_mm=x_mm if filtered and roi_valid else "NOT_AVAILABLE",
            y_workspace_mm=y_mm if filtered and roi_valid else "NOT_AVAILABLE",
            raw_inside_roi=raw_inside, filtered_inside_roi=filtered_inside,
            armed=bool(armed), target_valid=valid,
            x_target_mm=x_mm if valid else "NOT_AVAILABLE",
            y_target_mm=y_mm if valid else "NOT_AVAILABLE",
            tcp_connected=False, x_sent_mm="NOT_AVAILABLE", y_sent_mm="NOT_AVAILABLE",
            send_timestamp="NOT_AVAILABLE", sent_valid="NOT_AVAILABLE",
            event_type="FRAME", robot_feedback="NOT_AVAILABLE",
            data_source=data_source, frame_resized=resized,
        ))
    if motion_gate is not None:
        candidates = {
            row["wrist"]: (row["x_workspace_mm"], row["y_workspace_mm"])
            for row in rows if row["raw_inside_roi"] and row["filtered_inside_roi"]
        }
        motion_states = motion_gate.update(candidates, context=(
            finite_values(image_size, 2), roi, finite_values(work_mm, 2),
        ))
        accepted_wrists = {}
        for row in rows:
            side = row["wrist"]
            state = motion_states.get(side)
            if state is not None:
                x_accepted, y_accepted = state.point_mm
                pixel_x = roi[0] + (x_accepted / work_mm[0] + 0.5) * (roi[2] - roi[0])
                pixel_y = roi[1] + (0.5 - y_accepted / work_mm[1]) * (roi[3] - roi[1])
                accepted_wrists[side] = (pixel_x, pixel_y, wrists[side][2])
                row.update(
                    accepted_x_pixel=pixel_x, accepted_y_pixel=pixel_y,
                    accepted_x_mm=x_accepted, accepted_y_mm=y_accepted,
                    delta_x_mm=state.delta_mm[0], delta_y_mm=state.delta_mm[1],
                    movement_detected=state.moved,
                    movement_initialized=state.initialized,
                )
                if row["target_valid"]:
                    row["x_target_mm"], row["y_target_mm"] = state.point_mm
            else:
                row.update(
                    accepted_x_pixel="NOT_AVAILABLE", accepted_y_pixel="NOT_AVAILABLE",
                    accepted_x_mm="NOT_AVAILABLE", accepted_y_mm="NOT_AVAILABLE",
                    delta_x_mm="NOT_AVAILABLE", delta_y_mm="NOT_AVAILABLE",
                    movement_detected=False, movement_initialized=False,
                    target_valid=False, x_target_mm="NOT_AVAILABLE", y_target_mm="NOT_AVAILABLE",
                )
            row["movement_threshold_mm"] = motion_gate.threshold_mm
        wrists = accepted_wrists
    return wrists, roi, rows
