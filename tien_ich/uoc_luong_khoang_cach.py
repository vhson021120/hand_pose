"""Estimate camera-to-chest and camera-to-wrist range from a single RGB camera.

Chest range comes from the pinhole relation Z = fx * W * cos(yaw) / shoulder_px,
where W is the real shoulder width and yaw is the torso rotation measured from
MediaPipe world landmarks. Wrist range is chest range plus the MediaPipe world
depth difference between the wrist and the shoulder midpoint. All values are
estimates for a dry-run preview, not a safety measurement.
"""

import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

DEFAULT_SHOULDER_WIDTH_M = 0.38
DEFAULT_HFOV_DEG = 66.0
MAX_MAE_M = 0.03
MAX_ABS_ERROR_M = 0.05
MIN_RANGE_M = 0.3
MAX_RANGE_M = 4.0

COCO_SHOULDERS = (5, 6)
COCO_WRISTS = {"LEFT": 9, "RIGHT": 10}
WORLD_SHOULDERS = (11, 12)
WORLD_WRISTS = {"LEFT": 15, "RIGHT": 16}


@dataclass
class RangeCalibration:
    """Camera intrinsics plus the fitted shoulder-width range scale."""

    image_width: int = 1280
    image_height: int = 720
    fx_px: float = 0.0
    fy_px: float = 0.0
    cx_px: float = 0.0
    cy_px: float = 0.0
    reprojection_rmse_px: float = 0.0
    shoulder_width_m: float = DEFAULT_SHOULDER_WIDTH_M
    scale: float = 1.0
    confirmed: bool = False
    calibration_metrics: Dict[str, float] = field(default_factory=dict)
    validation_metrics: Dict[str, float] = field(default_factory=dict)
    yaw_metrics: Dict[str, float] = field(default_factory=dict)
    wrist_metrics: Dict[str, float] = field(default_factory=dict)

    @property
    def intrinsics_calibrated(self) -> bool:
        return self.fx_px > 0 and self.reprojection_rmse_px > 0

    def focal_px(self, width: int) -> float:
        if self.fx_px > 0:
            return self.fx_px * width / self.image_width
        return (width / 2.0) / math.tan(math.radians(DEFAULT_HFOV_DEG / 2.0))

    def focal_y_px(self, width: int) -> float:
        if self.fy_px > 0:
            return self.fy_px * width / self.image_width
        return self.focal_px(width)

    def principal_point(self, width: int, height: int) -> Tuple[float, float]:
        if self.cx_px > 0 and self.cy_px > 0:
            factor = width / self.image_width
            return self.cx_px * factor, self.cy_px * factor
        return width / 2.0, height / 2.0

    def range_constant(self, width: int) -> float:
        return self.scale * self.focal_px(width) * self.shoulder_width_m

    @classmethod
    def from_json(cls, path) -> "RangeCalibration":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        calibration = cls(
            image_width=int(data["image_width"]),
            image_height=int(data["image_height"]),
            fx_px=float(data.get("fx_px", 0.0)),
            fy_px=float(data.get("fy_px", 0.0)),
            cx_px=float(data.get("cx_px", 0.0)),
            cy_px=float(data.get("cy_px", 0.0)),
            reprojection_rmse_px=float(data.get("reprojection_rmse_px", 0.0)),
            shoulder_width_m=float(data["shoulder_width_m"]),
            scale=float(data["scale"]),
            confirmed=data.get("confirmed") is True,
            calibration_metrics=dict(data.get("calibration_metrics", {})),
            validation_metrics=dict(data.get("validation_metrics", {})),
            yaw_metrics=dict(data.get("yaw_metrics", {})),
            wrist_metrics=dict(data.get("wrist_metrics", {})),
        )
        calibration.check()
        return calibration

    def check(self) -> None:
        values = (self.image_width, self.image_height, self.shoulder_width_m, self.scale)
        if not all(np.isfinite(values)) or min(values) <= 0:
            raise ValueError("range calibration values must be positive and finite")
        optional = (self.fx_px, self.fy_px, self.cx_px, self.cy_px, self.reprojection_rmse_px)
        if not all(np.isfinite(optional)) or min(optional) < 0:
            raise ValueError("camera intrinsics must be finite and non-negative")

    def save(self, path) -> None:
        self.check()
        Path(path).write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


@dataclass
class WristRange:
    depth_m: float
    distance_m: float
    camera_xyz_m: Tuple[float, float, float]
    delta_z_m: float


@dataclass
class RangeMeasurement:
    chest_distance_m: float
    yaw_deg: float
    shoulder_px: float
    range_feature: float
    chest_xyz_m: Tuple[float, float, float]
    wrists: Dict[str, WristRange]


class ShoulderRangeEstimator:
    """Measure chest and wrist range for one person per call."""

    def __init__(
        self,
        calibration: Optional[RangeCalibration] = None,
        max_yaw_deg: float = 25.0,
        min_confidence: float = 0.6,
        min_shoulder_fraction: float = 0.03,
    ) -> None:
        if not 0.0 < max_yaw_deg < 90.0:
            raise ValueError("max_yaw_deg must be between 0 and 90")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.calibration = calibration or RangeCalibration()
        self.max_yaw_deg = max_yaw_deg
        self.min_confidence = min_confidence
        self.min_shoulder_fraction = min_shoulder_fraction
        self.last_reason = "NO MEASUREMENT"

    def range_feature(
        self, pose: np.ndarray, world: np.ndarray, image_shape: Sequence[int]
    ) -> Optional[Tuple[float, float, float]]:
        """Return (cos(yaw)/shoulder_px, yaw_deg, shoulder_px) or None with a reason."""
        height, width = image_shape[:2]
        pose = np.asarray(pose)
        world = np.asarray(world)
        if pose.ndim != 2 or pose.shape[0] < 11 or pose.shape[1] < 3:
            self.last_reason = "NO POSE"
            return None
        if world.ndim != 2 or world.shape[0] < 17 or world.shape[1] < 4:
            self.last_reason = "NO WORLD LANDMARKS"
            return None

        left, right = (pose[index] for index in COCO_SHOULDERS)
        world_left, world_right = (world[index] for index in WORLD_SHOULDERS)
        if (
            min(left[2], right[2], world_left[3], world_right[3]) < self.min_confidence
            or not np.isfinite(left[:2]).all()
            or not np.isfinite(right[:2]).all()
            or not np.isfinite(world_left[:3]).all()
            or not np.isfinite(world_right[:3]).all()
        ):
            self.last_reason = "LOW SHOULDER CONFIDENCE"
            return None
        for point in (left, right):
            if not (0 <= point[0] < width and 0 <= point[1] < height):
                self.last_reason = "SHOULDER OUT OF FRAME"
                return None

        shoulder_px = float(np.hypot(*(left[:2] - right[:2])))
        if shoulder_px < self.min_shoulder_fraction * width:
            self.last_reason = "SHOULDERS TOO SMALL"
            return None

        delta = world_right[:3] - world_left[:3]
        yaw_deg = math.degrees(math.atan2(abs(float(delta[2])), abs(float(delta[0])) + 1e-9))
        if yaw_deg > self.max_yaw_deg:
            self.last_reason = f"TORSO ROTATED {yaw_deg:.0f} DEG"
            return None
        return math.cos(math.radians(yaw_deg)) / shoulder_px, yaw_deg, shoulder_px

    def measure(
        self, pose: np.ndarray, world: np.ndarray, image_shape: Sequence[int]
    ) -> Optional[RangeMeasurement]:
        feature = self.range_feature(pose, world, image_shape)
        if feature is None:
            return None
        range_feature, yaw_deg, shoulder_px = feature
        height, width = image_shape[:2]
        pose = np.asarray(pose)
        world = np.asarray(world)

        chest_z = self.calibration.range_constant(width) * range_feature
        if not MIN_RANGE_M <= chest_z <= MAX_RANGE_M:
            self.last_reason = f"RANGE OUT OF BOUNDS {chest_z:.2f}M"
            return None

        focal = self.calibration.focal_px(width)
        focal_y = self.calibration.focal_y_px(width)
        center_x, center_y = self.calibration.principal_point(width, height)
        midpoint = (pose[COCO_SHOULDERS[0], :2] + pose[COCO_SHOULDERS[1], :2]) / 2.0
        chest_xyz = (
            float((midpoint[0] - center_x) * chest_z / focal),
            float((midpoint[1] - center_y) * chest_z / focal_y),
            float(chest_z),
        )

        midpoint_world_z = float(
            (world[WORLD_SHOULDERS[0], 2] + world[WORLD_SHOULDERS[1], 2]) / 2.0
        )
        wrists = {}
        for side, coco_index in COCO_WRISTS.items():
            wrist = pose[coco_index]
            world_wrist = world[WORLD_WRISTS[side]]
            if (
                min(wrist[2], world_wrist[3]) < self.min_confidence
                or not np.isfinite(wrist[:2]).all()
                or not np.isfinite(world_wrist[:3]).all()
            ):
                continue
            delta_z = float(world_wrist[2]) - midpoint_world_z
            depth = chest_z + delta_z
            if not MIN_RANGE_M <= depth <= MAX_RANGE_M:
                continue
            x = float((wrist[0] - center_x) * depth / focal)
            y = float((wrist[1] - center_y) * depth / focal_y)
            wrists[side] = WristRange(
                depth_m=float(depth),
                distance_m=float(math.sqrt(x * x + y * y + depth * depth)),
                camera_xyz_m=(x, y, float(depth)),
                delta_z_m=delta_z,
            )
        self.last_reason = "OK"
        return RangeMeasurement(
            chest_distance_m=float(chest_z),
            yaw_deg=yaw_deg,
            shoulder_px=shoulder_px,
            range_feature=range_feature,
            chest_xyz_m=chest_xyz,
            wrists=wrists,
        )


class MetricDistanceSmoother:
    """Per-track median filter that briefly holds the last value over dropouts."""

    def __init__(self, window_size: int = 7, max_missing: int = 3) -> None:
        if window_size < 1 or max_missing < 0:
            raise ValueError("window_size must be positive and max_missing non-negative")
        self.window_size = window_size
        self.max_missing = max_missing
        self._values: Dict[int, Deque[float]] = {}
        self._missing: Dict[int, int] = {}

    def update(self, track_id: int, distance_m: Optional[float]) -> Optional[float]:
        values = self._values.setdefault(track_id, deque(maxlen=self.window_size))
        if distance_m is None or not np.isfinite(distance_m) or distance_m <= 0:
            missing = self._missing.get(track_id, 0) + 1
            self._missing[track_id] = missing
            if missing > self.max_missing:
                values.clear()
                return None
        else:
            self._missing[track_id] = 0
            values.append(float(distance_m))
        return float(np.median(values)) if values else None

    def retain_tracks(self, track_ids: Iterable[int]) -> None:
        keep = set(track_ids)
        for stale in set(self._values) - keep:
            self._values.pop(stale, None)
            self._missing.pop(stale, None)

    def reset(self) -> None:
        self._values.clear()
        self._missing.clear()


class WorkingDistanceGate:
    """Report when one tracked person has stable range readings inside the work zone."""

    def __init__(
        self,
        target_distance_m: float = 1.0,
        tolerance_m: float = 0.2,
        required_consecutive: int = 2,
        latch: bool = True,
    ) -> None:
        if not np.isfinite(target_distance_m) or target_distance_m <= 0:
            raise ValueError("target_distance_m must be positive and finite")
        if not np.isfinite(tolerance_m) or tolerance_m <= 0:
            raise ValueError("tolerance_m must be positive and finite")
        if required_consecutive < 1:
            raise ValueError("required_consecutive must be positive")
        self.target_distance_m = target_distance_m
        self.tolerance_m = tolerance_m
        self.required_consecutive = required_consecutive
        self.latch = latch
        self._track_id: Optional[int] = None
        self._consecutive = 0
        self._activated = False
        self.status = "WAITING FOR ONE PERSON AT WORKING DISTANCE"

    def update(
        self,
        person_count: int,
        track_id: Optional[int],
        distance_m: Optional[float],
    ) -> bool:
        """Return True once (latched) or while stable in range (non-latched)."""
        if self._activated:
            self.status = "IN WORKING ZONE"
            return False
        if person_count != 1 or track_id is None:
            self.reset_candidate()
            self.status = "WAITING FOR EXACTLY ONE PERSON"
            return False
        if distance_m is None or not np.isfinite(distance_m) or distance_m <= 0:
            self.reset_candidate()
            self.status = "WAITING FOR VALID DISTANCE"
            return False

        difference = distance_m - self.target_distance_m
        if abs(difference) > self.tolerance_m:
            self.reset_candidate()
            self.status = "TOO FAR" if difference > 0 else "TOO NEAR"
            return False

        if self._track_id != track_id:
            self._track_id = track_id
            self._consecutive = 0
        self._consecutive = min(self._consecutive + 1, self.required_consecutive)
        if self._consecutive < self.required_consecutive:
            self.status = (
                f"IN RANGE - CONFIRMING "
                f"({self._consecutive}/{self.required_consecutive})"
            )
            return False

        if self.latch:
            self._activated = True
            self.status = "IN WORKING ZONE"
        else:
            self.status = "IN WORKING ZONE"
        return True

    def reset_candidate(self) -> None:
        self._track_id = None
        self._consecutive = 0


def range_error_metrics(true_m: Sequence[float], estimated_m: Sequence[float]) -> Dict[str, float]:
    true = np.asarray(true_m, dtype=np.float64)
    estimated = np.asarray(estimated_m, dtype=np.float64)
    if true.shape != estimated.shape or true.size == 0:
        raise ValueError("true and estimated distances must have the same non-zero length")
    errors = estimated - true
    return {
        "count": int(true.size),
        "mae_m": float(np.mean(np.abs(errors))),
        "rmse_m": float(np.sqrt(np.mean(errors ** 2))),
        "max_abs_error_m": float(np.max(np.abs(errors))),
        "mean_error_m": float(np.mean(errors)),
    }


def fit_range_scale(
    true_m: Sequence[float],
    range_features: Sequence[float],
    calibration: RangeCalibration,
    width: int,
) -> float:
    """Least-squares scale for Z = scale * fx * W * feature."""
    true = np.asarray(true_m, dtype=np.float64)
    feature = np.asarray(range_features, dtype=np.float64)
    if true.shape != feature.shape or true.size < 2 or len(set(np.round(true, 3))) < 2:
        raise ValueError("calibration needs at least two distinct real distances")
    if (true <= 0).any() or (feature <= 0).any():
        raise ValueError("calibration distances and features must be positive")
    base = calibration.focal_px(width) * calibration.shoulder_width_m * feature
    return float(np.sum(true * base) / np.sum(base * base))


def passes_validation(metrics: Dict[str, float]) -> bool:
    return (
        metrics.get("count", 0) >= 3
        and metrics["mae_m"] <= MAX_MAE_M
        and metrics["max_abs_error_m"] <= MAX_ABS_ERROR_M
    )


WRIST_MAX_MAE_M = 0.05
WRIST_MAX_ABS_ERROR_M = 0.08


def wrist_depth_metrics(
    true_forward_m: Sequence[float], delta_z_m: Sequence[float]
) -> Dict[str, float]:
    """Compare real wrist-forward offsets with the MediaPipe world dz (negative = nearer).

    The estimated forward offset is -dz. `gain` is the least-squares factor that would map
    the estimate onto the truth; a value far from 1 means a systematic depth scale bias.
    """
    true = np.asarray(true_forward_m, dtype=np.float64)
    estimated = -np.asarray(delta_z_m, dtype=np.float64)
    metrics = range_error_metrics(true, estimated)
    denominator = float(np.sum(estimated * estimated))
    metrics["gain"] = float(np.sum(true * estimated) / denominator) if denominator > 1e-9 else 0.0
    metrics["sign_correct"] = bool(
        np.all(np.sign(estimated[true > 0.05]) >= 0) if np.any(true > 0.05) else True
    )
    return metrics


def wrist_depth_acceptable(metrics: Dict[str, float]) -> bool:
    return (
        metrics.get("count", 0) >= 3
        and metrics.get("sign_correct") is True
        and metrics["mae_m"] <= WRIST_MAX_MAE_M
        and metrics["max_abs_error_m"] <= WRIST_MAX_ABS_ERROR_M
    )


def yaw_compensation_report(
    true_m: float, range_features: Sequence[float], yaw_deg: Sequence[float], constant: float
) -> Dict[str, float]:
    """Error of the chest range with and without the cos(yaw) correction at one real distance.

    `range_features` are cos(yaw)/shoulder_px values; the uncompensated range divides that
    cosine back out.
    """
    features = np.asarray(range_features, dtype=np.float64)
    yaw = np.asarray(yaw_deg, dtype=np.float64)
    if features.shape != yaw.shape or features.size == 0:
        raise ValueError("features and yaw must have the same non-zero length")
    compensated = constant * features
    uncompensated = compensated / np.cos(np.radians(yaw))
    return {
        "true_m": float(true_m),
        "mean_yaw_deg": float(np.mean(yaw)),
        "max_yaw_deg": float(np.max(yaw)),
        "compensated_error_m": float(np.mean(compensated) - true_m),
        "uncompensated_error_m": float(np.mean(uncompensated) - true_m),
    }
