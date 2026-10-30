import csv
import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristState


Vector3 = Tuple[float, float, float]
Matrix3 = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]


@dataclass(frozen=True)
class RobotMappingConfig:
    """Explicit, confirmed mapping from shoulder-relative cm to robot-frame mm."""

    robot_frame: str
    tracked_side: str
    home_robot_mm: Vector3
    mm_per_cm_matrix: Matrix3
    max_offset_mm: Vector3
    max_speed_mm_s: float
    max_sample_interval_s: float
    max_delta_step_cm: float
    min_confidence: float
    calibration_confirmed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.robot_frame, str) or not self.robot_frame.strip():
            raise ValueError("robot_frame must be a non-empty frame name")
        if self.tracked_side not in ("LEFT", "RIGHT"):
            raise ValueError("tracked_side must be LEFT or RIGHT")
        if self.calibration_confirmed is not True:
            raise ValueError("robot mapping must be explicitly calibration_confirmed")
        for name, value in (
            ("home_robot_mm", self.home_robot_mm),
            ("max_offset_mm", self.max_offset_mm),
        ):
            array = np.asarray(value, dtype=np.float64)
            if array.shape != (3,) or not np.isfinite(array).all():
                raise ValueError(f"{name} must contain three finite values")
        matrix = np.asarray(self.mm_per_cm_matrix, dtype=np.float64)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("mm_per_cm_matrix must be a finite 3x3 matrix")
        if any(value <= 0 for value in self.max_offset_mm):
            raise ValueError("max_offset_mm values must be positive")
        if not math.isfinite(self.max_speed_mm_s) or self.max_speed_mm_s <= 0:
            raise ValueError("max_speed_mm_s must be positive and finite")
        if not math.isfinite(self.max_sample_interval_s) or self.max_sample_interval_s <= 0:
            raise ValueError("max_sample_interval_s must be positive and finite")
        if not math.isfinite(self.max_delta_step_cm) or self.max_delta_step_cm <= 0:
            raise ValueError("max_delta_step_cm must be positive and finite")
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

    @classmethod
    def from_json(cls, path: str) -> "RobotMappingConfig":
        with Path(path).open("r", encoding="utf-8") as stream:
            values = json.load(stream)
        return cls(
            robot_frame=values["robot_frame"],
            tracked_side=values["tracked_side"],
            home_robot_mm=tuple(values["home_robot_mm"]),
            mm_per_cm_matrix=tuple(tuple(row) for row in values["mm_per_cm_matrix"]),
            max_offset_mm=tuple(values["max_offset_mm"]),
            max_speed_mm_s=float(values["max_speed_mm_s"]),
            max_sample_interval_s=float(values["max_sample_interval_s"]),
            max_delta_step_cm=float(values["max_delta_step_cm"]),
            min_confidence=float(values["min_confidence"]),
            calibration_confirmed=values["calibration_confirmed"],
        )


@dataclass(frozen=True)
class RobotTargetPreview:
    """Dry-run-only target preview; never represents a robot command or robot state."""

    valid: bool
    status: str
    robot_frame: Optional[str]
    tracked_side: Optional[str]
    offset_mm: Optional[Vector3] = None
    target_mm: Optional[Vector3] = None
    timestamp: Optional[float] = None


class RobotTargetMapper:
    """Map calibrated shoulder-relative wrist displacement to a dry-run robot target."""

    def __init__(self, config: RobotMappingConfig) -> None:
        self.config = config
        self._last_timestamp: Optional[float] = None
        self._last_delta_cm: Optional[np.ndarray] = None
        self._last_target: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._last_timestamp = None
        self._last_delta_cm = None
        self._last_target = None

    def _invalid(self, status: str, timestamp: Optional[float] = None) -> RobotTargetPreview:
        self.reset()
        return RobotTargetPreview(
            valid=False,
            status=status,
            robot_frame=self.config.robot_frame,
            tracked_side=self.config.tracked_side,
            timestamp=timestamp,
        )

    def hold(self, status: str, timestamp: float) -> RobotTargetPreview:
        """Report that no fresh sample is available without extending its validity."""
        return RobotTargetPreview(
            valid=False,
            status=status,
            robot_frame=self.config.robot_frame,
            tracked_side=self.config.tracked_side,
            timestamp=timestamp,
        )

    def update(
        self,
        state: RelativeWristState,
        timestamp: float,
    ) -> RobotTargetPreview:
        if not math.isfinite(timestamp):
            return self._invalid("INVALID TIMESTAMP")
        if state.side != self.config.tracked_side:
            return self._invalid("WRONG TRACKED SIDE", timestamp)
        if state.status != "MODEL RELATIVE" or not state.calibrated or state.delta_cm is None:
            return self._invalid(state.status, timestamp)
        if state.confidence is None or not math.isfinite(state.confidence):
            return self._invalid("INVALID CONFIDENCE", timestamp)
        if state.confidence < self.config.min_confidence:
            return self._invalid("LOW CONFIDENCE", timestamp)

        delta_cm = np.asarray(state.delta_cm, dtype=np.float64)
        if delta_cm.shape != (3,) or not np.isfinite(delta_cm).all():
            return self._invalid("INVALID 3D DELTA", timestamp)
        offset_mm = np.asarray(self.config.mm_per_cm_matrix) @ delta_cm
        if not np.isfinite(offset_mm).all():
            return self._invalid("INVALID MAPPED OFFSET", timestamp)
        if np.any(np.abs(offset_mm) > np.asarray(self.config.max_offset_mm)):
            return self._invalid("OFFSET OUT OF RANGE", timestamp)

        if self._last_timestamp is None:
            self._last_timestamp = timestamp
            self._last_delta_cm = delta_cm
            self._last_target = np.asarray(self.config.home_robot_mm, dtype=np.float64)
            return RobotTargetPreview(
                False, "INITIALIZING", self.config.robot_frame,
                self.config.tracked_side, timestamp=timestamp,
            )

        dt = timestamp - self._last_timestamp
        if not math.isfinite(dt) or dt <= 0:
            return self._invalid("INVALID DT", timestamp)
        if dt > self.config.max_sample_interval_s:
            return self._invalid("STALE FRAME", timestamp)
        if (
            self._last_delta_cm is not None
            and float(np.linalg.norm(delta_cm - self._last_delta_cm)) > self.config.max_delta_step_cm
        ):
            return self._invalid("DELTA JUMP", timestamp)

        desired_target = np.asarray(self.config.home_robot_mm, dtype=np.float64) + offset_mm
        previous_target = self._last_target
        if previous_target is None:
            return self._invalid("REINITIALIZING", timestamp)
        difference = desired_target - previous_target
        max_step = self.config.max_speed_mm_s * dt
        step_length = float(np.linalg.norm(difference))
        limited = step_length > max_step
        if limited:
            difference *= max_step / step_length
        target = previous_target + difference

        self._last_timestamp = timestamp
        self._last_delta_cm = delta_cm
        self._last_target = target
        actual_offset = target - np.asarray(self.config.home_robot_mm, dtype=np.float64)
        return RobotTargetPreview(
            valid=True,
            status="RATE LIMITED" if limited else "DRY RUN",
            robot_frame=self.config.robot_frame,
            tracked_side=self.config.tracked_side,
            offset_mm=tuple(float(value) for value in actual_offset),
            target_mm=tuple(float(value) for value in target),
            timestamp=timestamp,
        )


class RobotMappingCsvLogger:
    """Write dry-run mapping telemetry and raw shoulder/wrist landmarks to CSV."""

    _FIELDS = (
        "timestamp_iso", "timestamp_monotonic_s", "side", "status", "confidence",
        "shoulder_x_m", "shoulder_y_m", "shoulder_z_m",
        "wrist_x_m", "wrist_y_m", "wrist_z_m",
        "delta_x_cm", "delta_y_cm", "delta_z_cm",
        "offset_x_mm", "offset_y_mm", "offset_z_mm",
        "target_x_mm", "target_y_mm", "target_z_mm", "robot_frame", "valid",
    )

    def __init__(self, path: str) -> None:
        self._stream = Path(path).open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._stream, fieldnames=self._FIELDS)
        self._writer.writeheader()

    @staticmethod
    def _xyz(pose: Optional[np.ndarray], index: int) -> Tuple[Optional[float], ...]:
        if pose is None or pose.ndim != 2 or pose.shape[0] <= index or pose.shape[1] < 3:
            return (None, None, None)
        point = pose[index, :3]
        if not np.isfinite(point).all():
            return (None, None, None)
        return tuple(float(value) for value in point)

    @staticmethod
    def _xyz_values(values: Optional[Sequence[float]]) -> Tuple[Optional[float], ...]:
        if values is None:
            return (None, None, None)
        return tuple(float(value) for value in values)

    def write(
        self,
        monotonic_timestamp: float,
        state: RelativeWristState,
        preview: RobotTargetPreview,
        pose_world: Optional[np.ndarray],
    ) -> None:
        shoulder_index, wrist_index = (11, 15) if state.side == "LEFT" else (12, 16)
        shoulder = self._xyz(pose_world, shoulder_index)
        wrist = self._xyz(pose_world, wrist_index)
        row = {
            "timestamp_iso": datetime.now().astimezone().isoformat(),
            "timestamp_monotonic_s": monotonic_timestamp,
            "side": state.side,
            "status": preview.status,
            "confidence": state.confidence,
            "shoulder_x_m": shoulder[0], "shoulder_y_m": shoulder[1], "shoulder_z_m": shoulder[2],
            "wrist_x_m": wrist[0], "wrist_y_m": wrist[1], "wrist_z_m": wrist[2],
            "delta_x_cm": self._xyz_values(state.delta_cm)[0],
            "delta_y_cm": self._xyz_values(state.delta_cm)[1],
            "delta_z_cm": self._xyz_values(state.delta_cm)[2],
            "offset_x_mm": self._xyz_values(preview.offset_mm)[0],
            "offset_y_mm": self._xyz_values(preview.offset_mm)[1],
            "offset_z_mm": self._xyz_values(preview.offset_mm)[2],
            "target_x_mm": self._xyz_values(preview.target_mm)[0],
            "target_y_mm": self._xyz_values(preview.target_mm)[1],
            "target_z_mm": self._xyz_values(preview.target_mm)[2],
            "robot_frame": preview.robot_frame,
            "valid": preview.valid,
        }
        self._writer.writerow(row)
        self._stream.flush()

    def close(self) -> None:
        self._stream.close()
