from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class RelativeWristState:
    side: str
    delta_cm: Optional[Tuple[float, float, float]]
    status: str
    confidence: Optional[float]
    calibration_frames: int
    calibrated: bool


class RelativeWristPreview:
    """Preview wrist displacement from the midpoint of both shoulders."""

    _LANDMARKS = {"LEFT": 15, "RIGHT": 16}

    def __init__(
        self,
        calibration_frames: int = 20,
        min_confidence: float = 0.6,
        smoothing: float = 0.35,
        max_arm_length_m: float = 1.5,
    ) -> None:
        if calibration_frames < 1:
            raise ValueError("calibration_frames must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        if max_arm_length_m <= 0:
            raise ValueError("max_arm_length_m must be positive")

        self.calibration_frames = calibration_frames
        self.min_confidence = min_confidence
        self.smoothing = smoothing
        self.max_arm_length_m = max_arm_length_m
        self._samples: Dict[str, List[np.ndarray]] = {"LEFT": [], "RIGHT": []}
        self._neutral: Dict[str, Optional[np.ndarray]] = {"LEFT": None, "RIGHT": None}
        self._filtered_delta: Dict[str, Optional[np.ndarray]] = {"LEFT": None, "RIGHT": None}

    def reset(self) -> None:
        for side in self._LANDMARKS:
            self._samples[side].clear()
            self._neutral[side] = None
            self._filtered_delta[side] = None

    def _invalid_state(self, side: str, status: str, confidence: Optional[float] = None) -> RelativeWristState:
        return RelativeWristState(
            side=side,
            delta_cm=None,
            status=status,
            confidence=confidence,
            calibration_frames=len(self._samples[side]),
            calibrated=self._neutral[side] is not None,
        )

    def _state(self, side: str, pose: np.ndarray) -> RelativeWristState:
        wrist_index = self._LANDMARKS[side]
        if pose.ndim != 2 or pose.shape[0] <= wrist_index or pose.shape[1] < 4:
            return self._invalid_state(side, "NO 3D POSE")
        left_shoulder = pose[11]
        right_shoulder = pose[12]
        wrist = pose[wrist_index]
        confidence = min(
            float(left_shoulder[3]),
            float(right_shoulder[3]),
            float(wrist[3]),
        )
        if (
            not np.isfinite(left_shoulder).all()
            or not np.isfinite(right_shoulder).all()
            or not np.isfinite(wrist).all()
            or confidence < self.min_confidence
        ):
            return self._invalid_state(side, "LOW CONFIDENCE", confidence)

        shoulder_origin = (left_shoulder[:3] + right_shoulder[:3]) * 0.5
        shoulder_relative_wrist = wrist[:3] - shoulder_origin
        arm_length = float(np.linalg.norm(shoulder_relative_wrist))
        if arm_length > self.max_arm_length_m:
            return self._invalid_state(side, "INVALID 3D GEOMETRY", confidence)

        neutral = self._neutral[side]
        samples = self._samples[side]
        if neutral is None:
            samples.append(shoulder_relative_wrist.astype(np.float64))
            if len(samples) < self.calibration_frames:
                return self._invalid_state(side, "CALIBRATING", confidence)
            neutral = np.median(np.asarray(samples), axis=0)
            self._neutral[side] = neutral
            self._samples[side].clear()
            delta = np.zeros(3, dtype=np.float64)
        else:
            delta = shoulder_relative_wrist - neutral

        previous = self._filtered_delta[side]
        filtered = (
            delta if previous is None
            else previous + self.smoothing * (delta - previous)
        )
        self._filtered_delta[side] = filtered
        delta_cm = tuple(float(component * 100.0) for component in filtered)
        return RelativeWristState(
            side=side,
            delta_cm=delta_cm,
            status="MODEL RELATIVE",
            confidence=confidence,
            calibration_frames=self.calibration_frames,
            calibrated=True,
        )

    def update(self, pose_world_landmarks: Sequence[np.ndarray]) -> Tuple[RelativeWristState, RelativeWristState]:
        if not pose_world_landmarks:
            return (
                self._invalid_state("LEFT", "NO 3D POSE"),
                self._invalid_state("RIGHT", "NO 3D POSE"),
            )
        if len(pose_world_landmarks) > 1:
            self.reset()
            status = "WAITING FOR EXACTLY ONE PERSON"
            return (
                self._invalid_state("LEFT", status),
                self._invalid_state("RIGHT", status),
            )

        pose = np.asarray(pose_world_landmarks[0])
        return self._state("LEFT", pose), self._state("RIGHT", pose)
