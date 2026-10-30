from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class ArmJogState:
    side: str
    shoulder: Optional[Tuple[int, int]]
    wrist: Optional[Tuple[int, int]]
    delta_x_px: Optional[float]
    delta_y_px: Optional[float]
    direction: str
    calibration_frames: int
    calibrated: bool


class ArmJogPreview:
    """Preview independent screen-plane wrist jog directions; never sends commands."""

    def __init__(
        self,
        calibration_frames: int = 20,
        deadband: float = 0.12,
        min_confidence: float = 0.35,
    ) -> None:
        if calibration_frames < 1:
            raise ValueError("calibration_frames must be at least 1")
        if deadband <= 0:
            raise ValueError("deadband must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")

        self.calibration_frames = calibration_frames
        self.deadband = deadband
        self.min_confidence = min_confidence
        self._samples: Dict[str, List[np.ndarray]] = {"LEFT": [], "RIGHT": []}
        self._neutral: Dict[str, Optional[np.ndarray]] = {"LEFT": None, "RIGHT": None}

    def reset(self) -> None:
        for side in self._samples:
            self._samples[side].clear()
            self._neutral[side] = None

    def _point(self, pose: np.ndarray, index: int) -> Optional[np.ndarray]:
        if pose.ndim != 2 or pose.shape[0] <= index or pose.shape[1] < 3:
            return None
        point = pose[index]
        if (not np.isfinite(point).all() or point[2] < self.min_confidence or
                not np.isfinite(point[:2]).all()):
            return None
        return point[:2].astype(float)

    def _state(
        self,
        side: str,
        pose: np.ndarray,
        shoulder_index: int,
        wrist_index: int,
        box_height: float,
    ) -> ArmJogState:
        shoulder = self._point(pose, shoulder_index)
        wrist = self._point(pose, wrist_index)
        samples = self._samples[side]
        neutral = self._neutral[side]
        if shoulder is None or wrist is None:
            return ArmJogState(
                side, None, None, None, None, "NO POSE",
                len(samples), neutral is not None,
            )

        current = (wrist - shoulder) / box_height
        if neutral is None:
            samples.append(current)
            if len(samples) >= self.calibration_frames:
                self._neutral[side] = np.mean(samples, axis=0)
                self._samples[side].clear()
                neutral = self._neutral[side]
            else:
                return ArmJogState(
                    side,
                    tuple(map(int, np.rint(shoulder))),
                    tuple(map(int, np.rint(wrist))),
                    None,
                    None,
                    "CALIBRATING",
                    len(samples),
                    False,
                )

        delta = (current - neutral) * box_height
        dx = float(delta[0])
        dy_up = float(-delta[1])
        threshold = self.deadband * box_height
        if max(abs(dx), abs(dy_up)) < threshold:
            direction = "HOLD"
        elif abs(dx) >= abs(dy_up):
            direction = "SCREEN RIGHT" if dx > 0 else "SCREEN LEFT"
        else:
            direction = "SCREEN UP" if dy_up > 0 else "SCREEN DOWN"

        return ArmJogState(
            side,
            tuple(map(int, np.rint(shoulder))),
            tuple(map(int, np.rint(wrist))),
            dx,
            dy_up,
            direction,
            self.calibration_frames if neutral is not None else len(samples),
            neutral is not None,
        )

    def update(
        self,
        boxes: Sequence[Sequence[float]],
        poses: Sequence[np.ndarray],
    ) -> Tuple[ArmJogState, ArmJogState]:
        if len(boxes) != 1 or len(poses) != 1:
            self.reset()
            message = "WAITING FOR EXACTLY ONE PERSON"
            return (
                ArmJogState("LEFT", None, None, None, None, message, 0, False),
                ArmJogState("RIGHT", None, None, None, None, message, 0, False),
            )

        box = boxes[0]
        box_height = max(1.0, float(box[3]) - float(box[1]))
        pose = np.asarray(poses[0])
        return (
            self._state("LEFT", pose, 5, 9, box_height),
            self._state("RIGHT", pose, 6, 10, box_height),
        )
