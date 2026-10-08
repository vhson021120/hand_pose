from typing import Optional, Sequence

import numpy as np


class HighLevelCommandRecognizer:
    """Recognize stable arm poses as high-level commands."""

    def __init__(
        self,
        stable_frames: int = 8,
        min_confidence: float = 0.35,
        vertical_margin: float = 0.12,
    ) -> None:
        if stable_frames < 1:
            raise ValueError("stable_frames must be at least 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if vertical_margin <= 0:
            raise ValueError("vertical_margin must be positive")

        self.stable_frames = stable_frames
        self.min_confidence = min_confidence
        self.vertical_margin = vertical_margin
        self.active_command: Optional[str] = None
        self._candidate: Optional[str] = None
        self._candidate_frames = 0

    def _arm_state(self, pose: np.ndarray, shoulder_index: int, wrist_index: int,
                   box_height: float) -> Optional[str]:
        shoulder = pose[shoulder_index]
        wrist = pose[wrist_index]
        if (shoulder[2] < self.min_confidence or wrist[2] < self.min_confidence or
                not np.isfinite(shoulder[:2]).all() or not np.isfinite(wrist[:2]).all()):
            return None

        margin = box_height * self.vertical_margin
        vertical_delta = shoulder[1] - wrist[1]
        if vertical_delta >= margin:
            return "raised"
        if vertical_delta <= -margin:
            return "lowered"
        return None

    def _classify(self, boxes: Sequence[Sequence[float]],
                  poses: Sequence[np.ndarray]) -> Optional[str]:
        if len(boxes) != 1 or len(poses) != 1:
            return None

        box = boxes[0]
        box_height = max(1.0, float(box[3]) - float(box[1]))
        pose = np.asarray(poses[0])
        if pose.ndim != 2 or pose.shape[0] <= 10 or pose.shape[1] < 3:
            return None

        left = self._arm_state(pose, 5, 9, box_height)
        right = self._arm_state(pose, 6, 10, box_height)
        if left == "raised" and right == "raised":
            return "STOP"
        if left == "raised" and right == "lowered":
            return "LEFT"
        if left == "lowered" and right == "raised":
            return "RIGHT"
        if left == "lowered" and right == "lowered":
            return "START"
        return None

    def update(self, boxes: Sequence[Sequence[float]],
               poses: Sequence[np.ndarray]) -> Optional[str]:
        """Return a command once after its pose is stable for enough frames."""
        command = self._classify(boxes, poses)
        if command is None:
            self._candidate = None
            self._candidate_frames = 0
            return None

        if command == self._candidate:
            self._candidate_frames += 1
        else:
            self._candidate = command
            self._candidate_frames = 1

        if self._candidate_frames < self.stable_frames or command == self.active_command:
            return None

        self.active_command = command
        return command

    @property
    def candidate_command(self) -> Optional[str]:
        return self._candidate

    @property
    def candidate_frames(self) -> int:
        return self._candidate_frames
