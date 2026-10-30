from typing import Optional

import numpy as np


class CaptureGestureController:
    """Apply state-specific, time-based gesture controls to webcam capture."""

    WAITING = "WAITING"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    STOPPED = "STOPPED"

    _DURATIONS = {
        "THUMBS_UP": 1.0,
        "OPEN_PALM": 3.0,
        "FIST": 0.4,
        "V_SIGN": 2.0,
    }

    def __init__(self) -> None:
        self.state = self.WAITING
        self._candidate: Optional[str] = None
        self._candidate_since: Optional[float] = None
        self.progress = 0.0
        self.instruction = "THUMBS UP 1s TO START"

    def update(self, gesture: Optional[str], timestamp: float) -> Optional[str]:
        if self.state == self.STOPPED:
            return None

        allowed = {
            self.WAITING: {"THUMBS_UP": "start"},
            self.RUNNING: {"OPEN_PALM": "pause", "V_SIGN": "stop"},
            self.PAUSED: {"FIST": "resume", "V_SIGN": "stop"},
        }[self.state]
        action = allowed.get(gesture)
        if action is None:
            self._candidate = None
            self._candidate_since = None
            self.progress = 0.0
            self._update_instruction()
            return None

        if gesture != self._candidate or self._candidate_since is None:
            self._candidate = gesture
            self._candidate_since = timestamp
            self.progress = 0.0
            self._update_instruction()
            return None

        elapsed = max(0.0, timestamp - self._candidate_since)
        duration = self._DURATIONS[gesture]
        self.progress = min(1.0, elapsed / duration)
        if elapsed + 1e-9 < duration:
            self._update_instruction()
            return None

        self._candidate = None
        self._candidate_since = None
        self.progress = 0.0
        if action == "start" or action == "resume":
            self.state = self.RUNNING
        elif action == "pause":
            self.state = self.PAUSED
        else:
            self.state = self.STOPPED
        self._update_instruction()
        return action

    def start_automatically(self) -> bool:
        if self.state != self.WAITING:
            return False
        self._candidate = None
        self._candidate_since = None
        self.progress = 0.0
        self.state = self.RUNNING
        self._update_instruction()
        return True

    def _update_instruction(self) -> None:
        if self.state == self.WAITING:
            state_text = "WAITING - THUMBS UP 1s TO START"
        elif self.state == self.RUNNING:
            state_text = "RUNNING - OPEN PALM 3s: PAUSE | V 2s: STOP"
        elif self.state == self.PAUSED:
            state_text = "PAUSED - FIST: RESUME | V 2s: STOP"
        else:
            state_text = "STOPPED"

        if self._candidate is not None:
            duration = self._DURATIONS[self._candidate]
            state_text += f" | HOLD {self._candidate}: {self.progress * duration:.1f}/{duration:.1f}s"
        self.instruction = state_text


def classify_control_gesture(landmarks: np.ndarray) -> Optional[str]:
    """Classify a vertical thumbs-up, open palm, fist, or index-middle V sign."""
    if (
        landmarks.ndim != 2
        or landmarks.shape[0] < 21
        or landmarks.shape[1] < 2
        or not np.isfinite(landmarks[:, :2]).all()
    ):
        return None

    wrist = landmarks[0, :2]
    palm_size = float(np.linalg.norm(landmarks[9, :2] - wrist))
    if palm_size <= 1e-6:
        return None

    fingers = []
    for tip_index, pip_index in ((8, 6), (12, 10), (16, 14), (20, 18)):
        tip_distance = np.linalg.norm(landmarks[tip_index, :2] - wrist)
        pip_distance = np.linalg.norm(landmarks[pip_index, :2] - wrist)
        fingers.append(tip_distance > pip_distance * 1.15)

    index, middle, ring, pinky = fingers
    thumb_mcp = landmarks[2, :2]
    thumb_tip = landmarks[4, :2]
    thumb_up = (
        thumb_mcp[1] - thumb_tip[1] >= palm_size * 0.35
        and np.linalg.norm(thumb_tip - landmarks[5, :2]) >= palm_size * 0.55
    )
    if not any(fingers) and thumb_up:
        return "THUMBS_UP"
    if all(fingers):
        return "OPEN_PALM"
    if index and middle and not ring and not pinky:
        return "V_SIGN"
    if not any(fingers):
        return "FIST"
    return None
