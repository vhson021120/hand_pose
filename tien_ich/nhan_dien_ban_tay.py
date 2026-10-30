from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
from time import perf_counter_ns
from typing import List, Optional

import cv2
import numpy as np


@dataclass(frozen=True)
class MediaPipeHand:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    is_left: bool
    landmarks: np.ndarray
    person_index: int | None = None
    world_landmarks: Optional[np.ndarray] = None


class MediaPipeHandDetector:
    """Adapter around MediaPipe Tasks HandLandmarker."""

    def __init__(
        self,
        model_path: str,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision.core import image as image_module
        except ImportError as exc:
            raise ImportError("MediaPipe is not installed. Install it with: pip install mediapipe") from exc

        model_path = str(Path(model_path).absolute())
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe hand model was not found: {model_path}. "
                "Download the HandLandmarker model or pass --hand-model."
            )

        if Path(model_path).is_absolute() and Path(model_path).drive:
            short_path = ctypes.create_unicode_buffer(260)
            if ctypes.windll.kernel32.GetShortPathNameW(model_path, short_path, len(short_path)):
                model_path = short_path.value

        self.vision = vision
        self.image_module = image_module
        self.last_timestamp = 0
        options = vision.HandLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_presence_confidence=min_detection_confidence,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        working_directory = Path.cwd()
        if os.name == "nt":
            short_cwd = ctypes.create_unicode_buffer(260)
            if ctypes.windll.kernel32.GetShortPathNameW(str(working_directory), short_cwd, len(short_cwd)):
                os.chdir(short_cwd.value)
        try:
            self.hand_landmarker = vision.HandLandmarker.create_from_options(options)
        finally:
            os.chdir(working_directory)

    def detect(self, image: np.ndarray) -> List[MediaPipeHand]:
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        task_image = self.image_module.Image(self.image_module.ImageFormat.SRGB, rgb)
        timestamp_ms = perf_counter_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        result = self.hand_landmarker.detect_for_video(task_image, timestamp_ms)
        hands = []

        if not result.hand_landmarks:
            return hands

        for hand_index, (landmarks, handedness) in enumerate(zip(
            result.hand_landmarks,
            result.handedness,
        )):
            if not landmarks:
                continue
            coordinates = np.asarray(
                [[point.x, point.y, point.z] for point in landmarks],
                dtype=np.float32,
            ).copy()
            coordinates[:, :2] *= [width, height]
            world_points = None
            if hand_index < len(result.hand_world_landmarks):
                world_points = np.asarray(
                    [
                        [point.x, point.y, point.z]
                        for point in result.hand_world_landmarks[hand_index]
                    ],
                    dtype=np.float32,
                )
            xs = coordinates[:, 0]
            ys = coordinates[:, 1]
            hands.append(
                MediaPipeHand(
                    x=float((xs.min() + xs.max()) / 2),
                    y=float((ys.min() + ys.max()) / 2),
                    width=float(xs.max() - xs.min()),
                    height=float(ys.max() - ys.min()),
                    confidence=float(handedness[0].score),
                    is_left=bool(handedness[0].category_name == "Left"),
                    landmarks=coordinates,
                    world_landmarks=world_points,
                )
            )
        return hands

    def close(self) -> None:
        self.hand_landmarker.close()
