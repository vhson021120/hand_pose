import ctypes
import math
import os
from pathlib import Path
from time import perf_counter_ns
from typing import Dict, List, Tuple

import cv2
import numpy as np


class MediaPipeWristDetector:
    """Return only visible left/right wrist coordinates in image pixels."""

    def __init__(
        self,
        model_path: str,
        num_poses: int = 4,
        min_confidence: float = 0.5,
        min_visibility: float = 0.7,
    ) -> None:
        if num_poses < 1:
            raise ValueError("num_poses must be positive")
        if not math.isfinite(min_confidence) or not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between 0 and 1")
        if not math.isfinite(min_visibility) or not 0 <= min_visibility <= 1:
            raise ValueError("min_visibility must be between 0 and 1")
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision.core import image as image_module
        except ImportError as exc:
            raise ImportError("MediaPipe is required. Install it with: pip install mediapipe") from exc

        model_path = str(Path(model_path).absolute())
        if not Path(model_path).is_file():
            raise FileNotFoundError(f"MediaPipe pose model was not found: {model_path}")
        if os.name == "nt":
            short_path = ctypes.create_unicode_buffer(260)
            if ctypes.windll.kernel32.GetShortPathNameW(model_path, short_path, len(short_path)):
                model_path = short_path.value

        self.image_module = image_module
        self.min_visibility = min_visibility
        self.last_timestamp = 0
        options = vision.PoseLandmarkerOptions(
            base_options=base_options.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        working_directory = Path.cwd()
        if os.name == "nt":
            short_cwd = ctypes.create_unicode_buffer(260)
            if ctypes.windll.kernel32.GetShortPathNameW(str(working_directory), short_cwd, len(short_cwd)):
                os.chdir(short_cwd.value)
        try:
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
        finally:
            os.chdir(working_directory)

    def detect(self, image: np.ndarray) -> List[Dict[str, Tuple[float, float, float]]]:
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        task_image = self.image_module.Image(self.image_module.ImageFormat.SRGB, rgb)
        timestamp_ms = perf_counter_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        result = self.landmarker.detect_for_video(task_image, timestamp_ms)

        poses = []
        for landmarks in result.pose_landmarks:
            wrists = {}
            for side, index in (("LEFT", 15), ("RIGHT", 16)):
                point = landmarks[index]
                visibility = float(point.visibility)
                if visibility >= self.min_visibility:
                    wrists[side] = (
                        float(point.x * width),
                        float(point.y * height),
                        visibility,
                    )
            poses.append(wrists)
        return poses

    def close(self) -> None:
        self.landmarker.close()