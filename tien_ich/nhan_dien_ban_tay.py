from dataclasses import dataclass
import math
import os
from pathlib import Path
import tempfile
from time import perf_counter_ns
from typing import List, Optional

import cv2
import numpy as np


def open_camera(args):
    source = int(args.pipe) if args.pipe.isdecimal() else args.pipe
    camera = cv2.VideoCapture(source)
    if not camera.isOpened() and os.name == "nt":
        camera.release()
        camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Unable to open webcam source {args.pipe}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.camera_fourcc:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc))
    if args.camera_fps > 0:
        camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    print(
        f"Camera capture: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@{camera.get(cv2.CAP_PROP_FPS):.0f}fps"
    )
    return camera


def create_mediapipe_task(factory, options):
    """Create a native task without exposing Unicode cwd to Windows lookup."""
    working_directory = Path.cwd()
    if os.name == "nt":
        # The model is already held in BaseOptions.model_asset_buffer. Native
        # runfiles lookup also needs an ASCII cwd when short paths are absent.
        init_directory = tempfile.gettempdir()
        if not init_directory.isascii():
            init_directory = os.environ.get("SystemRoot", "C:/Windows")
        if not init_directory.isascii():
            raise RuntimeError("MediaPipe initialization needs an ASCII temporary directory")
        os.chdir(init_directory)
    try:
        return factory(options)
    finally:
        os.chdir(working_directory)


@dataclass(frozen=True)
class MediaPipeHand:
    confidence: float
    is_left: bool
    landmarks: np.ndarray
    world_landmarks: Optional[np.ndarray] = None
    gesture_name: Optional[str] = None
    gesture_confidence: Optional[float] = None


class MediaPipeHandDetector:
    """Detect hands, optionally using a trained gesture classifier in the same pass."""

    def __init__(
        self,
        model_path: str,
        num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        gesture_model_path: Optional[str] = None,
    ) -> None:
        try:
            from mediapipe.tasks.python import vision
            from mediapipe.tasks.python.core import base_options
            from mediapipe.tasks.python.vision.core import image as image_module
        except ImportError as exc:
            raise ImportError("MediaPipe is not installed. Install it with: pip install mediapipe") from exc

        model_path = str(Path(gesture_model_path or model_path).absolute())
        if not Path(model_path).is_file():
            raise FileNotFoundError(
                f"MediaPipe hand/gesture model was not found: {model_path}. "
                "Check --gesture-model or --hand-model."
            )

        # Keep native Windows model loading independent of Unicode paths.
        model_bytes = Path(model_path).read_bytes()

        self.image_module = image_module
        self.last_timestamp = 0
        self.gesture_recognizer = None
        self.hand_landmarker = None
        options_type = (vision.GestureRecognizerOptions if gesture_model_path
                        else vision.HandLandmarkerOptions)
        options = options_type(
            base_options=base_options.BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_presence_confidence=min_detection_confidence,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        factory = (vision.GestureRecognizer.create_from_options if gesture_model_path
                   else vision.HandLandmarker.create_from_options)
        task = create_mediapipe_task(factory, options)
        if gesture_model_path:
            self.gesture_recognizer = task
        else:
            self.hand_landmarker = task

    def detect(self, image: np.ndarray) -> List[MediaPipeHand]:
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        task_image = self.image_module.Image(self.image_module.ImageFormat.SRGB, rgb)
        timestamp_ms = perf_counter_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        recognizer = getattr(self, "gesture_recognizer", None)
        result = (recognizer.recognize_for_video(task_image, timestamp_ms)
                  if recognizer is not None
                  else self.hand_landmarker.detect_for_video(task_image, timestamp_ms))
        hands = []

        if not result.hand_landmarks:
            return hands

        for hand_index, (landmarks, handedness) in enumerate(zip(
            result.hand_landmarks,
            result.handedness,
        )):
            if len(landmarks) != 21 or not handedness:
                continue
            coordinates = np.asarray(
                [[point.x, point.y, point.z] for point in landmarks],
                dtype=np.float32,
            ).copy()
            if not np.isfinite(coordinates).all():
                continue
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
            gesture_name, gesture_confidence = None, None
            if recognizer is not None:
                # Preserve the winning class across ALL gestures. Restricting
                # the model to palm/fist could promote a lower-scored wrong class.
                gesture_name, gesture_confidence = "None", 0.0
                categories = result.gestures[hand_index] if hand_index < len(result.gestures) else ()
                valid_categories = []
                for category in categories:
                    score = float(category.score)
                    if math.isfinite(score) and 0 <= score <= 1:
                        valid_categories.append((score, category.category_name))
                if valid_categories:
                    gesture_confidence, gesture_name = max(valid_categories, key=lambda item: item[0])
                    gesture_name = str(gesture_name or "None")
            hands.append(
                MediaPipeHand(
                    confidence=float(handedness[0].score),
                    is_left=bool(handedness[0].category_name == "Left"),
                    landmarks=coordinates,
                    world_landmarks=world_points,
                    gesture_name=gesture_name,
                    gesture_confidence=gesture_confidence,
                )
            )
        return hands

    def close(self) -> None:
        recognizer = getattr(self, "gesture_recognizer", None)
        if recognizer is not None:
            recognizer.close()
        elif self.hand_landmarker is not None:
            self.hand_landmarker.close()
