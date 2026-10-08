import ctypes
import os
from pathlib import Path
from time import perf_counter_ns
from typing import List, Sequence

import cv2
import numpy as np

# COCO arm joints (LS, RS, LE, RE, LW, RW) -> MediaPipe pose landmark indices
COCO_TO_MEDIAPIPE = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16}


class MediaPipeArmRefiner:
    """Use MediaPipe PoseLandmarker to refine KAPAO shoulder/elbow/wrist keypoints."""

    def __init__(
        self,
        model_path: str,
        num_poses: int = 2,
        min_confidence: float = 0.5,
        min_visibility: float = 0.5,
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
                f"MediaPipe pose model was not found: {model_path}. "
                "Download pose_landmarker_full.task or pass --arm-model."
            )
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

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Return one (6, 3) array per person: x, y in pixels and visibility, in COCO arm order."""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        task_image = self.image_module.Image(self.image_module.ImageFormat.SRGB, rgb)
        timestamp_ms = perf_counter_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        result = self.landmarker.detect_for_video(task_image, timestamp_ms)

        people = []
        for landmarks in result.pose_landmarks:
            arm = np.zeros((len(COCO_TO_MEDIAPIPE), 3), dtype=np.float32)
            for row, mp_index in enumerate(COCO_TO_MEDIAPIPE.values()):
                point = landmarks[mp_index]
                arm[row] = (point.x * width, point.y * height, point.visibility)
            people.append(arm)
        return people

    def detect_people(self, image: np.ndarray):
        """Standalone mode: return person boxes and COCO-17 style poses (only arm joints 5-10 filled)."""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        task_image = self.image_module.Image(self.image_module.ImageFormat.SRGB, rgb)
        timestamp_ms = perf_counter_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        result = self.landmarker.detect_for_video(task_image, timestamp_ms)

        boxes, poses = [], []
        for landmarks in result.pose_landmarks:
            points = np.asarray([[p.x * width, p.y * height, p.visibility] for p in landmarks], dtype=np.float32)
            shown = points[points[:, 2] >= 0.3]
            if len(shown) < 3:
                continue
            x1, y1 = shown[:, :2].min(axis=0)
            x2, y2 = shown[:, :2].max(axis=0)
            pad = 0.1 * max(x2 - x1, y2 - y1)
            boxes.append((max(0.0, x1 - pad), max(0.0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)))
            pose = np.zeros((17, 3), dtype=np.float32)
            for coco_index, mp_index in COCO_TO_MEDIAPIPE.items():
                x, y, visibility = points[mp_index]
                if visibility >= self.min_visibility:
                    pose[coco_index] = (x, y, 1.0)
            poses.append(pose)
        return boxes, poses

    def refine(
        self,
        boxes: Sequence[Sequence[float]],
        poses: Sequence[np.ndarray],
        arms: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """Overwrite KAPAO arm joints with the MediaPipe pose that best matches each person box."""
        refined = [np.asarray(pose).copy() for pose in poses]
        used = set()
        for person_index, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            margin = 0.15 * max(x2 - x1, y2 - y1)
            best, best_score = None, 0.0
            for arm_index, arm in enumerate(arms):
                if arm_index in used:
                    continue
                visible = arm[arm[:, 2] >= self.min_visibility]
                if len(visible) == 0:
                    continue
                inside = np.mean(
                    (visible[:, 0] >= x1 - margin) & (visible[:, 0] <= x2 + margin)
                    & (visible[:, 1] >= y1 - margin) & (visible[:, 1] <= y2 + margin)
                )
                score = float(inside) * len(visible)
                if score > best_score:
                    best, best_score = arm_index, score
            if best is None or best_score < 2.0:
                continue
            used.add(best)
            for row, coco_index in enumerate(COCO_TO_MEDIAPIPE):
                x, y, visibility = arms[best][row]
                if visibility >= self.min_visibility:
                    refined[person_index][coco_index] = (x, y, 1.0)
        return refined

    def close(self) -> None:
        self.landmarker.close()
