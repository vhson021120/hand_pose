import ctypes
import os
from pathlib import Path
from time import perf_counter_ns
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# COCO arm joints (LS, RS, LE, RE, LW, RW) -> MediaPipe pose landmark indices
COCO_TO_MEDIAPIPE = {5: 11, 6: 12, 7: 13, 8: 14, 9: 15, 10: 16}
COCO_TO_MEDIAPIPE_POSE = (
    0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
)


class MediaPipeArmRefiner:
    """Use MediaPipe PoseLandmarker to refine KAPAO shoulder/elbow/wrist keypoints."""

    def __init__(
        self,
        model_path: str,
        num_poses: int = 4,
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
        self.last_world_landmarks: List[np.ndarray] = []
        self.last_pose_boxes: List[Tuple[float, float, float, float]] = []
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
        self.last_world_landmarks = [
            np.asarray(
                [[point.x, point.y, point.z, point.visibility] for point in landmarks],
                dtype=np.float32,
            )
            for landmarks in result.pose_world_landmarks
        ]
        self.last_pose_boxes = [
            self._landmark_box(landmarks, width, height)
            for landmarks in result.pose_landmarks
        ]

        people = []
        for landmarks in result.pose_landmarks:
            arm = np.zeros((len(COCO_TO_MEDIAPIPE), 3), dtype=np.float32)
            for row, mp_index in enumerate(COCO_TO_MEDIAPIPE.values()):
                point = landmarks[mp_index]
                arm[row] = (point.x * width, point.y * height, point.visibility)
            people.append(arm)
        return people

    def detect_people(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float, float]], List[np.ndarray]]:
        """Return person boxes and mapped COCO-17 body poses."""
        boxes, poses, _ = self.detect_people_with_world(image)
        return boxes, poses

    def detect_people_with_world(
        self, image: np.ndarray
    ) -> Tuple[List[Tuple[float, float, float, float]], List[np.ndarray], List[np.ndarray]]:
        """Also return MediaPipe's 33 hip-centered world landmarks per person."""
        height, width = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        task_image = self.image_module.Image(self.image_module.ImageFormat.SRGB, rgb)
        timestamp_ms = perf_counter_ns() // 1_000_000
        if timestamp_ms <= self.last_timestamp:
            timestamp_ms = self.last_timestamp + 1
        self.last_timestamp = timestamp_ms
        result = self.landmarker.detect_for_video(task_image, timestamp_ms)
        self.last_world_landmarks = [
            np.asarray(
                [[point.x, point.y, point.z, point.visibility] for point in landmarks],
                dtype=np.float32,
            )
            for landmarks in result.pose_world_landmarks
        ]
        self.last_pose_boxes = [
            self._landmark_box(landmarks, width, height)
            for landmarks in result.pose_landmarks
        ]

        boxes, poses, world_poses = [], [], []
        for pose_index, landmarks in enumerate(result.pose_landmarks):
            points = np.asarray([[p.x * width, p.y * height, p.visibility] for p in landmarks], dtype=np.float32)
            shown = points[points[:, 2] >= 0.3]
            if len(shown) < 3:
                continue
            x1, y1 = shown[:, :2].min(axis=0)
            x2, y2 = shown[:, :2].max(axis=0)
            pad = 0.1 * max(x2 - x1, y2 - y1)
            boxes.append((max(0.0, x1 - pad), max(0.0, y1 - pad), min(width, x2 + pad), min(height, y2 + pad)))
            pose = np.zeros((17, 3), dtype=np.float32)
            for coco_index, mp_index in enumerate(COCO_TO_MEDIAPIPE_POSE):
                x, y, visibility = points[mp_index]
                if visibility >= self.min_visibility:
                    pose[coco_index] = (x, y, visibility)
            poses.append(pose)
            if pose_index < len(self.last_world_landmarks):
                world = self.last_world_landmarks[pose_index].copy()
            else:
                world = np.empty((0, 4), dtype=np.float32)
            world_poses.append(world)
        return boxes, poses, world_poses

    @staticmethod
    def _landmark_box(landmarks, width: int, height: int) -> Tuple[float, float, float, float]:
        points = np.asarray(
            [[point.x * width, point.y * height, point.visibility] for point in landmarks],
            dtype=np.float32,
        )
        visible = points[points[:, 2] >= 0.3]
        if len(visible) < 3:
            return (0.0, 0.0, 0.0, 0.0)
        x1, y1 = visible[:, :2].min(axis=0)
        x2, y2 = visible[:, :2].max(axis=0)
        pad = 0.1 * max(x2 - x1, y2 - y1)
        return (
            max(0.0, float(x1 - pad)),
            max(0.0, float(y1 - pad)),
            min(float(width), float(x2 + pad)),
            min(float(height), float(y2 + pad)),
        )

    def match_world_to_boxes(
        self, boxes: Sequence[Sequence[float]]
    ) -> List[np.ndarray]:
        """Return world poses in detector-box order, preventing cross-person 3D swaps."""
        matches = self._match_pose_indices(boxes)
        return [
            self.last_world_landmarks[pose_index].copy()
            if pose_index is not None and pose_index < len(self.last_world_landmarks)
            else np.empty((0, 4), dtype=np.float32)
            for pose_index in matches
        ]

    def _match_pose_indices(
        self, boxes: Sequence[Sequence[float]]
    ) -> List[int | None]:
        unmatched = set(range(min(len(self.last_pose_boxes), len(self.last_world_landmarks))))
        matched: List[int | None] = []
        for box in boxes:
            x1, y1, x2, y2 = map(float, box)
            target_area = max(1.0, (x2 - x1) * (y2 - y1))
            target_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            target_diagonal = max(1.0, float(np.hypot(x2 - x1, y2 - y1)))
            best_index = None
            best_score = 0.0
            for pose_index in unmatched:
                px1, py1, px2, py2 = self.last_pose_boxes[pose_index]
                intersection = max(0.0, min(x2, px2) - max(x1, px1)) * max(
                    0.0, min(y2, py2) - max(y1, py1)
                )
                overlap = intersection / target_area
                pose_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                distance = float(np.hypot(
                    target_center[0] - pose_center[0],
                    target_center[1] - pose_center[1],
                )) / target_diagonal
                score = overlap + max(0.0, 0.2 - distance) * 0.25
                if (overlap >= 0.05 or distance <= 0.2) and score > best_score:
                    best_index = pose_index
                    best_score = score
            if best_index is None:
                matched.append(None)
                continue
            unmatched.remove(best_index)
            matched.append(best_index)
        return matched

    def refine(
        self,
        boxes: Sequence[Sequence[float]],
        poses: Sequence[np.ndarray],
        arms: Sequence[np.ndarray],
    ) -> List[np.ndarray]:
        """Overwrite KAPAO arm joints with the MediaPipe pose that best matches each person box."""
        refined = [np.asarray(pose).copy() for pose in poses]
        for person_index, arm_index in enumerate(self._match_pose_indices(boxes)):
            if arm_index is None or arm_index >= len(arms):
                continue
            visible = arms[arm_index][arms[arm_index][:, 2] >= self.min_visibility]
            if len(visible) < 2:
                continue
            for row, coco_index in enumerate(COCO_TO_MEDIAPIPE):
                x, y, visibility = arms[arm_index][row]
                if visibility >= self.min_visibility:
                    refined[person_index][coco_index] = (x, y, 1.0)
        return refined

    def close(self) -> None:
        self.landmarker.close()
