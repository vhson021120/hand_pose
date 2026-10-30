from typing import Dict, List, Sequence

import numpy as np


CONTROL_POINTS = (5, 6, 7, 8, 9, 10)


class ControlPoseTracker:
    """Smooth selected COCO arm joints and predict through brief detector dropouts."""

    def __init__(
        self,
        max_missing_frames: int = 2,
        smoothing: float = 0.35,
        min_confidence: float = 0.35,
        deadband_fraction: float = 0.01,
        outlier_jump_fraction: float = 0.45,
        outlier_confirm_fraction: float = 0.08,
    ) -> None:
        if max_missing_frames < 0:
            raise ValueError("max_missing_frames must be non-negative")
        if not 0.0 < smoothing <= 1.0:
            raise ValueError("smoothing must be greater than 0 and at most 1")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if not 0.0 <= deadband_fraction <= 0.1:
            raise ValueError("deadband_fraction must be between 0 and 0.1")
        if not 0.0 < outlier_jump_fraction <= 1.0:
            raise ValueError("outlier_jump_fraction must be greater than 0 and at most 1")
        if not 0.0 < outlier_confirm_fraction <= outlier_jump_fraction:
            raise ValueError("outlier_confirm_fraction must be positive and no greater than outlier_jump_fraction")
        self.max_missing_frames = max_missing_frames
        self.smoothing = smoothing
        self.min_confidence = min_confidence
        self.deadband_fraction = deadband_fraction
        self.outlier_jump_fraction = outlier_jump_fraction
        self.outlier_confirm_fraction = outlier_confirm_fraction
        self.tracks: Dict[int, dict] = {}
        self.next_track_id = 0
        self.last_track_ids: List[int] = []

    @staticmethod
    def _box_iou(first: Sequence[float], second: Sequence[float]) -> float:
        x1 = max(first[0], second[0])
        y1 = max(first[1], second[1])
        x2 = min(first[2], second[2])
        y2 = min(first[3], second[3])
        intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        return intersection / max(1.0, first_area + second_area - intersection)

    @staticmethod
    def _center_distance(first: Sequence[float], second: Sequence[float]) -> float:
        first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
        second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
        diagonal = max(1.0, float(np.hypot(first[2] - first[0], first[3] - first[1])))
        return float(np.hypot(first_center[0] - second_center[0], first_center[1] - second_center[1])) / diagonal

    def update(self, boxes: Sequence[Sequence[float]], poses: Sequence[np.ndarray]) -> List[np.ndarray]:
        current_boxes = [tuple(map(float, box)) for box in boxes]
        candidates = []
        for detection_index, box in enumerate(current_boxes):
            for track_id, track in self.tracks.items():
                overlap = self._box_iou(box, track["box"])
                distance = self._center_distance(box, track["box"])
                if overlap >= 0.02 or distance <= 0.45:
                    candidates.append((overlap - 0.25 * distance, detection_index, track_id))

        assignments = {}
        used_tracks = set()
        for _, detection_index, track_id in sorted(candidates, reverse=True):
            if detection_index not in assignments and track_id not in used_tracks:
                assignments[detection_index] = track_id
                used_tracks.add(track_id)

        for track_id in list(self.tracks):
            if track_id not in used_tracks:
                self.tracks[track_id]["missed"] += 1
                if self.tracks[track_id]["missed"] > self.max_missing_frames:
                    del self.tracks[track_id]

        output = []
        self.last_track_ids = []
        for detection_index, (box, raw_pose) in enumerate(zip(current_boxes, poses)):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                initial_pose = np.asarray(raw_pose).copy()
                self.tracks[track_id] = {
                    "box": box,
                    "pose": initial_pose,
                    "joint_misses": {index: 0 for index in CONTROL_POINTS},
                    "joint_velocity": {
                        index: np.zeros(2, dtype=np.float32) for index in CONTROL_POINTS
                    },
                    "joint_candidates": {
                        index: None for index in CONTROL_POINTS
                    },
                    "missed": 0,
                }

            track = self.tracks[track_id]
            pose = np.asarray(raw_pose).copy()
            previous_pose = track["pose"]
            box_diagonal = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
            for point_index in CONTROL_POINTS:
                point = pose[point_index]
                previous = previous_pose[point_index]
                if point[2] > 0 and point[2] >= self.min_confidence and np.isfinite(point).all():
                    misses = track["joint_misses"][point_index]
                    jump = float(np.linalg.norm(point[:2] - previous[:2])) if previous[2] > 0 else float("inf")
                    candidate = track["joint_candidates"][point_index]
                    if previous[2] > 0 and jump > box_diagonal * self.outlier_jump_fraction:
                        if (
                            candidate is None
                            or float(np.linalg.norm(point[:2] - candidate))
                            > max(2.0, box_diagonal * self.outlier_confirm_fraction)
                        ):
                            track["joint_candidates"][point_index] = point[:2].copy()
                            pose[point_index, 2] = 0.0
                            point = pose[point_index]
                        else:
                            track["joint_candidates"][point_index] = None
                    else:
                        track["joint_candidates"][point_index] = None
                if point[2] > 0 and point[2] >= self.min_confidence and np.isfinite(point).all():
                    misses = track["joint_misses"][point_index]
                    jump = float(np.linalg.norm(point[:2] - previous[:2])) if previous[2] > 0 else float("inf")
                    if previous[2] > 0 and jump <= box_diagonal * 0.2:
                        deadband = max(0.5, box_diagonal * self.deadband_fraction)
                        if jump > deadband:
                            alpha = self.smoothing * float(np.clip(point[2], 0.25, 1.0))
                            pose[point_index, :2] = (
                                alpha * point[:2] + (1.0 - alpha) * previous[:2]
                            )
                        else:
                            pose[point_index, :2] = previous[:2]
                    if previous[2] > 0:
                        observed_velocity = (
                            pose[point_index, :2] - previous[:2]
                        ) / (misses + 1)
                        max_velocity = box_diagonal * 0.5
                        velocity_magnitude = float(np.linalg.norm(observed_velocity))
                        if velocity_magnitude > max_velocity:
                            observed_velocity *= max_velocity / velocity_magnitude
                        track["joint_velocity"][point_index] = (
                            0.5 * observed_velocity
                            + 0.5 * track["joint_velocity"][point_index]
                        )
                    track["joint_misses"][point_index] = 0
                elif previous[2] > 0 and track["joint_misses"][point_index] < self.max_missing_frames:
                    velocity = track["joint_velocity"][point_index]
                    pose[point_index, :2] = previous[:2] + velocity
                    pose[point_index, 2] = min(
                        previous[2] * 0.5,
                        max(0.0, self.min_confidence - np.finfo(np.float32).eps),
                    )
                    track["joint_misses"][point_index] += 1
                    track["joint_velocity"][point_index] *= 0.75
                else:
                    pose[point_index, 2] = 0.0
                    track["joint_velocity"][point_index].fill(0.0)
            track["box"] = box
            track["pose"] = pose.copy()
            track["missed"] = 0
            output.append(pose)
            self.last_track_ids.append(track_id)
        return output