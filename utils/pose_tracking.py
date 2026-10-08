from typing import Dict, List, Sequence

import numpy as np


CONTROL_POINTS = (5, 6, 7, 8, 9, 10)


class ControlPoseTracker:
    """Smooth selected COCO arm joints and bridge brief detector dropouts."""

    def __init__(self, max_missing_frames: int = 2, smoothing: float = 0.35) -> None:
        self.max_missing_frames = max_missing_frames
        self.smoothing = smoothing
        self.tracks: Dict[int, dict] = {}
        self.next_track_id = 0

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
        for detection_index, (box, raw_pose) in enumerate(zip(current_boxes, poses)):
            track_id = assignments.get(detection_index)
            if track_id is None:
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = {
                    "box": box,
                    "pose": np.asarray(raw_pose).copy(),
                    "joint_misses": {index: 0 for index in CONTROL_POINTS},
                    "missed": 0,
                }

            track = self.tracks[track_id]
            pose = np.asarray(raw_pose).copy()
            previous_pose = track["pose"]
            box_diagonal = max(1.0, float(np.hypot(box[2] - box[0], box[3] - box[1])))
            for point_index in CONTROL_POINTS:
                point = pose[point_index]
                previous = previous_pose[point_index]
                if point[2] > 0 and np.isfinite(point[:2]).all():
                    jump = float(np.linalg.norm(point[:2] - previous[:2])) if previous[2] > 0 else float("inf")
                    if previous[2] > 0 and jump <= box_diagonal * 0.2:
                        pose[point_index, :2] = (self.smoothing * point[:2] +
                                                 (1.0 - self.smoothing) * previous[:2])
                    track["joint_misses"][point_index] = 0
                elif previous[2] > 0 and track["joint_misses"][point_index] < self.max_missing_frames:
                    pose[point_index, :2] = previous[:2]
                    pose[point_index, 2] = previous[2] * 0.5
                    track["joint_misses"][point_index] += 1
            track["box"] = box
            track["pose"] = pose.copy()
            track["missed"] = 0
            output.append(pose)
        return output