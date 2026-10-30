from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class PoseFlowTracker:
    """Propagate keypoints with forward-backward checked Lucas-Kanade optical flow."""

    def __init__(
        self,
        win_size: int = 21,
        max_error: float = 30.0,
        max_forward_backward_error: float = 1.5,
        min_good_points: int = 2,
        min_confidence: float = 0.35,
    ) -> None:
        if win_size < 1:
            raise ValueError("win_size must be positive")
        if max_error <= 0:
            raise ValueError("max_error must be positive")
        if max_forward_backward_error <= 0:
            raise ValueError("max_forward_backward_error must be positive")
        if min_good_points < 1:
            raise ValueError("min_good_points must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        self.win_size = (win_size, win_size)
        self.max_error = max_error
        self.max_forward_backward_error = max_forward_backward_error
        self.min_good_points = min_good_points
        self.min_confidence = min_confidence
        self.prev_gray: Optional[np.ndarray] = None
        self.boxes: List[Tuple[float, ...]] = []
        self.poses: List[np.ndarray] = []

    def reset(self, gray: np.ndarray, boxes: Sequence[Sequence[float]], poses: Sequence[np.ndarray]) -> None:
        self.prev_gray = gray
        self.boxes = [tuple(map(float, box)) for box in boxes]
        self.poses = [np.asarray(pose, dtype=np.float32).copy() for pose in poses]

    def track(
        self, gray: np.ndarray
    ) -> Tuple[List[Tuple[float, ...]], List[np.ndarray]]:
        if self.prev_gray is None or not self.poses:
            self.prev_gray = gray
            return list(self.boxes), list(self.poses)

        for index, (box, pose) in enumerate(zip(self.boxes, self.poses)):
            valid = np.flatnonzero(
                (pose[:, 2] >= self.min_confidence)
                & np.isfinite(pose[:, :2]).all(axis=1)
            )
            if len(valid) == 0:
                continue
            previous = pose[valid, :2].reshape(-1, 1, 2).astype(np.float32)
            current, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, previous, None, winSize=self.win_size, maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            moved = pose.copy()
            if current is None or status is None or error is None:
                moved[valid, 2] = 0.0
                self.poses[index] = moved
                continue

            if (
                current.size != len(valid) * 2
                or status.size != len(valid)
                or error.size != len(valid)
            ):
                moved[valid, 2] = 0.0
                self.poses[index] = moved
                continue

            forward_points = current.reshape(-1, 2)
            forward_valid = (
                (status.reshape(-1) == 1)
                & np.isfinite(error.reshape(-1))
                & (error.reshape(-1) < self.max_error)
                & np.isfinite(forward_points).all(axis=1)
            )
            forward_rows = np.flatnonzero(forward_valid)
            if len(forward_rows):
                backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray,
                    self.prev_gray,
                    current[forward_rows].reshape(-1, 1, 2),
                    None,
                    winSize=self.win_size,
                    maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
                )
                if (
                    backward is None
                    or backward_status is None
                    or backward.size != len(forward_rows) * 2
                    or backward_status.size != len(forward_rows)
                ):
                    forward_valid[forward_rows] = False
                else:
                    backward_error = np.linalg.norm(
                        backward.reshape(-1, 2) - previous[forward_rows, 0, :],
                        axis=1,
                    )
                    forward_valid[forward_rows] &= (
                        (backward_status.reshape(-1) == 1)
                        & np.isfinite(backward_error)
                        & (backward_error <= self.max_forward_backward_error)
                    )

            good_rows = np.flatnonzero(forward_valid)
            if len(good_rows) < self.min_good_points:
                moved[valid, 2] = 0.0
                self.poses[index] = moved
                continue

            shifts = forward_points[good_rows] - previous[good_rows, 0, :]
            shift = np.median(shifts, axis=0)
            moved = pose.copy()
            moved[valid[~forward_valid], 2] = 0.0
            for row in good_rows:
                point_index = valid[row]
                moved[point_index, :2] = forward_points[row]
            self.poses[index] = moved
            self.boxes[index] = (box[0] + shift[0], box[1] + shift[1], box[2] + shift[0], box[3] + shift[1])

        self.prev_gray = gray
        return list(self.boxes), list(self.poses)
