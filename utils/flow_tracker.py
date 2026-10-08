from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np


class PoseFlowTracker:
    """Propagate keypoints and person boxes between detector frames with Lucas-Kanade optical flow."""

    def __init__(self, win_size: int = 21, max_error: float = 30.0) -> None:
        self.win_size = (win_size, win_size)
        self.max_error = max_error
        self.prev_gray: Optional[np.ndarray] = None
        self.boxes: List[Tuple[float, ...]] = []
        self.poses: List[np.ndarray] = []

    def reset(self, gray: np.ndarray, boxes: Sequence[Sequence[float]], poses: Sequence[np.ndarray]) -> None:
        self.prev_gray = gray
        self.boxes = [tuple(map(float, box)) for box in boxes]
        self.poses = [np.asarray(pose, dtype=np.float32).copy() for pose in poses]

    def track(self, gray: np.ndarray):
        if self.prev_gray is None or not self.poses:
            self.prev_gray = gray
            return list(self.boxes), list(self.poses)

        for index, (box, pose) in enumerate(zip(self.boxes, self.poses)):
            valid = np.flatnonzero(pose[:, 2] > 0)
            if len(valid) == 0:
                continue
            previous = pose[valid, :2].reshape(-1, 1, 2).astype(np.float32)
            current, status, error = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, previous, None, winSize=self.win_size, maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
            )
            good = (status.ravel() == 1) & (error.ravel() < self.max_error)
            if not good.any():
                continue
            shift = np.median(current.reshape(-1, 2)[good] - previous.reshape(-1, 2)[good], axis=0)
            moved = pose.copy()
            for row, point_index in enumerate(valid):
                moved[point_index, :2] = current[row, 0] if good[row] else pose[point_index, :2] + shift
            self.poses[index] = moved
            self.boxes[index] = (box[0] + shift[0], box[1] + shift[1], box[2] + shift[0], box[3] + shift[1])

        self.prev_gray = gray
        return list(self.boxes), list(self.poses)
