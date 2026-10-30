import unittest
from unittest.mock import patch

import numpy as np

from tien_ich.theo_doi_quang_hoc import PoseFlowTracker


class PoseFlowTrackerTest(unittest.TestCase):
    def setUp(self):
        self.gray = np.zeros((32, 32), dtype=np.uint8)
        self.pose = np.zeros((17, 3), dtype=np.float32)
        self.pose[5] = [5, 5, 1]
        self.pose[7] = [10, 10, 1]
        self.pose[9] = [15, 15, 1]
        self.boxes = [(0, 0, 20, 20)]
        self.tracker = PoseFlowTracker(min_good_points=2)
        self.tracker.reset(self.gray, self.boxes, [self.pose])

    def test_handles_missing_lucas_kanade_outputs(self):
        with patch("tien_ich.theo_doi_quang_hoc.cv2.calcOpticalFlowPyrLK", return_value=(None, None, None)):
            boxes, poses = self.tracker.track(self.gray.copy())

        self.assertEqual(boxes, self.boxes)
        self.assertTrue(np.all(poses[0][[5, 7, 9], 2] == 0))

    def test_rejects_forward_backward_mismatch_per_keypoint(self):
        forward = self.pose[[5, 7, 9], :2] + [3, 0]
        backward = self.pose[[5, 7, 9], :2].copy()
        backward[2] += [8, 0]
        outputs = [
            (forward.reshape(-1, 1, 2), np.ones((3, 1), dtype=np.uint8), np.zeros((3, 1))),
            (backward.reshape(-1, 1, 2), np.ones((3, 1), dtype=np.uint8), np.zeros((3, 1))),
        ]
        with patch("tien_ich.theo_doi_quang_hoc.cv2.calcOpticalFlowPyrLK", side_effect=outputs):
            _, poses = self.tracker.track(self.gray.copy())

        np.testing.assert_allclose(poses[0][[5, 7], :2], self.pose[[5, 7], :2] + [3, 0])
        self.assertEqual(poses[0][9, 2], 0)

    def test_rejects_malformed_lucas_kanade_output(self):
        malformed = (
            np.zeros((1, 1, 2), dtype=np.float32),
            np.ones((1, 1), dtype=np.uint8),
            np.zeros((1, 1), dtype=np.float32),
        )
        with patch("tien_ich.theo_doi_quang_hoc.cv2.calcOpticalFlowPyrLK", return_value=malformed):
            _, poses = self.tracker.track(self.gray.copy())

        self.assertTrue(np.all(poses[0][[5, 7, 9], 2] == 0))


if __name__ == "__main__":
    unittest.main()
