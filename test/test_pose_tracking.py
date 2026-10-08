import unittest

import numpy as np

from utils.pose_tracking import ControlPoseTracker


class ControlPoseTrackerTest(unittest.TestCase):
    def setUp(self):
        self.tracker = ControlPoseTracker(max_missing_frames=2, smoothing=0.35)
        self.box = [(0, 0, 100, 100)]
        self.pose = np.zeros((17, 3), dtype=np.float32)
        self.pose[9] = [50, 50, 1]
        self.tracker.update(self.box, [self.pose])

    def test_smooths_small_wrist_jitter(self):
        pose = self.pose.copy()
        pose[9] = [54, 50, 1]

        result = self.tracker.update(self.box, [pose])[0]

        self.assertAlmostEqual(result[9, 0], 51.4, places=4)

    def test_bridges_only_two_missing_wrist_frames(self):
        missing_pose = np.zeros((17, 3), dtype=np.float32)
        first = self.tracker.update(self.box, [missing_pose])[0]
        second = self.tracker.update(self.box, [missing_pose])[0]
        third = self.tracker.update(self.box, [missing_pose])[0]

        self.assertGreater(first[9, 2], 0)
        self.assertGreater(second[9, 2], 0)
        self.assertEqual(third[9, 2], 0)

    def test_does_not_smooth_large_fast_motion(self):
        pose = self.pose.copy()
        pose[9] = [100, 50, 1]

        result = self.tracker.update(self.box, [pose])[0]

        self.assertEqual(result[9, 0], 100)


if __name__ == "__main__":
    unittest.main()