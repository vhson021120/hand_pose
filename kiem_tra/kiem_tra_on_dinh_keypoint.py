import unittest

import numpy as np

from tien_ich.on_dinh_keypoint import ControlPoseTracker


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

    def test_deadband_rejects_subpixel_wrist_jitter(self):
        for x in (50.6, 49.4, 50.8, 49.2):
            pose = self.pose.copy()
            pose[9] = [x, 50, 1]
            result = self.tracker.update(self.box, [pose])[0]

        self.assertAlmostEqual(result[9, 0], 50.0, places=5)

    def test_low_confidence_keypoint_is_smoothed_more_strongly(self):
        pose = self.pose.copy()
        pose[9] = [60, 50, 0.4]

        result = self.tracker.update(self.box, [pose])[0]

        self.assertGreater(result[9, 0], 50.0)
        self.assertLess(result[9, 0], 53.5)

    def test_bridges_only_two_missing_wrist_frames(self):
        missing_pose = np.zeros((17, 3), dtype=np.float32)
        first = self.tracker.update(self.box, [missing_pose])[0]
        second = self.tracker.update(self.box, [missing_pose])[0]
        third = self.tracker.update(self.box, [missing_pose])[0]

        self.assertGreater(first[9, 2], 0)
        self.assertGreater(second[9, 2], 0)
        self.assertEqual(third[9, 2], 0)

    def test_predicts_wrist_motion_during_a_brief_detection_dropout(self):
        moving_pose = self.pose.copy()
        moving_pose[9] = [60, 50, 1]
        observed = self.tracker.update(self.box, [moving_pose])[0]

        missing_pose = np.zeros((17, 3), dtype=np.float32)
        predicted = self.tracker.update(self.box, [missing_pose])[0]

        self.assertGreater(predicted[9, 0], observed[9, 0])
        self.assertGreater(predicted[9, 2], 0)

    def test_does_not_smooth_large_fast_motion(self):
        pose = self.pose.copy()
        pose[9] = [100, 50, 1]

        result = self.tracker.update(self.box, [pose])[0]

        self.assertEqual(result[9, 0], 100)

    def test_requires_consistent_second_frame_for_isolated_large_jump(self):
        jumped = self.pose.copy()
        jumped[9] = [125, 50, 1]

        first = self.tracker.update(self.box, [jumped])[0]
        second = self.tracker.update(self.box, [jumped])[0]

        self.assertEqual(first[9, 0], 50)
        self.assertLess(first[9, 2], self.tracker.min_confidence)
        self.assertEqual(second[9, 0], 125)
        self.assertEqual(second[9, 2], 1)

    def test_subthreshold_wrist_confidence_is_not_treated_as_a_valid_detection(self):
        pose = self.pose.copy()
        pose[9] = [90, 90, 0.2]

        result = self.tracker.update(self.box, [pose])[0]

        self.assertLess(result[9, 2], self.tracker.min_confidence)

    def test_track_identifier_changes_after_track_expires(self):
        initial_id = self.tracker.last_track_ids[0]
        self.tracker.update([], [])
        self.tracker.update([], [])
        self.tracker.update([], [])

        self.tracker.update(self.box, [self.pose])

        self.assertNotEqual(self.tracker.last_track_ids[0], initial_id)


if __name__ == "__main__":
    unittest.main()