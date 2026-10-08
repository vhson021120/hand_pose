import unittest

import numpy as np

from utils.arm_jog_preview import ArmJogPreview


class ArmJogPreviewTest(unittest.TestCase):
    def setUp(self):
        self.preview = ArmJogPreview(calibration_frames=2, deadband=0.1)
        self.box = [(0, 0, 100, 200)]

    def pose(self, left_wrist=(50, 100), right_wrist=(150, 100)):
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[5] = [50, 50, 1]
        pose[9] = [*left_wrist, 1]
        pose[6] = [150, 50, 1]
        pose[10] = [*right_wrist, 1]
        return pose

    def calibrate(self):
        self.preview.update(self.box, [self.pose()])
        return self.preview.update(self.box, [self.pose()])

    def test_independent_screen_direction_per_arm(self):
        left, right = self.calibrate()
        self.assertTrue(left.calibrated)
        self.assertTrue(right.calibrated)

        left, right = self.preview.update(
            self.box, [self.pose(left_wrist=(90, 100), right_wrist=(150, 60))]
        )

        self.assertEqual(left.direction, "SCREEN RIGHT")
        self.assertAlmostEqual(left.delta_x_px, 40)
        self.assertEqual(right.direction, "SCREEN UP")
        self.assertAlmostEqual(right.delta_y_px, 40)

    def test_screen_left_and_down_are_reported(self):
        self.calibrate()

        left, right = self.preview.update(
            self.box, [self.pose(left_wrist=(10, 100), right_wrist=(150, 150))]
        )

        self.assertEqual(left.direction, "SCREEN LEFT")
        self.assertEqual(right.direction, "SCREEN DOWN")

    def test_small_motion_is_hold_and_missing_point_is_no_pose(self):
        self.calibrate()
        left, _ = self.preview.update(self.box, [self.pose(left_wrist=(55, 100))])
        self.assertEqual(left.direction, "HOLD")

        pose = self.pose()
        pose[9, 2] = 0
        left, _ = self.preview.update(self.box, [pose])
        self.assertEqual(left.direction, "NO POSE")

    def test_multiple_people_reset_neutral_calibration(self):
        self.calibrate()

        left, right = self.preview.update(self.box * 2, [self.pose(), self.pose()])

        self.assertFalse(left.calibrated)
        self.assertFalse(right.calibrated)
        self.assertEqual(left.direction, "WAITING FOR EXACTLY ONE PERSON")

    def test_explicit_recalibration_clears_baseline(self):
        self.calibrate()
        self.preview.reset()

        left, right = self.preview.update(self.box, [self.pose()])

        self.assertFalse(left.calibrated)
        self.assertFalse(right.calibrated)
        self.assertEqual(left.direction, "CALIBRATING")


if __name__ == "__main__":
    unittest.main()
