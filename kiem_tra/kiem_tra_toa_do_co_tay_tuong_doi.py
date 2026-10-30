import unittest

import numpy as np

from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristPreview


class RelativeWristPreviewTest(unittest.TestCase):
    def setUp(self):
        self.preview = RelativeWristPreview(calibration_frames=2, smoothing=1.0)

    @staticmethod
    def pose_world(left_wrist=(0.4, 0.1, 0.0), right_wrist=(0.4, 0.1, 0.0), confidence=0.9):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[11, :3] = (-0.2, 0.0, 0.0)
        pose[15, :3] = left_wrist
        pose[12, :3] = (0.2, 0.0, 0.0)
        pose[16, :3] = right_wrist
        pose[[11, 12, 15, 16], 3] = confidence
        return pose

    def test_reports_shoulder_relative_xyz_displacement_after_neutral_calibration(self):
        self.preview.update([self.pose_world()])
        left, right = self.preview.update([self.pose_world()])
        self.assertEqual(left.status, "MODEL RELATIVE")
        np.testing.assert_allclose(left.delta_cm, (0.0, 0.0, 0.0), atol=1e-5)

        left, right = self.preview.update([
            self.pose_world(left_wrist=(0.5, 0.2, -0.1), right_wrist=(0.3, -0.1, 0.2))
        ])

        np.testing.assert_allclose(left.delta_cm, (10.0, 10.0, -10.0), atol=1e-5)
        np.testing.assert_allclose(right.delta_cm, (-10.0, -20.0, 20.0), atol=1e-5)
        self.assertEqual(right.status, "MODEL RELATIVE")

    def test_uses_midpoint_of_both_shoulders_as_xyz_origin(self):
        self.preview = RelativeWristPreview(calibration_frames=1, smoothing=1.0)
        pose = self.pose_world(left_wrist=(0.0, 0.2, 0.0))
        self.preview.update([pose])

        moved_other_shoulder = pose.copy()
        moved_other_shoulder[12, 0] = 0.4
        left, _ = self.preview.update([moved_other_shoulder])

        self.assertAlmostEqual(left.delta_cm[0], -10.0, places=4)

    def test_uses_median_neutral_to_reject_one_calibration_outlier(self):
        self.preview = RelativeWristPreview(calibration_frames=3, smoothing=1.0)
        self.preview.update([self.pose_world(left_wrist=(0.4, 0.1, 0.0))])
        self.preview.update([self.pose_world(left_wrist=(1.2, 0.1, 0.0))])
        left, _ = self.preview.update([self.pose_world(left_wrist=(0.42, 0.1, 0.0))])
        self.assertEqual(left.status, "MODEL RELATIVE")

        left, _ = self.preview.update([self.pose_world(left_wrist=(0.5, 0.1, 0.0))])

        self.assertAlmostEqual(left.delta_cm[0], 8.0, places=4)

    def test_rejects_low_confidence_and_implausible_arm_geometry(self):
        left, _ = self.preview.update([self.pose_world(confidence=0.4)])
        self.assertEqual(left.status, "LOW CONFIDENCE")
        self.assertEqual(left.calibration_frames, 0)

        pose = self.pose_world(left_wrist=(2.0, 0.0, 0.0))
        left, _ = self.preview.update([pose])
        self.assertEqual(left.status, "INVALID 3D GEOMETRY")
        self.assertEqual(left.calibration_frames, 0)

    def test_missing_person_does_not_discard_neutral_but_multiple_people_does(self):
        pose = self.pose_world()
        self.preview.update([pose])
        self.preview.update([pose])
        left, _ = self.preview.update([])
        self.assertTrue(left.calibrated)
        self.assertEqual(left.status, "NO 3D POSE")

        left, right = self.preview.update([pose, pose])

        self.assertFalse(left.calibrated)
        self.assertFalse(right.calibrated)
        self.assertEqual(left.status, "WAITING FOR EXACTLY ONE PERSON")


if __name__ == "__main__":
    unittest.main()
