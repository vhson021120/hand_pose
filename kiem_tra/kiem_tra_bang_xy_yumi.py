"""The integrated preview uses accepted XY snapshots instead of live raw pose."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from tien_ich.ve_mo_hinh_3d import render_relative_3d_canvas


def snapshot(status="READY"):
    pose = np.zeros((33, 4), dtype=np.float32)
    pose[11] = (-.04, -.04, 1., 1.)
    pose[12] = (.04, -.04, 1., 1.)
    pose[13] = (-.07, .0, 1., 1.)
    pose[14] = (.07, .0, 1., 1.)
    pose[15] = (-.09, .06, 1., 1.)
    pose[16] = (.09, .06, 1., 1.)
    arms = []
    for side in ("LEFT", "RIGHT"):
        rows = tuple(SimpleNamespace(axis=axis, angle_deg=axis * 2.,
                                     velocity_deg_s=1.5, predicted_deg=axis * 2. + .3)
                     for axis in range(1, 8))
        arms.append(SimpleNamespace(side=side, status=status, reason=None,
                                    calibration_frames=20, required=20,
                                    camera_position_m=(.01, -.02, 1.),
                                    robot_position_m=(.448, .2, .604),
                                    fixed_robot_z_m=.604, rows=rows,
                                    filter_state="TRACKING"))
    return SimpleNamespace(status=status, analysis_hz=5., camera_depth_m=1.,
                           arms=tuple(arms), model_pose=pose, model_hands=())


class NativeXYCanvasTest(unittest.TestCase):
    def test_vietnamese_reason_is_readable_with_opencv_font_without_mutating_state(self):
        state = snapshot()
        original = "X/Y da dung yen; Z camera dat truoc"
        vietnamese = "X/Y đã đứng yên; Z camera đặt trước"
        state.arms[0].reason = vietnamese
        with patch("tien_ich.ve_mo_hinh_3d.cv2.putText") as draw:
            render_relative_3d_canvas([], [], native_xy=state)
        texts = [call.args[1] for call in draw.call_args_list]
        self.assertIn(original, texts)
        self.assertTrue(all(text.isascii() for text in texts))
        self.assertEqual(state.arms[0].reason, vietnamese)

    def test_xy_render_ignores_live_raw_pose_and_does_not_mutate_snapshot(self):
        fixed = snapshot()
        before = fixed.model_pose.copy()
        raw_first = np.ones((33, 4), dtype=np.float32)
        raw_second = raw_first * 100.
        first = render_relative_3d_canvas([raw_first], [], native_xy=fixed)
        second = render_relative_3d_canvas([raw_second], [], native_xy=fixed)
        self.assertEqual(first.shape, (720, 1000, 3))
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(fixed.model_pose, before)

    def test_joint_rows_and_coordinate_systems_are_labeled_separately(self):
        with patch("tien_ich.ve_mo_hinh_3d.cv2.putText") as draw:
            render_relative_3d_canvas([], [], native_xy=snapshot())
        texts = [call.args[1] for call in draw.call_args_list]
        for axis in range(1, 8):
            self.assertEqual(texts.count(f"J{axis}"), 2)
        self.assertIn("CAMERA XY PLANE | Z = 1000 mm (SET)", texts)
        self.assertIn("Camera XYZ mm: +10.0, -20.0, +1000.0", texts)
        self.assertIn("Robot XYZ mm:  +448.0, +200.0, +604.0", texts)
        self.assertTrue(any("NO ROBOT COMMANDS" in text for text in texts))
        self.assertEqual(texts.count("+200ms deg"), 2)

    def test_held_snapshot_keeps_angles_but_hides_velocity_and_forecast(self):
        held = snapshot("HOLD")
        with patch("tien_ich.ve_mo_hinh_3d.cv2.putText") as draw:
            render_relative_3d_canvas([], [], native_xy=held)
        texts = [call.args[1] for call in draw.call_args_list]
        self.assertIn("+14.00", texts)
        self.assertNotIn("+14.30", texts)
        self.assertNotIn("+1.50", texts)
        self.assertEqual(texts.count("--"), 28)

    def test_missing_model_or_calibration_is_renderable_in_existing_window(self):
        pending = snapshot("CALIBRATING")
        pending.model_pose = None
        for arm in pending.arms:
            arm.rows = ()
            arm.calibration_frames = 3
            arm.camera_position_m = None
            arm.robot_position_m = None
        with patch("tien_ich.ve_mo_hinh_3d.cv2.putText") as draw:
            image = render_relative_3d_canvas([], [], native_xy=pending,
                                              width=640, height=480)
        texts = [call.args[1] for call in draw.call_args_list]
        self.assertEqual(image.shape, (480, 640, 3))
        self.assertEqual(texts.count("Hold neutral pose: 3/20 samples"), 2)
        self.assertEqual(texts.count("--"), 42)


if __name__ == "__main__":
    unittest.main()
