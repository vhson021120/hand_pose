import unittest

import numpy as np

from tien_ich.dieu_khien_webcam import CaptureGestureController, classify_control_gesture


def make_hand(extended_fingers):
    landmarks = np.zeros((21, 3), dtype=np.float32)
    landmarks[0, :2] = (100, 200)
    landmarks[1, :2] = (88, 195)
    landmarks[2, :2] = (80, 190)
    landmarks[3, :2] = (88, 190)
    landmarks[4, :2] = (95, 190)
    finger_indices = ((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
    for index, (mcp, pip, tip) in enumerate(finger_indices):
        x = 70 + index * 20
        landmarks[mcp, :2] = (x, 170)
        landmarks[pip, :2] = (x, 140)
        landmarks[tip, :2] = (x, 90 if index in extended_fingers else 155)
    return landmarks


class WebcamControlsTest(unittest.TestCase):
    def test_classifies_control_gestures(self):
        self.assertEqual(classify_control_gesture(make_hand({0, 1, 2, 3})), "OPEN_PALM")
        self.assertEqual(classify_control_gesture(make_hand(set())), "FIST")
        self.assertEqual(classify_control_gesture(make_hand({0, 1})), "V_SIGN")

    def test_classifies_thumbs_up_separately_from_fist(self):
        landmarks = make_hand(set())
        landmarks[2, :2] = (80, 190)
        landmarks[3, :2] = (70, 145)
        landmarks[4, :2] = (65, 100)

        self.assertEqual(classify_control_gesture(landmarks), "THUMBS_UP")

    def test_ignores_unmapped_hand_pose(self):
        self.assertIsNone(classify_control_gesture(make_hand({0})))

    def test_state_specific_controls_use_hold_durations(self):
        controller = CaptureGestureController()
        self.assertIsNone(controller.update("OPEN_PALM", 0.0))
        self.assertIsNone(controller.update("THUMBS_UP", 0.0))
        self.assertIsNone(controller.update("THUMBS_UP", 0.99))
        self.assertEqual(controller.update("THUMBS_UP", 1.0), "start")
        self.assertEqual(controller.state, CaptureGestureController.RUNNING)

        self.assertIsNone(controller.update("OPEN_PALM", 2.0))
        self.assertIsNone(controller.update("OPEN_PALM", 4.99))
        self.assertEqual(controller.update("OPEN_PALM", 5.0), "pause")
        self.assertEqual(controller.state, CaptureGestureController.PAUSED)

        self.assertIsNone(controller.update("FIST", 6.0))
        self.assertEqual(controller.update("FIST", 6.4), "resume")
        self.assertEqual(controller.state, CaptureGestureController.RUNNING)

        self.assertIsNone(controller.update("V_SIGN", 7.0))
        self.assertIsNone(controller.update("V_SIGN", 8.99))
        self.assertEqual(controller.update("V_SIGN", 9.0), "stop")
        self.assertEqual(controller.state, CaptureGestureController.STOPPED)

    def test_lost_or_changed_gesture_restarts_hold_timer(self):
        controller = CaptureGestureController()
        controller.update("THUMBS_UP", 0.0)
        controller.update(None, 0.8)
        self.assertIsNone(controller.update("THUMBS_UP", 0.9))
        self.assertIsNone(controller.update("THUMBS_UP", 1.8))
        self.assertEqual(controller.update("THUMBS_UP", 1.9), "start")

    def test_terminal_state_cannot_emit_more_actions(self):
        controller = CaptureGestureController()
        controller.update("THUMBS_UP", 0.0)
        controller.update("THUMBS_UP", 1.0)
        controller.update("V_SIGN", 2.0)
        self.assertEqual(controller.update("V_SIGN", 4.0), "stop")
        self.assertIsNone(controller.update("THUMBS_UP", 10.0))

    def test_automatic_work_zone_start_transitions_waiting_state(self):
        controller = CaptureGestureController()

        self.assertTrue(controller.start_automatically())
        self.assertEqual(controller.state, CaptureGestureController.RUNNING)
        self.assertFalse(controller.start_automatically())


if __name__ == "__main__":
    unittest.main()
