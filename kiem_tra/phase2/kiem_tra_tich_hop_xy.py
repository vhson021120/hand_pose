"""Phase 2 integration checks using SYNTHETIC pixels and IPv4 loopback only.

MediaPipe model output is mocked; the production detector adapter, coordinate
pipeline, EMA, mapping, target publication and TCP transport are exercised.
Nothing here opens a camera, loads a model, or addresses a real controller.
"""

import queue
import socket
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from kiem_tra.kiem_tra_ket_noi_yumi import FakeRobotTask
from run import process_wrist_frame
from tien_ich.ket_noi_yumi import YumiArmLink
from tien_ich.nhan_dien_co_tay import MediaPipeWristDetector
from tien_ich.theo_doi_co_tay_2d import WristXYTracker


WORK_MM = (400.0, 300.0)
HOLD = "0,0.0,0.0\n"


class SyntheticCoordinateIntegrationTests(unittest.TestCase):
    def setUp(self):
        original_connect = socket.create_connection

        def loopback_only(address, *args, **kwargs):
            if address[0] != "127.0.0.1":
                raise AssertionError("Synthetic integration tests allow only 127.0.0.1")
            return original_connect(address, *args, **kwargs)

        guard = patch("tien_ich.ket_noi_yumi.socket.create_connection",
                      side_effect=loopback_only)
        guard.start()
        self.addCleanup(guard.stop)
        self.robots = {}
        self.links = {}
        self.sent_events = queue.Queue()
        for arm in ("ROB_L", "ROB_R"):
            robot = FakeRobotTask()
            self.robots[arm] = robot
            self.addCleanup(robot.close)
            link = YumiArmLink(arm, "127.0.0.1", robot.port,
                               max_age=30, connect_timeout=0.3,
                               retry_delay=0.03, on_send=self.sent_events.put)
            self.links[arm] = link
            self.addCleanup(link.close)
            link.start()
            robot.accept()

        self.detector = MediaPipeWristDetector.__new__(MediaPipeWristDetector)
        self.detector.image_module = Mock()
        self.detector.image_module.ImageFormat.SRGB = 1
        self.detector.image_module.Image.side_effect = lambda *_args: _args[-1]
        self.detector.last_timestamp = 0
        self.detector.min_visibility = 0.7
        self.detector.landmarker = Mock()
        self.tracker = WristXYTracker(smoothing=1.0, min_visibility=0.7)
        self.frame_id = 0

    def publish_frame(self, pixels, image_size=(1280, 720), armed=True, swap=False):
        """Inject synthetic model output, then follow the actual production path."""
        width, height = image_size
        landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.0)
                     for _ in range(33)]
        for wrist, index in (("LEFT", 15), ("RIGHT", 16)):
            if wrist in pixels:
                point = pixels[wrist]
                x, y = point[:2]
                visibility = point[2] if len(point) == 3 else 0.9
                landmarks[index] = SimpleNamespace(
                    x=x / width, y=y / height, visibility=visibility)
        self.detector.landmarker.detect_for_video.return_value = SimpleNamespace(
            pose_landmarks=[landmarks])
        detections = self.detector.detect(np.zeros((height, width, 3), dtype=np.uint8))[0]
        wrists, roi, rows = process_wrist_frame(
            self.tracker, detections, image_size, WORK_MM, 0.8,
            armed=armed, swap_arms=swap)
        self.frame_id += 1
        for row in rows:
            row.update(frame_id=self.frame_id, data_source="SYNTHETIC")
            target = ((row["x_target_mm"], row["y_target_mm"])
                      if row["target_valid"] else None)
            self.links[row["arm"]].set_target(target, context=row)
        return wrists, roi, {row["wrist"]: row for row in rows}

    def assert_wire(self, expected_left, expected_right=HOLD):
        self.assertEqual(self.robots["ROB_L"].request(), expected_left)
        self.assertEqual(self.robots["ROB_R"].request(), expected_right)

    def test_positive_and_negative_x_preserve_sign_and_axis(self):
        for point, expected in [((736, 360), "1,50.0,0.0\n"),
                                ((544, 360), "1,-50.0,0.0\n")]:
            with self.subTest(pixel=point):
                _, _, rows = self.publish_frame({"LEFT": point})
                self.assertTrue(rows["LEFT"]["target_valid"])
                self.assert_wire(expected)

    def test_positive_and_negative_y_preserve_sign_and_axis(self):
        for point, expected in [((640, 264), "1,0.0,50.0\n"),
                                ((640, 456), "1,0.0,-50.0\n")]:
            with self.subTest(pixel=point):
                self.publish_frame({"LEFT": point})
                self.assert_wire(expected)

    def test_all_roi_corners_remain_inclusive(self):
        cases = [((256, 72), "1,-200.0,150.0\n"),
                 ((1024, 72), "1,200.0,150.0\n"),
                 ((256, 648), "1,-200.0,-150.0\n"),
                 ((1024, 648), "1,200.0,-150.0\n")]
        for point, expected in cases:
            with self.subTest(pixel=point):
                _, _, rows = self.publish_frame({"LEFT": point})
                self.assertTrue(rows["LEFT"]["raw_inside_roi"])
                self.assertTrue(rows["LEFT"]["filtered_inside_roi"])
                self.assert_wire(expected)

    def test_outside_each_roi_edge_must_send_hold(self):
        for point in ((255, 360), (1025, 360), (640, 71), (640, 649)):
            with self.subTest(pixel=point):
                _, _, rows = self.publish_frame({"LEFT": point})
                self.assertFalse(rows["LEFT"]["raw_inside_roi"])
                self.assertFalse(rows["LEFT"]["target_valid"])
                self.assert_wire(HOLD)

    def test_d01_raw_outside_filtered_inside_must_send_hold(self):
        self.tracker = WristXYTracker(smoothing=0.25)
        self.publish_frame({"LEFT": (640, 360)})
        self.assert_wire("1,0.0,0.0\n")
        wrists, _, rows = self.publish_frame({"LEFT": (1100, 360)})
        self.assertEqual(wrists["LEFT"][:2], (755.0, 360.0))
        self.assertFalse(rows["LEFT"]["raw_inside_roi"])
        self.assertTrue(rows["LEFT"]["filtered_inside_roi"])
        self.assertFalse(rows["LEFT"]["target_valid"])
        self.assert_wire(HOLD)

    def test_filtered_outside_raw_inside_also_must_send_hold(self):
        self.tracker = WristXYTracker(smoothing=0.25)
        self.publish_frame({"LEFT": (1100, 360)})
        self.assert_wire(HOLD)
        _, _, rows = self.publish_frame({"LEFT": (1024, 360)})
        self.assertTrue(rows["LEFT"]["raw_inside_roi"])
        self.assertFalse(rows["LEFT"]["filtered_inside_roi"])
        self.assertFalse(rows["LEFT"]["target_valid"])
        self.assert_wire(HOLD)

    def test_left_and_right_route_to_their_default_links(self):
        _, _, rows = self.publish_frame({"LEFT": (736, 264), "RIGHT": (544, 456)})
        self.assertEqual(rows["LEFT"]["arm"], "ROB_L")
        self.assertEqual(rows["RIGHT"]["arm"], "ROB_R")
        self.assert_wire("1,50.0,50.0\n", "1,-50.0,-50.0\n")

    def test_swap_arms_changes_destination_not_coordinate_sign(self):
        _, _, rows = self.publish_frame(
            {"LEFT": (736, 264), "RIGHT": (544, 456)}, swap=True)
        self.assertEqual(rows["LEFT"]["arm"], "ROB_R")
        self.assertEqual(rows["RIGHT"]["arm"], "ROB_L")
        self.assert_wire("1,-50.0,-50.0\n", "1,50.0,50.0\n")

    def test_disarmed_overrides_valid_positions_with_hold(self):
        _, _, rows = self.publish_frame(
            {"LEFT": (736, 264), "RIGHT": (544, 456)}, armed=False)
        for row in rows.values():
            self.assertTrue(row["raw_inside_roi"])
            self.assertTrue(row["filtered_inside_roi"])
            self.assertFalse(row["armed"])
            self.assertFalse(row["target_valid"])
        self.assert_wire(HOLD)

    def test_missing_then_reacquired_point_does_not_reuse_old_ema(self):
        self.tracker = WristXYTracker(smoothing=0.25)
        self.publish_frame({"LEFT": (640, 360)})
        self.assert_wire("1,0.0,0.0\n")
        wrists, _, rows = self.publish_frame({})
        self.assertEqual(wrists, {})
        self.assertFalse(rows["LEFT"]["target_valid"])
        self.assert_wire(HOLD)
        wrists, _, rows = self.publish_frame({"LEFT": (736, 360)})
        self.assertEqual(wrists["LEFT"][:2], (736.0, 360.0))
        self.assertTrue(rows["LEFT"]["target_valid"])
        self.assert_wire("1,50.0,0.0\n")

    def test_low_visibility_is_hold_without_replaying_previous_target(self):
        self.publish_frame({"LEFT": (736, 360)})
        self.assert_wire("1,50.0,0.0\n")
        _, _, rows = self.publish_frame({"LEFT": (736, 360, 0.69)})
        self.assertFalse(rows["LEFT"]["target_valid"])
        self.assert_wire(HOLD)

    def test_resolution_change_keeps_new_frame_center_at_origin(self):
        self.tracker = WristXYTracker(smoothing=0.25)
        self.publish_frame({"LEFT": (640, 360)})
        self.assert_wire("1,0.0,0.0\n")
        wrists, _, rows = self.publish_frame(
            {"LEFT": (320, 180)}, image_size=(640, 360))
        self.assertEqual(wrists["LEFT"][:2], (320.0, 180.0))
        self.assertTrue(rows["LEFT"]["frame_resized"])
        self.assertEqual((rows["LEFT"]["x_workspace_mm"],
                          rows["LEFT"]["y_workspace_mm"]), (0.0, 0.0))
        self.assert_wire("1,0.0,0.0\n")

    def test_stale_target_transmits_hold_even_if_frame_was_valid(self):
        self.links["ROB_L"].max_age = 0.3
        with patch("tien_ich.ket_noi_yumi.time.monotonic", return_value=100.0):
            _, _, rows = self.publish_frame({"LEFT": (736, 360)})
        self.assertTrue(rows["LEFT"]["target_valid"])
        with patch("tien_ich.ket_noi_yumi.time.monotonic", return_value=100.301):
            self.assertEqual(self.robots["ROB_L"].request(), HOLD)

    def test_idle_receive_timeout_preserves_connection_and_target(self):
        self.publish_frame({"LEFT": (736, 264)})
        time.sleep(0.65)  # Synthetic loopback only; exceeds recv timeout 0.5 s.
        self.assert_wire("1,50.0,50.0\n")

    def test_disconnect_reconnect_uses_newest_target(self):
        self.publish_frame({"LEFT": (736, 264)})
        self.assert_wire("1,50.0,50.0\n")
        self.robots["ROB_L"].client.close()
        self.robots["ROB_L"].accept()
        self.publish_frame({"LEFT": (544, 456)})
        self.assert_wire("1,-50.0,-50.0\n")

    def test_created_target_and_actual_rounded_send_are_distinct(self):
        # Synthetic pixel x=755 gives X=59.895833..., rounded wire X=59.9.
        _, _, rows = self.publish_frame({"LEFT": (755, 360)})
        row = rows["LEFT"]
        self.assertAlmostEqual(row["x_target_mm"], 59.89583333333333)
        self.assertEqual(self.robots["ROB_L"].request(), "1,59.9,0.0\n")
        event = self.sent_events.get(timeout=2)
        self.assertEqual(event["event_type"], "SEND")
        self.assertEqual(event["frame_id"], self.frame_id)
        self.assertEqual(event["arm"], "ROB_L")
        self.assertEqual(event["wrist"], "LEFT")
        self.assertEqual(event["data_source"], "SYNTHETIC")
        self.assertAlmostEqual(event["x_target_mm"], row["x_target_mm"])
        self.assertEqual(event["x_sent_mm"], 59.9)
        self.assertEqual(event["y_sent_mm"], 0.0)
        self.assertTrue(event["sent_valid"])
        self.assertNotEqual(event["send_timestamp"], "NOT_AVAILABLE")
        self.assertEqual(event["robot_feedback"], "NOT_AVAILABLE")


if __name__ == "__main__":
    unittest.main()
