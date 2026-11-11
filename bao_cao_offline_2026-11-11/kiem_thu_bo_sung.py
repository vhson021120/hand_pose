"""Supplemental offline checks; synthetic inputs, never measured camera data.

Phase 2 updates turn corrected limitations into regression expectations.
The original 31-test snapshot is preserved in the Phase 2 backup folder.
TCP checks reuse FakeRobotTask and allow only 127.0.0.1 connections.
Run from the repository: python bao_cao_offline_2026-11-11/kiem_thu_bo_sung.py -v
"""

import math
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from kiem_tra.kiem_tra_ket_noi_yumi import FakeRobotTask
from run import compute_workspace_roi
from tien_ich.ket_noi_yumi import YumiArmLink, format_target
from tien_ich.theo_doi_co_tay_2d import (
    WristXYTracker,
    screen_pixel_to_centered_mm,
    wrist_to_workspace_mm,
)

WORK = (400.0, 300.0)
ROI = (256.0, 72.0, 1024.0, 648.0)


class MappingChecks(unittest.TestCase):
    def assertMapping(self, pixel, expected, roi=ROI):
        x, y, inside = wrist_to_workspace_mm(pixel, roi, WORK)
        self.assertAlmostEqual(x, expected[0])
        self.assertAlmostEqual(y, expected[1])
        self.assertEqual(inside, expected[2])

    def test_default_roi(self):
        self.assertEqual(compute_workspace_roi((1280, 720), WORK, 0.8), ROI)

    def test_other_resolution(self):
        roi = compute_workspace_roi((640, 480), WORK, 0.8)
        self.assertEqual(roi, (64.0, 48.0, 576.0, 432.0))
        self.assertMapping((384, 240), (50.0, 0.0, True), roi)

    def test_aspect_ratio_and_fill(self):
        left, top, right, bottom = compute_workspace_roi((1000, 1000), WORK, 0.8)
        self.assertAlmostEqual((right - left) / (bottom - top), 4 / 3)
        self.assertAlmostEqual(right - left, 800)
        self.assertAlmostEqual(bottom - top, 600)

    def test_axes_corners_and_center(self):
        cases = [((640, 360), (0, 0, True)),
                 ((256, 72), (-200, 150, True)),
                 ((1024, 72), (200, 150, True)),
                 ((256, 648), (-200, -150, True)),
                 ((1024, 648), (200, -150, True)),
                 ((736, 360), (50, 0, True)),
                 ((544, 360), (-50, 0, True)),
                 ((640, 264), (0, 50, True)),
                 ((640, 456), (0, -50, True))]
        for pixel, expected in cases:
            with self.subTest(pixel=pixel):
                self.assertMapping(pixel, expected)

    def test_symmetry_about_center(self):
        for dx, dy in [(17, 23), (123, -56), (384, 288)]:
            a = wrist_to_workspace_mm((640 + dx, 360 + dy), ROI, WORK)
            b = wrist_to_workspace_mm((640 - dx, 360 - dy), ROI, WORK)
            self.assertAlmostEqual(a[0], -b[0])
            self.assertAlmostEqual(a[1], -b[1])

    def test_boundary_and_just_outside(self):
        for pixel, expected in [((1025, 360), (200, 0, False)),
                                ((255, 360), (-200, 0, False)),
                                ((640, 71), (0, 150, False)),
                                ((640, 649), (0, -150, False)),
                                ((1024, 360), (200, 0, True))]:
            with self.subTest(pixel=pixel):
                self.assertMapping(pixel, expected)

    def test_legacy_helper_has_different_scale(self):
        self.assertEqual(screen_pixel_to_centered_mm((740, 360), (1280, 720), 0.35),
                         (35.0, 0.0))
        self.assertMapping((740, 360), (100 * 25 / 48, 0, True))

    def test_zero_width_roi_is_invalid(self):
        self.assertEqual(wrist_to_workspace_mm((0, 0), (0.0, 0.0, 0.0, 1.0), WORK),
                         (0.0, 0.0, False))

    def test_nonfinite_mapping_characterization(self):
        result = wrist_to_workspace_mm((float('nan'), 360), ROI, WORK)
        self.assertEqual(result, (0.0, 0.0, False))
        self.assertMapping((float('inf'), 360), (0, 0, False))
        self.assertMapping((-float('inf'), 360), (0, 0, False))

    def test_flip_reflects_x_without_touching_y(self):
        # A synthetic three-channel array; no camera or model is opened.
        image = np.zeros((5, 8, 3), dtype=np.uint8)
        image[2, 1] = (11, 22, 33)
        flipped = cv2.flip(image, 1)
        np.testing.assert_array_equal(flipped[2, 8 - 1 - 1], (11, 22, 33))


class TrackerChecks(unittest.TestCase):
    def test_constructor_rejects_nonfinite_and_out_of_range(self):
        for value in [0, -1, 1.1, float('nan'), float('inf')]:
            with self.subTest(smoothing=value), self.assertRaises(ValueError):
                WristXYTracker(smoothing=value)
        for value in [-0.1, 1.1, float('nan'), float('inf')]:
            with self.subTest(visibility=value), self.assertRaises(ValueError):
                WristXYTracker(min_visibility=value)

    def test_arms_are_filtered_independently(self):
        tracker = WristXYTracker(0.25)
        tracker.update({'LEFT': (100, 100, 0.9), 'RIGHT': (900, 200, 0.8)})
        result = tracker.update({'LEFT': (140, 100, 0.9), 'RIGHT': (860, 240, 0.8)})
        self.assertEqual(result['LEFT'][:2], (110.0, 100.0))
        self.assertEqual(result['RIGHT'][:2], (890.0, 210.0))

    def test_invalid_shapes_are_omitted(self):
        for point in [(1, 2), (1, 2, 3, 4), [[1, 2, 0.9]], ()]:
            with self.subTest(point=point):
                self.assertEqual(WristXYTracker().update({'LEFT': point}), {})

    def test_nonfinite_each_component_is_omitted(self):
        for value in [float('nan'), float('inf'), -float('inf')]:
            for index in range(3):
                point = [100, 200, 0.9]
                point[index] = value
                with self.subTest(value=value, index=index):
                    self.assertEqual(WristXYTracker().update({'LEFT': point}), {})

    def test_nonnumeric_input_is_omitted(self):
        self.assertEqual(WristXYTracker().update({'LEFT': ('bad', 20, 0.9)}), {})

    def test_missing_one_arm_resets_only_that_history(self):
        tracker = WristXYTracker(0.25)
        tracker.update({'LEFT': (100, 100, 0.9), 'RIGHT': (200, 100, 0.9)})
        tracker.update({'RIGHT': (240, 100, 0.9)})
        result = tracker.update({'LEFT': (500, 100, 0.9), 'RIGHT': (250, 100, 0.9)})
        self.assertEqual(result['LEFT'][:2], (500.0, 100.0))
        self.assertEqual(result['RIGHT'][:2], (220.0, 100.0))

    def test_raw_outside_can_remain_inside_after_ema(self):
        # EMA is unchanged; Phase 2 target gating independently rejects raw-outside.
        from run import process_wrist_frame
        pipeline_tracker = WristXYTracker(0.25)
        process_wrist_frame(pipeline_tracker, {'LEFT': (640, 360, 0.9)},
                            (1280, 720), WORK, 0.8, armed=True)
        _, _, rows = process_wrist_frame(pipeline_tracker, {'LEFT': (1100, 360, 0.9)},
                                         (1280, 720), WORK, 0.8, armed=True)
        self.assertFalse(rows[0]['raw_inside_roi'])
        self.assertTrue(rows[0]['filtered_inside_roi'])
        self.assertFalse(rows[0]['target_valid'])
        tracker = WristXYTracker(0.25)
        tracker.update({'LEFT': (640, 360, 0.9)})
        self.assertFalse(wrist_to_workspace_mm((1100, 360), ROI, WORK)[2])
        result = tracker.update({'LEFT': (1100, 360, 0.9)})['LEFT']
        self.assertEqual(result[:2], (755.0, 360.0))
        mapped = wrist_to_workspace_mm(result, ROI, WORK)
        self.assertAlmostEqual(mapped[0], 59.89583333333333)
        self.assertTrue(mapped[2])
        for _ in range(5):
            result = tracker.update({'LEFT': (1100, 360, 0.9)})['LEFT']
            self.assertTrue(wrist_to_workspace_mm(result, ROI, WORK)[2])
        result = tracker.update({'LEFT': (1100, 360, 0.9)})['LEFT']
        self.assertFalse(wrist_to_workspace_mm(result, ROI, WORK)[2])

    def test_finite_extreme_ema_output_is_omitted(self):
        tracker = WristXYTracker(0.25)
        tracker.update({'LEFT': (1e308, 0, 0.9)})
        with np.errstate(over='ignore', invalid='ignore'):
            result = tracker.update({'LEFT': (-1e308, 0, 0.9)})
        self.assertEqual(result, {})

    def test_frame_resize_resets_old_pixel_history(self):
        tracker = WristXYTracker(0.25)
        tracker.update({'LEFT': (640, 360, 0.9)}, image_size=(1280, 720))
        # New frame is half width and height; normalized wrist is still centered.
        result = tracker.update({'LEFT': (320, 180, 0.9)}, image_size=(640, 360))['LEFT']
        new_roi = compute_workspace_roi((640, 360), WORK, 0.8)
        mapped = wrist_to_workspace_mm(result, new_roi, WORK)
        self.assertEqual(mapped, (0.0, 0.0, True))


class MessageChecks(unittest.TestCase):
    def test_signed_xy_order(self):
        self.assertEqual(format_target((12.34, -56.78)), b'1,12.3,-56.8\n')
        self.assertEqual(format_target((-12.34, 56.78)), b'1,-12.3,56.8\n')

    def test_rounding_error_per_axis_at_most_point_zero_five(self):
        for value in [-199.999, -12.345, -0.05, -0.049, 0.05, 12.345, 199.999]:
            fields = format_target((value, -value)).decode('ascii').strip().split(',')
            self.assertLessEqual(abs(float(fields[1]) - value), 0.050000000001)
            self.assertLessEqual(abs(float(fields[2]) + value), 0.050000000001)

    def test_negative_zero_is_numerically_zero(self):
        self.assertEqual(format_target((-0.001, 0.001)), b'1,-0.0,0.0\n')
        self.assertEqual(float('-0.0'), 0.0)

    def test_nonfinite_targets_and_missing_are_hold(self):
        self.assertEqual(format_target(None), b'0,0.0,0.0\n')
        for value in [float('nan'), float('inf'), -float('inf')]:
            self.assertEqual(format_target((value, 2)), b'0,0.0,0.0\n')
            self.assertEqual(format_target((2, value)), b'0,0.0,0.0\n')

    def test_malformed_target_is_hold(self):
        for target in [(1,), (1, 2, 3), ('bad', 1)]:
            with self.subTest(target=target):
                self.assertEqual(format_target(target), b'0,0.0,0.0\n')

    def test_latest_target_age_and_expiry_without_sleep(self):
        link = YumiArmLink('TEST', '127.0.0.1', 1, max_age=0.5)
        with patch('tien_ich.ket_noi_yumi.time.monotonic', return_value=10.0):
            link.set_target((10, 20))
            link.set_target((-5, 7.5))
        with patch('tien_ich.ket_noi_yumi.time.monotonic', return_value=10.5):
            self.assertEqual(link.current_message(), b'1,-5.0,7.5\n')
        with patch('tien_ich.ket_noi_yumi.time.monotonic', return_value=10.5001):
            self.assertEqual(link.current_message(), b'0,0.0,0.0\n')

    def test_connect_timeout_enters_retry_path_without_network(self):
        link = YumiArmLink('MOCK_ONLY', '127.0.0.1', 1,
                           connect_timeout=0.2, retry_delay=0.7)
        with patch('tien_ich.ket_noi_yumi.socket.create_connection',
                   side_effect=socket.timeout('synthetic connect timeout')) as connect:
            with patch.object(link._stop, 'wait',
                              side_effect=lambda _delay: link._stop.set()) as retry_wait:
                link._run()
        connect.assert_called_once_with(('127.0.0.1', 1), timeout=0.2)
        retry_wait.assert_called_once_with(0.7)
        self.assertFalse(link.connected)


class LoopbackChecks(unittest.TestCase):
    def setUp(self):
        original_create_connection = socket.create_connection

        def loopback_only(address, *args, **kwargs):
            if address[0] != '127.0.0.1':
                raise AssertionError('Only explicit IPv4 loopback is allowed')
            return original_create_connection(address, *args, **kwargs)

        self.guard = patch('tien_ich.ket_noi_yumi.socket.create_connection',
                           side_effect=loopback_only)
        self.guard.start()
        self.addCleanup(self.guard.stop)
        self.robot = FakeRobotTask()
        self.addCleanup(self.robot.close)
        self.link = YumiArmLink('OFFLINE', '127.0.0.1', self.robot.port,
                                max_age=30, connect_timeout=0.3, retry_delay=0.05)
        self.addCleanup(self.link.close)
        self.link.start()
        self.robot.accept()
        self.link.set_target((12.34, -56.78))

    def read_lines(self, count):
        data = b''
        while data.count(b'\n') < count:
            chunk = self.robot.client.recv(64)
            if not chunk:
                self.fail('Unexpected connection close before complete response')
            data += chunk
        return data.splitlines(keepends=True)

    def test_fragmented_request_waits_for_newline(self):
        self.robot.client.sendall(b'R')
        self.robot.client.settimeout(0.1)
        with self.assertRaises(socket.timeout):
            self.robot.client.recv(64)
        self.link.set_target((-9.8, 7.6))
        self.robot.client.settimeout(5)
        self.robot.client.sendall(b'\n')
        self.assertEqual(self.read_lines(1), [b'1,-9.8,7.6\n'])

    def test_multiple_requests_in_one_chunk(self):
        self.robot.client.sendall(b'R\nR\nR\n')
        self.assertEqual(self.read_lines(3), [b'1,12.3,-56.8\n'] * 3)

    def test_receive_timeout_does_not_close_idle_connection(self):
        time.sleep(0.65)  # Exceeds the Python receive timeout of 0.5 s.
        self.assertEqual(self.robot.request(), '1,12.3,-56.8\n')

    def test_reconnect_after_idle_and_disconnect(self):
        time.sleep(0.65)
        self.robot.client.close()
        self.robot.accept()
        self.link.set_target((-1, 2))
        self.assertEqual(self.robot.request(), '1,-1.0,2.0\n')

    def test_unexpected_newline_message_gets_hold(self):
        # Phase 2 preserves the reply format but rejects requests other than R.
        self.robot.client.sendall(b'NOT_R\n')
        self.assertEqual(self.read_lines(1), [b'0,0.0,0.0\n'])


if __name__ == '__main__':
    unittest.main()
