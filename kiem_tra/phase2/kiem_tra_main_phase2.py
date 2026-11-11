"""Synthetic tests of run.main with every device, model, link and window mocked.

No camera constructor, model loader, socket or OpenCV window is used. These
checks exercise CLI wiring and the production loop; all pixel data is synthetic.
"""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import Mock, patch

import numpy as np

import run as application


class MockMainPhase2Tests(unittest.TestCase):
    def invoke_main(self, args=(), detections=None, sizes=None, keys=None,
                    read_failure=False, logger_error=None, logger_drained=True):
        """Run only against explicitly mocked I/O and return its observations."""
        if detections is None:
            detections = [{"LEFT": (736, 264, 0.9),
                           "RIGHT": (544, 456, 0.9)}]
        if sizes is None:
            sizes = [(1280, 720)] * len(detections)
        if keys is None:
            keys = [ord("q")]

        camera = Mock(name="SYNTHETIC_CAMERA")
        frames = [(True, np.zeros((height, width, 3), dtype=np.uint8))
                  for width, height in sizes]
        camera.read.side_effect = [(False, None)] if read_failure else frames
        pose = Mock(name="SYNTHETIC_POSE")
        pose.detect.side_effect = [[points] for points in detections]
        hand = Mock(name="SYNTHETIC_HAND")
        hand.detect.return_value = []
        links = {}

        def make_link(name, host, port, **kwargs):
            link = Mock()
            link.name = name
            link.connected = True
            link.start.return_value = link
            links[name] = link
            return link

        logger = Mock(name="SYNTHETIC_LOGGER")
        logger.error = logger_error
        logger.dropped_rows = 0
        logger.close.return_value = logger_drained
        output = io.StringIO()
        observation = dict(camera=camera, pose=pose, hand=hand, links=links,
                           logger=logger, output=output)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(output))
            stack.enter_context(redirect_stderr(output))
            stack.enter_context(patch.object(application.sys, "argv", ["run.py", *args]))
            # Guard the actual entry point too, so a future refactor cannot
            # accidentally bypass open_camera and touch a physical camera.
            stack.enter_context(patch.object(application.cv2, "VideoCapture",
                                            side_effect=AssertionError("Camera forbidden")))
            patches = {
                "open_camera": dict(return_value=camera),
                "MediaPipeWristDetector": dict(return_value=pose),
                "MediaPipeHandDetector": dict(return_value=hand),
                "YumiArmLink": dict(side_effect=make_link),
                "AsyncDiagnosticLogger": dict(return_value=logger),
                "draw_webcam_overlay": dict(side_effect=lambda frame, *_: frame.copy()),
                "draw_workspace_rect": {},
                "draw_wrist_xy_dashboard": dict(return_value=np.zeros((720, 960, 3), dtype=np.uint8)),
                "draw_robot_status": {},
                "draw_coordinate_diagnostics": dict(return_value=np.zeros((720, 1120, 3), dtype=np.uint8)),
            }
            for name, kwargs in patches.items():
                observation[name] = stack.enter_context(patch.object(application, name, **kwargs))
            for name, kwargs in {
                "namedWindow": {}, "imshow": {},
                "waitKey": dict(side_effect=keys),
                "getWindowProperty": dict(return_value=1),
                "destroyAllWindows": {},
            }.items():
                observation[name] = stack.enter_context(patch.object(application.cv2, name, **kwargs))
            try:
                observation["return_code"] = application.main()
            except (SystemExit, RuntimeError) as exc:
                observation["exception"] = exc
        return observation

    def assert_cleaned_up(self, observed):
        observed["camera"].release.assert_called_once_with()
        observed["pose"].close.assert_called_once_with()
        observed["hand"].close.assert_called_once_with()
        observed["destroyAllWindows"].assert_called_once_with()
        for link in observed["links"].values():
            link.close.assert_called_once_with()

    def test_default_windows_and_disabled_logging_are_preserved(self):
        observed = self.invoke_main()
        self.assertEqual(observed["return_code"], 0)
        self.assertEqual([call.args[0] for call in observed["namedWindow"].call_args_list],
                         ["Webcam", "Wrist XY"])
        self.assertEqual([call.args[0] for call in observed["imshow"].call_args_list],
                         ["Webcam", "Wrist XY"])
        observed["AsyncDiagnosticLogger"].assert_not_called()
        observed["draw_coordinate_diagnostics"].assert_not_called()
        self.assertEqual(set(observed["links"]), {"ROB_L", "ROB_R"})
        for call in observed["YumiArmLink"].call_args_list:
            self.assertIsNone(call.kwargs["on_send"])
        for link in observed["links"].values():
            self.assertIsNone(link.set_target.call_args.args[0])  # Starts disarmed.
        self.assert_cleaned_up(observed)

    def test_no_robot_does_not_construct_or_start_any_link(self):
        observed = self.invoke_main(["--no-robot"])
        self.assertEqual(observed["return_code"], 0)
        observed["YumiArmLink"].assert_not_called()
        observed["AsyncDiagnosticLogger"].assert_not_called()
        self.assertEqual(len(observed["namedWindow"].call_args_list), 2)
        self.assert_cleaned_up(observed)

    def test_diagnostic_options_add_panel_and_frame_records_only(self):
        observed = self.invoke_main(["--diagnostic", "--diagnostic-log", "SYNTHETIC.csv"])
        self.assertEqual(observed["return_code"], 0)
        observed["AsyncDiagnosticLogger"].assert_called_once_with("SYNTHETIC.csv")
        self.assertEqual([call.args[0] for call in observed["namedWindow"].call_args_list],
                         ["Webcam", "Wrist XY", "Coordinate Diagnostic"])
        observed["draw_coordinate_diagnostics"].assert_called_once()
        self.assertEqual(observed["logger"].record.call_count, 2)
        for call in observed["logger"].record.call_args_list:
            row = call.args[0]
            self.assertEqual(row["event_type"], "FRAME")
            self.assertEqual(row["robot_feedback"], "NOT_AVAILABLE")
            self.assertEqual(row["x_sent_mm"], "NOT_AVAILABLE")
            self.assertEqual(row["send_timestamp"], "NOT_AVAILABLE")
        for call in observed["YumiArmLink"].call_args_list:
            self.assertEqual(call.kwargs["on_send"], observed["logger"].record)
        observed["logger"].close.assert_called_once_with()
        self.assert_cleaned_up(observed)

    def test_nonfinite_fps_is_rejected_before_any_device_or_logger(self):
        for value in ("nan", "inf", "-inf"):
            with self.subTest(fps=value):
                observed = self.invoke_main(["--camera-fps=" + value])
                self.assertIsInstance(observed["exception"], SystemExit)
                self.assertEqual(observed["exception"].code, 2)
                for name in ("open_camera", "MediaPipeWristDetector", "MediaPipeHandDetector",
                             "YumiArmLink", "AsyncDiagnosticLogger", "namedWindow"):
                    observed[name].assert_not_called()

    def test_swap_cli_routes_generated_targets_after_space_arms(self):
        points = {"LEFT": (736, 264, 0.9), "RIGHT": (544, 456, 0.9)}
        observed = self.invoke_main(["--swap-arms"], detections=[points, points],
                                    keys=[ord(" "), ord("q")])
        self.assertEqual(observed["return_code"], 0)
        left_calls = observed["links"]["ROB_L"].set_target.call_args_list
        right_calls = observed["links"]["ROB_R"].set_target.call_args_list
        self.assertIsNone(left_calls[0].args[0])
        self.assertIsNone(right_calls[0].args[0])
        # Mapping remains unrounded; binary floats can differ around 1e-14.
        # Exact signs/order/one-decimal bytes are checked over real loopback.
        for actual, expected in zip(left_calls[1].args[0], (-50.0, -50.0)):
            self.assertAlmostEqual(actual, expected, places=10)
        for actual, expected in zip(right_calls[1].args[0], (50.0, 50.0)):
            self.assertAlmostEqual(actual, expected, places=10)
        self.assertEqual(left_calls[1].kwargs["context"]["wrist"], "RIGHT")
        self.assertEqual(right_calls[1].kwargs["context"]["wrist"], "LEFT")
        self.assertEqual(left_calls[1].kwargs["context"]["arm"], "ROB_L")
        self.assertEqual(right_calls[1].kwargs["context"]["arm"], "ROB_R")
        self.assertTrue(left_calls[1].kwargs["context"]["armed"])
        self.assert_cleaned_up(observed)

    def test_main_d01_raw_outside_never_publishes_filtered_inside_target(self):
        observed = self.invoke_main(detections=[
            {"LEFT": (640, 360, 0.9)},
            {"LEFT": (1100, 360, 0.9)},
            {"LEFT": (1100, 360, 0.9)},
        ], keys=[ord(" "), -1, ord("q")])
        self.assertEqual(observed["return_code"], 0)
        calls = observed["links"]["ROB_L"].set_target.call_args_list
        self.assertEqual(len(calls), 3)
        self.assertEqual(calls[1].kwargs["context"]["filtered_x_pixel"], 755.0)
        for call in calls[1:]:
            row = call.kwargs["context"]
            self.assertTrue(row["armed"])
            self.assertFalse(row["raw_inside_roi"])
            self.assertTrue(row["filtered_inside_roi"])
            self.assertFalse(row["target_valid"])
            self.assertIsNone(call.args[0])
        self.assert_cleaned_up(observed)

    def test_main_resize_resets_history_before_publishing_new_center(self):
        observed = self.invoke_main(detections=[
            {"LEFT": (640, 360, 0.9)}, {"LEFT": (320, 180, 0.9)},
        ], sizes=[(1280, 720), (640, 360)], keys=[ord(" "), ord("q")])
        self.assertEqual(observed["return_code"], 0)
        call = observed["links"]["ROB_L"].set_target.call_args_list[1]
        self.assertEqual(call.args[0], (0.0, 0.0))
        row = call.kwargs["context"]
        self.assertEqual((row["filtered_x_pixel"], row["filtered_y_pixel"]), (320.0, 180.0))
        self.assertTrue(row["frame_resized"])
        self.assertTrue(row["target_valid"])
        self.assert_cleaned_up(observed)

    def test_camera_read_failure_cleans_up_without_publishing_target(self):
        observed = self.invoke_main(read_failure=True)
        self.assertIsInstance(observed["exception"], RuntimeError)
        self.assertIn("Camera read failed", str(observed["exception"]))
        for link in observed["links"].values():
            link.set_target.assert_not_called()
        self.assert_cleaned_up(observed)

    def test_log_failure_and_incomplete_drain_do_not_interrupt_cleanup(self):
        observed = self.invoke_main(["--no-robot", "--diagnostic-log", "SYNTHETIC.csv"],
                                    logger_error="SYNTHETIC: lỗi ghi tệp", logger_drained=False)
        self.assertEqual(observed["return_code"], 0)
        self.assertIn("drained=False", observed["output"].getvalue())
        self.assert_cleaned_up(observed)

    def test_diagnostic_warning_tolerates_legacy_console_encoding(self):
        buffer = io.BytesIO()
        stream = io.TextIOWrapper(buffer, encoding="cp1252", errors="strict")
        with patch.object(application.sys, "stderr", stream):
            application.report_diagnostic_warning("SYNTHETIC: lỗi đường dẫn")
        stream.close()


if __name__ == "__main__":
    unittest.main()
