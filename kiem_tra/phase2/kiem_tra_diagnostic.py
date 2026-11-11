"""Synthetic diagnostic checks. No camera, model, or network access."""

import csv
import io
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from tien_ich.chan_doan_toa_do import (
    AsyncDiagnosticLogger, CSV_FIELDS, NOT_AVAILABLE,
    draw_coordinate_diagnostics, utc_timestamp,
)


class DiagnosticLoggerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "synthetic.csv"

    def read_rows(self):
        with self.path.open(encoding="utf-8", newline="") as stream:
            return list(csv.DictReader(stream))

    def test_required_columns_missing_values_and_no_robot_feedback(self):
        logger = AsyncDiagnosticLogger(self.path)
        self.assertTrue(logger.record({"frame_id": 1, "data_source": "SYNTHETIC",
                                       "robot_feedback": "invented position"}))
        self.assertTrue(logger.close())
        rows = self.read_rows()
        self.assertEqual(set(rows[0]), set(CSV_FIELDS))
        self.assertEqual(rows[0]["raw_x_pixel"], NOT_AVAILABLE)
        self.assertEqual(rows[0]["robot_feedback"], NOT_AVAILABLE)
        self.assertEqual(rows[0]["data_source"], "SYNTHETIC")
        self.assertEqual(rows[0]["event_type"], "FRAME")
        self.assertTrue(rows[0]["timestamp"].endswith("+00:00"))

    def test_frame_and_send_are_distinct_and_send_uses_supplied_payload(self):
        logger = AsyncDiagnosticLogger(self.path)
        context = {"timestamp": "SYNTHETIC_FRAME_TIME", "frame_id": 4,
                   "arm": "ROB_R", "wrist": "LEFT", "target_valid": True,
                   "x_target_mm": 12.345, "y_target_mm": -9.876,
                   "data_source": "SYNTHETIC"}
        logger.record(dict(context, x_sent_mm=999, send_timestamp="not a send"))
        logger.record(dict(context, event_type="SEND", x_sent_mm=12.3,
                           y_sent_mm=-9.9, sent_valid=True,
                           send_timestamp="SYNTHETIC_SEND_TIME"))
        self.assertTrue(logger.close())
        frame, sent = self.read_rows()
        self.assertEqual(frame["x_target_mm"], "12.345")
        self.assertEqual(frame["x_sent_mm"], NOT_AVAILABLE)
        self.assertEqual(frame["send_timestamp"], NOT_AVAILABLE)
        self.assertEqual(frame["sent_valid"], NOT_AVAILABLE)
        self.assertEqual(sent["x_sent_mm"], "12.3")
        self.assertEqual(sent["y_sent_mm"], "-9.9")
        self.assertEqual(sent["send_timestamp"], "SYNTHETIC_SEND_TIME")
        self.assertEqual(sent["frame_id"], frame["frame_id"])
        self.assertEqual(sent["arm"], "ROB_R")

    def test_record_takes_snapshot_and_nonfinite_values_are_unavailable(self):
        logger = AsyncDiagnosticLogger(self.path)
        row = {"frame_id": 1, "x_workspace_mm": float("nan"),
               "y_workspace_mm": float("inf"), "data_source": "SYNTHETIC"}
        logger.record(row)
        row["frame_id"] = 999
        self.assertTrue(logger.close())
        result = self.read_rows()[0]
        self.assertEqual(result["frame_id"], "1")
        self.assertEqual(result["x_workspace_mm"], NOT_AVAILABLE)
        self.assertEqual(result["y_workspace_mm"], NOT_AVAILABLE)

    def test_reopening_appends_without_duplicate_header(self):
        for frame_id in (1, 2):
            logger = AsyncDiagnosticLogger(self.path)
            logger.record({"frame_id": frame_id, "data_source": "SYNTHETIC"})
            self.assertTrue(logger.close())
        self.assertEqual([row["frame_id"] for row in self.read_rows()], ["1", "2"])

    def test_write_error_is_reported_and_future_records_are_dropped(self):
        entered = threading.Event()
        fail = threading.Event()

        def failing_open(_self):
            entered.set()
            if not fail.wait(timeout=2):
                raise RuntimeError("synthetic test synchronization timeout")
            raise OSError("synthetic disk failure")

        with patch.object(AsyncDiagnosticLogger, "_open_stream", failing_open):
            logger = AsyncDiagnosticLogger(self.path)
            self.assertTrue(entered.wait(timeout=2))
            self.assertTrue(logger.record({"data_source": "SYNTHETIC"}))
            fail.set()
            self.assertTrue(logger.close())
        self.assertIn("synthetic disk failure", logger.error)
        self.assertFalse(logger.record({"data_source": "SYNTHETIC"}))
        self.assertEqual(logger.dropped_rows, 2)

    def test_slow_disk_never_waits_in_record_and_queue_is_bounded(self):
        entered = threading.Event()
        release = threading.Event()
        finished = threading.Event()
        accepted = []

        def delayed_open(_self):
            entered.set()
            if not release.wait(timeout=2):
                raise RuntimeError("synthetic test synchronization timeout")
            return io.StringIO()

        with patch.object(AsyncDiagnosticLogger, "_open_stream", delayed_open):
            logger = AsyncDiagnosticLogger(self.path, queue_size=1)
            self.assertTrue(entered.wait(timeout=2))

            def publish():
                accepted.extend([logger.record({"frame_id": 1}),
                                 logger.record({"frame_id": 2})])
                finished.set()

            producer = threading.Thread(target=publish, daemon=True)
            producer.start()
            try:
                self.assertTrue(finished.wait(timeout=0.5), "record waited for disk")
                self.assertEqual(accepted, [True, False])
                self.assertEqual(logger.dropped_rows, 1)
                self.assertFalse(logger.close(timeout=0.01))
            finally:
                release.set()
                producer.join(timeout=2)
                self.assertTrue(logger.close())
        self.assertEqual(logger.written_rows, 1)

    def test_closed_logger_and_malformed_records_do_not_raise(self):
        logger = AsyncDiagnosticLogger(self.path)
        self.assertFalse(logger.record(123))
        self.assertTrue(logger.close())
        self.assertFalse(logger.record({"data_source": "SYNTHETIC"}))
        self.assertEqual(logger.dropped_rows, 2)

    def test_invalid_queue_and_close_timeout_are_rejected(self):
        for size in (0, -1, 0.5, True):
            with self.subTest(size=size), self.assertRaises(ValueError):
                AsyncDiagnosticLogger(self.path, queue_size=size)
        logger = AsyncDiagnosticLogger(self.path)
        try:
            for timeout in (-1, float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    logger.close(timeout)
        finally:
            logger.close()


class DiagnosticDisplayTests(unittest.TestCase):
    def test_render_is_optional_canvas_only_and_marks_expected_not_feedback(self):
        row = {"wrist": "LEFT", "arm": "ROB_L", "armed": True,
               "target_valid": True, "tcp_connected": False,
               "raw_x_pixel": 700.0, "raw_y_pixel": 300.0,
               "filtered_x_pixel": 680.0, "filtered_y_pixel": 320.0,
               "x_workspace_mm": 20.0, "y_workspace_mm": 30.0,
               "raw_inside_roi": True, "filtered_inside_roi": True,
               "x_target_mm": 20.0, "y_target_mm": 30.0,
               "data_source": "SYNTHETIC"}
        labels = []

        def collect(canvas, label, *_args, **_kwargs):
            labels.append(label)
            return canvas

        with patch("tien_ich.chan_doan_toa_do.cv2.putText", side_effect=collect), \
             patch("cv2.VideoCapture", side_effect=AssertionError("camera forbidden")), \
             patch("cv2.imshow", side_effect=AssertionError("window forbidden")):
            canvas = draw_coordinate_diagnostics([row], "ON; dropped=0")
        self.assertEqual(canvas.shape, (720, 1120, 3))
        text = "\n".join(labels)
        for expected in ("EXPECTED TCP TARGET", "ROBOT FEEDBACK: NOT_AVAILABLE",
                         "RAW pixel", "EMA pixel", "WORKSPACE mm", "ROI raw",
                         "ARMED", "DISCONNECTED", "SYNTHETIC", "LAST OBSERVED SEND"):
            self.assertIn(expected, text)

    def test_missing_data_render_hold(self):
        canvas = draw_coordinate_diagnostics([{}])
        self.assertEqual(canvas.shape, (720, 1120, 3))

    def test_timestamp_is_explicit_utc(self):
        self.assertTrue(utc_timestamp().endswith("+00:00"))


if __name__ == "__main__":
    unittest.main()
