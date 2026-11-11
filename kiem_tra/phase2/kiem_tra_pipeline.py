"""Synthetic malformed-input and coordinate regression checks; no devices."""

import unittest

from run import compute_workspace_roi, process_wrist_frame
from tien_ich.theo_doi_co_tay_2d import WristXYTracker, wrist_to_workspace_mm


class PipelineValidationTests(unittest.TestCase):
    def test_malformed_observations_omit_and_clear_history(self):
        for value in [None, "bad", ("bad", 2, 0.9), [[1, 2], [3]],
                      (1, 2), (1, 2, 0.9, 4), (1, 2, 1.1), (1, 2, -1)]:
            with self.subTest(value=value):
                tracker = WristXYTracker()
                tracker.update({"LEFT": (100, 200, 0.9)})
                self.assertEqual(tracker.update({"LEFT": value}), {})
                self.assertEqual(tracker.update({"LEFT": (500, 200, 0.9)})["LEFT"][0], 500)

    def test_non_mapping_detections_and_invalid_size_hold(self):
        tracker = WristXYTracker()
        for detections in [None, [], "bad"]:
            _, _, rows = process_wrist_frame(tracker, detections, (1280, 720),
                                             (400, 300), 0.8, armed=True)
            self.assertTrue(all(not row["target_valid"] for row in rows))
        self.assertEqual(tracker.update({"LEFT": (1, 2, 0.9)}, image_size=(0, 720)), {})

    def test_invalid_roi_or_workspace_never_maps_as_valid(self):
        for roi in [(0, 0, 0, 1), (2, 0, 1, 1), (0, 1, 1, 0),
                    (0, 0, float("inf"), 1), (0, 0, 1), "bad", None]:
            with self.subTest(roi=roi):
                self.assertEqual(wrist_to_workspace_mm((1, 2), roi, (400, 300)),
                                 (0.0, 0.0, False))
        for work in [(0, 300), (-1, 2), (float("nan"), 2), (400,), None]:
            with self.subTest(work=work):
                self.assertFalse(wrist_to_workspace_mm((640, 360), (256, 72, 1024, 648), work)[2])

    def test_invalid_coordinate_shape_or_value_never_maps_as_valid(self):
        for point in [None, (), (1,), (1, 2, 3, 4), ("bad", 2),
                      (float("nan"), 2), (1, float("inf")), [[1], [2]]]:
            with self.subTest(point=point):
                self.assertEqual(wrist_to_workspace_mm(point, (0, 0, 100, 100), (400, 300)),
                                 (0.0, 0.0, False))

    def test_invalid_roi_configuration_is_rejected_or_hold(self):
        for size, work, fill in [((0, 720), (400, 300), 0.8),
                                 ((1280, 720), (0, 300), 0.8),
                                 ((1280, 720), (400, 300), float("nan")),
                                 ((1280, 720), (400, 300), "bad")]:
            with self.subTest(size=size, work=work, fill=fill):
                with self.assertRaises(ValueError):
                    compute_workspace_roi(size, work, fill)
                _, _, rows = process_wrist_frame(WristXYTracker(), {"LEFT": (640, 360, 0.9)},
                                                 size, work, fill, armed=True)
                self.assertFalse(rows[0]["target_valid"])
                self.assertEqual(rows[0]["x_workspace_mm"], "NOT_AVAILABLE")
                self.assertEqual(rows[0]["y_workspace_mm"], "NOT_AVAILABLE")

    def test_history_resets_on_repeated_resolution_changes(self):
        tracker = WristXYTracker()
        for size in [(1280, 720), (640, 480), (1280, 720)]:
            center = (size[0] / 2, size[1] / 2, 0.9)
            _, _, rows = process_wrist_frame(tracker, {"LEFT": center}, size,
                                             (400, 300), 0.8, armed=True)
            self.assertEqual((rows[0]["x_target_mm"], rows[0]["y_target_mm"]), (0.0, 0.0))

    def test_ema_equation_and_default_coefficient_are_preserved(self):
        tracker = WristXYTracker()
        tracker.update({"LEFT": (100, 100, 0.9)}, image_size=(1280, 720))
        self.assertEqual(tracker.update({"LEFT": (120, 100, 0.9)}, image_size=(1280, 720))["LEFT"],
                         (105.0, 100.0, 0.9))

    def test_frame_row_is_unrounded_intent_not_a_send(self):
        _, _, rows = process_wrist_frame(WristXYTracker(), {"LEFT": (755, 360, 0.9)},
                                         (1280, 720), (400, 300), 0.8,
                                         armed=True, frame_id=8, data_source="SYNTHETIC")
        row = rows[0]
        self.assertEqual(row["event_type"], "FRAME")
        self.assertEqual(row["frame_id"], 8)
        self.assertAlmostEqual(row["x_target_mm"], 59.89583333333333)
        for key in ("x_sent_mm", "y_sent_mm", "send_timestamp", "robot_feedback"):
            self.assertEqual(row[key], "NOT_AVAILABLE")


if __name__ == "__main__":
    unittest.main()
