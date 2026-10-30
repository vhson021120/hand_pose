import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tien_ich.uoc_luong_khoang_cach import (
    MetricDistanceSmoother,
    RangeCalibration,
    ShoulderRangeEstimator,
    WorkingDistanceGate,
    fit_range_scale,
    passes_validation,
    range_error_metrics,
    wrist_depth_acceptable,
    wrist_depth_metrics,
    yaw_compensation_report,
)

SHAPE = (720, 1280, 3)


def make_inputs(distance_m, yaw_deg=0.0, wrist_dz=-0.2, calibration=None, visibility=0.9):
    calibration = calibration or RangeCalibration()
    focal = calibration.focal_px(SHAPE[1])
    width = calibration.shoulder_width_m
    shoulder_px = focal * width * math.cos(math.radians(yaw_deg)) / distance_m
    pose = np.zeros((17, 3), dtype=np.float32)
    pose[5] = (640 - shoulder_px / 2, 300, visibility)
    pose[6] = (640 + shoulder_px / 2, 300, visibility)
    pose[9] = (500, 450, visibility)
    pose[10] = (780, 450, visibility)
    world = np.zeros((33, 4), dtype=np.float32)
    half_x = width / 2 * math.cos(math.radians(yaw_deg))
    half_z = width / 2 * math.sin(math.radians(yaw_deg))
    world[11] = (-half_x, 0, -half_z, visibility)
    world[12] = (half_x, 0, half_z, visibility)
    world[15] = (-0.3, 0.3, wrist_dz, visibility)
    world[16] = (0.3, 0.3, wrist_dz, visibility)
    return pose, world


class ShoulderRangeEstimatorTest(unittest.TestCase):
    def test_recovers_known_chest_distance(self):
        for distance in (0.8, 1.0, 1.2):
            measurement = ShoulderRangeEstimator().measure(*make_inputs(distance), SHAPE)
            self.assertAlmostEqual(measurement.chest_distance_m, distance, places=2)

    def test_yaw_compensation_keeps_distance(self):
        measurement = ShoulderRangeEstimator().measure(*make_inputs(1.0, yaw_deg=20), SHAPE)
        self.assertAlmostEqual(measurement.chest_distance_m, 1.0, places=2)
        self.assertAlmostEqual(measurement.yaw_deg, 20, places=1)

    def test_rejects_large_rotation_and_low_confidence(self):
        estimator = ShoulderRangeEstimator()
        self.assertIsNone(estimator.measure(*make_inputs(1.0, yaw_deg=40), SHAPE))
        self.assertIn('ROTATED', estimator.last_reason)
        self.assertIsNone(estimator.measure(*make_inputs(1.0, visibility=0.3), SHAPE))
        self.assertEqual(estimator.last_reason, 'LOW SHOULDER CONFIDENCE')

    def test_wrist_depth_uses_world_offset(self):
        measurement = ShoulderRangeEstimator().measure(*make_inputs(1.0, wrist_dz=-0.25), SHAPE)
        self.assertAlmostEqual(measurement.wrists['LEFT'].depth_m, 0.75, places=2)
        self.assertGreater(measurement.wrists['LEFT'].distance_m, 0.75)
        self.assertLess(measurement.wrists['LEFT'].camera_xyz_m[0], 0)

    def test_missing_wrist_is_omitted(self):
        pose, world = make_inputs(1.0)
        pose[10, 2] = 0.0
        measurement = ShoulderRangeEstimator().measure(pose, world, SHAPE)
        self.assertNotIn('RIGHT', measurement.wrists)
        self.assertIn('LEFT', measurement.wrists)

    def test_scale_changes_distance(self):
        calibration = RangeCalibration(scale=0.5)
        pose, world = make_inputs(1.0)
        measurement = ShoulderRangeEstimator(calibration).measure(pose, world, SHAPE)
        self.assertAlmostEqual(measurement.chest_distance_m, 0.5, places=2)


class CalibrationTest(unittest.TestCase):
    def test_fit_scale_recovers_true_scale(self):
        calibration = RangeCalibration()
        base = calibration.focal_px(1280) * calibration.shoulder_width_m
        distances = [0.8, 1.0, 1.2]
        features = [0.6 * d / base for d in distances]  # camera "sees" 0.6x the nominal range
        self.assertAlmostEqual(
            fit_range_scale(distances, features, calibration, 1280), 1 / 0.6, places=4
        )

    def test_fit_requires_distinct_distances(self):
        with self.assertRaises(ValueError):
            fit_range_scale([1.0, 1.0], [0.001, 0.001], RangeCalibration(), 1280)

    def test_metrics_and_validation_thresholds(self):
        metrics = range_error_metrics([0.8, 1.0, 1.2], [0.81, 0.99, 1.22])
        self.assertAlmostEqual(metrics['mae_m'], 0.04 / 3, places=6)
        self.assertTrue(passes_validation(metrics))
        bad = range_error_metrics([0.8, 1.0, 1.2], [0.9, 1.0, 1.2])
        self.assertFalse(passes_validation(bad))

    def test_json_roundtrip_and_confirmation_must_be_true(self):
        calibration = RangeCalibration(scale=1.3, confirmed=True)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cal.json'
            calibration.save(path)
            self.assertTrue(RangeCalibration.from_json(path).confirmed)
            text = path.read_text(encoding='utf-8').replace('true', '"yes"')
            path.write_text(text, encoding='utf-8')
            self.assertFalse(RangeCalibration.from_json(path).confirmed)

    def test_invalid_calibration_rejected(self):
        with self.assertRaises(ValueError):
            RangeCalibration(scale=-1).check()


class SmootherAndGateTest(unittest.TestCase):
    def test_median_rejects_outlier_and_holds_short_dropout(self):
        smoother = MetricDistanceSmoother(window_size=5, max_missing=2)
        for value in (1.0, 1.0, 2.5, 1.0):
            result = smoother.update(1, value)
        self.assertAlmostEqual(result, 1.0)
        self.assertAlmostEqual(smoother.update(1, None), 1.0)
        smoother.update(1, None)
        self.assertIsNone(smoother.update(1, None))

    def test_gate_latches_after_consecutive_frames(self):
        gate = WorkingDistanceGate(1.0, 0.2, required_consecutive=3)
        self.assertEqual([gate.update(1, 7, 1.05) for _ in range(3)], [False, False, True])
        self.assertFalse(gate.update(1, 7, 1.05))

    def test_gate_rejects_far_near_multiple_and_resets(self):
        gate = WorkingDistanceGate(1.0, 0.2, required_consecutive=2)
        self.assertFalse(gate.update(1, 1, 1.5))
        self.assertEqual(gate.status, 'TOO FAR')
        self.assertFalse(gate.update(1, 1, 0.5))
        self.assertEqual(gate.status, 'TOO NEAR')
        self.assertFalse(gate.update(2, None, 1.0))
        self.assertFalse(gate.update(1, 1, 1.0))
        self.assertFalse(gate.update(1, 1, 1.5))
        self.assertFalse(gate.update(1, 1, 1.0))

    def test_zone_edges_are_80_to_120_cm(self):
        gate = WorkingDistanceGate(1.0, 0.2, required_consecutive=1, latch=False)
        self.assertTrue(gate.update(1, 1, 0.81))
        self.assertTrue(gate.update(1, 1, 1.19))
        self.assertFalse(gate.update(1, 1, 0.78))
        self.assertFalse(gate.update(1, 1, 1.22))

    def test_non_latching_gate_reports_every_stable_frame(self):
        gate = WorkingDistanceGate(1.0, 0.2, required_consecutive=2, latch=False)
        self.assertEqual([gate.update(1, 3, 1.0) for _ in range(4)], [False, True, True, True])
        self.assertFalse(gate.update(1, 3, 1.5))


class IntrinsicsAndEvaluationTest(unittest.TestCase):
    def test_intrinsics_used_for_range_and_xyz(self):
        calibration = RangeCalibration(
            image_width=1280, image_height=720, fx_px=900.0, fy_px=910.0,
            cx_px=650.0, cy_px=350.0, reprojection_rmse_px=0.4,
        )
        self.assertTrue(calibration.intrinsics_calibrated)
        self.assertAlmostEqual(calibration.focal_px(640), 450.0)
        self.assertEqual(calibration.principal_point(640, 360), (325.0, 175.0))
        pose, world = make_inputs(1.0, calibration=calibration)
        measurement = ShoulderRangeEstimator(calibration).measure(pose, world, SHAPE)
        self.assertAlmostEqual(measurement.chest_distance_m, 1.0, places=2)
        self.assertAlmostEqual(measurement.chest_xyz_m[0], (640 - 650) / 900.0, places=2)
        self.assertAlmostEqual(measurement.chest_xyz_m[1], (300 - 350) / 910.0, places=2)

    def test_defaults_are_not_marked_calibrated(self):
        self.assertFalse(RangeCalibration().intrinsics_calibrated)
        self.assertEqual(RangeCalibration().principal_point(1280, 720), (640.0, 360.0))

    def test_intrinsics_roundtrip_and_old_files_load(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'cal.json'
            RangeCalibration(fx_px=900.0, fy_px=905.0, cx_px=640.0, cy_px=360.0,
                             reprojection_rmse_px=0.3).save(path)
            loaded = RangeCalibration.from_json(path)
            self.assertEqual((loaded.fx_px, loaded.fy_px, loaded.reprojection_rmse_px), (900.0, 905.0, 0.3))
            path.write_text('{"image_width":1280,"image_height":720,"shoulder_width_m":0.38,'
                            '"scale":1.0}', encoding='utf-8')
            self.assertEqual(RangeCalibration.from_json(path).fy_px, 0.0)

    def test_negative_intrinsics_rejected(self):
        with self.assertRaises(ValueError):
            RangeCalibration(cx_px=-1.0).check()

    def test_yaw_report_shows_compensation_helps(self):
        calibration = RangeCalibration()
        constant = calibration.range_constant(1280)
        yaw = 30.0
        pixel = calibration.focal_px(1280) * calibration.shoulder_width_m * math.cos(math.radians(yaw)) / 1.0
        feature = math.cos(math.radians(yaw)) / pixel
        report = yaw_compensation_report(1.0, [feature] * 5, [yaw] * 5, constant)
        self.assertAlmostEqual(report['compensated_error_m'], 0.0, places=4)
        self.assertGreater(abs(report['uncompensated_error_m']), 0.0)

    def test_wrist_depth_metrics_good_and_biased(self):
        good = wrist_depth_metrics([0.0, 0.2, 0.4], [0.0, -0.21, -0.39])
        self.assertTrue(good['sign_correct'])
        self.assertTrue(wrist_depth_acceptable(good))
        self.assertAlmostEqual(good['gain'], 1.0, delta=0.1)
        biased = wrist_depth_metrics([0.0, 0.2, 0.4], [0.0, -0.1, -0.2])
        self.assertFalse(wrist_depth_acceptable(biased))
        self.assertAlmostEqual(biased['gain'], 2.0, places=3)

    def test_wrist_depth_wrong_sign_rejected(self):
        flipped = wrist_depth_metrics([0.0, 0.2, 0.4], [0.0, 0.2, 0.4])
        self.assertFalse(flipped['sign_correct'])
        self.assertFalse(wrist_depth_acceptable(flipped))


if __name__ == '__main__':
    unittest.main()
