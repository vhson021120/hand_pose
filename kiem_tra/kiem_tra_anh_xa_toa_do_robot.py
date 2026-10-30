import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristState
from tien_ich.anh_xa_toa_do_robot import (
    RobotMappingConfig,
    RobotMappingCsvLogger,
    RobotTargetMapper,
)


class RobotTargetMappingTest(unittest.TestCase):
    def setUp(self):
        self.config = RobotMappingConfig(
            robot_frame="confirmed_base",
            tracked_side="RIGHT",
            home_robot_mm=(300.0, 100.0, 200.0),
            mm_per_cm_matrix=(
                (10.0, 0.0, 0.0),
                (0.0, -10.0, 0.0),
                (0.0, 0.0, 20.0),
            ),
            max_offset_mm=(100.0, 100.0, 100.0),
            max_speed_mm_s=1000.0,
            max_sample_interval_s=0.5,
            max_delta_step_cm=20.0,
            min_confidence=0.6,
            calibration_confirmed=True,
        )
        self.mapper = RobotTargetMapper(self.config)

    @staticmethod
    def state(delta_cm=(0.0, 0.0, 0.0), confidence=0.9, side="RIGHT"):
        return RelativeWristState(
            side=side,
            delta_cm=delta_cm,
            status="MODEL RELATIVE",
            confidence=confidence,
            calibration_frames=30,
            calibrated=True,
        )

    def test_zero_relative_offset_maps_to_the_configured_home(self):
        initializing = self.mapper.update(self.state(), 1.0)
        target = self.mapper.update(self.state(), 1.1)

        self.assertFalse(initializing.valid)
        self.assertEqual(initializing.status, "INITIALIZING")
        self.assertTrue(target.valid)
        self.assertEqual(target.target_mm, self.config.home_robot_mm)
        self.assertEqual(target.offset_mm, (0.0, 0.0, 0.0))

    def test_maps_cm_to_mm_with_axis_reflection_without_recalibrating_neutral(self):
        self.mapper.update(self.state(), 1.0)
        target = self.mapper.update(self.state((2.0, 3.0, -1.0)), 1.1)

        np.testing.assert_allclose(target.offset_mm, (20.0, -30.0, -20.0))
        np.testing.assert_allclose(target.target_mm, (320.0, 70.0, 180.0))

    def test_supports_calibrated_axis_permutation(self):
        config = RobotMappingConfig(
            **{
                **self.config.__dict__,
                "mm_per_cm_matrix": (
                    (0.0, 0.0, 20.0),
                    (10.0, 0.0, 0.0),
                    (0.0, -10.0, 0.0),
                ),
            }
        )
        mapper = RobotTargetMapper(config)
        mapper.update(self.state(), 1.0)

        result = mapper.update(self.state((2.0, 3.0, -1.0)), 1.1)

        np.testing.assert_allclose(result.offset_mm, (-20.0, 20.0, -30.0))

    def test_rejects_offset_outside_configured_workspace(self):
        self.mapper.update(self.state(), 1.0)
        result = self.mapper.update(self.state((11.0, 0.0, 0.0)), 1.1)

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "OFFSET OUT OF RANGE")
        self.assertIsNone(result.target_mm)

    def test_limits_target_speed(self):
        config = RobotMappingConfig(
            **{
                **self.config.__dict__,
                "max_speed_mm_s": 10.0,
            }
        )
        mapper = RobotTargetMapper(config)
        mapper.update(self.state(), 1.0)
        result = mapper.update(self.state((5.0, 0.0, 0.0)), 1.1)

        self.assertTrue(result.valid)
        self.assertEqual(result.status, "RATE LIMITED")
        self.assertAlmostEqual(result.target_mm[0], 301.0)

    def test_invalid_dt_reinitializes_without_producing_target(self):
        self.mapper.update(self.state(), 1.0)
        result = self.mapper.update(self.state(), 1.0)

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "INVALID DT")
        self.assertIsNone(result.target_mm)

    def test_stale_tracking_and_bad_confidence_do_not_produce_targets(self):
        self.mapper.update(self.state(), 1.0)
        stale = self.mapper.update(self.state(), 2.0)
        low_confidence = self.mapper.update(self.state(confidence=0.2), 2.1)

        self.assertEqual(stale.status, "STALE FRAME")
        self.assertIsNone(stale.target_mm)
        self.assertEqual(low_confidence.status, "LOW CONFIDENCE")
        self.assertFalse(low_confidence.valid)

    def test_rejects_nan_delta_large_jumps_and_wrong_side(self):
        self.mapper.update(self.state(), 1.0)
        nan_result = self.mapper.update(
            self.state((float("nan"), 0.0, 0.0)), 1.1
        )
        side_result = self.mapper.update(
            self.state(side="LEFT"), 1.2
        )

        self.assertEqual(nan_result.status, "INVALID 3D DELTA")
        self.assertEqual(side_result.status, "WRONG TRACKED SIDE")
        self.assertIsNone(side_result.target_mm)

    def test_rejects_a_large_relative_delta_jump(self):
        config = RobotMappingConfig(
            **{
                **self.config.__dict__,
                "max_offset_mm": (1000.0, 1000.0, 1000.0),
            }
        )
        mapper = RobotTargetMapper(config)
        mapper.update(self.state(), 1.0)

        result = mapper.update(self.state((21.0, 0.0, 0.0)), 1.1)

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "DELTA JUMP")
        self.assertIsNone(result.target_mm)

    def test_requires_explicit_calibration_confirmation(self):
        values = dict(self.config.__dict__)
        values["calibration_confirmed"] = False

        with self.assertRaisesRegex(ValueError, "calibration_confirmed"):
            RobotMappingConfig(**values)

    def test_loads_confirmed_mapping_configuration_from_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.json"
            path.write_text(json.dumps(self.config.__dict__), encoding="utf-8")

            loaded = RobotMappingConfig.from_json(str(path))

        self.assertEqual(loaded.robot_frame, "confirmed_base")
        self.assertEqual(loaded.home_robot_mm, self.config.home_robot_mm)
        self.assertEqual(loaded.mm_per_cm_matrix, self.config.mm_per_cm_matrix)

    def test_reset_requires_a_new_initialization_sample(self):
        self.mapper.update(self.state(), 1.0)
        self.assertTrue(self.mapper.update(self.state(), 1.1).valid)
        self.mapper.reset()

        result = self.mapper.update(self.state(), 1.2)

        self.assertFalse(result.valid)
        self.assertEqual(result.status, "INITIALIZING")

    def test_csv_logs_source_relative_delta_and_preview_target(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mapping.csv"
            logger = RobotMappingCsvLogger(str(path))
            self.mapper.update(self.state(), 1.0)
            preview = self.mapper.update(self.state((1.0, 0.0, 0.0)), 1.1)
            pose = np.zeros((33, 4), dtype=np.float32)
            pose[12, :3] = (0.1, 0.2, 0.3)
            pose[16, :3] = (0.4, 0.5, 0.6)
            logger.write(1.1, self.state((1.0, 0.0, 0.0)), preview, pose)
            logger.close()

            with path.open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))

        self.assertAlmostEqual(float(row["shoulder_x_m"]), 0.1, places=6)
        self.assertEqual(row["delta_x_cm"], "1.0")
        self.assertEqual(row["target_x_mm"], "310.0")
        self.assertEqual(row["robot_frame"], "confirmed_base")
        self.assertEqual(row["valid"], "True")


if __name__ == "__main__":
    unittest.main()
