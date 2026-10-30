"""Native XY control: sample cadence, anti-jitter, model plane and actual YuMi IK."""
from dataclasses import FrozenInstanceError
from types import SimpleNamespace
import unittest

import numpy as np

from tien_ich.dieu_khien_xy_yumi import NativeXYController, XYControlConfig
from tien_ich.dong_hoc_yumi import ArmModel, JOINT_ORDER


class NativeXYControllerTest(unittest.TestCase):
    @staticmethod
    def pose(dx=0., dy=0., *, translate=(0., 0.)):
        points = np.zeros((17, 3), dtype=np.float64)
        points[0] = (320., 90., .99)
        points[1] = (308., 82., .99)
        points[2] = (332., 82., .99)
        for shoulder, elbow, wrist, x in ((5, 7, 9, 260.), (6, 8, 10, 380.)):
            points[shoulder] = (x, 140., .99)
            points[elbow] = (x, 210., .99)
            points[wrist] = (x + dx, 280. + dy, .99)
        points[11] = (275., 350., .99)
        points[12] = (365., 350., .99)
        points[:, :2] += translate
        return points

    def engine(self, **kwargs):
        values = {'calibration_frames': 1}
        values.update(kwargs)
        return NativeXYController(XYControlConfig(**values))

    def update(self, engine, index, pose=None, **kwargs):
        return engine.update(
            [self.pose() if pose is None else pose],
            [[180., 60., 460., 420.]], (640, 480),
            1000. + index * .2, **kwargs)

    def test_five_hz_sampling_precedes_neutral_calibration(self):
        engine = self.engine(calibration_frames=100)
        samples = 0
        for index in range(30):
            result = engine.update([self.pose()], [[180., 60., 460., 420.]],
                                   (640, 480), 1000. + index / 30.)
            samples += bool(result.sampled)
        self.assertEqual(samples, 5)
        for arm in result.arms:
            self.assertEqual(arm.calibration_frames, 5)
            self.assertEqual(arm.status, 'CALIBRATING')

    def test_xy_movement_preserves_imposed_camera_depth_and_robot_flange_z(self):
        engine = self.engine()
        initial = self.update(engine, 0)
        for index in range(1, 24):
            result = self.update(engine, index, self.pose(dx=20., dy=-15.))
            for arm in result.arms:
                self.assertEqual(arm.status, 'READY', arm.reason)
                original = initial.arm(arm.side)
                self.assertEqual(arm.camera_position_m[2], 1.)
                self.assertEqual(arm.robot_position_m[2], original.robot_position_m[2])
                fk = ArmModel(arm.side.lower()).forward(arm.joints_deg)
                self.assertLess(abs(fk[2, 3] - arm.fixed_robot_z_m), 1e-5)
        for arm in result.arms:
            delta = np.asarray(arm.robot_position_m) - initial.arm(arm.side).robot_position_m
            self.assertGreater(delta[0], .004)
            self.assertGreater(delta[1], .003)

    def test_small_static_wrist_noise_does_not_move_robot_or_joint_targets(self):
        engine = self.engine()
        initial = self.update(engine, 0)
        rng = np.random.default_rng(93)
        for index, (dx, dy) in enumerate(rng.uniform(-2., 2., (40, 2)), 1):
            result = self.update(engine, index, self.pose(dx, dy))
            for arm in result.arms:
                self.assertEqual(arm.status, 'READY', arm.reason)
                np.testing.assert_allclose(arm.robot_position_m,
                                           initial.arm(arm.side).robot_position_m, atol=1e-12)
                np.testing.assert_allclose(arm.joints_deg,
                                           initial.arm(arm.side).joints_deg, atol=1e-8)

    def test_single_spike_is_suppressed_and_rejected_outliers_do_not_poison_filter(self):
        engine = self.engine(deadband_m=0., ema_alpha=1.)
        initial = self.update(engine, 0)
        for index, dx in enumerate((0., 15., 0., 0.), 1):
            result = self.update(engine, index, self.pose(dx=dx))
            for arm in result.arms:
                np.testing.assert_allclose(arm.robot_position_m,
                                           initial.arm(arm.side).robot_position_m, atol=1e-12)
        for index in range(5, 9):
            result = self.update(engine, index, self.pose(dx=600.))
            self.assertTrue(all(arm.status != 'READY' for arm in result.arms))
        for index in range(9, 15):
            result = self.update(engine, index)
            for arm in result.arms:
                self.assertEqual(arm.status, 'READY', arm.reason)
                np.testing.assert_allclose(arm.robot_position_m,
                                           initial.arm(arm.side).robot_position_m, atol=1e-12)

    def test_common_body_translation_is_cancelled_by_shoulder_midpoint_reference(self):
        engine = self.engine(deadband_m=0., ema_alpha=1.)
        initial = self.update(engine, 0)
        for index in range(1, 8):
            result = self.update(engine, index,
                                 self.pose(translate=(index * 15., index * -8.)))
            for arm in result.arms:
                np.testing.assert_allclose(arm.robot_position_m,
                                           initial.arm(arm.side).robot_position_m, atol=1e-12)

    def test_actual_xy_and_joint_steps_obey_limits_without_leaving_plane(self):
        engine = self.engine(median_window=1, ema_alpha=1., deadband_m=0.,
                             max_xy_speed_m_s=.005, max_joint_step_deg=.15)
        previous = self.update(engine, 0)
        for index in range(1, 12):
            result = self.update(engine, index, self.pose(dx=80., dy=-60.))
            for arm in result.arms:
                self.assertEqual(arm.status, 'READY', arm.reason)
                old = previous.arm(arm.side)
                step = np.asarray(arm.robot_position_m) - old.robot_position_m
                self.assertLessEqual(np.linalg.norm(step), .001 + 1e-9)
                self.assertLessEqual(np.max(np.abs(np.asarray(arm.joints_deg) - old.joints_deg)), .15 + 1e-6)
                self.assertEqual(arm.robot_position_m[2], old.robot_position_m[2])
            previous = result

    def test_pause_holds_joints_and_resume_recalibrates_without_seed_jump(self):
        engine = self.engine()
        for index in range(14):
            before = self.update(engine, index, self.pose(dx=0. if index == 0 else 30.))
        paused = self.update(engine, 14, active=False)
        self.assertTrue(all(arm.status != 'READY' for arm in paused.arms))
        for arm in paused.arms:
            np.testing.assert_allclose(arm.joints_deg, before.arm(arm.side).joints_deg, atol=1e-8)
            self.assertTrue(all(row.velocity_deg_s is None and row.predicted_deg is None
                                for row in arm.rows))
        resumed = self.update(engine, 15, self.pose(dx=-40., dy=15.))
        for arm in resumed.arms:
            self.assertEqual(arm.status, 'READY', arm.reason)
            np.testing.assert_allclose(arm.joints_deg, before.arm(arm.side).joints_deg, atol=1e-5)
            self.assertEqual(arm.fixed_robot_z_m, before.arm(arm.side).fixed_robot_z_m)

    def test_track_change_restarts_calibration_even_if_no_empty_frame_was_seen(self):
        engine = self.engine(calibration_frames=2)
        self.update(engine, 0, track_ids=(1,))
        self.assertEqual(self.update(engine, 1, track_ids=(1,)).status, 'READY')
        changed = self.update(engine, 2, track_ids=(2,))
        self.assertTrue(all(arm.status != 'READY' for arm in changed.arms))
        self.assertTrue(all(arm.calibration_frames <= 1 for arm in changed.arms))

    def test_cached_or_flow_frame_never_advances_calibration(self):
        engine = self.engine(calibration_frames=20)
        first = self.update(engine, 0)
        cached = self.update(engine, 1, self.pose(dx=100.), fresh=False)
        self.assertFalse(cached.sampled)
        self.assertEqual([a.calibration_frames for a in cached.arms],
                         [a.calibration_frames for a in first.arms])

    def test_low_confidence_or_predicted_shoulders_suspend_both_arms(self):
        engine = self.engine()
        self.update(engine, 0)
        pose = self.pose()
        pose[6, 2] = .1
        result = self.update(engine, 1, pose)
        self.assertTrue(all(arm.status != 'READY' for arm in result.arms))
        result = self.update(engine, 2, joint_misses={5: 1})
        self.assertTrue(all(arm.status != 'READY' for arm in result.arms))

    def test_uncertain_source_wrist_is_not_used_as_fresh_target(self):
        engine = self.engine()
        self.update(engine, 0)
        result = self.update(engine, 1,
                             wrist_reports={'LEFT': SimpleNamespace(state='UNCERTAIN')})
        self.assertNotEqual(result.arm('LEFT').status, 'READY')
        self.assertEqual(result.arm('RIGHT').status, 'READY')

    def test_snapshot_expiry_clears_velocity_and_forecast_without_replaying_input(self):
        engine = self.engine()
        self.update(engine, 0)
        ready = self.update(engine, 1)
        expired = engine.snapshot(timestamp=1000.8)
        self.assertFalse(expired.sampled)
        for arm in expired.arms:
            self.assertNotEqual(arm.status, 'READY')
            self.assertTrue(all(row.velocity_deg_s is None and row.predicted_deg is None
                                for row in arm.rows))
            np.testing.assert_allclose(arm.joints_deg, ready.arm(arm.side).joints_deg)

    def test_output_model_is_immutable_on_one_plane_and_input_arrays_are_untouched(self):
        engine = self.engine()
        pose = self.pose()
        original = pose.copy()
        hand = np.zeros((21, 3), dtype=float)
        hand[:, :2] = (260., 280.)
        hand[5, :2] = (250., 265.)
        hand[17, :2] = (270., 265.)
        hand_original = hand.copy()
        snapshot = self.update(engine, 0, pose, hands=((hand, True),))
        np.testing.assert_array_equal(pose, original)
        np.testing.assert_array_equal(hand, hand_original)
        self.assertIsNotNone(snapshot.model_pose)
        self.assertFalse(snapshot.model_pose.flags.writeable)
        shown = snapshot.model_pose[:, 3] > 0
        np.testing.assert_allclose(snapshot.model_pose[shown, 2], 1.)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            snapshot.status = 'changed'
        for model_hand, _ in snapshot.model_hands:
            self.assertFalse(model_hand.flags.writeable)
            np.testing.assert_allclose(model_hand[:, 2], 1.)

    def test_joint_rows_have_abb_axis_order_and_never_show_seed_as_measurement(self):
        engine = self.engine(calibration_frames=2)
        first = self.update(engine, 0)
        for arm in first.arms:
            self.assertEqual([row.axis for row in arm.rows], list(range(1, 8)))
            self.assertTrue(all(row.angle_deg is None for row in arm.rows))
        ready = self.update(engine, 1)
        for arm in ready.arms:
            physical = dict(zip(JOINT_ORDER, arm.joints_deg))
            self.assertEqual([row.axis for row in arm.rows], list(range(1, 8)))
            for row in arm.rows:
                self.assertAlmostEqual(row.angle_deg, physical[row.axis])

    def test_all_displayed_body_points_share_sampling_and_small_noise_rejection(self):
        engine = self.engine()
        initial = self.update(engine, 0)
        moved = self.pose()
        moved[7, 0] += 20.
        between = engine.update([moved], [[180., 60., 460., 420.]], (640, 480), 1000.1)
        self.assertFalse(between.sampled)
        np.testing.assert_array_equal(between.model_pose, initial.model_pose)
        rng = np.random.default_rng(15)
        for index, noise in enumerate(rng.uniform(-2., 2., (20, 4)), 1):
            pose = self.pose()
            pose[7, :2] += noise[:2]
            pose[8, :2] += noise[2:]
            snapshot = self.update(engine, index, pose)
            # MediaPipe display indices 13/14 are the two COCO elbows.
            np.testing.assert_allclose(snapshot.model_pose[[13, 14], :3],
                                       initial.model_pose[[13, 14], :3], atol=1e-12)

    def test_sustained_model_motion_above_deadband_accumulates_through_ema(self):
        engine = self.engine()
        initial = self.update(engine, 0)
        pose = self.pose()
        # 10 px is 4.167 mm at the default gain. Its first EMA increment is
        # only 1.458 mm, so a filter that discards its EMA state would stick.
        pose[7, 0] += 10.
        original = pose.copy()
        for index in range(1, 14):
            snapshot = self.update(engine, index, pose)
        self.assertGreater(snapshot.model_pose[13, 0] - initial.model_pose[13, 0], .002)
        self.assertEqual(snapshot.model_pose[13, 2], 1.)
        np.testing.assert_array_equal(pose, original)


class NativeXYConfigTest(unittest.TestCase):
    def test_invalid_nonfinite_or_unbounded_settings_are_rejected(self):
        for keyword, value in [('camera_depth_m', 0.), ('analysis_hz', float('nan')),
                               ('analysis_hz', True), ('median_window', 2),
                               ('ema_alpha', 1.1), ('deadband_m', -.001),
                               ('camera_focal_px', (0., 800.)),
                               ('max_xy_speed_m_s', float('inf'))]:
            with self.subTest(keyword=keyword), self.assertRaises(ValueError):
                NativeXYController(XYControlConfig(**{keyword: value}))


if __name__ == '__main__':
    unittest.main()
