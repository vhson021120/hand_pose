import unittest

import numpy as np

from tien_ich.on_dinh_co_tay import (
    OneEuroFilter3D,
    StabilityBenchmark,
    StabilizerConfig,
    TrackState,
    WristStabilizer,
)
from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristPreview


def make_pose(wrist_right=(0.30, 0.0, -0.10), vis=0.95):
    pose = np.zeros((33, 4))
    pose[:, 3] = vis
    pose[11] = (-0.20, 0.0, 0.0, vis)
    pose[12] = (0.20, 0.0, 0.0, vis)
    pose[13] = (-0.30, 0.25, 0.0, vis)
    pose[14] = (0.30, 0.25, 0.0, vis)
    pose[15] = (-0.30, 0.50, 0.0, vis)
    pose[16] = (*wrist_right, vis)
    pose[16, :3] = np.array(wrist_right)
    pose[14, :3] = (0.30, 0.0, -0.10) + np.array([0.0, 0.0, 0.0])
    return pose


def fast_config(**kw):
    base = dict(reacquire_frames=3, uncertain_max_frames=3)
    base.update(kw)
    return StabilizerConfig(**base)


def pose_with_wrist(offset):
    pose = make_pose()
    pose[14, :3] = (0.30, 0.20, 0.0)
    pose[16, :3] = np.array([0.30, 0.20, 0.0]) + np.array(offset)
    return pose


class OneEuroTests(unittest.TestCase):
    def test_constant_signal_is_preserved(self):
        f = OneEuroFilter3D(1.0, 5.0, 1.0)
        for k in range(20):
            out = f(k * 0.033, np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(out, [1.0, 2.0, 3.0], atol=1e-9)

    def test_noise_reduced_and_step_follows(self):
        rng = np.random.default_rng(1)
        f = OneEuroFilter3D(1.0, 5.0, 1.0)
        raw, out = [], []
        for k in range(200):
            v = np.array([0.1, 0.0, 0.0]) + rng.normal(0, 0.005, 3)
            raw.append(v)
            out.append(f(k * 0.033, v))
        self.assertLess(np.std(np.array(out)[50:, 0]), np.std(np.array(raw)[50:, 0]))
        for k in range(200, 215):
            last = f(k * 0.033, np.array([0.5, 0.0, 0.0]))
        self.assertGreater(last[0], 0.45)

    def test_alpha_depends_on_dt(self):
        slow, fast = OneEuroFilter3D(1.0, 0.0, 1.0), OneEuroFilter3D(1.0, 0.0, 1.0)
        for f in (slow, fast):
            f(0.0, np.zeros(3))
        a = slow(0.01, np.array([1.0, 0.0, 0.0]))[0]
        b = fast(0.2, np.array([1.0, 0.0, 0.0]))[0]
        self.assertGreater(b, a)

    def test_non_increasing_timestamp_returns_previous(self):
        f = OneEuroFilter3D(1.0, 1.0, 1.0)
        f(1.0, np.zeros(3))
        np.testing.assert_allclose(f(1.0, np.ones(3)), np.zeros(3))


class StabilizerTests(unittest.TestCase):
    def run_frames(self, stab, poses, dt=0.033, start=0.0):
        reports = []
        for k, pose in enumerate(poses):
            _, rep = stab.process(pose, start + k * dt)
            reports.append(rep["RIGHT"])
        return reports

    def test_disabled_returns_identical_copy(self):
        stab = WristStabilizer(StabilizerConfig(enabled=False))
        pose = make_pose()
        out, reports = stab.process(pose, 0.0)
        np.testing.assert_array_equal(out, pose)
        self.assertEqual(reports, {})
        self.assertIsNot(out, pose)

    def test_input_is_not_mutated(self):
        stab = WristStabilizer(fast_config())
        pose = pose_with_wrist((0.0, 0.2, 0.0))
        before = pose.copy()
        stab.process(pose, 0.0)
        np.testing.assert_array_equal(pose, before)

    def test_reacquiring_then_tracked(self):
        stab = WristStabilizer(fast_config())
        poses = [pose_with_wrist((0.0, 0.2, 0.0))] * 5
        states = [r.state for r in self.run_frames(stab, poses)]
        self.assertEqual(states[0], TrackState.REACQUIRING)
        self.assertEqual(states[2], TrackState.TRACKED)
        self.assertEqual(states[-1], TrackState.TRACKED)

    def test_reacquiring_output_blocked(self):
        stab = WristStabilizer(fast_config())
        out, _ = stab.process(pose_with_wrist((0.0, 0.2, 0.0)), 0.0)
        self.assertEqual(out[16, 3], 0.0)

    def test_short_dropout_is_uncertain_then_recovers(self):
        stab = WristStabilizer(fast_config())
        good = pose_with_wrist((0.0, 0.2, 0.0))
        bad = good.copy()
        bad[16, 3] = 0.1
        reports = self.run_frames(stab, [good] * 4 + [bad] * 2 + [good] * 2)
        self.assertEqual(reports[4].state, TrackState.UNCERTAIN)
        self.assertEqual(reports[5].state, TrackState.UNCERTAIN)
        self.assertEqual(reports[6].state, TrackState.TRACKED)

    def test_long_dropout_becomes_lost(self):
        stab = WristStabilizer(fast_config())
        good = pose_with_wrist((0.0, 0.2, 0.0))
        bad = good.copy()
        bad[16, 3] = 0.1
        reports = self.run_frames(stab, [good] * 4 + [bad] * 6)
        self.assertEqual(reports[-1].state, TrackState.LOST)

    def test_single_frame_jump_rejected_and_output_unchanged(self):
        stab = WristStabilizer(fast_config())
        good = pose_with_wrist((0.0, 0.2, 0.0))
        jump = pose_with_wrist((0.0, 0.2, 0.0))
        jump[16, :3] = jump[14, :3] + (0.2, 0.0, 0.0)
        reports = self.run_frames(stab, [good] * 5 + [jump, good])
        self.assertEqual(reports[5].state, TrackState.UNCERTAIN)
        self.assertEqual(reports[5].reason, "SPEED OUTLIER")
        self.assertEqual(reports[6].state, TrackState.TRACKED)

    def test_bone_length_outlier_rejected(self):
        stab = WristStabilizer(fast_config(max_speed_m_s=1000.0))
        good = pose_with_wrist((0.0, 0.2, 0.0))
        bad = pose_with_wrist((0.0, 0.2, 0.0))
        bad[16, :3] = good[14, :3] + (0.0, 0.0, 0.9)
        reports = self.run_frames(stab, [good] * 4 + [bad])
        self.assertEqual(reports[4].reason, "BONE LENGTH OUTLIER")

    def test_sustained_fast_move_reacquires_at_new_position(self):
        stab = WristStabilizer(fast_config(max_speed_m_s=1.0, uncertain_max_frames=1))
        a = pose_with_wrist((0.0, 0.2, 0.0))
        b = pose_with_wrist((0.0, 0.2, 0.0))
        b[16, :3] += (0.15, 0.0, 0.0)
        reports = self.run_frames(stab, [a] * 4 + [b] * 8)
        self.assertEqual(reports[-1].state, TrackState.TRACKED)
        self.assertIn(TrackState.REACQUIRING, [r.state for r in reports])

    def test_large_time_gap_drops_to_lost(self):
        stab = WristStabilizer(fast_config())
        good = pose_with_wrist((0.0, 0.2, 0.0))
        self.run_frames(stab, [good] * 4)
        _, rep = stab.process(good, 10.0)
        self.assertEqual(rep["RIGHT"].state, TrackState.REACQUIRING)

    def test_filtered_wrist_keeps_pose_world_frame(self):
        stab = WristStabilizer(fast_config())
        good = pose_with_wrist((0.0, 0.2, 0.0))
        out = None
        for k in range(6):
            out, _ = stab.process(good, k * 0.033)
        np.testing.assert_allclose(out[16, :3], good[16, :3], atol=1e-9)
        np.testing.assert_array_equal(out[11], good[11])

    def test_hand_confirmation_lowers_visibility_floor_only_in_same_pixel_frame(self):
        cfg = fast_config(hand_refine=True)
        stab = WristStabilizer(cfg)
        pose = pose_with_wrist((0.0, 0.2, 0.0))
        pose[16, 3] = 0.4
        pose2d = np.zeros((17, 3))
        pose2d[10] = (300.0, 200.0, 0.4)
        shape = (480, 640, 3)
        _, no_hand = stab.process(pose, 0.0, pose2d, shape, None)
        self.assertEqual(no_hand["RIGHT"].reason, "LOW WRIST CONFIDENCE")
        stab.reset()
        _, far = stab.process(pose, 0.0, pose2d, shape, [(600.0, 50.0)])
        self.assertEqual(far["RIGHT"].reason, "LOW WRIST CONFIDENCE")
        stab.reset()
        _, near = stab.process(pose, 0.0, pose2d, shape, [(305.0, 202.0)])
        self.assertTrue(near["RIGHT"].hand_confirmed)
        self.assertEqual(near["RIGHT"].reason, "REACQUIRING")

    def test_hand_refine_off_never_confirms(self):
        stab = WristStabilizer(fast_config(hand_refine=False))
        pose = pose_with_wrist((0.0, 0.2, 0.0))
        pose2d = np.zeros((17, 3))
        pose2d[10] = (300.0, 200.0, 0.9)
        _, rep = stab.process(pose, 0.0, pose2d, (480, 640, 3), [(300.0, 200.0)])
        self.assertFalse(rep["RIGHT"].hand_confirmed)

    def test_invalid_config_rejected(self):
        with self.assertRaises(ValueError):
            WristStabilizer(StabilizerConfig(min_cutoff_hz=0))


class PipelineTests(unittest.TestCase):
    def test_invalid_data_never_yields_model_relative(self):
        stab = WristStabilizer(fast_config())
        preview = RelativeWristPreview(calibration_frames=3, smoothing=1.0)
        good = pose_with_wrist((0.0, 0.2, 0.0))
        bad = good.copy()
        bad[16, 3] = 0.0
        statuses = []
        for k, p in enumerate([good] * 8 + [bad]):
            out, _ = stab.process(p, k * 0.033)
            states = preview.update([out])
            statuses.append([s.status for s in states if s.side == "RIGHT"][0])
        self.assertEqual(statuses[-1], "LOW CONFIDENCE")
        self.assertEqual(statuses[7], "MODEL RELATIVE")


class BenchmarkTests(unittest.TestCase):
    def test_summary_reports_dropout_jitter_and_latency(self):
        stab = WristStabilizer(fast_config())
        bench = StabilityBenchmark()
        rng = np.random.default_rng(0)
        for k in range(60):
            pose = pose_with_wrist((0.0, 0.2, 0.0))
            pose[16, :3] += rng.normal(0, 0.004, 3)
            if k == 30:
                pose[16, 3] = 0.0
            _, rep = stab.process(pose, k * 0.033)
            bench.record(k * 0.033, rep["RIGHT"], 0.2)
        s = bench.summary("RIGHT")
        self.assertEqual(s["frames"], 60)
        self.assertGreater(s["dropout_rate"], 0.0)
        self.assertLess(s["jitter_rms_mm"], s["raw_jitter_rms_mm"])
        self.assertAlmostEqual(s["process_ms_mean"], 0.2, places=3)


if __name__ == "__main__":
    unittest.main()
