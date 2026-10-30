"""Native five-Hz XY preview for run.py; no transport or secondary process.

Image keypoints define a control model on an imposed camera-depth plane.
They are not measurements of scene depth. The model and the bounded YuMi
base-to-flange IK are kept in this application's existing process.
"""
from collections import deque
from dataclasses import dataclass, field
import math
from typing import Optional, Tuple

import numpy as np

from tien_ich.dong_hoc_yumi import ArmModel, IKError, JOINT_ORDER


@dataclass(frozen=True)
class XYControlConfig:
    camera_depth_m: float = 1.
    analysis_hz: float = 5.
    calibration_frames: int = 20
    min_confidence: float = .5
    planar_scale_m: float = .2
    median_window: int = 3
    ema_alpha: float = .35
    deadband_m: float = .002
    max_xy_speed_m_s: float = .03
    max_translation_m: float = .08
    max_joint_speed_deg_s: float = 20.
    max_joint_step_deg: float = 4.
    max_age_s: float = .5
    prediction_horizon_s: float = .2
    camera_focal_px: Optional[Tuple[float, float]] = None
    camera_principal_point_px: Optional[Tuple[float, float]] = None
    fixed_robot_z_m: Optional[float] = None

    def __post_init__(self):
        for name in ('camera_depth_m', 'analysis_hz', 'planar_scale_m', 'ema_alpha',
                     'max_xy_speed_m_s', 'max_translation_m', 'max_joint_speed_deg_s',
                     'max_joint_step_deg', 'max_age_s', 'prediction_horizon_s'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.analysis_hz > 60 or self.ema_alpha > 1 or self.camera_depth_m > 100:
            raise ValueError('Invalid analysis rate, filter alpha or imposed depth')
        if type(self.calibration_frames) is not int or not 1 <= self.calibration_frames <= 1000:
            raise ValueError('calibration_frames must be an integer in [1,1000]')
        if type(self.median_window) is not int or not 1 <= self.median_window <= 9 or self.median_window % 2 == 0:
            raise ValueError('median_window must be an odd integer in [1,9]')
        if isinstance(self.deadband_m, bool) or not isinstance(self.deadband_m, (float, int)) or not math.isfinite(self.deadband_m) or self.deadband_m < 0:
            raise ValueError('deadband_m must be finite and nonnegative')
        if isinstance(self.min_confidence, bool) or not isinstance(self.min_confidence, (float, int)) or not math.isfinite(self.min_confidence) or not 0 < self.min_confidence <= 1:
            raise ValueError('min_confidence must be in (0,1]')
        if self.fixed_robot_z_m is not None and (isinstance(self.fixed_robot_z_m, bool) or not isinstance(self.fixed_robot_z_m, (float, int)) or not math.isfinite(self.fixed_robot_z_m)):
            raise ValueError('fixed_robot_z_m must be null or a finite robot-base height')
        for name, positive in (('camera_focal_px', True), ('camera_principal_point_px', False)):
            pair = getattr(self, name)
            if pair is not None and (not isinstance(pair, (tuple, list)) or len(pair) != 2 or any(
                    isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value)
                    or (positive and value <= 0) for value in pair)):
                raise ValueError(f'{name} must be null or two finite numbers')
            if pair is not None:
                object.__setattr__(self, name, tuple(float(value) for value in pair))


@dataclass(frozen=True)
class JointRow:
    axis: int
    angle_deg: Optional[float] = None
    velocity_deg_s: Optional[float] = None
    predicted_deg: Optional[float] = None
    direction: str = 'UNKNOWN'


@dataclass(frozen=True)
class NativeXYArmState:
    side: str
    status: str
    reason: str
    calibration_frames: int
    required: int
    camera_position_m: Optional[tuple] = None
    robot_position_m: Optional[tuple] = None
    actual_robot_position_m: Optional[tuple] = None
    fixed_robot_z_m: Optional[float] = None
    joints_deg: Optional[tuple] = None
    rows: tuple = ()
    filter_state: str = 'HOLD'


@dataclass(frozen=True)
class NativeXYSnapshot:
    timestamp: Optional[float]
    sampled: bool
    status: str
    reason: str
    analysis_hz: float
    camera_depth_m: float
    arms: tuple
    model_pose: Optional[np.ndarray] = None
    model_hands: tuple = ()
    camera_intrinsics_provided: bool = False

    def arm(self, side):
        return next(arm for arm in self.arms if arm.side == str(side).upper())


@dataclass
class _Arm:
    model: ArmModel
    joints: np.ndarray
    seed_pose: np.ndarray
    fixed_z: float
    samples: list = field(default_factory=list)
    camera_samples: list = field(default_factory=list)
    neutral: Optional[np.ndarray] = None
    anchor: Optional[np.ndarray] = None
    camera_anchor: Optional[np.ndarray] = None
    target: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    history: deque = field(default_factory=deque)
    filtered: Optional[np.ndarray] = None
    last_valid: Optional[float] = None
    last_ready: Optional[float] = None
    velocity: Optional[np.ndarray] = None
    view: Optional[NativeXYArmState] = None


def _readonly(value):
    result = np.asarray(value, dtype=float).copy()
    result.flags.writeable = False
    return result


class NativeXYController:
    """Consume in-process COCO poses, preserving the camera's normal frame rate."""

    _COCO_TO_MP = {0: 0, 1: 2, 2: 5, 3: 7, 4: 8, 5: 11, 6: 12,
                   7: 13, 8: 14, 9: 15, 10: 16, 11: 23, 12: 24,
                   13: 25, 14: 26, 15: 27, 16: 28}

    def __init__(self, config=None):
        self.config = config or XYControlConfig()
        if not isinstance(self.config, XYControlConfig):
            raise ValueError('config must be XYControlConfig')
        self._arms = {}
        for side in ('LEFT', 'RIGHT'):
            model = ArmModel(side.lower())
            joints = np.asarray(model.default_seed_deg, dtype=float)
            seed = model.forward(joints)
            fixed_z = float(seed[2, 3] if self.config.fixed_robot_z_m is None else self.config.fixed_robot_z_m)
            target = seed[:3, 3].copy()
            target[2] = fixed_z
            if np.linalg.norm(target - seed[:3, 3]) > self.config.max_translation_m:
                raise ValueError(f'{side}: fixed robot Z exceeds the initial workspace')
            try:
                solution = model.solve(target, seed[:3, :3], joints, position_tolerance_m=1e-5)
            except IKError as exc:
                raise ValueError(f'{side}: fixed robot Z is unreachable: {exc}') from exc
            state = _Arm(model, np.asarray(solution.joints_deg), seed, fixed_z)
            state.view = self._view(state, side, 'WAITING', 'Chờ nhận diện hoạt động')
            self._arms[side] = state
        self._last_input = self._last_analysis = None
        self._track_ids = None
        self._image_size = None
        self._model_points = {}
        self._model_history = {}
        self._model_ema = {}
        self._hand_points = {}
        self._hand_history = {}
        self._hand_ema = {}
        self._model_origin = None
        self._model_pose = None
        self._model_hands = ()
        self._snapshot = self._make_snapshot(None, False)

    def _view(self, state, side, status, reason, **values):
        prior = state.view
        rows = tuple(JointRow(axis=row.axis, angle_deg=row.angle_deg) for row in prior.rows) if prior else tuple(JointRow(axis) for axis in range(1, 8))
        defaults = dict(side=side, status=status, reason=reason,
                        calibration_frames=len(state.samples) if state.neutral is None else self.config.calibration_frames,
                        required=self.config.calibration_frames,
                        camera_position_m=prior.camera_position_m if prior else None,
                        robot_position_m=prior.robot_position_m if prior else None,
                        actual_robot_position_m=prior.actual_robot_position_m if prior else None,
                        fixed_robot_z_m=state.fixed_z,
                        joints_deg=prior.joints_deg if prior else None, rows=rows)
        defaults.update(values)
        return NativeXYArmState(**defaults)

    def _clear_observation(self, state):
        state.samples.clear()
        state.camera_samples.clear()
        state.neutral = state.anchor = state.camera_anchor = None
        state.history.clear()
        state.filtered = state.last_valid = state.velocity = None

    def reset(self, reason='Hiệu chuẩn lại'):
        for side, state in self._arms.items():
            self._clear_observation(state)
            state.view = self._view(state, side, 'HOLD', reason)
        self._track_ids = None
        self._model_points.clear()
        self._model_history.clear()
        self._model_ema.clear()
        self._hand_points.clear()
        self._hand_history.clear()
        self._hand_ema.clear()
        self._model_origin = None
        self._snapshot = self._make_snapshot(self._last_input, False)
        return self._snapshot

    def _make_snapshot(self, timestamp, sampled, reason=''):
        arms = tuple(state.view for state in self._arms.values())
        status = ('READY' if any(arm.status == 'READY' for arm in arms)
                  else 'CALIBRATING' if any(arm.status == 'CALIBRATING' for arm in arms)
                  else 'STALE' if any(arm.status == 'STALE' for arm in arms)
                  else 'WAITING' if all(arm.status == 'WAITING' for arm in arms) else 'HOLD')
        return NativeXYSnapshot(timestamp, sampled, status, reason,
                                self.config.analysis_hz, self.config.camera_depth_m,
                                arms, self._model_pose, self._model_hands,
                                self.config.camera_focal_px is not None)

    def snapshot(self, timestamp=None):
        if timestamp is not None:
            timestamp = self._time(timestamp)
            for side, state in self._arms.items():
                if state.last_valid is not None and timestamp - state.last_valid > self.config.max_age_s:
                    self._clear_observation(state)
                    state.view = self._view(state, side, 'STALE', 'Keypoint đã cũ')
            self._snapshot = self._make_snapshot(timestamp, False)
        return self._snapshot

    @staticmethod
    def _time(timestamp):
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError('timestamp must be a finite monotonic time')
        return float(timestamp)

    def _project(self, pixels, size):
        width, height = size
        center = self.config.camera_principal_point_px or (width / 2., height / 2.)
        with np.errstate(over='ignore', invalid='ignore'):
            xy = np.asarray(pixels, dtype=float) - center
            if self.config.camera_focal_px is not None:
                xy *= self.config.camera_depth_m / np.asarray(self.config.camera_focal_px)
            else:
                xy *= self.config.planar_scale_m / height
        if not np.isfinite(xy).all() or np.max(np.abs(xy)) > 100:
            raise ValueError('Tọa độ camera không hợp lệ')
        return np.asarray([xy[0], xy[1], self.config.camera_depth_m])

    def _restore_filter(self, state):
        accepted = state.target - state.anchor[:3, 3]
        accepted[2] = 0.
        state.filtered = accepted.copy()
        state.history = deque([accepted.copy() for _ in range(self.config.median_window)], maxlen=self.config.median_window)

    def _arm_update(self, side, state, pose, size, timestamp):
        cfg = self.config
        wrist = 9 if side == 'LEFT' else 10
        if state.last_valid is not None and timestamp - state.last_valid > cfg.max_age_s:
            self._clear_observation(state)
        if min(pose[index, 2] for index in (5, 6, wrist)) < cfg.min_confidence:
            state.history.clear()
            state.filtered = state.velocity = None
            if state.neutral is None:
                state.samples.clear()
                state.camera_samples.clear()
            return self._view(state, side, 'HOLD', 'Hai vai hoặc cổ tay chưa đủ tin cậy')
        camera = self._project(pose[wrist, :2], size)
        origin = self._project((pose[5, :2] + pose[6, :2]) * .5, size)
        relative = (camera - origin) * [1., -1., 0.]
        state.last_valid = timestamp
        if state.neutral is None:
            state.samples.append(relative)
            state.camera_samples.append(camera)
            if len(state.samples) < cfg.calibration_frames:
                return self._view(state, side, 'CALIBRATING', 'Giữ tư thế trung lập')
            state.neutral = np.median(state.samples, axis=0)
            state.anchor = state.model.forward(state.joints)
            state.target = state.anchor[:3, 3].copy()
            state.target[2] = state.fixed_z
            state.rotation = state.anchor[:3, :3].copy()
            state.camera_anchor = np.median(state.camera_samples, axis=0)
            self._restore_filter(state)
        if state.filtered is None:
            self._restore_filter(state)
        displacement = relative - state.neutral
        if np.linalg.norm(displacement[:2]) > cfg.max_translation_m:
            self._restore_filter(state)
            state.velocity = None
            return self._view(state, side, 'HOLD', 'Mục tiêu vượt vùng chuyển động')
        state.history.append(displacement)
        median = np.median(np.asarray(state.history), axis=0)
        state.filtered += cfg.ema_alpha * (median - state.filtered)
        requested = state.anchor[:3, 3] + state.filtered
        requested[2] = state.fixed_z
        if np.linalg.norm(requested - state.seed_pose[:3, 3]) > cfg.max_translation_m:
            self._restore_filter(state)
            state.velocity = None
            return self._view(state, side, 'HOLD', 'Mục tiêu vượt vùng gốc của robot')
        delta = requested - state.target
        delta[2] = 0.
        dt = min(1. / cfg.analysis_hz, timestamp - state.last_ready) if state.last_ready is not None else 1. / cfg.analysis_hz
        dt = max(0., dt)
        distance = np.linalg.norm(delta)
        filter_state = 'TRACKING'
        if distance <= cfg.deadband_m:
            delta[:] = 0.
            filter_state = 'DEADBAND'
        elif distance > cfg.max_xy_speed_m_s * dt:
            delta *= cfg.max_xy_speed_m_s * dt / distance
            filter_state = 'SPEED_LIMITED'
        allowance = min(cfg.max_joint_step_deg, cfg.max_joint_speed_deg_s * dt)
        solution = None
        for attempt in range(7):
            position = state.target + delta * .5 ** attempt
            position[2] = state.fixed_z
            try:
                candidate = state.model.solve(position, state.rotation, state.joints, position_tolerance_m=1e-5)
            except IKError:
                continue
            joints = np.asarray(candidate.joints_deg)
            actual = state.model.forward(joints)
            if abs(actual[2, 3] - state.fixed_z) <= 1e-5 and np.max(np.abs(joints - state.joints)) <= allowance + 1e-6:
                solution = candidate
                if attempt:
                    filter_state = 'JOINT_LIMITED'
                break
        if solution is None:
            self._restore_filter(state)
            state.velocity = None
            return self._view(state, side, 'HOLD', 'IK chưa đạt giới hạn bước và Z cố định')
        velocity = None
        if state.view.status == 'READY' and state.last_ready is not None:
            elapsed = timestamp - state.last_ready
            if 1e-6 < elapsed <= cfg.max_age_s:
                raw = (joints - state.joints) / elapsed
                velocity = np.zeros(7) if filter_state == 'DEADBAND' else raw if state.velocity is None else .3 * raw + .7 * state.velocity
        state.joints = joints
        state.target = position
        state.last_ready = timestamp
        state.velocity = velocity
        camera = state.camera_anchor + (position - state.anchor[:3, 3]) * [1., -1., 0.]
        camera[2] = cfg.camera_depth_m
        rows = []
        for axis in range(1, 8):
            index = JOINT_ORDER.index(axis)
            v = None if velocity is None else float(velocity[index])
            prediction = None if v is None else float(np.clip(joints[index] + v * cfg.prediction_horizon_s, *state.model.limits_deg[index]))
            direction = 'UNKNOWN' if v is None else 'UP' if v > .001 else 'DOWN' if v < -.001 else 'STILL'
            rows.append(JointRow(axis, float(joints[index]), v, prediction, direction))
        return self._view(state, side, 'READY', 'X/Y đã lọc; Z camera đặt trước',
                          camera_position_m=tuple(float(v) for v in camera),
                          robot_position_m=tuple(float(v) for v in position),
                          actual_robot_position_m=tuple(float(v) for v in actual[:3, 3]),
                          joints_deg=tuple(float(v) for v in joints), rows=tuple(rows), filter_state=filter_state)

    def _model_filter(self, key, point, previous, histories, ema):
        cfg = self.config
        prior = previous.get(key)
        if prior is None:
            histories[key] = deque([point.copy() for _ in range(cfg.median_window)], maxlen=cfg.median_window)
            previous[key] = point.copy()
            ema[key] = point.copy()
            return point.copy()
        # A rejected spike must not contaminate the later EMA/model pose.
        if np.linalg.norm(point[:2] - prior[:2]) > cfg.max_translation_m:
            histories[key] = deque([prior.copy() for _ in range(cfg.median_window)], maxlen=cfg.median_window)
            ema[key] = prior.copy()
            return prior.copy()
        histories[key].append(point.copy())
        # Keep the hidden EMA independently of the accepted position. Small
        # deliberate motions can then accumulate past the deadband instead of
        # being discarded forever at each 200-ms sample.
        target = ema[key] + cfg.ema_alpha * (np.median(np.asarray(histories[key]), axis=0) - ema[key])
        ema[key] = target.copy()
        delta = target - prior
        delta[2] = 0.
        length = np.linalg.norm(delta)
        if length <= cfg.deadband_m:
            delta[:] = 0.
        elif length > cfg.max_xy_speed_m_s / cfg.analysis_hz:
            delta *= cfg.max_xy_speed_m_s / cfg.analysis_hz / length
        result = prior + delta
        result[2] = cfg.camera_depth_m
        previous[key] = result.copy()
        return result

    def _update_model(self, pose, size, hands):
        cfg = self.config
        if min(pose[5, 2], pose[6, 2]) < cfg.min_confidence:
            return
        origin = self._project((pose[5, :2] + pose[6, :2]) * .5, size)
        if self._model_origin is None:
            self._model_origin = origin.copy()
        model = np.zeros((33, 4), dtype=float)
        model[:, 2] = cfg.camera_depth_m
        for coco, mp in self._COCO_TO_MP.items():
            if pose[coco, 2] < cfg.min_confidence:
                continue
            point = self._project(pose[coco, :2], size) - origin + self._model_origin
            point[2] = cfg.camera_depth_m
            point = self._model_filter(mp, point, self._model_points, self._model_history, self._model_ema)
            model[mp] = (*point, float(pose[coco, 2]))
        # The displayed wrist is the accepted robot-control target, never a
        # separate noisy preview that disagrees with the angle table.
        for side, mp in (('LEFT', 15), ('RIGHT', 16)):
            view = self._arms[side].view
            if view.status == 'READY' and view.camera_position_m is not None:
                model[mp] = (*view.camera_position_m, float(pose[9 if side == 'LEFT' else 10, 2]))
            elif view.status != 'CALIBRATING':
                model[mp, 3] = 0.
        self._model_pose = _readonly(model)
        output_hands = []
        seen = set()
        for raw, is_left in hands:
            points = np.asarray(raw, dtype=float)
            side = 'LEFT' if is_left else 'RIGHT'
            wrist = model[15 if is_left else 16]
            if points.shape != (21, 3) or not np.isfinite(points).all() or wrist[3] < cfg.min_confidence or side in seen:
                continue
            seen.add(side)
            projected = np.array([self._project(point[:2], size) for point in points])
            projected += wrist[:3] - projected[0]
            filtered = np.array([self._model_filter((side, index), point, self._hand_points, self._hand_history, self._hand_ema)
                                 for index, point in enumerate(projected)])
            # Translate the whole filtered hand so its wrist remains attached.
            filtered += wrist[:3] - filtered[0]
            filtered[:, 2] = cfg.camera_depth_m
            output_hands.append((_readonly(filtered), bool(is_left)))
        self._model_hands = tuple(output_hands)

    def update(self, poses, bboxes, image_size, timestamp, *, active=True,
               fresh=True, track_ids=(), joint_misses=None, wrist_reports=None, hands=()):
        timestamp = self._time(timestamp)
        if self._last_input is not None and timestamp <= self._last_input:
            self.reset('Thời gian khung hình không tăng')
            self._last_input = timestamp
            return self._make_snapshot(timestamp, False)
        self._last_input = timestamp
        if not active:
            self.reset('Nguồn đang chờ hoặc tạm dừng')
            return self._make_snapshot(timestamp, False)
        if len(poses) != 1 or len(bboxes) != 1:
            self.reset('Cần đúng một người trong khung hình')
            return self._make_snapshot(timestamp, False)
        token = tuple(track_ids)
        if self._track_ids is not None and token != self._track_ids:
            self.reset('Người theo dõi thay đổi; hiệu chuẩn lại')
            self._track_ids = token
            return self._make_snapshot(timestamp, False)
        self._track_ids = token
        if not fresh:
            return self.snapshot(timestamp)
        if self._last_analysis is not None and timestamp - self._last_analysis < 1. / self.config.analysis_hz - 1e-9:
            return self.snapshot(timestamp)
        self._last_analysis = timestamp
        try:
            if (not isinstance(image_size, (tuple, list)) or len(image_size) != 2
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v <= 0 for v in image_size)):
                raise ValueError('Kích thước ảnh không hợp lệ')
            if self._image_size is not None and tuple(image_size) != self._image_size:
                self.reset('Kích thước ảnh thay đổi; hiệu chuẩn lại')
            self._image_size = tuple(image_size)
            pose = np.asarray(poses[0], dtype=float).copy()
            if pose.shape != (17, 3) or not np.isfinite(pose).all() or np.any(pose[:, 2] < 0) or np.any(pose[:, 2] > 1):
                raise ValueError('Keypoint không hợp lệ')
            misses = joint_misses or {}
            for index in (5, 6, 7, 8, 9, 10):
                if misses.get(index, 0) > 0:
                    pose[index, 2] = 0.
            for side, wrist in (('LEFT', 9), ('RIGHT', 10)):
                report = (wrist_reports or {}).get(side)
                quality = getattr(report, 'state', None)
                quality = getattr(quality, 'value', quality)
                if report is not None and quality != 'TRACKED':
                    pose[wrist, 2] = 0.
            for side, state in self._arms.items():
                state.view = self._arm_update(side, state, pose, self._image_size, timestamp)
            self._update_model(pose, self._image_size, hands)
        except (ValueError, TypeError, IndexError, OverflowError) as exc:
            self.reset(str(exc))
        self._snapshot = self._make_snapshot(timestamp, True)
        return self._snapshot
