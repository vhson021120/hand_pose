"""Ổn định cổ tay MediaPipe: One Euro theo timestamp, máy trạng thái, kiểm tra ngoại lai.

Bộ lọc chỉ làm việc trong hệ tọa độ Pose world (vector cổ tay so với trung điểm hai
vai). Hand Landmarker chỉ được dùng trong hệ pixel ảnh để xác nhận cổ tay; tọa độ
Hand world không bao giờ được trộn vào Pose world.
"""

import csv
import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LEFT = "LEFT"
RIGHT = "RIGHT"
# (vai, khuỷu, cổ tay) theo chỉ số Pose 33 điểm và chỉ số cổ tay COCO-17.
SIDE_INDEXES = {
    LEFT: (11, 13, 15, 9),
    RIGHT: (12, 14, 16, 10),
}


class TrackState(str, Enum):
    TRACKED = "TRACKED"
    UNCERTAIN = "UNCERTAIN"
    LOST = "LOST"
    REACQUIRING = "REACQUIRING"


@dataclass
class StabilizerConfig:
    enabled: bool = True
    min_cutoff_hz: float = 1.5
    beta: float = 6.0
    derivative_cutoff_hz: float = 1.0
    min_visibility: float = 0.5
    hand_confirmed_min_visibility: float = 0.3
    max_arm_length_m: float = 1.5
    max_speed_m_s: float = 6.0
    bone_tolerance: float = 0.35
    uncertain_max_frames: int = 6
    uncertain_max_seconds: float = 0.4
    reacquire_frames: int = 4
    reacquire_tolerance_m: float = 0.08
    max_gap_seconds: float = 0.5
    hand_refine: bool = False
    hand_match_px_fraction: float = 0.08

    def validate(self) -> None:
        if self.min_cutoff_hz <= 0 or self.derivative_cutoff_hz <= 0 or self.beta < 0:
            raise ValueError("One Euro parameters must be positive (beta >= 0)")
        if self.reacquire_frames < 1 or self.uncertain_max_frames < 0:
            raise ValueError("frame counters are out of range")
        if self.max_speed_m_s <= 0 or self.max_arm_length_m <= 0:
            raise ValueError("speed and arm length limits must be positive")


class OneEuroFilter3D:
    """One Euro filter cho vector 3D; hệ số lọc tính từ dt thực của timestamp."""

    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.reset()

    def reset(self) -> None:
        self._x: Optional[np.ndarray] = None
        self._dx = np.zeros(3)
        self._t: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, timestamp: float, value: np.ndarray) -> np.ndarray:
        value = np.asarray(value, dtype=np.float64)
        if self._x is None or self._t is None or timestamp <= self._t:
            if self._x is not None and self._t is not None and timestamp <= self._t:
                return self._x.copy()
            self._x, self._t = value.copy(), timestamp
            self._dx = np.zeros(3)
            return value.copy()
        dt = timestamp - self._t
        raw_velocity = (value - self._x) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        self._dx = a_d * raw_velocity + (1.0 - a_d) * self._dx
        cutoff = self.min_cutoff + self.beta * float(np.linalg.norm(self._dx))
        a = self._alpha(cutoff, dt)
        self._x = a * value + (1.0 - a) * self._x
        self._t = timestamp
        return self._x.copy()


@dataclass
class SideReport:
    side: str
    state: TrackState
    reason: str
    raw_relative: Optional[np.ndarray]
    output_relative: Optional[np.ndarray]
    hand_confirmed: bool = False

    @property
    def usable(self) -> bool:
        return self.state == TrackState.TRACKED


class _SideTracker:
    def __init__(self, side: str, config: StabilizerConfig):
        self.side = side
        self.config = config
        self.filter = OneEuroFilter3D(
            config.min_cutoff_hz, config.beta, config.derivative_cutoff_hz
        )
        self.reset()

    def reset(self) -> None:
        self.state = TrackState.LOST
        self.filter.reset()
        self.last_accepted: Optional[np.ndarray] = None
        self.last_accepted_t: Optional[float] = None
        self.last_output: Optional[np.ndarray] = None
        self.forearm_ref: Optional[float] = None
        self.upper_ref: Optional[float] = None
        self.misses = 0
        self.uncertain_since: Optional[float] = None
        self.candidate: Optional[np.ndarray] = None
        self.candidate_count = 0

    def _bone_ok(self, upper: Optional[float], fore: Optional[float]) -> bool:
        tol = self.config.bone_tolerance
        for value, ref in ((upper, self.upper_ref), (fore, self.forearm_ref)):
            if value is None or ref is None:
                continue
            if abs(value - ref) > tol * ref:
                return False
        return True

    def _update_bones(self, upper: Optional[float], fore: Optional[float]) -> None:
        if upper is not None:
            self.upper_ref = upper if self.upper_ref is None else 0.9 * self.upper_ref + 0.1 * upper
        if fore is not None:
            self.forearm_ref = fore if self.forearm_ref is None else 0.9 * self.forearm_ref + 0.1 * fore

    def validate(
        self,
        timestamp: float,
        rel: Optional[np.ndarray],
        wrist_vis: float,
        shoulder_vis: float,
        upper: Optional[float],
        fore: Optional[float],
        in_frame: bool,
        hand_confirmed: bool,
    ) -> str:
        """Trả về chuỗi rỗng nếu hợp lệ, ngược lại là lý do loại mẫu."""
        cfg = self.config
        if rel is None or not np.all(np.isfinite(rel)):
            return "NON FINITE"
        vis_floor = cfg.min_visibility
        if hand_confirmed:
            vis_floor = min(vis_floor, cfg.hand_confirmed_min_visibility)
        if wrist_vis < vis_floor:
            return "LOW WRIST CONFIDENCE"
        if shoulder_vis < cfg.min_visibility:
            return "LOW SHOULDER CONFIDENCE"
        if not in_frame:
            return "WRIST OUTSIDE FRAME"
        if float(np.linalg.norm(rel)) > cfg.max_arm_length_m:
            return "ARM TOO LONG"
        if not self._bone_ok(upper, fore):
            return "BONE LENGTH OUTLIER"
        if (
            self.state in (TrackState.TRACKED, TrackState.UNCERTAIN)
            and self.last_accepted is not None
            and self.last_accepted_t is not None
        ):
            dt = timestamp - self.last_accepted_t
            if dt > 0:
                speed = float(np.linalg.norm(rel - self.last_accepted)) / dt
                if speed > cfg.max_speed_m_s:
                    return "SPEED OUTLIER"
        return ""

    def step(
        self, timestamp: float, rel: Optional[np.ndarray], reason: str,
        upper: Optional[float], fore: Optional[float],
    ) -> Tuple[TrackState, str, Optional[np.ndarray]]:
        cfg = self.config
        if (
            self.last_accepted_t is not None
            and timestamp - self.last_accepted_t > cfg.max_gap_seconds
            and self.state in (TrackState.TRACKED, TrackState.UNCERTAIN)
        ):
            self._go_lost()
        if reason:
            return self._on_invalid(timestamp, reason)
        assert rel is not None
        if self.state in (TrackState.TRACKED, TrackState.UNCERTAIN):
            self._accept(timestamp, rel, upper, fore)
            self.misses = 0
            self.uncertain_since = None
            self.state = TrackState.TRACKED
            self.last_output = self.filter(timestamp, rel)
            return self.state, "", self.last_output
        # LOST hoặc REACQUIRING: cần liên tiếp các mẫu nhất quán.
        if (
            self.state == TrackState.REACQUIRING
            and self.candidate is not None
            and float(np.linalg.norm(rel - self.candidate)) <= cfg.reacquire_tolerance_m
        ):
            self.candidate_count += 1
        else:
            self.candidate_count = 1
        self.candidate = rel.copy()
        self.state = TrackState.REACQUIRING
        if self.candidate_count >= cfg.reacquire_frames:
            self.filter.reset()
            self.upper_ref = self.forearm_ref = None
            self._accept(timestamp, rel, upper, fore)
            self.state = TrackState.TRACKED
            self.last_output = self.filter(timestamp, rel)
            return self.state, "", self.last_output
        return self.state, "REACQUIRING", None

    def _accept(self, timestamp, rel, upper, fore) -> None:
        self.last_accepted = rel.copy()
        self.last_accepted_t = timestamp
        self._update_bones(upper, fore)

    def _go_lost(self) -> None:
        self.state = TrackState.LOST
        self.filter.reset()
        self.last_accepted = None
        self.last_accepted_t = None
        self.last_output = None
        self.candidate = None
        self.candidate_count = 0
        self.uncertain_since = None

    def _on_invalid(self, timestamp: float, reason: str):
        cfg = self.config
        if self.state in (TrackState.TRACKED, TrackState.UNCERTAIN):
            if self.state == TrackState.TRACKED:
                self.uncertain_since = timestamp
                self.misses = 0
            self.misses += 1
            expired = (
                self.misses > cfg.uncertain_max_frames
                or (self.uncertain_since is not None
                    and timestamp - self.uncertain_since > cfg.uncertain_max_seconds)
            )
            if not expired:
                self.state = TrackState.UNCERTAIN
                return self.state, reason, self.last_output
            self._go_lost()
            return self.state, reason, None
        if self.state == TrackState.REACQUIRING:
            self.candidate = None
            self.candidate_count = 0
            self.state = TrackState.LOST
        return self.state, reason, None


class WristStabilizer:
    """Lọc cổ tay Pose world; trả bản sao landmarks đã được kiểm soát."""

    def __init__(self, config: Optional[StabilizerConfig] = None):
        self.config = config or StabilizerConfig()
        self.config.validate()
        self._trackers = {s: _SideTracker(s, self.config) for s in SIDE_INDEXES}

    def reset(self) -> None:
        for tracker in self._trackers.values():
            tracker.reset()

    def state_of(self, side: str) -> TrackState:
        return self._trackers[side].state

    def process(
        self,
        pose_world: Sequence,
        timestamp: float,
        pose_2d: Optional[np.ndarray] = None,
        image_shape: Optional[Tuple[int, ...]] = None,
        hand_wrists_px: Optional[Sequence[Tuple[float, float]]] = None,
    ) -> Tuple[np.ndarray, Dict[str, SideReport]]:
        """pose_world: 33x4 (x, y, z, visibility) của đúng một người."""
        landmarks = np.array(pose_world, dtype=np.float64, copy=True)
        if not self.config.enabled:
            return landmarks, {}
        reports: Dict[str, SideReport] = {}
        shoulders = landmarks[[11, 12], :3]
        shoulder_ok = bool(np.all(np.isfinite(shoulders)))
        mid = shoulders.mean(axis=0) if shoulder_ok else None
        for side, (si, ei, wi, ci) in SIDE_INDEXES.items():
            tracker = self._trackers[side]
            wrist = landmarks[wi]
            rel = (wrist[:3] - mid) if mid is not None else None
            elbow = landmarks[ei]
            upper = fore = None
            if (
                shoulder_ok
                and elbow[3] >= self.config.min_visibility
                and np.all(np.isfinite(elbow[:3]))
                and np.all(np.isfinite(wrist[:3]))
            ):
                upper = float(np.linalg.norm(elbow[:3] - landmarks[si, :3]))
                fore = float(np.linalg.norm(wrist[:3] - elbow[:3]))
            in_frame, wrist_px = True, None
            if pose_2d is not None and image_shape is not None and len(pose_2d) > ci:
                px = np.asarray(pose_2d[ci], dtype=np.float64)
                if px[2] > 0:
                    wrist_px = px[:2]
                    h, w = image_shape[:2]
                    in_frame = -0.05 * w <= px[0] <= 1.05 * w and -0.05 * h <= px[1] <= 1.05 * h
            confirmed = self._hand_confirms(wrist_px, hand_wrists_px, image_shape)
            reason = tracker.validate(
                timestamp, rel, float(wrist[3]), float(min(landmarks[11, 3], landmarks[12, 3])),
                upper, fore, in_frame, confirmed,
            )
            state, reason, out = tracker.step(timestamp, rel, reason, upper, fore)
            if state == TrackState.TRACKED and out is not None and mid is not None:
                landmarks[wi, :3] = mid + out
            else:
                landmarks[wi, 3] = 0.0
            reports[side] = SideReport(side, state, reason, rel, out, confirmed)
        return landmarks, reports

    def _hand_confirms(self, wrist_px, hand_wrists_px, image_shape) -> bool:
        if not self.config.hand_refine or wrist_px is None or not hand_wrists_px or image_shape is None:
            return False
        limit = self.config.hand_match_px_fraction * float(max(image_shape[:2]))
        return any(
            math.hypot(hx - wrist_px[0], hy - wrist_px[1]) <= limit
            for hx, hy in hand_wrists_px
        )


class StabilityBenchmark:
    """Ghi nhận jitter, dropout, độ trễ; xuất CSV từng frame và tóm tắt."""

    FIELDS = (
        "timestamp", "side", "state", "reason", "raw_x_mm", "raw_y_mm", "raw_z_mm",
        "out_x_mm", "out_y_mm", "out_z_mm", "process_ms", "detect_ms",
    )

    def __init__(self, csv_path: Optional[str] = None):
        self.rows: List[dict] = []
        self._file = None
        self._writer = None
        if csv_path:
            self._file = open(csv_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
            self._writer.writeheader()

    def record(self, timestamp: float, report: SideReport, process_ms: float,
               detect_ms: Optional[float] = None) -> None:
        def mm(v):
            return [None] * 3 if v is None else [round(float(c) * 1000.0, 3) for c in v]
        raw, out = mm(report.raw_relative), mm(report.output_relative if report.usable else None)
        row = {
            "timestamp": round(timestamp, 6), "side": report.side,
            "state": report.state.value, "reason": report.reason,
            "raw_x_mm": raw[0], "raw_y_mm": raw[1], "raw_z_mm": raw[2],
            "out_x_mm": out[0], "out_y_mm": out[1], "out_z_mm": out[2],
            "process_ms": round(process_ms, 4),
            "detect_ms": None if detect_ms is None else round(detect_ms, 4),
        }
        self.rows.append(row)
        if self._writer:
            self._writer.writerow(row)

    def close(self) -> None:
        if self._file:
            self._file.close()
            self._file = None

    def summary(self, side: Optional[str] = None) -> dict:
        return summarize_rows(self.rows, side)


def _rms_second_difference(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    second = points[2:] - 2.0 * points[1:-1] + points[:-2]
    return float(np.sqrt(np.mean(np.sum(second ** 2, axis=1))))


def summarize_rows(rows: Sequence[dict], side: Optional[str] = None) -> dict:
    rows = [r for r in rows if side is None or r["side"] == side]
    total = len(rows)
    if total == 0:
        return {"frames": 0}
    tracked = [r for r in rows if r["out_x_mm"] is not None]

    def series(prefix):
        return np.array(
            [[r[f"{prefix}_x_mm"], r[f"{prefix}_y_mm"], r[f"{prefix}_z_mm"]] for r in tracked],
            dtype=np.float64,
        ).reshape(-1, 3)

    proc = np.array([r["process_ms"] for r in rows], dtype=np.float64)
    return {
        "frames": total,
        "dropout_rate": 1.0 - len(tracked) / total,
        "jitter_rms_mm": _rms_second_difference(series("out")),
        "raw_jitter_rms_mm": _rms_second_difference(
            np.array([[r["raw_x_mm"], r["raw_y_mm"], r["raw_z_mm"]] for r in rows
                      if r["raw_x_mm"] is not None], dtype=np.float64).reshape(-1, 3)
        ),
        "process_ms_mean": float(proc.mean()),
        "process_ms_p95": float(np.percentile(proc, 95)),
    }
