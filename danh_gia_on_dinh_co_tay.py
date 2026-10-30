"""Benchmark tổng hợp trước/sau cho ổn định cổ tay (không cần webcam).

Chuỗi mô phỏng: cổ tay phải xoay quanh khuỷu (chiều dài cẳng tay cố định) với nhiễu
Gaussian, đoạn thiếu sáng (nhiễu lớn, visibility dao động), nhảy ngoại lai ngắn và
một đoạn mất điểm. "Trước" = RelativeWristPreview(EMA 0.35) như cũ; "Sau" = WristStabilizer
+ RelativeWristPreview(smoothing=1.0).

Chạy: python danh_gia_on_dinh_co_tay.py [--seed 0] [--csv đường_dẫn]
"""

import argparse
import json

import numpy as np

from tien_ich.on_dinh_co_tay import StabilizerConfig, WristStabilizer
from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristPreview

FPS = 30.0
FRAMES = 900
FOREARM_M = 0.25
ELBOW = np.array([0.30, 0.20, 0.0])


def build_sequence(seed: int):
    rng = np.random.default_rng(seed)
    t = np.arange(FRAMES) / FPS + rng.uniform(-0.003, 0.003, FRAMES).cumsum() * 0.0
    t = t + rng.uniform(-0.003, 0.003, FRAMES)
    t = np.maximum.accumulate(t + np.arange(FRAMES) * 1e-6)
    theta = 0.6 * np.sin(2 * np.pi * 0.5 * t)
    truth = ELBOW + FOREARM_M * np.stack(
        [np.sin(theta), -np.cos(theta), np.zeros_like(theta)], axis=1
    )
    sigma = np.full(FRAMES, 0.005)
    vis = np.full(FRAMES, 0.95)
    low_light = slice(300, 600)
    sigma[low_light] = 0.015
    vis[low_light] = np.where(rng.random(300) < 0.3, 0.45, 0.8)
    measured = truth + rng.normal(0, 1, (FRAMES, 3)) * sigma[:, None]
    for start in (150, 450, 700):  # ngoại lai ngắn
        measured[start:start + 2, 2] += 0.35
    vis[800:810] = 0.0  # mất điểm
    return t, truth, measured, vis


def make_pose(wrist, vis):
    pose = np.zeros((33, 4))
    pose[:, 3] = 0.95
    pose[11] = (-0.20, 0.0, 0.0, 0.95)
    pose[12] = (0.20, 0.0, 0.0, 0.95)
    pose[14] = (*ELBOW, 0.95)
    pose[16] = (*wrist, vis)
    return pose


def metrics(t, truth, delivered):
    ok = ~np.isnan(delivered[:, 0])
    truth_delta = (truth - truth[:30].mean(axis=0)) * 100.0
    err = np.linalg.norm(delivered[ok] - truth_delta[ok], axis=1)
    pts = delivered[ok]
    second = pts[2:] - 2 * pts[1:-1] + pts[:-2]
    best_lag, best_rmse = 0, float("inf")
    for lag in range(0, 9):
        a = delivered[lag:][ok[lag:]]
        b = truth_delta[: FRAMES - lag][ok[lag:]]
        rmse = float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))
        if rmse < best_rmse:
            best_lag, best_rmse = lag, rmse
    return {
        "delivered_frames": int(ok.sum()),
        "dropout_pct": round(100.0 * (1 - ok.mean()), 2),
        "rmse_cm": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "jitter_rms_cm": round(float(np.sqrt(np.mean(np.sum(second ** 2, axis=1)))), 3),
        "outliers_over_5cm": int((err > 5.0).sum()),
        "max_error_cm": round(float(err.max()), 2),
        "lag_ms": round(best_lag / FPS * 1000.0, 1),
    }


def run_pipeline(t, measured, vis, stabilize: bool):
    preview = RelativeWristPreview(
        calibration_frames=30, smoothing=1.0 if stabilize else 0.35
    )
    stabilizer = WristStabilizer(StabilizerConfig(enabled=stabilize))
    out = np.full((FRAMES, 3), np.nan)
    for k in range(FRAMES):
        pose = make_pose(measured[k], vis[k])
        if stabilize:
            pose, _ = stabilizer.process(pose, float(t[k]))
        state = [s for s in preview.update([pose]) if s.side == "RIGHT"][0]
        if state.status == "MODEL RELATIVE" and state.delta_cm is not None:
            out[k] = state.delta_cm
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str)
    args = parser.parse_args()
    t, truth, measured, vis = build_sequence(args.seed)
    report = {
        "before_legacy_ema": metrics(t, truth, run_pipeline(t, measured, vis, False)),
        "after_stabilizer": metrics(t, truth, run_pipeline(t, measured, vis, True)),
    }
    print(json.dumps(report, indent=2))
    if args.csv:
        with open(args.csv, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)


if __name__ == "__main__":
    main()
