import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from tien_ich.nhan_dien_ban_tay import MediaPipeHandDetector
from tien_ich.nhan_dien_co_tay import MediaPipeWristDetector
from tien_ich.theo_doi_co_tay_2d import WristXYTracker


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
WRIST_SPECS = (("LEFT", 4, (70, 220, 110)), ("RIGHT", 5, (220, 90, 220)))


def draw_webcam_overlay(frame, wrists, hands):
    output = frame.copy()
    wrist_colors = {side: color for side, _, color in WRIST_SPECS}
    for side, (x, y, _) in wrists.items():
        color = wrist_colors[side]
        point = (int(round(x)), int(round(y)))
        cv2.circle(output, point, 9, color, 2, cv2.LINE_AA)
        cv2.putText(output, f"{side} WRIST", (point[0] + 10, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for hand in hands:
        landmarks = np.asarray(hand.landmarks)
        if landmarks.shape != (21, 3) or not np.isfinite(landmarks[:, :2]).all():
            continue
        color = (70, 220, 110) if hand.is_left else (220, 90, 220)
        for first, second in HAND_CONNECTIONS:
            start = tuple(np.rint(landmarks[first, :2]).astype(int))
            end = tuple(np.rint(landmarks[second, :2]).astype(int))
            cv2.line(output, start, end, color, 2, cv2.LINE_AA)
        for x, y in landmarks[:, :2]:
            cv2.circle(output, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
    return output


def compute_workspace_roi(image_size, work_mm, fill):
    width, height = image_size
    scale = min(width * fill / work_mm[0], height * fill / work_mm[1])
    roi_w, roi_h = work_mm[0] * scale, work_mm[1] * scale
    left, top = (width - roi_w) / 2.0, (height - roi_h) / 2.0
    return left, top, left + roi_w, top + roi_h


def draw_workspace_rect(frame, roi):
    p1 = (int(round(roi[0])), int(round(roi[1])))
    p2 = (int(round(roi[2])), int(round(roi[3])))
    cv2.rectangle(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "WORKSPACE", (p1[0] + 6, p1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)


def draw_wrist_xy_dashboard(wrists, roi, work_mm, pose_count):
    canvas = np.full((720, 960, 3), (20, 27, 35), dtype=np.uint8)
    max_width, max_height = 776, 484
    fit = min(max_width / work_mm[0], max_height / work_mm[1])
    plot_width, plot_height = int(work_mm[0] * fit), int(work_mm[1] * fit)
    plot_left = 92 + (max_width - plot_width) // 2
    plot_top = 116 + (max_height - plot_height) // 2
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height
    center_x = plot_left + plot_width // 2
    center_y = plot_top + plot_height // 2

    if pose_count == 0:
        status = "NO POSE DETECTED"
        status_color = (0, 185, 255)
    else:
        status = "TRACKING"
        status_color = (100, 235, 150)
    cv2.putText(canvas, "WRIST XY | MILLIMETERS", (28, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, status, (28, 77), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, status_color, 1, cv2.LINE_AA)

    for division in range(5):
        x = plot_left + division * plot_width // 4
        y = plot_top + division * plot_height // 4
        color = (80, 105, 120) if division == 2 else (44, 58, 70)
        thickness = 2 if division == 2 else 1
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), color, thickness)
        cv2.line(canvas, (plot_left, y), (plot_right, y), color, thickness)
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (80, 96, 110), 1)
    cv2.circle(canvas, (center_x, center_y), 4, (235, 240, 245), -1, cv2.LINE_AA)
    cv2.putText(canvas, "O", (center_x + 8, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "X+", (plot_right - 28, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (145, 190, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y+", (center_x + 8, plot_top + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (145, 220, 170), 1, cv2.LINE_AA)

    value_lines = []
    for side, _, color in WRIST_SPECS:
        wrist = wrists.get(side)
        if wrist is None:
            value_lines.append(f"{side}: NOT DETECTED")
            continue
        x, y, _ = wrist
        u = float(np.clip((x - roi[0]) / (roi[2] - roi[0]), 0, 1))
        v = float(np.clip((y - roi[1]) / (roi[3] - roi[1]), 0, 1))
        x_mm = (u - 0.5) * work_mm[0]
        y_mm = (0.5 - v) * work_mm[1]
        inside = roi[0] <= x <= roi[2] and roi[1] <= y <= roi[3]
        dot_x = plot_left + int(u * plot_width)
        dot_y = plot_top + int(v * plot_height)
        cv2.circle(canvas, (dot_x, dot_y), 9, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (dot_x, dot_y), 14, color, 2, cv2.LINE_AA)
        suffix = "" if inside else "  (OUT OF WORKSPACE)"
        value_lines.append(f"{side}: X {x_mm:+.1f} mm   Y {y_mm:+.1f} mm{suffix}")

    for index, (text, color) in enumerate(zip(value_lines, ((70, 220, 110), (220, 90, 220)))):
        cv2.putText(canvas, text, (28, 642 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    x_limit = work_mm[0] / 2.0
    y_limit = work_mm[1] / 2.0
    cv2.putText(canvas, f"Workspace {work_mm[0]:g}x{work_mm[1]:g} mm | "
                       f"X: {-x_limit:+.1f}..{x_limit:+.1f} mm | "
                       f"Y: {-y_limit:+.1f}..{y_limit:+.1f} mm",
                (28, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 181, 194), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Origin: workspace center | fixed size and aspect ratio | "
                       "display scale, not physical calibration",
                (28, 698), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 181, 194), 1, cv2.LINE_AA)
    return canvas


def open_camera(args):
    source = int(args.pipe) if args.pipe.isdecimal() else args.pipe
    camera = cv2.VideoCapture(source)
    if not camera.isOpened() and os.name == "nt":
        camera.release()
        camera = cv2.VideoCapture(source, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Unable to open webcam source {args.pipe}")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if args.camera_fourcc:
        camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.camera_fourcc))
    if args.camera_fps > 0:
        camera.set(cv2.CAP_PROP_FPS, args.camera_fps)
    print(
        f"Camera capture: {int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
        f"@{camera.get(cv2.CAP_PROP_FPS):.0f}fps"
    )
    return camera


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Track MediaPipe wrists and hand landmarks on a centered 2D XY plane."
    )
    parser.add_argument("--pipe", default="0", help="webcam index or video source")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=0)
    parser.add_argument("--camera-fourcc", default="")
    parser.add_argument(
        "--pose-model", default=str(ROOT / "mo_hinh" / "pose_landmarker_lite.task")
    )
    parser.add_argument(
        "--hand-model", default=str(ROOT / "mo_hinh" / "hand_landmarker.task")
    )
    parser.add_argument("--work-width-mm", type=float, default=400.0)
    parser.add_argument("--work-height-mm", type=float, default=300.0)
    parser.add_argument("--work-fill", type=float, default=0.8,
                        help="fraction of the camera frame the workspace occupies")
    parser.add_argument("--wrist-visibility", type=float, default=0.7)
    parser.add_argument("--smoothing", type=float, default=0.25)
    args = parser.parse_args()

    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    if args.camera_fps < 0:
        parser.error("camera FPS must be non-negative")
    if len(args.camera_fourcc) not in (0, 4):
        parser.error("camera FOURCC must contain exactly four characters")
    for name in ("work_width_mm", "work_height_mm"):
        value = getattr(args, name)
        if not np.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.work_fill <= 1:
        parser.error("--work-fill must be greater than 0 and at most 1")
    if not 0 <= args.wrist_visibility <= 1:
        parser.error("--wrist-visibility must be between 0 and 1")
    if not 0 < args.smoothing <= 1:
        parser.error("--smoothing must be greater than 0 and at most 1")

    camera = open_camera(args)

    pose_detector = None
    hand_detector = None
    tracker = WristXYTracker(args.smoothing, args.wrist_visibility)
    try:
        pose_detector = MediaPipeWristDetector(
            args.pose_model,
            num_poses=1,
            min_confidence=args.wrist_visibility,
            min_visibility=args.wrist_visibility,
        )
        hand_detector = MediaPipeHandDetector(args.hand_model)
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist XY", cv2.WINDOW_NORMAL)

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                if args.pipe.isdecimal():
                    raise RuntimeError(f"Camera read failed for source {args.pipe}")
                break
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            pose_results = pose_detector.detect(frame)
            detections = pose_results[0] if len(pose_results) == 1 else {}
            wrists = tracker.update(detections)
            hands = hand_detector.detect(frame)

            work_mm = (args.work_width_mm, args.work_height_mm)
            roi = compute_workspace_roi((width, height), work_mm, args.work_fill)
            webcam_view = draw_webcam_overlay(frame, wrists, hands)
            draw_workspace_rect(webcam_view, roi)
            xy_view = draw_wrist_xy_dashboard(wrists, roi, work_mm, len(pose_results))
            cv2.imshow("Webcam", webcam_view)
            cv2.imshow("Wrist XY", xy_view)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if (
                cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1
                or cv2.getWindowProperty("Wrist XY", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
    finally:
        camera.release()
        if pose_detector is not None:
            pose_detector.close()
        if hand_detector is not None:
            hand_detector.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
