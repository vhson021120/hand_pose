"""Start wrist/gesture tracking and coordinate the YuMi control loop."""

import argparse
import math
from pathlib import Path
import sys

import cv2

from tien_ich.nhan_dien_ban_tay import MediaPipeHandDetector, open_camera
from tien_ich.nhan_dien_co_tay import MediaPipeWristDetector
from tien_ich.ket_noi_yumi import YumiArmLink, utc_timestamp
from tien_ich.dieu_khien_gripper import GripperController
from tien_ich.theo_doi_co_tay_2d import (
    WristMotionGate, WristXYTracker, process_wrist_frame,
)
from tien_ich.chan_doan_toa_do import (
    AsyncDiagnosticLogger, draw_coordinate_diagnostics, draw_webcam_overlay,
    draw_workspace_rect, draw_gripper_status, draw_wrist_motion_status,
    draw_wrist_xy_dashboard, draw_robot_status, report_diagnostic_warning,
)

ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Control YuMi wrist XY and matching grippers with open-palm/fist signs."
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
    parser.add_argument(
        "--gesture-model", default=str(ROOT / "mo_hinh" / "gesture_recognizer.task"),
        help="trained palm/fist model; pass an empty string to use landmark geometry only",
    )
    parser.add_argument("--work-width-mm", type=float, default=400.0)
    parser.add_argument("--work-height-mm", type=float, default=300.0)
    parser.add_argument("--work-fill", type=float, default=0.8,
                        help="fraction of the camera frame the workspace occupies")
    parser.add_argument("--wrist-visibility", type=float, default=0.7)
    parser.add_argument("--smoothing", type=float, default=0.25)
    parser.add_argument("--motion-threshold-mm", type=float, default=10.0,
                        help="hold wrist keypoints until either axis changes by more than this mm")
    parser.add_argument("--robot-ip", default="192.168.125.1",
                        help="YuMi controller IP running robot/HandFollow.mod")
    parser.add_argument("--left-port", type=int, default=1025, help="port of T_ROB_L")
    parser.add_argument("--right-port", type=int, default=1026, help="port of T_ROB_R")
    parser.add_argument("--swap-arms", action="store_true",
                        help="left wrist drives ROB_R and right wrist drives ROB_L")
    parser.add_argument("--gripper-hand", choices=("both", "left", "right"), default="both",
                        help="hands allowed to control grippers (same mapping as wrists)")
    parser.add_argument("--no-gripper", action="store_true",
                        help="disable gripper commands while keeping wrist tracking")
    parser.add_argument("--gripper-stable-ms", type=float, default=250.0,
                        help="continuous stable gesture time before a command, in milliseconds")
    parser.add_argument("--gripper-confidence", type=float, default=0.7,
                        help="minimum handedness and gesture confidence for gripper control")
    parser.add_argument("--diagnostic", action="store_true",
                        help="show raw/filtered ROI and expected targets (not robot feedback)")
    parser.add_argument("--diagnostic-log", metavar="CSV", default=None,
                        help="enable asynchronous coordinate CSV logging to this path (off by default)")
    args = parser.parse_args(sys.argv[1:])

    if args.camera_width <= 0 or args.camera_height <= 0:
        parser.error("camera dimensions must be positive")
    if not math.isfinite(args.camera_fps) or args.camera_fps < 0:
        parser.error("camera FPS must be finite and non-negative")
    if len(args.camera_fourcc) not in (0, 4):
        parser.error("camera FOURCC must contain exactly four characters")
    for name in ("work_width_mm", "work_height_mm"):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.work_fill <= 1:
        parser.error("--work-fill must be greater than 0 and at most 1")
    if not 0 <= args.wrist_visibility <= 1:
        parser.error("--wrist-visibility must be between 0 and 1")
    if not 0 < args.smoothing <= 1:
        parser.error("--smoothing must be greater than 0 and at most 1")
    if not math.isfinite(args.motion_threshold_mm) or args.motion_threshold_mm <= 0:
        parser.error("--motion-threshold-mm must be finite and positive")
    if not math.isfinite(args.gripper_stable_ms) or args.gripper_stable_ms <= 0:
        parser.error("--gripper-stable-ms must be finite and positive")
    if not 0 <= args.gripper_confidence <= 1:
        parser.error("--gripper-confidence must be between 0 and 1")

    camera = open_camera(args)

    pose_detector = None
    hand_detector = None
    tracker = WristXYTracker(args.smoothing, args.wrist_visibility)
    motion_gate = WristMotionGate(args.motion_threshold_mm)
    gripper_controller = GripperController(
        stable_time=args.gripper_stable_ms / 1000.0,
        min_confidence=args.gripper_confidence,
        min_visibility=args.wrist_visibility,
    )
    links = {}
    armed = False
    diagnostic_logger = None
    frame_id = 0
    reported_log_error = None
    try:
        if args.diagnostic_log:
            diagnostic_logger = AsyncDiagnosticLogger(args.diagnostic_log)
        on_send = diagnostic_logger.record if diagnostic_logger else None
        links["RIGHT" if args.swap_arms else "LEFT"] = YumiArmLink(
            "ROB_L", args.robot_ip, args.left_port, on_send=on_send
        ).start()
        links["LEFT" if args.swap_arms else "RIGHT"] = YumiArmLink(
            "ROB_R", args.robot_ip, args.right_port, on_send=on_send
        ).start()
        print(f"Robot links -> {args.robot_ip}:{args.left_port} (ROB_L), "
              f"{args.robot_ip}:{args.right_port} (ROB_R). Press SPACE to arm.")
        pose_detector = MediaPipeWristDetector(
            args.pose_model,
            num_poses=1,
            min_confidence=args.wrist_visibility,
            min_visibility=args.wrist_visibility,
        )
        hand_detector = MediaPipeHandDetector(
            args.hand_model, gesture_model_path=args.gesture_model or None,
        )
        print("Gripper recognition: trained Gesture Recognizer" if args.gesture_model
              else "Gripper recognition: landmark geometry")
        cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
        cv2.namedWindow("Wrist XY", cv2.WINDOW_NORMAL)
        if args.diagnostic:
            cv2.namedWindow("Coordinate Diagnostic", cv2.WINDOW_NORMAL)

        while True:
            ok, frame = camera.read()
            if not ok or frame is None:
                if args.pipe.isdecimal():
                    raise RuntimeError(f"Camera read failed for source {args.pipe}")
                break
            frame = cv2.flip(frame, 1)
            frame_id += 1
            frame_timestamp = utc_timestamp()
            height, width = frame.shape[:2]
            pose_results = pose_detector.detect(frame)
            detections = pose_results[0] if len(pose_results) == 1 else {}
            work_mm = (args.work_width_mm, args.work_height_mm)
            wrists, roi, diagnostic_rows = process_wrist_frame(
                tracker, detections, (width, height), work_mm, args.work_fill,
                armed=armed, swap_arms=args.swap_arms, frame_id=frame_id,
                timestamp=frame_timestamp,
                motion_gate=motion_gate,
            )
            hands = hand_detector.detect(frame)
            eligible_sides = {
                row["wrist"] for row in diagnostic_rows
                if row["raw_inside_roi"] and row["filtered_inside_roi"]
                and not args.no_gripper
                and args.gripper_hand in ("both", row["wrist"].lower())
            }
            gripper_states = gripper_controller.update(
                hands, detections, roi, (width, height),
                armed=armed, eligible_sides=eligible_sides,
            )
            for row in diagnostic_rows:
                state = gripper_states[row["wrist"]]
                enabled = (not args.no_gripper
                           and args.gripper_hand in ("both", row["wrist"].lower()))
                can_command = enabled and row["wrist"] in eligible_sides and armed
                row.update(
                    gripper_gesture=state.gesture,
                    gripper_command=state.command if can_command else None,
                    gripper_reason=state.reason if enabled else "DISABLED",
                    gripper_model_gesture=getattr(state, "model_gesture", None),
                    gripper_model_confidence=getattr(state, "gesture_confidence", None),
                )

            webcam_view = draw_webcam_overlay(frame, wrists, hands)
            draw_workspace_rect(webcam_view, roi)
            draw_gripper_status(webcam_view, diagnostic_rows)
            draw_wrist_motion_status(webcam_view, diagnostic_rows)
            xy_view = draw_wrist_xy_dashboard(
                wrists, roi, work_mm, len(pose_results), motion_rows=diagnostic_rows,
                gripper_rows=diagnostic_rows,
            )
            for row in diagnostic_rows:
                link = links[row["wrist"]]
                row["tcp_connected"] = bool(link.connected)
                target = (row["x_target_mm"], row["y_target_mm"]) if row["target_valid"] else None
                link.set_target(target, context=row, gripper=row["gripper_command"])
                if diagnostic_logger:
                    diagnostic_logger.record(row)
            draw_robot_status(xy_view, armed, links)
            cv2.imshow("Webcam", webcam_view)
            cv2.imshow("Wrist XY", xy_view)
            log_status = "CSV OFF"
            if diagnostic_logger:
                log_status = f"CSV drops={diagnostic_logger.dropped_rows} error={diagnostic_logger.error or 'none'}"
                if diagnostic_logger.error and diagnostic_logger.error != reported_log_error:
                    reported_log_error = diagnostic_logger.error
                    report_diagnostic_warning(f"Diagnostic logging unavailable: {reported_log_error}")
            if args.diagnostic:
                cv2.imshow("Coordinate Diagnostic", draw_coordinate_diagnostics(diagnostic_rows, log_status))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                armed = not armed
                gripper_controller.reset()
                if not armed:
                    for link in set(links.values()):
                        link.set_target(None)
                print("Robot ARMED" if armed else "Robot HOLD")
            if (
                cv2.getWindowProperty("Webcam", cv2.WND_PROP_VISIBLE) < 1
                or cv2.getWindowProperty("Wrist XY", cv2.WND_PROP_VISIBLE) < 1
            ):
                break
    finally:
        # Invalidate both arms before any worker join can delay cleanup.
        for link in set(links.values()):
            link.set_target(None)
        for link in set(links.values()):
            link.close()
        camera.release()
        if pose_detector is not None:
            pose_detector.close()
        if hand_detector is not None:
            hand_detector.close()
        cv2.destroyAllWindows()
        if diagnostic_logger:
            drained = diagnostic_logger.close()
            if not drained or diagnostic_logger.dropped_rows or diagnostic_logger.error:
                report_diagnostic_warning(
                    f"Diagnostic CSV: drained={drained}, dropped={diagnostic_logger.dropped_rows}, "
                    f"error={diagnostic_logger.error}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
