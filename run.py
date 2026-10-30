import os
import sys
from pathlib import Path
from time import monotonic

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
FILE = Path(__file__).absolute()
sys.path.insert(0, str(ROOT))

import argparse
from typing import Optional, Tuple
import yaml
import cv2
import numpy as np
import torch
from tien_ich.tien_ich_pytorch import select_device, time_sync
from tien_ich.ham_dung_chung import check_img_size
from tien_ich.bo_tai_du_lieu import LoadWebcam
from mo_hinh.nap_mo_hinh import attempt_load
from danh_gia import run_nms, post_process_batch
from tien_ich.ket_hop_ban_tay_pose import HandKapoFusion
from tien_ich.on_dinh_keypoint import ControlPoseTracker
from tien_ich.theo_doi_quang_hoc import PoseFlowTracker
from tien_ich.xem_truoc_huong_tay import ArmJogPreview, ArmJogState
from tien_ich.ve_mo_hinh_3d import render_relative_3d_canvas
from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristPreview, RelativeWristState
from tien_ich.on_dinh_co_tay import (
    StabilityBenchmark, StabilizerConfig, TrackState, WristStabilizer,
)
from tien_ich.anh_xa_toa_do_robot import (
    RobotMappingConfig,
    RobotMappingCsvLogger,
    RobotTargetMapper,
    RobotTargetPreview,
)
from tien_ich.uoc_luong_khoang_cach import (
    MetricDistanceSmoother,
    RangeCalibration,
    ShoulderRangeEstimator,
    WorkingDistanceGate,
)
from tien_ich.dieu_khien_webcam import CaptureGestureController, classify_control_gesture

FACE_POINTS = (0, 1, 2, 3, 4)
ARM_POINTS = (5, 6, 7, 8, 9, 10)
FACE_CONNECTIONS = ((0, 1), (0, 2), (1, 3), (2, 4))
ARM_CONNECTIONS = ((5, 7), (7, 9), (6, 8), (8, 10))
HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
LEFT_BODY_POINTS = (5, 7, 9)
RIGHT_BODY_POINTS = (6, 8, 10)
BODY_POINT_LABELS = {5: 'LS', 6: 'RS', 7: 'LE', 8: 'RE', 9: 'LW', 10: 'RW'}
LEFT_COLOR = (0, 220, 0)
RIGHT_COLOR = (255, 0, 255)
FACE_COLOR = (255, 190, 80)


def draw_capture_status(image, controller: CaptureGestureController) -> None:
    state_color = {
        controller.WAITING: (0, 200, 255),
        controller.RUNNING: (100, 255, 100),
        controller.PAUSED: (0, 200, 255),
        controller.STOPPED: (0, 0, 255),
    }[controller.state]
    cv2.putText(image, controller.instruction, (12, image.shape[0] - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, state_color, 1, cv2.LINE_AA)


def draw_hand_status(
    image,
    detector_status: str,
    detected_count: int,
    shown_count: int,
    frame_age: int,
) -> None:
    """Show whether hand detection is active and how many hands were found."""
    x1, y1 = 10, 10
    x2 = min(image.shape[1] - 1, 300)
    y2 = min(image.shape[0] - 1, 94)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (24, 30, 38), thickness=-1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    status_color = (100, 255, 100) if detector_status == 'ON' else (0, 170, 255)
    cv2.putText(image, f'HAND DETECTOR: {detector_status}', (x1 + 10, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, status_color, 1, cv2.LINE_AA)
    if detector_status == 'ON':
        cv2.putText(image, f'DETECTED: {detected_count} | DRAWN: {shown_count}',
                    (x1 + 10, y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                    (240, 240, 240), 1, cv2.LINE_AA)
        cv2.putText(image, f'LAST DETECTION: {frame_age} FRAME(S) AGO',
                    (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (210, 210, 210), 1, cv2.LINE_AA)
    elif detector_status == 'UNAVAILABLE':
        cv2.putText(image, 'INIT ERROR: CHECK CONSOLE',
                    (x1 + 10, y1 + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (180, 180, 180), 1, cv2.LINE_AA)


def _format_point(point: Optional[Tuple[int, int]]) -> str:
    return f'({point[0]},{point[1]})px' if point is not None else '(--,--)px'


def draw_joint_preview(
    image,
    left_arm: ArmJogState,
    right_arm: ArmJogState,
) -> None:
    """Show direction-only jog previews, not robot joint targets."""
    panel_width = min(960, max(320, image.shape[1] - 20))
    panel_height = 152
    x1 = 10
    y1 = max(10, image.shape[0] - panel_height - 10)
    x2 = min(image.shape[1] - 1, x1 + panel_width)
    y2 = min(image.shape[0] - 1, y1 + panel_height)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (24, 30, 38), thickness=-1)
    cv2.addWeighted(overlay, 0.86, image, 0.14, 0, image)

    lines = [
        ("CARTESIAN JOG PREVIEW - IMAGE DIRECTIONS ONLY - NOT SENT", (255, 255, 255)),
    ]
    for label, arm, color in (
        ("LEFT HUMAN ARM -> LEFT YUMI ARM", left_arm, LEFT_COLOR),
        ("RIGHT HUMAN ARM -> RIGHT YUMI ARM", right_arm, RIGHT_COLOR),
    ):
        delta = (
            f"dx {arm.delta_x_px:+.0f}px dy {arm.delta_y_px:+.0f}px"
            if arm.delta_x_px is not None and arm.delta_y_px is not None
            else "dx -- dy --"
        )
        if arm.direction == "CALIBRATING":
            progress = f"{arm.calibration_frames} frames"
        elif arm.calibrated and arm.direction != "NO POSE":
            progress = "ready"
        else:
            progress = "--"
        lines.append((
            f"{label} | {arm.direction} | {delta} | S{_format_point(arm.shoulder)} "
            f"W{_format_point(arm.wrist)} | {progress}",
            color if arm.direction not in ("NO POSE", "CALIBRATING", "WAITING FOR EXACTLY ONE PERSON")
            else (0, 190, 255),
        ))
    lines.append(("C = RECALIBRATE NEUTRAL | LEFT/RIGHT/UP/DOWN ARE SCREEN DIRECTIONS", (180, 180, 180)))
    lines.append(("NEAR/FAR NEEDS DEPTH; NO ROBOT MOTION OR SAFETY FUNCTION", (160, 160, 160)))

    font_scale = min(0.47, max(0.30, (x2 - x1 - 20) / 1400))
    text_y = y1 + 25
    for text, color in lines:
        cv2.putText(image, text, (x1 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, 1, cv2.LINE_AA)
        text_y += 27


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pipe', default='0', help='webcam pipe, 0 for default camera')
    parser.add_argument('--weights', default=str(FILE.parents[0] / 'weights' / 'kapao_s_coco.pt'))
    parser.add_argument('--data', default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--camera-width', type=int, default=1280)
    parser.add_argument('--camera-height', type=int, default=720)
    parser.add_argument('--camera-fps', type=float, default=0,
                        help='request a capture FPS from the driver (0 keeps the driver default)')
    parser.add_argument('--camera-fourcc', type=str, default='',
                        help='request a capture codec such as MJPG (empty keeps the driver default)')
    stabilizer_group = parser.add_mutually_exclusive_group()
    stabilizer_group.add_argument('--wrist-stabilizer', dest='wrist_stabilizer', action='store_true',
                                  help='One Euro + state machine + outlier gating for wrist 3D (default)')
    stabilizer_group.add_argument('--no-wrist-stabilizer', dest='wrist_stabilizer', action='store_false',
                                  help='restore the previous unfiltered wrist pipeline')
    parser.set_defaults(wrist_stabilizer=True)
    parser.add_argument('--wrist-min-cutoff', type=float, default=1.5,
                        help='One Euro min cutoff (Hz); lower = smoother but more lag')
    parser.add_argument('--wrist-beta', type=float, default=6.0,
                        help='One Euro speed coefficient; higher = less lag during fast motion')
    parser.add_argument('--wrist-max-speed', type=float, default=6.0,
                        help='reject wrist samples faster than this many m/s')
    parser.add_argument('--wrist-min-visibility', type=float, default=0.5)
    parser.add_argument('--wrist-uncertain-frames', type=int, default=6,
                        help='frames a wrist may be missing before it is declared LOST')
    parser.add_argument('--wrist-reacquire-frames', type=int, default=4,
                        help='consistent valid frames needed to leave REACQUIRING')
    parser.add_argument('--wrist-hand-refine', action='store_true',
                        help='let a nearby Hand Landmarker wrist (pixel space) confirm a low-visibility Pose wrist')
    parser.add_argument('--wrist-keep-preview-smoothing', action='store_true',
                        help='keep the legacy EMA in the relative wrist preview on top of One Euro')
    parser.add_argument('--wrist-benchmark-csv', type=str,
                        help='write per-frame wrist stability telemetry; a summary is printed on exit')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or cpu')
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument('--hand', dest='use_hand', action='store_true', help='enable MediaPipe hand detection')
    hand_group.add_argument('--no-hand', dest='use_hand', action='store_false', help='disable MediaPipe hand detection')
    parser.set_defaults(use_hand=True)
    parser.add_argument('--hand-model', type=str, default=str(FILE.parents[0] / 'mo_hinh' / 'hand_landmarker.task'))
    parser.add_argument('--hand-stride', type=int, default=2,
                        help='run MediaPipe once every N webcam frames')
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument('--arm-mediapipe', dest='use_arm_mediapipe', action='store_true',
                           help='refine shoulder/elbow/wrist with MediaPipe PoseLandmarker (default)')
    arm_group.add_argument('--no-arm-mediapipe', dest='use_arm_mediapipe', action='store_false',
                           help='use only KAPAO arm keypoints')
    parser.set_defaults(use_arm_mediapipe=True)
    parser.add_argument('--arm-model', type=str, default=str(FILE.parents[0] / 'mo_hinh' / 'pose_landmarker_lite.task'))
    parser.add_argument('--arm-stride', type=int, default=1,
                        help='run MediaPipe arm detection every N KAPAO detector runs')
    parser.add_argument('--model-stride', type=int, default=0,
                        help='run the detector every N frames, optical flow in between '
                             '(0 = auto: 1 for mediapipe, 2 for kapao)')
    parser.add_argument('--detector', choices=('mediapipe', 'kapao'), default='mediapipe',
                        help='mediapipe = fast low-latency pose detector (default); kapao = slower KAPAO network')
    parser.add_argument('--show-person-boxes', action='store_true', help='show person bounding boxes')
    parser.add_argument('--conf-thres', type=float, default=0.3)
    parser.add_argument('--iou-thres', type=float, default=0.45)
    parser.add_argument('--no-kp-dets', action='store_true')
    parser.add_argument('--conf-thres-kp', type=float, default=0.5)
    parser.add_argument('--conf-thres-kp-person', type=float, default=0.2)
    parser.add_argument('--iou-thres-kp', type=float, default=0.45)
    parser.add_argument('--overwrite-tol', type=int, default=50, help='pixel tolerance for kp det overwrite')
    parser.add_argument('--kp-size', type=int, default=2)
    parser.add_argument('--kp-thick', type=int, default=2)
    parser.add_argument('--line-thick', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--no-fps', action='store_true', help='disable FPS overlay and logging')
    jog_preview_group = parser.add_mutually_exclusive_group()
    jog_preview_group.add_argument('--joint-preview', dest='show_joint_preview', action='store_true',
                                   help='show the optional 2D direction-only arm preview')
    jog_preview_group.add_argument('--no-joint-preview', dest='show_joint_preview', action='store_false',
                                   help='hide the optional 2D arm preview (default)')
    parser.set_defaults(show_joint_preview=False)
    parser.add_argument('--jog-calibration-frames', type=int, default=20,
                        help='valid neutral frames before the 2D/3D wrist previews are active')
    parser.add_argument('--jog-deadband', type=float, default=0.12,
                        help='minimum wrist shift as a fraction of person-box height')
    parser.add_argument('--mapping-config', type=str,
                        help='confirmed camera-to-robot mapping JSON; enables dry-run target preview only')
    parser.add_argument('--mapping-log', type=str,
                        help='optional CSV path for raw 3D landmarks and dry-run mapping telemetry')
    parser.add_argument('--start-mode', choices=('zone-thumbs-up', 'zone', 'thumbs-up'),
                        default='zone-thumbs-up',
                        help='zone-thumbs-up: one person inside the work zone AND thumbs-up (default); '
                             'zone: start automatically in the zone; thumbs-up: ignore distance')
    parser.add_argument('--working-distance-m', type=float, default=1.0)
    parser.add_argument('--working-distance-tolerance-m', type=float, default=0.2)
    parser.add_argument('--range-calibration', type=str, default='data/hieu_chuan_khoang_cach.json',
                        help='shoulder-width range calibration JSON')
    parser.add_argument('--range-calibrate', action='store_true',
                        help='interactively fit the range scale at --range-distances, then exit')
    parser.add_argument('--range-validate', action='store_true',
                        help='validate the saved range scale at --range-distances, then exit')
    parser.add_argument('--range-distances', type=float, nargs='+', default=[0.8, 1.0, 1.2],
                        help='real camera-to-chest distances in meters for calibrate/validate')
    parser.add_argument('--range-shoulder-width-m', type=float, default=None,
                        help='real shoulder width in meters (default: value in calibration, else 0.38)')
    parser.add_argument('--focal-calibrate', action='store_true',
                        help='calibrate focal length/principal point with a printed chessboard, then exit')
    parser.add_argument('--chessboard-size', type=int, nargs=2, default=[9, 6], metavar=('COLS', 'ROWS'),
                        help='chessboard inner corners (columns rows)')
    parser.add_argument('--chessboard-square-mm', type=float, default=25.0,
                        help='printed chessboard square size in millimeters (measure it)')
    parser.add_argument('--max-yaw-deg', type=float, default=25.0,
                        help='reject chest range when the torso is turned more than this angle')
    parser.add_argument('--yaw-test', action='store_true',
                        help='measure range error with the torso turned (at the first --range-distances), then exit')
    parser.add_argument('--yaw-test-angles', type=float, nargs='+', default=[0.0, 15.0, 30.0],
                        help='approximate torso angles to try in --yaw-test')
    parser.add_argument('--wrist-validate', action='store_true',
                        help='evaluate wrist depth error against real forward reach, then exit')
    parser.add_argument('--wrist-offsets', type=float, nargs='+', default=[0.0, 0.2, 0.4],
                        help='real wrist distance in front of the chest plane (m) for --wrist-validate')
    parser.add_argument('--wrist-chest-distance', type=float, default=1.0,
                        help='approximate chest distance used for --wrist-validate prompts (m)')
    parser.add_argument('--wrist-side', choices=('LEFT', 'RIGHT'), default='RIGHT')
    parser.add_argument('--allow-uncalibrated-range', action='store_true',
                        help='let the work-zone gate run before the range scale is validated (unreliable)')

    args = parser.parse_args()
    if not args.use_hand:
        parser.error('gesture-controlled capture requires MediaPipe hand detection; remove --no-hand')
    if args.model_stride < 0:
        parser.error('--model-stride must be non-negative')
    if args.arm_stride < 1:
        parser.error('--arm-stride must be at least 1')
    if args.hand_stride < 1:
        parser.error('--hand-stride must be at least 1')
    if args.jog_calibration_frames < 1:
        parser.error('--jog-calibration-frames must be at least 1')
    if args.jog_deadband <= 0:
        parser.error('--jog-deadband must be positive')
    if not np.isfinite(args.working_distance_m) or args.working_distance_m <= 0:
        parser.error('--working-distance-m must be positive')
    if (
        not np.isfinite(args.working_distance_tolerance_m)
        or args.working_distance_tolerance_m <= 0
    ):
        parser.error('--working-distance-tolerance-m must be positive')
    modes = [args.focal_calibrate, args.range_calibrate, args.range_validate,
             args.yaw_test, args.wrist_validate]
    if sum(bool(mode) for mode in modes) > 1:
        parser.error('choose only one calibration/validation mode at a time')
    if any(not np.isfinite(d) or d <= 0 for d in args.range_distances):
        parser.error('--range-distances must be positive')
    if not 0 < args.max_yaw_deg < 90:
        parser.error('--max-yaw-deg must be between 0 and 90')
    if args.chessboard_size[0] < 3 or args.chessboard_size[1] < 3 or args.chessboard_size[0] == args.chessboard_size[1]:
        parser.error('--chessboard-size needs two different values of at least 3 (inner corners)')
    if not np.isfinite(args.chessboard_square_mm) or args.chessboard_square_mm <= 0:
        parser.error('--chessboard-square-mm must be positive')
    if any(not np.isfinite(o) or o < 0 for o in args.wrist_offsets) or len(args.wrist_offsets) < 3:
        parser.error('--wrist-offsets needs at least three non-negative values')
    if any(not np.isfinite(a) or not 0 <= a < 90 for a in args.yaw_test_angles):
        parser.error('--yaw-test-angles must be between 0 and 90')
    if args.range_shoulder_width_m is not None and not (
        np.isfinite(args.range_shoulder_width_m) and args.range_shoulder_width_m > 0
    ):
        parser.error('--range-shoulder-width-m must be positive')
    args.auto_work_zone = args.start_mode in ('zone', 'zone-thumbs-up')
    args.require_thumbs_up = args.start_mode in ('zone-thumbs-up', 'thumbs-up')

    try:
        mapping_config = (
            RobotMappingConfig.from_json(args.mapping_config)
            if args.mapping_config else None
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.error(f'invalid --mapping-config: {exc}')

    with open(args.data) as f:
        data = yaml.safe_load(f)

    data['imgsz'] = args.imgsz
    data['conf_thres'] = args.conf_thres
    data['iou_thres'] = args.iou_thres
    data['use_kp_dets'] = not args.no_kp_dets
    data['conf_thres_kp'] = args.conf_thres_kp
    data['iou_thres_kp'] = args.iou_thres_kp
    data['conf_thres_kp_person'] = args.conf_thres_kp_person
    data['overwrite_tol'] = args.overwrite_tol
    data['scales'] = [1]
    data['flips'] = [None]
    data['count_fused'] = False

    device = select_device(args.device, batch_size=1)
    print('Using device:', device)

    use_kapao = args.detector == 'kapao'
    arm_refiner = None
    if args.use_arm_mediapipe or not use_kapao:
        from tien_ich.nhan_dien_tu_the import MediaPipeArmRefiner
        try:
            arm_refiner = MediaPipeArmRefiner(args.arm_model)
            print(f'MediaPipe arm detection enabled; model={args.arm_model}')
        except Exception as exc:
            if not use_kapao:
                raise RuntimeError(
                    'MediaPipe PoseLandmarker is required by the default webcam detector. '
                    f'Could not initialize it: {exc}'
                ) from exc
            print(f'Warning: MediaPipe arm refinement unavailable: {exc}')
            use_kapao = True
    if mapping_config is not None and arm_refiner is None:
        raise RuntimeError(
            '--mapping-config requires MediaPipe PoseLandmarker world landmarks; '
            'enable arm MediaPipe refinement.'
        )

    range_calibration = RangeCalibration()
    range_calibration_path = Path(args.range_calibration)
    if range_calibration_path.is_file():
        try:
            range_calibration = RangeCalibration.from_json(range_calibration_path)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise RuntimeError(f'Invalid range calibration {range_calibration_path}: {exc}') from exc
    if args.range_shoulder_width_m is not None:
        range_calibration.shoulder_width_m = args.range_shoulder_width_m
    if args.focal_calibrate:
        from tien_ich.hieu_chuan_khoang_cach import run_focal_calibration
        if arm_refiner is not None:
            arm_refiner.close()
        sys.exit(run_focal_calibration(args, range_calibration, range_calibration_path))
    if args.range_calibrate or args.range_validate or args.yaw_test or args.wrist_validate:
        if arm_refiner is None:
            raise RuntimeError('Range calibration requires MediaPipe PoseLandmarker world landmarks.')
        from tien_ich import hieu_chuan_khoang_cach as calibration_sessions
        try:
            if args.yaw_test:
                code = calibration_sessions.run_yaw_test(
                    args, arm_refiner, range_calibration, range_calibration_path)
            elif args.wrist_validate:
                code = calibration_sessions.run_wrist_validation(
                    args, arm_refiner, range_calibration, range_calibration_path)
            else:
                code = calibration_sessions.run_range_session(
                    args, arm_refiner, range_calibration, range_calibration_path,
                    validate=args.range_validate,
                )
            sys.exit(code)
        finally:
            arm_refiner.close()
            cv2.destroyAllWindows()

    model = None
    half = False
    stride = 32
    imgsz = args.imgsz
    if use_kapao:
        model = attempt_load(args.weights, map_location=device)
        half = args.half & (device.type != 'cpu')
        if half:
            model.half()
        stride = int(model.stride.max())
        imgsz = check_img_size(args.imgsz, s=stride)

    dataset = LoadWebcam(
        pipe=args.pipe,
        img_size=imgsz,
        stride=stride,
        width=args.camera_width,
        height=args.camera_height,
        preprocess=use_kapao,
        fps=args.camera_fps,
        fourcc=args.camera_fourcc,
    )
    if not dataset.cap.isOpened():
        raise RuntimeError(f'Unable to open webcam source {args.pipe}')
    pose_tracker = ControlPoseTracker()
    arm_jog_preview = ArmJogPreview(
        calibration_frames=args.jog_calibration_frames,
        deadband=args.jog_deadband,
    )
    wrist_stabilizer_config = StabilizerConfig(
        enabled=args.wrist_stabilizer,
        min_cutoff_hz=args.wrist_min_cutoff,
        beta=args.wrist_beta,
        max_speed_m_s=args.wrist_max_speed,
        min_visibility=args.wrist_min_visibility,
        uncertain_max_frames=args.wrist_uncertain_frames,
        reacquire_frames=args.wrist_reacquire_frames,
        hand_refine=args.wrist_hand_refine,
    )
    wrist_stabilizer = WristStabilizer(wrist_stabilizer_config)
    wrist_benchmark = (
        StabilityBenchmark(args.wrist_benchmark_csv)
        if args.wrist_stabilizer and args.wrist_benchmark_csv else None
    )
    relative_wrist_preview = RelativeWristPreview(
        calibration_frames=(
            max(args.jog_calibration_frames, 30)
            if mapping_config is not None else args.jog_calibration_frames
        ),
        # One Euro đã lọc theo thời gian; tránh lọc EMA lần hai làm tăng độ trễ.
        smoothing=(
            1.0 if args.wrist_stabilizer and not args.wrist_keep_preview_smoothing else 0.35
        ),
    )
    _preview_reset = relative_wrist_preview.reset

    def _reset_wrist_pipeline():
        _preview_reset()
        wrist_stabilizer.reset()

    relative_wrist_preview.reset = _reset_wrist_pipeline
    target_mapper = RobotTargetMapper(mapping_config) if mapping_config is not None else None
    hand_detector = None
    hand_fusion = None
    hand_detector_status = 'UNAVAILABLE'
    from tien_ich.nhan_dien_ban_tay import MediaPipeHandDetector
    try:
        hand_detector = MediaPipeHandDetector(args.hand_model)
        hand_detector_status = 'ON'
        print(f'MediaPipe hand detection enabled; model={args.hand_model}')
        hand_fusion = HandKapoFusion(
            source_width=0,
            source_height=0,
            target_width=0,
            target_height=0,
            person_box_threshold=0.05,
        )
    except Exception as exc:
        dataset.release()
        if arm_refiner is not None:
            arm_refiner.close()
        raise RuntimeError(
            'Gesture-controlled webcam capture requires the MediaPipe hand model. '
            f'Could not initialize it: {exc}'
        ) from exc

    if device.type != 'cpu' and model is not None:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    model_stride = args.model_stride or (2 if use_kapao else 1)
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    cv2.namedWindow("3D Model", cv2.WINDOW_NORMAL)
    capture_controller = CaptureGestureController()
    is_capturing = False
    cached_bboxes = []
    cached_poses = []
    cached_pose_world_landmarks = []
    displayed_hand_results = []
    raw_hand_results = []
    wrist_reports = {}
    last_target_track_id = None
    mapping_logger = None
    range_ready = range_calibration.confirmed or args.allow_uncalibrated_range
    range_estimator = ShoulderRangeEstimator(range_calibration, max_yaw_deg=args.max_yaw_deg)
    distance_smoother = MetricDistanceSmoother(window_size=7, max_missing=5)
    work_zone_gate = WorkingDistanceGate(
        target_distance_m=args.working_distance_m,
        tolerance_m=args.working_distance_tolerance_m,
        required_consecutive=5,
        latch=not args.require_thumbs_up,
    )
    chest_distance = None
    wrist_ranges = {}
    range_reason = 'NO MEASUREMENT'
    zone_ready = False
    if args.auto_work_zone and not range_ready:
        print('Range scale is not confirmed (run --range-calibrate then --range-validate); '
              'falling back to thumbs-up start without the work-zone check.')
        args.auto_work_zone = False
        args.require_thumbs_up = True
    try:
        fps_smooth = None
        arm_results = []
        detector_runs = 0
        flow_tracker = PoseFlowTracker()
        detected_hand_count = 0
        last_hand_detection_frame = 0
        mapping_logger = RobotMappingCsvLogger(args.mapping_log) if args.mapping_log else None
        for i, (path, img, im0, _) in enumerate(dataset):
            t_frame_start = time_sync()
            frame_monotonic = monotonic()
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            t_model = t_nms = 0.0
            t_post_start = time_sync()
            pose_world_landmarks = cached_pose_world_landmarks
            world_measurement_fresh = False
            auto_waiting = (
                args.auto_work_zone
                and capture_controller.state == capture_controller.WAITING
                and not is_capturing
            )
            run_detector = (
                (is_capturing or auto_waiting)
                and (i % model_stride == 0 or not flow_tracker.poses)
            )
            if not is_capturing and not auto_waiting:
                bboxes = cached_bboxes
                poses = cached_poses
            elif run_detector and not use_kapao:
                t_model_start = time_sync()
                bboxes, poses, pose_world_landmarks = arm_refiner.detect_people_with_world(im0)
                world_measurement_fresh = True
                t_model = time_sync() - t_model_start
                t_post_start = time_sync()
                detector_runs += 1
                poses = pose_tracker.update(bboxes, poses)
                flow_tracker.reset(gray, bboxes, poses)
            elif run_detector:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()
                img = img / 255.0
                if len(img.shape) == 3:
                    img = img[None]
                t_model_start = time_sync()
                with torch.no_grad():
                    out = model(img, augment=False, kp_flip=data['kp_flip'], scales=data['scales'],
                                flips=data['flips'])[0]
                t_model = time_sync() - t_model_start

                t_nms_start = time_sync()
                person_dets, kp_dets = run_nms(data, out)
                t_nms = time_sync() - t_nms_start

                t_post_start = time_sync()
                bboxes, poses, _, _, _ = post_process_batch(data, img, [], [[im0.shape[:2]]], person_dets, kp_dets)
                if arm_refiner is not None:
                    if detector_runs % args.arm_stride == 0:
                        arm_results = arm_refiner.detect(im0)
                        pose_world_landmarks = arm_refiner.match_world_to_boxes(bboxes)
                        world_measurement_fresh = True
                        poses = arm_refiner.refine(bboxes, poses, arm_results)
                detector_runs += 1
                poses = pose_tracker.update(bboxes, poses)
                flow_tracker.reset(gray, bboxes, poses)
            elif is_capturing:
                bboxes, poses = flow_tracker.track(gray)
                poses = pose_tracker.update(bboxes, poses)
                flow_tracker.reset(gray, bboxes, poses)
            if is_capturing:
                cached_bboxes = list(bboxes)
                cached_poses = [pose.copy() for pose in poses]
                cached_pose_world_landmarks = [pose.copy() for pose in pose_world_landmarks]
                pose_world_landmarks = cached_pose_world_landmarks
            auto_start = False
            if capture_controller.state != capture_controller.WAITING:
                zone_ready = False
            if world_measurement_fresh and (is_capturing or auto_waiting):
                current_track_ids = pose_tracker.last_track_ids
                distance_smoother.retain_tracks(current_track_ids)
                single_person = (
                    len(bboxes) == 1
                    and len(poses) == 1
                    and len(pose_world_landmarks) == 1
                    and len(current_track_ids) == 1
                )
                measurement = None
                if single_person:
                    measurement = range_estimator.measure(
                        poses[0], pose_world_landmarks[0], im0.shape
                    )
                    range_reason = range_estimator.last_reason
                else:
                    range_reason = 'NEED EXACTLY ONE PERSON'
                gate_track_id = current_track_ids[0] if single_person else None
                chest_distance = (
                    distance_smoother.update(
                        gate_track_id,
                        measurement.chest_distance_m if measurement is not None else None,
                    )
                    if gate_track_id is not None else None
                )
                wrist_ranges = measurement.wrists if measurement is not None else {}
                if auto_waiting:
                    in_zone = work_zone_gate.update(
                        1 if single_person else 0,
                        gate_track_id,
                        chest_distance if range_ready else None,
                    )
                    zone_ready = in_zone
                    auto_start = in_zone and not args.require_thumbs_up
            t_post = time_sync() - t_post_start

            gesture_action = None
            if i % args.hand_stride == 0:
                raw_hand_results = hand_detector.detect(im0)
                detected_hand_count = len(raw_hand_results)
                last_hand_detection_frame = i
                if len(raw_hand_results) == 1:
                    gesture = classify_control_gesture(raw_hand_results[0].landmarks)
                else:
                    gesture = None
                if args.auto_work_zone and capture_controller.state == capture_controller.WAITING:
                    if not (args.require_thumbs_up and zone_ready):
                        gesture = None
                gesture_action = capture_controller.update(gesture, monotonic())
                if is_capturing:
                    person_boxes = [tuple(map(int, bbox)) for bbox in bboxes]
                    hand_fusion.source_width = im0.shape[1]
                    hand_fusion.source_height = im0.shape[0]
                    hand_fusion.target_width = im0.shape[1]
                    hand_fusion.target_height = im0.shape[0]
                    displayed_hand_results = hand_fusion.associate_hands_with_arms(
                        raw_hand_results, person_boxes, poses
                    )
                elif capture_controller.state == capture_controller.WAITING:
                    displayed_hand_results = raw_hand_results

            if gesture_action in ("start", "resume"):
                is_capturing = capture_controller.state == capture_controller.RUNNING
                flow_tracker.reset(gray, [], [])
                pose_tracker = ControlPoseTracker()
                distance_smoother.reset()
                chest_distance = None
                wrist_ranges = {}
                zone_ready = False
                last_target_track_id = None
                arm_jog_preview.reset()
                relative_wrist_preview.reset()
                if target_mapper is not None:
                    target_mapper.reset()
                print("Pose detection started." if gesture_action == "start" else "Pose detection resumed.")
            elif auto_start and capture_controller.start_automatically():
                is_capturing = True
                flow_tracker.reset(gray, bboxes, poses)
                cached_bboxes = list(bboxes)
                cached_poses = [pose.copy() for pose in poses]
                cached_pose_world_landmarks = [
                    pose.copy() for pose in pose_world_landmarks
                ]
                relative_wrist_preview.reset()
                arm_jog_preview.reset()
                if target_mapper is not None:
                    target_mapper.reset()
                print(
                    f"Recognition started automatically at "
                    f"{work_zone_gate.target_distance_m:.2f}m "
                    f"+/- {work_zone_gate.tolerance_m:.2f}m."
                )
            elif gesture_action == "pause":
                is_capturing = False
                bboxes = cached_bboxes
                poses = cached_poses
                pose_world_landmarks = cached_pose_world_landmarks
                flow_tracker.reset(gray, [], [])
                last_target_track_id = None
                arm_jog_preview.reset()
                relative_wrist_preview.reset()
                if target_mapper is not None:
                    target_mapper.reset()
                print("Pose detection paused.")
            elif gesture_action == "stop":
                print("Webcam session ended.")
                break

            jog_states = arm_jog_preview.update(bboxes, poses) if is_capturing else arm_jog_preview.update([], [])
            target_side = mapping_config.tracked_side if mapping_config is not None else "RIGHT"
            if world_measurement_fresh and is_capturing:
                if (
                    len(bboxes) == 1
                    and len(pose_world_landmarks) == 1
                    and len(pose_tracker.last_track_ids) == 1
                ):
                    current_track_id = pose_tracker.last_track_ids[0]
                    if (
                        last_target_track_id is not None
                        and current_track_id != last_target_track_id
                    ):
                        relative_wrist_preview.reset()
                        if target_mapper is not None:
                            target_mapper.reset()
                    last_target_track_id = current_track_id
                else:
                    relative_wrist_preview.reset()
                    if target_mapper is not None:
                        target_mapper.reset()
                    last_target_track_id = None
            stabilized_world_landmarks = pose_world_landmarks
            if (
                is_capturing
                and world_measurement_fresh
                and wrist_stabilizer_config.enabled
                and len(pose_world_landmarks) == 1
            ):
                t_stabilize_start = time_sync()
                # Chỉ dùng pixel Hand Landmarker (cùng hệ ảnh); không trộn hand world.
                hand_wrist_pixels = [
                    (float(h.landmarks[0][0]), float(h.landmarks[0][1]))
                    for h in raw_hand_results
                    if len(h.landmarks) > 0
                ]
                stabilized_world, wrist_reports = wrist_stabilizer.process(
                    pose_world_landmarks[0],
                    frame_monotonic,
                    poses[0] if len(poses) == 1 else None,
                    im0.shape,
                    hand_wrist_pixels,
                )
                stabilized_world_landmarks = [stabilized_world]
                if wrist_benchmark is not None and target_side in wrist_reports:
                    wrist_benchmark.record(
                        frame_monotonic,
                        wrist_reports[target_side],
                        (time_sync() - t_stabilize_start) * 1000.0,
                        (time_sync() - t_frame_start) * 1000.0,
                    )
            else:
                wrist_reports = {}
            wrist_states = (
                relative_wrist_preview.update(stabilized_world_landmarks)
                if is_capturing and world_measurement_fresh
                else relative_wrist_preview.update([])
            )
            selected_wrist_state = next(
                (state for state in wrist_states if state.side == target_side),
                None,
            )
            if target_mapper is None:
                target_preview = RobotTargetPreview(
                    False, "MAPPING NOT CONFIGURED", None, target_side
                )
            elif not is_capturing:
                target_mapper.reset()
                target_preview = target_mapper.hold("NOT CAPTURING", frame_monotonic)
            elif not world_measurement_fresh:
                target_preview = target_mapper.hold(
                    "WAITING FOR FRESH 3D FRAME", frame_monotonic
                )
            elif selected_wrist_state is None:
                target_preview = target_mapper.hold(
                    "NO TRACKED WRIST", frame_monotonic
                )
            elif (
                target_side in wrist_reports
                and wrist_reports[target_side].state == TrackState.UNCERTAIN
            ):
                target_preview = target_mapper.hold(
                    "WRIST UNCERTAIN", frame_monotonic
                )
            else:
                target_preview = target_mapper.update(
                    selected_wrist_state, frame_monotonic
                )
            if mapping_logger is not None:
                log_state = selected_wrist_state or RelativeWristState(
                    side=target_side,
                    delta_cm=None,
                    status=target_preview.status,
                    confidence=None,
                    calibration_frames=0,
                    calibrated=False,
                )
                raw_world_pose = (
                    np.asarray(pose_world_landmarks[0])
                    if world_measurement_fresh and len(pose_world_landmarks) == 1
                    else None
                )
                mapping_logger.write(
                    frame_monotonic, log_state, target_preview, raw_world_pose
                )

            im0_copy = im0.copy()
            t_draw_start = time_sync()
            for j, (bbox, pose) in enumerate(zip(bboxes, poses)):
                x1, y1, x2, y2 = bbox
                track_ids = pose_tracker.last_track_ids
                track_id = track_ids[j] if j < len(track_ids) else j
                distance = chest_distance if j == 0 and len(bboxes) == 1 else None
                distance_is_fresh = distance is not None
                if distance_is_fresh:
                    in_working_range = (
                        abs(distance - args.working_distance_m)
                        <= args.working_distance_tolerance_m
                    )
                    distance_label = (
                        f'CHEST {distance:.2f}m '
                        f'{"IN" if in_working_range else "OUT"} OF WORK RANGE'
                        f'{"" if range_ready else " (UNCALIBRATED)"}'
                    )
                else:
                    distance_label = f'CHEST --: {range_reason}'
                cv2.putText(
                    im0_copy,
                    distance_label,
                    (int(x1), min(im0_copy.shape[0] - 8, max(36, int(y1) + 18))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 220, 255) if distance_is_fresh else (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )
                if distance_is_fresh:
                    wrist_verified = range_calibration.wrist_metrics.get('acceptable') is True
                    for wrist_row, (side, wrist_range) in enumerate(sorted(wrist_ranges.items())):
                        wx, wy, wz = wrist_range.camera_xyz_m
                        cv2.putText(
                            im0_copy,
                            f'{side[0]}W dist {wrist_range.distance_m:.2f}m '
                            f'XYZ {wx:+.2f} {wy:+.2f} {wz:.2f}'
                            f'{"" if wrist_verified else " (DEPTH UNVERIFIED)"}',
                            (int(x1), min(im0_copy.shape[0] - 8, max(36, int(y1) + 38 + 18 * wrist_row))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45,
                            LEFT_COLOR if side == 'LEFT' else RIGHT_COLOR,
                            1,
                            cv2.LINE_AA,
                        )
                if not is_capturing:
                    continue
                if args.show_person_boxes:
                    cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)
                cv2.putText(im0_copy, f'P{j + 1}', (int(x1), max(18, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, FACE_COLOR, 1, cv2.LINE_AA)
                for first, second in FACE_CONNECTIONS:
                    if pose[first, -1] > 0.5 and pose[second, -1] > 0.5:
                        pt1 = (int(pose[first, 0]), int(pose[first, 1]))
                        pt2 = (int(pose[second, 0]), int(pose[second, 1]))
                        cv2.line(im0_copy, pt1, pt2, FACE_COLOR, 1, cv2.LINE_AA)
                for first, second in ARM_CONNECTIONS:
                    color = LEFT_COLOR if first in LEFT_BODY_POINTS else RIGHT_COLOR
                    if pose[first, -1] > 0.5 and pose[second, -1] > 0.5:
                        pt1 = (int(pose[first, 0]), int(pose[first, 1]))
                        pt2 = (int(pose[second, 0]), int(pose[second, 1]))
                        cv2.line(im0_copy, pt1, pt2, color, max(2, args.line_thick), cv2.LINE_AA)
                for point_index in FACE_POINTS + ARM_POINTS:
                    x, y, confidence = pose[point_index]
                    if confidence > 0.5:
                        color = (
                            FACE_COLOR if point_index in FACE_POINTS
                            else LEFT_COLOR if point_index in LEFT_BODY_POINTS
                            else RIGHT_COLOR
                        )
                        radius = 2 if point_index in FACE_POINTS else args.kp_size + 1
                        cv2.circle(im0_copy, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
                        if point_index in BODY_POINT_LABELS:
                            cv2.putText(im0_copy, BODY_POINT_LABELS[point_index], (int(x) + 3, int(y) - 3),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            for hand in displayed_hand_results:
                color = (0, 255, 0) if hand.is_left else (255, 0, 255)
                for first, second in HAND_CONNECTIONS:
                    pt1 = (int(hand.landmarks[first, 0]), int(hand.landmarks[first, 1]))
                    pt2 = (int(hand.landmarks[second, 0]), int(hand.landmarks[second, 1]))
                    cv2.line(im0_copy, pt1, pt2, color, 2, cv2.LINE_AA)
                for point_index in range(21):
                    point = hand.landmarks[point_index]
                    cv2.circle(im0_copy, (int(point[0]), int(point[1])), 2, color, -1, cv2.LINE_AA)
            im0 = cv2.addWeighted(im0, args.alpha, im0_copy, 1 - args.alpha, gamma=0)
            three_d_canvas = render_relative_3d_canvas(
                pose_world_landmarks,
                [hand.world_landmarks for hand in displayed_hand_results],
                hand_person_indices=[hand.person_index for hand in displayed_hand_results],
                hand_is_left=[hand.is_left for hand in displayed_hand_results],
                wrist_preview=wrist_states,
                target_preview=target_preview,
            )
            t_draw = time_sync() - t_draw_start

            t_frame = time_sync() - t_frame_start
            fps = 1.0 / (t_frame + 1e-8)
            fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.9 + fps * 0.1)

            if not args.no_fps:
                text = f'FPS {fps_smooth:.1f} | Model {t_model*1000:.0f}ms | NMS {t_nms*1000:.0f}ms | Post {t_post*1000:.0f}ms | Draw {t_draw*1000:.0f}ms'
                fps_position = (5, 278)
                cv2.putText(im0, text, fps_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # log every 30 frames
            if i % 30 == 0 and not args.no_fps:
                print(f'Frame {i}: FPS={fps_smooth:.2f}, hands={detected_hand_count}, '
                      f'hand_frame_age={i - last_hand_detection_frame}, model={t_model*1000:.1f}ms, '
                      f'nms={t_nms*1000:.1f}ms, post={t_post*1000:.1f}ms, '
                      f'draw={t_draw*1000:.1f}ms, total={t_frame*1000:.1f}ms')

            draw_hand_status(
                im0,
                hand_detector_status,
                detected_hand_count,
                len(displayed_hand_results),
                i - last_hand_detection_frame,
            )
            if args.show_joint_preview:
                draw_joint_preview(im0, *jog_states)
            draw_capture_status(im0, capture_controller)
            if args.auto_work_zone and capture_controller.state == capture_controller.WAITING:
                zone_status = (
                    'RANGE UNCALIBRATED (run --range-calibrate)' if not range_ready
                    else work_zone_gate.status
                )
                if zone_ready and args.require_thumbs_up:
                    zone_status = 'IN ZONE - SHOW THUMBS-UP TO START'
                cv2.putText(
                    im0,
                    f'WORK ZONE: {zone_status} '
                    f'({args.working_distance_m:.1f}m +/- '
                    f'{args.working_distance_tolerance_m:.1f}m)',
                    (12, max(24, im0.shape[0] - 42)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (0, 220, 255),
                    1,
                    cv2.LINE_AA,
                )
            cv2.imshow('Webcam', im0)
            cv2.imshow('3D Model', three_d_canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                arm_jog_preview.reset()
                relative_wrist_preview.reset()
                if target_mapper is not None:
                    target_mapper.reset()
                print('Wrist preview recalibration started; hold a neutral pose.')
            if key == ord('q'):
                print("Webcam session ended.")
                break
            if (
                cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1
                or cv2.getWindowProperty('3D Model', cv2.WND_PROP_VISIBLE) < 1
            ):
                print('A display window was closed. Stopping...')
                break
    except KeyboardInterrupt:
        print('Keyboard interrupt received. Stopping webcam...')
    finally:
        dataset.release()
        if hand_detector is not None:
            hand_detector.close()
        if arm_refiner is not None:
            arm_refiner.close()
        if mapping_logger is not None:
            mapping_logger.close()
        if wrist_benchmark is not None:
            wrist_benchmark.close()
            summary_side = mapping_config.tracked_side if mapping_config is not None else "RIGHT"
            print(f"Wrist stability summary ({summary_side}): {wrist_benchmark.summary(summary_side)}")
        cv2.destroyAllWindows()
