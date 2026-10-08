import sys
from pathlib import Path
FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[1].as_posix())  # add kapao/ to path

import argparse
from typing import Optional, Tuple
import yaml
import cv2
import torch
from utils.torch_utils import select_device, time_sync
from utils.general import check_img_size
from utils.datasets import LoadWebcam
from models.experimental import attempt_load
from val import run_nms, post_process_batch
from utils.hand_kapo_fusion import HandKapoFusion
from utils.pose_tracking import ControlPoseTracker
from utils.flow_tracker import PoseFlowTracker
from utils.high_level_commands import HighLevelCommandRecognizer
from utils.command_transport import TcpCommandSender
from utils.arm_jog_preview import ArmJogPreview, ArmJogState

CONTROL_BODY_POINTS = (5, 6, 7, 8, 9, 10)
CONTROL_BODY_CONNECTIONS = ((5, 7), (7, 9), (6, 8), (8, 10))
CONTROL_HAND_POINTS = (0, 1, 2, 3, 4, 5, 6, 7, 8)
CONTROL_HAND_CONNECTIONS = ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8))
LEFT_BODY_POINTS = (5, 7, 9)
RIGHT_BODY_POINTS = (6, 8, 10)
BODY_POINT_LABELS = {5: 'LS', 6: 'RS', 7: 'LE', 8: 'RE', 9: 'LW', 10: 'RW'}
LEFT_COLOR = (0, 220, 0)
RIGHT_COLOR = (255, 0, 255)


def draw_command_status(
    image,
    recognizer: HighLevelCommandRecognizer,
    command_mode: str,
    last_command: str,
) -> None:
    """Draw command recognition and transport mode without implying robot state."""
    panel_width = min(360, max(260, image.shape[1] - 20))
    panel_height = 154
    x1 = max(10, image.shape[1] - panel_width - 10)
    y1 = 10
    x2 = min(image.shape[1] - 1, x1 + panel_width)
    y2 = min(image.shape[0] - 1, y1 + panel_height)
    if x2 <= x1 or y2 <= y1:
        return

    overlay = image.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (24, 30, 38), thickness=-1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)

    candidate = recognizer.candidate_command
    if candidate is None:
        pose_status = "POSE: NO VALID COMMAND POSE"
        pose_color = (180, 180, 180)
    else:
        stable_frames = min(recognizer.candidate_frames, recognizer.stable_frames)
        pose_status = f"POSE: {candidate} ({stable_frames}/{recognizer.stable_frames})"
        pose_color = (0, 220, 255) if stable_frames < recognizer.stable_frames else (100, 255, 100)

    lines = (
        ("KAPAO HIGH-LEVEL COMMANDS", (255, 255, 255)),
        (f"MODE: {command_mode}", (100, 220, 255)),
        (pose_status, pose_color),
        (f"LAST DETECTED: {last_command or '--'}", (230, 230, 230)),
        ("UP: LEFT/RIGHT | BOTH UP: STOP", (200, 200, 200)),
        ("BOTH DOWN: START | NOT ROBOT STATE", (160, 160, 160)),
    )
    text_y = y1 + 24
    for text, color in lines:
        cv2.putText(image, text, (x1 + 10, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, color, 1, cv2.LINE_AA)
        text_y += 24


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
    parser.add_argument('--weights', default=str(FILE.parents[1] / 'weights' / 'kapao_s_coco.pt'))
    parser.add_argument('--data', default='data/coco-kp.yaml')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--camera-width', type=int, default=1280)
    parser.add_argument('--camera-height', type=int, default=720)
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or cpu')
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument('--hand', dest='use_hand', action='store_true', help='enable MediaPipe hand detection')
    hand_group.add_argument('--no-hand', dest='use_hand', action='store_false', help='disable MediaPipe hand detection')
    parser.set_defaults(use_hand=True)
    parser.add_argument('--hand-model', type=str, default=str(FILE.parents[1] / 'models' / 'hand_landmarker.task'))
    parser.add_argument('--hand-stride', type=int, default=2,
                        help='run MediaPipe once every N webcam frames')
    parser.add_argument('--full-pose', action='store_true', help='show all 17 body keypoints instead of the control subset')
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument('--arm-mediapipe', dest='use_arm_mediapipe', action='store_true',
                           help='refine shoulder/elbow/wrist with MediaPipe PoseLandmarker (default)')
    arm_group.add_argument('--no-arm-mediapipe', dest='use_arm_mediapipe', action='store_false',
                           help='use only KAPAO arm keypoints')
    parser.set_defaults(use_arm_mediapipe=True)
    parser.add_argument('--arm-model', type=str, default=str(FILE.parents[1] / 'models' / 'pose_landmarker_lite.task'))
    parser.add_argument('--arm-stride', type=int, default=1,
                        help='run MediaPipe arm detection every N KAPAO detector runs')
    parser.add_argument('--model-stride', type=int, default=0,
                        help='run the detector every N frames, optical flow in between '
                             '(0 = auto: 1 for mediapipe, 2 for kapao)')
    parser.add_argument('--detector', choices=('mediapipe', 'kapao'), default='mediapipe',
                        help='mediapipe = fast low-latency pose detector (default); kapao = slower KAPAO network')
    hand_overlay_group = parser.add_mutually_exclusive_group()
    hand_overlay_group.add_argument('--full-hand', dest='full_hand', action='store_true',
                                    help='show all 21 hand keypoints (default)')
    hand_overlay_group.add_argument('--control-hand', dest='full_hand', action='store_false',
                                    help='show only wrist, thumb and index finger')
    parser.set_defaults(full_hand=True)
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
    parser.add_argument('--commands', action='store_true',
                        help='enable stable arm-pose commands (dry-run unless --robot-host is set)')
    parser.add_argument('--command-stable-frames', type=int, default=8,
                        help='consecutive frames required before emitting a command')
    parser.add_argument('--robot-host', type=str,
                        help='TCP host for a RAPID command receiver; omit for dry-run')
    parser.add_argument('--robot-port', type=int, default=5000,
                        help='TCP port for a RAPID command receiver (default: 5000)')
    parser.add_argument('--no-joint-preview', action='store_true',
                        help='hide the 2D direction-only Cartesian jog preview')
    parser.add_argument('--jog-calibration-frames', type=int, default=20,
                        help='neutral arm frames to average before the direction preview is active')
    parser.add_argument('--jog-deadband', type=float, default=0.12,
                        help='minimum wrist shift as a fraction of person-box height')

    args = parser.parse_args()
    if args.model_stride < 0:
        parser.error('--model-stride must be non-negative')
    if args.arm_stride < 1:
        parser.error('--arm-stride must be at least 1')
    if args.hand_stride < 1:
        parser.error('--hand-stride must be at least 1')
    if args.command_stable_frames < 1:
        parser.error('--command-stable-frames must be at least 1')
    if args.jog_calibration_frames < 1:
        parser.error('--jog-calibration-frames must be at least 1')
    if args.jog_deadband <= 0:
        parser.error('--jog-deadband must be positive')
    if not 1 <= args.robot_port <= 65535:
        parser.error('--robot-port must be between 1 and 65535')
    if args.robot_host and not args.commands:
        parser.error('--robot-host requires --commands')

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
        from utils.mediapipe_arm import MediaPipeArmRefiner
        try:
            arm_refiner = MediaPipeArmRefiner(args.arm_model)
            print(f'MediaPipe arm detection enabled; model={args.arm_model}')
        except (ImportError, FileNotFoundError) as exc:
            print(f'Warning: {exc}')
            if not use_kapao:
                print('Falling back to the KAPAO detector.')
                use_kapao = True

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
    )
    if not dataset.cap.isOpened():
        raise RuntimeError(f'Unable to open webcam source {args.pipe}')
    pose_tracker = ControlPoseTracker()
    arm_jog_preview = ArmJogPreview(
        calibration_frames=args.jog_calibration_frames,
        deadband=args.jog_deadband,
    )
    command_recognizer = HighLevelCommandRecognizer(stable_frames=args.command_stable_frames)
    command_sender = None
    command_mode = 'DRY RUN - NOT SENT'
    if args.commands:
        if args.robot_host:
            command_sender = TcpCommandSender(args.robot_host, args.robot_port)
            command_mode = f'TCP {args.robot_host}:{args.robot_port} (ACK in console)'
            print(f'High-level commands enabled; TCP target={args.robot_host}:{args.robot_port}')
        else:
            print('High-level commands enabled in dry-run mode; no commands will be sent.')
    last_command = ''
    hand_detector = None
    hand_fusion = None
    hand_detector_status = 'OFF (--no-hand)'
    if args.use_hand:
        from utils.mediapipe_hand import MediaPipeHandDetector
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
        except (ImportError, FileNotFoundError) as exc:
            print(f'Warning: {exc}')
            hand_detector_status = 'UNAVAILABLE'
            hand_detector = None
            hand_fusion = None

    if device.type != 'cpu' and model is not None:
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    model_stride = args.model_stride or (2 if use_kapao else 1)
    try:
        fps_smooth = None
        hand_results = []
        arm_results = []
        detector_runs = 0
        flow_tracker = PoseFlowTracker()
        detected_hand_count = 0
        last_hand_detection_frame = 0
        for i, (path, img, im0, _) in enumerate(dataset):
            t_frame_start = time_sync()
            gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
            run_detector = i % model_stride == 0 or not flow_tracker.poses
            t_model = t_nms = 0.0
            t_post_start = time_sync()
            if run_detector and not use_kapao:
                t_model_start = time_sync()
                bboxes, poses = arm_refiner.detect_people(im0)
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
                        poses = arm_refiner.refine(bboxes, poses, arm_results)
                detector_runs += 1
                poses = pose_tracker.update(bboxes, poses)
                flow_tracker.reset(gray, bboxes, poses)
            else:
                bboxes, poses = flow_tracker.track(gray)
            if args.commands:
                command = command_recognizer.update(bboxes, poses)
                if command is not None:
                    last_command = command
                    if command_sender is None:
                        print(f'[DRY RUN] {command}')
                    else:
                        command_sender.submit(command)
            t_post = time_sync() - t_post_start

            if hand_detector is not None:
                if i % args.hand_stride == 0:
                    hand_results = hand_detector.detect(im0)
                    detected_hand_count = len(hand_results)
                    last_hand_detection_frame = i
                if hand_fusion is not None:
                    person_boxes = [tuple(map(int, bbox)) for bbox in bboxes]
                    hand_fusion.source_width = im0.shape[1]
                    hand_fusion.source_height = im0.shape[0]
                    hand_fusion.target_width = im0.shape[1]
                    hand_fusion.target_height = im0.shape[0]
                    hand_results = hand_fusion.associate_hands_with_arms(
                        hand_results, person_boxes, poses
                    )

            jog_states = arm_jog_preview.update(bboxes, poses)

            im0_copy = im0.copy()
            t_draw_start = time_sync()
            for j, (bbox, pose) in enumerate(zip(bboxes, poses)):
                x1, y1, x2, y2 = bbox
                if args.show_person_boxes:
                    cv2.rectangle(im0_copy, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 255), thickness=1)
                if args.full_pose:
                    segments = ((segment, (255, 255, 255)) for segment in data['segments'].values())
                    body_point_indices = range(len(pose))
                else:
                    segments = (((first, second), color) for first, second, color in (
                        (5, 7, LEFT_COLOR), (7, 9, LEFT_COLOR),
                        (6, 8, RIGHT_COLOR), (8, 10, RIGHT_COLOR),
                    ))
                    body_point_indices = CONTROL_BODY_POINTS
                for (first, second), color in segments:
                    if pose[first, -1] and pose[second, -1]:
                        pt1 = (int(pose[first, 0]), int(pose[first, 1]))
                        pt2 = (int(pose[second, 0]), int(pose[second, 1]))
                        cv2.line(im0_copy, pt1, pt2, color, args.line_thick)
                if data['use_kp_dets']:
                    for point_index in body_point_indices:
                        x, y, c = pose[point_index]
                        if c:
                            color = (LEFT_COLOR if point_index in LEFT_BODY_POINTS else
                                     RIGHT_COLOR if point_index in RIGHT_BODY_POINTS else (0, 255, 255))
                            cv2.circle(im0_copy, (int(x), int(y)), args.kp_size + 1, color, -1)
                            if not args.full_pose:
                                cv2.putText(im0_copy, BODY_POINT_LABELS[point_index], (int(x) + 3, int(y) - 3),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            hand_points = range(21) if args.full_hand else CONTROL_HAND_POINTS
            hand_connections = (
                ((0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                 (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                 (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17))
                if args.full_hand else CONTROL_HAND_CONNECTIONS
            )
            for hand in hand_results:
                color = (0, 255, 0) if hand.is_left else (255, 0, 255)
                cv2.rectangle(
                    im0_copy,
                    (int(max(0, hand.x - hand.width / 2)), int(max(0, hand.y - hand.height / 2))),
                    (int(min(im0.shape[1], hand.x + hand.width / 2)), int(min(im0.shape[0], hand.y + hand.height / 2))),
                    color,
                    thickness=2,
                )
                for first, second in hand_connections:
                    pt1 = (int(hand.landmarks[first, 0]), int(hand.landmarks[first, 1]))
                    pt2 = (int(hand.landmarks[second, 0]), int(hand.landmarks[second, 1]))
                    cv2.line(im0_copy, pt1, pt2, color, 2)
                for point_index in hand_points:
                    point = hand.landmarks[point_index]
                    cv2.circle(im0_copy, (int(point[0]), int(point[1])), 3, color, -1)
            t_draw = time_sync() - t_draw_start
            im0 = cv2.addWeighted(im0, args.alpha, im0_copy, 1 - args.alpha, gamma=0)

            t_frame = time_sync() - t_frame_start
            fps = 1.0 / (t_frame + 1e-8)
            fps_smooth = fps if fps_smooth is None else (fps_smooth * 0.9 + fps * 0.1)

            if not args.no_fps:
                text = f'FPS {fps_smooth:.1f} | Model {t_model*1000:.0f}ms | NMS {t_nms*1000:.0f}ms | Post {t_post*1000:.0f}ms | Draw {t_draw*1000:.0f}ms'
                fps_position = (5, 178) if args.commands else (5, 120)
                cv2.putText(im0, text, fps_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # log every 30 frames
            if i % 30 == 0 and not args.no_fps:
                print(f'Frame {i}: FPS={fps_smooth:.2f}, hands={len(hand_results)}, '
                      f'hand_frame_age={i - last_hand_detection_frame}, model={t_model*1000:.1f}ms, '
                      f'nms={t_nms*1000:.1f}ms, post={t_post*1000:.1f}ms, '
                      f'draw={t_draw*1000:.1f}ms, total={t_frame*1000:.1f}ms')

            draw_hand_status(
                im0,
                hand_detector_status,
                detected_hand_count,
                len(hand_results),
                i - last_hand_detection_frame,
            )
            if args.commands:
                draw_command_status(im0, command_recognizer, command_mode, last_command)
            if not args.no_joint_preview:
                draw_joint_preview(im0, *jog_states)
            cv2.imshow('Webcam', im0)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c'):
                arm_jog_preview.reset()
                print('Arm jog preview recalibration started; hold both arms in a neutral pose.')
            if key == ord('q'):
                break
            if cv2.getWindowProperty('Webcam', cv2.WND_PROP_VISIBLE) < 1:
                print('Webcam window closed. Stopping...')
                break
    except KeyboardInterrupt:
        print('Keyboard interrupt received. Stopping webcam...')
    finally:
        dataset.release()
        if command_sender is not None:
            command_sender.close()
        if hand_detector is not None:
            hand_detector.close()
        if arm_refiner is not None:
            arm_refiner.close()
        cv2.destroyAllWindows()
