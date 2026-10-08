import argparse
from pathlib import Path

from common import PROJECT_ROOT, run_python_script


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run KAPAO with a webcam")
    parser.add_argument("--pipe", type=str, default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--hand-stride", type=int, default=2)
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "kapao_s_coco.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument("--hand", dest="use_hand", action="store_true", help="enable MediaPipe hand detection")
    hand_group.add_argument("--no-hand", dest="use_hand", action="store_false", help="disable MediaPipe hand detection")
    parser.set_defaults(use_hand=True)
    parser.add_argument("--hand-model", type=str, default=str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
    parser.add_argument("--full-pose", action="store_true")
    arm_group = parser.add_mutually_exclusive_group()
    arm_group.add_argument("--arm-mediapipe", dest="use_arm_mediapipe", action="store_true",
                           help="refine arm keypoints with MediaPipe pose (default)")
    arm_group.add_argument("--no-arm-mediapipe", dest="use_arm_mediapipe", action="store_false",
                           help="use only KAPAO arm keypoints")
    parser.set_defaults(use_arm_mediapipe=True)
    parser.add_argument("--arm-model", type=str, default=str(PROJECT_ROOT / "models" / "pose_landmarker_lite.task"))
    parser.add_argument("--arm-stride", type=int, default=1)
    parser.add_argument("--model-stride", type=int, default=0,
                        help="run the detector every N frames (0 = auto)")
    parser.add_argument("--detector", choices=("mediapipe", "kapao"), default="mediapipe")
    hand_overlay_group = parser.add_mutually_exclusive_group()
    hand_overlay_group.add_argument("--full-hand", dest="full_hand", action="store_true",
                                    help="show all hand keypoints (default)")
    hand_overlay_group.add_argument("--control-hand", dest="full_hand", action="store_false",
                                    help="show only wrist, thumb and index finger")
    parser.set_defaults(full_hand=True)
    parser.add_argument("--show-person-boxes", action="store_true")
    parser.add_argument("--commands", action="store_true",
                        help="enable high-level arm-pose commands (dry-run by default)")
    parser.add_argument("--command-stable-frames", type=int, default=8)
    parser.add_argument("--robot-host", type=str, help="TCP host for a RAPID command receiver")
    parser.add_argument("--robot-port", type=int, default=5000)
    parser.add_argument("--no-joint-preview", action="store_true",
                        help="hide the 2D direction-only Cartesian jog preview")
    parser.add_argument("--jog-calibration-frames", type=int, default=20)
    parser.add_argument("--jog-deadband", type=float, default=0.12)
    args = parser.parse_args(argv)

    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    result = run_python_script(
        "demos/webcam.py",
        [
            "--pipe",
            args.pipe,
            "--imgsz",
            str(args.imgsz),
            "--camera-width",
            str(args.camera_width),
            "--camera-height",
            str(args.camera_height),
            "--weights",
            str(weights_path.resolve()),
            "--data",
            str(PROJECT_ROOT / "data" / "coco-kp.yaml"),
            "--device",
            args.device,
            "--hand" if args.use_hand else "--no-hand",
            "--hand-model",
            args.hand_model,
            "--hand-stride",
            str(args.hand_stride),
            *( ["--full-pose"] if args.full_pose else [] ),
            "--arm-mediapipe" if args.use_arm_mediapipe else "--no-arm-mediapipe",
            "--arm-model",
            args.arm_model,
            "--arm-stride",
            str(args.arm_stride),
            "--model-stride",
            str(args.model_stride),
            "--detector",
            args.detector,
            "--full-hand" if args.full_hand else "--control-hand",
            *( ["--show-person-boxes"] if args.show_person_boxes else [] ),
            *( ["--commands"] if args.commands else [] ),
            "--command-stable-frames",
            str(args.command_stable_frames),
            *( ["--robot-host", args.robot_host] if args.robot_host else [] ),
            "--robot-port",
            str(args.robot_port),
            *( ["--no-joint-preview"] if args.no_joint_preview else [] ),
            "--jog-calibration-frames",
            str(args.jog_calibration_frames),
            "--jog-deadband",
            str(args.jog_deadband),
        ],
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
