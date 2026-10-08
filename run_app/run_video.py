import argparse
from pathlib import Path

from common import PROJECT_ROOT, choose_file, run_python_script


def main():
    parser = argparse.ArgumentParser(description="Run KAPAO on a video")
    parser.add_argument("--video", type=str, default="", help="Path to the video file")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "kapao_s_coco.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--face", action="store_true")
    hand_group = parser.add_mutually_exclusive_group()
    hand_group.add_argument("--hand", dest="use_hand", action="store_true", default=True)
    hand_group.add_argument("--no-hand", dest="use_hand", action="store_false")
    parser.add_argument("--hand-model", type=str, default=str(PROJECT_ROOT / "models" / "hand_landmarker.task"))
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--full-pose", action="store_true")
    parser.add_argument("--full-hand", action="store_true")
    parser.add_argument("--show-person-boxes", action="store_true")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=-1)
    args = parser.parse_args()

    video_path = args.video
    if not video_path:
        video_path = choose_file(
            "Chọn video cần phân tích",
            [("Video files", "*.mp4;*.avi;*.mov;*.mkv;*.wmv;*.m4v")],
        )
    if not video_path:
        print("Không chọn video. Thoát.")
        return 0

    video_path = str(Path(video_path).resolve())
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    cmd_args = [
        "--video-path",
        video_path,
        "--imgsz",
        str(args.imgsz),
        "--weights",
        str(weights_path.resolve()),
        "--data",
        str(PROJECT_ROOT / "data" / "coco-kp.yaml"),
        "--device",
        args.device,
        "--start",
        str(args.start),
        "--end",
        str(args.end),
        "--hand-model",
        args.hand_model,
        "--frame-stride",
        str(args.frame_stride),
    ]
    cmd_args.append("--hand" if args.use_hand else "--no-hand")
    if args.face:
        cmd_args.append("--face")
    if args.full_pose:
        cmd_args.append("--full-pose")
    if args.full_hand:
        cmd_args.append("--full-hand")
    if args.show_person_boxes:
        cmd_args.append("--show-person-boxes")

    result = run_python_script(
        "demos/video.py",
        cmd_args,
        cwd=str(Path(video_path).resolve().parent),
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
