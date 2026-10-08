import argparse
from pathlib import Path

from common import PROJECT_ROOT, choose_file, run_python_script


def main():
    parser = argparse.ArgumentParser(description="Chạy KAPAO trên ảnh")
    parser.add_argument("--image", type=str, default="", help="Đường dẫn ảnh nếu truyền trực tiếp")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--weights", type=str, default=str(PROJECT_ROOT / "weights" / "kapao_l_coco.pt"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bbox", action="store_true")
    parser.add_argument("--pose", action="store_true")
    parser.add_argument("--face", action="store_true")
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        image_path = choose_file(
            "Chọn ảnh cần phân tích",
            [("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.webp")],
        )
    if not image_path:
        print("Không chọn ảnh. Thoát.")
        return 0

    image_path = str(Path(image_path).resolve())
    weights_path = Path(args.weights)
    if not weights_path.is_absolute():
        weights_path = PROJECT_ROOT / weights_path
    cmd_args = [
        "--img-path",
        image_path,
        "--imgsz",
        str(args.imgsz),
        "--weights",
        str(weights_path.resolve()),
        "--data",
        str(PROJECT_ROOT / "data" / "coco-kp.yaml"),
        "--device",
        args.device,
    ]
    if args.bbox:
        cmd_args.append("--bbox")
    if args.pose:
        cmd_args.append("--pose")
    if args.face:
        cmd_args.append("--face")
    if not any(flag in cmd_args for flag in ["--bbox", "--pose", "--face"]):
        cmd_args.extend(["--bbox", "--pose", "--face"])

    result = run_python_script(
        "demos/image.py",
        cmd_args,
        cwd=str(Path(image_path).resolve().parent),
    )
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
