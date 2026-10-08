import argparse


def main():
    parser = argparse.ArgumentParser(description="KAPAO Runner", add_help=False)
    parser.add_argument("-h", "--help", action="store_true")
    parser.add_argument("--mode", choices=["image", "video", "webcam", "all"], default=None)
    args, remaining_args = parser.parse_known_args()

    if args.help:
        if args.mode == "webcam":
            import run_webcam
            return run_webcam.main(["--help", *remaining_args])
        parser.print_help()
        return 0

    if args.mode is None:
        print("=== KAPAO Runner ===")
        print("1. Ảnh")
        print("2. Video")
        print("3. Webcam")
        print("4. Thoát")
        choice = input("Chọn chế độ (1/2/3/4): ").strip()
        mode = {"1": "image", "2": "video", "3": "webcam", "4": "exit"}.get(choice, "exit")
    else:
        mode = args.mode

    if mode == "exit":
        print("Thoát.")
        return 0

    if mode == "image":
        if remaining_args:
            parser.error("unrecognized arguments: " + " ".join(remaining_args))
        import run_image
        return run_image.main()
    if mode == "video":
        if remaining_args:
            parser.error("unrecognized arguments: " + " ".join(remaining_args))
        import run_video
        return run_video.main()
    if mode == "webcam":
        import run_webcam
        return run_webcam.main(remaining_args)
    if mode == "all":
        if remaining_args:
            parser.error("unrecognized arguments: " + " ".join(remaining_args))
        print("=== Chế độ Ảnh ===")
        print("Khởi chạy: python run_app/run_image.py")
        print("=== Chế độ Video ===")
        print("Khởi chạy: python run_app/run_video.py")
        print("=== Chế độ Webcam ===")
        print("Khởi chạy: python run_app/run_webcam.py")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
