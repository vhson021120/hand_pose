import argparse
from collections import Counter
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("\n[RUN]", " ".join(str(part) for part in command), flush=True)
    return subprocess.run(command, cwd=ROOT).returncode


def main():
    parser = argparse.ArgumentParser(description="Interactive KAPAO fine-tuning in 10-epoch rounds")
    parser.add_argument("--data", default="data/custom-kp.yaml")
    parser.add_argument("--weights", default="weights/kapao_s_coco.pt")
    parser.add_argument("--hyp", default="data/hyps/hyp.kp.yaml")
    parser.add_argument("--img", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--video", default="media/videos/demo/yBZ0Y2t0ceo_480p.mp4")
    parser.add_argument("--preview-seconds", type=int, default=5)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    data_path = (ROOT / args.data).resolve()
    if not data_path.is_file():
        raise SystemExit(f"Dataset config not found: {data_path}")
    if args.max_rounds < 1:
        raise SystemExit("max-rounds must be at least 1")

    with data_path.open(encoding="utf-8") as stream:
        data_config = yaml.safe_load(stream)
    dataset_root = Path(data_config.get("path", ""))
    if not dataset_root.is_absolute():
        dataset_root = ROOT / dataset_root
    for split in ("train", "val"):
        manifest = dataset_root / data_config[split]
        if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
            raise SystemExit(f"{split} manifest is missing or empty: {manifest}. Annotate frames first.")
    issue_counts = Counter()
    unique_tracks = set()
    labels_root = dataset_root / data_config.get("labels", "labels")
    for split in ("train", "val"):
        manifest = dataset_root / data_config[split]
        for image_entry in manifest.read_text(encoding="utf-8").splitlines():
            metadata_path = labels_root / split / f"{Path(image_entry).stem}.json"
            if not metadata_path.is_file():
                continue
            with metadata_path.open(encoding="utf-8") as stream:
                metadata = json.load(stream)
            issue_counts.update(metadata.get("issues", []))
            unique_tracks.update(person["track_id"] for person in metadata.get("people", []))
    print("Annotated issue tags:")
    for issue in ("jitter", "misplaced_point", "missing_point", "left_right_confusion",
                  "occlusion", "multiple_people", "fast_motion", "viewpoint_deformation"):
        print(f"  {issue}: {issue_counts[issue]} frames")
    print(f"Unique person track IDs: {len(unique_tracks)}")

    weights = str((ROOT / args.weights).resolve())
    preview_video = (ROOT / args.video).resolve()
    if not args.no_preview and not preview_video.is_file():
        raise SystemExit(f"Preview video not found: {preview_video}")
    run_root = ROOT / "runs" / "video_finetune"

    def validate(checkpoint, title):
        command = [
            sys.executable, str(ROOT / "val.py"),
            "--data", str(data_path), "--weights", checkpoint,
            "--imgsz", str(args.img), "--batch-size", "1", "--device", args.device,
            "--task", "val", "--scales", "1", "--flips", "-1",
        ]
        print(f"\n========== Validation score: {title} ==========")
        if run(command):
            raise SystemExit("Validation failed; check the terminal error above.")

    validate(weights, "baseline")

    for round_number in range(1, args.max_rounds + 1):
        round_name = f"round_{round_number:03d}"
        project = run_root / round_name
        train_command = [
            sys.executable, str(ROOT / "train.py"),
            "--data", str(data_path), "--weights", weights,
            "--hyp", str((ROOT / args.hyp).resolve()),
            "--img", str(args.img), "--batch-size", str(args.batch_size),
            "--epochs", "10", "--workers", str(args.workers),
            "--device", args.device, "--val-scales", "1", "--val-flips", "-1",
            "--project", str(project), "--name", "train", "--exist-ok",
        ]
        print(f"\n========== Training round {round_number}: 10 epochs ==========")
        if run(train_command):
            raise SystemExit("Training failed; check the terminal error above.")

        weights_dir = project / "train" / "weights"
        best_weights = weights_dir / "best.pt"
        if not best_weights.is_file():
            best_weights = weights_dir / "last.pt"
        if not best_weights.is_file():
            raise SystemExit(f"No checkpoint was produced in {weights_dir}")

        validate(str(best_weights), f"round {round_number}")
        weights = str(best_weights)
        print(f"\nCheckpoint for this round: {weights}")

        if not args.no_preview:
            preview_command = [
                sys.executable, str(ROOT / "demos" / "video.py"),
                "--video-path", str(preview_video), "--data", str(data_path),
                "--weights", weights, "--device", args.device, "--imgsz", str(args.img),
                "--display", "--start", "0", "--end", str(args.preview_seconds),
            ]
            print("\n========== Visual review ==========")
            if run(preview_command):
                raise SystemExit("Preview failed; check the terminal error above.")

        if round_number == args.max_rounds:
            break
        if input("Continue with another 10-epoch round? [y/N]: ").strip().lower() != "y":
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())