import argparse
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
SKELETON = (
    (5, 6), (5, 11), (11, 12), (12, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 1), (0, 2), (1, 3), (2, 4),
)
ISSUES = {
    "1": "jitter",
    "2": "misplaced_point",
    "3": "missing_point",
    "4": "left_right_confusion",
    "5": "occlusion",
    "6": "multiple_people",
    "7": "fast_motion",
    "8": "viewpoint_deformation",
}


def read_image(path):
    encoded = np.frombuffer(Path(path).read_bytes(), dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR)


def write_image(path, image):
    success, encoded = cv2.imencode(Path(path).suffix or ".jpg", image)
    if not success:
        raise OSError(f"Could not encode image: {path}")
    Path(path).write_bytes(encoded.tobytes())


def person_to_yolo_label(person, width, height):
    x1, y1, x2, y2 = person["box"]
    row = [0, (x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
           (x2 - x1) / width, (y2 - y1) / height]
    for point in person["points"]:
        row.extend((0, 0, 0) if point is None else
                   (point[0] / width, point[1] / height, point[2]))
    return " ".join(f"{value:.6f}" for value in row)


def load_saved_people(label_path, width, height, metadata_path=None):
    if not Path(label_path).is_file():
        return [], set()
    rows = [line.split() for line in Path(label_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    metadata = {}
    if metadata_path and Path(metadata_path).is_file():
        metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
    people = []
    for row_index, values in enumerate(rows):
        if len(values) != 56:
            continue
        values = [float(value) for value in values]
        cx, cy, box_width, box_height = values[1:5]
        box = (
            int((cx - box_width / 2) * width),
            int((cy - box_height / 2) * height),
            int((cx + box_width / 2) * width),
            int((cy + box_height / 2) * height),
        )
        points = []
        for point_index in range(17):
            x, y, visibility = values[5 + point_index * 3:8 + point_index * 3]
            points.append((x * width, y * height, visibility) if visibility > 0 else None)
        saved_person = metadata.get("people", [])
        track_id = saved_person[row_index].get("track_id", row_index + 1) if row_index < len(saved_person) else row_index + 1
        people.append({"box": box, "points": points, "index": 17, "track_id": track_id})
    return people, set(metadata.get("issues", []))


def carry_track_ids(previous_people, current_people, first_new_id):
    unused_previous = set(range(len(previous_people)))
    next_id = first_new_id
    for current in current_people:
        cx1, cy1, cx2, cy2 = current["box"]
        current_center = ((cx1 + cx2) / 2, (cy1 + cy2) / 2)
        candidates = []
        for index in unused_previous:
            previous = previous_people[index]
            px1, py1, px2, py2 = previous["box"]
            intersection = max(0, min(cx2, px2) - max(cx1, px1)) * max(0, min(cy2, py2) - max(cy1, py1))
            union = max(1, (cx2 - cx1) * (cy2 - cy1) + (px2 - px1) * (py2 - py1) - intersection)
            overlap = intersection / union
            previous_center = ((px1 + px2) / 2, (py1 + py2) / 2)
            distance = np.hypot(current_center[0] - previous_center[0], current_center[1] - previous_center[1])
            candidates.append((overlap, distance, index))
        if candidates:
            overlap, distance, previous_index = max(candidates, key=lambda item: (item[0], -item[1]))
            if overlap >= 0.01 or distance <= 0.2 * np.hypot(cx2 - cx1, cy2 - cy1):
                current["track_id"] = previous_people[previous_index]["track_id"]
                unused_previous.remove(previous_index)
                continue
        current["track_id"] = next_id
        next_id += 1
    return current_people, next_id


class KapaoDraftPredictor:
    def __init__(self, weights, device, image_size):
        sys.path.append(str(ROOT))
        import torch
        import yaml
        from models.experimental import attempt_load
        from utils.general import check_img_size
        from utils.torch_utils import select_device
        from val import post_process_batch, run_nms

        self.torch = torch
        self.post_process_batch = post_process_batch
        self.run_nms = run_nms
        with (ROOT / "data" / "coco-kp.yaml").open(encoding="utf-8") as stream:
            self.data = yaml.safe_load(stream)
        self.data.update({
            "imgsz": image_size,
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "use_kp_dets": True,
            "conf_thres_kp": 0.05,
            "iou_thres_kp": 0.45,
            "conf_thres_kp_person": 0.05,
            "overwrite_tol": 80,
            "scales": [1],
            "flips": [None],
            "count_fused": False,
        })
        self.device = select_device(device, batch_size=1)
        self.model = attempt_load(str(weights), map_location=self.device)
        self.stride = int(self.model.stride.max())
        self.image_size = check_img_size(image_size, s=self.stride)
        print(f"KAPAO draft suggestions enabled: {weights} on {self.device}")

    def detect(self, image):
        from utils.augmentations import letterbox

        prepared = letterbox(image, self.image_size, stride=self.stride)[0]
        prepared = prepared[:, :, ::-1].transpose((2, 0, 1)).copy()
        tensor = self.torch.from_numpy(prepared).to(self.device).float().div_(255.0)[None]
        with self.torch.no_grad():
            output = self.model(
                tensor,
                augment=True,
                kp_flip=self.data["kp_flip"],
                scales=self.data["scales"],
                flips=self.data["flips"],
            )[0]
        person_dets, kp_dets = self.run_nms(self.data, output)
        boxes, poses, _, _, _ = self.post_process_batch(
            self.data, tensor, [], [[image.shape[:2]]], person_dets, kp_dets
        )
        people = []
        for track_id, (box, pose) in enumerate(zip(boxes, poses), start=1):
            points = []
            for x, y, confidence in pose:
                if confidence > 0 and np.isfinite(x) and np.isfinite(y):
                    points.append((float(x), float(y), 2))
                else:
                    points.append(None)
            people.append({
                "box": tuple(int(value) for value in box),
                "points": points,
                "index": 17,
                "track_id": track_id,
                "draft": True,
                "reviewed": [False] * 17,
            })
        return people


class FrameAnnotator:
    def __init__(self, image, track_id_start=1, people=None):
        self.image = image
        self.height, self.width = image.shape[:2]
        self.scale = min(1.0, 1280 / self.width)
        self.people = people or []
        self.selected_person = 0 if self.people else None
        self.selected_keypoint = None
        self.dragging_point = None
        self.placing_missing = False
        self.active_box = None
        self.drawing_box = False
        self.current_person = None
        self.next_visibility = 2
        self.next_track_id = track_id_start
        self.issue_tags = set()

    def point_from_display(self, x, y):
        return int(x / self.scale), int(y / self.scale)

    def mouse(self, event, x, y, flags, _):
        if event == cv2.EVENT_LBUTTONDOWN and self.drawing_box:
            self.active_box = [self.point_from_display(x, y), None]
        elif event == cv2.EVENT_LBUTTONDOWN and self.placing_missing and self.selected_person is not None and self.selected_keypoint is not None:
            px, py = self.point_from_display(x, y)
            person = self.people[self.selected_person]
            person["points"][self.selected_keypoint] = (px, py, 2)
            person.setdefault("reviewed", [True] * 17)[self.selected_keypoint] = True
            self.placing_missing = False
        elif event == cv2.EVENT_LBUTTONDOWN and self.current_person is None:
            nearest = None
            for person_index, person in enumerate(self.people):
                for point_index, point in enumerate(person["points"]):
                    if point is None:
                        continue
                    dx = point[0] * self.scale - x
                    dy = point[1] * self.scale - y
                    distance = dx * dx + dy * dy
                    if distance <= 18 * 18 and (nearest is None or distance < nearest[0]):
                        nearest = (distance, person_index, point_index)
            if nearest is not None:
                _, self.selected_person, self.selected_keypoint = nearest
                self.dragging_point = (self.selected_person, self.selected_keypoint)
        elif event == cv2.EVENT_MOUSEMOVE and self.active_box is not None:
            self.active_box[1] = self.point_from_display(x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging_point is not None:
            px, py = self.point_from_display(x, y)
            person_index, point_index = self.dragging_point
            point = self.people[person_index]["points"][point_index]
            self.people[person_index]["points"][point_index] = (px, py, point[2])
            self.people[person_index].setdefault("reviewed", [True] * 17)[point_index] = True
        elif event == cv2.EVENT_LBUTTONUP and self.active_box is not None:
            self.active_box[1] = self.point_from_display(x, y)
            first, second = self.active_box
            x1, x2 = sorted((max(0, min(self.width - 1, first[0])), max(0, min(self.width - 1, second[0]))))
            y1, y2 = sorted((max(0, min(self.height - 1, first[1])), max(0, min(self.height - 1, second[1]))))
            if x2 - x1 >= 2 and y2 - y1 >= 2:
                self.current_person = {
                    "box": (x1, y1, x2, y2),
                    "points": [None] * 17,
                    "index": 0,
                    "track_id": self.next_track_id,
                    "reviewed": [True] * 17,
                }
                self.next_track_id += 1
                self.drawing_box = False
            self.active_box = None
        elif event == cv2.EVENT_LBUTTONUP and self.dragging_point is not None:
            person_index, point_index = self.dragging_point
            self.people[person_index].setdefault("reviewed", [True] * 17)[point_index] = True
            self.dragging_point = None
        elif event == cv2.EVENT_LBUTTONDOWN and self.current_person is not None:
            index = self.current_person["index"]
            if index < len(KEYPOINTS):
                px, py = self.point_from_display(x, y)
                px = max(0, min(self.width - 1, px))
                py = max(0, min(self.height - 1, py))
                self.current_person["points"][index] = (px, py, self.next_visibility)
                self.current_person["reviewed"][index] = True
                self.current_person["index"] += 1
                if self.current_person["index"] == len(KEYPOINTS):
                    self.finish_person()
                self.next_visibility = 2

    def finish_person(self):
        self.people.append(self.current_person)
        self.selected_person = len(self.people) - 1
        self.selected_keypoint = 0
        self.current_person = None

    def mark_unknown(self):
        if self.current_person is not None:
            self.current_person["index"] += 1
            if self.current_person["index"] == len(KEYPOINTS):
                self.finish_person()
        elif self.selected_person is not None and self.selected_keypoint is not None:
            person = self.people[self.selected_person]
            person["points"][self.selected_keypoint] = None
            person.setdefault("reviewed", [True] * 17)[self.selected_keypoint] = True
        self.next_visibility = 2

    def mark_occluded(self):
        if self.selected_person is not None and self.selected_keypoint is not None:
            person = self.people[self.selected_person]
            point = person["points"][self.selected_keypoint]
            if point is not None:
                person["points"][self.selected_keypoint] = (point[0], point[1], 1)
                person.setdefault("reviewed", [True] * 17)[self.selected_keypoint] = True
        else:
            self.next_visibility = 1

    def select_person(self, offset):
        if self.people:
            current = self.selected_person if self.selected_person is not None else 0
            self.selected_person = (current + offset) % len(self.people)
            self.selected_keypoint = 0

    def select_keypoint(self, offset):
        if self.selected_person is not None:
            self.selected_keypoint = ((self.selected_keypoint or 0) + offset) % 17

    def accept_selected_person(self):
        if self.selected_person is not None:
            self.people[self.selected_person]["reviewed"] = [True] * 17

    def undo_point(self):
        if self.current_person is not None and self.current_person["index"] > 0:
            self.current_person["index"] -= 1
            self.current_person["points"][self.current_person["index"]] = None

    def render(self, frame_number, frame_total):
        canvas = cv2.resize(self.image, None, fx=self.scale, fy=self.scale)
        for person_index, person in enumerate(self.people + ([self.current_person] if self.current_person else [])):
            if person is None:
                continue
            x1, y1, x2, y2 = person["box"]
            cv2.rectangle(canvas, (int(x1 * self.scale), int(y1 * self.scale)),
                          (int(x2 * self.scale), int(y2 * self.scale)), (40, 220, 80), 2)
            cv2.putText(canvas, f"ID {person['track_id']}",
                        (int(x1 * self.scale), max(14, int(y1 * self.scale) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 220, 80), 2)
            points = person["points"]
            for first, second in SKELETON:
                if points[first] and points[second]:
                    cv2.line(canvas,
                             tuple(int(c * self.scale) for c in points[first][:2]),
                             tuple(int(c * self.scale) for c in points[second][:2]),
                             (0, 210, 255), 2)
            for index, point in enumerate(points):
                if point:
                    x, y, _ = point
                    location = (int(x * self.scale), int(y * self.scale))
                    reviewed = person.get("reviewed", [True] * 17)[index]
                    color = (0, 210, 255) if person.get("draft") and not reviewed else (0, 0, 255)
                    if person_index == self.selected_person and index == self.selected_keypoint:
                        cv2.circle(canvas, location, 9, (255, 255, 0), 2)
                    cv2.circle(canvas, location, 4, color, -1)
                    cv2.putText(canvas, str(index + 1), location, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 1)
        if self.active_box and self.active_box[1]:
            first, second = self.active_box
            cv2.rectangle(canvas, tuple(int(c * self.scale) for c in first),
                          tuple(int(c * self.scale) for c in second), (40, 220, 80), 2)

        instructions = [
            f"Sample {frame_number + 1}/{frame_total} | people: {len(self.people)} | tracks: {','.join(str(p['track_id']) for p in self.people)}",
            "Orange = draft. Inspect then e accepts person; drag a dot to fix. [ ] person, , . point",
            "r: repair missing with click | x: unknown | v: occluded | t: assign track ID",
            "a: add missed person | b: remove selected | s: save | n: skip | q: quit",
            "Issue tags: 1 jitter 2 misplaced 3 missing 4 left/right 5 occlusion 6 multi 7 fast 8 viewpoint",
            "COCO left/right means the person's anatomical sides, not screen-left/screen-right.",
        ]
        if self.drawing_box:
            instructions[1] = "Drag left mouse to draw a person box"
        elif self.current_person:
            index = self.current_person["index"]
            if index < len(KEYPOINTS):
                instructions[1] = f"Click point {index + 1}/17: {KEYPOINTS[index]}"
        elif self.selected_person is not None and self.selected_keypoint is not None:
            instructions[1] = (f"Selected ID {self.people[self.selected_person]['track_id']}: "
                               f"{self.selected_keypoint + 1}/17 {KEYPOINTS[self.selected_keypoint]}")
        tags = " ".join(f"{key}:{'x' if issue in self.issue_tags else '-'}" for key, issue in ISSUES.items())
        instructions.append(f"Tags [{tags}]")
        for line, text in enumerate(instructions):
            cv2.putText(canvas, text, (8, 20 + line * 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 0), 3)
            cv2.putText(canvas, text, (8, 20 + line * 21), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)
        return canvas


def write_manifests(dataset_root):
    for split in ("train", "val"):
        images_dir = dataset_root / "images" / split
        labels_dir = dataset_root / "labels" / split
        manifest = dataset_root / "images" / f"{split}.txt"
        files = sorted(images_dir.glob("*.jpg")) if images_dir.exists() else []
        labeled = [image for image in files if (labels_dir / f"{image.stem}.txt").is_file()]
        manifest.write_text("".join(f"./images/{split}/{image.name}\n" for image in labeled), encoding="utf-8")
        print(f"{split}: {len(labeled)} annotated frames -> {manifest}")


def main():
    parser = argparse.ArgumentParser(description="Extract video frames and annotate COCO-17 body keypoints")
    parser.add_argument("--video", default=str(ROOT / "media" / "videos" / "demo" / "yBZ0Y2t0ceo_480p.mp4"))
    parser.add_argument("--dataset", default=str(ROOT / "data" / "datasets" / "custom_pose"))
    parser.add_argument("--sample-seconds", type=float, default=0.5)
    parser.add_argument("--start-sec", type=float, default=90.0)
    parser.add_argument("--end-sec", type=float, default=180.0, help="-1 means the end of the video")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--weights", default=str(ROOT / "weights" / "kapao_s_coco.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--manual-only", action="store_true", help="do not use KAPAO draft suggestions")
    parser.add_argument("--overwrite", action="store_true", help="allow editing frames that already have labels")
    args = parser.parse_args()

    video_path = Path(args.video).resolve()
    dataset_root = Path(args.dataset).resolve()
    if not video_path.is_file():
        raise SystemExit(f"Video not found: {video_path}")
    if args.sample_seconds <= 0 or args.start_sec < 0 or args.end_sec == 0 or args.end_sec < -1:
        raise SystemExit("sample-seconds must be positive; start-sec nonnegative; end-sec positive or -1")
    if args.end_sec != -1 and args.end_sec <= args.start_sec:
        raise SystemExit("end-sec must be later than start-sec")
    if not 0 < args.val_fraction < 0.5:
        raise SystemExit("val-fraction must be between 0 and 0.5")
    if args.imgsz < 32:
        raise SystemExit("imgsz must be at least 32")

    predictor = None if args.manual_only else KapaoDraftPredictor(args.weights, args.device, args.imgsz)

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    interval = max(1, round(fps * args.sample_seconds))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    start_frame = min(total_frames, round(args.start_sec * fps))
    end_frame = total_frames if args.end_sec == -1 else min(total_frames, round(args.end_sec * fps))
    sample_count = math.ceil(max(0, end_frame - start_frame) / interval)
    val_start = int(sample_count * (1 - args.val_fraction))
    print(f"Video: {video_path.name}, {sample_count} sampled frames from {args.start_sec:g}s to {args.end_sec:g}s, one every {args.sample_seconds:g}s")
    print(f"Temporal split: {val_start} train samples, {sample_count - val_start} validation samples")

    for split in ("train", "val"):
        (dataset_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    samples = []
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame
    sample_id = 0
    while frame_id < end_frame:
        ok, frame = capture.read()
        if not ok:
            break
        if frame_id % interval == 0:
            split = "val" if sample_id >= val_start else "train"
            filename = f"{video_path.stem}_{frame_id:08d}.jpg"
            image_path = dataset_root / "images" / split / filename
            write_image(image_path, frame)
            samples.append((split, image_path, frame_id))
            sample_id += 1
        frame_id += 1
    capture.release()

    window = "KAPAO video keypoint labeling"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    annotator = None
    current_sample = None
    selected_track_id = 1
    next_track_id = 1
    previous_drafts = []
    cv2.setMouseCallback(window, lambda event, x, y, flags, data: annotator.mouse(event, x, y, flags, data)
                         if annotator is not None else None)

    for index, (split, image_path, frame_number) in enumerate(samples):
        label_path = dataset_root / "labels" / split / f"{image_path.stem}.txt"
        metadata_path = label_path.with_suffix(".json")
        if label_path.exists() and metadata_path.exists() and not args.overwrite:
            continue
        image = read_image(image_path)
        if image is None:
            raise SystemExit(f"Could not decode sampled frame: {image_path}")
        saved_people, saved_issues = load_saved_people(
            label_path, image.shape[1], image.shape[0], metadata_path
        )
        if predictor is not None and not label_path.exists():
            draft_people = predictor.detect(image)
            draft_people, next_track_id = carry_track_ids(previous_drafts, draft_people, next_track_id)
            previous_drafts = draft_people
            annotator = FrameAnnotator(image, selected_track_id, draft_people)
        else:
            annotator = FrameAnnotator(image, selected_track_id, saved_people)
        annotator.issue_tags = saved_issues
        if annotator.people:
            annotator.selected_person = 0
            annotator.selected_keypoint = 0
        current_sample = index
        while True:
            cv2.imshow(window, annotator.render(index, len(samples)))
            key = cv2.waitKey(20) & 0xFF
            if key == ord("a") and annotator.current_person is None:
                annotator.drawing_box = True
            elif key == ord("v"):
                annotator.mark_occluded()
            elif key == ord("e"):
                annotator.accept_selected_person()
            elif key == ord("x"):
                annotator.mark_unknown()
            elif key == ord("r") and annotator.selected_person is not None and annotator.selected_keypoint is not None:
                annotator.placing_missing = True
            elif key == ord("]"):
                annotator.select_person(1)
            elif key == ord("["):
                annotator.select_person(-1)
            elif key == ord("."):
                annotator.select_keypoint(1)
            elif key == ord(","):
                annotator.select_keypoint(-1)
            elif key == ord("u"):
                annotator.undo_point()
            elif key == ord("b") and annotator.people:
                selected = annotator.selected_person if annotator.selected_person is not None else len(annotator.people) - 1
                annotator.people.pop(selected)
                annotator.selected_person = min(selected, len(annotator.people) - 1) if annotator.people else None
                annotator.selected_keypoint = 0 if annotator.people else None
            elif key == ord("t"):
                value = input("Track ID for selected person (reuse it in later frames): ").strip()
                if value.isdigit() and int(value) > 0:
                    selected_track_id = int(value)
                    if annotator.selected_person is not None:
                        annotator.people[annotator.selected_person]["track_id"] = selected_track_id
                    elif annotator.current_person is not None:
                        annotator.current_person["track_id"] = selected_track_id
                    annotator.next_track_id = selected_track_id + 1
                else:
                    print("Track ID must be a positive integer")
            elif chr(key) in ISSUES:
                issue = ISSUES[chr(key)]
                if issue in annotator.issue_tags:
                    annotator.issue_tags.remove(issue)
                else:
                    annotator.issue_tags.add(issue)
            elif key == ord("s"):
                if annotator.current_person is not None:
                    print("Finish all 17 points (use x for unknown points) or cancel before saving.")
                    continue
                unreviewed = sum(
                    not reviewed
                    for person in annotator.people
                    for reviewed in person.get("reviewed", [True] * 17)
                )
                if unreviewed:
                    print(f"Review {unreviewed} orange points, press e to accept a checked person's proposal, or mark points unknown.")
                    continue
                rows = []
                for person in annotator.people:
                    rows.append(person_to_yolo_label(person, annotator.width, annotator.height))
                label_path.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
                metadata = {
                    "video": video_path.name,
                    "frame_index": frame_number,
                    "timestamp_sec": round(frame_number / fps, 3),
                    "issues": sorted(annotator.issue_tags),
                    "people": [
                        {
                            "track_id": person["track_id"],
                            "bbox_xyxy": list(person["box"]),
                            "keypoints_xy_visibility": person["points"],
                        }
                        for person in annotator.people
                    ],
                }
                metadata_path.write_text(json.dumps(metadata, ensure_ascii=True), encoding="utf-8")
                print(f"Saved {label_path.relative_to(dataset_root)} ({len(rows)} people)")
                break
            elif key == ord("n"):
                print(f"Skipped {image_path.name}")
                break
            elif key == ord("q"):
                current_sample = None
                break
        if current_sample is None:
            break

    cv2.destroyAllWindows()
    write_manifests(dataset_root)
    print("Dataset config: data/custom-kp.yaml")
    print("Issue tags and person track IDs are saved as adjacent JSON metadata; KAPAO training uses the COCO-format TXT labels.")
    print("Skipped frames are excluded; saving an empty frame explicitly creates a negative example.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())