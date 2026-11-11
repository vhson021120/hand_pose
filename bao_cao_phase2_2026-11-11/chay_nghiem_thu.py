"""Phase 2 acceptance: cameras forbidden; real TCP only to 127.0.0.1.

Runs original 9 tests, updated 31 supplemental regressions, then Phase 2 tests.
Produces SYNTHETIC CSV via a mocked detector and real loopback socket.
Does not invoke the actual application, load models, or overwrite Phase 1 logs.
"""

import csv
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import time
import unittest
from unittest.mock import Mock
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import cv2
import numpy as np


def forbidden_camera(*args, **kwargs):
    raise AssertionError("No camera may be opened in Phase 2 acceptance")


cv2.VideoCapture = forbidden_camera
# Guard the socket method as well as create_connection; test mocks cannot
# accidentally bypass the real-connection destination restriction.
real_connect = socket.socket.connect
real_connect_ex = socket.socket.connect_ex
real_bind = socket.socket.bind
connections = []


def validate_address(address):
    if not isinstance(address, tuple) or address[0] != "127.0.0.1":
        raise AssertionError(f"Non-loopback network access forbidden: {address!r}")


def guarded_connect(sock, address):
    validate_address(address)
    connections.append(address)
    return real_connect(sock, address)


def guarded_connect_ex(sock, address):
    validate_address(address)
    connections.append(address)
    return real_connect_ex(sock, address)


def guarded_bind(sock, address):
    validate_address(address)
    return real_bind(sock, address)


socket.socket.connect = guarded_connect
socket.socket.connect_ex = guarded_connect_ex
socket.socket.bind = guarded_bind


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


rapid_before = file_hash(ROOT / "robot/HandFollow.mod")
results = {}


def run_suite(label, suite):
    stream = io.StringIO()
    begin = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - begin
    output = stream.getvalue()
    (HERE / (label + ".txt")).write_text(output, encoding="utf-8")
    print(output)
    results[label] = dict(tests=result.testsRun,
                         passed=result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
                         failed=len(result.failures), errors=len(result.errors),
                         skipped=len(result.skipped), seconds=elapsed, success=result.wasSuccessful())


artifacts_only = sys.argv[1:] == ["--artifacts-only"]
if artifacts_only:
    previous = json.loads((HERE / "ket_qua_nghiem_thu.json").read_text(encoding="utf-8"))
    for relative_path, expected_hash in previous["sha256_after"].items():
        if file_hash(ROOT / relative_path) != expected_hash:
            raise RuntimeError("Source changed: full test run required before regenerating artifacts")
    results = previous["results"]
else:
    run_suite("test_goc_9", unittest.TestLoader().discover(str(ROOT / "kiem_tra"), "kiem_tra_*.py"))
    spec = importlib.util.spec_from_file_location("supplemental_phase2", ROOT / "bao_cao_offline_2026-11-11/kiem_thu_bo_sung.py")
    supplemental = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(supplemental)
    run_suite("test_bo_sung_31", unittest.TestLoader().loadTestsFromModule(supplemental))
    run_suite("test_phase2", unittest.TestLoader().discover(str(ROOT / "kiem_tra/phase2"), "kiem_tra_*.py"))

# A real successful sendall to the existing FakeRobotTask, with synthetic
# MediaPipe model output. Never label these events as real robot feedback.
from kiem_tra.kiem_tra_ket_noi_yumi import FakeRobotTask
from run import process_wrist_frame
from tien_ich.chan_doan_toa_do import AsyncDiagnosticLogger, draw_coordinate_diagnostics, utc_timestamp
from tien_ich.ket_noi_yumi import YumiArmLink
from tien_ich.nhan_dien_co_tay import MediaPipeWristDetector
from tien_ich.theo_doi_co_tay_2d import WristXYTracker

sample = HERE / "diagnostic_SYNTHETIC.csv"
# Regeneration is confined to this explicitly named generated artifact.
sample.write_text("", encoding="utf-8")
logger = AsyncDiagnosticLogger(sample)
robot = FakeRobotTask()
link = YumiArmLink("ROB_L", "127.0.0.1", robot.port, max_age=30,
                   retry_delay=0.03, on_send=logger.record)
sample_messages = []
try:
    link.start()
    robot.accept()
    detector = MediaPipeWristDetector.__new__(MediaPipeWristDetector)
    detector.image_module = Mock()
    detector.image_module.ImageFormat.SRGB = 1
    detector.image_module.Image.side_effect = lambda *_args: _args[-1]
    detector.last_timestamp = 0
    detector.min_visibility = 0.7
    detector.landmarker = Mock()
    tracker = WristXYTracker()
    for frame_id, (x, y) in enumerate([(755, 360), (1100, 360), (None, None)], 1):
        landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0) for _ in range(33)]
        if x is not None:
            landmarks[15] = SimpleNamespace(x=x / 1280, y=y / 720, visibility=0.9)
        detector.landmarker.detect_for_video.return_value = SimpleNamespace(pose_landmarks=[landmarks])
        detections = detector.detect(np.zeros((720, 1280, 3), dtype=np.uint8))[0]
        _, _, rows = process_wrist_frame(tracker, detections, (1280, 720), (400, 300), 0.8,
                                         armed=True, frame_id=frame_id, timestamp=utc_timestamp(),
                                         data_source="SYNTHETIC")
        for row in rows:
            row["tcp_connected"] = link.connected if row["arm"] == "ROB_L" else False
            logger.record(row)
            if row["arm"] == "ROB_L":
                target = (row["x_target_mm"], row["y_target_mm"]) if row["target_valid"] else None
                link.set_target(target, context=row)
        sample_messages.append(robot.request())
finally:
    link.close()
    robot.close()
    drained = logger.close()

with sample.open(encoding="utf-8", newline="") as stream:
    sample_rows = list(csv.DictReader(stream))
preview_rows = []
for source_row in sample_rows[:2]:
    row = source_row.copy()
    for key in ("armed", "target_valid", "tcp_connected", "raw_inside_roi",
                "filtered_inside_roi", "frame_resized"):
        row[key] = row[key] == "True"
    preview_rows.append(row)
preview_written, preview_bytes = cv2.imencode(
    ".png", draw_coordinate_diagnostics(preview_rows, "SYNTHETIC CSV preview"))
if preview_written:
    # pathlib supports this Windows workspace's Vietnamese path; imwrite may not.
    (HERE / "diagnostic_SYNTHETIC.png").write_bytes(preview_bytes.tobytes())
sample_valid = (drained and logger.error is None and logger.dropped_rows == 0
                and len(sample_rows) == 9
                and all(row["data_source"] == "SYNTHETIC" and row["robot_feedback"] == "NOT_AVAILABLE"
                        for row in sample_rows)
                and sample_messages == ["1,59.9,0.0\n", "0,0.0,0.0\n", "0,0.0,0.0\n"])
files = [ROOT / "run.py", *sorted((ROOT / "tien_ich").glob("*.py")),
         ROOT / "bao_cao_offline_2026-11-11/kiem_thu_bo_sung.py",
         *sorted((ROOT / "kiem_tra/phase2").glob("*.py")), ROOT / "robot/HandFollow.mod"]
evidence = dict(results=results, environment={
    "python": sys.version,
    "packages": {name: importlib.metadata.version(name) for name in
                 ("numpy", "opencv-contrib-python", "mediapipe")}},
    sample_csv_valid=sample_valid, sample_csv_rows=len(sample_rows),
    preview_png_valid=bool(preview_written), test_results_reused_after_hash_check=artifacts_only,
    sample_tcp_messages=sample_messages,
    network_destinations=connections,
    non_loopback_connections=sum(address[0] != "127.0.0.1" for address in connections),
    camera_access="BLOCKED_BY_TEST_GUARD", robot_feedback="NOT_AVAILABLE",
    rapid_unchanged=rapid_before == file_hash(ROOT / "robot/HandFollow.mod"),
    sha256_after={str(path.relative_to(ROOT)).replace("\\", "/"): file_hash(path) for path in files})
(HERE / "ket_qua_nghiem_thu.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({key: evidence[key] for key in
                  ("results", "sample_csv_valid", "sample_csv_rows", "rapid_unchanged", "non_loopback_connections")},
                 ensure_ascii=False, indent=2))
raise SystemExit(0 if all(row["success"] for row in results.values()) and sample_valid
                 and preview_written and evidence["rapid_unchanged"] else 1)
