"""Audit runner: existing tests with line coverage, then supplemental checks.

No production files are changed. Camera access is blocked. TCP destinations
are restricted to explicit IPv4 loopback. All generated evidence stays here.
"""

import contextlib
import hashlib
import importlib.metadata
import importlib.util
import io
import json
from pathlib import Path
import socket
import sys
import threading
import time
import trace
import unittest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True
sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

SOURCES = [
    "run.py", "tien_ich/nhan_dien_co_tay.py",
    "tien_ich/theo_doi_co_tay_2d.py", "tien_ich/ket_noi_yumi.py",
    "robot/HandFollow.mod", "tien_ich/nhan_dien_ban_tay.py",
    "kiem_tra/kiem_tra_nhan_dien_co_tay.py",
    "kiem_tra/kiem_tra_theo_doi_co_tay_2d.py",
    "kiem_tra/kiem_tra_ket_noi_yumi.py", "README.md", "requirements.txt",
]


def hashes():
    return {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in SOURCES}


before = hashes()
environment = {
    "python": sys.version,
    "executable": sys.executable,
    "packages": {name: importlib.metadata.version(name) for name in
                 ("numpy", "opencv-contrib-python", "mediapipe")},
    "coverage_package_available": importlib.util.find_spec("coverage") is not None,
}

import cv2


def camera_forbidden(*args, **kwargs):
    raise AssertionError("Camera access is forbidden during this audit")


cv2.VideoCapture = camera_forbidden
real_create_connection = socket.create_connection


def loopback_only(address, *args, **kwargs):
    if address[0] != "127.0.0.1":
        raise AssertionError("Only explicit 127.0.0.1 is allowed in offline tests")
    return real_create_connection(address, *args, **kwargs)


socket.create_connection = loopback_only
targets = {str((ROOT / name).resolve()): name for name in SOURCES if name.endswith(".py")}
observed = {name: set() for name in targets.values()}


def trace_lines(frame, event, arg):
    name = targets.get(frame.f_code.co_filename)
    if event == "line" and name is not None:
        observed[name].add(frame.f_lineno)
    return trace_lines


def run_suite(suite, label):
    stream = io.StringIO()
    started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    elapsed = time.perf_counter() - started
    output = stream.getvalue()
    (HERE / (label + ".txt")).write_text(output, encoding="utf-8")
    print(output)
    return {"tests": result.testsRun,
            "pass": result.testsRun - len(result.failures) - len(result.errors) - len(result.skipped),
            "fail": len(result.failures), "error": len(result.errors),
            "skipped": len(result.skipped), "elapsed_seconds": elapsed,
            "success": result.wasSuccessful()}


sys.settrace(trace_lines)
threading.settrace(trace_lines)
try:
    original = unittest.defaultTestLoader.discover(
        str(ROOT / "kiem_tra"), pattern="kiem_tra_*.py")
    original_result = run_suite(original, "unit_test_co_san_co_coverage")
finally:
    sys.settrace(None)
    threading.settrace(None)

coverage = {}
for name in SOURCES[:4] + ["tien_ich/nhan_dien_ban_tay.py"]:
    executable = {line for line in trace._find_executable_linenos(str(ROOT / name))
                  if line > 0}  # Python 3.11 can expose synthetic bytecode line 0.
    hits = observed[name] & executable
    coverage[name] = {
        "executable_lines": len(executable), "executed_lines": len(hits),
        "percent": round(100 * len(hits) / len(executable), 2) if executable else 100.0,
        "missing_lines": sorted(executable - hits),
    }

spec = importlib.util.spec_from_file_location("audit_supplemental", HERE / "kiem_thu_bo_sung.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
supplemental = unittest.defaultTestLoader.loadTestsFromModule(module)
supplemental_result = run_suite(supplemental, "unit_test_bo_sung")

after = hashes()
evidence = {
    "environment": environment, "original_with_trace": original_result,
    "supplemental": supplemental_result, "original_line_coverage": coverage,
    "coverage_method": "Python trace executable lines; includes imports and definitions; background threads traced; not branch coverage",
    "sha256_before": before, "sha256_after": after,
    "source_files_unchanged": before == after,
}
(HERE / "bang_chung_kiem_thu.json").write_text(
    json.dumps(evidence, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps({key: evidence[key] for key in
                  ("environment", "original_with_trace", "supplemental", "source_files_unchanged")},
                 indent=2, ensure_ascii=False))
print("Original suite line coverage:")
for name, data in coverage.items():
    print(f"{name}: {data['executed_lines']}/{data['executable_lines']} ({data['percent']}%)")
raise SystemExit(0 if original_result["success"] and supplemental_result["success"] and before == after else 1)
