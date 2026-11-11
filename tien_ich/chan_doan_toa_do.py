"""Optional coordinate diagnostics; all file I/O runs on a bounded writer thread.

FRAME records describe application decisions. SEND records are emitted by the
TCP observer only after sendall succeeds; neither record confirms robot motion.
"""

import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import queue
import threading

import cv2
import numpy as np


NOT_AVAILABLE = "NOT_AVAILABLE"
CSV_FIELDS = (
    "timestamp", "frame_id", "arm", "raw_x_pixel", "raw_y_pixel",
    "filtered_x_pixel", "filtered_y_pixel", "x_workspace_mm", "y_workspace_mm",
    "raw_inside_roi", "filtered_inside_roi", "armed", "target_valid",
    "x_target_mm", "y_target_mm", "tcp_connected", "x_sent_mm", "y_sent_mm",
    "send_timestamp", "event_type", "wrist", "sent_valid", "robot_feedback",
    "data_source", "frame_resized",
)


def utc_timestamp():
    """Wall-clock UTC for CSV correlation, separate from target-age monotonic time."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _csv_value(value):
    if value is None or (isinstance(value, str) and value == ""):
        return NOT_AVAILABLE
    if isinstance(value, (float, np.floating)) and not math.isfinite(value):
        return NOT_AVAILABLE
    if not isinstance(value, (str, bool, int, float, np.generic)):
        return NOT_AVAILABLE
    return value


def _normalise_record(row):
    result = {field: _csv_value(row.get(field)) for field in CSV_FIELDS}
    result["event_type"] = "SEND" if row.get("event_type") == "SEND" else "FRAME"
    if result["timestamp"] == NOT_AVAILABLE:
        result["timestamp"] = utc_timestamp()
    if result["data_source"] == NOT_AVAILABLE:
        result["data_source"] = "CAMERA_UNVERIFIED"
    # The wire protocol has no position acknowledgement. Never infer feedback.
    result["robot_feedback"] = NOT_AVAILABLE
    if result["event_type"] != "SEND":
        for field in ("x_sent_mm", "y_sent_mm", "send_timestamp", "sent_valid"):
            result[field] = NOT_AVAILABLE
    return result


class AsyncDiagnosticLogger:
    """Append UTF-8 CSV without waiting for the disk in record().

    Queue overflow drops the new row and increments dropped_rows. File errors
    are exposed through error and disable further logging; they never propagate
    into camera/TCP callers. close() drains accepted rows when possible and has
    a finite timeout, so a hung disk cannot indefinitely block application exit.
    """

    def __init__(self, path, queue_size=4096):
        if not isinstance(queue_size, int) or isinstance(queue_size, bool) or queue_size < 1:
            raise ValueError("queue_size must be a positive integer")
        self.path = Path(path)
        self._queue = queue.Queue(maxsize=queue_size)
        self._state_lock = threading.Lock()
        self._closing = threading.Event()
        self._error = None
        self._dropped_rows = 0
        self._written_rows = 0
        self._thread = threading.Thread(target=self._write_loop,
                                        name="coordinate-csv", daemon=True)
        self._thread.start()

    @property
    def error(self):
        with self._state_lock:
            return self._error

    @property
    def dropped_rows(self):
        with self._state_lock:
            return self._dropped_rows

    @property
    def written_rows(self):
        with self._state_lock:
            return self._written_rows

    @property
    def closed(self):
        return self._closing.is_set()

    def record(self, row):
        """Publish a shallow snapshot of a flat diagnostic row; return acceptance."""
        try:
            snapshot = dict(row)
        except (TypeError, ValueError):
            with self._state_lock:
                self._dropped_rows += 1
            return False
        with self._state_lock:
            if self._closing.is_set() or self._error is not None:
                self._dropped_rows += 1
                return False
            try:
                self._queue.put_nowait(snapshot)
            except queue.Full:
                self._dropped_rows += 1
                return False
        return True

    def close(self, timeout=2.0):
        """Stop accepting rows; return True only when the writer has finished."""
        if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("close timeout must be finite and non-negative")
        with self._state_lock:
            self._closing.set()
        self._thread.join(timeout=timeout)
        return not self._thread.is_alive()

    def _open_stream(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        return self.path.open("a", encoding="utf-8", newline="")

    def _write_loop(self):
        active_row = False
        try:
            with self._open_stream() as stream:
                writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS,
                                        extrasaction="ignore")
                if stream.tell() == 0:
                    writer.writeheader()
                    stream.flush()
                while True:
                    try:
                        row = self._queue.get(timeout=0.05)
                    except queue.Empty:
                        if self._closing.is_set():
                            return
                        continue
                    active_row = True
                    writer.writerow(_normalise_record(row))
                    stream.flush()
                    active_row = False
                    with self._state_lock:
                        self._written_rows += 1
        except Exception as exc:
            # CSV/filesystem failures are diagnostic errors, not control errors.
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._dropped_rows += self._queue.qsize() + int(active_row)
                self._closing.set()


def _display(value):
    if value is None or (isinstance(value, str) and value == NOT_AVAILABLE):
        return NOT_AVAILABLE
    if isinstance(value, (float, np.floating)):
        return f"{value:.3f}" if math.isfinite(value) else NOT_AVAILABLE
    return str(value)


def _pair(row, first, second):
    return f"({_display(row.get(first))}, {_display(row.get(second))})"


def draw_coordinate_diagnostics(rows, logging_status=""):
    """Render an optional panel only; never opens a camera, socket, or window."""
    rows = list(rows)
    canvas = np.full((max(720, 160 + 260 * len(rows)), 1120, 3),
                     (20, 27, 35), dtype=np.uint8)

    def text(value, y, color=(220, 230, 240), scale=0.55):
        cv2.putText(canvas, str(value), (24, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)

    text("COORDINATE DIAGNOSTICS | EXPECTED TARGETS, NOT ROBOT POSITION", 34,
         scale=0.66)
    text("ROBOT FEEDBACK: NOT_AVAILABLE | workspace mm are uncalibrated", 65,
         (0, 190, 255))
    text(f"CSV: {logging_status or 'OFF'}", 94, (160, 180, 200))
    for index, row in enumerate(rows):
        y = 135 + index * 260
        armed = row.get("armed") is True
        valid = row.get("target_valid") is True
        connected = row.get("tcp_connected") is True
        text(f"{_display(row.get('wrist'))} -> {_display(row.get('arm'))} | "
             f"{'ARMED' if armed else 'HOLD'} | "
             f"TCP {'CONNECTED' if connected else 'DISCONNECTED'} | "
             f"SOURCE: {_display(row.get('data_source'))}", y, (90, 220, 170))
        text("RAW pixel: " + _pair(row, "raw_x_pixel", "raw_y_pixel"), y + 32)
        text("EMA pixel: " + _pair(row, "filtered_x_pixel", "filtered_y_pixel"), y + 64)
        text("WORKSPACE mm: " + _pair(row, "x_workspace_mm", "y_workspace_mm"), y + 96)
        text(f"ROI raw: {_display(row.get('raw_inside_roi'))} | "
             f"filtered: {_display(row.get('filtered_inside_roi'))} | "
             f"frame resized: {_display(row.get('frame_resized'))}", y + 128)
        expected = _pair(row, "x_target_mm", "y_target_mm") if valid else "HOLD (0,0.0,0.0)"
        text("EXPECTED TCP TARGET mm: " + expected, y + 160, (0, 190, 255))
        text("LAST OBSERVED SEND mm: " + _pair(row, "x_sent_mm", "y_sent_mm") +
             " | " + _display(row.get("send_timestamp")), y + 192, scale=0.48)
        text("FRAME " + _display(row.get("frame_id")) + " | " +
             _display(row.get("timestamp")), y + 222, (160, 180, 200), scale=0.48)
    return canvas
