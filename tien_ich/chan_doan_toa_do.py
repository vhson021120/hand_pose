"""Optional coordinate diagnostics; all file I/O runs on a bounded writer thread.

FRAME records describe application decisions. SEND records are emitted by the
TCP observer only after sendall succeeds; neither record confirms robot motion.
"""

import csv
import math
from pathlib import Path
import queue
import sys
import threading

import cv2
import numpy as np

from tien_ich.ket_noi_yumi import utc_timestamp
from tien_ich.theo_doi_co_tay_2d import WRIST_SPECS, wrist_to_workspace_mm


NOT_AVAILABLE = "NOT_AVAILABLE"
CSV_FIELDS = (
    "timestamp", "frame_id", "arm", "raw_x_pixel", "raw_y_pixel",
    "filtered_x_pixel", "filtered_y_pixel", "x_workspace_mm", "y_workspace_mm",
    "accepted_x_pixel", "accepted_y_pixel", "accepted_x_mm", "accepted_y_mm",
    "delta_x_mm", "delta_y_mm", "movement_detected", "movement_initialized",
    "movement_threshold_mm",
    "raw_inside_roi", "filtered_inside_roi", "armed", "target_valid",
    "x_target_mm", "y_target_mm", "tcp_connected", "x_sent_mm", "y_sent_mm",
    "send_timestamp", "event_type", "wrist", "sent_valid", "robot_feedback",
    "data_source", "frame_resized",
    "gripper_gesture", "gripper_command", "gripper_reason",
    "gripper_protocol", "gripper_sent",
    "gripper_model_gesture", "gripper_model_confidence",
)


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
        for field in ("x_sent_mm", "y_sent_mm", "send_timestamp", "sent_valid",
                      "gripper_protocol", "gripper_sent"):
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
        # Detector startup temporarily changes cwd for Windows native code.
        self.path = Path(path).resolve()
        self._queue = queue.Queue(maxsize=queue_size)
        self._state_lock = threading.Lock()
        self._closing = threading.Event()
        self._error = None
        self._dropped_rows = 0
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
    canvas = np.full((max(720, 160 + 360 * len(rows)), 1120, 3),
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
        y = 135 + index * 360
        armed = row.get("armed") is True
        valid = row.get("target_valid") is True
        connected = row.get("tcp_connected") is True
        text(f"{_display(row.get('wrist'))} -> {_display(row.get('arm'))} | "
             f"{'ARMED' if armed else 'HOLD'} | "
             f"TCP {'CONNECTED' if connected else 'DISCONNECTED'} | "
             f"SOURCE: {_display(row.get('data_source'))}", y, (90, 220, 170))
        text("RAW pixel: " + _pair(row, "raw_x_pixel", "raw_y_pixel"), y + 32)
        text("EMA pixel: " + _pair(row, "filtered_x_pixel", "filtered_y_pixel"), y + 64)
        text("WORKSPACE mm (EMA candidate): " +
             _pair(row, "x_workspace_mm", "y_workspace_mm"), y + 96)
        text("ACCEPTED pixel: " + _pair(row, "accepted_x_pixel", "accepted_y_pixel"), y + 128)
        text("ACCEPTED mm: " + _pair(row, "accepted_x_mm", "accepted_y_mm"), y + 160)
        accepted_values = (row.get("accepted_x_mm"), row.get("accepted_y_mm"))
        has_accepted = all(isinstance(value, (int, float, np.integer, np.floating))
                           and not isinstance(value, (bool, np.bool_))
                           and math.isfinite(value) for value in accepted_values)
        if not has_accepted:
            movement_status = "NO_POINT"
        elif row.get("movement_initialized") is True:
            movement_status = "INIT"
        elif row.get("movement_detected") is True:
            movement_status = "MOVE"
        else:
            movement_status = "HOLD"
        text("DELTA mm: " + _pair(row, "delta_x_mm", "delta_y_mm") +
             f" | {movement_status} | axis threshold > " +
             _display(row.get("movement_threshold_mm")) + " mm", y + 192,
             (0, 190, 255), scale=0.50)
        text(f"ROI raw: {_display(row.get('raw_inside_roi'))} | "
             f"filtered: {_display(row.get('filtered_inside_roi'))} | "
             f"frame resized: {_display(row.get('frame_resized'))}", y + 224)
        expected = _pair(row, "x_target_mm", "y_target_mm") if valid else "HOLD (0,0.0,0.0)"
        text("EXPECTED TCP TARGET mm: " + expected, y + 256, (0, 190, 255))
        text("LAST OBSERVED SEND mm: " + _pair(row, "x_sent_mm", "y_sent_mm") +
             " | " + _display(row.get("send_timestamp")), y + 288, scale=0.48)
        text("FRAME " + _display(row.get("frame_id")) + " | " +
             _display(row.get("timestamp")), y + 318, (160, 180, 200), scale=0.48)
    return canvas


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)


def report_diagnostic_warning(message):
    """Optional logging failures must not interrupt processing or cleanup."""
    try:
        print(message, file=sys.stderr)
    except (UnicodeError, OSError, ValueError):
        pass


def draw_webcam_overlay(frame, wrists, hands):
    output = frame.copy()
    wrist_colors = dict(WRIST_SPECS)
    for side, (x, y, _) in wrists.items():
        color = wrist_colors[side]
        point = (int(round(x)), int(round(y)))
        cv2.circle(output, point, 9, color, 2, cv2.LINE_AA)
        cv2.putText(output, f"{side} WRIST", (point[0] + 10, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    for hand in hands:
        landmarks = np.asarray(hand.landmarks)
        if landmarks.shape != (21, 3) or not np.isfinite(landmarks[:, :2]).all():
            continue
        color = (70, 220, 110) if hand.is_left else (220, 90, 220)
        for first, second in HAND_CONNECTIONS:
            start = tuple(np.rint(landmarks[first, :2]).astype(int))
            end = tuple(np.rint(landmarks[second, :2]).astype(int))
            cv2.line(output, start, end, color, 2, cv2.LINE_AA)
        for x, y in landmarks[:, :2]:
            cv2.circle(output, (int(round(x)), int(round(y))), 3, color, -1, cv2.LINE_AA)
    return output


def draw_workspace_rect(frame, roi):
    p1 = (int(round(roi[0])), int(round(roi[1])))
    p2 = (int(round(roi[2])), int(round(roi[3])))
    cv2.rectangle(frame, p1, p2, (0, 200, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, "WORKSPACE", (p1[0] + 6, p1[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1, cv2.LINE_AA)


def draw_gripper_status(frame, rows):
    """Show requested signals and source hand; these are not robot feedback."""
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 82), (20, 27, 35), -1)
    cv2.putText(frame, "GRIPPER COMMANDS (not feedback)", (12, 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (220, 225, 230), 1, cv2.LINE_AA)
    for index, row in enumerate(rows):
        command = row["gripper_command"]
        signal = {1: "OPEN(1)", 0: "CLOSE(0)"}.get(command, "HOLD(-1)")
        text = (f"{row['wrist']} -> {row['arm']} | HAND {row['gripper_gesture']} | "
                f"CMD {signal} | {row['gripper_reason']}")
        score = row.get("gripper_model_confidence")
        if isinstance(score, (int, float)) and np.isfinite(score):
            text += f" | model {row.get('gripper_model_gesture', 'None')} {score:.0%}"
        color = (70, 220, 110) if row["wrist"] == "LEFT" else (220, 90, 220)
        cv2.putText(frame, text, (12, 44 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def wrist_motion_label(row):
    if row.get("accepted_x_mm") in (None, "NOT_AVAILABLE"):
        return "NO POINT"
    if row.get("movement_initialized"):
        return "INIT"
    return "MOVE" if row.get("movement_detected") else "HOLD"


def draw_wrist_motion_status(frame, rows):
    cv2.rectangle(frame, (0, 82), (frame.shape[1], 140), (20, 27, 35), -1)
    for index, row in enumerate(rows):
        delta = ""
        if row["delta_x_mm"] != "NOT_AVAILABLE":
            delta = f" dX {row['delta_x_mm']:+.2f} mm dY {row['delta_y_mm']:+.2f} mm |"
        text = (f"{row['wrist']} WRIST |{delta} {wrist_motion_label(row)} | "
                f"threshold > {row['movement_threshold_mm']:g} mm")
        color = (70, 220, 110) if row["wrist"] == "LEFT" else (220, 90, 220)
        cv2.putText(frame, text, (12, 104 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def draw_robot_status(canvas, armed, links):
    if armed:
        text, color = "ROBOT: ARMED  [SPACE = STOP]", (60, 60, 255)
    else:
        text, color = "ROBOT: HOLD  [SPACE = START]", (0, 185, 255)
    cv2.putText(canvas, text, (560, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
    parts = [f"{link.name} {'OK' if link.connected else '--'}" for link in links.values()]
    cv2.putText(canvas, "   ".join(parts), (560, 77), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, (165, 181, 194), 1, cv2.LINE_AA)


def draw_wrist_xy_dashboard(wrists, roi, work_mm, pose_count, motion_rows=None,
                            gripper_rows=None):
    height = 802 if gripper_rows is not None else 720
    canvas = np.full((height, 960, 3), (20, 27, 35), dtype=np.uint8)
    max_width, max_height = 776, 484
    fit = min(max_width / work_mm[0], max_height / work_mm[1])
    plot_width, plot_height = int(work_mm[0] * fit), int(work_mm[1] * fit)
    plot_left = 92 + (max_width - plot_width) // 2
    plot_top = 116 + (max_height - plot_height) // 2
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height
    center_x = plot_left + plot_width // 2
    center_y = plot_top + plot_height // 2

    if pose_count == 0:
        status = "NO POSE DETECTED"
        status_color = (0, 185, 255)
    else:
        status = "TRACKING"
        status_color = (100, 235, 150)
    cv2.putText(canvas, "WRIST XY | MILLIMETERS", (28, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.78, (240, 244, 248), 2, cv2.LINE_AA)
    cv2.putText(canvas, status, (28, 77), cv2.FONT_HERSHEY_SIMPLEX,
                0.52, status_color, 1, cv2.LINE_AA)

    for division in range(5):
        x = plot_left + division * plot_width // 4
        y = plot_top + division * plot_height // 4
        color = (80, 105, 120) if division == 2 else (44, 58, 70)
        thickness = 2 if division == 2 else 1
        cv2.line(canvas, (x, plot_top), (x, plot_bottom), color, thickness)
        cv2.line(canvas, (plot_left, y), (plot_right, y), color, thickness)
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_right, plot_bottom), (80, 96, 110), 1)
    cv2.circle(canvas, (center_x, center_y), 4, (235, 240, 245), -1, cv2.LINE_AA)
    cv2.putText(canvas, "O", (center_x + 8, center_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (235, 240, 245), 1, cv2.LINE_AA)
    cv2.putText(canvas, "X+", (plot_right - 28, center_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (145, 190, 220), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Y+", (center_x + 8, plot_top + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (145, 220, 170), 1, cv2.LINE_AA)

    value_lines = []
    motion_by_side = {row["wrist"]: row for row in motion_rows or ()}
    for side, color in WRIST_SPECS:
        wrist = wrists.get(side)
        if wrist is None:
            value_lines.append(f"{side}: NOT DETECTED")
            continue
        x_mm, y_mm, inside = wrist_to_workspace_mm(wrist, roi, work_mm)
        u = x_mm / work_mm[0] + 0.5
        v = 0.5 - y_mm / work_mm[1]
        dot_x = plot_left + int(u * plot_width)
        dot_y = plot_top + int(v * plot_height)
        cv2.circle(canvas, (dot_x, dot_y), 9, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (dot_x, dot_y), 14, color, 2, cv2.LINE_AA)
        suffix = "" if inside else "  (OUT OF WORKSPACE)"
        if side in motion_by_side:
            row = motion_by_side[side]
            suffix += (f" | dX {row['delta_x_mm']:+.2f}  dY {row['delta_y_mm']:+.2f} mm"
                       f" | {wrist_motion_label(row)}")
        value_lines.append(f"{side}: X {x_mm:+.1f} mm   Y {y_mm:+.1f} mm{suffix}")

    for index, (text, color) in enumerate(zip(value_lines, ((70, 220, 110), (220, 90, 220)))):
        cv2.putText(canvas, text, (28, 642 + index * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    x_limit = work_mm[0] / 2.0
    y_limit = work_mm[1] / 2.0
    cv2.putText(canvas, f"Workspace {work_mm[0]:g}x{work_mm[1]:g} mm | "
                       f"X: {-x_limit:+.1f}..{x_limit:+.1f} mm | "
                       f"Y: {-y_limit:+.1f}..{y_limit:+.1f} mm",
                (28, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (165, 181, 194), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Origin: workspace center | fixed size and aspect ratio | "
                       "display scale, not physical calibration",
                (28, 698), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (165, 181, 194), 1, cv2.LINE_AA)
    if gripper_rows is not None:
        draw_gripper_status(canvas[720:], gripper_rows)
    return canvas
