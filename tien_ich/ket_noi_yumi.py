"""TCP link that feeds wrist targets to one YuMi arm running robot/HandFollow.mod.

The robot task is the TCP server. It sends "R\\n" whenever it is ready for the
next motion step and this link answers with the newest target only, so stale
targets never queue up on the controller. "R,G\\n" opts into a fourth gripper
field: -1 HOLD, 1 OPEN, 0 CLOSED. Legacy "R\\n" keeps the three-field reply.
"""

import math
import socket
import threading
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Callable, Optional, Tuple

Target = Optional[Tuple[float, float]]
HOLD_MESSAGE = b"0,0.0,0.0\n"
MAX_REQUEST_BYTES = 64


def utc_timestamp():
    """UTC wall-clock time for frame/send diagnostics, not target-age timing."""
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _normalize_gripper(command):
    # Reject booleans, strings and fractional values rather than coercing an
    # uncertain command into an actuator request.
    if isinstance(command, bool) or not isinstance(command, (int, float)):
        return None
    return int(command) if command in (0, 1) else None


def _normalize_target(target) -> Target:
    """Copy a finite numeric pair; malformed input requests HOLD."""
    if target is None or isinstance(target, (str, bytes, bytearray, Mapping)):
        return None
    try:
        if len(target) != 2:
            return None
        x, y = (float(value) for value in target)
    except (TypeError, ValueError, OverflowError):
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return x, y


def format_target(target: Target, gripper=None, *, include_gripper=False) -> bytes:
    """Encode a legacy XY reply, optionally adding the negotiated gripper field."""
    target = _normalize_target(target)
    if target is None:
        message = HOLD_MESSAGE
        gripper = None
    else:
        x, y = target
        message = f"1,{x:.1f},{y:.1f}\n".encode("ascii")
    if include_gripper:
        command = _normalize_gripper(gripper)
        return message[:-1] + f",{command if command is not None else -1}\n".encode("ascii")
    return message


class YumiArmLink:
    def __init__(
        self,
        name: str,
        host: str,
        port: int,
        max_age: float = 0.3,
        connect_timeout: float = 2.0,
        retry_delay: float = 1.0,
        on_send: Optional[Callable[[dict], None]] = None,
    ) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.max_age = max_age
        self.connect_timeout = connect_timeout
        self.retry_delay = retry_delay
        self.on_send = on_send
        self.connected = False
        self._lock = threading.Lock()
        self._target: Target = None
        self._gripper = None
        self._context = {}
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"yumi-{name}", daemon=True)

    def start(self) -> "YumiArmLink":
        self._thread.start()
        return self

    def set_target(self, target: Target, context=None, *, gripper=None) -> None:
        """Publish an immutable target and its frame metadata, or HOLD.

        ``context`` is a flat diagnostic mapping. It describes the generated
        target, whereas a later SEND event describes the bytes actually sent.
        """
        target = _normalize_target(target)
        context = dict(context) if isinstance(context, Mapping) else {}
        with self._lock:
            self._target = target
            self._gripper = _normalize_gripper(gripper) if target is not None else None
            self._context = context
            self._stamp = time.monotonic()

    def _snapshot_message(self, request_valid=True, include_gripper=False):
        with self._lock:
            target, stamp = self._target, self._stamp
            gripper = self._gripper
            context = self._context.copy()
        # This prevents publishing a stale motion target; it does not cancel
        # a MoveL already accepted by the controller.
        if not request_valid or time.monotonic() - stamp > self.max_age:
            target = None
        return format_target(target, gripper, include_gripper=include_gripper), context


    def _report_send(self, message: bytes, context: dict) -> None:
        if self.on_send is None:
            return
        send_timestamp = utc_timestamp()
        fields = message.decode("ascii").strip().split(",")
        valid, x, y = fields[:3]
        event = context.copy()
        event.setdefault("timestamp", send_timestamp)
        event.setdefault("data_source", "CAMERA_UNVERIFIED")
        event.update(
            event_type="SEND",
            arm=self.name,
            sent_valid=valid == "1",
            x_sent_mm=float(x),
            y_sent_mm=float(y),
            send_timestamp=send_timestamp,
            tcp_connected=True,
            robot_feedback="NOT_AVAILABLE",
            gripper_protocol=len(fields) == 4,
            gripper_sent=({"1": "OPEN", "0": "CLOSED"}.get(fields[3], "HOLD")
                          if len(fields) == 4 else "HOLD"),
        )
        # Diagnostics are optional and cannot terminate the control worker.
        # Callers should use a nonblocking observer (the CSV logger queues).
        try:
            self.on_send(event)
        except Exception:
            pass

    def close(self) -> None:
        self.set_target(None)
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=3)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                with socket.create_connection(
                    (self.host, self.port), timeout=self.connect_timeout
                ) as sock:
                    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sock.settimeout(0.5)
                    self.connected = True
                    self._serve(sock)
            except OSError:
                pass
            finally:
                self.connected = False
            self._stop.wait(self.retry_delay)

    def _serve(self, sock: socket.socket) -> None:
        pending = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(64)
            except socket.timeout:
                continue
            if not chunk:
                return
            pending += chunk
            while b"\n" in pending:
                request, pending = pending.split(b"\n", 1)
                if len(request) > MAX_REQUEST_BYTES:
                    raise OSError("TCP request exceeds maximum line length")
                message, context = self._snapshot_message(
                    request_valid=request in (b"R", b"R\r", b"R,G", b"R,G\r"),
                    include_gripper=request in (b"R,G", b"R,G\r"),
                )
                sock.sendall(message)
                # A successful sendall means bytes were handed to the socket,
                # not that ABB executed the target or returned robot position.
                self._report_send(message, context)
            if len(pending) > MAX_REQUEST_BYTES:
                raise OSError("TCP request exceeds maximum line length")
