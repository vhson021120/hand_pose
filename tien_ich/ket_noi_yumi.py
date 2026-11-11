"""TCP link that feeds wrist targets to one YuMi arm running robot/HandFollow.mod.

The robot task is the TCP server. It sends "R\\n" whenever it is ready for the
next motion step and this link answers with the newest target only, so stale
targets never queue up on the controller.
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


def format_target(target: Target) -> bytes:
    """Encode a target as "<valid>,<x_mm>,<y_mm>\\n"; None means hold position."""
    target = _normalize_target(target)
    if target is None:
        return HOLD_MESSAGE
    x, y = target
    return f"1,{x:.1f},{y:.1f}\n".encode("ascii")


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
        self._context = {}
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"yumi-{name}", daemon=True)

    def start(self) -> "YumiArmLink":
        self._thread.start()
        return self

    def set_target(self, target: Target, context=None) -> None:
        """Publish an immutable target and its frame metadata, or HOLD.

        ``context`` is a flat diagnostic mapping. It describes the generated
        target, whereas a later SEND event describes the bytes actually sent.
        """
        target = _normalize_target(target)
        context = dict(context) if isinstance(context, Mapping) else {}
        with self._lock:
            self._target = target
            self._context = context
            self._stamp = time.monotonic()

    def _snapshot_message(self, request_valid=True):
        with self._lock:
            target, stamp = self._target, self._stamp
            context = self._context.copy()
        # This prevents publishing a stale motion target; it does not cancel
        # a MoveL already accepted by the controller.
        if not request_valid or time.monotonic() - stamp > self.max_age:
            target = None
        return format_target(target), context

    def current_message(self) -> bytes:
        return self._snapshot_message()[0]

    def _report_send(self, message: bytes, context: dict) -> None:
        if self.on_send is None:
            return
        send_timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        valid, x, y = message.decode("ascii").strip().split(",")
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
        )
        # Diagnostics are optional and cannot terminate the control worker.
        # Callers should use a nonblocking observer (the CSV logger queues).
        try:
            self.on_send(event)
        except Exception:
            pass

    def close(self) -> None:
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
                    request_valid=request in (b"R", b"R\r")
                )
                sock.sendall(message)
                # A successful sendall means bytes were handed to the socket,
                # not that ABB executed the target or returned robot position.
                self._report_send(message, context)
            if len(pending) > MAX_REQUEST_BYTES:
                raise OSError("TCP request exceeds maximum line length")
