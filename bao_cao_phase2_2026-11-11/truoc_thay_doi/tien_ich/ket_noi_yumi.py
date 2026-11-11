"""TCP link that feeds wrist targets to one YuMi arm running robot/HandFollow.mod.

The robot task is the TCP server. It sends "R\\n" whenever it is ready for the
next motion step and this link answers with the newest target only, so stale
targets never queue up on the controller.
"""

import math
import socket
import threading
import time
from typing import Optional, Tuple

Target = Optional[Tuple[float, float]]


def format_target(target: Target) -> bytes:
    """Encode a target as "<valid>,<x_mm>,<y_mm>\\n"; None means hold position."""
    if target is None:
        return b"0,0.0,0.0\n"
    x, y = (float(value) for value in target)
    if not (math.isfinite(x) and math.isfinite(y)):
        return b"0,0.0,0.0\n"
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
    ) -> None:
        self.name = name
        self.host = host
        self.port = port
        self.max_age = max_age
        self.connect_timeout = connect_timeout
        self.retry_delay = retry_delay
        self.connected = False
        self._lock = threading.Lock()
        self._target: Target = None
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"yumi-{name}", daemon=True)

    def start(self) -> "YumiArmLink":
        self._thread.start()
        return self

    def set_target(self, target: Target) -> None:
        """Publish the newest target in workspace mm, or None to hold position."""
        with self._lock:
            self._target = target
            self._stamp = time.monotonic()

    def current_message(self) -> bytes:
        with self._lock:
            target, stamp = self._target, self._stamp
        # If the camera loop stalls, stop the robot instead of replaying old data.
        if time.monotonic() - stamp > self.max_age:
            target = None
        return format_target(target)

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
                _, pending = pending.split(b"\n", 1)
                sock.sendall(self.current_message())
