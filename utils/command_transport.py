import queue
import socket
import threading
import time
from typing import Optional, Tuple


class TcpCommandSender:
    """Send one newline-terminated command per TCP connection and await an ACK."""

    def __init__(self, host: str, port: int, timeout: float = 1.0,
                 max_command_age: float = 2.0) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.max_command_age = max_command_age
        self._commands: "queue.Queue[Tuple[str, float]]" = queue.Queue(maxsize=8)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, name="abb-command-sender", daemon=True)
        self._worker.start()

    def submit(self, command: str) -> bool:
        try:
            self._commands.put_nowait((command, time.monotonic()))
        except queue.Full:
            print(f"[TCP ERROR] Command queue is full; rejected {command}.")
            return False
        return True

    def _send(self, command: str) -> None:
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as connection:
            connection.settimeout(self.timeout)
            connection.sendall((command + "\n").encode("ascii"))
            with connection.makefile("r", encoding="ascii", newline="\n") as response_stream:
                response = response_stream.readline().strip()

        if not response:
            print(f"[TCP ERROR] No ACK received for {command}.")
        elif response.upper() == "ACK" or response.upper() == f"ACK {command}":
            print(f"[TCP ACK] {command}")
        else:
            print(f"[TCP REJECTED] {command}: {response}")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                item: Optional[Tuple[str, float]] = self._commands.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                return

            command, queued_at = item
            if time.monotonic() - queued_at > self.max_command_age:
                print(f"[TCP ERROR] Dropped stale command {command}.")
                continue
            try:
                self._send(command)
            except (OSError, UnicodeError) as exc:
                print(f"[TCP ERROR] Could not send {command} to {self.host}:{self.port}: {exc}")

    def close(self) -> None:
        self._stop.set()
        self._worker.join(timeout=self.timeout * 2 + 0.5)
