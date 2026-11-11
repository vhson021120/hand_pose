"""Phase 2 TCP tests: synthetic data and socket mocks, never a controller."""

import socket
import unittest
from unittest.mock import Mock, patch

from tien_ich.ket_noi_yumi import (
    HOLD_MESSAGE,
    MAX_REQUEST_BYTES,
    YumiArmLink,
    format_target,
)


class TcpPhase2Test(unittest.TestCase):
    def make_link(self, observer=None, max_age=30):
        # These tests do not start the link. _serve receives a socket mock.
        return YumiArmLink("ROB_L", "127.0.0.1", 1,
                           max_age=max_age, on_send=observer)

    def make_socket(self, chunks):
        sock = Mock()
        sock.recv.side_effect = list(chunks) + [b""]
        return sock

    def test_malformed_targets_become_hold(self):
        for target in [None, (), (1,), (1, 2, 3), ("bad", 2),
                       "12", b"12", {1: 2, 3: 4}, 12, True,
                       (float("nan"), 1), (1, float("inf")),
                       (1, -float("inf")), ([1], 2)]:
            with self.subTest(target=target):
                self.assertEqual(format_target(target), HOLD_MESSAGE)

    def test_wire_format_signs_axes_and_precision_are_unchanged(self):
        self.assertEqual(format_target((12.345, -40)), b"1,12.3,-40.0\n")
        self.assertEqual(format_target((-12.345, 40)), b"1,-12.3,40.0\n")

    def test_target_and_flat_context_are_copied_at_publish(self):
        events = []
        link = self.make_link(events.append)
        target = [12.345, -40]
        context = {"frame_id": 7, "wrist": "LEFT", "target_valid": True,
                   "x_target_mm": 12.345, "y_target_mm": -40,
                   "timestamp": "synthetic-frame-time", "data_source": "SYNTHETIC"}
        link.set_target(target, context)
        target[0] = 999
        context["frame_id"] = 8
        link._serve(self.make_socket([b"R\n"]))
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event["frame_id"], 7)
        self.assertEqual(event["x_target_mm"], 12.345)
        self.assertEqual((event["x_sent_mm"], event["y_sent_mm"]), (12.3, -40.0))
        self.assertTrue(event["sent_valid"])
        self.assertTrue(event["target_valid"])
        self.assertEqual(event["event_type"], "SEND")
        self.assertEqual(event["arm"], "ROB_L")
        self.assertEqual(event["timestamp"], "synthetic-frame-time")
        self.assertEqual(event["robot_feedback"], "NOT_AVAILABLE")
        self.assertEqual(event["data_source"], "SYNTHETIC")
        self.assertTrue(event["send_timestamp"].endswith("+00:00"))

    def test_stale_target_send_is_hold_but_retains_frame_context(self):
        events = []
        link = self.make_link(events.append, max_age=0.3)
        with patch("tien_ich.ket_noi_yumi.time.monotonic", return_value=10):
            link.set_target((1, 2), {"frame_id": 9, "target_valid": True})
        sock = self.make_socket([b"R\n"])
        with patch("tien_ich.ket_noi_yumi.time.monotonic", return_value=10.31):
            link._serve(sock)
        sock.sendall.assert_called_once_with(HOLD_MESSAGE)
        self.assertEqual(events[0]["frame_id"], 9)
        self.assertTrue(events[0]["target_valid"])
        self.assertFalse(events[0]["sent_valid"])
        self.assertEqual((events[0]["x_sent_mm"], events[0]["y_sent_mm"]), (0, 0))

    def test_send_failure_never_creates_success_event(self):
        observer = Mock()
        link = self.make_link(observer)
        link.set_target((1, 2))
        sock = self.make_socket([b"R\n"])
        sock.sendall.side_effect = OSError("synthetic send failure")
        with self.assertRaises(OSError):
            link._serve(sock)
        observer.assert_not_called()

    def test_observer_failure_does_not_end_worker_session(self):
        observer = Mock(side_effect=RuntimeError("synthetic log failure"))
        link = self.make_link(observer)
        link.set_target((1, -2))
        sock = self.make_socket([b"R\nR\n"])
        link._serve(sock)
        self.assertEqual(sock.sendall.call_count, 2)
        self.assertEqual(observer.call_count, 2)

    def test_invalid_request_gets_hold_then_valid_request_recovers(self):
        events = []
        link = self.make_link(events.append)
        link.set_target((1, -2), {"frame_id": 1})
        sock = self.make_socket([b"NOT_R\nR\r\nR\n"])
        link._serve(sock)
        self.assertEqual([call.args[0] for call in sock.sendall.call_args_list],
                         [HOLD_MESSAGE, b"1,1.0,-2.0\n", b"1,1.0,-2.0\n"])
        self.assertEqual([event["sent_valid"] for event in events], [False, True, True])

    def test_fragmented_request_and_newest_context_stay_correlated(self):
        events = []
        link = self.make_link(events.append)
        link.set_target((1, 2), {"frame_id": 1})
        sock = self.make_socket([])
        calls = iter([b"R", b"\n", b""])

        def receive(_size):
            chunk = next(calls)
            if chunk == b"\n":
                link.set_target((-3, 4), {"frame_id": 2})
            return chunk

        sock.recv.side_effect = receive
        link._serve(sock)
        sock.sendall.assert_called_once_with(b"1,-3.0,4.0\n")
        self.assertEqual(events[0]["frame_id"], 2)

    def test_socket_timeout_keeps_pending_fragment(self):
        link = self.make_link()
        link.set_target((3, 4))
        sock = self.make_socket([b"R", socket.timeout(), b"\n"])
        link._serve(sock)
        sock.sendall.assert_called_once_with(b"1,3.0,4.0\n")

    def test_overlong_unterminated_request_resets_connection(self):
        link = self.make_link()
        sock = self.make_socket([b"X" * MAX_REQUEST_BYTES, b"X"])
        with self.assertRaises(OSError):
            link._serve(sock)
        sock.sendall.assert_not_called()

    def test_overlong_complete_request_resets_connection(self):
        link = self.make_link()
        sock = self.make_socket([b"X" * (MAX_REQUEST_BYTES + 1) + b"\n"])
        with self.assertRaises(OSError):
            link._serve(sock)
        sock.sendall.assert_not_called()

    def test_malformed_target_does_not_prevent_next_valid_send(self):
        events = []
        link = self.make_link(events.append)
        link.set_target(("bad", 1))
        first = self.make_socket([b"R\n"])
        link._serve(first)
        first.sendall.assert_called_once_with(HOLD_MESSAGE)
        link.set_target((5, -6))
        second = self.make_socket([b"R\n"])
        link._serve(second)
        second.sendall.assert_called_once_with(b"1,5.0,-6.0\n")
        self.assertEqual([event["sent_valid"] for event in events], [False, True])

    def test_reconnect_path_handles_overlong_request_like_socket_failure(self):
        link = self.make_link()
        bad_socket = self.make_socket([b"X" * (MAX_REQUEST_BYTES + 1)])
        connection = Mock()
        connection.__enter__ = Mock(return_value=bad_socket)
        connection.__exit__ = Mock(return_value=False)
        with patch("tien_ich.ket_noi_yumi.socket.create_connection",
                   return_value=connection) as connect:
            with patch.object(link._stop, "wait", side_effect=lambda _delay: link._stop.set()) as wait:
                link._run()
        connect.assert_called_once_with(("127.0.0.1", 1), timeout=2.0)
        wait.assert_called_once_with(1.0)
        self.assertFalse(link.connected)


if __name__ == "__main__":
    unittest.main()
