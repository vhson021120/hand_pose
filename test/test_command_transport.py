import socket
import threading
import unittest

from utils.command_transport import TcpCommandSender


class TcpCommandSenderTest(unittest.TestCase):
    def test_sends_one_line_and_receives_ack(self):
        server = socket.socket()
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        server.settimeout(2)
        received = []
        server_error = []

        def receive_command():
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(2)
                    data = b""
                    while not data.endswith(b"\n"):
                        data += connection.recv(64)
                    received.append(data)
                    connection.sendall(b"ACK LEFT\n")
            except OSError as exc:
                server_error.append(exc)
            finally:
                server.close()

        receiver = threading.Thread(target=receive_command)
        receiver.start()
        sender = TcpCommandSender("127.0.0.1", server.getsockname()[1])
        try:
            self.assertTrue(sender.submit("LEFT"))
            receiver.join(timeout=3)
            self.assertFalse(receiver.is_alive())
            self.assertEqual(received, [b"LEFT\n"])
            self.assertEqual(server_error, [])
        finally:
            sender.close()
            if receiver.is_alive():
                server.close()
                receiver.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
