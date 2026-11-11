import socket
import time
import unittest

from tien_ich.ket_noi_yumi import YumiArmLink, format_target
from tien_ich.theo_doi_co_tay_2d import wrist_to_workspace_mm


class FakeRobotTask:
    """Behaves like HandFollow.mod: TCP server that asks "R\\n" and reads one line."""

    def __init__(self):
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.server.settimeout(5)
        self.port = self.server.getsockname()[1]
        self.client = None

    def accept(self):
        self.client, _ = self.server.accept()
        self.client.settimeout(5)

    def request(self):
        self.client.sendall(b"R\n")
        data = b""
        while not data.endswith(b"\n"):
            data += self.client.recv(64)
        return data.decode("ascii")

    def close(self):
        if self.client is not None:
            self.client.close()
        self.server.close()


class FormatTargetTest(unittest.TestCase):
    def test_valid_target_and_hold(self):
        self.assertEqual(format_target((12.345, -40)), b"1,12.3,-40.0\n")
        self.assertEqual(format_target(None), b"0,0.0,0.0\n")
        self.assertEqual(format_target((float("nan"), 1)), b"0,0.0,0.0\n")


class WorkspaceMappingTest(unittest.TestCase):
    def test_center_corners_and_outside(self):
        roi = (100.0, 50.0, 500.0, 350.0)  # 400 x 300 px for a 400 x 300 mm workspace
        work = (400.0, 300.0)
        self.assertEqual(wrist_to_workspace_mm((300, 200), roi, work), (0.0, 0.0, True))
        self.assertEqual(wrist_to_workspace_mm((500, 50), roi, work), (200.0, 150.0, True))
        self.assertEqual(wrist_to_workspace_mm((100, 350), roi, work), (-200.0, -150.0, True))
        self.assertEqual(wrist_to_workspace_mm((900, 200), roi, work), (200.0, 0.0, False))


class YumiArmLinkTest(unittest.TestCase):
    def setUp(self):
        self.robot = FakeRobotTask()
        self.link = YumiArmLink("ROB_L", "127.0.0.1", self.robot.port, max_age=0.3).start()
        self.robot.accept()

    def tearDown(self):
        self.link.close()
        self.robot.close()

    def test_replies_with_newest_target_only(self):
        self.link.set_target((10.0, 20.0))
        self.link.set_target((-5.0, 7.5))
        self.assertEqual(self.robot.request(), "1,-5.0,7.5\n")
        self.link.set_target(None)
        self.assertEqual(self.robot.request(), "0,0.0,0.0\n")

    def test_stale_target_becomes_hold(self):
        self.link.set_target((10.0, 20.0))
        time.sleep(0.4)
        self.assertEqual(self.robot.request(), "0,0.0,0.0\n")

    def test_reconnects_after_robot_drops_connection(self):
        self.robot.client.close()
        self.robot.accept()
        self.link.set_target((1.0, 2.0))
        self.assertEqual(self.robot.request(), "1,1.0,2.0\n")


if __name__ == "__main__":
    unittest.main()
