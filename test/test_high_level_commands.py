import unittest

import numpy as np

from utils.high_level_commands import HighLevelCommandRecognizer


class HighLevelCommandRecognizerTest(unittest.TestCase):
    def setUp(self):
        self.recognizer = HighLevelCommandRecognizer(stable_frames=3)
        self.box = [(0, 0, 100, 200)]

    def pose(self, left="lowered", right="lowered", confidence=1.0):
        pose = np.zeros((17, 3), dtype=np.float32)
        for shoulder_index, wrist_index, state in (
            (5, 9, left),
            (6, 10, right),
        ):
            pose[shoulder_index] = [30 if shoulder_index == 5 else 70, 80, confidence]
            wrist_y = {"raised": 30, "lowered": 130, "unknown": 80}[state]
            point_confidence = 0.0 if state == "unknown" else confidence
            pose[wrist_index] = [30 if wrist_index == 9 else 70, wrist_y, point_confidence]
        return pose

    def stable_command(self, left, right):
        results = [
            self.recognizer.update(self.box, [self.pose(left, right)])
            for _ in range(3)
        ]
        return results

    def test_emits_start_once_after_stable_lowered_arms(self):
        self.assertEqual(self.stable_command("lowered", "lowered"), [None, None, "START"])
        self.assertEqual(self.recognizer.candidate_command, "START")
        self.assertEqual(self.recognizer.candidate_frames, 3)
        self.assertIsNone(self.recognizer.update(self.box, [self.pose()]))

    def test_recognizes_left_right_and_stop(self):
        self.assertEqual(self.stable_command("raised", "lowered")[-1], "LEFT")
        self.assertEqual(self.stable_command("lowered", "raised")[-1], "RIGHT")
        self.assertEqual(self.stable_command("raised", "raised")[-1], "STOP")

    def test_uncertain_pose_and_multiple_people_do_not_emit(self):
        for _ in range(5):
            self.assertIsNone(self.recognizer.update(self.box, [self.pose("unknown", "lowered")]))
            self.assertIsNone(self.recognizer.update(self.box * 2, [self.pose(), self.pose()]))

    def test_low_confidence_does_not_emit(self):
        for _ in range(5):
            self.assertIsNone(self.recognizer.update(self.box, [self.pose(confidence=0.2)]))

    def test_pose_change_restarts_stability_count(self):
        self.assertIsNone(self.recognizer.update(self.box, [self.pose("raised", "lowered")]))
        self.assertIsNone(self.recognizer.update(self.box, [self.pose("lowered", "raised")]))
        self.assertIsNone(self.recognizer.update(self.box, [self.pose("lowered", "raised")]))
        self.assertEqual(
            self.recognizer.update(self.box, [self.pose("lowered", "raised")]),
            "RIGHT",
        )


if __name__ == "__main__":
    unittest.main()
