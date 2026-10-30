import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tien_ich.nhan_dien_tu_the import MediaPipeArmRefiner


class MediaPipeArmWorldAssociationTest(unittest.TestCase):
    def test_mapped_pose_preserves_mediapipe_visibility(self):
        refiner = MediaPipeArmRefiner.__new__(MediaPipeArmRefiner)
        refiner.image_module = Mock()
        refiner.image_module.ImageFormat.SRGB = 1
        refiner.image_module.Image.side_effect = lambda *_args: _args[-1]
        refiner.last_timestamp = 0
        refiner.min_visibility = 0.5
        landmarks = [
            SimpleNamespace(x=0.5, y=0.5, visibility=0.8)
            for _ in range(33)
        ]
        landmarks[11].visibility = 0.73
        world_landmarks = [
            SimpleNamespace(x=0.0, y=0.0, z=0.0, visibility=0.8)
            for _ in range(33)
        ]
        refiner.landmarker = Mock()
        refiner.landmarker.detect_for_video.return_value = SimpleNamespace(
            pose_landmarks=[landmarks],
            pose_world_landmarks=[world_landmarks],
        )

        _, poses, _ = refiner.detect_people_with_world(np.zeros((100, 100, 3), dtype=np.uint8))

        self.assertEqual(len(poses), 1)
        self.assertAlmostEqual(float(poses[0][5, 2]), 0.73, places=5)

    def test_world_poses_follow_detector_box_order(self):
        refiner = MediaPipeArmRefiner.__new__(MediaPipeArmRefiner)
        refiner.last_pose_boxes = [
            (0.0, 0.0, 100.0, 100.0),
            (200.0, 0.0, 300.0, 100.0),
        ]
        refiner.last_world_landmarks = [
            np.full((33, 4), 1.0, dtype=np.float32),
            np.full((33, 4), 2.0, dtype=np.float32),
        ]

        matched = refiner.match_world_to_boxes([
            (200.0, 0.0, 300.0, 100.0),
            (0.0, 0.0, 100.0, 100.0),
        ])

        self.assertEqual(len(matched), 2)
        self.assertTrue(np.all(matched[0] == 2.0))
        self.assertTrue(np.all(matched[1] == 1.0))

    def test_unmatched_person_gets_no_world_pose_instead_of_another_persons(self):
        refiner = MediaPipeArmRefiner.__new__(MediaPipeArmRefiner)
        refiner.last_pose_boxes = [(0.0, 0.0, 100.0, 100.0)]
        refiner.last_world_landmarks = [np.ones((33, 4), dtype=np.float32)]

        matched = refiner.match_world_to_boxes([
            (0.0, 0.0, 100.0, 100.0),
            (500.0, 500.0, 600.0, 600.0),
        ])

        self.assertEqual(matched[0].shape, (33, 4))
        self.assertEqual(matched[1].shape, (0, 4))


if __name__ == "__main__":
    unittest.main()
