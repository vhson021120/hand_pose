import unittest

import numpy as np

from tien_ich.ket_hop_ban_tay_pose import HandKapoFusion, HandResult


class HandKapoFusionTest(unittest.TestCase):
    def test_coordinate_conversion_uses_source_and_target_dimensions(self):
        fusion = HandKapoFusion(source_width=640, source_height=480, target_width=1280, target_height=960)
        hand = HandResult(
            x=100,
            y=80,
            width=200,
            height=120,
            confidence=0.9,
            is_left=True,
            landmarks=np.zeros((21, 3), dtype=np.float32),
        )

        result = fusion.associate_hands_with_arms([hand], [], [])[0]

        self.assertEqual(result.x, 200)
        self.assertEqual(result.y, 160)
        self.assertEqual(result.width, 400)
        self.assertEqual(result.height, 240)
        self.assertEqual(result.confidence, 0.9)
        self.assertTrue(result.is_left)

    def test_world_landmarks_survive_coordinate_conversion(self):
        world_landmarks = np.ones((21, 3), dtype=np.float32)
        fusion = HandKapoFusion(640, 480, 1280, 960)
        hand = HandResult(
            100, 80, 200, 120, 0.9, True,
            np.zeros((21, 3), dtype=np.float32),
            world_landmarks=world_landmarks,
        )

        result = fusion.to_target(hand)

        np.testing.assert_array_equal(result.world_landmarks, world_landmarks)

    def test_hand_is_associated_with_the_matching_person(self):
        fusion = HandKapoFusion(source_width=640, source_height=480, target_width=1280, target_height=960)
        person_boxes = [
            np.array([100, 100, 400, 400]),
            np.array([500, 100, 800, 400]),
        ]
        hand = HandResult(
            x=250,
            y=250,
            width=80,
            height=80,
            confidence=0.9,
            is_left=True,
            landmarks=np.zeros((21, 3), dtype=np.float32),
        )

        matches = fusion.match_person(hand, person_boxes)

        self.assertEqual(matches, 0)

    def test_hand_outside_person_box_is_rejected(self):
        fusion = HandKapoFusion(source_width=640, source_height=480, target_width=1280, target_height=960)
        person_boxes = [np.array([100, 100, 400, 400])]
        hand = HandResult(
            x=500,
            y=250,
            width=80,
            height=80,
            confidence=0.9,
            is_left=True,
            landmarks=np.zeros((21, 3), dtype=np.float32),
        )

        self.assertIsNone(fusion.match_person(hand, person_boxes))

    def test_hand_side_comes_from_matching_body_wrist(self):
        fusion = HandKapoFusion(200, 200, 200, 200)
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[9] = [35, 100, 1]
        pose[10] = [165, 100, 1]
        person_box = (0, 0, 200, 200)
        hands = [
            HandResult(35, 100, 20, 30, 0.9, False, np.tile([35, 100, 0], (21, 1))),
            HandResult(165, 100, 20, 30, 0.8, True, np.tile([165, 100, 0], (21, 1))),
        ]

        results = fusion.associate_hands_with_arms(hands, [person_box], [pose])

        self.assertEqual(len(results), 2)
        self.assertEqual({hand.is_left for hand in results}, {True, False})
        self.assertEqual({hand.person_index for hand in results}, {0})

    def test_duplicate_hand_side_keeps_highest_confidence(self):
        fusion = HandKapoFusion(200, 200, 200, 200)
        pose = np.zeros((17, 3), dtype=np.float32)
        pose[9] = [35, 100, 1]
        person_box = (0, 0, 200, 200)
        hands = [
            HandResult(32, 100, 20, 30, 0.6, False, np.tile([32, 100, 0], (21, 1))),
            HandResult(38, 100, 20, 30, 0.9, True, np.tile([38, 100, 0], (21, 1))),
        ]

        results = fusion.associate_hands_with_arms(hands, [person_box], [pose])

        self.assertEqual(len(results), 1)
        self.assertTrue(results[0].is_left)
        self.assertEqual(results[0].confidence, 0.9)


if __name__ == "__main__":
    unittest.main()
