import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from tien_ich.nhan_dien_co_tay import MediaPipeWristDetector


class MediaPipeWristDetectorTest(unittest.TestCase):
    def test_extracts_only_visible_wrist_pixels(self):
        detector = MediaPipeWristDetector.__new__(MediaPipeWristDetector)
        detector.image_module = Mock()
        detector.image_module.ImageFormat.SRGB = 1
        detector.image_module.Image.side_effect = lambda *_args: _args[-1]
        detector.last_timestamp = 0
        detector.min_visibility = 0.7
        landmarks = [SimpleNamespace(x=0.5, y=0.5, visibility=0.9) for _ in range(33)]
        landmarks[15] = SimpleNamespace(x=0.25, y=0.75, visibility=0.9)
        landmarks[16] = SimpleNamespace(x=0.8, y=0.4, visibility=0.6)
        detector.landmarker = Mock()
        detector.landmarker.detect_for_video.return_value = SimpleNamespace(
            pose_landmarks=[landmarks]
        )

        result = detector.detect(np.zeros((100, 200, 3), dtype=np.uint8))

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], {"LEFT": (50.0, 75.0, 0.9)})


if __name__ == "__main__":
    unittest.main()