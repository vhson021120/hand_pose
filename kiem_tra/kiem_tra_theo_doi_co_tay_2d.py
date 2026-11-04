import unittest

from tien_ich.theo_doi_co_tay_2d import WristXYTracker, screen_pixel_to_centered_mm


class WristXYTrackerTest(unittest.TestCase):
    def test_center_is_origin_and_screen_up_is_positive_y(self):
        origin = screen_pixel_to_centered_mm((640, 360), (1280, 720), 0.35)
        up = screen_pixel_to_centered_mm((640, 260), (1280, 720), 0.35)
        right = screen_pixel_to_centered_mm((740, 360), (1280, 720), 0.35)

        self.assertEqual(origin, (0.0, 0.0))
        self.assertEqual(up, (0.0, 35.0))
        self.assertEqual(right, (35.0, 0.0))

    def test_smoothing_uses_observations_and_does_not_predict_missing_points(self):
        tracker = WristXYTracker(smoothing=0.25, min_visibility=0.7)
        first = tracker.update({"LEFT": (100, 100, 0.9)})
        second = tracker.update({"LEFT": (120, 100, 0.9)})
        missing = tracker.update({})
        reacquired = tracker.update({"LEFT": (130, 100, 0.9)})

        self.assertEqual(first["LEFT"][:2], (100.0, 100.0))
        self.assertEqual(second["LEFT"][:2], (105.0, 100.0))
        self.assertEqual(missing, {})
        self.assertEqual(reacquired["LEFT"][:2], (130.0, 100.0))

    def test_low_visibility_and_invalid_points_are_omitted(self):
        tracker = WristXYTracker(min_visibility=0.7)
        result = tracker.update({
            "LEFT": (10, 20, 0.69),
            "RIGHT": (float("nan"), 20, 0.99),
        })

        self.assertEqual(result, {})


if __name__ == "__main__":
    unittest.main()