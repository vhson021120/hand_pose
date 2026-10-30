import unittest
from unittest.mock import patch

import numpy as np

from tien_ich.ve_mo_hinh_3d import (
    _build_arm_faces,
    _draw_model,
    _model_origin,
    _project,
    anchor_hand_to_wrist,
    render_relative_3d_canvas,
)
from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristState
from tien_ich.anh_xa_toa_do_robot import RobotTargetPreview


class MediaPipe3DOverlayTest(unittest.TestCase):
    def test_renders_pose_and_hand_in_standalone_canvas(self):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 0] = np.linspace(-0.2, 0.2, 33)
        pose[:, 1] = np.linspace(-0.4, 0.4, 33)
        pose[:, 2] = np.linspace(-0.1, 0.1, 33)
        pose[:, 3] = 1.0
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[:, 0] = np.linspace(-0.05, 0.05, 21)
        hand[:, 1] = np.linspace(-0.1, 0.1, 21)
        hand[:, 2] = np.linspace(-0.02, 0.02, 21)

        image = render_relative_3d_canvas(
            [pose], [hand], [0], [True], width=640, height=480
        )

        self.assertEqual(image.shape, (480, 640, 3))
        self.assertTrue(np.any(image != (24, 29, 36)))

    def test_builds_only_arm_and_face_volumes_not_torso_or_legs(self):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 3] = 1.0
        pose[0, :3] = (0, -0.85, 0)
        pose[2, :3] = (-0.05, -0.86, 0)
        pose[5, :3] = (0.05, -0.86, 0)
        pose[7, :3] = (-0.12, -0.85, 0)
        pose[8, :3] = (0.12, -0.85, 0)
        pose[11, :3] = (-0.2, -0.7, 0)
        pose[12, :3] = (0.2, -0.7, 0)
        pose[13, :3] = (-0.35, -0.4, 0)
        pose[14, :3] = (0.35, -0.4, 0)
        pose[15, :3] = (-0.4, -0.1, 0)
        pose[16, :3] = (0.4, -0.1, 0)

        faces = _build_arm_faces(pose)
        pose[23:33, :3] = 100.0
        faces_without_lower_body = _build_arm_faces(pose)

        self.assertGreaterEqual(len(faces), 30)
        self.assertTrue(all(vertices.shape == (4, 3) for vertices, _ in faces))
        self.assertEqual(len(faces), len(faces_without_lower_body))
        for (vertices, _), (lower_body_vertices, _) in zip(faces, faces_without_lower_body):
            np.testing.assert_array_equal(vertices, lower_body_vertices)

    def test_hand_world_points_are_translated_to_the_matching_pose_wrist(self):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[15, :3] = (0.4, -0.6, 0.2)
        pose[15, 3] = 1.0
        hand = np.zeros((21, 3), dtype=np.float32)
        hand[0] = (-0.03, 0.01, -0.02)
        hand[8] = (0.05, 0.1, 0.03)

        anchored = anchor_hand_to_wrist(hand, pose, 15)

        np.testing.assert_allclose(anchored[0], pose[15, :3])
        np.testing.assert_allclose(
            anchored[8] - anchored[0],
            hand[8] - hand[0],
            atol=1e-7,
        )

    def test_world_y_projects_down_in_image_coordinates(self):
        origin = np.zeros(3, dtype=np.float32)
        top = _project(np.array((0, -0.5, 0), dtype=np.float32), origin, (0, 0), 100)
        bottom = _project(np.array((0, 0.5, 0), dtype=np.float32), origin, (0, 0), 100)

        self.assertLess(top[1], bottom[1])

    def test_model_origin_is_exact_midpoint_of_both_valid_shoulders(self):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[11] = (-0.3, 0.5, 0.2, 1.0)
        pose[12] = (0.5, 0.1, -0.2, 1.0)

        origin = _model_origin(pose, [])

        np.testing.assert_allclose(origin, (0.1, 0.3, 0.0), atol=1e-7)
        pose[12, 3] = 0.1
        self.assertIsNone(_model_origin(pose, []))

    def test_shows_waiting_state_when_world_landmarks_are_unavailable(self):
        image = render_relative_3d_canvas([], [None], width=640, height=480)

        self.assertTrue(np.any(image != (24, 29, 36)))

    def test_renders_relative_wrist_preview_status(self):
        state = RelativeWristState(
            side="LEFT",
            delta_cm=(1.0, -2.0, 3.0),
            status="MODEL RELATIVE",
            confidence=0.9,
            calibration_frames=20,
            calibrated=True,
        )

        image = render_relative_3d_canvas(
            [], [], wrist_preview=(state,), width=640, height=480
        )

        self.assertEqual(image.shape, (480, 640, 3))
        self.assertTrue(np.any(image != (24, 29, 36)))

    def test_renders_robot_target_as_dry_run_with_explicit_frame_and_units(self):
        target = RobotTargetPreview(
            valid=True,
            status="DRY RUN",
            robot_frame="confirmed_base",
            tracked_side="RIGHT",
            offset_mm=(10.0, -20.0, 30.0),
            target_mm=(310.0, 80.0, 230.0),
        )
        with patch("tien_ich.ve_mo_hinh_3d.cv2.putText") as draw_text:
            image = render_relative_3d_canvas(
                [], [], target_preview=target, width=640, height=480
            )

        self.assertEqual(image.shape, (480, 640, 3))
        texts = [call.args[1] for call in draw_text.call_args_list]
        self.assertIn(
            "DRY-RUN TARGET | confirmed_base XYZ +310.0, +80.0, +230.0 mm | DRY RUN",
            texts,
        )

    def test_maps_relative_wrist_delta_to_a_vector_on_the_3d_model(self):
        pose = np.zeros((33, 4), dtype=np.float32)
        pose[:, 3] = 1.0
        pose[11, :3] = (-0.2, 0.0, 0.0)
        pose[12, :3] = (0.2, 0.0, 0.0)
        pose[13, :3] = (-0.3, -0.2, 0.0)
        pose[15, :3] = (-0.4, -0.4, 0.0)
        pose[14, :3] = (0.3, -0.2, 0.0)
        pose[16, :3] = (0.4, -0.4, 0.0)
        state = RelativeWristState(
            side="LEFT",
            delta_cm=(10.0, 0.0, 0.0),
            status="MODEL RELATIVE",
            confidence=0.9,
            calibration_frames=20,
            calibrated=True,
        )
        image = np.zeros((480, 640, 3), dtype=np.uint8)

        with patch("tien_ich.ve_mo_hinh_3d.cv2.arrowedLine") as draw_arrow:
            _draw_model(image, pose, [], (0, 0, 640, 480), "PERSON", (state,))

        expected_start = _project(
            np.array((-0.5, -0.4, 0.0), dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            (320, 240),
            272.5,
        )
        expected_end = _project(
            pose[15, :3],
            np.zeros(3, dtype=np.float32),
            (320, 240),
            272.5,
        )
        calls = [
            call.args[1:3]
            for call in draw_arrow.call_args_list
            if call.args[0] is image
        ]
        self.assertIn((expected_start, expected_end), calls)


if __name__ == "__main__":
    unittest.main()
