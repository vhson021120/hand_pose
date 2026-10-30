from math import ceil
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from tien_ich.toa_do_co_tay_tuong_doi import RelativeWristState
from tien_ich.anh_xa_toa_do_robot import RobotTargetPreview


HAND_CONNECTIONS = (
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
    (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17),
)
_ARM_CONNECTIONS = (
    (11, 13, (210, 135, 55)),
    (13, 15, (210, 135, 55)),
    (12, 14, (185, 75, 190)),
    (14, 16, (185, 75, 190)),
    (11, 12, (150, 165, 185)),
)
_FACE_CONNECTIONS = ((7, 2), (2, 0), (0, 5), (5, 8), (2, 5))
_FACE_INDICES = (0, 2, 5, 7, 8)
_MIN_CONFIDENCE = 0.5


def _point(points: np.ndarray, index: int) -> Optional[np.ndarray]:
    if index >= len(points):
        return None
    point = points[index]
    if not np.isfinite(point[:3]).all():
        return None
    if len(point) > 3 and (
        not np.isfinite(point[3]) or point[3] < _MIN_CONFIDENCE
    ):
        return None
    return point[:3].astype(np.float32)


def anchor_hand_to_wrist(hand_world: np.ndarray, pose_world: np.ndarray, wrist_index: int) -> np.ndarray:
    """Translate hand-local landmarks so its wrist meets the matching pose wrist."""
    wrist = _point(pose_world, wrist_index)
    if wrist is None or hand_world.ndim != 2 or len(hand_world) == 0:
        return hand_world.copy()
    hand_wrist = hand_world[0, :3]
    if not np.isfinite(hand_wrist).all():
        return hand_world.copy()
    anchored = hand_world.copy()
    anchored[:, :3] += wrist - hand_wrist
    return anchored


def _make_bone(
    start: np.ndarray,
    end: np.ndarray,
    radius: float,
    color: Tuple[int, int, int],
) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    axis = end - start
    length = float(np.linalg.norm(axis))
    if length < 1e-5:
        return []
    axis /= length
    reference = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    if abs(float(np.dot(axis, reference))) > 0.92:
        reference = np.array((1.0, 0.0, 0.0), dtype=np.float32)
    side = np.cross(axis, reference)
    side /= max(float(np.linalg.norm(side)), 1e-6)
    depth_axis = np.cross(axis, side)
    depth_axis /= max(float(np.linalg.norm(depth_axis)), 1e-6)

    start_ring = [
        start + side * radius * np.cos(angle) + depth_axis * radius * np.sin(angle)
        for angle in np.linspace(0, 2 * np.pi, 5)[:-1]
    ]
    end_ring = [
        end + side * radius * np.cos(angle) + depth_axis * radius * np.sin(angle)
        for angle in np.linspace(0, 2 * np.pi, 5)[:-1]
    ]
    faces = [(start_ring, color), (end_ring, color)]
    for index in range(4):
        next_index = (index + 1) % 4
        faces.append((
            [start_ring[index], start_ring[next_index], end_ring[next_index], end_ring[index]],
            color,
        ))
    return [(np.asarray(vertices, dtype=np.float32), face_color) for vertices, face_color in faces]


def _build_arm_faces(
    points: np.ndarray,
    hands: Sequence[Tuple[np.ndarray, bool]] = (),
) -> List[Tuple[np.ndarray, Tuple[int, int, int]]]:
    if points.ndim != 2 or points.shape[1] < 3 or len(points) < 33:
        return []

    joints = {index: _point(points, index) for index in range(33)}
    shoulder_points = [joints[index] for index in (11, 12) if joints[index] is not None]
    shoulder_width = (
        float(np.linalg.norm(joints[11] - joints[12]))
        if len(shoulder_points) == 2 else 0.36
    )
    arm_radius = max(0.014, shoulder_width * 0.07)
    faces: List[Tuple[np.ndarray, Tuple[int, int, int]]] = []

    for first, second, color in _ARM_CONNECTIONS:
        if joints[first] is not None and joints[second] is not None:
            faces.extend(_make_bone(joints[first], joints[second], arm_radius, color))

    for first, second in _FACE_CONNECTIONS:
        if joints[first] is not None and joints[second] is not None:
            faces.extend(_make_bone(joints[first], joints[second], 0.006, (190, 205, 220)))

    for hand, is_left in hands:
        if hand.ndim != 2 or hand.shape[0] < 21 or hand.shape[1] < 3:
            continue
        color = (100, 210, 150) if is_left else (210, 130, 210)
        hand_points = [_point(hand, index) for index in range(21)]
        for first, second in HAND_CONNECTIONS:
            if hand_points[first] is not None and hand_points[second] is not None:
                faces.extend(_make_bone(hand_points[first], hand_points[second], 0.007, color))
    return faces


def _project(
    point: np.ndarray,
    origin: np.ndarray,
    center: Tuple[float, float],
    scale: float,
) -> Tuple[int, int]:
    relative = point - origin
    # A shallow fixed isometric view reveals depth without changing pose scale per frame.
    x = relative[0] - relative[2] * 0.5
    y = relative[1] - relative[2] * 0.3
    return int(round(center[0] + x * scale)), int(round(center[1] + y * scale))


def _model_origin(
    pose: Optional[np.ndarray],
    hands: Sequence[Tuple[np.ndarray, bool]],
) -> Optional[np.ndarray]:
    if pose is not None and pose.ndim == 2 and pose.shape[0] >= 33:
        left_shoulder = _point(pose, 11)
        right_shoulder = _point(pose, 12)
        if left_shoulder is None or right_shoulder is None:
            return None
        return (left_shoulder + right_shoulder) * 0.5
    if hands:
        wrist = _point(hands[0][0], 0)
        if wrist is not None:
            return wrist
    return np.zeros(3, dtype=np.float32)


def _draw_model(
    image: np.ndarray,
    pose: Optional[np.ndarray],
    hands: Sequence[Tuple[np.ndarray, bool]],
    region: Tuple[int, int, int, int],
    label: str,
    wrist_preview: Sequence[RelativeWristState] = (),
) -> bool:
    if pose is not None and (pose.ndim != 2 or pose.shape[0] < 33 or pose.shape[1] < 3):
        pose = None

    anchored_hands = []
    for hand, is_left in hands:
        if hand.ndim != 2 or hand.shape[0] < 21 or hand.shape[1] < 3:
            continue
        if pose is not None:
            wrist_index = 15 if is_left else 16
            hand = anchor_hand_to_wrist(hand, pose, wrist_index)
        anchored_hands.append((hand, is_left))

    faces = _build_arm_faces(pose, anchored_hands) if pose is not None else []
    if pose is None:
        for hand, is_left in anchored_hands:
            color = (100, 210, 150) if is_left else (210, 130, 210)
            hand_points = [_point(hand, index) for index in range(21)]
            for first, second in HAND_CONNECTIONS:
                if hand_points[first] is not None and hand_points[second] is not None:
                    faces.extend(_make_bone(hand_points[first], hand_points[second], 0.007, color))
    if not faces:
        return False

    origin = _model_origin(pose, anchored_hands)
    if origin is None:
        return False
    x1, y1, x2, y2 = region
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    # Keep one fixed model-space scale per person slot to make wrist motion visible.
    scale = float(max(1.0, min((width - 30) / 2.0, (height - 44) / 1.6)))
    center = (x1 + width / 2, y1 + height / 2)

    z_values = np.concatenate([face[0][:, 2] for face in faces])
    z_min, z_max = float(z_values.min()), float(z_values.max())
    projected_faces = []
    for vertices, color in faces:
        depth = float(np.mean(vertices[:, 2]))
        projected = np.asarray(
            [_project(vertex, origin, center, scale) for vertex in vertices],
            dtype=np.int32,
        )
        projected_faces.append((depth, projected, color))

    # MediaPipe world z is smaller for points closer to camera: draw far faces first.
    for depth, vertices, color in sorted(projected_faces, key=lambda face: face[0], reverse=True):
        if len(vertices) >= 3:
            shaded = _depth_shade(color, depth, z_min, z_max)
            cv2.fillConvexPoly(image, vertices, shaded, cv2.LINE_AA)
            cv2.polylines(image, [vertices], True, (225, 230, 238), 1, cv2.LINE_AA)

    axis_origin = _project(origin, origin, center, scale)
    for axis, color, label in (
        (np.array((0.25, 0.0, 0.0), dtype=np.float32), (80, 190, 255), "X"),
        (np.array((0.0, 0.25, 0.0), dtype=np.float32), (100, 230, 130), "Y"),
        (np.array((0.0, 0.0, 0.25), dtype=np.float32), (255, 170, 90), "Z"),
    ):
        endpoint = _project(origin + axis, origin, center, scale)
        cv2.arrowedLine(image, axis_origin, endpoint, color, 1, cv2.LINE_AA, tipLength=0.2)
        cv2.putText(image, label, (endpoint[0] + 2, endpoint[1] - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)

    if pose is not None:
        for index in _FACE_INDICES:
            point = _point(pose, index)
            if point is not None:
                cv2.circle(image, _project(point, origin, center, scale), 4, (220, 230, 240), -1, cv2.LINE_AA)
        for index, color in (
            (11, (210, 135, 55)), (13, (210, 135, 55)), (15, (210, 135, 55)),
            (12, (185, 75, 190)), (14, (185, 75, 190)), (16, (185, 75, 190)),
        ):
            point = _point(pose, index)
            if point is not None:
                cv2.circle(image, _project(point, origin, center, scale), 5, color, -1, cv2.LINE_AA)
        for state in wrist_preview:
            if state.delta_cm is None or state.side not in ("LEFT", "RIGHT"):
                continue
            displacement = np.asarray(state.delta_cm, dtype=np.float32) / 100.0
            if displacement.shape != (3,) or not np.isfinite(displacement).all():
                continue
            wrist_index = 15 if state.side == "LEFT" else 16
            wrist = _point(pose, wrist_index)
            if wrist is None:
                continue
            neutral_wrist = wrist - displacement
            start = _project(neutral_wrist, origin, center, scale)
            end = _project(wrist, origin, center, scale)
            color = (100, 240, 170) if state.side == "LEFT" else (230, 145, 235)
            cv2.circle(image, start, 4, color, 1, cv2.LINE_AA)
            cv2.arrowedLine(image, start, end, color, 2, cv2.LINE_AA, tipLength=0.2)
    for hand, is_left in anchored_hands:
        color = (100, 210, 150) if is_left else (210, 130, 210)
        for index in range(21):
            point = _point(hand, index)
            if point is not None:
                cv2.circle(image, _project(point, origin, center, scale), 2, color, -1, cv2.LINE_AA)

    cv2.putText(image, label, (x1 + 8, y1 + 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
    return True


def _depth_shade(
    color: Tuple[int, int, int],
    depth: float,
    z_min: float,
    z_max: float,
) -> Tuple[int, int, int]:
    if z_max - z_min < 1e-6:
        factor = 0.82
    else:
        near_ratio = float(np.clip((z_max - depth) / (z_max - z_min), 0.0, 1.0))
        factor = 0.55 + near_ratio * 0.55
    return tuple(int(np.clip(channel * factor, 0, 255)) for channel in color)


def render_relative_3d_canvas(
    pose_world_landmarks: Sequence[np.ndarray],
    hand_world_landmarks: Sequence[Optional[np.ndarray]],
    hand_person_indices: Optional[Sequence[Optional[int]]] = None,
    hand_is_left: Optional[Sequence[bool]] = None,
    wrist_preview: Optional[Sequence[RelativeWristState]] = None,
    width: int = 1000,
    height: int = 720,
    target_preview: Optional[RobotTargetPreview] = None,
) -> np.ndarray:
    """Render a stable-scale arm model and midpoint-of-shoulders wrist preview."""
    canvas = np.full((height, width, 3), (24, 29, 36), dtype=np.uint8)
    cv2.putText(canvas, "MEDIAPIPE 3D | SHOULDERS - ARMS - HANDS - FACE",
                (16, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(canvas, "MODEL-RELATIVE ESTIMATE | NOT ROBOT COORDINATES | NO COMMANDS SENT",
                (16, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (80, 190, 255), 1, cv2.LINE_AA)

    people = [
        (index, pose)
        for index, pose in enumerate(pose_world_landmarks)
        if pose.ndim == 2 and pose.shape[0] >= 33 and pose.shape[1] >= 3
    ]
    targets: List[Tuple[Optional[np.ndarray], List[Tuple[np.ndarray, bool]], str]] = []
    attached_indices = set()
    for person_index, pose in people:
        attached_hands = []
        if hand_person_indices is not None and hand_is_left is not None:
            for hand_index, hand in enumerate(hand_world_landmarks):
                if (
                    hand is None
                    or hand_index >= len(hand_person_indices)
                    or hand_index >= len(hand_is_left)
                    or hand_person_indices[hand_index] != person_index
                    or hand.ndim != 2
                    or hand.shape[0] < 21
                ):
                    continue
                attached_hands.append((hand, hand_is_left[hand_index]))
                attached_indices.add(hand_index)
        targets.append((pose, attached_hands, f"PERSON {person_index + 1}"))

    for hand_index, hand in enumerate(hand_world_landmarks):
        if (
            hand is None
            or hand_index in attached_indices
            or hand.ndim != 2
            or hand.shape[0] < 21
        ):
            continue
        is_left = (
            hand_is_left[hand_index]
            if hand_is_left is not None and hand_index < len(hand_is_left)
            else False
        )
        targets.append((None, [(hand, is_left)], f"HAND {hand_index + 1}"))

    if not targets:
        cv2.putText(canvas, "Waiting for a person or hand...",
                    (width // 2 - 150, height // 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (175, 185, 195), 1, cv2.LINE_AA)
    else:
        content_x, content_y = 12, 48
        content_width = width - 24
        content_height = height - (
            155 if target_preview is not None
            else 108 if wrist_preview is not None
            else 76
        )
        columns = 1 if len(targets) == 1 else (2 if len(targets) <= 4 else 3)
        rows = ceil(len(targets) / columns)
        cell_width = content_width // columns
        cell_height = content_height // rows
        for index, (pose, hands, label) in enumerate(targets):
            row, column = divmod(index, columns)
            x1 = content_x + column * cell_width
            y1 = content_y + row * cell_height
            x2 = content_x + (column + 1) * cell_width - 5
            y2 = content_y + (row + 1) * cell_height - 5
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (65, 75, 88), 1, cv2.LINE_AA)
            if not _draw_model(
                canvas,
                pose,
                hands,
                (x1 + 4, y1 + 4, x2 - 4, y2 - 4),
                label,
                wrist_preview or (),
            ):
                cv2.putText(canvas, f"{label}: no usable landmarks",
                            (x1 + 12, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX,
                            0.42, (0, 170, 255), 1, cv2.LINE_AA)

    if wrist_preview is not None:
        for index, state in enumerate(wrist_preview[:2]):
            y = height - 52 + index * 19
            if state.delta_cm is None:
                details = state.status
            else:
                dx, dy, dz = state.delta_cm
                details = f"MODEL dXYZ {dx:+.1f}, {dy:+.1f}, {dz:+.1f} cm"
            confidence = (
                f" | confidence {state.confidence:.2f}"
                if state.confidence is not None and np.isfinite(state.confidence)
                else ""
            )
            cv2.putText(
                canvas,
                f"{state.side}: {details}{confidence}",
                (16, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (150, 225, 170) if state.delta_cm is not None else (0, 190, 255),
                1,
                cv2.LINE_AA,
            )
    if target_preview is not None:
        if target_preview.valid and target_preview.target_mm is not None:
            x, y, z = target_preview.target_mm
            frame = target_preview.robot_frame or "UNSPECIFIED FRAME"
            text = (
                f"DRY-RUN TARGET | {frame} XYZ {x:+.1f}, {y:+.1f}, {z:+.1f} mm"
                f" | {target_preview.status}"
            )
            color = (150, 225, 170)
        else:
            text = f"DRY-RUN TARGET | {target_preview.status}"
            color = (0, 190, 255)
        cv2.putText(
            canvas,
            text,
            (16, height - 74),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )
    return canvas
