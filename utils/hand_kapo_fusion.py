from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class HandResult:
    x: float
    y: float
    width: float
    height: float
    confidence: float
    is_left: bool
    landmarks: np.ndarray
    person_index: Optional[int] = None


class HandKapoFusion:
    """Convert MediaPipe hand results and associate them with KAPAO people."""

    def __init__(
        self,
        source_width: int,
        source_height: int,
        target_width: int,
        target_height: int,
        person_box_threshold: float = 0.05,
    ) -> None:
        self.source_width = source_width
        self.source_height = source_height
        self.target_width = target_width
        self.target_height = target_height
        self.person_box_threshold = person_box_threshold

    def scale(self, value: float, source_size: int, target_size: int) -> float:
        if source_size <= 0 or target_size <= 0:
            raise ValueError("Image dimensions must be positive")
        return value * target_size / source_size

    def to_target(self, hand: HandResult) -> HandResult:
        landmarks = hand.landmarks.copy()
        if landmarks.size:
            landmarks[:, 0] = self.scale(landmarks[:, 0], self.source_width, self.target_width)
            landmarks[:, 1] = self.scale(landmarks[:, 1], self.source_height, self.target_height)

        return HandResult(
            x=self.scale(hand.x, self.source_width, self.target_width),
            y=self.scale(hand.y, self.source_height, self.target_height),
            width=self.scale(hand.width, self.source_width, self.target_width),
            height=self.scale(hand.height, self.source_height, self.target_height),
            confidence=hand.confidence,
            is_left=hand.is_left,
            landmarks=landmarks,
            person_index=hand.person_index,
        )

    def box_overlap(self, hand: HandResult, person_box: Sequence[float]) -> float:
        px1, py1, px2, py2 = person_box
        hx1, hy1 = hand.x - hand.width / 2, hand.y - hand.height / 2
        hx2, hy2 = hand.x + hand.width / 2, hand.y + hand.height / 2
        overlap_x = max(0.0, min(px2, hx2) - max(px1, hx1))
        overlap_y = max(0.0, min(py2, hy2) - max(py1, hy1))
        return overlap_x * overlap_y / max(1.0, (px2 - px1) * (py2 - py1))

    def match_person(self, hand: HandResult, person_boxes: Iterable[Sequence[float]]) -> Optional[int]:
        matches = []
        contained = []
        for person_index, person_box in enumerate(person_boxes):
            px1, py1, px2, py2 = person_box
            if px1 <= hand.x <= px2 and py1 <= hand.y <= py2:
                person_center = ((px1 + px2) / 2, (py1 + py2) / 2)
                diagonal = max(1.0, float(np.hypot(px2 - px1, py2 - py1)))
                distance = float(np.hypot(hand.x - person_center[0], hand.y - person_center[1])) / diagonal
                contained.append((distance, person_index))
            overlap = self.box_overlap(hand, person_box)
            if overlap >= self.person_box_threshold:
                matches.append((overlap, person_index))
        if contained:
            return min(contained)[1]
        if not matches:
            return None
        return max(matches, key=lambda item: item[0])[1]

    def match_persons(self, hands: Sequence[HandResult], person_boxes: Sequence[Sequence[float]]) -> List[Optional[int]]:
        return [self.match_person(hand, person_boxes) for hand in hands]

    def associate_hands_with_arms(
        self,
        hands: Sequence[HandResult],
        person_boxes: Sequence[Sequence[float]],
        poses: Sequence[np.ndarray],
        wrist_distance_threshold: float = 0.35,
    ) -> List[HandResult]:
        converted_hands = [self.to_target(hand) for hand in hands]
        person_indices = self.match_persons(converted_hands, person_boxes)
        grouped = {}
        for index, person_index in enumerate(person_indices):
            if person_index is not None:
                grouped.setdefault(person_index, []).append(index)

        assigned_sides = {}
        for person_index, hand_indices in grouped.items():
            pose = np.asarray(poses[person_index])
            if pose.shape[0] <= 10:
                continue
            wrist_points = {}
            for is_left, keypoint_index in ((True, 9), (False, 10)):
                wrist = pose[keypoint_index]
                if wrist[2] > 0 and np.isfinite(wrist[:2]).all():
                    wrist_points[is_left] = wrist[:2]
            if not wrist_points:
                continue

            box = person_boxes[person_index]
            max_distance = wrist_distance_threshold * np.hypot(box[2] - box[0], box[3] - box[1])
            hand_wrist_points = {}
            for hand_index in hand_indices:
                landmarks = converted_hands[hand_index].landmarks
                if landmarks.size:
                    hand_wrist_points[hand_index] = landmarks[0, :2]

            if len(hand_indices) >= 2 and len(wrist_points) == 2:
                left_hand, right_hand = hand_indices[:2]
                if left_hand in hand_wrist_points and right_hand in hand_wrist_points:
                    direct = (np.linalg.norm(hand_wrist_points[left_hand] - wrist_points[True]) +
                              np.linalg.norm(hand_wrist_points[right_hand] - wrist_points[False]))
                    swapped = (np.linalg.norm(hand_wrist_points[left_hand] - wrist_points[False]) +
                               np.linalg.norm(hand_wrist_points[right_hand] - wrist_points[True]))
                    proposed = ({left_hand: True, right_hand: False} if direct <= swapped else
                                {left_hand: False, right_hand: True})
                    for hand_index, is_left in proposed.items():
                        if np.linalg.norm(hand_wrist_points[hand_index] - wrist_points[is_left]) <= max_distance:
                            assigned_sides[hand_index] = is_left

            for hand_index in hand_indices:
                if hand_index in assigned_sides or hand_index not in hand_wrist_points:
                    continue
                nearest_side = min(
                    wrist_points,
                    key=lambda side: np.linalg.norm(hand_wrist_points[hand_index] - wrist_points[side]),
                )
                distance = np.linalg.norm(hand_wrist_points[hand_index] - wrist_points[nearest_side])
                if distance <= max_distance:
                    assigned_sides[hand_index] = nearest_side

        associated = []
        for index, hand in enumerate(converted_hands):
            person_index = person_indices[index]
            is_left = assigned_sides.get(index, hand.is_left)
            associated.append(HandResult(
                hand.x, hand.y, hand.width, hand.height, hand.confidence,
                is_left, hand.landmarks, person_index,
            ))

        best_per_side = {}
        unassociated = []
        for hand in associated:
            if hand.person_index is None:
                unassociated.append(hand)
                continue
            key = (hand.person_index, hand.is_left)
            previous = best_per_side.get(key)
            if previous is None or hand.confidence > previous.confidence:
                best_per_side[key] = hand
        return unassociated + list(best_per_side.values())
