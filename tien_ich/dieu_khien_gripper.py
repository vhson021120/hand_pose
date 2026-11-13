"""Recognize a palm/fist and bind its gripper command to an observed wrist.

Image x/y are pixels but image z is normalized, so image fallback geometry
uses x/y only. Metric world landmarks, when supplied, allow 3D joint geometry.
No gesture or debounce state is retained through an uncertain observation.
"""

from dataclasses import dataclass
import math
import time
from typing import Mapping, Optional

import numpy as np

from tien_ich.theo_doi_co_tay_2d import finite_values


SIDES = ("LEFT", "RIGHT")
OPEN = "OPEN"
CLOSED = "CLOSED"
UNKNOWN = "UNKNOWN"
NO_HAND = "NO_HAND"


@dataclass(frozen=True)
class GripperState:
    gesture: str = NO_HAND
    command: Optional[int] = None
    reason: str = "NO_HAND"
    model_gesture: Optional[str] = None
    gesture_confidence: Optional[float] = None


def _confidence(hand, threshold):
    try:
        confidence = float(hand.confidence)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    return math.isfinite(confidence) and threshold <= confidence <= 1.0


def _landmarks(value, dimensions):
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if points.shape != (21, 3) or not np.isfinite(points).all():
        return None
    return points[:, :dimensions]


def _joint_angle(first, joint, last):
    a, b = first - joint, last - joint
    lengths = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not math.isfinite(lengths) or lengths < 1e-12:
        return None
    cosine = float(np.dot(a, b)) / lengths
    return math.degrees(math.acos(min(1.0, max(-1.0, cosine))))


def _finger_states(points, strict=False):
    """Classify curl per finger; a noisy joint may abstain rather than veto.

    Ratios use the palm length, and angles use each finger's own bone chain.
    A long finger may fold into a fist without fitting an absolute reach cap.
    ``strict`` is used only when image geometry substitutes uncertain 3D data.
    """
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        points = points - points[0]
        palm = float(np.median(np.linalg.norm(points[[5, 9, 13, 17]], axis=1)))
        if not math.isfinite(palm) or palm <= 1e-8:
            return None
        points = points / palm
        # A nearly edge-on image hand has insufficient geometry for a sign.
        width = float(np.linalg.norm(points[5] - points[17]))
        if not math.isfinite(width) or width < 0.15:
            return None

        states = []
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12),
                                   (13, 14, 15, 16), (17, 18, 19, 20)):
            chain = points[[mcp, pip, dip, tip]]
            bones = np.linalg.norm(np.diff(chain, axis=0), axis=1)
            if not np.isfinite(bones).all() or np.any(bones < 0.025) or np.any(bones > 1.5):
                states.append(UNKNOWN)
                continue
            reach = float(np.linalg.norm(chain[-1] - chain[0]))
            straightness = reach / float(bones.sum())
            pip_angle = _joint_angle(chain[0], chain[1], chain[2])
            dip_angle = _joint_angle(chain[1], chain[2], chain[3])
            if pip_angle is None or dip_angle is None:
                states.append(UNKNOWN)
                continue
            tip_radius = float(np.linalg.norm(points[tip]))
            pip_radius = float(np.linalg.norm(points[pip]))
            open_pip, open_dip = (155, 150) if strict else (145, 135)
            curl = (180 - pip_angle) + (180 - dip_angle)
            if (pip_angle >= open_pip and dip_angle >= open_dip and straightness >= 0.88
                    and tip_radius >= pip_radius + 0.10):
                states.append(OPEN)
            elif (pip_angle <= (125 if strict else 145)
                  and curl >= (110 if strict else 100)
                  and straightness <= (0.72 if strict else 0.78)
                  and tip_radius <= pip_radius + 0.10):
                states.append(CLOSED)
            else:
                states.append(UNKNOWN)
    return tuple(states)


def _gesture_consensus(states, unanimous=False):
    if states is None:
        return UNKNOWN
    required = 4 if unanimous else 3
    if states.count(OPEN) >= required and CLOSED not in states:
        return OPEN
    if states.count(CLOSED) >= required and OPEN not in states:
        return CLOSED
    return UNKNOWN


def _classify_geometry(image_points, world):
    """Combine conservative image/world geometry independently of model labels."""
    image_states = _finger_states(image_points)
    if world is None:
        return _gesture_consensus(image_states)
    world_points = _landmarks(world, 3)
    if world_points is None:
        return UNKNOWN
    world_states = _finger_states(world_points)
    world_gesture = _gesture_consensus(world_states)
    if world_gesture != UNKNOWN:
        opposite = CLOSED if world_gesture == OPEN else OPEN
        if image_states is not None and opposite in image_states:
            return UNKNOWN
        return world_gesture

    # Uncertain finite world estimates may recover only from all four strong
    # image fingers. Plausible 3D data must also provide two supporting fingers:
    # a half-curled hand can project to straight 2D lines, so 2D alone must not
    # override four genuinely uncertain (rather than degenerate) 3D fingers.
    image_gesture = _gesture_consensus(_finger_states(image_points, strict=True), unanimous=True)
    opposite = CLOSED if image_gesture == OPEN else OPEN
    if world_states is not None:
        if opposite in world_states or world_states.count(image_gesture) < 2:
            return UNKNOWN
    return image_gesture


def classify_hand(hand, min_confidence=0.7):
    """Combine the trained top category with conservative landmark geometry.

    A confident Open_Palm/Closed_Fist can resolve uncertain geometry but cannot
    override the opposite sign. A model None or a weak supported category falls
    back to geometry; explicit other signs always veto actuation.

    Geometry allows one uncertain finger among three agreeing fingers, but a
    clearly opposing finger vetoes the sign. Image fallback never mixes
    normalized z with pixel x/y, or overrides contradictory 3D evidence.
    """
    if not _confidence(hand, min_confidence):
        return UNKNOWN
    image_points = _landmarks(getattr(hand, "landmarks", None), 2)
    if image_points is None:
        return UNKNOWN
    world = getattr(hand, "world_landmarks", None)
    gesture_name = getattr(hand, "gesture_name", None)
    if gesture_name is None:
        return _classify_geometry(image_points, world)
    if not isinstance(gesture_name, str):
        return UNKNOWN
    try:
        confidence = float(getattr(hand, "gesture_confidence", None))
    except (TypeError, ValueError, OverflowError):
        return UNKNOWN
    if not math.isfinite(confidence) or not 0 <= confidence <= 1:
        return UNKNOWN
    trained = {"Open_Palm": OPEN, "Closed_Fist": CLOSED}.get(gesture_name)
    if trained is None and gesture_name != "None":
        return UNKNOWN

    geometry = _classify_geometry(image_points, world)
    if trained is not None:
        if geometry not in (trained, UNKNOWN):
            return UNKNOWN
        if confidence >= min_confidence:
            return trained
    return geometry


def _inside(point, roi):
    return roi[0] <= point[0] <= roi[2] and roi[1] <= point[1] <= roi[3]


class GripperController:
    """Debounce independent wrist-associated signs before producing commands.

    ``command`` is 1=open, 0=close, None=hold. A ready command is refreshed on
    every valid frame; network expiry is the responsibility of YumiArmLink.
    Neither x ordering nor Hand Landmarker handedness selects the robot arm.
    """

    def __init__(self, stable_time=0.25, min_confidence=0.7,
                 min_visibility=0.7, max_frame_gap=0.3):
        for name, value in (("stable_time", stable_time), ("max_frame_gap", max_frame_gap)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name, value in (("min_confidence", min_confidence), ("min_visibility", min_visibility)):
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0, 1]")
        self.stable_time = stable_time
        self.min_confidence = min_confidence
        self.min_visibility = min_visibility
        self.max_frame_gap = max_frame_gap
        self._image_size = None
        self._last_time = None
        self._pending = {}

    def reset(self):
        self._pending.clear()
        self._image_size = None
        self._last_time = None

    def _associate(self, hands, wrists, size):
        valid_wrists = {}
        if isinstance(wrists, Mapping):
            for side in SIDES:
                point = finite_values(wrists.get(side), 3)
                if point and self.min_visibility <= point[2] <= 1:
                    valid_wrists[side] = point
        matches = {side: [] for side in SIDES}
        ambiguous = set()
        diagonal = math.hypot(*size)
        max_distance = diagonal * 0.10
        ambiguity_margin = diagonal * 0.02
        for hand in hands:
            if not _confidence(hand, self.min_confidence):
                continue
            points = _landmarks(getattr(hand, "landmarks", None), 2)
            if points is None:
                continue
            wrist = points[0]
            distances = sorted((math.hypot(wrist[0] - point[0], wrist[1] - point[1]), side)
                               for side, point in valid_wrists.items())
            if not distances or distances[0][0] > max_distance:
                continue
            if len(distances) > 1 and distances[1][0] - distances[0][0] < ambiguity_margin:
                ambiguous.update(side for _, side in distances)
                continue
            matches[distances[0][1]].append((hand, wrist))
        for side in SIDES:
            if len(matches[side]) > 1:
                ambiguous.add(side)
        return matches, ambiguous, valid_wrists

    def update(self, hands, wrists, roi, image_size, *, armed=False,
               eligible_sides=None, now=None):
        """Consume one fresh frame and return a GripperState for each side.

        ``wrists`` must be raw Pose wrists from this frame. ``eligible_sides``
        can additionally require the caller's filtered wrist to be in the ROI.
        A lost/ambiguous hand, invalid sign, disarm, frame gap or resize requires
        a completely new stable interval before any later command is emitted.
        """
        size = finite_values(image_size, 2)
        bounds = finite_values(roi, 4)
        try:
            now = time.monotonic() if now is None else float(now)
        except (TypeError, ValueError, OverflowError):
            now = float("nan")
        if (size is None or min(size) <= 0 or bounds is None
                or bounds[2] <= bounds[0] or bounds[3] <= bounds[1]
                or not math.isfinite(now)):
            self.reset()
            return {side: GripperState(reason="INVALID_FRAME") for side in SIDES}
        if (self._image_size != size or self._last_time is None
                or now < self._last_time or now - self._last_time > self.max_frame_gap):
            self._pending.clear()
        self._image_size, self._last_time = size, now
        matches, ambiguous, valid_wrists = self._associate(hands, wrists, size)
        eligible = set(SIDES) if eligible_sides is None else set(eligible_sides)
        result = {}
        for side in SIDES:
            command = None
            model_gesture, gesture_confidence = None, None
            if side in ambiguous:
                gesture, reason = NO_HAND, "AMBIGUOUS"
            elif not matches[side]:
                gesture, reason = NO_HAND, "NO_HAND"
            else:
                hand, hand_wrist = matches[side][0]
                model_gesture = getattr(hand, "gesture_name", None)
                gesture_confidence = getattr(hand, "gesture_confidence", None)
                gesture = classify_hand(hand, self.min_confidence)
                if not armed:
                    reason = "DISARMED"
                elif not _inside(valid_wrists[side], bounds) or not _inside(hand_wrist, bounds):
                    reason = "OUTSIDE"
                elif side not in eligible:
                    reason = "INELIGIBLE"
                elif gesture not in (OPEN, CLOSED):
                    reason = "UNKNOWN"
                else:
                    previous = self._pending.get(side)
                    if previous is None or previous[0] != gesture:
                        self._pending[side] = (gesture, now)
                    if now - self._pending[side][1] >= self.stable_time:
                        command = 1 if gesture == OPEN else 0
                        reason = "READY"
                    else:
                        reason = "STABILIZING"
            if reason not in ("READY", "STABILIZING"):
                self._pending.pop(side, None)
            result[side] = GripperState(gesture, command, reason, model_gesture, gesture_confidence)
        return result
