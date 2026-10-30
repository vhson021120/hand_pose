"""YuMi kinematics in metres/radians internally and ABB degrees at the API.

Geometry comes from the pinned, vendored KTH ROS IRB14000 description, not a
generic 7-axis model. The endpoint is the URDF ``yumi_link_7_*`` frame: a
RobotStudio workobject transform and a tool/TCP transform must be calibrated
separately. Numerical IK is local, seeded and joint-limited; it does not provide
collision checking or approval for physical robot motion.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


JOINT_ORDER = (1, 2, 7, 3, 4, 5, 6)
MODEL_DIRECTORY = Path(__file__).resolve().parent / "du_lieu_yumi"


class IKError(ValueError):
    """The requested pose cannot be reached under the supplied constraints."""


@dataclass(frozen=True)
class IKResult:
    joints_deg: Tuple[float, ...]
    position_error_m: float
    orientation_error_rad: float


def _vector(value: Iterable[float], length: int, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain {length} finite numbers") from error
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return array


class ArmModel:
    """Base-to-flange-link FK and bounded local IK for one IRB14000 arm.

    All seven joint arrays use physical chain order ``[1, 2, 7, 3, 4, 5, 6]``.
    ABB RAPID jointtarget serialization uses axes 1..6 in ``robax`` and axis 7
    in the controller's configured additional axis; serialization is outside
    this class. ``limits_deg`` has shape (7, 2), lower/upper in that same order.
    """

    def __init__(self, side: str):
        if side not in ("left", "right"):
            raise ValueError("side must be 'left' or 'right'")
        data = json.loads((MODEL_DIRECTORY / "model.json").read_text())
        arm = data["arms"][side]
        self.side = side
        self.base_frame = data["base_frame"]
        self.flange_frame = arm["flange_frame"]
        self.joint_names = tuple(joint["name"] for joint in arm["joints"])
        self.urdf_joint_names = tuple(joint["urdf_joint_name"] for joint in arm["joints"])
        self.limits_deg = np.asarray([joint["limits_deg"] for joint in arm["joints"]])
        self.limits_deg.flags.writeable = False
        self.default_seed_deg = tuple(float(v) for v in data["default_seed_deg"])
        self._limits_rad = np.deg2rad(self.limits_deg)
        self._origins = []
        self._axes = []
        for joint in arm["joints"]:
            origin = np.eye(4)
            # URDF extrinsic roll, pitch, yaw: Rz(yaw) @ Ry(pitch) @ Rx(roll).
            origin[:3, :3] = Rotation.from_euler("xyz", joint["rpy_rad"]).as_matrix()
            origin[:3, 3] = joint["xyz_m"]
            self._origins.append(origin)
            self._axes.append(np.asarray(joint["axis"], dtype=float))

    def _joints(self, joints_deg: Iterable[float]) -> np.ndarray:
        joints = _vector(joints_deg, 7, "joints_deg")
        if np.any(joints < self.limits_deg[:, 0] - 1e-9) or np.any(
            joints > self.limits_deg[:, 1] + 1e-9
        ):
            raise ValueError("joints_deg exceeds IRB14000 physical joint limits")
        return np.deg2rad(joints)

    def _forward_rad(self, joints_rad: np.ndarray) -> np.ndarray:
        transform = np.eye(4)
        for origin, axis, angle in zip(self._origins, self._axes, joints_rad):
            motion = np.eye(4)
            motion[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            transform = transform @ origin @ motion
        return transform

    def forward(self, joints_deg: Iterable[float]) -> np.ndarray:
        """Return a 4x4 homogeneous transform from base to flange-link frame."""
        return self._forward_rad(self._joints(joints_deg))

    def solve(
        self,
        position_m: Iterable[float],
        rotation3x3: Iterable[Iterable[float]],
        seed_deg: Optional[Iterable[float]] = None,
        *,
        position_tolerance_m: float = 0.001,
        orientation_tolerance_rad: float = 0.01,
        max_nfev: int = 120,
    ) -> IKResult:
        """Solve a pose locally around seed, or raise IKError on residual failure.

        Reuse the previous successful result as the next seed for continuity.
        A small seed penalty resolves the redundant seventh joint. No alternate
        seeds or discontinuous branch searches are attempted. Invalid input
        raises ValueError; unreachable/convergence failures raise IKError.
        """
        position = _vector(position_m, 3, "position_m")
        rotation = np.asarray(rotation3x3, dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("rotation3x3 must be a finite 3x3 rotation matrix")
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or not np.isclose(
            np.linalg.det(rotation), 1.0, atol=1e-7, rtol=0
        ):
            raise ValueError("rotation3x3 must be orthonormal with determinant +1")
        if not np.isfinite(position_tolerance_m) or position_tolerance_m <= 0:
            raise ValueError("position_tolerance_m must be finite and positive")
        if not np.isfinite(orientation_tolerance_rad) or orientation_tolerance_rad <= 0:
            raise ValueError("orientation_tolerance_rad must be finite and positive")
        if isinstance(max_nfev, bool) or not isinstance(max_nfev, int) or max_nfev <= 0:
            raise ValueError("max_nfev must be a positive integer")
        seed = self._joints(self.default_seed_deg if seed_deg is None else seed_deg)

        def residual(joints_rad: np.ndarray) -> np.ndarray:
            transform = self._forward_rad(joints_rad)
            orientation = Rotation.from_matrix(rotation @ transform[:3, :3].T).as_rotvec()
            return np.concatenate(
                (transform[:3, 3] - position, 0.2 * orientation, 1e-5 * (joints_rad - seed))
            )

        optimization = least_squares(
            residual,
            seed,
            bounds=(self._limits_rad[:, 0], self._limits_rad[:, 1]),
            max_nfev=max_nfev,
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        transform = self._forward_rad(optimization.x)
        position_error = float(np.linalg.norm(transform[:3, 3] - position))
        orientation_error = float(
            np.linalg.norm(Rotation.from_matrix(rotation @ transform[:3, :3].T).as_rotvec())
        )
        if position_error > position_tolerance_m or orientation_error > orientation_tolerance_rad:
            raise IKError(
                f"{self.side} IK rejected: position residual {position_error:.6f} m "
                f"(limit {position_tolerance_m:.6f}), orientation residual "
                f"{orientation_error:.6f} rad (limit {orientation_tolerance_rad:.6f}); "
                f"{optimization.message}"
            )
        return IKResult(
            joints_deg=tuple(float(value) for value in np.rad2deg(optimization.x)),
            position_error_m=position_error,
            orientation_error_rad=orientation_error,
        )
