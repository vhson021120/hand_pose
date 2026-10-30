# IRB14000 model provenance

The native joint preview uses the **community KTH ROS model for ABB YuMi IRB14000**, not an
ABB-certified RobotStudio model. Geometry and joint limits were extracted from:

- Repository: https://github.com/kth-ros-pkg/yumi
- Commit: `4520af912c0ab3532225d84fed2cabe4870379bd`
- Source: [`yumi_description/urdf/yumi.xacro`](https://github.com/kth-ros-pkg/yumi/blob/4520af912c0ab3532225d84fed2cabe4870379bd/yumi_description/urdf/yumi.xacro)
- Vendored file: `yumi.xacro`, unchanged
- SHA-256: `f496a8d59bfa3ba091bfdaa717b23f6f04e60ecd0f0dc0ab2eca595311f5b127`
- License: BSD-2-Clause; original copyright, conditions and disclaimer are in
  the adjacent `LICENSE`, copied from upstream `license.dat`.

`model.json` contains only the seven revolute joints per arm, in physical
chain order **axis1, axis2, axis7, axis3, axis4, axis5, axis6**. Translation
uses metres; URDF RPY uses radians and extrinsic roll/pitch/yaw
(`Rz(yaw) Ry(pitch) Rx(roll)`). The source's `${PI ...}` expressions are
evaluated using mathematical pi; joint limits are exported in degrees.

The base frame is `yumi_base_link`, coincident with the model's `yumi_body`.
The endpoint is `yumi_link_7_l` or `yumi_link_7_r`. These are the URDF's final
arm-link frames; **no gripper, tool/TCP, pedestal, workobject or World transform
is included**. In particular, upstream's separate gripper attachment offset
`xyz=[0, 0, 0.007]` and `rpy=[0, 0, pi]` is not applied. Validate the base and
endpoint conventions against the selected RobotStudio robot/controller and
calibrate any required transforms before motion.

Both arms' limits in physical order are:

| Axis | Minimum (degrees) | Maximum (degrees) |
| --- | ---: | ---: |
| 1 | -168.5 | 168.5 |
| 2 | -143.5 | 43.5 |
| 7 | -168.5 | 168.5 |
| 3 | -123.5 | 80 |
| 4 | -290 | 290 |
| 5 | -88 | 138 |
| 6 | -229 | 229 |

`default_calibration.json` records FK at canonical joint angles
`[0, -60, 0, 30, 0, 40, 0]` for each arm, plus the resulting position and
orientation. These anchors are numerical examples, not validated collision-free
starting poses. Native tests check fixed-Z XY motion, continuity after tracking
loss, and joint bounds from these anchors. The numerical
IK has joint bounds and continuity regularization, but **no collision model**.
