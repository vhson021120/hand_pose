"""Interactive camera, range, torso-yaw and wrist-depth calibration/validation sessions."""

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np

from tien_ich.bo_tai_du_lieu import LoadWebcam
from tien_ich.uoc_luong_khoang_cach import (
    RangeCalibration,
    ShoulderRangeEstimator,
    fit_range_scale,
    passes_validation,
    range_error_metrics,
    wrist_depth_acceptable,
    wrist_depth_metrics,
    yaw_compensation_report,
)

SAMPLES_PER_POINT = 40
MIN_CHESSBOARD_VIEWS = 12
MAX_REPROJECTION_RMSE_PX = 1.0
WINDOW = 'Calibration'


def _open_camera(args) -> Optional[LoadWebcam]:
    dataset = LoadWebcam(
        pipe=args.pipe,
        img_size=640,
        stride=32,
        width=args.camera_width,
        height=args.camera_height,
        preprocess=False,
    )
    if not dataset.cap.isOpened():
        print(f'Unable to open webcam source {args.pipe}')
        return None
    return dataset


def _draw_lines(image, lines, color=(0, 220, 255)) -> None:
    for row, text in enumerate(lines):
        cv2.putText(image, text, (12, 28 + 26 * row), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)


def _collect(
    dataset,
    arm_refiner,
    sample_fn: Callable,
    prompt: str,
) -> Optional[Tuple[list, int, int]]:
    """Wait for SPACE, then gather SAMPLES_PER_POINT valid samples from sample_fn.

    sample_fn(pose, world, frame_shape) -> (sample or None, reason).
    Returns (samples, width, height), or None if aborted.
    """
    collecting = False
    samples: list = []
    for _, _, frame, _ in dataset:
        height, width = frame.shape[:2]
        _, poses, worlds = arm_refiner.detect_people_with_world(frame)
        reason = 'NEED EXACTLY ONE PERSON'
        if len(poses) == 1 and len(worlds) == 1:
            sample, reason = sample_fn(poses[0], worlds[0], frame.shape)
            if sample is not None:
                reason = 'OK'
                if collecting:
                    samples.append(sample)
        shown = frame.copy()
        for pose in poses:
            for index in (5, 6, 9, 10):
                if pose[index, 2] > 0.3:
                    cv2.circle(shown, (int(pose[index, 0]), int(pose[index, 1])), 5, (0, 255, 0), -1)
        _draw_lines(shown, (
            prompt,
            f'Status: {reason}',
            f'Collecting {len(samples)}/{SAMPLES_PER_POINT}' if collecting
            else 'Press SPACE to record, Q to abort',
        ))
        cv2.imshow(WINDOW, shown)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            return None
        if key == ord(' '):
            collecting = True
            samples.clear()
        if len(samples) >= SAMPLES_PER_POINT:
            return samples, width, height
    return None


def run_focal_calibration(args, calibration: RangeCalibration, path: Path) -> int:
    """Estimate fx, fy and the principal point from a printed chessboard (OpenCV)."""
    cols, rows = args.chessboard_size
    square_m = args.chessboard_square_mm / 1000.0
    pattern = (cols, rows)
    grid = np.zeros((cols * rows, 3), np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    dataset = _open_camera(args)
    if dataset is None:
        return 1
    object_points, image_points = [], []
    size = None
    last_capture = -100
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    try:
        for index, (_, _, frame, _) in enumerate(dataset):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            size = gray.shape[::-1]
            found, corners = cv2.findChessboardCorners(
                gray, pattern, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            shown = frame.copy()
            if found:
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(shown, pattern, corners, found)
            _draw_lines(shown, (
                f'Chessboard {cols}x{rows} inner corners, square {args.chessboard_square_mm:.1f} mm',
                f'Views {len(object_points)}/{MIN_CHESSBOARD_VIEWS}+ | '
                f'{"BOARD FOUND - press SPACE" if found else "board not found"}',
                'Tilt/move the board between views. ENTER = finish, Q = abort',
            ))
            cv2.imshow(WINDOW, shown)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Aborted; nothing saved.')
                return 1
            if key == ord(' ') and found and index - last_capture > 5:
                object_points.append(grid)
                image_points.append(corners)
                last_capture = index
            if key in (13, 10) and len(object_points) >= MIN_CHESSBOARD_VIEWS:
                break
    finally:
        dataset.release()
        cv2.destroyAllWindows()

    if size is None or len(object_points) < MIN_CHESSBOARD_VIEWS:
        print(f'Need at least {MIN_CHESSBOARD_VIEWS} chessboard views; nothing saved.')
        return 1
    rmse, matrix, _distortion, _, _ = cv2.calibrateCamera(object_points, image_points, size, None, None)
    print(f'Image {size[0]}x{size[1]}: fx={matrix[0, 0]:.1f}, fy={matrix[1, 1]:.1f}, '
          f'cx={matrix[0, 2]:.1f}, cy={matrix[1, 2]:.1f}, reprojection RMSE={rmse:.3f} px')
    if rmse > MAX_REPROJECTION_RMSE_PX:
        print(f'Reprojection RMSE above {MAX_REPROJECTION_RMSE_PX} px; nothing saved. '
              'Use more varied, sharper, well-lit views.')
        return 1
    calibration.image_width, calibration.image_height = int(size[0]), int(size[1])
    calibration.fx_px, calibration.fy_px = float(matrix[0, 0]), float(matrix[1, 1])
    calibration.cx_px, calibration.cy_px = float(matrix[0, 2]), float(matrix[1, 2])
    calibration.reprojection_rmse_px = float(rmse)
    # The focal length changed, so any earlier scale and confirmation no longer apply.
    calibration.scale = 1.0
    calibration.confirmed = False
    calibration.calibration_metrics = {}
    calibration.validation_metrics = {}
    calibration.save(path)
    print(f'Saved {path}. Lens distortion is not corrected. '
          'Now run --range-calibrate, then --range-validate.')
    return 0


def _yaw_sample(estimator: ShoulderRangeEstimator):
    def sample(pose, world, shape):
        feature = estimator.range_feature(pose, world, shape)
        return ((feature[0], feature[1]) if feature is not None else None), estimator.last_reason
    return sample


def run_range_session(
    args,
    arm_refiner,
    calibration: RangeCalibration,
    calibration_path: Path,
    validate: bool,
) -> int:
    """Calibrate (fit scale) or validate (fixed scale) at the real distances in args."""
    distances = list(args.range_distances)
    if not validate and len(set(distances)) < 2:
        print('Calibration needs at least two distinct --range-distances.')
        return 1
    if validate and not calibration_path.is_file():
        print(f'No calibration file at {calibration_path}; run --range-calibrate first.')
        return 1

    dataset = _open_camera(args)
    if dataset is None:
        return 1
    estimator = ShoulderRangeEstimator(calibration, max_yaw_deg=args.max_yaw_deg)
    recorded: List[np.ndarray] = []
    width = height = 0
    try:
        for distance in distances:
            result = _collect(
                dataset, arm_refiner, _yaw_sample(estimator),
                f'{"VALIDATE" if validate else "CALIBRATE"}: chest {distance:.2f} m from camera, face camera',
            )
            if result is None:
                print('Aborted; nothing saved.')
                return 1
            samples, width, height = result
            recorded.append(np.asarray([s[0] for s in samples], dtype=np.float64))
    finally:
        dataset.release()
        cv2.destroyAllWindows()

    true_all = np.concatenate([np.full(len(f), d) for f, d in zip(recorded, distances)])
    feature_all = np.concatenate(recorded)
    medians = [float(np.median(f)) for f in recorded]

    if validate:
        constant = calibration.range_constant(width)
        metrics = range_error_metrics(true_all, constant * feature_all)
        for distance, features in zip(distances, recorded):
            print(f'  true {distance:.2f} m -> median {constant * float(np.median(features)):.3f} m '
                  f'(std {constant * float(np.std(features)):.3f} m)')
        calibration.validation_metrics = metrics
        calibration.confirmed = passes_validation(metrics)
        print(f'Validation: MAE={metrics["mae_m"]:.3f} m, RMSE={metrics["rmse_m"]:.3f} m, '
              f'max={metrics["max_abs_error_m"]:.3f} m -> '
              f'{"CONFIRMED" if calibration.confirmed else "NOT CONFIRMED"}')
    else:
        if width != calibration.image_width and calibration.fx_px > 0:
            print(f'Warning: camera is {width}px wide but intrinsics were calibrated at '
                  f'{calibration.image_width}px; focal length is rescaled proportionally.')
        else:
            calibration.image_width, calibration.image_height = width, height
        calibration.scale = 1.0
        scale = fit_range_scale(distances, medians, calibration, width)
        calibration.scale = scale
        metrics = range_error_metrics(true_all, calibration.range_constant(width) * feature_all)
        loo_errors = []
        for held in range(len(distances)):
            keep = [k for k in range(len(distances)) if k != held]
            if len({distances[k] for k in keep}) < 2:
                continue
            probe = RangeCalibration(
                image_width=calibration.image_width,
                image_height=calibration.image_height,
                fx_px=calibration.fx_px,
                shoulder_width_m=calibration.shoulder_width_m,
            )
            probe.scale = fit_range_scale(
                [distances[k] for k in keep], [medians[k] for k in keep], probe, width
            )
            loo_errors.append(probe.range_constant(width) * medians[held] - distances[held])
        calibration.calibration_metrics = metrics
        calibration.confirmed = False
        print(f'Fitted scale={scale:.4f}; implied shoulder width '
              f'{scale * calibration.shoulder_width_m:.3f} m '
              f'(entered {calibration.shoulder_width_m:.3f} m)')
        print(f'Fit: MAE={metrics["mae_m"]:.3f} m, RMSE={metrics["rmse_m"]:.3f} m, '
              f'max={metrics["max_abs_error_m"]:.3f} m')
        if loo_errors:
            print('Leave-one-out errors (m): ' + ', '.join(f'{e:+.3f}' for e in loo_errors))
        print('Scale saved UNCONFIRMED. Run --range-validate at independent distances to confirm.')

    calibration.save(calibration_path)
    print(f'Saved {calibration_path}')
    return 0


def run_yaw_test(args, arm_refiner, calibration: RangeCalibration, path: Path) -> int:
    """Measure the same real distance while the torso is turned, with and without cos(yaw)."""
    if not path.is_file():
        print(f'No calibration file at {path}; run --range-calibrate first.')
        return 1
    distance = args.range_distances[0]
    dataset = _open_camera(args)
    if dataset is None:
        return 1
    estimator = ShoulderRangeEstimator(calibration, max_yaw_deg=60.0)
    reports = []
    try:
        for target in args.yaw_test_angles:
            result = _collect(
                dataset, arm_refiner, _yaw_sample(estimator),
                f'YAW TEST: chest {distance:.2f} m, turn torso ~{target:.0f} deg, keep chest distance',
            )
            if result is None:
                print('Aborted; nothing saved.')
                return 1
            samples, width, _ = result
            reports.append(yaw_compensation_report(
                distance,
                [s[0] for s in samples],
                [s[1] for s in samples],
                calibration.range_constant(width),
            ))
    finally:
        dataset.release()
        cv2.destroyAllWindows()

    for target, report in zip(args.yaw_test_angles, reports):
        print(f'  target {target:4.0f} deg | measured yaw {report["mean_yaw_deg"]:5.1f} deg '
              f'| error with cos(yaw) {report["compensated_error_m"]:+.3f} m '
              f'| without {report["uncompensated_error_m"]:+.3f} m')
    calibration.yaw_metrics = {
        'distance_m': float(distance),
        'max_abs_compensated_error_m': float(max(abs(r['compensated_error_m']) for r in reports)),
        'max_tested_yaw_deg': float(max(r['max_yaw_deg'] for r in reports)),
        'allowed_yaw_deg': float(args.max_yaw_deg),
    }
    calibration.save(path)
    print(f'Torso beyond {args.max_yaw_deg:.0f} deg is rejected at run time. Saved {path}')
    return 0


def run_wrist_validation(args, arm_refiner, calibration: RangeCalibration, path: Path) -> int:
    """Check MediaPipe wrist depth (dz) against real forward reach measured with a ruler."""
    side = args.wrist_side
    wrist_index = 15 if side == 'LEFT' else 16
    coco_index = 9 if side == 'LEFT' else 10
    dataset = _open_camera(args)
    if dataset is None:
        return 1
    estimator = ShoulderRangeEstimator(calibration, max_yaw_deg=args.max_yaw_deg)

    def sample(pose, world, shape):
        if estimator.range_feature(pose, world, shape) is None:
            return None, estimator.last_reason
        if min(pose[coco_index, 2], world[wrist_index, 3]) < estimator.min_confidence:
            return None, 'LOW WRIST CONFIDENCE'
        midpoint_z = (world[11, 2] + world[12, 2]) / 2.0
        return float(world[wrist_index, 2] - midpoint_z), 'OK'

    medians = []
    try:
        for offset in args.wrist_offsets:
            result = _collect(
                dataset, arm_refiner, sample,
                f'WRIST: {side.lower()} wrist {offset:.2f} m in FRONT of chest plane '
                f'(chest ~{args.wrist_chest_distance:.2f} m)',
            )
            if result is None:
                print('Aborted; nothing saved.')
                return 1
            medians.append(float(np.median(result[0])))
    finally:
        dataset.release()
        cv2.destroyAllWindows()

    metrics = wrist_depth_metrics(args.wrist_offsets, medians)
    for offset, delta_z in zip(args.wrist_offsets, medians):
        print(f'  real forward {offset:.2f} m -> MediaPipe dz {delta_z:+.3f} m '
              f'(estimated forward {-delta_z:.3f} m)')
    acceptable = wrist_depth_acceptable(metrics)
    metrics['side'] = side
    metrics['acceptable'] = acceptable
    calibration.wrist_metrics = metrics
    if path.is_file():
        calibration.save(path)
    print(f'Wrist depth: MAE={metrics["mae_m"]:.3f} m, RMSE={metrics["rmse_m"]:.3f} m, '
          f'max={metrics["max_abs_error_m"]:.3f} m, gain={metrics["gain"]:.2f}, '
          f'sign {"OK" if metrics["sign_correct"] else "WRONG"} -> '
          f'{"ACCEPTABLE for dry-run preview" if acceptable else "NOT ACCEPTABLE"}')
    print('This is a laptop-webcam estimate, not a robot-grade measurement.')
    return 0
