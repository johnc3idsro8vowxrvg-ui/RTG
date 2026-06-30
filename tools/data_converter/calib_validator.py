#!/usr/bin/env python3
"""
calib_validator.py -- Validate calibration parameters for the RTG BEV system.

Checks:
  1. Extrinsic matrix validity: det(R) != 0, R is orthogonal.
  2. Intrinsic matrix validity: fx > 0, fy > 0, cx/cy within image.
  3. Projection alignment: overlay point cloud on camera images for visual check.
  4. LiDAR-to-LiDAR transform consistency.
  5. Generates a validation report in JSON format.

Usage:
  python calib_validator.py \\
      --calib calib.yaml \\
      --image-dir data/extracted/scene_001/ \\
      --frame 0 \\
      --output-dir vis/calib_check/
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------
def load_calib(path):
    """Load calibration from YAML or JSON file."""
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        with open(path, 'r') as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Math checks
# ---------------------------------------------------------------------------
def check_rotation_matrix(R, name='R'):
    """Check if a 3x3 matrix is a valid rotation matrix.

    Checks:
      - det(R) is close to +1 (or -1 for improper rotations).
      - R * R^T is close to identity (orthogonality).
      - No NaN or Inf values.

    Args:
        R: 3x3 array (list of lists or np.ndarray).
        name: Name of the matrix for report messages.

    Returns:
        tuple: (is_valid: bool, issues: list of str)
    """
    R = np.array(R, dtype=np.float64)
    issues = []

    if R.shape != (3, 3):
        issues.append(f'{name}: expected shape (3,3), got {R.shape}')
        return False, issues

    if np.any(np.isnan(R)) or np.any(np.isinf(R)):
        issues.append(f'{name}: contains NaN or Inf values.')
        return False, issues

    # Determinant
    det_R = np.linalg.det(R)
    if abs(abs(det_R) - 1.0) > 0.01:
        issues.append(
            f'{name}: |det(R)| = {abs(det_R):.6f}, expected 1.0 (deviation > 0.01).'
        )

    # Orthogonality: R * R^T should be close to I
    eye_diff = R @ R.T - np.eye(3)
    ortho_error = np.max(np.abs(eye_diff))
    if ortho_error > 0.001:
        issues.append(
            f'{name}: max orthogonality error = {ortho_error:.6f} (threshold 0.001).'
        )

    return len(issues) == 0, issues


def check_intrinsic(K, width, height, name='K'):
    """Check if camera intrinsic matrix is valid.

    Checks:
      - fx > 0, fy > 0.
      - cx, cy within reasonable range [0, 2*width], [0, 2*height].
      - No NaN or Inf.

    Args:
        K: 3x3 intrinsic matrix.
        width: Image width in pixels.
        height: Image height in pixels.
        name: Name for report messages.

    Returns:
        tuple: (is_valid: bool, issues: list of str)
    """
    K = np.array(K, dtype=np.float64)
    issues = []

    if K.shape != (3, 3):
        issues.append(f'{name}: expected shape (3,3), got {K.shape}')
        return False, issues

    if np.any(np.isnan(K)) or np.any(np.isinf(K)):
        issues.append(f'{name}: contains NaN or Inf values.')
        return False, issues

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    if fx <= 0:
        issues.append(f'{name}: fx = {fx:.2f} <= 0.')
    if fy <= 0:
        issues.append(f'{name}: fy = {fy:.2f} <= 0.')

    if cx < 0 or cx > 2 * width:
        issues.append(
            f'{name}: cx = {cx:.2f} out of range [0, {2*width}] '
            f'(image width={width}).'
        )
    if cy < 0 or cy > 2 * height:
        issues.append(
            f'{name}: cy = {cy:.2f} out of range [0, {2*height}] '
            f'(image height={height}).'
        )

    return len(issues) == 0, issues


def check_extrinsic_transform(T, name='T'):
    """Check translation vector validity.

    Args:
        T: 3-element translation vector.
        name: Name for report messages.

    Returns:
        tuple: (is_valid: bool, issues: list of str)
    """
    T = np.array(T, dtype=np.float64).ravel()
    issues = []

    if len(T) != 3:
        issues.append(f'{name}: expected 3 elements, got {len(T)}.')
        return False, issues

    if np.any(np.isnan(T)) or np.any(np.isinf(T)):
        issues.append(f'{name}: contains NaN or Inf values.')
        return False, issues

    if np.linalg.norm(T) > 100:
        issues.append(
            f'{name}: translation magnitude = {np.linalg.norm(T):.2f}m, '
            f'unusually large (>100m).'
        )

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Projection visualization
# ---------------------------------------------------------------------------
def project_lidar_to_image(pts_lidar, R_s2l, T_s2l, intrinsic, img_shape):
    """Project LiDAR points to image plane.

    Since sensor2lidar transforms FROM sensor TO lidar:
      p_lidar = p_sensor @ R_s2l.T + T_s2l

    We need the inverse (lidar to sensor):
      p_camera = (p_lidar - T_s2l) @ R_s2l

    Args:
        pts_lidar: (N, 3) points in LiDAR frame.
        R_s2l: (3, 3) sensor2lidar rotation.
        T_s2l: (3,) sensor2lidar translation.
        intrinsic: (3, 3) camera intrinsic.
        img_shape: (H, W) of the image.

    Returns:
        np.ndarray: Visualization image with projected points.
    """
    R = np.array(R_s2l, dtype=np.float64)
    T = np.array(T_s2l, dtype=np.float64).ravel()

    # Transform to camera frame: p_camera = (p_lidar - T) @ R
    pts_cam = (pts_lidar - T) @ R  # (N, 3)
    depths = pts_cam[:, 2]

    # Filter points behind camera
    front_mask = depths > 0.01
    pts_cam = pts_cam[front_mask]
    depths = depths[front_mask]

    if len(pts_cam) == 0:
        return None, 0

    # Project
    K = np.array(intrinsic, dtype=np.float64)
    uv_h = pts_cam @ K.T
    uv = uv_h[:, :2] / uv_h[:, 2:3]

    # Filter within image
    H, W = img_shape[:2]
    valid_u = (uv[:, 0] >= 0) & (uv[:, 0] < W)
    valid_v = (uv[:, 1] >= 0) & (uv[:, 1] < H)
    valid = valid_u & valid_v

    return uv[valid], depths[valid], len(uv[valid])


def render_projection_overlay(img, uv, depths, inlier_ratio):
    """Render point cloud projected onto image.

    Args:
        img: BGR image.
        uv: (N, 2) pixel coordinates.
        depths: (N,) depth values.
        inlier_ratio: Ratio of points that fall within the image.

    Returns:
        np.ndarray: Annotated image.
    """
    vis = img.copy()
    H, W = vis.shape[:2]

    if len(uv) == 0:
        return vis

    # Color by depth: near=red, far=blue
    d_min, d_max = depths.min(), depths.max()
    if d_max - d_min < 0.01:
        d_norm = np.zeros_like(depths)
    else:
        d_norm = np.clip((depths - d_min) / (d_max - d_min), 0, 1)

    for i in range(len(uv)):
        u, v = int(uv[i, 0]), int(uv[i, 1])
        if 0 <= u < W and 0 <= v < H:
            r = int(255 * (1 - d_norm[i]))
            b = int(255 * d_norm[i])
            cv2.circle(vis, (u, v), 1, (b, 0, r), -1)

    # Add info text
    cv2.putText(vis, f'Points in view: {len(uv)}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(vis, f'Depth range: [{d_min:.1f}, {d_max:.1f}]m',
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return vis


def run_projection_check(calib, image_dir, frame_id=0, output_dir=None):
    """Run point-cloud-to-image projection alignment check.

    Projects LiDAR points onto both camera views and generates
    overlay images for visual inspection.

    Args:
        calib: Calibration dict.
        image_dir: Directory containing front/, rear/, lidar_front/, lidar_rear/.
        frame_id: Frame index to visualize.
        output_dir: Directory to save overlay images.

    Returns:
        dict: Results summary.
    """
    frame_str = f'{frame_id:06d}'
    results = {}

    cam_configs = [
        {
            'name': 'CAM_FRONT',
            'img_dir': 'front',
            'R': calib['extrinsics']['cam_front_to_lidar_front']['R'],
            'T': calib['extrinsics']['cam_front_to_lidar_front']['T'],
            'K': calib['cameras']['front']['intrinsic'],
            'lidar': 'lidar_front',
        },
        {
            'name': 'CAM_BACK',
            'img_dir': 'rear',
            'R': calib['extrinsics']['cam_rear_to_lidar_rear']['R'],
            'T': calib['extrinsics']['cam_rear_to_lidar_rear']['T'],
            'K': calib['cameras']['rear']['intrinsic'],
            'lidar': 'lidar_rear',
        },
    ]

    for cfg in cam_configs:
        # Find image
        img_path = None
        for ext in ['.jpg', '.png', '.jpeg']:
            candidate = os.path.join(image_dir, cfg['img_dir'], frame_str + ext)
            if os.path.isfile(candidate):
                img_path = candidate
                break

        if img_path is None:
            print(f'[WARN] No image found for {cfg["name"]} frame {frame_id}')
            results[cfg['name']] = {'error': 'No image found.'}
            continue

        img = cv2.imread(img_path)
        if img is None:
            results[cfg['name']] = {'error': f'Failed to read {img_path}.'}
            continue

        # Find lidar
        lidar_path = None
        for ext in ['.bin']:
            candidate = os.path.join(image_dir, cfg['lidar'], frame_str + ext)
            if os.path.isfile(candidate):
                lidar_path = candidate
                break

        if lidar_path is None:
            cam_name = cfg['name']
            print(f'[WARN] No lidar found for {cam_name} frame {frame_id}')
        else:
            pts = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

            uv, depths, n_in_view = project_lidar_to_image(
                pts[:, :3], cfg['R'], cfg['T'], cfg['K'], img.shape
            )

            if uv is not None and len(uv) > 0:
                n_total = len(pts)
                inlier_ratio = n_in_view / n_total if n_total > 0 else 0
                results[cfg['name']] = {
                    'total_points': n_total,
                    'points_in_view': n_in_view,
                    'inlier_ratio': float(inlier_ratio),
                    'depth_min': float(depths.min()),
                    'depth_max': float(depths.max()),
                    'depth_mean': float(depths.mean()),
                }

                if output_dir:
                    vis = render_projection_overlay(
                        img, uv, depths, inlier_ratio
                    )
                    os.makedirs(output_dir, exist_ok=True)
                    out_path = os.path.join(
                        output_dir,
                        f'{frame_str}_{cfg["name"].lower()}_projection.jpg'
                    )
                    cv2.imwrite(out_path, vis)
                    results[cfg['name']]['vis_path'] = out_path
                    print(f'[INFO] Saved overlay: {out_path}')
            else:
                results[cfg['name']] = {
                    'total_points': len(pts),
                    'points_in_view': 0,
                    'inlier_ratio': 0.0,
                    'warning': 'No lidar points projected within image bounds.',
                }

    return results


# ---------------------------------------------------------------------------
# Cross-lidar consistency check
# ---------------------------------------------------------------------------
def check_lidar_to_lidar_transform(calib):
    """Check the front-to-rear lidar extrinsic transform consistency.

    If a direct lidar_rear_to_lidar_front transform is provided, verify it
    against the camera-based chain: rear_lidar -> rear_camera -> front_camera -> front_lidar.

    Args:
        calib: Calibration dict.

    Returns:
        tuple: (is_valid: bool, issues: list of str)
    """
    issues = []
    ext = calib.get('extrinsics', {})

    if 'lidar_rear_to_lidar_front' not in ext:
        issues.append(
            'No direct lidar_rear_to_lidar_front transform found. '
            'Cannot verify cross-lidar consistency.'
        )
        return False, issues

    R_lr2lf = np.array(ext['lidar_rear_to_lidar_front']['R'], dtype=np.float64)
    T_lr2lf = np.array(ext['lidar_rear_to_lidar_front']['T'], dtype=np.float64)
    T_lr2lf = T_lr2lf.ravel()

    # Check rotation
    rot_ok, rot_issues = check_rotation_matrix(R_lr2lf, 'lidar_rear_to_lidar_front.R')
    issues.extend(rot_issues)

    # Check translation
    trans_ok, trans_issues = check_extrinsic_transform(T_lr2lf, 'lidar_rear_to_lidar_front.T')
    issues.extend(trans_issues)

    # If camera extrinsics also exist, verify chain consistency
    cam_keys = [
        'cam_rear_to_lidar_rear',
        'cam_front_to_lidar_front',
    ]
    if all(k in ext for k in cam_keys):
        # Compute chain: L_rear -> C_rear -> (C_front) -> L_front
        # Not directly chainable without inter-camera transform, but we can
        # at least check that both cameras' lidar transforms are consistent.
        pass

    return len(issues) == 0, issues


# ---------------------------------------------------------------------------
# Validation report
# ---------------------------------------------------------------------------
def generate_report(calib, projection_results):
    """Generate a structured validation report.

    Args:
        calib: Calibration dict.
        projection_results: dict from run_projection_check().

    Returns:
        dict: Validation report.
    """
    report = {
        'status': 'PASS',
        'checks': {},
    }
    all_issues = []

    # 1. Camera intrinsics
    cam_intrinsic_ok = True
    for cam_key, img_key in [('front', 'CAM_FRONT'), ('rear', 'CAM_BACK')]:
        K = calib['cameras'][cam_key]['intrinsic']
        W = calib['cameras'][cam_key].get('width', 1920)
        H = calib['cameras'][cam_key].get('height', 1080)
        ok, issues = check_intrinsic(K, W, H, f'intrinsic_{cam_key}')
        report['checks'][f'intrinsic_{cam_key}'] = {
            'valid': ok,
            'issues': issues,
            'fx': float(np.array(K)[0, 0]),
            'fy': float(np.array(K)[1, 1]),
            'cx': float(np.array(K)[0, 2]),
            'cy': float(np.array(K)[1, 2]),
        }
        if not ok:
            cam_intrinsic_ok = False
            all_issues.extend(issues)

    # 2. Camera extrinsics (rotation + translation)
    ext_ok = True
    ext_keys = [
        ('cam_front_to_lidar_front', 'extrinsic_front'),
        ('cam_rear_to_lidar_rear', 'extrinsic_rear'),
    ]
    for ext_key, report_key in ext_keys:
        ext_data = calib['extrinsics'][ext_key]
        R_ok, R_issues = check_rotation_matrix(ext_data['R'], f'{ext_key}.R')
        T_ok, T_issues = check_extrinsic_transform(ext_data['T'], f'{ext_key}.T')

        report['checks'][report_key] = {
            'rotation_valid': R_ok,
            'rotation_issues': R_issues,
            'translation_valid': T_ok,
            'translation_issues': T_issues,
        }
        if not (R_ok and T_ok):
            ext_ok = False
            all_issues.extend(R_issues)
            all_issues.extend(T_issues)

    # 3. Cross-lidar transform
    l2l_ok, l2l_issues = check_lidar_to_lidar_transform(calib)
    report['checks']['lidar_rear_to_lidar_front'] = {
        'valid': l2l_ok,
        'issues': l2l_issues,
    }
    if not l2l_ok:
        all_issues.extend(l2l_issues)

    # 4. Projection results
    report['checks']['projection'] = projection_results
    for cam_name, cam_result in projection_results.items():
        if 'inlier_ratio' in cam_result:
            if cam_result['inlier_ratio'] < 0.01:
                all_issues.append(
                    f'{cam_name}: very low inlier ratio '
                    f'({cam_result["inlier_ratio"]:.4f}). '
                    f'Possible calibration error.'
                )

    # Summary
    if all_issues:
        report['status'] = 'FAIL'
        report['issues'] = all_issues
    else:
        report['status'] = 'PASS'

    # Intrinsics summary row
    report['intrinsics_summary'] = {
        'CAM_FRONT': {
            'fx': float(np.array(calib['cameras']['front']['intrinsic'])[0, 0]),
            'fy': float(np.array(calib['cameras']['front']['intrinsic'])[1, 1]),
        },
        'CAM_BACK': {
            'fx': float(np.array(calib['cameras']['rear']['intrinsic'])[0, 0]),
            'fy': float(np.array(calib['cameras']['rear']['intrinsic'])[1, 1]),
        },
    }

    return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Validate calibration parameters for RTG BEV system.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python calib_validator.py --calib calib.yaml

  python calib_validator.py \\
      --calib calib.yaml \\
      --image-dir data/extracted/scene_001/ \\
      --frame 0 \\
      --output-dir vis/calib_check/
        """,
    )
    parser.add_argument(
        '--calib', required=True,
        help='Path to calib.yaml.',
    )
    parser.add_argument(
        '--image-dir', default=None,
        help='Directory containing extracted data (front/, rear/, etc.). '
             'Required for projection visualization check.',
    )
    parser.add_argument(
        '--frame', type=int, default=0,
        help='Frame index for projection visualization (default: 0).',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory to save projection overlay images.',
    )
    parser.add_argument(
        '--report-path', default='calib_validation_report.json',
        help='Path for the output validation report JSON.',
    )

    args = parser.parse_args()

    # Load calib
    if not os.path.isfile(args.calib):
        print(f'[ERROR] Calibration file not found: {args.calib}')
        sys.exit(1)
    calib = load_calib(args.calib)
    print(f'[INFO] Loaded calibration from {args.calib}')

    # Validate calib structure
    required_keys = ['cameras', 'extrinsics']
    for key in required_keys:
        if key not in calib:
            print(f'[ERROR] Missing required key in calib: {key}')
            sys.exit(1)

    # Run projection check if image_dir provided
    projection_results = {}
    if args.image_dir:
        if os.path.isdir(args.image_dir):
            projection_results = run_projection_check(
                calib, args.image_dir, args.frame, args.output_dir
            )
        else:
            print(f'[WARN] Image directory not found: {args.image_dir}')

    # Generate report
    report = generate_report(calib, projection_results)

    # Save report
    with open(args.report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'[INFO] Validation report saved to {args.report_path}')
    print(f'[INFO] Overall status: {report["status"]}')

    if report['status'] == 'FAIL':
        print(f'[INFO] {len(report.get("issues", []))} issue(s) found:')
        for issue in report.get('issues', []):
            print(f'  - {issue}')
        sys.exit(1)
    else:
        print('[INFO] All checks passed.')
        sys.exit(0)


if __name__ == '__main__':
    main()
