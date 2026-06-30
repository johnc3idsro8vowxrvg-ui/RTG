#!/usr/bin/env python3
"""
project_3d_to_2d.py -- Project 3D bounding boxes onto 2D camera views.

Annotation auxiliary tool. Given 3D box annotations and calibration files,
projects 3D boxes onto each camera view and renders visualization images.

This is used to:
  - Validate calibration accuracy (check alignment of projected boxes).
  - Verify annotation correctness (check if 3D boxes match image content).
  - Debug camera-LiDAR fusion issues.

Usage:
  python project_3d_to_2d.py \\
      --ann-file annotations.json \\
      --calib calib.yaml \\
      --image-dir data/extracted/scene_001/ \\
      --output-dir vis/projections/ \\
      --cameras CAM_FRONT CAM_BACK \\
      --frames 0,5,10,20,50
"""

import argparse
import csv
import json
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Math utilities
# ---------------------------------------------------------------------------
def _load_yaml_or_json(path):
    """Load YAML or JSON file."""
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        with open(path, 'r') as f:
            return json.load(f)


def rotation_matrix_from_quaternion(q):
    """Convert quaternion [x, y, z, w] to 3x3 rotation matrix.

    Args:
        q: Quaternion as [x, y, z, w] or [w, x, y, z].

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    if len(q) != 4:
        raise ValueError(f'Quaternion must have 4 elements, got {len(q)}')
    # Assume [x, y, z, w] convention
    x, y, z, w = q
    R = np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=np.float64)
    return R


def build_view_matrix(intrinsic):
    """Build 3x4 view projection matrix from intrinsic (3x3)."""
    K = np.array(intrinsic, dtype=np.float64)
    P = np.zeros((3, 4), dtype=np.float64)
    P[:3, :3] = K
    return P


def camera_to_lidar_transform(R_cl, T_cl):
    """Camera-to-LiDAR transform: camera points are rotated/translated to LiDAR frame.

    Given sensor2lidar (S2L): transforms a point FROM sensor frame TO lidar frame.
      p_lidar = p_sensor @ R_s2l.T + T_s2l

    For projecting lidar->camera, we need the inverse:
      p_camera = (p_lidar - T_s2l) @ R_s2l

    Args:
        R_cl: 3x3 rotation matrix (sensor2lidar_rotation).
        T_cl: 3-element translation (sensor2lidar_translation).

    Returns:
        tuple: (R_lc (3,3), T_lc (3,)) representing lidar-to-camera transform
               where p_camera = p_lidar @ R_lc.T + T_lc
    """
    R_s2l = np.array(R_cl, dtype=np.float64)
    T_s2l = np.array(T_cl, dtype=np.float64).ravel()

    # Invert: lidar_to_camera
    R_l2c = R_s2l  # R_s2l = R_l2c (rotation from lidar to camera)
    T_l2c = -T_s2l @ R_s2l.T  # p_camera = (p_lidar - T_s2l) @ R_s2l

    return R_l2c, T_l2c


def project_points_lidar_to_image(points_lidar, R_l2c, T_l2c, intrinsic):
    """Project 3D points from LiDAR frame to image plane.

    Args:
        points_lidar: (N, 3) array of points in LiDAR frame.
        R_l2c: (3, 3) rotation from lidar to camera.
        T_l2c: (3,) translation from lidar to camera.
        intrinsic: (3, 3) camera intrinsic matrix.

    Returns:
        tuple: (uv (N, 2), depths (N,), in_front_mask (N, bool))
    """
    N = points_lidar.shape[0]
    # Transform to camera frame
    pts_cam = points_lidar @ R_l2c.T + T_l2c  # (N, 3)

    # Depth in camera frame (z)
    depths = pts_cam[:, 2]

    # Only points in front of camera
    in_front = depths > 0.01

    # Project
    uv_h = pts_cam @ intrinsic.T  # (N, 3)
    uv = uv_h[:, :2] / uv_h[:, 2:3]  # normalize

    return uv, depths, in_front


def get_3d_box_corners(x, y, z, w, l, h, yaw):
    """Get 8 corners of a 3D box in LiDAR frame.

    Box definition: center (x, y, z), dimensions (w, l, h), yaw rotation around z-axis.

    Args:
        x, y, z: Center of the box.
        w, l, h: Width, length, height.
        yaw: Rotation around z-axis (radians).

    Returns:
        np.ndarray: (8, 3) corner coordinates.
    """
    # Corners in local frame (centered at origin)
    dx = w / 2.0
    dy = l / 2.0
    dz = h / 2.0
    corners_local = np.array([
        [-dx, -dy, -dz],
        [-dx, -dy,  dz],
        [-dx,  dy, -dz],
        [-dx,  dy,  dz],
        [ dx, -dy, -dz],
        [ dx, -dy,  dz],
        [ dx,  dy, -dz],
        [ dx,  dy,  dz],
    ], dtype=np.float64)

    # Rotation around z-axis
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    Rz = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ], dtype=np.float64)

    corners_rotated = corners_local @ Rz.T
    corners_global = corners_rotated + np.array([x, y, z])

    return corners_global


# ---------------------------------------------------------------------------
# Drawing utilities
# ---------------------------------------------------------------------------
# Color palette for categories
CATEGORY_COLORS = {
    'person': (0, 255, 0),          # Green
    'truck': (255, 0, 0),           # Blue
    'car': (0, 0, 255),             # Red
    'other_obstacle': (255, 255, 0), # Cyan
}
DEFAULT_COLOR = (255, 255, 255)     # White


EDGES = [
    [0, 1], [0, 2], [0, 4],
    [1, 3], [1, 5],
    [2, 3], [2, 6],
    [3, 7],
    [4, 5], [4, 6],
    [5, 7],
    [6, 7],
]

# Edges grouped by face for rendering (front, back, top, bottom)
FACE_EDGES = {
    'front':  [[0, 2], [2, 6], [6, 4], [4, 0]],
    'back':   [[1, 3], [3, 7], [7, 5], [5, 1]],
    'top':    [[2, 3], [3, 7], [7, 6], [6, 2]],
    'bottom': [[0, 1], [1, 5], [5, 4], [4, 0]],
    'left':   [[0, 1], [1, 3], [3, 2], [2, 0]],
    'right':  [[4, 5], [5, 7], [7, 6], [6, 4]],
}


def draw_3d_box_on_image(image, corners_uv, depths, in_front, color,
                          line_width=2, show_depth=True):
    """Draw a projected 3D box on an image.

    Args:
        image: BGR image (numpy array).
        corners_uv: (8, 2) projected corner coordinates.
        depths: (8,) depth values.
        in_front: (8,) boolean mask of which corners are in front of camera.
        color: BGR tuple.
        line_width: Line thickness.
        show_depth: If True, annotate with average depth.

    Returns:
        np.ndarray: Annotated image.
    """
    # Skip if all corners are behind camera
    if not np.any(in_front):
        return image

    img = image.copy()
    corners = corners_uv.astype(np.int32)
    h, w = img.shape[:2]

    def _in_bounds(pt):
        return 0 <= pt[0] < w and 0 <= pt[1] < h

    # Draw edges
    for i, j in EDGES:
        if in_front[i] and in_front[j]:
            pt1 = tuple(corners[i])
            pt2 = tuple(corners[j])
            if _in_bounds(pt1) or _in_bounds(pt2):
                cv2.line(img, pt1, pt2, color, line_width)

    # Draw corners
    for i in range(8):
        if in_front[i]:
            pt = tuple(corners[i])
            if _in_bounds(pt):
                cv2.circle(img, pt, 3, color, -1)

    # Draw center
    center_uv = corners_uv.mean(axis=0).astype(np.int32)
    center_pt = tuple(center_uv)
    if _in_bounds(center_pt):
        cv2.circle(img, center_pt, 5, color, -1)

    # Depth annotation
    if show_depth:
        mean_depth = depths[in_front].mean() if np.any(in_front) else 0
        text = f'{mean_depth:.1f}m'
        text_pos = (center_pt[0] + 8, center_pt[1] - 8)
        if _in_bounds(text_pos):
            cv2.putText(img, text, text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img


# ---------------------------------------------------------------------------
# Main projection logic
# ---------------------------------------------------------------------------
def project_frame(frame_id, annotations, calib, image_dir, output_dir,
                  camera_names=None, show_lidar_projection=True):
    """Project all 3D boxes in a single frame onto camera views.

    Args:
        frame_id: Integer frame index.
        annotations: dict {frame_id: [annotation dicts]}.
        calib: dict from load_calib() or similar.
        image_dir: Directory containing front/ and rear/ subdirectories.
        output_dir: Directory to save visualization images.
        camera_names: List of camera names (e.g., ['CAM_FRONT', 'CAM_BACK']).
        show_lidar_projection: If True, also project LiDAR points onto image.

    Returns:
        dict: Per-camera paths to saved visualization images.
    """
    if camera_names is None:
        camera_names = ['CAM_FRONT', 'CAM_BACK']

    anns = annotations.get(frame_id, [])
    if not anns:
        print(f'[WARN] No annotations for frame {frame_id}')
        return {}

    # Find image files for this frame
    frame_str = f'{frame_id:06d}'

    # Look for images in common patterns
    cam_image_map = {
        'CAM_FRONT': None,
        'CAM_BACK': None,
    }
    for ext in ['.jpg', '.png', '.jpeg']:
        front_candidate = os.path.join(image_dir, 'front', frame_str + ext)
        rear_candidate = os.path.join(image_dir, 'rear', frame_str + ext)
        if cam_image_map['CAM_FRONT'] is None and os.path.isfile(front_candidate):
            cam_image_map['CAM_FRONT'] = front_candidate
        if cam_image_map['CAM_BACK'] is None and os.path.isfile(rear_candidate):
            cam_image_map['CAM_BACK'] = rear_candidate

    # Calibration lookup
    cam_calib = {
        'CAM_FRONT': {
            'intrinsic': calib['cameras']['front']['intrinsic'],
            'R': calib['extrinsics']['cam_front_to_lidar_front']['R'],
            'T': calib['extrinsics']['cam_front_to_lidar_front']['T'],
        },
        'CAM_BACK': {
            'intrinsic': calib['cameras']['rear']['intrinsic'],
            'R': calib['extrinsics']['cam_rear_to_lidar_rear']['R'],
            'T': calib['extrinsics']['cam_rear_to_lidar_rear']['T'],
        },
    }

    output_paths = {}
    for cam_name in camera_names:
        img_path = cam_image_map.get(cam_name)
        if img_path is None:
            print(f'[WARN] No image found for {cam_name} frame {frame_id}')
            continue

        img = cv2.imread(img_path)
        if img is None:
            print(f'[WARN] Failed to read image: {img_path}')
            continue

        cal = cam_calib.get(cam_name)
        if cal is None:
            continue

        intrinsic = np.array(cal['intrinsic'], dtype=np.float64)
        R_l2c, T_l2c = camera_to_lidar_transform(cal['R'], cal['T'])

        # Optionally project LiDAR points
        if show_lidar_projection:
            # Look for lidar point file
            lidar_patterns = [
                os.path.join(image_dir, 'lidar_concat', frame_str + '.bin'),
                os.path.join(image_dir, 'lidar_front', frame_str + '.bin'),
            ]
            for lp in lidar_patterns:
                if os.path.isfile(lp):
                    pts = np.fromfile(lp, dtype=np.float32).reshape(-1, 5)
                    uv, depths, in_front = project_points_lidar_to_image(
                        pts[:, :3], R_l2c, T_l2c, intrinsic
                    )
                    # Draw point cloud on image
                    h, w = img.shape[:2]
                    for j in range(len(uv)):
                        if in_front[j]:
                            u, v = int(uv[j, 0]), int(uv[j, 1])
                            if 0 <= u < w and 0 <= v < h:
                                d = depths[j]
                                # Color by depth: near=red, far=blue
                                d_norm = min(d / 60.0, 1.0)  # normalize to 0-60m
                                r = int(255 * (1 - d_norm))
                                b = int(255 * d_norm)
                                cv2.circle(img, (u, v), 1, (b, 0, r), -1)
                    break
            else:
                print(f'[INFO] No lidar points found for frame {frame_id}, '
                      f'skipping point projection.')

        # Draw 3D boxes
        for ann in anns:
            bbox = ann['bbox_3d']
            cat = ann.get('category', 'other_obstacle')
            color = CATEGORY_COLORS.get(cat, DEFAULT_COLOR)

            corners_3d = get_3d_box_corners(
                bbox['x'], bbox['y'], bbox['z'],
                bbox['w'], bbox['l'], bbox['h'],
                bbox['yaw'],
            )

            uv, depths, in_front = project_points_lidar_to_image(
                corners_3d, R_l2c, T_l2c, intrinsic
            )

            img = draw_3d_box_on_image(
                img, uv, depths, in_front, color,
                line_width=2, show_depth=True,
            )

            # Draw category label near center
            center_uv = uv.mean(axis=0).astype(np.int32)
            h, w = img.shape[:2]
            if 0 <= center_uv[0] < w and 0 <= center_uv[1] < h:
                cv2.putText(img, cat, tuple(center_uv),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Save visualization
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir, f'{frame_str}_{cam_name.lower()}.jpg'
        )
        cv2.imwrite(out_path, img)
        output_paths[cam_name] = out_path
        print(f'[INFO] Saved {out_path}')

    return output_paths


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Project 3D bounding boxes onto 2D camera views.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python project_3d_to_2d.py \\
      --ann-file annotations.json \\
      --calib calib.yaml \\
      --image-dir data/extracted/scene_001/ \\
      --frames 0,10,20,50,100

  python project_3d_to_2d.py \\
      --ann-file annotations.json \\
      --calib calib.yaml \\
      --image-dir data/extracted/scene_001/ \\
      --frames 0-100-10 \\
      --cameras CAM_FRONT
        """,
    )
    parser.add_argument(
        '--ann-file', required=True,
        help='Path to annotations.json containing 3D box annotations.',
    )
    parser.add_argument(
        '--calib', required=True,
        help='Path to calib.yaml.',
    )
    parser.add_argument(
        '--image-dir', required=True,
        help='Directory containing front/ and rear/ subdirectories with images.',
    )
    parser.add_argument(
        '--output-dir', default='vis/projections',
        help='Output directory for visualization images.',

    )
    parser.add_argument(
        '--cameras', nargs='+', default=['CAM_FRONT', 'CAM_BACK'],
        help='Camera names to project onto (default: CAM_FRONT CAM_BACK).',
    )
    parser.add_argument(
        '--frames', default=None,
        help='Frames to visualize. Formats: "0,5,10" or "0-100" or "0-100-10" '
             '(start-end-step). If omitted, projects all annotated frames.',
    )
    parser.add_argument(
        '--no-lidar', action='store_true',
        help='Do not project lidar points onto images.',
    )

    args = parser.parse_args()

    # Load annotations
    with open(args.ann_file, 'r') as f:
        ann_data = json.load(f)
    annotations_list = ann_data.get('annotations', [])
    annotations_by_frame = {}
    for ann in annotations_list:
        fid = ann['frame_id']
        annotations_by_frame.setdefault(fid, []).append(ann)
    print(f'[INFO] Loaded annotations for {len(annotations_by_frame)} frames.')

    # Load calib
    calib = _load_yaml_or_json(args.calib)

    # Parse frame list
    if args.frames:
        frame_ids = _parse_frame_range(args.frames)
    else:
        frame_ids = sorted(annotations_by_frame.keys())
        print(f'[INFO] Processing all {len(frame_ids)} annotated frames.')

    # Process each frame
    for fid in frame_ids:
        print(f'\n[INFO] Processing frame {fid}...')
        project_frame(
            frame_id=fid,
            annotations=annotations_by_frame,
            calib=calib,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            camera_names=args.cameras,
            show_lidar_projection=not args.no_lidar,
        )

    print(f'\n[DONE] Visualizations saved to {args.output_dir}')


def _parse_frame_range(spec):
    """Parse frame range specification.

    Formats:
      "0,5,10,20"          -> [0, 5, 10, 20]
      "0-10"               -> [0, 1, ..., 10]
      "0-100-10"           -> [0, 10, 20, ..., 100]
    """
    spec = spec.strip()
    if ',' in spec:
        return [int(x.strip()) for x in spec.split(',')]
    elif '-' in spec:
        parts = spec.split('-')
        if len(parts) == 2:
            return list(range(int(parts[0]), int(parts[1]) + 1))
        elif len(parts) == 3:
            return list(range(int(parts[0]), int(parts[1]) + 1, int(parts[2])))
    return [int(spec)]


if __name__ == '__main__':
    main()
