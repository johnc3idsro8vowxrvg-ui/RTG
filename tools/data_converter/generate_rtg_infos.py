#!/usr/bin/env python3
"""
generate_rtg_infos.py -- Convert extracted data to CenterPoint-compatible info.pkl.

Converts the output of rosbag_extract into the nuScenes-compatible info.pkl format
that CenterPoint's DataLoader expects.

Inputs:
  data_root/
    front/                    -- Front camera images (*.jpg)
    rear/                     -- Rear camera images (*.jpg)
    lidar_front/              -- Front lidar point clouds (*.bin)
    lidar_rear/               -- Rear lidar point clouds (*.bin)
    timestamps.csv            -- Frame index with timestamps and paths
    calib.yaml                -- Calibration parameters
    annotations.json          -- 3D bounding box annotations (optional, for training)

Outputs:
  rtg_infos_train.pkl         -- Training set info file
  rtg_infos_val.pkl           -- Validation set info file

Usage:
  python generate_rtg_infos.py data/extracted/scene_001/ \\
      --ann-file data/annotations/annotations_train.json \\
      --output-prefix rtg \\
      --val-ratio 0.1 \\
      --continuous-seq scene_001:10-60,5-30 scene_002:0-100
"""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# RTG Project Constants
# ---------------------------------------------------------------------------
RTG_CLASSES = ('person', 'truck', 'car', 'other_obstacle')

# Default class name mapping (annotation format -> RTG class)
DEFAULT_CLASS_MAPPING = {
    'person': 'person',
    'pedestrian': 'person',
    'truck': 'truck',
    'container_truck': 'truck',
    'car': 'car',
    'vehicle': 'car',
    'other': 'other_obstacle',
    'other_obstacle': 'other_obstacle',
    'obstacle': 'other_obstacle',
    'cone': 'other_obstacle',
    'barrier': 'other_obstacle',
    'forklift': 'other_obstacle',
    'agv': 'other_obstacle',
}

# Camera names compatible with CenterPoint
CAM_NAMES = ['CAM_FRONT', 'CAM_BACK']

# Default lidar file path (concatenated point cloud saved during info generation)
LIDAR_CONCAT_FILENAME = 'lidar_concat.bin'


# ---------------------------------------------------------------------------
# Calibration I/O
# ---------------------------------------------------------------------------
def load_calib(calib_path):
    """Load calibration from YAML file.

    Expected YAML structure:
      cameras:
        front:
          intrinsic: [[fx,0,cx],[0,fy,cy],[0,0,1]]
          width: 1920
          height: 1080
        rear:
          intrinsic: [[fx,0,cx],[0,fy,cy],[0,0,1]]
          width: 1920
          height: 1080

      extrinsics:
        cam_front_to_lidar_front:
          R: [[r11,r12,r13],[r21,r22,r23],[r31,r32,r33]]
          T: [tx, ty, tz]
        cam_rear_to_lidar_rear:
          R: [[...]]
          T: [tx, ty, tz]
        lidar_rear_to_lidar_front:
          R: [[...]]
          T: [tx, ty, tz]

    Falls back to JSON if YAML is not available.

    Args:
        calib_path: Path to calib.yaml.

    Returns:
        dict: Parsed calibration.
    """
    data = _load_yaml_or_json(calib_path)
    return data


def _load_yaml_or_json(path):
    """Load a YAML or JSON file."""
    try:
        import yaml
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except ImportError:
        with open(path, 'r') as f:
            return json.load(f)


# ---------------------------------------------------------------------------
# Annotation I/O
# ---------------------------------------------------------------------------
def load_annotations(ann_path):
    """Load 3D bounding box annotations from JSON.

    Expected JSON structure:
      {
        "annotations": [
          {
            "frame_id": 0,
            "category": "truck",
            "bbox_3d": {
              "x": 10.5, "y": 2.3, "z": 1.5,
              "w": 2.5, "l": 12.0, "h": 4.0,
              "yaw": 0.05
            },
            "occlusion": 0,
            "track_id": 1
          },
          ...
        ],
        "metadata": {
          "categories": ["person", "truck", "car", "other_obstacle"],
          "num_frames": 4500,
          ...
        }
      }

    Args:
        ann_path: Path to annotations.json.

    Returns:
        dict: Parsed annotations grouped by frame_id.
    """
    with open(ann_path, 'r') as f:
        data = json.load(f)

    anns_by_frame = defaultdict(list)
    for ann in data.get('annotations', []):
        anns_by_frame[ann['frame_id']].append(ann)

    metadata = data.get('metadata', {})
    return dict(anns_by_frame), metadata


# ---------------------------------------------------------------------------
# Timestamp I/O
# ---------------------------------------------------------------------------
def load_timestamps(csv_path):
    """Load timestamps.csv.

    Args:
        csv_path: Path to timestamps.csv.

    Returns:
        list of dict: Each row as a dictionary.
    """
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['frame_id'] = int(row['frame_id'])
            row['timestamp'] = float(row['timestamp'])
            row['num_lidar_front_pts'] = int(row.get('num_lidar_front_pts', 0))
            row['num_lidar_rear_pts'] = int(row.get('num_lidar_rear_pts', 0))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Calibration matrix utilities
# ---------------------------------------------------------------------------
def build_extrinsic_matrix(R, T):
    """Build 4x4 extrinsic matrix from rotation (3x3) and translation (3,).

    Args:
        R: 3x3 rotation matrix (list of lists or np.ndarray).
        T: 3-element translation vector (list or np.ndarray).

    Returns:
        np.ndarray: 4x4 homogeneous transform matrix.
    """
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(R, dtype=np.float64)
    mat[:3, 3] = np.array(T, dtype=np.float64)
    return mat


def rotation_matrix_to_list(R):
    """Ensure rotation is a list-of-lists format."""
    return np.array(R, dtype=np.float64).tolist()


def translation_to_list(T):
    """Ensure translation is a flat list of 3 floats."""
    arr = np.array(T, dtype=np.float64).ravel()
    return arr[:3].tolist()


# ---------------------------------------------------------------------------
# Lidar concatenation
# ---------------------------------------------------------------------------
def concat_lidar_points(lidar_front_path, lidar_rear_path,
                        lidar_rear_to_front_R=None,
                        lidar_rear_to_front_T=None):
    """Concatenate front and rear lidar point clouds into a unified frame.

    If rear-to-front extrinsics are provided, rear points are transformed
    into the front lidar coordinate system before concatenation.

    Args:
        lidar_front_path: Absolute path to front lidar .bin file.
        lidar_rear_path: Absolute path to rear lidar .bin file.
        lidar_rear_to_front_R: 3x3 rotation (rear -> front).
        lidar_rear_to_front_T: 3-element translation (rear -> front).

    Returns:
        np.ndarray: Concatenated point cloud (M, 5).
    """
    pts_front = np.fromfile(lidar_front_path, dtype=np.float32).reshape(-1, 5)
    pts_rear = np.fromfile(lidar_rear_path, dtype=np.float32).reshape(-1, 5)

    if lidar_rear_to_front_R is not None and lidar_rear_to_front_T is not None:
        R = np.array(lidar_rear_to_front_R, dtype=np.float64)
        T = np.array(lidar_rear_to_front_T, dtype=np.float64).ravel()
        # Transform rear points (xyz only)
        pts_rear_xyz = pts_rear[:, :3].astype(np.float64)
        pts_rear_xyz = pts_rear_xyz @ R.T + T
        pts_rear[:, :3] = pts_rear_xyz.astype(np.float32)

    return np.concatenate([pts_front, pts_rear], axis=0)


# ---------------------------------------------------------------------------
# GT box conversion
# ---------------------------------------------------------------------------
def convert_annotations_to_gt_boxes(anns, class_mapping=None):
    """Convert annotation dicts to CenterPoint gt_boxes/gt_names format.

    gt_boxes format: [x, y, z, w, l, h, yaw] (SECOND format, 7 values)
    where yaw is -theta - pi/2 (nuScenes convention).

    Args:
        anns: list of annotation dicts for one frame.
        class_mapping: dict mapping annotation category -> RTG class.

    Returns:
        tuple: (gt_boxes np.ndarray [M,7], gt_names np.ndarray [M,])
    """
    if class_mapping is None:
        class_mapping = DEFAULT_CLASS_MAPPING

    boxes = []
    names = []
    for ann in anns:
        bbox = ann['bbox_3d']
        # bbox_3d keys: x, y, z, w, l, h, yaw
        # GT boxes in SECOND format: [x, y, z, w, l, h, yaw]
        box = [
            bbox['x'], bbox['y'], bbox['z'],
            bbox['w'], bbox['l'], bbox['h'],
            bbox['yaw'],
        ]
        cat = ann.get('category', 'other_obstacle')
        mapped_cat = class_mapping.get(cat, 'other_obstacle')
        if mapped_cat not in RTG_CLASSES:
            mapped_cat = 'other_obstacle'
        boxes.append(box)
        names.append(mapped_cat)

    if not boxes:
        return (
            np.zeros((0, 7), dtype=np.float32),
            np.array([], dtype='<U32'),
        )
    return np.array(boxes, dtype=np.float32), np.array(names, dtype='<U32')


# ---------------------------------------------------------------------------
# Info generation
# ---------------------------------------------------------------------------
def generate_infos(data_root,
                   timestamps,
                   annotations_by_frame,
                   calib,
                   output_path,
                   selected_indices=None):
    """Generate info.pkl compatible with CenterPoint.

    Args:
        data_root: Absolute path to the extracted data directory.
        timestamps: list of dict from load_timestamps().
        annotations_by_frame: dict {frame_id: [annotation dicts]}.
        calib: dict from load_calib().
        output_path: Path to write the .pkl file.
        selected_indices: Optional list of frame indices to include.
            If None, all frames are included.

    Returns:
        list of dict: Generated info entries.
    """
    if selected_indices is None:
        selected_indices = list(range(len(timestamps)))

    # Extract calibration parameters
    cam_front_intrinsic = calib['cameras']['front']['intrinsic']
    cam_rear_intrinsic = calib['cameras']['rear']['intrinsic']

    cam_front_to_lidar_R = calib['extrinsics']['cam_front_to_lidar_front']['R']
    cam_front_to_lidar_T = calib['extrinsics']['cam_front_to_lidar_front']['T']
    cam_rear_to_lidar_R = calib['extrinsics']['cam_rear_to_lidar_rear']['R']
    cam_rear_to_lidar_T = calib['extrinsics']['cam_rear_to_lidar_rear']['T']

    # Rear lidar to front lidar extrinsics (for point cloud concatenation)
    lr2lf_R = None
    lr2lf_T = None
    if 'lidar_rear_to_lidar_front' in calib.get('extrinsics', {}):
        lr2lf_R = calib['extrinsics']['lidar_rear_to_lidar_front']['R']
        lr2lf_T = calib['extrinsics']['lidar_rear_to_lidar_front']['T']

    # Build lidar-to-world transform
    # We set the front lidar position as the world origin (identity)
    lidar2world = np.eye(4).tolist()

    infos = []
    for idx in selected_indices:
        ts_row = timestamps[idx]
        frame_id = ts_row['frame_id']

        # Generate concatenated lidar file
        lidar_front_abs = os.path.join(
            data_root, ts_row['lidar_front_path']
        )
        lidar_rear_abs = os.path.join(
            data_root, ts_row['lidar_rear_path']
        )
        concat_pts = concat_lidar_points(
            lidar_front_abs, lidar_rear_abs, lr2lf_R, lr2lf_T
        )
        concat_path_rel = f'lidar_concat/{frame_id:06d}.bin'
        concat_path_abs = os.path.join(data_root, concat_path_rel)
        os.makedirs(os.path.dirname(concat_path_abs), exist_ok=True)
        concat_pts.astype(np.float32).tofile(concat_path_abs)

        # Build info dict
        info = {
            'token': f'rtg_{frame_id:06d}',
            'timestamp': ts_row['timestamp'],

            # Lidar
            'lidar_path': concat_path_rel,
            'num_lidar_pts': concat_pts.shape[0],

            # Cameras (2 views)
            'cams': {
                'CAM_FRONT': {
                    'data_path': ts_row['cam_front_path'],
                    'sensor2lidar_rotation': rotation_matrix_to_list(
                        cam_front_to_lidar_R
                    ),
                    'sensor2lidar_translation': translation_to_list(
                        cam_front_to_lidar_T
                    ),
                    'cam_intrinsic': cam_front_intrinsic,
                },
                'CAM_BACK': {
                    'data_path': ts_row['cam_rear_path'],
                    'sensor2lidar_rotation': rotation_matrix_to_list(
                        cam_rear_to_lidar_R
                    ),
                    'sensor2lidar_translation': translation_to_list(
                        cam_rear_to_lidar_T
                    ),
                    'cam_intrinsic': cam_rear_intrinsic,
                },
            },

            # Lidar-to-world transform (identity: front lidar = world origin)
            'lidar2world': lidar2world,

            # For compatibility with CenterPoint's NuScenesDataset
            'lidar2ego_translation': [0.0, 0.0, 0.0],
            'lidar2ego_rotation': [1.0, 0.0, 0.0, 0.0],
            'ego2global_translation': [0.0, 0.0, 0.0],
            'ego2global_rotation': [1.0, 0.0, 0.0, 0.0],

            # Sweeps (single frame mode: empty list)
            'sweeps': [],
        }

        # GT annotations (if available)
        anns = annotations_by_frame.get(frame_id, [])
        if anns:
            gt_boxes, gt_names = convert_annotations_to_gt_boxes(anns)
            info['gt_boxes'] = gt_boxes
            info['gt_names'] = gt_names
            info['gt_velocity'] = np.zeros((len(gt_boxes), 2), dtype=np.float32)
            info['num_lidar_pts'] = np.ones(len(gt_boxes), dtype=np.int32)
            info['num_radar_pts'] = np.zeros(len(gt_boxes), dtype=np.int32)
            info['valid_flag'] = np.ones(len(gt_boxes), dtype=bool)
        else:
            # No annotations: provide empty arrays for compatibility
            info['gt_boxes'] = np.zeros((0, 7), dtype=np.float32)
            info['gt_names'] = np.array([], dtype='<U32')
            info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
            info['num_lidar_pts'] = np.zeros(0, dtype=np.int32)
            info['num_radar_pts'] = np.zeros(0, dtype=np.int32)
            info['valid_flag'] = np.zeros(0, dtype=bool)

        infos.append(info)

    # Save pkl
    metadata = {
        'version': 'rtg_v1.0',
        'num_cameras': 2,
        'camera_names': CAM_NAMES,
        'categories': list(RTG_CLASSES),
        'data_root': data_root,
    }
    data = {
        'infos': infos,
        'metadata': metadata,
    }
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f'[INFO] Saved {len(infos)} infos to {output_path}')
    return infos


# ---------------------------------------------------------------------------
# Train/Val Split
# ---------------------------------------------------------------------------
def parse_continuous_sequences(seq_spec):
    """Parse continuous sequence specification string.

    Format: "scene_001:10-60,5-30 scene_002:0-100"
    Meaning: scene_001 frames [10,60] and [5,30] are continuous sequences;
             scene_002 frames [0,100] is a continuous sequence.
    Frame ranges are inclusive.

    Args:
        seq_spec: Space-separated string of "scene:range1,range2,...".

    Returns:
        set of int: All frame indices belonging to continuous sequences.
    """
    if not seq_spec:
        return set()

    continuous_indices = set()
    for part in seq_spec.split():
        if ':' not in part:
            continue
        scene_name, ranges = part.split(':', 1)
        for rng in ranges.split(','):
            rng = rng.strip()
            if '-' in rng:
                start, end = rng.split('-')
                continuous_indices.update(
                    range(int(start), int(end) + 1)
                )
            else:
                continuous_indices.add(int(rng))
    return continuous_indices


def split_train_val(infos, val_ratio=0.1, continuous_seq_indices=None,
                    random_seed=42):
    """Split infos into training and validation sets.

    Strategy:
      1. Continuous sequences are placed entirely in the validation set
         (for evaluating tracking and warning continuity).
      2. Remaining frames are randomly split as train/val.

    Args:
        infos: list of info dicts.
        val_ratio: Fraction of non-continuous data to use for validation.
        continuous_seq_indices: set of info indices belonging to continuous
            sequences. These go entirely to val.
        random_seed: Random seed for reproducibility.

    Returns:
        tuple: (train_infos, val_infos) as lists of info dicts.
    """
    rng = np.random.RandomState(random_seed)

    if continuous_seq_indices is None:
        continuous_seq_indices = set()

    n_total = len(infos)
    indices = list(range(n_total))

    # Separate continuous sequence indices from the rest
    continuous_set = set()
    non_continuous = []
    for i in indices:
        if i in continuous_seq_indices:
            continuous_set.add(i)
        else:
            non_continuous.append(i)

    # Shuffle non-continuous
    rng.shuffle(non_continuous)

    # Split non-continuous
    n_val_non_cont = int(len(non_continuous) * val_ratio)
    val_non_cont = set(non_continuous[:n_val_non_cont])
    train_non_cont = set(non_continuous[n_val_non_cont:])

    # Build final splits
    val_indices = continuous_set | val_non_cont
    train_indices = train_non_cont

    train_infos = [infos[i] for i in sorted(train_indices)]
    val_infos = [infos[i] for i in sorted(val_indices)]

    print(f'[INFO] Split: {len(train_infos)} train, {len(val_infos)} val '
          f'(continuous val: {len(continuous_set)}, '
          f'sampled val: {len(val_non_cont)})')
    return train_infos, val_infos


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Generate CenterPoint-compatible info.pkl from RTG data.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_rtg_infos.py data/extracted/scene_001/ \\
      --ann-file data/annotations/annotations.json \\
      --output-prefix rtg

  python generate_rtg_infos.py data/extracted/scene_001/ \\
      --ann-file data/annotations/annotations.json \\
      --val-ratio 0.1 \\
      --continuous-seq "scene_001:10-60,5-30"

  python generate_rtg_infos.py data/extracted/ \\
      --ann-file data/annotations/annotations_train.json \\
      --output-dir data/ \\
      --output-prefix rtg \\
      --no-split
        """,
    )
    parser.add_argument(
        'data_root',
        help='Root directory of extracted data (contains front/, rear/, '
             'lidar_front/, lidar_rear/, timestamps.csv, calib.yaml).',
    )
    parser.add_argument(
        '--ann-file', default=None,
        help='Path to annotations.json (optional). If omitted, no GT is included.',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Output directory for .pkl files. Default: same as data_root.',
    )
    parser.add_argument(
        '--output-prefix', default='rtg',
        help='Output file prefix (default: rtg).',
    )
    parser.add_argument(
        '--val-ratio', type=float, default=0.1,
        help='Fraction of non-continuous data for validation (default: 0.1).',
    )
    parser.add_argument(
        '--continuous-seq',
        default=None,
        help='Continuous sequence spec: "scene:start-end,start-end ..."',
    )
    parser.add_argument(
        '--no-split', action='store_true',
        help='Do not split; generate a single info file.',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for train/val split (default: 42).',
    )

    args = parser.parse_args()

    data_root = os.path.abspath(args.data_root)
    output_dir = os.path.abspath(args.output_dir) if args.output_dir else data_root
    os.makedirs(output_dir, exist_ok=True)

    # Load inputs
    calib_path = os.path.join(data_root, 'calib.yaml')
    if not os.path.isfile(calib_path):
        print(f'[ERROR] calib.yaml not found at {calib_path}')
        sys.exit(1)
    calib = load_calib(calib_path)

    csv_path = os.path.join(data_root, 'timestamps.csv')
    if not os.path.isfile(csv_path):
        print(f'[ERROR] timestamps.csv not found at {csv_path}')
        sys.exit(1)
    timestamps = load_timestamps(csv_path)
    print(f'[INFO] Loaded {len(timestamps)} frames from timestamps.csv')

    # Load annotations
    annotations_by_frame = {}
    if args.ann_file:
        if os.path.isfile(args.ann_file):
            annotations_by_frame, ann_meta = load_annotations(args.ann_file)
            print(f'[INFO] Loaded annotations for '
                  f'{len(annotations_by_frame)} frames')
        else:
            print(f'[WARN] Annotation file not found: {args.ann_file}')

    # Parse continuous sequences
    continuous_set = parse_continuous_sequences(args.continuous_seq or '')
    if continuous_set:
        print(f'[INFO] Continuous sequence frames: {len(continuous_set)}')

    # Generate all infos
    all_output = os.path.join(
        output_dir, f'{args.output_prefix}_infos_all.pkl'
    )
    infos = generate_infos(
        data_root=data_root,
        timestamps=timestamps,
        annotations_by_frame=annotations_by_frame,
        calib=calib,
        output_path=all_output,
    )

    if args.no_split:
        # Rename _all to _train for convenience
        train_path = os.path.join(
            output_dir, f'{args.output_prefix}_infos_train.pkl'
        )
        os.rename(all_output, train_path)
        print(f'[INFO] Single file mode: {train_path}')
    else:
        # Split into train/val
        train_infos, val_infos = split_train_val(
            infos,
            val_ratio=args.val_ratio,
            continuous_seq_indices=continuous_set,
            random_seed=args.seed,
        )

        # Write train
        train_path = os.path.join(
            output_dir, f'{args.output_prefix}_infos_train.pkl'
        )
        metadata = {
            'version': 'rtg_v1.0',
            'num_cameras': 2,
            'camera_names': CAM_NAMES,
            'categories': list(RTG_CLASSES),
            'data_root': data_root,
        }
        with open(train_path, 'wb') as f:
            pickle.dump({'infos': train_infos, 'metadata': metadata}, f)
        print(f'[INFO] Train: {len(train_infos)} infos -> {train_path}')

        # Write val
        val_path = os.path.join(
            output_dir, f'{args.output_prefix}_infos_val.pkl'
        )
        with open(val_path, 'wb') as f:
            pickle.dump({'infos': val_infos, 'metadata': metadata}, f)
        print(f'[INFO] Val: {len(val_infos)} infos -> {val_path}')

        # Remove intermediate all file
        if os.path.exists(all_output):
            os.remove(all_output)

    print('[DONE] Info generation complete.')


if __name__ == '__main__':
    main()
