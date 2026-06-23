#!/usr/bin/env python3
"""
rosbag_extract.py -- Extract images and point clouds from ROS1 rosbag.

Usage:
  python rosbag_extract.py input.bag -o output_dir/ \\
      --camera-front-topic /camera/front/image_raw \\
      --camera-rear-topic /camera/rear/image_raw \\
      --lidar-front-topic /lidar/os1_front/points \\
      --lidar-rear-topic /lidar/os1_rear/points \\
      --sync-window 0.05 \\
      --extract-tf

Output directory structure:
  output_dir/
    front/<timestamp>.jpg
    rear/<timestamp>.jpg
    lidar_front/<timestamp>.bin
    lidar_rear/<timestamp>.bin
    timestamps.csv
    calib.yaml        (if --extract-tf)
    tf_static.yaml    (if --extract-tf)
    quality_report.json
"""

import argparse
import csv
import json
import os
import sys
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Graceful degradation for ROS1 dependencies
# ---------------------------------------------------------------------------
_ROS_AVAILABLE = False
_rosbag_module = None
_cv_bridge_module = None
_sensor_msgs_module = None
_tf2_msgs_module = None
_rospy_module = None
_genpy_module = None

try:
    import rosbag
    _rosbag_module = rosbag
    _ROS_AVAILABLE = True
except ImportError:
    pass

try:
    from cv_bridge import CvBridge
    _cv_bridge_module = True
except ImportError:
    pass

try:
    from sensor_msgs.msg import Image, PointCloud2
    _sensor_msgs_module = True
except ImportError:
    pass

try:
    from tf2_msgs.msg import TFMessage
    _tf2_msgs_module = True
except ImportError:
    pass

try:
    import rospy
    _rospy_module = rospy
except ImportError:
    pass

try:
    import genpy
    _genpy_module = genpy
except ImportError:
    pass


def _check_ros():
    """Raise RuntimeError if ROS dependencies are missing."""
    if not _ROS_AVAILABLE:
        raise RuntimeError(
            "ROS1 rosbag library not available. "
            "Install with: pip install rosbag (or use system ROS1 environment)."
        )
    if not _cv_bridge_module:
        raise RuntimeError(
            "cv_bridge not available. Install with ROS1 or "
            "pip install cv_bridge."
        )
    if not _sensor_msgs_module:
        raise RuntimeError(
            "sensor_msgs not available. Install with ROS1 environment."
        )


# ---------------------------------------------------------------------------
# PointCloud2 decoding (without ROS deserialization overhead)
# ---------------------------------------------------------------------------
def _decode_pointcloud2(msg):
    """Decode a sensor_msgs/PointCloud2 message into an (N, 5) float32 array.

    Extracts fields: x, y, z, intensity, timestamp (if available).
    For Ouster OS1 bags, uses reflectivity as intensity and t as timestamp.

    Args:
        msg: sensor_msgs.msg.PointCloud2 message.

    Returns:
        np.ndarray of shape (N, 5) with columns [x, y, z, intensity, timestamp].
        Missing fields are filled with zeros.
    """
    from sensor_msgs.point_cloud2 import read_points

    available = {field.name for field in msg.fields}
    if not {'x', 'y', 'z'}.issubset(available):
        return np.zeros((0, 5), dtype=np.float32)

    intensity_field = None
    if 'intensity' in available:
        intensity_field = 'intensity'
    elif 'reflectivity' in available:
        intensity_field = 'reflectivity'

    timestamp_field = None
    if 'timestamp' in available:
        timestamp_field = 'timestamp'
    elif 't' in available:
        timestamp_field = 't'

    read_fields = ['x', 'y', 'z']
    if intensity_field is not None:
        read_fields.append(intensity_field)
    if timestamp_field is not None:
        read_fields.append(timestamp_field)

    points_list = list(read_points(msg, field_names=read_fields, skip_nans=True))
    if not points_list:
        return np.zeros((0, 5), dtype=np.float32)

    arr = np.array(points_list, dtype=np.float32)
    N = arr.shape[0]
    result = np.zeros((N, 5), dtype=np.float32)
    result[:, :3] = arr[:, :3]
    col = 3
    if intensity_field is not None:
        result[:, 3] = arr[:, col]
        col += 1
    if timestamp_field is not None:
        result[:, 4] = arr[:, col]
    return result


# ---------------------------------------------------------------------------
# TF extraction helpers
# ---------------------------------------------------------------------------
def _extract_tf_messages(bag, output_dir):
    """Extract /tf and /tf_static transforms from rosbag.

    Args:
        bag: Opened rosbag.Bag instance.
        output_dir: Output directory string.

    Returns:
        dict: {frame_id: {child_frame_id: {translation: [x,y,z], rotation: [x,y,z,w]}}}
    """
    tf_data = []         # timestamped transforms
    tf_static_data = []  # static transforms

    for topic, msg, t in bag.read_messages(topics=['/tf', '/tf_static']):
        for transform in msg.transforms:
            entry = {
                'header': {
                    'stamp': transform.header.stamp.to_sec(),
                    'frame_id': transform.header.frame_id,
                },
                'child_frame_id': transform.child_frame_id,
                'translation': {
                    'x': transform.transform.translation.x,
                    'y': transform.transform.translation.y,
                    'z': transform.transform.translation.z,
                },
                'rotation': {
                    'x': transform.transform.rotation.x,
                    'y': transform.transform.rotation.y,
                    'z': transform.transform.rotation.z,
                    'w': transform.transform.rotation.w,
                },
            }
            if topic == '/tf':
                tf_data.append(entry)
            else:
                tf_static_data.append(entry)

    # Write TF data
    if tf_data:
        tf_out = os.path.join(output_dir, 'tf.yaml')
        _save_yaml(tf_out, {'transforms': tf_data})
        print(f'[TF] Saved {len(tf_data)} transforms to {tf_out}')

    if tf_static_data:
        tf_static_out = os.path.join(output_dir, 'tf_static.yaml')
        _save_yaml(tf_static_out, {'transforms': tf_static_data})
        print(f'[TF] Saved {len(tf_static_data)} static transforms to {tf_static_out}')

    # Build a combined lookup table
    combined = _build_tf_lookup(tf_data, tf_static_data)
    return combined


def _build_tf_lookup(tf_data, tf_static_data):
    """Build per-frame transform lookup from TF messages.

    Returns:
        dict: {timestamp: {frame: {parent, translation, rotation}}}
    """
    lookup = defaultdict(dict)
    all_entries = tf_static_data + tf_data
    for entry in all_entries:
        ts = entry['header']['stamp']
        child = entry['child_frame_id']
        parent = entry['header']['frame_id']
        lookup[ts][child] = {
            'parent': parent,
            'translation': [
                entry['translation']['x'],
                entry['translation']['y'],
                entry['translation']['z'],
            ],
            'rotation': [
                entry['rotation']['x'],
                entry['rotation']['y'],
                entry['rotation']['z'],
                entry['rotation']['w'],
            ],
        }
    return dict(lookup)


def _save_yaml(path, data):
    """Save data as YAML. Falls back to JSON if yaml is not available."""
    try:
        import yaml
        with open(path, 'w') as f:
            yaml.safe_dump(data, f, default_flow_style=False)
    except ImportError:
        # Fallback: save as JSON with .yaml extension
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)


# ---------------------------------------------------------------------------
# Main Extractor Class
# ---------------------------------------------------------------------------
class RosbagExtractor:
    """Extract synchronized image and point cloud data from a ROS1 rosbag.

    Attributes:
        bag_path: Path to the input rosbag file.
        output_dir: Root output directory.
        sync_window_sec: Synchronization window in seconds (default: 0.050).
        bridge: CvBridge instance for image conversion.
    """

    def __init__(self, bag_path, output_dir, sync_window_sec=0.050):
        _check_ros()
        self.bag_path = bag_path
        self.output_dir = output_dir
        self.sync_window_sec = sync_window_sec
        self.bridge = CvBridge()

        self._sync_groups = []
        self._stats = {
            'total_cam_front': 0,
            'total_cam_rear': 0,
            'total_lidar_front': 0,
            'total_lidar_rear': 0,
            'synced_groups': 0,
            'missed_cam_front': 0,
            'missed_cam_rear': 0,
            'missed_lidar_front': 0,
            'missed_lidar_rear': 0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def extract(self,
                front_camera_topic,
                rear_camera_topic,
                front_lidar_topic,
                rear_lidar_topic,
                extract_tf=False,
                compress_images=True):
        """Main extraction pipeline.

        Args:
            front_camera_topic: ROS topic name for front camera.
            rear_camera_topic: ROS topic name for rear camera.
            front_lidar_topic: ROS topic name for front lidar.
            rear_lidar_topic: ROS topic name for rear lidar.
            extract_tf: Whether to extract TF transforms.
            compress_images: Whether to save images as JPEG (True) or PNG (False).
        """
        print(f'[INFO] Opening rosbag: {self.bag_path}')
        bag = rosbag.Bag(self.bag_path)

        # 1. Parse bag info and collect messages
        print('[INFO] Collecting messages by topic...')
        groups = self._collect_and_sync(
            bag,
            front_camera_topic,
            rear_camera_topic,
            front_lidar_topic,
            rear_lidar_topic,
        )
        self._sync_groups = groups

        # 2. Create output directories
        for subdir in ['front', 'rear', 'lidar_front', 'lidar_rear']:
            os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)

        # 3. Write extracted data
        print(f'[INFO] Writing {len(groups)} synchronized groups...')
        timestamps_rows = self._write_frames(groups, compress_images)

        # 4. Write timestamps.csv
        self._write_timestamps_csv(timestamps_rows)

        # 5. Extract TF (if requested)
        if extract_tf:
            print('[INFO] Extracting TF transforms...')
            _extract_tf_messages(bag, self.output_dir)

        # 6. Generate quality report
        self._generate_quality_report(bag, groups)

        bag.close()
        print(f'[DONE] Extraction complete. Output: {self.output_dir}')

    # ------------------------------------------------------------------
    # Soft-sync collection
    # ------------------------------------------------------------------
    def _collect_and_sync(self, bag, cam_f_topic, cam_r_topic,
                          lidar_f_topic, lidar_r_topic):
        """Collect messages from all 4 topics and perform soft-sync.

        Strategy:
            - Build per-topic list of (timestamp, msg).
            - Use the lidar front topic as the reference timeline.
            - For each reference message, find the closest message on each
              other topic within sync_window_sec.
            - Skip groups where any sensor is missing.

        Returns:
            list of dict: Each dict has keys 'cam_front', 'cam_rear',
            'lidar_front', 'lidar_rear' with (timestamp, msg) tuples.
        """
        # Collect
        cam_f_msgs = []
        cam_r_msgs = []
        lidar_f_msgs = []
        lidar_r_msgs = []

        topics_to_collect = [
            cam_f_topic, cam_r_topic, lidar_f_topic, lidar_r_topic
        ]

        for topic, msg, t in bag.read_messages(topics=topics_to_collect):
            ts = t.to_sec()
            entry = (ts, msg)
            if topic == cam_f_topic:
                cam_f_msgs.append(entry)
            elif topic == cam_r_topic:
                cam_r_msgs.append(entry)
            elif topic == lidar_f_topic:
                lidar_f_msgs.append(entry)
            elif topic == lidar_r_topic:
                lidar_r_msgs.append(entry)

        self._stats['total_cam_front'] = len(cam_f_msgs)
        self._stats['total_cam_rear'] = len(cam_r_msgs)
        self._stats['total_lidar_front'] = len(lidar_f_msgs)
        self._stats['total_lidar_rear'] = len(lidar_r_msgs)

        # Use lidar front as reference
        reference = lidar_f_msgs
        if len(reference) == 0:
            print('[WARN] No lidar_front messages. Trying camera_front as reference.')
            reference = cam_f_msgs
        if len(reference) == 0:
            print('[ERROR] No reference messages found.')
            return []

        sync_groups = []
        used_cam_f = set()
        used_cam_r = set()
        used_lidar_r = set()

        for ref_ts, ref_msg in reference:
            # Find closest on other topics
            cam_f_match = self._find_closest(cam_f_msgs, ref_ts, used_cam_f)
            cam_r_match = self._find_closest(cam_r_msgs, ref_ts, used_cam_r)
            lidar_r_match = self._find_closest(lidar_r_msgs, ref_ts, used_lidar_r)

            # Check all within sync window
            valid = True
            if cam_f_match is None:
                self._stats['missed_cam_front'] += 1
                valid = False
            if cam_r_match is None:
                self._stats['missed_cam_rear'] += 1
                valid = False
            if lidar_r_match is None:
                self._stats['missed_lidar_rear'] += 1
                valid = False
            # Reference (lidar_front) is always present by construction

            if valid:
                sync_groups.append({
                    'cam_front': cam_f_match,
                    'cam_rear': cam_r_match,
                    'lidar_front': (ref_ts, ref_msg),
                    'lidar_rear': lidar_r_match,
                })
                used_cam_f.add(cam_f_match[0])
                used_cam_r.add(cam_r_match[0])
                used_lidar_r.add(lidar_r_match[0])

        self._stats['synced_groups'] = len(sync_groups)

        # Handle missed ref messages
        ref_count = len(reference)
        self._stats['missed_lidar_front'] = ref_count - len(sync_groups)

        return sync_groups

    def _find_closest(self, msg_list, ref_ts, used_set):
        """Find the message in msg_list nearest to ref_ts within the sync window.

        Args:
            msg_list: list of (timestamp, msg).
            ref_ts: Reference timestamp.
            used_set: Set of timestamps already assigned.

        Returns:
            Tuple (timestamp, msg) or None.
        """
        best = None
        best_dt = float('inf')
        for ts, msg in msg_list:
            if ts in used_set:
                continue
            dt = abs(ts - ref_ts)
            if dt < best_dt:
                best_dt = dt
                best = (ts, msg)
        if best is not None and best_dt <= self.sync_window_sec:
            return best
        return None

    # ------------------------------------------------------------------
    # Write frames
    # ------------------------------------------------------------------
    def _write_frames(self, groups, compress_images):
        """Write image and point cloud files for each sync group.

        Returns:
            list of dict: Rows for timestamps.csv.
        """
        ext = '.jpg' if compress_images else '.png'
        timestamps_rows = []

        for idx, group in enumerate(groups):
            frame_name = f'{idx:06d}'

            # Camera front
            _, cam_f_msg = group['cam_front']
            cam_f_img = self.bridge.imgmsg_to_cv2(
                cam_f_msg, desired_encoding='bgr8'
            )
            cam_f_path = os.path.join(self.output_dir, 'front', frame_name + ext)
            cv2.imwrite(cam_f_path, cam_f_img)

            # Camera rear
            _, cam_r_msg = group['cam_rear']
            cam_r_img = self.bridge.imgmsg_to_cv2(
                cam_r_msg, desired_encoding='bgr8'
            )
            cam_r_path = os.path.join(self.output_dir, 'rear', frame_name + ext)
            cv2.imwrite(cam_r_path, cam_r_img)

            # Lidar front
            _, lidar_f_msg = group['lidar_front']
            pts_f = _decode_pointcloud2(lidar_f_msg)
            lidar_f_path = os.path.join(
                self.output_dir, 'lidar_front', frame_name + '.bin'
            )
            pts_f.astype(np.float32).tofile(lidar_f_path)

            # Lidar rear
            _, lidar_r_msg = group['lidar_rear']
            pts_r = _decode_pointcloud2(lidar_r_msg)
            lidar_r_path = os.path.join(
                self.output_dir, 'lidar_rear', frame_name + '.bin'
            )
            pts_r.astype(np.float32).tofile(lidar_r_path)

            # Timestamp row
            timestamps_rows.append({
                'frame_id': idx,
                'frame_name': frame_name,
                'timestamp': group['cam_front'][0],  # use cam_front ts as canonical
                'cam_front_ts': group['cam_front'][0],
                'cam_rear_ts': group['cam_rear'][0],
                'lidar_front_ts': group['lidar_front'][0],
                'lidar_rear_ts': group['lidar_rear'][0],
                'cam_front_path': f'front/{frame_name}{ext}',
                'cam_rear_path': f'rear/{frame_name}{ext}',
                'lidar_front_path': f'lidar_front/{frame_name}.bin',
                'lidar_rear_path': f'lidar_rear/{frame_name}.bin',
                'num_lidar_front_pts': pts_f.shape[0],
                'num_lidar_rear_pts': pts_r.shape[0],
            })

        return timestamps_rows

    def _write_timestamps_csv(self, rows):
        """Write timestamps.csv index file."""
        csv_path = os.path.join(self.output_dir, 'timestamps.csv')
        fieldnames = [
            'frame_id', 'frame_name', 'timestamp',
            'cam_front_ts', 'cam_rear_ts', 'lidar_front_ts', 'lidar_rear_ts',
            'cam_front_path', 'cam_rear_path',
            'lidar_front_path', 'lidar_rear_path',
            'num_lidar_front_pts', 'num_lidar_rear_pts',
        ]
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f'[INFO] timestamps.csv written with {len(rows)} rows.')

    # ------------------------------------------------------------------
    # Quality report
    # ------------------------------------------------------------------
    def _generate_quality_report(self, bag, groups):
        """Generate data quality report JSON."""
        # Duration
        duration = bag.get_end_time() - bag.get_start_time()

        # Timestamp jitter (per-topic std of inter-frame dt)
        def _compute_jitter(ts_list):
            if len(ts_list) < 2:
                return 0.0
            dts = np.diff(sorted(ts_list))
            return float(np.std(dts))

        timestamps = defaultdict(list)
        for g in groups:
            timestamps['cam_front'].append(g['cam_front'][0])
            timestamps['cam_rear'].append(g['cam_rear'][0])
            timestamps['lidar_front'].append(g['lidar_front'][0])
            timestamps['lidar_rear'].append(g['lidar_rear'][0])

        # Monotonicity check
        def _check_monotonic(ts_list):
            for i in range(1, len(ts_list)):
                if ts_list[i] < ts_list[i - 1]:
                    return False
            return True

        total_sync = self._stats['synced_groups']
        sync_loss = (
            (self._stats['total_lidar_front'] - total_sync)
            / max(self._stats['total_lidar_front'], 1)
        )

        report = {
            'bag_path': self.bag_path,
            'duration_sec': duration,
            'total_synced_frames': total_sync,
            'sync_loss_rate': round(sync_loss, 4),
            'sync_window_sec': self.sync_window_sec,
            'message_counts': {
                'cam_front': self._stats['total_cam_front'],
                'cam_rear': self._stats['total_cam_rear'],
                'lidar_front': self._stats['total_lidar_front'],
                'lidar_rear': self._stats['total_lidar_rear'],
            },
            'missed_per_sync': {
                'cam_front': self._stats['missed_cam_front'],
                'cam_rear': self._stats['missed_cam_rear'],
                'lidar_front': self._stats['missed_lidar_front'],
                'lidar_rear': self._stats['missed_lidar_rear'],
            },
            'fps_estimate': {
                # synced_frames / duration
                'synced_fps': round(total_sync / max(duration, 0.001), 2),
                'cam_front': round(
                    self._stats['total_cam_front'] / max(duration, 0.001), 2
                ),
                'cam_rear': round(
                    self._stats['total_cam_rear'] / max(duration, 0.001), 2
                ),
                'lidar_front': round(
                    self._stats['total_lidar_front'] / max(duration, 0.001), 2
                ),
                'lidar_rear': round(
                    self._stats['total_lidar_rear'] / max(duration, 0.001), 2
                ),
            },
            'timestamp_jitter_std': {
                k: round(_compute_jitter(v), 6) for k, v in timestamps.items()
            },
            'timestamp_monotonic': {
                k: _check_monotonic(v) for k, v in timestamps.items()
            },
        }
        report_path = os.path.join(self.output_dir, 'quality_report.json')
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f'[INFO] Quality report: {report_path}')
        return report


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Extract images and point clouds from ROS1 rosbag.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python rosbag_extract.py scene_001.bag -o data/extracted/scene_001/
  python rosbag_extract.py scene_001.bag -o out/ --extract-tf --no-compress
  python rosbag_extract.py scene_001.bag -o out/ --sync-window 0.03
        """,
    )
    parser.add_argument('bag_path', help='Path to input ROS1 rosbag file.')
    parser.add_argument(
        '-o', '--output-dir', required=True,
        help='Output directory for extracted data.',
    )
    parser.add_argument(
        '--front-camera-topic', default='/camera/front/image_raw',
        help='ROS topic for front camera (default: /camera/front/image_raw).',
    )
    parser.add_argument(
        '--rear-camera-topic', default='/camera/rear/image_raw',
        help='ROS topic for rear camera (default: /camera/rear/image_raw).',
    )
    parser.add_argument(
        '--front-lidar-topic', default='/lidar/os1_front/points',
        help='ROS topic for front lidar (default: /lidar/os1_front/points).',
    )
    parser.add_argument(
        '--rear-lidar-topic', default='/lidar/os1_rear/points',
        help='ROS topic for rear lidar (default: /lidar/os1_rear/points).',
    )
    parser.add_argument(
        '--sync-window', type=float, default=0.050,
        help='Synchronization window in seconds (default: 0.050 = 50 ms).',
    )
    parser.add_argument(
        '--extract-tf', action='store_true',
        help='Extract /tf and /tf_static transforms from the rosbag.',
    )
    parser.add_argument(
        '--no-compress', action='store_true',
        help='Save images as PNG instead of JPEG.',
    )

    args = parser.parse_args()

    # Validate inputs
    if not os.path.isfile(args.bag_path):
        print(f'[ERROR] Bag file not found: {args.bag_path}')
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    extractor = RosbagExtractor(
        bag_path=args.bag_path,
        output_dir=args.output_dir,
        sync_window_sec=args.sync_window,
    )

    try:
        extractor.extract(
            front_camera_topic=args.front_camera_topic,
            rear_camera_topic=args.rear_camera_topic,
            front_lidar_topic=args.front_lidar_topic,
            rear_lidar_topic=args.rear_lidar_topic,
            extract_tf=args.extract_tf,
            compress_images=not args.no_compress,
        )
    except RuntimeError as e:
        print(f'[ERROR] {e}')
        print(
            '[HINT] Make sure you are in a ROS1 environment '
            '(source /opt/ros/<distro>/setup.bash).'
        )
        sys.exit(1)


if __name__ == '__main__':
    main()
