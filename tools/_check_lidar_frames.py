"""
Check LiDAR coordinate frames from rosbag data.

For each LiDAR, read one frame and determine:
  - Which axis is "up" (ground plane direction)
  - Orientation relative to the scene

Method for Ouster OS1 (L1/L2):
  - ring 0 = bottom-most beam, ring 127 = top-most beam
  - In ROS frame: +z = up, so ring 0 has lowest z, ring 127 has highest z
  - This lets us confirm which axis is the vertical in the output

Usage:
  python tools/_check_lidar_frames.py [bag_path]
"""
import sys
import os
import numpy as np

# Add rosbag path
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')

import rosbag
from sensor_msgs.point_cloud2 import read_points


def analyze_ouster_frame(points_np, name):
    """Analyze an Ouster OS1 frame with ring field.

    Ouster fields: x, y, z, t, ring, range, reflectivity, near_ir
    ring: 0 (bottom beam, most downward) to 127 (top beam, most upward)
    ROS frame: +x forward, +y left, +z up
    """
    print(f"\n{'='*60}")
    print(f"  {name} — Ouster OS1 128-line")
    print(f"{'='*60}")
    print(f"  Points: {points_np.shape[0]:,}")
    print(f"  Fields: x, y, z, t, ring, range, reflectivity, near_ir")

    x, y, z = points_np[:, 0], points_np[:, 1], points_np[:, 2]
    ring = points_np[:, 4].astype(np.int32)

    # Overall stats per axis
    for axis_name, arr in [('x', x), ('y', y), ('z', z)]:
        print(f"  {axis_name}: [{arr.min():.1f}, {arr.max():.1f}], mean={arr.mean():.1f}, std={arr.std():.1f}")

    # Check ring vs Z: ring 0 should have LOWEST z (ground), ring 127 HIGHEST z
    ring_min = ring.min()
    ring_max = ring.max()
    mean_z_per_ring = np.array([z[ring == r].mean() for r in range(ring_min, ring_max + 1)])

    # Correlation between ring number and mean Z
    rings = np.arange(ring_min, ring_max + 1)
    valid = ~np.isnan(mean_z_per_ring)
    if valid.sum() > 10:
        correlation = np.corrcoef(rings[valid], mean_z_per_ring[valid])[0, 1]
    else:
        correlation = 0

    print(f"\n  Ring range: {ring_min} ~ {ring_max}")
    print(f"  Mean Z at ring={ring_min} (bottom beam): {mean_z_per_ring[0]:.2f}m")
    print(f"  Mean Z at ring={ring_max} (top beam):    {mean_z_per_ring[-1]:.2f}m")
    print(f"  Correlation(ring, Z): {correlation:.4f}")

    if correlation > 0.8:
        print(f"  >>> +Z = UP (ring increases with Z) ✓ ROS standard")
    elif correlation < -0.8:
        print(f"  >>> -Z = UP (ring decreases with Z) — Z is flipped!")
    else:
        print(f"  >>> WARNING: ring-Z correlation is weak, check orientation")

    # Check ring vs X and ring vs Y to see if ring correlates with horizontal axes
    for axis_name, arr in [('x', x), ('y', y)]:
        mean_per_ring = np.array([arr[ring == r].mean() for r in range(ring_min, ring_max + 1)])
        corr = np.corrcoef(rings[valid], mean_per_ring[valid])[0, 1]
        print(f"  Correlation(ring, {axis_name}): {corr:.4f}")

    # Spatial extent: where are most points?
    print(f"\n  Point distribution:")
    # Count points in 4 quadrants (x>0/x<0  ×  y>0/y<0)
    xp_yp = ((x > 0) & (y > 0)).sum()
    xp_yn = ((x > 0) & (y < 0)).sum()
    xn_yp = ((x < 0) & (y > 0)).sum()
    xn_yn = ((x < 0) & (y < 0)).sum()
    total = len(x)
    print(f"  x>0, y>0: {xp_yp:>8,} ({100*xp_yp/total:.1f}%)")
    print(f"  x>0, y<0: {xp_yn:>8,} ({100*xp_yn/total:.1f}%)")
    print(f"  x<0, y>0: {xn_yp:>8,} ({100*xn_yp/total:.1f}%)")
    print(f"  x<0, y<0: {xn_yn:>8,} ({100*xn_yn/total:.1f}%)")

    # Ground height: bottom 10% of points by Z
    z_sorted = np.sort(z)
    ground_z = z_sorted[:len(z_sorted)//10].mean()
    print(f"\n  Ground Z (bottom 10%): {ground_z:.2f}m")
    print(f"  Sensor height ≈ {-ground_z:.1f}m above ground (if mounted ~1.5m, ground ≈ -1.5m)")

    return correlation


def analyze_rs32_frame(points_np, name):
    """Analyze a RoboSense 32-line frame (rotated 90°).

    RS32 fields: x, y, z, intensity (4 fields, no ring)
    Normal orientation: 360° horizontal, ~70° vertical
    After 90° rotation: ~70° horizontal (along lane), 360° vertical

    Without ring, use scene geometry: ground plane fitting.
    """
    print(f"\n{'='*60}")
    print(f"  {name} — RoboSense 32-line (rotated 90°)")
    print(f"{'='*60}")
    print(f"  Points: {points_np.shape[0]:,}")
    print(f"  Fields: x, y, z, intensity (no ring)")

    x, y, z = points_np[:, 0], points_np[:, 1], points_np[:, 2]

    for axis_name, arr in [('x', x), ('y', y), ('z', z)]:
        print(f"  {axis_name}: [{arr.min():.1f}, {arr.max():.1f}], mean={arr.mean():.1f}, std={arr.std():.1f}")

    # Simple ground plane: find axis with smallest variance at low height
    # For RS32 rotated 90°: the 360° plane is along one horizontal axis
    # Check which axis has the widest range → that's the "scanning plane"
    ranges = {'x': x.max() - x.min(), 'y': y.max() - y.min(), 'z': z.max() - z.min()}
    print(f"\n  Axis ranges: { {k: f'{v:.1f}m' for k, v in ranges.items()} }")

    largest_axis = max(ranges, key=ranges.get)
    smallest_axis = min(ranges, key=ranges.get)
    print(f"  Largest range: {largest_axis} ({ranges[largest_axis]:.1f}m)")
    print(f"  Smallest range: {smallest_axis} ({ranges[smallest_axis]:.1f}m)")

    # Quadrant distribution
    xp_yp = ((x > 0) & (y > 0)).sum()
    xp_yn = ((x > 0) & (y < 0)).sum()
    xn_yp = ((x < 0) & (y > 0)).sum()
    xn_yn = ((x < 0) & (y < 0)).sum()
    total = max(len(x), 1)
    print(f"\n  Point distribution:")
    print(f"  x>0, y>0: {xp_yp:>8,} ({100*xp_yp/total:.1f}%)")
    print(f"  x>0, y<0: {xp_yn:>8,} ({100*xp_yn/total:.1f}%)")
    print(f"  x<0, y>0: {xn_yp:>8,} ({100*xn_yp/total:.1f}%)")
    print(f"  x<0, y<0: {xn_yn:>8,} ({100*xn_yn/total:.1f}%)")

    # Ground height
    z_sorted = np.sort(z)
    ground_z = z_sorted[:len(z_sorted)//10].mean()
    print(f"\n  Ground Z (bottom 10%): {ground_z:.2f}m")


def main():
    bag_path = sys.argv[1] if len(sys.argv) > 1 else \
        '/media/shen/data/zk/Auto/CenterPoint/data/rtg/raw/lidar_and_camera_2026-05-19-15-44-34.bag'

    if not os.path.exists(bag_path):
        print(f"ERROR: bag not found: {bag_path}")
        sys.exit(1)

    print(f"Opening bag: {bag_path}")
    bag = rosbag.Bag(bag_path)

    # Topic → name mapping
    lidar_topics = {
        '/ouster1/points':          'L1 (集卡侧前, OS1 128线)',
        '/ouster2/points':          'L2 (集卡侧后, OS1 128线)',
        '/lidar3/rslidar3_points':  'L3 (禁行侧前, RS32 旋转90°)',
        '/lidar4/rslidar4_points':  'L4 (禁行侧后, RS32 旋转90°)',
    }

    for topic, name in lidar_topics.items():
        # Get first message
        for _, msg, t in bag.read_messages(topics=[topic]):
            points_list = list(read_points(msg, field_names=('x', 'y', 'z', 'intensity'),
                                            skip_nans=True))
            if not points_list:
                continue

            pts = np.array(points_list, dtype=np.float32)

            # Determine type by point count and topic
            if 'ouster' in topic:
                # Re-read with all fields for ring analysis
                all_points = list(read_points(msg, skip_nans=True))
                pts_full = np.array([list(p) for p in all_points], dtype=np.float32)
                analyze_ouster_frame(pts_full, name)
            else:
                analyze_rs32_frame(pts, name)

            break  # Only first frame
        else:
            print(f"\n  {name}: NO MESSAGES FOUND")

    bag.close()

    print(f"\n{'='*60}")
    print("SUMMARY: How to interpret the results")
    print(f"{'='*60}")
    print("""
For Ouster OS1 (L1/L2):
  - If ring-Z correlation is STRONG POSITIVE → +Z is UP (ROS standard) ✓
  - If ring-Z correlation is STRONG NEGATIVE → Z axis is flipped
  - The axis with ring correlation near 0 is the horizontal plane

  In ROS convention:
    +x = sensor "forward" (opposite cable)
    +y = sensor "left"
    +z = sensor "up"

For RoboSense 32 rotated 90° (L3/L4):
  - The axis with the LARGEST range (~360°) is the vertical scan direction
  - The axis with the SMALLEST range is the narrow FOV direction
""")

    print("\nNext step: compare results with BEV convention:")
    print("  BEV +x = 大车道方向, +y = 集卡侧→禁行侧, +z = 竖直向上")


if __name__ == '__main__':
    main()
