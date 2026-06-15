#!/usr/bin/env python3
"""Quick rosbag extraction helper — used directly with system python"""
import sys, os
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import rosbag, cv2, numpy as np, csv, json
from sensor_msgs import point_cloud2

bag_path = sys.argv[1]
out_dir = sys.argv[2]
scene_name = os.path.basename(bag_path).replace('.bag', '')

os.makedirs(f'{out_dir}/{scene_name}/front', exist_ok=True)
os.makedirs(f'{out_dir}/{scene_name}/lidar_front', exist_ok=True)

bag = rosbag.Bag(bag_path)
cam_msgs, lidar_msgs = [], []

for topic, msg, t in bag.read_messages():
    if 'Camera_Raw_Img' in topic:
        cam_msgs.append((msg, msg.header.stamp.to_sec()))
    elif 'rslidar' in topic or 'points' in topic:
        lidar_msgs.append((msg, msg.header.stamp.to_sec()))
bag.close()

print(f'{scene_name}: Camera={len(cam_msgs)}, LiDAR={len(lidar_msgs)}')

# Soft sync
records = []
sync_window = 0.1
for cam_msg, cam_ts in cam_msgs:
    best, best_dt = None, float('inf')
    for lm, lt in lidar_msgs:
        dt = abs(cam_ts - lt)
        if dt < best_dt:
            best_dt = dt; best = (lm, lt)
    if best_dt <= sync_window:
        records.append((cam_msg, cam_ts, best[0], best[1], best_dt))
        lidar_msgs = [(m,t) for m,t in lidar_msgs if t != best[1]]

print(f'  Synced: {len(records)} pairs (unmatched lidar: {len(lidar_msgs)})')
if not records:
    # LiDAR-only bag
    for i, (lidar_msg, lidar_ts) in enumerate(lidar_msgs + [(m,t) for m,t in [(None,0)]]):
        if lidar_msg is None: break
        pts = np.array(list(point_cloud2.read_points(
            lidar_msg, field_names=['x','y','z','intensity'], skip_nans=True)), dtype=np.float32)
        pts.tofile(f'{out_dir}/{scene_name}/lidar_front/{i:06d}.bin')
    print(f'  LiDAR-only: {len(lidar_msgs)} frames extracted')
    # Save quality report
    quality = {
        'type': 'lidar_only',
        'bag_file': os.path.basename(bag_path),
        'lidar_frames': len(lidar_msgs),
    }
else:
    timestamps = []
    for i, (cam_msg, cam_ts, lidar_msg, lidar_ts, sync_dt) in enumerate(records):
        raw = np.frombuffer(cam_msg.data, dtype=np.uint8).reshape(cam_msg.height, cam_msg.width, 3)
        cv2.imwrite(f'{out_dir}/{scene_name}/front/{i:06d}.jpg', raw)
        pts = np.array(list(point_cloud2.read_points(
            lidar_msg, field_names=['x','y','z','intensity'], skip_nans=True)), dtype=np.float32)
        pts.tofile(f'{out_dir}/{scene_name}/lidar_front/{i:06d}.bin')
        timestamps.append({
            'frame_idx': i, 'cam_timestamp': cam_ts, 'lidar_timestamp': lidar_ts,
            'sync_diff_ms': round(sync_dt*1000, 1), 'num_points': len(pts)
        })
    with open(f'{out_dir}/{scene_name}/timestamps.csv', 'w') as f:
        w = csv.DictWriter(f, fieldnames=timestamps[0].keys()); w.writeheader(); w.writerows(timestamps)
    quality = {
        'type': 'camera_lidar',
        'bag_file': os.path.basename(bag_path),
        'synchronized_frames': len(records),
        'camera_fps': 5.0, 'lidar_fps': 10.0,
        'camera_resolution': f'{cam_msgs[0][0].width}x{cam_msgs[0][0].height}',
        'camera_encoding': cam_msgs[0][0].encoding,
        'avg_points_per_frame': sum(r['num_points'] for r in timestamps) / max(len(timestamps), 1),
        'sync_loss_rate': (len(cam_msgs)-len(records))/max(len(cam_msgs),1),
    }
    print(f'  Frames extracted: {len(records)}')

with open(f'{out_dir}/{scene_name}/quality_report.json', 'w') as f:
    json.dump(quality, f, indent=2)

print(f'  Avg pts/frame: {quality.get("avg_points_per_frame", "N/A")}')
print(f'  Report: {out_dir}/{scene_name}/quality_report.json')
