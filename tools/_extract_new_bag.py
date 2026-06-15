"""Extract new RTG bag (4 cameras + 4 radars, 集卡侧 + 禁行侧)"""
import sys, os, csv, json
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import rosbag, cv2, numpy as np
from sensor_msgs import point_cloud2

BAG = '/media/shen/data/zk/Auto/CenterPoint/data/rtg/raw/lidar_and_camera_2026-05-19-15-44-34.bag'
OUT = '/media/shen/data/zk/Auto/CenterPoint/data/rtg/extracted/scene_001'

for s in ['cam_01','cam_02','cam_03','cam_04','lidar_01','lidar_02','lidar_03','lidar_04']:
    os.makedirs(f'{OUT}/{s}', exist_ok=True)

bag = rosbag.Bag(BAG)

# Collect messages
cam = {f'cam_0{i}': [] for i in range(1,5)}
lidar = {f'lidar_0{i}': [] for i in range(1,5)}

for topic, msg, t in bag.read_messages():
    ts = msg.header.stamp.to_sec()
    if 'Camera_Raw_Img_01' in topic:
        cam['cam_01'].append((msg, ts))
    elif 'Camera_Raw_Img_02' in topic:
        cam['cam_02'].append((msg, ts))
    elif 'Camera_Raw_Img_03' in topic:
        cam['cam_03'].append((msg, ts))
    elif 'Camera_Raw_Img_04' in topic:
        cam['cam_04'].append((msg, ts))
    elif 'ouster1' in topic:
        # Ouster uses internal clock, correct to Unix epoch
        # Check if timestamp is in GPS/internal epoch (small) vs Unix (large)
        if ts < 1e10:  # Internal clock
            pass  # Will be corrected later
        lidar['lidar_01'].append((msg, ts))
    elif 'ouster2' in topic:
        if ts < 1e10:
            pass
        lidar['lidar_02'].append((msg, ts))
    elif 'lidar3' in topic:
        lidar['lidar_03'].append((msg, ts))
    elif 'lidar4' in topic:
        lidar['lidar_04'].append((msg, ts))

# Correct ouster timestamps to Unix epoch
# Compute offset from first camera message to first ouster message
if lidar['lidar_01'] and cam['cam_01']:
    cam_start = cam['cam_01'][0][1]
    lidar_start = lidar['lidar_01'][0][1]
    if lidar_start < 1e10:  # Internal clock → needs correction
        offset = cam_start - lidar_start
        lidar['lidar_01'] = [(m, ts + offset) for m, ts in lidar['lidar_01']]
        print(f"Ouster1 offset corrected: +{offset:.0f}s")
    if lidar['lidar_02'] and lidar['lidar_02'][0][1] < 1e10:
        offset2 = cam_start - lidar['lidar_02'][0][1]
        lidar['lidar_02'] = [(m, ts + offset2) for m, ts in lidar['lidar_02']]
        print(f"Ouster2 offset corrected: +{offset2:.0f}s")

bag.close()

print("Camera messages:")
for k, v in cam.items(): print(f"  {k}: {len(v)}")
print("LiDAR messages:")
for k, v in lidar.items(): print(f"  {k}: {len(v)}")

# Sync: use lidar_01 (Ouster front, 10Hz) as reference
ref = lidar['lidar_01']
sync_window = 0.10  # 100ms (10Hz sensors, half-period tolerance)
records = []

for lidar_msg, lidar_ts in ref:
    frame = {'lidar_01': (lidar_msg, lidar_ts)}

    # Find closest messages for each other sensor
    for sensor_name, msgs in {**cam, **{k:v for k,v in lidar.items() if k != 'lidar_01'}}.items():
        best, best_dt = None, float('inf')
        for msg, ts in msgs:
            dt = abs(lidar_ts - ts)
            if dt < best_dt:
                best_dt = dt; best = (msg, ts)
        if best_dt <= sync_window:
            frame[sensor_name] = best
            frame[f'{sensor_name}_dt'] = best_dt

    records.append(frame)

# Compute availability stats
avail = {}
for s in {**cam, **lidar}:
    avail[s] = sum(1 for r in records if s in r)
print(f"\nSynchronized frames: {len(records)}")
print("Frames with each sensor:")
for s, n in avail.items():
    print(f"  {s}: {n} / {len(records)} ({100*n/max(len(records),1):.0f}%)")

# Extract
timestamps = []
for i, frame in enumerate(records):
    ts_row = {'frame_idx': i}

    # Cameras (all 4: C1/C2 集卡侧 + C3/C4 禁行侧)
    for cam_key in ['cam_01', 'cam_02', 'cam_03', 'cam_04']:
        if cam_key in frame:
            msg, ts = frame[cam_key]
            raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
            cv2.imwrite(f'{OUT}/{cam_key}/{i:06d}.jpg', raw)
            ts_row[f'{cam_key}_ts'] = ts

    # LiDARs (all 4: L1/L2 集卡侧 OS1 + L3/L4 禁行侧 Helios32)
    for lidar_key in ['lidar_01', 'lidar_02', 'lidar_03', 'lidar_04']:
        if lidar_key in frame:
            msg, ts = frame[lidar_key]
            fields = [f.name for f in msg.fields]
            if 'ring' in fields:
                # Ouster OS1: keep x,y,z,intensity,ring
                pts = np.array(list(point_cloud2.read_points(
                    msg, field_names=['x','y','z','reflectivity','ring'],
                    skip_nans=True)), dtype=np.float32)
            else:
                # Helios32: keep x,y,z,intensity
                pts = np.array(list(point_cloud2.read_points(
                    msg, field_names=['x','y','z','intensity'],
                    skip_nans=True)), dtype=np.float32)
            pts.tofile(f'{OUT}/{lidar_key}/{i:06d}.bin')
            ts_row[f'{lidar_key}_ts'] = ts
            ts_row[f'{lidar_key}_pts'] = len(pts)

    timestamps.append(ts_row)

    if i < 5:
        parts = [f"Frame {i}:"]
        for ck in ['cam_01','cam_02','cam_03','cam_04']:
            if f'{ck}_ts' in ts_row:
                parts.append(f"{ck}={ts_row[f'{ck}_ts']:.3f}")
        for lk in ['lidar_01','lidar_02','lidar_03','lidar_04']:
            if f'{lk}_pts' in ts_row:
                parts.append(f"{lk}={ts_row[f'{lk}_pts']}pts")
        print('  '.join(parts))

# Save timestamps
with open(f'{OUT}/timestamps.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=timestamps[0].keys())
    w.writeheader()
    w.writerows(timestamps)

# Quality report
quality = {
    'bag_file': os.path.basename(BAG),
    'duration_seconds': 51.7,
    'synchronized_frames': len(records),
    'cameras': {'cam_01': len(cam['cam_01']), 'cam_02': len(cam['cam_02']),
                'cam_03': len(cam['cam_03']), 'cam_04': len(cam['cam_04'])},
    'lidars': {'lidar_01': len(lidar['lidar_01']), 'lidar_02': len(lidar['lidar_02']),
               'lidar_03': len(lidar['lidar_03']), 'lidar_04': len(lidar['lidar_04'])},
    'camera_resolution': '1080x1920',
    'lidar_01_type': 'Ouster OS1 128-line (集卡侧前)',
    'lidar_02_type': 'Ouster OS1 128-line (集卡侧后)',
    'lidar_03_type': 'RoboSense Helios 32 (禁行侧前)',
    'lidar_04_type': 'RoboSense Helios 32 (禁行侧后)',
    'frames_with_all_4_cameras':
        sum(1 for r in records
            if all(f'{s}_ts' in r for s in ['cam_01','cam_02','cam_03','cam_04'])),
    'frames_with_truck_lane':
        sum(1 for r in records
            if all(s in r for s in ['cam_01','cam_02','lidar_01','lidar_02'])),
    'frames_with_forbidden':
        sum(1 for r in records
            if all(s in r for s in ['cam_03','cam_04','lidar_03','lidar_04'])),
    'sensor_availability': {s: f'{avail[s]}/{len(records)}' for s in sorted(avail)},
}

with open(f'{OUT}/quality_report.json', 'w') as f:
    json.dump(quality, f, indent=2)

print(f"\nExtraction complete: {len(records)} frames")
print(f"Output: {OUT}")
for k, v in quality.items():
    print(f"  {k}: {v}")
