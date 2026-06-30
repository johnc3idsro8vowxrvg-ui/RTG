#!/usr/bin/env python3
"""Generate info.pkl from extracted RTG data with synthetic calibration.
Used for pipeline verification only - NOT for accuracy evaluation.
"""
import sys, os, pickle, csv, json, yaml, argparse, numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', required=True, help='extracted scene directory')
    parser.add_argument('--calib', required=True, help='calibration YAML file')
    parser.add_argument('--output', required=True, help='output pkl path')
    parser.add_argument('--label', default='train', help='dataset label')
    args = parser.parse_args()

    # Load calibration
    with open(args.calib) as f:
        calib = yaml.safe_load(f)

    # Load timestamps
    ts_path = os.path.join(args.data_dir, 'timestamps.csv')
    if os.path.exists(ts_path):
        with open(ts_path) as f:
            timestamps = list(csv.DictReader(f))
    else:
        # LiDAR-only: generate from bin files
        lidar_dir = os.path.join(args.data_dir, 'lidar_front')
        files = sorted(os.listdir(lidar_dir)) if os.path.exists(lidar_dir) else []
        timestamps = [{'frame_idx': i, 'cam_timestamp': float(i)/10.0,
                       'lidar_timestamp': float(i)/10.0, 'sync_diff_ms': 0,
                       'num_points': 0} for i in range(len(files))]

    has_camera = os.path.exists(os.path.join(args.data_dir, 'front', '000000.jpg'))

    # Camera intrinsic
    cam_intrinsic = calib['sensors']['CAM_FR']['intrinsic']
    K = np.array([[cam_intrinsic['fx'], 0, cam_intrinsic['cx']],
                  [0, cam_intrinsic['fy'], cam_intrinsic['cy']],
                  [0, 0, 1]], dtype=np.float32)

    # Extrinsics
    R_cl = np.array(calib['extrinsics']['CAM_FR_to_LIDAR_FR']['R'], dtype=np.float32)
    T_cl = np.array(calib['extrinsics']['CAM_FR_to_LIDAR_FR']['T'], dtype=np.float32)
    lidar2cam = np.eye(4, dtype=np.float32)
    lidar2cam[:3, :3] = R_cl.T
    lidar2cam[:3, 3] = -R_cl.T @ T_cl

    # Camera intrinsic 4x4
    cam_intrinsic_4x4 = np.eye(4, dtype=np.float32)
    cam_intrinsic_4x4[:3, :3] = K

    infos = []
    scene_name = os.path.basename(args.data_dir)

    for ts in timestamps:
        idx = int(ts['frame_idx'])
        info = {
            'token': f'{scene_name}_{idx:06d}',
            'timestamp': float(ts.get('cam_timestamp', ts.get('lidar_timestamp', 0))),
            'scene_name': scene_name,
            'frame_idx': idx,
            'lidar_path': f'lidar_front/{idx:06d}.bin',
            'num_lidar_pts': int(ts.get('num_points', 0)),
            'cams': {},
            'lidar2world': np.eye(4, dtype=np.float32).tolist(),
            'sweeps': [],
            'location': '',
            'scene_token': scene_name,
        }

        if has_camera:
            # Build lidar2img = cam_intrinsic @ lidar2cam
            lidar2img = cam_intrinsic_4x4 @ lidar2cam

            info['cams']['CAM_FRONT'] = {
                'data_path': f'front/{idx:06d}.jpg',
                'sensor2lidar_rotation': R_cl.T.tolist(),
                'sensor2lidar_translation': (-R_cl.T @ T_cl).tolist(),
                'cam_intrinsic': K.tolist(),
                'lidar2cam': lidar2cam.tolist(),
                'lidar2img': lidar2img.tolist(),
            }

        # GT boxes (empty - no annotations yet)
        info['gt_boxes'] = np.zeros((0, 9), dtype=np.float32)
        info['gt_names'] = np.array([], dtype=str)
        info['gt_velocity'] = np.zeros((0, 2), dtype=np.float32)
        info['num_lidar_pts'] = int(ts.get('num_points', 0))
        info['num_radar_pts'] = 0
        info['valid_flag'] = True

        infos.append(info)

    metadata = {
        'version': 'rtg_v1.0_synthetic',
        'num_cameras': 1,
        'camera_names': ['CAM_FRONT'] if has_camera else [],
        'categories': {
            'person': 0, 'truck': 1, 'car': 2, 'other_obstacle': 3
        },
        'num_samples': len(infos),
        'has_camera': has_camera,
        'calib_note': 'SYNTHETIC - approximate, for pipeline verification only',
    }

    output = {'infos': infos, 'metadata': metadata}

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'wb') as f:
        pickle.dump(output, f)

    print(f'Generated {len(infos)} infos → {args.output}')
    print(f'  Has camera: {has_camera}')
    print(f'  Cameras: {metadata["camera_names"]}')
    if has_camera:
        print(f'  Intrinsic: fx={cam_intrinsic["fx"]:.1f}, fy={cam_intrinsic["fy"]:.1f}')
    print(f'  WARNING: Synthetic calibration - for pipeline verification only!')

if __name__ == '__main__':
    main()
