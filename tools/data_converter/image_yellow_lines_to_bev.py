#!/usr/bin/env python3
"""
image_yellow_lines_to_bev.py — 图像黄线像素标注 → BEV 车道边界投影

用途:
  在相机图像中一次性标注黄线/车道边界的像素位置（静态配置），
  通过相机内参 + 外参投影到 z=0 地面平面，生成 BEV 车道边界坐标。

用法:
  python image_yellow_lines_to_bev.py \
      --geometry config/geometry.yaml \
      --calib path/to/calib.yaml \
      --output config/geometry.yaml  # 原地更新 bev_boundaries 字段

输入:
  geometry.yaml 中的 lanes.camera_XX.yellow_lines[].pixel_points

输出:
  geometry.yaml 中的 lanes.bev_boundaries (自动填充)
"""

import argparse
import math
import os
import sys

import numpy as np
import yaml


def load_yaml(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    with open(path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)


def pixel_to_camera_ray(u, v, K):
    """将像素坐标 (u,v) 转换为相机坐标系下的归一化方向向量。

    Args:
        u, v: 像素坐标
        K: 内参矩阵 [fx, fy, cx, cy]

    Returns:
        direction: [dx, dy, 1] 归一化方向向量 (相机坐标系)
    """
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    dx = (u - cx) / fx
    dy = (v - cy) / fy
    return np.array([dx, dy, 1.0])


def project_ray_to_ground(direction, R_cl, T_cl, ground_z=0.0):
    """将相机坐标系下的射线投影到 z=ground_z 的地面平面。

    Args:
        direction: 相机坐标系下的方向向量 [dx, dy, dz]
        R_cl: 相机到 LiDAR 的旋转矩阵 (3x3)
        T_cl: 相机到 LiDAR 的平移向量 (3,)

    Returns:
        point_2d: 地面平面上的 (x, y) 坐标 (LiDAR/BEV 坐标系)
    """
    # 射线在 LiDAR 坐标系下的方向
    ray_lidar = R_cl @ direction

    # 相机光心在 LiDAR 坐标系下的位置
    camera_center_lidar = T_cl

    # 射线与 z=ground_z 平面的交点
    # camera_center_lidar[2] + t * ray_lidar[2] = ground_z
    # t = (ground_z - camera_center_lidar[2]) / ray_lidar[2]
    if abs(ray_lidar[2]) < 1e-10:
        return None  # 射线平行于地面，无交点

    t = (ground_z - camera_center_lidar[2]) / ray_lidar[2]
    if t <= 0:
        return None  # 交点在相机后方

    x = camera_center_lidar[0] + t * ray_lidar[0]
    y = camera_center_lidar[1] + t * ray_lidar[1]
    return np.array([x, y])


def project_polyline(pixel_points, K, R_cl, T_cl, ground_z=0.0):
    """将像素多段线投影到 BEV 地面平面。

    Args:
        pixel_points: [[u1,v1], [u2,v2], ...]
        K: 内参
        R_cl: 相机→LiDAR 旋转矩阵 (3x3)
        T_cl: 相机→LiDAR 平移向量 (3,)

    Returns:
        bev_points: [[x1,y1], [x2,y2], ...] 有效的地面投影点
        invalid_count: 无效投影点数
    """
    bev_points = []
    invalid_count = 0

    for u, v in pixel_points:
        direction = pixel_to_camera_ray(u, v, K)
        point = project_ray_to_ground(direction, R_cl, T_cl, ground_z)
        if point is not None:
            bev_points.append(point.tolist())
        else:
            invalid_count += 1

    return bev_points, invalid_count


def compute_lane_boundary_y_range(bev_points, axis='y'):
    """从 BEV 投影点计算车道边界的 y 轴范围。

    对于大车道（沿 x 轴延伸），车道边界主要由 y 坐标范围定义。

    Args:
        bev_points: [[x,y], ...]
        axis: 要计算范围的轴 ('y' 或 'x')

    Returns:
        (min_val, max_val): 范围
    """
    if not bev_points:
        return 0.0, 0.0

    idx = 1 if axis == 'y' else 0
    values = [p[idx] for p in bev_points]
    return min(values), max(values)


def compute_extrinsics_from_yaml(calib_data, camera_key):
    """从 calib.yaml 提取指定相机的内参和外参。

    Args:
        calib_data: calib.yaml 内容
        camera_key: 相机标识，如 'CAM_FR'

    Returns:
        K: {'fx','fy','cx','cy'}
        R_cl: 3x3 numpy array (相机→LiDAR)
        T_cl: (3,) numpy array (相机→LiDAR)
    """
    # 查找相机→LiDAR 外参
    ext_key = f'{camera_key}_to_LIDAR_FR'

    if 'extrinsics' in calib_data and ext_key in calib_data['extrinsics']:
        ext = calib_data['extrinsics'][ext_key]
    elif ext_key in calib_data:
        ext = calib_data[ext_key]
    else:
        raise KeyError(f'Extrinsics not found for {ext_key} in calib file')

    R = np.array(ext['R'])
    T = np.array(ext['T'])

    # 内参
    if 'sensors' in calib_data and camera_key in calib_data['sensors']:
        sensor = calib_data['sensors'][camera_key]
        K = sensor['intrinsic']
    elif camera_key in calib_data and 'intrinsic' in calib_data[camera_key]:
        K = calib_data[camera_key]['intrinsic']
    else:
        raise KeyError(f'Intrinsics not found for {camera_key} in calib file')

    return K, R, T


def main():
    parser = argparse.ArgumentParser(
        description='图像黄线像素标注 → BEV 车道边界投影',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # 从 geometry.yaml 读取像素标注 + calib.yaml 读取内外参 → 更新 geometry.yaml
  python image_yellow_lines_to_bev.py \\
      --geometry config/geometry.yaml \\
      --calib config/calib.yaml \\
      --output config/geometry.yaml

  # 仅打印 BEV 边界，不写文件
  python image_yellow_lines_to_bev.py \\
      --geometry config/geometry.yaml \\
      --calib config/calib.yaml \\
      --dry-run
        ''')

    parser.add_argument('--geometry', required=True,
                        help='geometry.yaml 文件路径 (含 lanes.*.yellow_lines 像素标注)')
    parser.add_argument('--calib', required=True,
                        help='calib.yaml 文件路径 (含内参和外参)')
    parser.add_argument('--output', default=None,
                        help='输出 geometry.yaml 路径 (默认: 覆盖 --geometry)')
    parser.add_argument('--dry-run', action='store_true',
                        help='仅打印结果，不写文件')
    parser.add_argument('--ground-z', type=float, default=0.0,
                        help='地面平面 z 坐标 (默认: 0.0)')

    args = parser.parse_args()

    # 加载文件
    if not os.path.exists(args.geometry):
        print(f'[ERROR] geometry file not found: {args.geometry}')
        sys.exit(1)
    if not os.path.exists(args.calib):
        print(f'[ERROR] calib file not found: {args.calib}')
        sys.exit(1)

    geometry = load_yaml(args.geometry)
    calib = load_yaml(args.calib)

    # 相机到 geometry.yaml lane 标注的映射
    camera_mapping = {
        'camera_01': 'CAM_FR',
        'camera_02': 'CAM_RE',
    }

    bev_boundaries = {}
    total_valid, total_invalid = 0, 0

    for geo_key, calib_key in camera_mapping.items():
        if 'lanes' not in geometry or geo_key not in geometry['lanes']:
            print(f'[SKIP] {geo_key}: no yellow_lines in geometry.lanes')
            continue

        try:
            K, R_cl, T_cl = compute_extrinsics_from_yaml(calib, calib_key)
        except KeyError as e:
            print(f'[SKIP] {geo_key}: {e}')
            continue

        print(f'\n{"="*60}')
        print(f'Processing {geo_key} ({calib_key})')
        print(f'  Intrinsic: fx={K["fx"]:.1f}, fy={K["fy"]:.1f}, '
              f'cx={K["cx"]:.1f}, cy={K["cy"]:.1f}')
        print(f'  Camera→LiDAR T: [{T_cl[0]:.3f}, {T_cl[1]:.3f}, {T_cl[2]:.3f}]')

        yellow_lines = geometry['lanes'][geo_key].get('yellow_lines', [])
        if not yellow_lines:
            print(f'  No yellow_lines defined')
            continue

        for line_info in yellow_lines:
            name = line_info.get('name', 'unnamed')
            pixel_points = line_info.get('pixel_points', [])

            if not pixel_points:
                print(f'  [SKIP] {name}: empty pixel_points (待现场标注)')
                continue

            bev_points, invalid = project_polyline(
                pixel_points, K, R_cl, T_cl, args.ground_z)

            total_valid += len(bev_points)
            total_invalid += invalid

            y_min, y_max = compute_lane_boundary_y_range(bev_points)
            print(f'  {name}:')
            print(f'    pixel_points: {len(pixel_points)} → BEV points: {len(bev_points)} '
                  f'(invalid: {invalid})')
            print(f'    y_range: [{y_min:.2f}, {y_max:.2f}] m')

            # 将结果按名称归类
            if name not in bev_boundaries:
                bev_boundaries[name] = {
                    'y_min': float('inf'),
                    'y_max': float('-inf'),
                    'x_min': float('inf'),
                    'x_max': float('-inf'),
                }

            if bev_points:
                ys = [p[1] for p in bev_points]
                xs = [p[0] for p in bev_points]
                bev_boundaries[name]['y_min'] = min(bev_boundaries[name]['y_min'], min(ys))
                bev_boundaries[name]['y_max'] = max(bev_boundaries[name]['y_max'], max(ys))
                bev_boundaries[name]['x_min'] = min(bev_boundaries[name]['x_min'], min(xs))
                bev_boundaries[name]['x_max'] = max(bev_boundaries[name]['x_max'], max(xs))

    # 汇总
    print(f'\n{"="*60}')
    print(f'Summary: {total_valid} valid BEV points, {total_invalid} invalid')

    # 更新 geometry.yaml 的 bev_boundaries
    if 'lanes' not in geometry:
        geometry['lanes'] = {}

    # 将结果按语义区域合并
    merged = {}

    # 主车道左边界 → main_lane_left
    # 主车道右边界 + 集卡车道右边界 → main_lane_truck_side
    for name, bounds in bev_boundaries.items():
        if bounds['y_min'] == float('inf'):
            continue
        merged[name] = {
            'y_min': round(bounds['y_min'], 2),
            'y_max': round(bounds['y_max'], 2),
        }

    geometry['lanes']['bev_boundaries'] = merged

    # 输出
    output_path = args.output or args.geometry

    if args.dry_run:
        print('\n[Dry-run] BEV boundaries that would be written:')
        for name, b in merged.items():
            print(f'  {name}: y=[{b["y_min"]}, {b["y_max"]}] m')
        print(f'\n[Dry-run] Would write to: {output_path}')
    else:
        save_yaml(geometry, output_path)
        print(f'\nUpdated BEV boundaries written to: {output_path}')


if __name__ == '__main__':
    main()
