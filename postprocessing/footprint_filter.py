"""
自车 Footprint 点云过滤模块 — footprint_filter.py

从 geometry.yaml 读取 RTG 四条支腿坐标，构建两个独立矩形 footprint
（集卡侧 + 禁行侧，前后腿通过连接梁连成一体），在推理前剔除
落在自车结构上的 LiDAR 点云，防止模型将自车结构误检为障碍物。

使用方式:
    from postprocessing.footprint_filter import SelfFootprintFilter
    f = SelfFootprintFilter('config/geometry.yaml')
    mask = f.get_mask(points_bev_xy)         # True = 保留
    filtered_pts = points[mask]
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml

logger = logging.getLogger(__name__)


class FootprintZone:
    """单个矩形 footprint 区域（BEV 坐标系）。"""

    def __init__(self, name: str, x_min: float, x_max: float,
                 y_min: float, y_max: float):
        self.name = name
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """返回 bool 数组，True 表示点在区域内。"""
        return (x >= self.x_min) & (x <= self.x_max) & \
               (y >= self.y_min) & (y <= self.y_max)

    def __repr__(self):
        return (f'FootprintZone({self.name}: '
                f'x=[{self.x_min:.1f},{self.x_max:.1f}], '
                f'y=[{self.y_min:.1f},{self.y_max:.1f}])')


class SelfFootprintFilter:
    """自车 footprint 点云过滤器。

    根据 geometry.yaml 构建 footprint 矩形区域，提供点云过滤功能。

    RTG 不同于常规车辆——地面上仅有 4 条支腿，中间区域
    （集装箱、集卡）为正常检测空间。因此拆分为两个独立矩形:
      - 集卡侧: 连接集卡侧前后支腿 (y≈0)
      - 禁行侧: 连接禁行侧前后支腿 (y≈-23.5m)
    """

    def __init__(self, geometry_path: str = 'config/geometry.yaml'):
        """
        Args:
            geometry_path: geometry.yaml 文件路径
        """
        self._zones: List[FootprintZone] = []
        self._load(geometry_path)

    # ------------------------------------------------------------------
    # 加载
    # ------------------------------------------------------------------

    def _load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            geom = yaml.safe_load(f)

        # 优先新格式: ego_geometry
        ego = geom.get('ego_geometry', {})
        if ego:
            assemblies = [
                ('truck_side', ego.get('truck_side_assembly', {})),
                ('forbidden_side', ego.get('forbidden_side_assembly', {})),
            ]
            for name, assy in assemblies:
                if not assy:
                    continue
                x_min = float(assy.get('x_min', 0))
                x_max = float(assy.get('x_max', 0))
                y_min = float(assy.get('y_min', 0))
                y_max = float(assy.get('y_max', 0))
                if x_min == x_max or y_min == y_max:
                    continue

                zone = FootprintZone(
                    name=name,
                    x_min=x_min, x_max=x_max,
                    y_min=y_min, y_max=y_max,
                )
                self._zones.append(zone)
                logger.info(f'Footprint {name}: x=[{x_min},{x_max}], '
                            f'y=[{y_min},{y_max}]')

            if self._zones:
                return

        # 旧格式 fallback: ego_footprint = {truck_lane_side: {front_leg: {x,y}, ...}, ...}
        footprint = geom.get('ego_footprint', {})
        if not footprint:
            logger.warning('ego_footprint/ego_geometry 未在 geometry.yaml 中配置，'
                           '自车点云过滤将跳过')
            return

        for side_key, side in footprint.items():
            front = side.get('front_leg')
            rear = side.get('rear_leg')
            if not front or not rear:
                logger.warning(f'{side_key}: front_leg 或 rear_leg 缺失')
                continue

            fx, fy = float(front['x']), float(front['y'])
            rx, ry = float(rear['x']), float(rear['y'])
            fw = float(front.get('width', 1.0))
            fl = float(front.get('length', 1.0))
            rw = float(rear.get('width', 1.0))
            rl = float(rear.get('length', 1.0))

            x_min = min(rx - rl / 2, fx - fl / 2)
            x_max = max(rx + rl / 2, fx + fl / 2)
            y_min = min(fy - fw / 2, ry - rw / 2)
            y_max = max(fy + fw / 2, ry + rw / 2)

            zone = FootprintZone(
                name=side_key,
                x_min=round(x_min, 3),
                x_max=round(x_max, 3),
                y_min=round(y_min, 3),
                y_max=round(y_max, 3),
            )
            self._zones.append(zone)
            logger.info(f'Loaded {zone}')

    # ------------------------------------------------------------------
    # 过滤
    # ------------------------------------------------------------------

    def get_mask(self, points_xy: np.ndarray) -> np.ndarray:
        """返回 bool 掩码——True 表示保留（非自车），False 表示剔除。

        Args:
            points_xy: [N, 2] 点云在 BEV 坐标系下的 (x, y) 坐标

        Returns:
            keep: [N] bool 数组
        """
        if len(points_xy) == 0:
            return np.array([], dtype=bool)

        x = points_xy[:, 0]
        y = points_xy[:, 1]
        keep = np.ones(len(points_xy), dtype=bool)

        for zone in self._zones:
            in_zone = zone.contains(x, y)
            keep[in_zone] = False

        return keep

    def filter(self, points: np.ndarray,
               lidar_id: int = 1) -> np.ndarray:
        """过滤点云，返回非自车点。

        点云坐标自动从雷达坐标系转换到 BEV 坐标系后再过滤。

        Args:
            points: [N, D] 点云数组 (前 3 列为 x, y, z)
            lidar_id: 雷达编号 (1=L1集卡侧前, 2=L2集卡侧后)

        Returns:
            filtered: [M, D] 过滤后的点云
        """
        if len(points) == 0 or not self._zones:
            return points

        # 雷达坐标 → BEV 坐标 (仅 x,y 翻译，z 加雷达高度)
        pts_bev = points.copy()
        pts_bev[:, :2] = self._lidar_to_bev_xy(points[:, :2], lidar_id)

        mask = self.get_mask(pts_bev[:, :2])

        n_removed = (~mask).sum()
        if n_removed > 0:
            logger.debug(
                f'Footprint filter (lidar {lidar_id}): '
                f'removed {n_removed}/{len(points)} points '
                f'({100 * n_removed / len(points):.1f}%)')

        return points[mask]

    # ------------------------------------------------------------------
    # 坐标转换
    # ------------------------------------------------------------------

    @staticmethod
    def _lidar_to_bev_xy(points_xy: np.ndarray, lidar_id: int) -> np.ndarray:
        """雷达坐标系 → BEV 坐标系 (仅 x, y)。

        雷达 BEV 坐标 (geometry.yaml sensors.ego):
          L1: (0, 0, 1.5)        — 集卡侧前
          L2: (-12.0, 0, 1.5)    — 集卡侧后
          L3: (0, -23.5, 1.5)    — 禁行侧前
          L4: (-12.0, -23.5, 1.5) — 禁行侧后

        LiDAR 帧与 BEV 帧朝向一致（仅 z 平移），因此:
          BEV_x = lidar_x + sensor_bev_x
          BEV_y = lidar_y + sensor_bev_y

        BEV 位置来源: geometry.yaml — 权威值
        """
        offsets = {
            1: (0.0, 0.0),         # L1: BEV (0, 0)
            2: (-12.0, 0.0),       # L2: BEV (-12.0, 0)
            3: (0.0, -23.5),       # L3: BEV (0, -23.5)
            4: (-12.0, -23.5),     # L4: BEV (-12.0, -23.5)
        }
        dx, dy = offsets.get(lidar_id, (0.0, 0.0))
        return points_xy + np.array([[dx, dy]], dtype=points_xy.dtype)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def zones(self) -> List[FootprintZone]:
        return self._zones

    @property
    def has_footprint(self) -> bool:
        return len(self._zones) > 0

    def get_summary(self) -> str:
        """返回 footprint 配置摘要。"""
        if not self._zones:
            return 'No ego footprint configured'
        lines = [f'{len(self._zones)} footprint zones:']
        for z in self._zones:
            lines.append(f'  {z}')
        return '\n'.join(lines)


# ----------------------------------------------------------------------
# 便捷函数
# ----------------------------------------------------------------------

def create_filter(geometry_path: str = 'config/geometry.yaml') -> SelfFootprintFilter:
    """工厂函数。"""
    return SelfFootprintFilter(geometry_path)
