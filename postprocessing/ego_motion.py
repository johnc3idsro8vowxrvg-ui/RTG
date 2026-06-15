"""
运动估计模块 — ego_motion.py

基于静态背景点云的 ICP 运动估计：
  - 地面分割 (RANSAC) → 动态目标剔除 (基于检测框) → 静态结构提取 → 帧间 ICP
  - 输出: 静止/运动 + 方向(+x/-x) + 置信度
  - 基于 Open3D 实现
  - 不需要精确速度，只需要方向判断

独立模块，不依赖模型推理结果。
"""

import logging
import time
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# Open3D 可选依赖
try:
    import open3d as o3d
    HAS_OPEN3D = True
except ImportError:
    HAS_OPEN3D = False
    logger.warning(
        'Open3D not installed. EgoMotionEstimator will use numpy-only '
        'degraded mode (no ICP, threshold-based fallback).'
    )


# ==============================================================================
# 常量
# ==============================================================================
class EgoMotionState:
    STATIC = 0
    MOVING_PLUS_X = 1
    MOVING_MINUS_X = 2
    UNKNOWN = 3

    _names = {0: 'static', 1: 'moving_+x', 2: 'moving_-x', 3: 'unknown'}

    @classmethod
    def name(cls, state: int) -> str:
        return cls._names.get(state, 'unknown')


# ==============================================================================
# EGO MOTION ESTIMATOR
# ==============================================================================
class EgoMotionEstimator:
    """基于静态背景 ICP 的运动估计器。

    核心流程:
      1. 地面分割 (RANSAC 平面拟合)
      2. 动态目标剔除 (基于检测框)
      3. 静态结构提取 (地面以上物体)
      4. 帧间 ICP 配准
      5. 方向/运动判断 + 平滑

    Usage:
        estimator = EgoMotionEstimator(config)
        state = estimator.update(points, detections, timestamp)
    """

    DEFAULT_CONFIG = {
        # 地面分割
        'ground_ransac_distance_threshold': 0.15,  # RANSAC 平面内点距离阈值 (m)
        'ground_ransac_n': 3,
        'ground_ransac_max_iterations': 200,
        'ground_height_range': [-0.3, 0.3],        # 被认为地面点的高度范围

        # 点云下采样
        'voxel_size': 0.15,                         # ICP 前体素下采样 (m)

        # ICP 配准
        'icp_max_correspondence_distance': 1.0,     # ICP 对应点最大距离
        'icp_max_iterations': 50,
        'icp_relative_fitness': 1e-6,
        'icp_relative_rmse': 1e-6,

        # 运动判断
        'static_displacement_threshold': 0.05,      # 位移 < 此值判定为静止 (m)
        'slow_motion_threshold': 0.02,              # 极慢运动位移阈值 (m/帧)
        'confirmation_window': 5,                    # 方向变化需连续 N 帧确认
        'history_seconds': 1.0,                     # 用于位移计算的历史时长
        'min_points_for_icp': 500,                  # ICP 所需最少静态点数
        'min_fitness_for_valid': 0.3,               # ICP 最小 fitness

        # 动态目标剔除
        'detection_box_margin': 0.5,                # 检测框外扩边距 (m), 用于剔除
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}

        # 上一帧的静态点云 (用于 ICP)
        self._prev_static_cloud: Optional[np.ndarray] = None
        self._prev_timestamp: Optional[float] = None

        # 运动状态平滑
        self._state_history: deque = deque(maxlen=10)
        self._displacement_history: deque = deque(maxlen=20)
        self._current_state = EgoMotionState.UNKNOWN
        self._current_confidence = 0.0

        # 累积位移 (最近1秒)
        self._displacement_window: deque = deque()  # (timestamp, dx)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        points: np.ndarray,
        detections: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """处理一帧点云，估计 RTG 运动状态。

        Parameters
        ----------
        points : np.ndarray (N, 3+)  [x, y, z, ...]
            当前帧拼接点云 (BEV 坐标系)。
        detections : list of dict or None
            当前帧检测结果 (用于动态目标剔除)。每个 dict 含 x,y,z,w,l,h,yaw。
        timestamp : float or None
            当前帧时间戳 (秒)。

        Returns
        -------
        result : dict
            {
                'state': int (0/1/2/3),
                'state_name': str,
                'confidence': float,
                'displacement': float,  # 最近1秒累积位移 (m)
                'velocity_estimate': float,
                'frame_displacement': float,
                'icp_fitness': float,
                'num_static_points': int,
                'valid': bool,
            }
        """
        if timestamp is None:
            timestamp = time.time()

        # 预处理: 地面分割 + 动态剔除 → 静态点云
        static_points = self._extract_static_points(points, detections)

        if static_points is None or static_points.shape[0] < self.config['min_points_for_icp']:
            # 不足以做 ICP，使用历史状态
            logger.debug(
                'Insufficient static points (%d), reusing previous state',
                static_points.shape[0] if static_points is not None else 0,
            )
            result = self._make_result(
                self._current_state, self._current_confidence,
                displacement=self._compute_cumulative_displacement(timestamp),
            )
            result['valid'] = False
            return result

        # 当前帧点云 → Open3D
        current_cloud = self._to_o3d(static_points)

        if self._prev_static_cloud is not None and HAS_OPEN3D:
            # ICP 配准
            prev_cloud = self._to_o3d(self._prev_static_cloud)
            icp_result = self._run_icp(prev_cloud, current_cloud)
            dx = icp_result['translation'][0]
            fitness = icp_result['fitness']
        else:
            # 首帧或无 Open3D: 无法判断
            dx = 0.0
            fitness = 0.0

        # 存储当前帧作为"上一帧"
        self._prev_static_cloud = static_points.copy()
        self._prev_timestamp = timestamp

        # 更新位移窗口
        dt = 0.05
        if self._prev_timestamp is not None:
            dt_actual = timestamp - self._prev_timestamp
            if dt_actual > 0:
                dt = dt_actual
        self._update_displacement_window(timestamp, dx)

        # 判断运动状态
        state, conf = self._classify_motion(dx, fitness, dt)

        # 平滑
        self._state_history.append(state)
        smoothed_state = self._smooth_state()
        self._current_state = smoothed_state
        self._current_confidence = conf

        cum_disp = self._compute_cumulative_displacement(timestamp)

        result = self._make_result(smoothed_state, conf, cum_disp)
        result['frame_displacement'] = dx
        result['icp_fitness'] = fitness
        result['num_static_points'] = static_points.shape[0]
        result['valid'] = True
        return result

    # ------------------------------------------------------------------
    # Ground Segmentation
    # ------------------------------------------------------------------
    def _extract_static_points(
        self,
        points: np.ndarray,
        detections: Optional[List[Dict[str, Any]]],
    ) -> Optional[np.ndarray]:
        """提取用于 ICP 的静态点云: 地面分割 → 动态剔除。"""
        if points.shape[0] == 0:
            return None

        # 1) 地面分割 (RANSAC)
        non_ground = self._segment_ground(points)

        # 2) 动态目标剔除
        static = self._remove_dynamic_objects(non_ground, detections)

        # 3) 体素下采样
        static = self._voxel_downsample(static)

        return static

    def _segment_ground(self, points: np.ndarray) -> np.ndarray:
        """RANSAC 平面拟合地面分割，返回非地面点。

        在 BEV 坐标系中，地面大致位于 z≈0 平面。
        """
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])

            # RANSAC 平面拟合
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=self.config['ground_ransac_distance_threshold'],
                ransac_n=self.config['ground_ransac_n'],
                num_iterations=self.config['ground_ransac_max_iterations'],
            )

            # 验证平面法向量接近 z 轴
            a, b, c, d = plane_model
            z_alignment = abs(c) / np.sqrt(a**2 + b**2 + c**2)
            if z_alignment < 0.7:
                # 法向量不够垂直 → 不是地面，返回全部点
                logger.debug('RANSAC plane not ground-like (z_align=%.2f)', z_alignment)
                return points

            # 提取非地面点
            inlier_set = set(inliers)
            non_ground_idx = [i for i in range(len(points)) if i not in inlier_set]
            return points[non_ground_idx]

        else:
            # 无 Open3D 降级: 简单高度阈值
            z = points[:, 2]
            gh_min, gh_max = self.config['ground_height_range']
            non_ground_mask = (z < gh_min) | (z > gh_max)
            return points[non_ground_mask]

    @staticmethod
    def _remove_dynamic_objects(
        points: np.ndarray,
        detections: Optional[List[Dict[str, Any]]],
    ) -> np.ndarray:
        """剔除落在检测框内的点（动态目标）。"""
        if points.shape[0] == 0:
            return points
        if not detections:
            return points

        # 构造 mask
        keep = np.ones(len(points), dtype=bool)
        margin = 0.5

        for det in detections:
            dx = det.get('x', 0.0)
            dy = det.get('y', 0.0)
            w = det.get('w', 0.0) / 2.0 + margin
            l = det.get('l', 0.0) / 2.0 + margin
            yaw = det.get('yaw', 0.0)

            # 将点转换到检测框局部坐标系
            px = points[:, 0] - dx
            py = points[:, 1] - dy

            cos_yaw = np.cos(-yaw)
            sin_yaw = np.sin(-yaw)
            px_local = px * cos_yaw - py * sin_yaw
            py_local = px * sin_yaw + py * cos_yaw

            in_box = (np.abs(px_local) <= l) & (np.abs(py_local) <= w)
            keep[in_box] = False

        return points[keep]

    def _voxel_downsample(self, points: np.ndarray) -> np.ndarray:
        """体素下采样。"""
        if points.shape[0] == 0:
            return points
        if HAS_OPEN3D:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            pcd = pcd.voxel_down_sample(voxel_size=self.config['voxel_size'])
            return np.asarray(pcd.points)
        return points

    # ------------------------------------------------------------------
    # ICP
    # ------------------------------------------------------------------
    def _run_icp(
        self,
        source: 'o3d.geometry.PointCloud',
        target: 'o3d.geometry.PointCloud',
    ) -> Dict[str, Any]:
        """执行 Point-to-Plane ICP 配准。"""
        if not HAS_OPEN3D:
            return {'translation': (0.0, 0.0, 0.0), 'fitness': 0.0, 'rmse': 0.0}

        # 估计法向量 (target)
        target.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )

        threshold = self.config['icp_max_correspondence_distance']
        trans_init = np.eye(4)

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.config['icp_relative_fitness'],
                relative_rmse=self.config['icp_relative_rmse'],
                max_iteration=self.config['icp_max_iterations'],
            ),
        )

        tx, ty, tz = reg_p2p.transformation[:3, 3]
        return {
            'translation': (float(tx), float(ty), float(tz)),
            'fitness': float(reg_p2p.fitness),
            'rmse': float(reg_p2p.inlier_rmse),
        }

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------
    def _classify_motion(
        self,
        dx: float,
        fitness: float,
        dt: float,
    ) -> Tuple[int, float]:
        """根据帧间位移和 ICP 质量判断运动状态。

        Returns (state, confidence).
        """
        abs_dx = abs(dx)
        static_thresh = self.config['static_displacement_threshold']

        # 检查 ICP 质量
        min_fit = self.config['min_fitness_for_valid']
        if fitness < min_fit:
            # ICP 质量差，返回未知
            return EgoMotionState.UNKNOWN, max(0.1, fitness)

        if abs_dx < static_thresh:
            # 静止
            conf = 1.0 - (abs_dx / static_thresh) * 0.5
            conf = min(1.0, max(0.5, conf))
            return EgoMotionState.STATIC, conf
        else:
            # 运动: 根据 dx 符号判断方向
            if dx > 0:
                state = EgoMotionState.MOVING_PLUS_X
            else:
                state = EgoMotionState.MOVING_MINUS_X

            # 置信度: 位移量与 fitness 的乘积
            conf = min(1.0, fitness * (abs_dx / (static_thresh * 2)))
            conf = max(0.3, conf)
            return state, conf

    def _smooth_state(self) -> int:
        """对状态历史做多数投票平滑。"""
        if not self._state_history:
            return EgoMotionState.UNKNOWN
        # 多数投票
        from collections import Counter
        counter = Counter(self._state_history)
        return counter.most_common(1)[0][0]

    # ------------------------------------------------------------------
    # Displacement Tracking
    # ------------------------------------------------------------------
    def _update_displacement_window(self, timestamp: float, dx: float) -> None:
        """维护1秒位移窗口。"""
        self._displacement_window.append((timestamp, dx))
        # 清理过期条目
        cutoff = timestamp - self.config['history_seconds']
        while self._displacement_window and self._displacement_window[0][0] < cutoff:
            self._displacement_window.popleft()

    def _compute_cumulative_displacement(self, timestamp: float) -> float:
        """计算最近1秒的累积位移。"""
        cutoff = timestamp - self.config['history_seconds']
        total = 0.0
        for ts, dx in self._displacement_window:
            if ts >= cutoff:
                total += dx
        return abs(total)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_result(
        self,
        state: int,
        confidence: float,
        displacement: float = 0.0,
    ) -> Dict[str, Any]:
        vel = displacement / max(self.config['history_seconds'], 0.1)
        return {
            'state': state,
            'state_name': EgoMotionState.name(state),
            'confidence': confidence,
            'displacement': displacement,
            'velocity_estimate': vel,
            'frame_displacement': 0.0,
            'icp_fitness': 0.0,
            'num_static_points': 0,
            'valid': False,
        }

    @staticmethod
    def _to_o3d(points: np.ndarray) -> 'o3d.geometry.PointCloud':
        """numpy (N,3) → Open3D PointCloud。"""
        if not HAS_OPEN3D:
            raise RuntimeError('Open3D is required for ICP')
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3].copy())
        return pcd

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置估计器状态。"""
        self._prev_static_cloud = None
        self._prev_timestamp = None
        self._state_history.clear()
        self._displacement_history.clear()
        self._displacement_window.clear()
        self._current_state = EgoMotionState.UNKNOWN
        self._current_confidence = 0.0
        logger.info('EgoMotionEstimator reset')
