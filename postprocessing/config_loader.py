"""
统一配置加载模块 — config_loader.py

职责:
  - 加载 calib.yaml, geometry.yaml, warning.yaml, system.yaml
  - 配置验证（距离阈值单调性、外参矩阵有效、内参合理性）
  - 热更新支持（warning.yaml 通过 ROS1 service 接口重载）

所有阈值从配置读取，不做硬编码。
"""

import os
import logging
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 默认配置目录
# ---------------------------------------------------------------------------
DEFAULT_CONFIG_DIR = os.path.join(os.path.dirname(__file__), '..', 'config')


# ==============================================================================
# EXCEPTIONS
# ==============================================================================
class ConfigError(Exception):
    """配置错误基类。"""


class ConfigValidationError(ConfigError):
    """配置验证失败。"""


class ConfigNotFoundError(ConfigError):
    """配置文件未找到。"""


# ==============================================================================
# CONFIG LOADER
# ==============================================================================
class ConfigLoader:
    """统一配置加载器。

    所有后处理模块通过此类获取配置参数，避免零散读取。
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Parameters
        ----------
        config_dir : str or None
            配置目录路径，None 时使用项目默认 config/ 目录。
        """
        self._config_dir = os.path.abspath(
            config_dir or DEFAULT_CONFIG_DIR
        )
        self._lock = threading.Lock()

        # 内部存储
        self._calib: Dict[str, Any] = {}
        self._geometry: Dict[str, Any] = {}
        self._warning: Dict[str, Any] = {}
        self._system: Dict[str, Any] = {}

        self._loaded = False

    # ------------------------------------------------------------------
    # 加载入口
    # ------------------------------------------------------------------
    def load_all(self) -> None:
        """加载并验证全部配置文件。"""
        with self._lock:
            self._calib = self._load_yaml('calib.yaml', required=True)
            self._geometry = self._load_yaml('geometry.yaml', required=True)
            self._warning = self._load_yaml('warning.yaml', required=True)
            self._system = self._load_yaml('system.yaml', required=True)

            self._validate_all()
            self._loaded = True
            logger.info(
                'All configs loaded from %s', self._config_dir
            )

    def reload_warning(self) -> None:
        """热更新 warning.yaml（线程安全）。"""
        with self._lock:
            self._warning = self._load_yaml('warning.yaml', required=True)
            self._validate_warning()
            logger.info('warning.yaml hot-reloaded')

    # ------------------------------------------------------------------
    # 公开访问接口
    # ------------------------------------------------------------------
    @property
    def calib(self) -> Dict[str, Any]:
        return self._calib

    @property
    def geometry(self) -> Dict[str, Any]:
        return self._geometry

    @property
    def warning(self) -> Dict[str, Any]:
        return self._warning

    @property
    def system(self) -> Dict[str, Any]:
        return self._system

    # ------------------------------------------------------------------
    # 便捷查询方法
    # ------------------------------------------------------------------
    def get_distance_thresholds(self) -> Dict[str, Dict[str, float]]:
        """返回类别 → {danger, warning, info} 的距离阈值字典。"""
        return self._warning.get('distance_thresholds', {})

    def get_frame_confirmation(self) -> Dict[str, int]:
        """返回帧确认策略配置。"""
        return self._warning.get('frame_confirmation', {})

    def get_roi_zones(self) -> Dict[str, Dict]:
        """返回 ROI 区域权重配置。"""
        return self._warning.get('roi_zones', {})

    def get_ego_motion_config(self) -> Dict:
        """返回运动状态相关配置。"""
        return self._warning.get('ego_motion', {})

    def get_output_rules(self) -> Dict:
        """返回预警输出规则。"""
        return self._warning.get('output_rules', {})

    def get_ego_footprint_legs(self) -> List[Dict[str, Any]]:
        """返回 4 条支腿的地面投影坐标列表。

        每条腿: {x, y, width, length, side, position}
        """
        footprint = self._geometry.get('ego_footprint', {})
        legs = []
        for side_key, side in footprint.items():
            for pos_key in ('front_leg', 'rear_leg'):
                leg = side.get(pos_key)
                if leg:
                    legs.append({
                        'x': float(leg.get('x', 0)),
                        'y': float(leg.get('y', 0)),
                        'width': float(leg.get('width', 1.0)),
                        'length': float(leg.get('length', 1.0)),
                        'side': side_key,
                        'position': pos_key.replace('_leg', ''),
                    })
        return legs

    def get_extrinsics(self) -> Dict[str, Any]:
        """返回外参矩阵字典。"""
        return self._calib.get('extrinsics', {})

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        """返回相机内参。"""
        return self._calib.get('cameras', {})

    def get_ros_config(self) -> Dict[str, Any]:
        """返回 ROS1 topic 配置。"""
        return self._system.get('ros', {})

    def get_sensor_config(self) -> Dict[str, Any]:
        """返回传感器配置。"""
        return self._system.get('sensors', {})

    def get_performance_targets(self) -> Dict[str, float]:
        """返回性能目标配置。"""
        return self._system.get('performance_targets', {})

    # ------------------------------------------------------------------
    # 内部: YAML 加载
    # ------------------------------------------------------------------
    def _load_yaml(self, filename: str, required: bool = False) -> Dict[str, Any]:
        """加载单个 YAML 文件。"""
        try:
            import yaml
        except ImportError:
            raise ConfigError(
                'PyYAML is required. Install with: pip install pyyaml'
            )

        filepath = os.path.join(self._config_dir, filename)
        if not os.path.exists(filepath):
            if required:
                raise ConfigNotFoundError(f'Config file not found: {filepath}')
            logger.warning('Optional config file not found: %s', filepath)
            return {}

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        if data is None:
            data = {}
        logger.debug('Loaded config: %s', filepath)
        return data

    # ------------------------------------------------------------------
    # 内部: 验证
    # ------------------------------------------------------------------
    def _validate_all(self) -> None:
        """启动时完整验证。"""
        self._validate_calib()
        self._validate_geometry()
        self._validate_warning()
        self._validate_system()
        logger.info('All config validations passed')

    def _validate_calib(self) -> None:
        """验证标定参数。"""
        calib = self._calib

        # 验证相机内参
        cameras = calib.get('cameras', {})
        for cam_name, cam_data in cameras.items():
            intrinsic = cam_data.get('intrinsic')
            if intrinsic:
                fx, fy = intrinsic[0], intrinsic[1]
                if fx <= 0 or fy <= 0:
                    raise ConfigValidationError(
                        f'Invalid intrinsic for {cam_name}: fx={fx}, fy={fy}'
                    )

        # 验证外参矩阵（如有）
        extrinsics = calib.get('extrinsics', {})
        for name, matrix in extrinsics.items():
            self._validate_transform_matrix(name, matrix)

    def _validate_geometry(self) -> None:
        """验证几何配置。"""
        geom = self._geometry

        # 验证 footprint 坐标在 BEV 合理范围
        footprint = geom.get('ego_footprint', {})
        for side_key, side in footprint.items():
            for pos_key in ('front_leg', 'rear_leg'):
                leg = side.get(pos_key, {})
                x = float(leg.get('x', 0))
                y = float(leg.get('y', 0))
                if abs(x) > 60 or abs(y) > 25:
                    logger.warning(
                        'Footprint %s/%s at (%.1f, %.1f) outside BEV range',
                        side_key, pos_key, x, y,
                    )

        # 验证 span_width 合理性
        span = geom.get('rtg_dimensions', {}).get('span_width')
        if span is not None and (span < 15 or span > 35):
            logger.warning('RTG span_width %.1f seems unusual', span)

    def _validate_warning(self) -> None:
        """验证预警配置，特别是阈值单调性。"""
        thresholds = self._warning.get('distance_thresholds', {})
        for cls_name, thresh in thresholds.items():
            danger = thresh.get('danger')
            warning = thresh.get('warning')
            info = thresh.get('info')

            if danger is None or warning is None or info is None:
                raise ConfigValidationError(
                    f'Missing distance threshold for class "{cls_name}": '
                    f'danger={danger}, warning={warning}, info={info}'
                )

            # 验证单调性: danger < warning < info
            if not (0 < danger < warning < info):
                raise ConfigValidationError(
                    f'Distance threshold monotonicity violated for "{cls_name}": '
                    f'danger={danger}, warning={warning}, info={info} '
                    f'(expected: 0 < danger < warning < info)'
                )

        # 验证 ROI 权重
        roi_zones = self._warning.get('roi_zones', {})
        for zone_name, zone_cfg in roi_zones.items():
            weight = zone_cfg.get('weight')
            if weight is None or not (0 <= weight <= 1):
                logger.warning(
                    'ROI zone "%s" weight=%.2f outside [0,1]', zone_name, weight
                )

    def _validate_system(self) -> None:
        """验证系统配置。"""
        sys_cfg = self._system

        # 验证 ROS topic 名非空
        ros_cfg = sys_cfg.get('ros', {})
        for sub_key, sub_cfg in ros_cfg.get('subscribers', {}).items():
            topic = sub_cfg.get('topic', '')
            if not topic:
                logger.warning('Subscriber "%s" topic name is empty', sub_key)

        for pub_key, pub_cfg in ros_cfg.get('publishers', {}).items():
            topic = pub_cfg.get('topic', '')
            if not topic:
                logger.warning('Publisher "%s" topic name is empty', pub_key)

        # 验证同步窗口为正
        sync_win = ros_cfg.get('sync_window', 0)
        if sync_win <= 0:
            logger.warning('Sync window %.3f is non-positive', sync_win)

    @staticmethod
    def _validate_transform_matrix(name: str, matrix) -> None:
        """验证一个 4×4 变换矩阵是否合理。"""
        try:
            mat = np.asarray(matrix, dtype=np.float64)
        except (ValueError, TypeError):
            raise ConfigValidationError(
                f'Extrinsic "{name}" cannot be parsed as matrix'
            )
        if mat.shape != (4, 4):
            raise ConfigValidationError(
                f'Extrinsic "{name}" shape {mat.shape}, expected (4,4)'
            )
        # 检查旋转部分的行列式
        R = mat[:3, :3]
        det = np.linalg.det(R)
        if abs(det) < 1e-6:
            raise ConfigValidationError(
                f'Extrinsic "{name}" rotation is singular (det={det:.2e})'
            )
        if abs(det - 1.0) > 0.1:
            logger.warning(
                'Extrinsic "%s" rotation det=%.3f, expected ~1.0 (may have scale)',
                name, det,
            )

    # ------------------------------------------------------------------
    # 热更新 ROS1 service 接口 (预留)
    # ------------------------------------------------------------------
    def get_warning_param(self, key_path: str) -> Any:
        """通过点分隔路径获取 warning 配置项。

        Example: get_warning_param('distance_thresholds/person/danger') → 8.0
        """
        parts = key_path.strip('/').split('/')
        node = self._warning
        for part in parts:
            if isinstance(node, dict):
                node = node.get(part)
            else:
                return None
        return node

    def set_warning_param(self, key_path: str, value: Any) -> bool:
        """通过点分隔路径设置 warning 配置项（热更新）。

        Returns True on success.
        """
        with self._lock:
            parts = key_path.strip('/').split('/')
            node = self._warning
            for part in parts[:-1]:
                if part not in node:
                    node[part] = {}
                node = node[part]
            node[parts[-1]] = value
            logger.info('Hot-updated warning param: %s = %s', key_path, value)
            # 修改后重新验证
            try:
                self._validate_warning()
            except ConfigValidationError as e:
                logger.error('Hot-update validation failed: %s', e)
                return False
            return True


# ==============================================================================
# 便捷函数: 启动加载
# ==============================================================================
_config_loader_instance: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """获取全局单例 ConfigLoader。

    首次调用时自动加载全部配置。
    """
    global _config_loader_instance
    if _config_loader_instance is None:
        _config_loader_instance = ConfigLoader(config_dir)
        _config_loader_instance.load_all()
    return _config_loader_instance


def reset_config_loader() -> None:
    """重置全局配置加载器（测试用）。"""
    global _config_loader_instance
    _config_loader_instance = None
