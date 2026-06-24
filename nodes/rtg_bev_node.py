#!/usr/bin/env python
"""
RTG BEV 防撞系统主节点 — rtg_bev_node.py

ROS1 主节点骨架 (基于 rospy):
  - 订阅: 2相机 + 2雷达 topic（按 system.yaml 配置）
  - 同步器: 时间戳对齐（50ms 窗口）
  - 推理调用（CenterPoint LiDAR-only 推理接口）
  - 发布: detections/tracks/warnings/ego_motion/diagnostics
  - 配置加载: 启动时读取所有 YAML 配置文件
  - 诊断信息: 传感器帧率、时间戳异常、标定文件状态

依赖:
  - rospy (可选, try/except 优雅降级)
  - sensor_msgs, diagnostic_msgs (ROS1 标准)
  - rtg_bev_msgs (本项目的自定义消息包)
"""

import argparse
import os
import sys
import time
import logging
import traceback
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# torch 可选导入（推理时使用）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

# ---------------------------------------------------------------------------
# 尝试导入 ROS1 依赖 (优雅降级)
# ---------------------------------------------------------------------------
ROS_AVAILABLE = False
try:
    import rospy
    ROS_AVAILABLE = True
except ImportError:
    rospy = None
    print('[rtg_bev_node] rospy not available; running in standalone/offline mode')

try:
    import yaml
except ImportError:
    yaml = None

# 项目内部模块 (相对于项目根目录)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from postprocessing.config_loader import ConfigLoader, ConfigError
from postprocessing.footprint_filter import SelfFootprintFilter
from postprocessing.tracker import Tracker
from postprocessing.ego_motion import EgoMotionEstimator, EgoMotionState
from postprocessing.warning_engine import WarningEngine

# ── 类别映射 (模型输出 → 跟踪器输入) ──────────────────────────────────
NUSC_CLASSES = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer',
    'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone',
]
NUSC_TO_RTG_CLASS = {
    'pedestrian': 'person', 'truck': 'truck', 'car': 'car',
    'trailer': 'truck', 'bus': 'other_obstacle',
    'construction_vehicle': 'other_obstacle', 'motorcycle': 'car',
    'bicycle': 'other_obstacle', 'traffic_cone': 'other_obstacle',
    'barrier': 'other_obstacle',
}
RTG_CLASS_NAME_TO_ID = {
    'person': 0, 'truck': 1, 'car': 2, 'other_obstacle': 3,
}

logger = logging.getLogger('rtg_bev_node')


# ==============================================================================
# SENSOR BUFFER
# ==============================================================================
class SensorBuffer:
    """环形缓冲区，缓存最近 N 条传感器消息用于时间戳同步。"""

    def __init__(self, maxlen: int = 10):
        self._buffer = deque(maxlen=maxlen)
        self._last_emitted_lidar_pair_time: Optional[float] = None

    def add(self, msg, ts: float, sensor_key: str) -> None:
        self._buffer.append({
            'msg': msg,
            'timestamp': ts,
            'sensor': sensor_key,
        })

    def get_synced_frame(self, window: float = 0.05) -> Optional[Dict[str, Any]]:
        """查找一组时间窗口内对齐的传感器消息。

        Parameters
        ----------
        window : float
            时间容忍窗口 (秒)。

        Returns
        -------
        frame : dict or None
            {lidar_01: msg, lidar_02: msg, camera_01?: msg, timestamp: float}
        """
        if len(self._buffer) < 2:
            return None

        # 倒序查找: 从最新消息向前找匹配组
        buffer_list = list(self._buffer)
        for i in range(len(buffer_list) - 1, -1, -1):
            anchor = buffer_list[i]
            anchor_ts = anchor['timestamp']
            group = {anchor['sensor']: anchor['msg']}
            group_ts = {anchor['sensor']: anchor_ts}

            for j in range(len(buffer_list)):
                if j == i:
                    continue
                entry = buffer_list[j]
                if abs(entry['timestamp'] - anchor_ts) <= window:
                    if entry['sensor'] not in group:
                        group[entry['sensor']] = entry['msg']
                        group_ts[entry['sensor']] = entry['timestamp']

            # V1 LiDAR-only 闭环需要 L1/L2；相机是可视化/扩展输入。
            if 'lidar_01' in group and 'lidar_02' in group:
                pair_time = max(group_ts['lidar_01'], group_ts['lidar_02'])
                if (
                    self._last_emitted_lidar_pair_time is not None and
                    pair_time <= self._last_emitted_lidar_pair_time + 1e-9
                ):
                    return None
                self._last_emitted_lidar_pair_time = pair_time
                return {
                    **group,
                    'timestamp': pair_time,
                }

        return None

    def clear_old(self, before_ts: float) -> int:
        """清除时间戳早于 before_ts 的条目。返回清除数量。"""
        before = len(self._buffer)
        while self._buffer and self._buffer[0]['timestamp'] < before_ts:
            self._buffer.popleft()
        return before - len(self._buffer)


# ==============================================================================
# DIAGNOSTICS
# ==============================================================================
class DiagnosticsCollector:
    """诊断信息收集器。

    采集传感器帧率、推理延迟、时间戳异常等，仅记录日志不实时告警。
    """

    def __init__(self):
        # 帧率统计: {sensor_key: deque of timestamps}
        self._frame_times: Dict[str, deque] = {}
        self._last_inference_latency = 0.0
        self._output_frame_count = 0
        self._start_time = time.time()
        self._sensor_anomalies: List[str] = []
        self._calib_status = 'unknown'

    def record_sensor_frame(self, sensor_key: str, timestamp: float) -> None:
        """记录一帧传感器数据。"""
        if sensor_key not in self._frame_times:
            self._frame_times[sensor_key] = deque(maxlen=100)
        self._frame_times[sensor_key].append(timestamp)

    def record_inference(self, latency_ms: float) -> None:
        self._last_inference_latency = latency_ms

    def record_output_frame(self) -> None:
        self._output_frame_count += 1

    def set_calib_status(self, status: str) -> None:
        self._calib_status = status

    def get_fps(self, sensor_key: str) -> float:
        """估算传感器实际帧率。"""
        times = self._frame_times.get(sensor_key, deque())
        if len(times) < 2:
            return 0.0
        duration = times[-1] - times[0]
        if duration <= 0:
            return 0.0
        return len(times) / duration

    def get_output_fps(self) -> float:
        elapsed = time.time() - self._start_time
        if elapsed <= 0:
            return 0.0
        return self._output_frame_count / elapsed

    def get_diagnostics_report(self) -> Dict[str, Any]:
        """生成诊断报告。"""
        report = {
            'timestamp': time.time(),
            'uptime_seconds': time.time() - self._start_time,
            'sensor_fps': {},
            'inference_latency_ms': self._last_inference_latency,
            'output_fps': self.get_output_fps(),
            'calib_status': self._calib_status,
            'anomalies': list(self._sensor_anomalies),
        }
        for key in self._frame_times:
            report['sensor_fps'][key] = round(self.get_fps(key), 1)
        return report

    def log_diagnostics(self) -> None:
        """记录诊断日志。"""
        report = self.get_diagnostics_report()
        fps_str = ', '.join(f'{k}={v}Hz' for k, v in report['sensor_fps'].items())
        logger.info(
            'DIAG: uptime=%.1fs, sensor_fps=[%s], output_fps=%.1f, '
            'inference_latency=%.1fms, calib=%s',
            report['uptime_seconds'], fps_str,
            report['output_fps'], report['inference_latency_ms'],
            report['calib_status'],
        )


# ==============================================================================
# MAIN NODE
# ==============================================================================
class RTGBEVNode:
    """RTG BEV 防撞系统 ROS1 主节点。

    数据流:
      传感器回调 → 同步缓冲区 → 预处理 → 推理 → 跟踪 → 运动估计 → 预警 → 发布
    """

    def __init__(self, config_dir: str, ros_node_name: str = 'rtg_bev_node'):
        """
        Parameters
        ----------
        config_dir : str
            配置文件目录路径。
        ros_node_name : str
            ROS1 节点名称。
        """
        self._config_dir = os.path.abspath(config_dir)
        self._ros_node_name = ros_node_name

        # 加载配置
        logger.info('Loading configs from %s ...', self._config_dir)
        self._config = ConfigLoader(self._config_dir)
        self._config.load_all()

        self._system_cfg = self._config.system
        self._ros_cfg = self._system_cfg.get('ros', {})
        self._perf_targets = self._config.get_performance_targets()

        # 初始化模块
        self._tracker = Tracker()
        self._ego_motion = EgoMotionEstimator()
        self._warning_engine = WarningEngine(self._config)
        self._diagnostics = DiagnosticsCollector()
        # 自车 footprint 点云过滤
        geometry_path = os.path.join(self._config_dir, 'geometry.yaml')
        self._footprint_filter = SelfFootprintFilter(geometry_path) \
            if os.path.exists(geometry_path) else None

        # 模型推理
        self._model: Any = None
        self._model_ready = False
        self._cached_img_metas: Any = None
        self._model_class_names = list(NUSC_CLASSES)
        self._torch_device: Any = None
        self._model_fp16 = False
        try:
            self._model = self._load_model()
        except Exception as e:
            logger.warning('Model loading skipped: %s', e)
            self._model = None
            self._model_ready = False

        # 传感器缓冲区 (用于时间戳同步)
        self._sensor_buffer = SensorBuffer(maxlen=30)
        self._sync_window = self._ros_cfg.get('sync_window', 0.05)

        # ROS1 状态
        self._ros_initialized = False
        self._publishers: Dict[str, Any] = {}
        self._subscribers: List[Any] = []

        # 主循环控制
        self._running = False
        self._last_diag_time = time.time()
        self._diag_interval = 1.0  # 诊断输出周期 (秒)
        self._debug_frame_index = 0

        # 初始化 ROS1 (如果可用)
        if ROS_AVAILABLE:
            self._init_ros()

        # 验证标定文件
        self._diagnostics.set_calib_status(
            'ok' if self._config.calib else 'missing'
        )

        logger.info('RTGBEVNode initialized')

    # ------------------------------------------------------------------
    # ROS1 初始化
    # ------------------------------------------------------------------
    def _init_ros(self) -> None:
        """初始化 ROS1 节点、订阅者和发布者。"""
        rospy.init_node(self._ros_node_name, anonymous=False)

        # 创建发布者
        publishers_cfg = self._ros_cfg.get('publishers', {})
        for pub_name, pub_cfg in publishers_cfg.items():
            topic = pub_cfg.get('topic', '')
            msg_type_str = pub_cfg.get('type', '')
            queue_size = pub_cfg.get('queue_size', 10)

            msg_type = _resolve_msg_type(msg_type_str)
            if msg_type is not None and topic:
                self._publishers[pub_name] = rospy.Publisher(
                    topic, msg_type, queue_size=queue_size
                )
                logger.info('Publisher: %s → %s (%s)', pub_name, topic, msg_type_str)
            else:
                logger.warning('Cannot create publisher "%s": type=%s not resolved',
                               pub_name, msg_type_str)

        # 创建订阅者
        subscribers_cfg = self._ros_cfg.get('subscribers', {})
        for sub_name, sub_cfg in subscribers_cfg.items():
            topic = sub_cfg.get('topic', '')
            msg_type_str = sub_cfg.get('type', '')
            queue_size = sub_cfg.get('queue_size', 5)
            if not topic:
                continue

            msg_type = _resolve_msg_type(msg_type_str)
            if msg_type is not None:
                sub = rospy.Subscriber(
                    topic,
                    msg_type,
                    callback=self._make_sensor_callback(sub_name),
                    queue_size=queue_size,
                )
                self._subscribers.append(sub)
                logger.info('Subscriber: %s ← %s (%s)', sub_name, topic, msg_type_str)

        # 创建 ROS1 service (热更新 warning.yaml)
        try:
            from std_srvs.srv import Empty, EmptyResponse
            rospy.Service(
                '/rtg_bev/reload_warning_config',
                Empty,
                self._handle_reload_warning,
            )
            logger.info('Service: /rtg_bev/reload_warning_config')
        except Exception:
            logger.warning('Could not create reload_warning_config service')

        try:
            srv_type = _make_set_param_srv()
            if srv_type is not None:
                rospy.Service(
                    '/rtg_bev/set_warning_param',
                    srv_type,
                    self._handle_set_warning_param,
                )
                logger.info('Service: /rtg_bev/set_warning_param')
            else:
                logger.warning('set_warning_param service not available '
                               '(rtg_bev_msgs.srv not built)')
        except Exception:
            logger.warning('Could not create set_warning_param service')

        self._ros_initialized = True

    # ------------------------------------------------------------------
    # 传感器回调工厂
    # ------------------------------------------------------------------
    def _make_sensor_callback(self, sensor_key: str):
        """为每个传感器创建回调函数 (闭包)。"""
        def callback(msg):
            ts = _extract_timestamp(msg)
            self._sensor_buffer.add(msg, ts, sensor_key)
            self._diagnostics.record_sensor_frame(sensor_key, ts)
        return callback

    # ------------------------------------------------------------------
    # ROS1 Service handlers
    # ------------------------------------------------------------------
    def _handle_reload_warning(self, req):
        """处理 warning.yaml 重载请求。"""
        try:
            self._config.reload_warning()
            self._warning_engine.reset()
            logger.info('warning.yaml reloaded via service')
            if ROS_AVAILABLE:
                from std_srvs.srv import EmptyResponse
                return EmptyResponse()
        except Exception as e:
            logger.error('Failed to reload warning.yaml: %s', e)
        if ROS_AVAILABLE:
            from std_srvs.srv import EmptyResponse
            return EmptyResponse()

    def _handle_set_warning_param(self, req):
        """处理单个参数热更新请求。"""
        try:
            key = getattr(req, 'key', '')
            value_str = getattr(req, 'value', '')
            try:
                value = float(value_str)
            except ValueError:
                value = value_str
            success = self._config.set_warning_param(key, value)
            logger.info('set_warning_param %s=%s → %s', key, value_str, success)
            if ROS_AVAILABLE:
                from rtg_bev_msgs.srv import SetWarningParamResponse
                return SetWarningParamResponse(
                    success=success,
                    message=f'{key}={value_str}' if success else f'failed: {key}'
                )
        except Exception as e:
            logger.error('Failed to set warning param: %s', e)
            if ROS_AVAILABLE:
                from rtg_bev_msgs.srv import SetWarningParamResponse
                return SetWarningParamResponse(success=False, message=str(e))
        if ROS_AVAILABLE:
            from rtg_bev_msgs.srv import SetWarningParamResponse
            return SetWarningParamResponse(success=False, message='ROS not available')

    # ------------------------------------------------------------------
    # 主处理循环
    # ------------------------------------------------------------------
    def process_frame(
        self,
        sensor_frame: Dict[str, Any],
    ) -> Dict[str, Any]:
        """处理一帧同步后的传感器数据。

        Parameters
        ----------
        sensor_frame : dict
            {camera_01, camera_02, lidar_01, lidar_02, timestamp, ...}

        Returns
        -------
        output : dict
            {detections, tracks, warnings, ego_motion}
        """
        t_start = time.time()
        ts = sensor_frame.get('timestamp', time.time())

        # ---------------------------------------------------------------
        # 步骤 1: 预处理 — 点云拼接 + 图像 decode (预留)
        # ---------------------------------------------------------------
        lidar_points = self._preprocess_lidar(sensor_frame)
        images = self._preprocess_camera(sensor_frame)

        # ---------------------------------------------------------------
        # 步骤 2: 推理 — CenterPoint 模型推理
        # ---------------------------------------------------------------
        detections = self._run_inference(lidar_points, images, ts)
        self._annotate_detection_distances(detections)

        # ---------------------------------------------------------------
        # 步骤 3: 跟踪
        # ---------------------------------------------------------------
        tracks = self._tracker.update(detections, ts)

        # ---------------------------------------------------------------
        # 步骤 4: 运动估计
        # ---------------------------------------------------------------
        ego_result = self._ego_motion.update(lidar_points, detections, ts)

        # ---------------------------------------------------------------
        # 步骤 5: 预警
        # ---------------------------------------------------------------
        warning_result = self._warning_engine.evaluate(
            tracks, ego_result['state'], ts,
        )

        # ---------------------------------------------------------------
        # 汇总输出
        # ---------------------------------------------------------------
        latency_ms = (time.time() - t_start) * 1000.0
        self._diagnostics.record_inference(latency_ms)
        self._diagnostics.record_output_frame()

        output = {
            'timestamp': ts,
            'detections': detections,
            'tracks': tracks,
            'warnings': warning_result.get('warnings', []),
            'active_zones': warning_result.get('active_zones', []),
            'ego_motion': ego_result,
            'diagnostics': self._diagnostics.get_diagnostics_report(),
            'latency_ms': latency_ms,
        }

        self._maybe_save_bev_debug(output, lidar_points)

        # 调试日志
        logger.debug(
            'Frame ts=%.3f: %d dets, %d tracks, %d warnings, '
            'ego=%s, latency=%.1fms',
            ts,
            len(detections), len(tracks),
            len(warning_result.get('warnings', [])),
            ego_result['state_name'],
            latency_ms,
        )

        return output

    # ------------------------------------------------------------------
    # 预处理 (占位)
    # ------------------------------------------------------------------
    def _preprocess_lidar(
        self, sensor_frame: Dict[str, Any]
    ) -> np.ndarray:
        """点云预处理: 外参拼接 + self-footprint 过滤。

        处理流程:
          1. 读取 lidar_01 (L1, front/BEV origin) 和 lidar_02 (L2, rear) 点云
          2. L2 外参变换到 L1/BEV 坐标系 (LIDAR_RE_to_LIDAR_FR)
          3. 各自 footprint 过滤 (按 lidar_id)
          4. 拼接为统一 BEV 点云
        """
        # ---- 加载 L2→L1 外参 ----
        T_l2_to_l1 = np.eye(4, dtype=np.float32)
        calib = self._config.calib
        extrinsics = calib.get('extrinsics', {})
        lr2lf = extrinsics.get('LIDAR_RE_to_LIDAR_FR', {})
        if lr2lf:
            R = np.array(lr2lf.get('R', [[1,0,0],[0,1,0],[0,0,1]]), dtype=np.float32)
            T = np.array(lr2lf.get('T', [0,0,0]), dtype=np.float32)
            T_l2_to_l1[:3, :3] = R
            T_l2_to_l1[:3, 3] = T

        lidar_inputs = [
            (1, 'lidar_01'),  # (lidar_id, sensor_key): L1 = front/BEV origin
            (2, 'lidar_02'),  # L2 = rear
        ]

        merged_parts = []
        for lidar_id, key in lidar_inputs:
            msg = sensor_frame.get(key)
            if msg is None:
                continue
            pts = _pointcloud2_to_numpy(msg)
            if pts is None or len(pts) == 0:
                continue

            # L2 → L1/BEV 坐标变换
            filter_lidar_id = lidar_id
            if lidar_id == 2:
                N = pts.shape[0]
                pts_xyz = pts[:, :3]
                ones = np.ones((N, 1), dtype=np.float32)
                pts_h = np.hstack([pts_xyz, ones])
                pts_xyz_t = (T_l2_to_l1 @ pts_h.T).T[:, :3]
                pts[:, :3] = pts_xyz_t
                filter_lidar_id = 1  # 已变换到BEV，用L1偏移(即不加偏移)

            # 自车 footprint 过滤
            if self._footprint_filter is not None and \
               self._footprint_filter.has_footprint:
                pts = self._footprint_filter.filter(pts, lidar_id=filter_lidar_id)

            if len(pts) > 0:
                merged_parts.append(pts)

        if not merged_parts:
            return np.zeros((0, 5), dtype=np.float32)

        return np.vstack(merged_parts)

    def _preprocess_camera(
        self, sensor_frame: Dict[str, Any]
    ) -> Dict[str, Any]:
        """相机预处理: 图像 decode + resize + normalize。

        当前为占位实现。
        """
        images = {}
        for key in ('camera_01', 'camera_02'):
            msg = sensor_frame.get(key)
            if msg is not None:
                # 实际: sensor_msgs/Image → numpy
                img = _image_msg_to_numpy(msg)
                if img is not None:
                    images[key] = img
        return images

    # ------------------------------------------------------------------
    # 推理
    # ------------------------------------------------------------------
    def _load_model(self) -> Optional[Any]:
        """加载 CenterPoint 模型 (det3d 框架)。

        Returns None if model loading is disabled or fails.
        """
        model_cfg = self._system_cfg.get('model', {})
        enabled = model_cfg.get('enabled', None)
        if enabled is False:
            logger.info('Model loading disabled (system.yaml: model.enabled=false)')
            return None
        if not HAS_TORCH:
            logger.warning('PyTorch not installed; inference disabled')
            return None

        config_path = model_cfg.get('config_path', '')
        checkpoint_path = model_cfg.get('checkpoint_path', '')

        if not config_path or not checkpoint_path:
            logger.warning('Model config_path or checkpoint_path is empty')
            return None

        # 转为绝对路径 (基于 CenterPoint 目录)
        cp_root = os.path.join(_PROJECT_ROOT, 'CenterPoint')
        _ensure_import_path(cp_root)
        if not os.path.isabs(config_path):
            config_path = os.path.join(cp_root, config_path)
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(cp_root, checkpoint_path)

        if enabled is None and not os.path.exists(checkpoint_path):
            logger.warning('Model checkpoint not found; inference disabled: %s',
                           checkpoint_path)
            return None

        try:
            from det3d.torchie import Config
            from det3d.models import build_detector

            logger.info('Loading model config: %s', config_path)
            cfg = Config.fromfile(config_path)
            cfg_class_names = getattr(cfg, 'class_names', None)
            if cfg_class_names is None and hasattr(cfg, 'get'):
                cfg_class_names = cfg.get('class_names', None)
            if cfg_class_names:
                self._model_class_names = [str(name) for name in cfg_class_names]

            model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)

            logger.info('Loading checkpoint: %s', checkpoint_path)
            from det3d.torchie.trainer import load_checkpoint
            load_checkpoint(model, checkpoint_path, map_location='cpu')

            # 保存 test_cfg 供推理时使用
            model._test_cfg = cfg.test_cfg
            model._voxel_cfg = cfg.voxel_generator if hasattr(cfg, 'voxel_generator') else None

            # 设备
            gpu_cfg = self._system_cfg.get('gpu', {})
            device = _resolve_torch_device(gpu_cfg)
            fp16 = bool(model_cfg.get('fp16', False))
            if str(device) == 'cpu' and fp16:
                logger.warning('fp16 disabled for CPU inference fallback')
                fp16 = False

            model = model.to(device)
            model.eval()
            if fp16:
                model = model.half()
            self._torch_device = device
            self._model_fp16 = fp16

            # 初始化 voxel generator (用于推理时体素化)
            from det3d.core.input.voxel_generator import VoxelGenerator
            pc_range = cfg.get('point_cloud_range', [-60.0, -25.2, -3.0, 60.0, 25.2, 8.0])
            vs = cfg.get('voxel_size', [0.075, 0.075, 0.2])
            self._voxel_generator = VoxelGenerator(
                voxel_size=vs,
                point_cloud_range=pc_range,
                max_num_points=10,
                max_voxels=160000,
            )
            self._voxel_grid_size = tuple(self._voxel_generator.grid_size)

            self._model_ready = True
            logger.info('CenterPoint model loaded successfully')
            return model

        except Exception as e:
            logger.error('Failed to load model: %s', e, exc_info=True)
            self._model_ready = False
            return None

    def _run_inference(
        self,
        points: np.ndarray,
        images: Dict[str, Any],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """CenterPoint 推理接口 (LiDAR-only)。

        数据流: 体素化 → reader → backbone → neck → head → decode+NMS
        """
        if not getattr(self, '_model_ready', False) or self._model is None:
            return []

        model_cfg = self._system_cfg.get('model', {})
        score_thr = model_cfg.get('score_threshold', 0.1)
        fp16 = bool(getattr(self, '_model_fp16', False))
        device = getattr(self, '_torch_device', None)
        if device is None:
            logger.error('Model is ready but torch device is not set')
            return []

        try:
            if points.shape[0] == 0:
                return []

            # 限制点数 + 确保5通道
            max_pts = model_cfg.get('max_points', 200000)
            if points.shape[0] > max_pts:
                idx = np.random.choice(points.shape[0], max_pts, replace=False)
                points = points[idx]
            if points.shape[1] < 5:
                points = np.pad(points, ((0, 0), (0, 5 - points.shape[1])),
                                constant_values=0)

            # Step 1: 体素化
            voxels, coordinates, num_points = self._voxel_generator.generate(
                points, max_voxels=160000
            )
            coordinates = _add_batch_index_to_coordinates(coordinates)

            voxel_dtype = torch.float16 if fp16 else torch.float32
            voxels_t = torch.from_numpy(voxels).to(device=device, dtype=voxel_dtype)
            num_points_t = torch.from_numpy(num_points).to(device=device, dtype=torch.float32)
            coors_t = torch.from_numpy(coordinates).to(device=device, dtype=torch.int32)

            with torch.no_grad():
                # Step 2: Reader (VoxelFeatureExtractorV3)
                input_features = self._model.reader(voxels_t, num_points_t)
                if fp16:
                    input_features = input_features.half()

                # Step 3: Backbone (SpMiddleResNetFHD)
                x, _ = self._model.backbone(
                    input_features, coors_t, 1, self._voxel_grid_size
                )

                # Step 4: Neck (RPN)
                x = self._model.neck(x)

                # Step 5: Head (CenterHead)
                preds, _ = self._model.bbox_head(x)

                # Step 6: Decode + NMS
                example = {'metadata': []}
                result = self._model.bbox_head.predict(example, preds, self._model._test_cfg)

            # 解析结果
            if not result:
                return []

            return _centerpoint_results_to_detections(
                result,
                score_thr,
                self._model_class_names,
            )

        except Exception as e:
            logger.error('Inference error: %s', e, exc_info=True)
            return []

    def _annotate_detection_distances(self, detections: List[Dict[str, Any]]) -> None:
        """Fill detection distance to nearest RTG leg for ROS output/debugging."""
        legs = self._config.get_ego_footprint_legs()
        if not legs:
            return
        for det in detections:
            tx, ty = det.get('x', 0.0), det.get('y', 0.0)
            det['distance'] = float(min(
                np.hypot(tx - leg['x'], ty - leg['y']) for leg in legs
            ))

    def _maybe_save_bev_debug(
        self,
        output: Dict[str, Any],
        lidar_points: np.ndarray,
    ) -> None:
        """Save lightweight BEV debug PNGs when debug.bev_visualization.enabled."""
        debug_cfg = self._system_cfg.get('debug', {}).get('bev_visualization', {})
        if not debug_cfg.get('enabled', False):
            return

        save_every = int(debug_cfg.get('save_every_n_frames', 10))
        if save_every <= 0:
            save_every = 10
        self._debug_frame_index += 1
        if self._debug_frame_index % save_every != 0:
            return

        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            from tools._draw_tracking import draw_bev_scene_background, draw_track_box

            log_dir = self._system_cfg.get('logging', {}).get('log_dir', 'logs')
            save_dir = debug_cfg.get('save_dir', os.path.join(log_dir, 'bev_debug'))
            if not os.path.isabs(save_dir):
                save_dir = os.path.join(_PROJECT_ROOT, save_dir)
            os.makedirs(save_dir, exist_ok=True)

            fig, ax = plt.subplots(figsize=(10, 6))
            draw_bev_scene_background(ax, draw_labels=False)

            max_points = int(debug_cfg.get('max_points', 12000))
            if lidar_points is not None and len(lidar_points) > 0:
                pts = lidar_points
                if len(pts) > max_points:
                    idx = np.linspace(0, len(pts) - 1, max_points).astype(np.int64)
                    pts = pts[idx]
                ax.scatter(
                    pts[:, 0], pts[:, 1],
                    s=0.2, c=pts[:, 2], cmap='viridis',
                    alpha=0.25, linewidths=0, zorder=1,
                )

            if debug_cfg.get('show_detections', True):
                for det in output.get('detections', []):
                    draw_track_box(ax, {**det, 'track_id': -1, 'state': 0},
                                   alpha=0.35, linewidth=0.7,
                                   show_label=False, show_velocity=False)

            if debug_cfg.get('show_tracks', True):
                for trk in output.get('tracks', []):
                    draw_track_box(ax, trk, alpha=0.85, show_label=False)

            title = (
                f"ts={output.get('timestamp', 0):.3f} "
                f"det={len(output.get('detections', []))} "
                f"trk={len(output.get('tracks', []))} "
                f"warn={len(output.get('warnings', []))}"
            )
            ax.set_title(title, fontsize=9)
            fig.tight_layout()
            path = os.path.join(save_dir, f"bev_{self._debug_frame_index:06d}.png")
            fig.savefig(path, dpi=140)
            plt.close(fig)
        except Exception as e:
            logger.warning('Failed to save BEV debug visualization: %s', e)

    # ------------------------------------------------------------------
    # 发布
    # ------------------------------------------------------------------
    def publish(self, output: Dict[str, Any]) -> None:
        """发布所有输出 topic。"""
        if not self._ros_initialized:
            return

        ts = output['timestamp']

        # detections
        if 'detections' in self._publishers:
            det_msg = _build_detection_array(output['detections'], ts)
            self._publishers['detections'].publish(det_msg)

        # tracks
        if 'tracks' in self._publishers:
            trk_msg = _build_track_array(output['tracks'], ts)
            self._publishers['tracks'].publish(trk_msg)

        # warnings
        if 'warnings' in self._publishers:
            warn_msg = _build_warning_array(output, ts)
            self._publishers['warnings'].publish(warn_msg)

        # ego_motion
        if 'ego_motion' in self._publishers:
            ego_msg = _build_ego_motion_state(output['ego_motion'])
            self._publishers['ego_motion'].publish(ego_msg)

    # ------------------------------------------------------------------
    # 运行循环
    # ------------------------------------------------------------------
    def run(self) -> None:
        """主循环: 反复尝试同步传感器 → 处理 → 发布。"""
        self._running = True
        logger.info('RTGBEVNode running (ROS=%s)', ROS_AVAILABLE)

        if ROS_AVAILABLE:
            self._run_ros_spin()
        else:
            self._run_standalone()

    def _run_ros_spin(self) -> None:
        """ROS1 回调驱动模式。"""
        rate = rospy.Rate(20)  # 20Hz 处理循环

        while not rospy.is_shutdown() and self._running:
            # 尝试从缓冲区获取同步帧
            sensor_frame = self._sensor_buffer.get_synced_frame(self._sync_window)

            if sensor_frame is not None:
                try:
                    output = self.process_frame(sensor_frame)
                    self.publish(output)
                except Exception:
                    logger.error('Frame processing error:\n%s', traceback.format_exc())

            # 定期输出诊断
            now = time.time()
            if now - self._last_diag_time >= self._diag_interval:
                self._diagnostics.log_diagnostics()
                self._last_diag_time = now

            # 清理旧传感器数据 (ROS 环境用 sim time, 否则用 wall time)
            cutoff = (rospy.get_time() if ROS_AVAILABLE else time.time()) - 1.0
            self._sensor_buffer.clear_old(cutoff)

            rate.sleep()

    def _run_standalone(self) -> None:
        """无 ROS1 的独立模式 (仅用于测试)。"""
        logger.warning('Running in standalone mode (no ROS1)')
        try:
            while self._running:
                time.sleep(0.05)
                now = time.time()
                if now - self._last_diag_time >= self._diag_interval:
                    self._diagnostics.log_diagnostics()
                    self._last_diag_time = now
        except KeyboardInterrupt:
            self._running = False
            logger.info('Shutdown')

    def shutdown(self) -> None:
        """关闭节点。"""
        self._running = False
        logger.info('RTGBEVNode shutdown')


# ==============================================================================
# ROS1 Message Helpers
# ==============================================================================
def _extract_timestamp(msg) -> float:
    """从 ROS1 消息中提取时间戳。"""
    receive_time = time.time()
    if ROS_AVAILABLE and rospy is not None:
        try:
            receive_time = rospy.get_time()
        except Exception:
            receive_time = time.time()

    if hasattr(msg, 'header') and hasattr(msg.header, 'stamp'):
        stamp = msg.header.stamp
        if hasattr(stamp, 'to_sec'):
            ts = stamp.to_sec()
        else:
            ts = float(stamp) if stamp else 0.0
        # Ouster bags in this project use device/internal header time, and
        # archived bags can be far from wall time. In those cases use receive
        # time so L1/L2/camera callbacks can still form synchronized frames.
        if ts <= 0 or ts < 1e9 or abs(receive_time - ts) > 3600.0:
            return receive_time
        return ts
    return receive_time


def _tensor_to_numpy(value):
    """Convert torch tensors or array-like values to numpy without assuming CPU."""
    if value is None:
        return None
    if hasattr(value, 'detach'):
        value = value.detach()
    if hasattr(value, 'cpu'):
        value = value.cpu()
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def _centerpoint_results_to_detections(
    result: List[Dict[str, Any]],
    score_thr: float,
    class_names: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Convert CenterPoint prediction dicts into RTG detection dicts."""
    detections: List[Dict[str, Any]] = []
    for item in result:
        boxes = _tensor_to_numpy(item.get('box3d_lidar', None))
        scores = _tensor_to_numpy(item.get('scores', None))
        labels = _tensor_to_numpy(item.get('label_preds', None))

        if boxes is None or scores is None:
            continue

        boxes = np.asarray(boxes)
        scores = np.asarray(scores).reshape(-1)
        if labels is None:
            labels = np.zeros(scores.shape[0], dtype=np.int64)
        labels = np.asarray(labels).reshape(-1)
        if boxes.ndim == 1:
            boxes = boxes.reshape(1, -1)

        for j, box in enumerate(boxes):
            if j >= scores.shape[0] or scores[j] < score_thr or box.shape[0] < 6:
                continue

            cls_label = int(labels[j]) if j < labels.shape[0] else 0
            cls_id, cls_name = _map_model_label_to_rtg_class(cls_label, class_names)

            # CenterPoint boxes are [x,y,z,w,l,h,vx,vy,yaw] for 9-dim output.
            # Seven-dim boxes omit velocity and keep yaw at index 6.
            box_dim = int(box.shape[0])
            yaw = float(box[-1]) if box_dim >= 7 else 0.0
            vx = float(box[6]) if box_dim >= 9 else 0.0
            vy = float(box[7]) if box_dim >= 9 else 0.0

            detections.append({
                'class_id': cls_id,
                'class_name': cls_name,
                'confidence': float(scores[j]),
                'x': float(box[0]),
                'y': float(box[1]),
                'z': float(box[2]),
                'w': float(box[3]),
                'l': float(box[4]),
                'h': float(box[5]),
                'yaw': yaw,
                'vx': vx,
                'vy': vy,
            })

    return detections


def _map_model_label_to_rtg_class(
    label: int,
    class_names: Optional[List[str]] = None,
) -> Tuple[int, str]:
    rtg_names = list(RTG_CLASS_NAME_TO_ID.keys())
    label_names = class_names if class_names else NUSC_CLASSES
    source_name = None

    if 0 <= label < len(label_names):
        source_name = label_names[label]
    elif 0 <= label < len(rtg_names):
        source_name = rtg_names[label]

    if source_name in RTG_CLASS_NAME_TO_ID:
        rtg_name = source_name
    elif source_name in NUSC_TO_RTG_CLASS:
        rtg_name = NUSC_TO_RTG_CLASS[source_name]
    else:
        rtg_name = 'other_obstacle'

    return RTG_CLASS_NAME_TO_ID[rtg_name], rtg_name


def _resolve_torch_device(
    gpu_cfg: Dict[str, Any],
    torch_module: Optional[Any] = None,
) -> Any:
    """Return the torch device requested by gpu config."""
    torch_mod = torch if torch_module is None else torch_module
    if torch_mod is None:
        raise RuntimeError('PyTorch is not available')

    device_id = int(gpu_cfg.get('device_id', 0))
    cuda_available = bool(getattr(torch_mod, 'cuda', None) and torch_mod.cuda.is_available())
    if cuda_available:
        return torch_mod.device(f'cuda:{device_id}')

    if bool(gpu_cfg.get('allow_fallback_to_cpu', False)):
        return torch_mod.device('cpu')

    raise RuntimeError('CUDA is not available and gpu.allow_fallback_to_cpu=false')


def _ensure_import_path(path: str, path_list: Optional[List[str]] = None) -> None:
    """Prepend an absolute import path once."""
    paths = sys.path if path_list is None else path_list
    abs_path = os.path.abspath(path)
    if abs_path not in paths:
        paths.insert(0, abs_path)


def _add_batch_index_to_coordinates(
    coordinates: np.ndarray,
    batch_index: int = 0,
) -> np.ndarray:
    """Convert voxel coordinates from [z,y,x] to CenterPoint [batch,z,y,x]."""
    coords = np.asarray(coordinates)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(f'Expected voxel coordinates with shape (N, 3), got {coords.shape}')

    batch = np.full((coords.shape[0], 1), batch_index, dtype=coords.dtype)
    return np.hstack((batch, coords)).astype(coords.dtype, copy=False)


def _pointcloud2_to_numpy(msg) -> Optional[np.ndarray]:
    """sensor_msgs/PointCloud2 → numpy (N, 4+) [x, y, z, intensity, ...]。

    输出固定为 (N, 5): x, y, z, intensity, timestamp。
    - Ouster OS1: 使用 reflectivity 作为 intensity, t 作为点内时间。
    - RoboSense: 使用 intensity, timestamp 缺失时补 0。
    - 测试/离线模式可直接传入 numpy 数组。
    """
    if msg is None:
        return None

    if isinstance(msg, np.ndarray):
        return _normalize_points_array(msg)

    if not all(hasattr(msg, attr) for attr in ('fields', 'data', 'point_step')):
        return None

    try:
        fields = {field.name: field for field in msg.fields}
        if not all(name in fields for name in ('x', 'y', 'z')):
            logger.warning('PointCloud2 missing x/y/z fields: %s', sorted(fields))
            return None

        n_points = int(getattr(msg, 'width', 0)) * int(getattr(msg, 'height', 1))
        if n_points <= 0:
            return np.zeros((0, 5), dtype=np.float32)

        result = np.zeros((n_points, 5), dtype=np.float32)
        result[:, 0] = _point_field_array(msg, fields['x']).astype(np.float32)
        result[:, 1] = _point_field_array(msg, fields['y']).astype(np.float32)
        result[:, 2] = _point_field_array(msg, fields['z']).astype(np.float32)

        intensity_field = None
        if 'intensity' in fields:
            intensity_field = fields['intensity']
        elif 'reflectivity' in fields:
            intensity_field = fields['reflectivity']
        if intensity_field is not None:
            result[:, 3] = _point_field_array(msg, intensity_field).astype(np.float32)

        timestamp_field = None
        if 'timestamp' in fields:
            timestamp_field = fields['timestamp']
        elif 't' in fields:
            timestamp_field = fields['t']
        if timestamp_field is not None:
            result[:, 4] = _point_field_array(msg, timestamp_field).astype(np.float32)

        finite = np.isfinite(result[:, 0]) & np.isfinite(result[:, 1]) & np.isfinite(result[:, 2])
        return result[finite]
    except Exception as e:
        logger.warning('Failed to parse PointCloud2: %s', e)
        return None


def _normalize_points_array(points: np.ndarray) -> np.ndarray:
    """Normalize arbitrary point array to [x,y,z,intensity,timestamp]."""
    arr = np.asarray(points, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 3:
        return np.zeros((0, 5), dtype=np.float32)

    out = np.zeros((arr.shape[0], 5), dtype=np.float32)
    cols = min(arr.shape[1], 5)
    out[:, :cols] = arr[:, :cols]
    finite = np.isfinite(out[:, 0]) & np.isfinite(out[:, 1]) & np.isfinite(out[:, 2])
    return out[finite]


def _point_field_array(msg, field) -> np.ndarray:
    """Read one PointCloud2 field using the field offset and point_step."""
    endian = '>' if getattr(msg, 'is_bigendian', False) else '<'
    dtype_map = {
        1: np.dtype('i1'),          # INT8
        2: np.dtype('u1'),          # UINT8
        3: np.dtype(endian + 'i2'), # INT16
        4: np.dtype(endian + 'u2'), # UINT16
        5: np.dtype(endian + 'i4'), # INT32
        6: np.dtype(endian + 'u4'), # UINT32
        7: np.dtype(endian + 'f4'), # FLOAT32
        8: np.dtype(endian + 'f8'), # FLOAT64
    }
    dtype = dtype_map.get(field.datatype)
    if dtype is None:
        raise ValueError(f'Unsupported PointField datatype: {field.datatype}')

    width = int(getattr(msg, 'width', 0))
    height = int(getattr(msg, 'height', 1))
    row_step = int(getattr(msg, 'row_step', width * int(msg.point_step)))
    data = getattr(msg, 'data')
    if isinstance(data, np.ndarray):
        buffer = data.tobytes()
    else:
        buffer = bytes(data)

    return np.ndarray(
        shape=(height, width),
        dtype=dtype,
        buffer=buffer,
        offset=int(field.offset),
        strides=(row_step, int(msg.point_step)),
    ).reshape(-1)


def _image_msg_to_numpy(msg) -> Optional[Any]:
    """sensor_msgs/Image → numpy (H,W,C)。

    无 ROS1 返回 None。
    """
    if not ROS_AVAILABLE:
        return None
    try:
        from cv_bridge import CvBridge
        bridge = CvBridge()
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        return cv_img
    except Exception:
        pass

    # 降级: 手动 decode (需要 PIL/Pillow)
    try:
        import PIL.Image
        import io
        raw = bytes(msg.data)
        img = PIL.Image.open(io.BytesIO(raw))
        return np.array(img)
    except Exception:
        pass

    return None


def _resolve_msg_type(type_str: str) -> Optional[type]:
    """解析 ROS1 消息类型字符串 → Python class。"""
    if not ROS_AVAILABLE:
        return None
    mapping = {
        'sensor_msgs/Image': _import_or_none('sensor_msgs.msg', 'Image'),
        'sensor_msgs/PointCloud2': _import_or_none('sensor_msgs.msg', 'PointCloud2'),
        'diagnostic_msgs/DiagnosticArray': _import_or_none('diagnostic_msgs.msg', 'DiagnosticArray'),
    }

    if type_str in mapping:
        return mapping[type_str]

    # rtg_bev_msgs/* 类型
    if type_str.startswith('rtg_bev_msgs/'):
        msg_name = type_str.split('/')[-1]
        return _import_or_none('rtg_bev_msgs.msg', msg_name)

    return None


def _import_or_none(module: str, attr: str) -> Optional[type]:
    try:
        import importlib
        mod = importlib.import_module(module)
        return getattr(mod, attr, None)
    except ImportError:
        return None


def _make_set_param_srv():
    """返回 SetWarningParam service 类型，不可用时返回 None。"""
    if not ROS_AVAILABLE:
        return None
    try:
        from rtg_bev_msgs.srv import SetWarningParam
        return SetWarningParam
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# 消息构建辅助函数
# ---------------------------------------------------------------------------
def _build_detection_array(
    detections: List[Dict],
    timestamp: float,
) -> Any:
    """构建 DetectionArray ROS1 消息。"""
    from rtg_bev_msgs import Detection, DetectionArray, Header
    msg = DetectionArray()
    msg.header = Header()
    msg.header.stamp = _to_ros_time(timestamp)
    msg.header.frame_id = 'rtg_bev_origin'

    for i, det in enumerate(detections):
        d = Detection()
        d.id = i
        d.class_id = det.get('class_id', 0)
        d.class_name = det.get('class_name', '')
        d.confidence = det.get('confidence', 0.0)
        d.x = det.get('x', 0.0)
        d.y = det.get('y', 0.0)
        d.z = det.get('z', 0.0)
        d.w = det.get('w', 0.0)
        d.l = det.get('l', 0.0)
        d.h = det.get('h', 0.0)
        d.yaw = det.get('yaw', 0.0)
        d.vx = det.get('vx', 0.0)
        d.vy = det.get('vy', 0.0)
        d.distance = det.get('distance', 0.0)
        msg.detections.append(d)

    return msg


def _build_track_array(
    tracks: List[Dict],
    timestamp: float,
) -> Any:
    """构建 TrackArray ROS1 消息。"""
    from rtg_bev_msgs import Track, TrackArray, Header
    msg = TrackArray()
    msg.header = Header()
    msg.header.stamp = _to_ros_time(timestamp)
    msg.header.frame_id = 'rtg_bev_origin'

    for trk in tracks:
        t = Track()
        t.track_id = trk.get('track_id', -1)
        t.class_id = trk.get('class_id', 0)
        t.age = trk.get('age', 0)
        t.x = trk.get('x', 0.0)
        t.y = trk.get('y', 0.0)
        t.z = trk.get('z', 0.0)
        t.vx = trk.get('vx', 0.0)
        t.vy = trk.get('vy', 0.0)
        t.w = trk.get('w', 0.0)
        t.l = trk.get('l', 0.0)
        t.h = trk.get('h', 0.0)
        t.yaw = trk.get('yaw', 0.0)
        t.state = trk.get('state', 0)
        msg.tracks.append(t)

    return msg


def _build_warning_array(
    output: Dict[str, Any],
    timestamp: float,
) -> Any:
    """构建 WarningArray ROS1 消息。"""
    from rtg_bev_msgs import Warning, WarningArray, WarningZone, Header
    msg = WarningArray()
    msg.header = Header()
    msg.header.stamp = _to_ros_time(timestamp)
    msg.header.frame_id = 'rtg_bev_origin'
    msg.ego_motion_state = output.get('ego_motion', {}).get('state', 3)

    for w in output.get('warnings', []):
        wrn = Warning()
        wrn.track_id = w.get('track_id', -1)
        wrn.warning_level = w.get('warning_level', 0)
        wrn.target_class = w.get('target_class', 0)
        wrn.distance = w.get('distance', 0.0)
        wrn.zone = w.get('zone', '')
        wrn.trigger_reason = w.get('trigger_reason', '')
        wrn.trigger_time = _to_ros_time(w.get('trigger_time', timestamp))
        msg.warnings.append(wrn)

    for zone in output.get('active_zones', []):
        wz = WarningZone()
        wz.name = zone.get('name', '')
        wz.weight = zone.get('weight', 0.0)
        wz.y_min = zone.get('y_min', 0.0)
        wz.y_max = zone.get('y_max', 0.0)
        wz.description = zone.get('description', '')
        msg.active_zones.append(wz)

    return msg


def _build_ego_motion_state(ego_result: Dict[str, Any]) -> Any:
    """构建 EgoMotionState ROS1 消息。"""
    from rtg_bev_msgs import EgoMotionState as EgoMsg
    msg = EgoMsg()
    msg.state = ego_result.get('state', 3)
    msg.confidence = ego_result.get('confidence', 0.0)
    msg.displacement = ego_result.get('displacement', 0.0)
    msg.velocity_estimate = ego_result.get('velocity_estimate', 0.0)
    return msg


def _to_ros_time(timestamp: float) -> Any:
    """float timestamp → rospy.Time。"""
    if ROS_AVAILABLE and rospy is not None:
        return rospy.Time.from_sec(timestamp)
    return timestamp


# ==============================================================================
# Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description='RTG BEV Anti-Collision Node')
    parser.add_argument(
        '--config-dir',
        default=None,
        help='Path to config directory (default: PROJECT_ROOT/config)',
    )
    parser.add_argument(
        '--log-level',
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        help='Log level',
    )
    parser.add_argument(
        '--standalone',
        action='store_true',
        help='Run in standalone mode without ROS1',
    )
    args = parser.parse_args()

    # 日志设置
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%H:%M:%S',
    )

    # 配置目录
    config_dir = args.config_dir
    if config_dir is None:
        config_dir = os.path.join(_PROJECT_ROOT, 'config')

    # 创建并运行节点
    node = RTGBEVNode(config_dir=config_dir)
    try:
        node.run()
    except KeyboardInterrupt:
        logger.info('Interrupted by user')
    finally:
        node.shutdown()


if __name__ == '__main__':
    main()
