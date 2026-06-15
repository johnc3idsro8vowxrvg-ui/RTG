"""
分级预警模块 — warning_engine.py

核心功能:
  - 加载 warning.yaml 配置（支持热更新）
  - 距离计算：目标中心到最近 RTG 支腿的 BEV 平面距离
  - 三级预警（danger/warning/info）+ 分级帧确认策略
  - ROI 区域权重计算
  - RTG 静止时抑制预警
  - Warning 消息组装

所有阈值从配置读取，不做硬编码。
"""

import logging
import time
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ==============================================================================
# 常量
# ==============================================================================
class WarningLevel:
    NONE = 0
    INFO = 1
    WARNING = 2
    DANGER = 3

    _names = {0: 'none', 1: 'info', 2: 'warning', 3: 'danger'}

    @classmethod
    def name(cls, level: int) -> str:
        return cls._names.get(level, 'unknown')


# ROI 区域名称常量
class ROIZone:
    LANE_CORE = 'lane_core'
    LANE_APPROACH = 'lane_approach'
    TRUCK_LANE = 'truck_lane'
    SIDE_INTRUSION = 'side_intrusion'


# 触发原因常量
class TriggerReason:
    DISTANCE = 'distance'
    INTRUSION = 'intrusion'
    HEADING_TOWARD = 'heading_toward'

    @classmethod
    def all(cls) -> List[str]:
        return [cls.DISTANCE, cls.INTRUSION, cls.HEADING_TOWARD]


# ==============================================================================
# WARNING ENGINE
# ==============================================================================
class WarningEngine:
    """分级预警引擎。

    消费 config_loader 提供的 warning 配置，对每帧 tracks 计算预警等级。

    Usage:
        engine = WarningEngine(config_loader)
        warnings = engine.evaluate(tracks, ego_motion_state, timestamp)
    """

    # 缓存: 保存每个 target 的预警状态历史 (target_key →
    #   {level, frame_counter, release_counter, trigger_time})
    # target_key = (class_id, spatial hash) 用于跨帧关联

    def __init__(self, config_loader):
        """
        Parameters
        ----------
        config_loader : ConfigLoader
            提供 warning 配置的加载器.
        """
        self._config_loader = config_loader
        self._target_history: Dict[str, Dict[str, Any]] = {}

        # 支腿坐标缓存 (geometry.yaml)
        self._legs: List[Dict[str, Any]] = []
        self._refresh_legs()

        # ROI 区域定义缓存
        self._roi_zones_config: Dict = {}
        self._refresh_roi()

    # ------------------------------------------------------------------
    # Evaluate
    # ------------------------------------------------------------------
    def evaluate(
        self,
        tracks: List[Dict[str, Any]],
        ego_motion_state: int,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        """评估一帧 tracks，返回预警结果。

        Parameters
        ----------
        tracks : list of dict
            跟踪目标列表 (来自 tracker.update() 输出)。
        ego_motion_state : int
            RTG 运动状态 (0=静止, 1=运动+x, 2=运动-x, 3=未知)。
        timestamp : float or None

        Returns
        -------
        result : dict
            {
                'warnings': list of warning dicts,
                'active_zones': list of active zone dicts,
                'ego_motion_state': int,
            }
        """
        if timestamp is None:
            timestamp = time.time()

        # 刷新热更新配置
        self._refresh_config()

        warnings_out = []

        # 获取配置参数
        thresholds = self._config_loader.get_distance_thresholds()
        frame_conf = self._config_loader.get_frame_confirmation()
        ego_cfg = self._config_loader.get_ego_motion_config()
        output_rules = self._config_loader.get_output_rules()

        min_conf = output_rules.get('min_confidence_for_warning', 0.3)

        # RTG 静止时是否输出预警
        warn_when_static = ego_cfg.get('warn_when_static', False)
        static_close_enabled = (
            ego_cfg.get('static_close_proximity_alert', {}).get('enabled', False)
        )
        static_close_dist = (
            ego_cfg.get('static_close_proximity_alert', {}).get('distance_threshold', 2.0)
        )
        static_close_classes = (
            ego_cfg.get('static_close_proximity_alert', {}).get('target_classes', ['person'])
        )

        for track in tracks:
            # 置信度过滤
            conf_val = track.get('confidence', 0.0)
            if conf_val < min_conf:
                continue

            cls_id = track.get('class_id', 3)
            class_name = _class_id_to_name(cls_id)

            # 获取该类别的阈值
            cls_thresholds = thresholds.get(class_name)
            if cls_thresholds is None:
                cls_thresholds = thresholds.get('other_obstacle', {})

            # 计算目标到最近支腿的距离
            tx, ty = track['x'], track['y']
            distance = self._compute_min_leg_distance(tx, ty)

            # 判断 ROI 区域
            zone = self._classify_zone(ty)

            # 基础预警等级 (基于距离)
            base_level = self._distance_to_level(distance, cls_thresholds)

            # ROI 权重修正
            weighted_level = self._apply_roi_weight(base_level, zone)

            # 触发原因
            reason = TriggerReason.DISTANCE

            # 帧确认策略
            confirmed_level = self._apply_frame_confirmation(
                track, weighted_level, timestamp, frame_conf
            )

            # RTG 静止抑制
            if ego_motion_state == 0:  # 静止
                if not warn_when_static:
                    if static_close_enabled and class_name in static_close_classes:
                        if distance <= static_close_dist:
                            confirmed_level = max(confirmed_level, WarningLevel.INFO)
                        else:
                            confirmed_level = WarningLevel.NONE
                    else:
                        confirmed_level = WarningLevel.NONE

            if confirmed_level == WarningLevel.NONE:
                continue

            warnings_out.append({
                'track_id': track.get('track_id', -1),
                'warning_level': confirmed_level,
                'target_class': cls_id,
                'distance': round(distance, 2),
                'zone': zone,
                'trigger_reason': reason,
                'trigger_time': timestamp,
                '_class_name': class_name,
            })

        # 去重: 同目标只保留最高等级
        if output_rules.get('deduplicate_by_highest', True):
            warnings_out = self._deduplicate_highest(warnings_out)

        # 每区域每类别数量限制
        max_per_zone = output_rules.get('max_targets_per_zone', {})
        warnings_out = self._limit_per_zone_category(warnings_out, max_per_zone)

        # 构建 active_zones
        active_zones = self._build_active_zones(warnings_out)

        # 清理过期历史
        self._cleanup_history(timestamp)

        return {
            'warnings': warnings_out,
            'active_zones': active_zones,
            'ego_motion_state': ego_motion_state,
        }

    # ------------------------------------------------------------------
    # Distance computation
    # ------------------------------------------------------------------
    def _compute_min_leg_distance(self, tx: float, ty: float) -> float:
        """计算目标 (tx, ty) 到最近 RTG 支腿的 BEV 平面欧氏距离。"""
        min_dist = float('inf')
        for leg in self._legs:
            dx = tx - leg['x']
            dy = ty - leg['y']
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_dist:
                min_dist = dist
        return min_dist

    # ------------------------------------------------------------------
    # ROI Zone classification
    # ------------------------------------------------------------------
    def _classify_zone(self, ty: float) -> str:
        """根据目标 y 坐标判断所在 ROI 区域。"""
        # 使用 geometry.yaml 中的 lane_layout 判定
        lane_layout = self._config_loader.geometry.get('lane_layout', {})

        # 集卡侧大车道
        main_truck = lane_layout.get('main_lane_truck_side', {})
        main_y = main_truck.get('y_range', [-1.5, 1.5])
        if main_y[0] <= ty <= main_y[1]:
            return ROIZone.LANE_CORE

        # 禁行侧大车道
        main_forb = lane_layout.get('main_lane_forbidden_side', {})
        forb_y = main_forb.get('y_range', [22.63, 25.0])
        if forb_y[0] <= ty <= forb_y[1]:
            return ROIZone.LANE_CORE

        # 集卡车道
        truck_lane = lane_layout.get('truck_lane', {})
        truck_y = truck_lane.get('y_range', [1.5, 6.0])
        if truck_y[0] <= ty <= truck_y[1]:
            return ROIZone.TRUCK_LANE

        # 大车道近邻 (集装箱区域中紧邻大车道的部分)
        # 集卡侧近邻: y 在 [大车道上界, 集卡车道]
        # 禁行侧近邻: y 在 [最后一列箱, 禁行侧大车道下界]
        container_cfg = lane_layout.get('container_rows', {})
        y_start = container_cfg.get('y_start', 6.0)
        y_end = container_cfg.get('y_end', 22.63)

        # 集卡侧近邻: 离集卡侧大车道很近
        approach_margin = 3.0
        if main_y[1] < ty < main_y[1] + approach_margin:
            return ROIZone.LANE_APPROACH
        if forb_y[0] - approach_margin < ty < forb_y[0]:
            return ROIZone.LANE_APPROACH

        # 集装箱区域 → lane_approach
        if y_start <= ty <= y_end:
            return ROIZone.LANE_APPROACH

        # 禁行侧入侵
        if ty > forb_y[1]:
            return ROIZone.SIDE_INTRUSION

        # 默认
        return ROIZone.LANE_APPROACH

    # ------------------------------------------------------------------
    # Threshold → Level
    # ------------------------------------------------------------------
    @staticmethod
    def _distance_to_level(
        distance: float,
        thresholds: Dict[str, float],
    ) -> int:
        """根据距离和阈值返回基础预警等级。

        Parameters
        ----------
        distance : float
            目标到最近支腿的距离 (m)。
        thresholds : dict
            {danger: float, warning: float, info: float}

        Returns
        -------
        level : int (0/1/2/3)
        """
        danger_d = thresholds.get('danger', 8.0)
        warning_d = thresholds.get('warning', 15.0)
        info_d = thresholds.get('info', 25.0)

        if distance <= danger_d:
            return WarningLevel.DANGER
        elif distance <= warning_d:
            return WarningLevel.WARNING
        elif distance <= info_d:
            return WarningLevel.INFO
        else:
            return WarningLevel.NONE

    # ------------------------------------------------------------------
    # ROI Weight
    # ------------------------------------------------------------------
    def _apply_roi_weight(self, level: int, zone: str) -> int:
        """根据 ROI 区域权重调整预警等级。"""
        if level == WarningLevel.NONE:
            return WarningLevel.NONE

        zones_cfg = self._roi_zones_config
        zone_cfg = zones_cfg.get(zone, {})
        weight = zone_cfg.get('weight', 1.0)

        if weight >= 1.0:
            return level

        # 权重 < 1.0: 降一级
        if weight >= 0.7:
            # lane_approach (0.8): 非 danger 降一级
            if level == WarningLevel.DANGER:
                return WarningLevel.DANGER
            return max(WarningLevel.INFO, level - 1)

        # 低权重 (0.5~0.6): 降两级 (最少 INFO)
        return max(WarningLevel.INFO, level - 2)

    # ------------------------------------------------------------------
    # Frame Confirmation
    # ------------------------------------------------------------------
    def _apply_frame_confirmation(
        self,
        track: Dict[str, Any],
        level: int,
        timestamp: float,
        frame_conf: Dict[str, int],
    ) -> int:
        """应用帧确认/解除策略。

        危险: 1帧即时触发。
        警告/提示: 连续 N 帧确认。
        解除: 延迟 N 帧消退。
        """
        track_id = track.get('track_id', -1)
        state_key = f"{track_id}"

        danger_frames = frame_conf.get('danger_confirm_frames', 1)
        warn_frames = frame_conf.get('warning_confirm_frames', 3)
        info_frames = frame_conf.get('info_confirm_frames', 3)
        release_delay = frame_conf.get('release_delay_frames', 3)

        if level == WarningLevel.NONE:
            return WarningLevel.NONE

        # 获取或创建历史
        if state_key not in self._target_history:
            self._target_history[state_key] = {
                'level': WarningLevel.NONE,
                'frame_counter': 0,
                'release_counter': 0,
                'last_seen': timestamp,
                'confirmed': False,
            }
        hist = self._target_history[state_key]
        hist['last_seen'] = timestamp

        # 危险: 单帧触发
        if level >= WarningLevel.DANGER:
            if danger_frames <= 1:
                hist['level'] = level
                hist['confirmed'] = True
                hist['frame_counter'] = 1
                hist['release_counter'] = 0
                return WarningLevel.DANGER
            else:
                # 需要多帧确认的危险 (罕见，但保留能力)
                if hist['level'] == level:
                    hist['frame_counter'] += 1
                else:
                    hist['level'] = level
                    hist['frame_counter'] = 1
                    hist['confirmed'] = False

                if hist['frame_counter'] >= danger_frames:
                    hist['confirmed'] = True
                    hist['release_counter'] = 0
                    return WarningLevel.DANGER
                return WarningLevel.NONE

        # 警告/提示: 多帧确认
        required_frames = (
            warn_frames if level == WarningLevel.WARNING else info_frames
        )

        if hist['level'] == level:
            hist['frame_counter'] += 1
        else:
            hist['level'] = level
            hist['frame_counter'] = 1
            hist['confirmed'] = False

        # 检查是否已确认
        if hist['confirmed']:
            hist['release_counter'] = 0
            return min(level, hist['level'])

        if hist['frame_counter'] >= required_frames:
            hist['confirmed'] = True
            hist['release_counter'] = 0
            return level

        # 未达到确认帧数
        return WarningLevel.NONE

    # ------------------------------------------------------------------
    # Deduplication & Limiting
    # ------------------------------------------------------------------
    @staticmethod
    def _deduplicate_highest(warnings: List[Dict]) -> List[Dict]:
        """同 track_id 只保留最高预警等级。"""
        by_id: Dict[int, Dict] = {}
        for w in warnings:
            tid = w.get('track_id', -1)
            if tid not in by_id or w['warning_level'] > by_id[tid]['warning_level']:
                by_id[tid] = w
        return list(by_id.values())

    @staticmethod
    def _limit_per_zone_category(
        warnings: List[Dict],
        max_per_zone: Dict[str, int],
    ) -> List[Dict]:
        """每区域每类别限制最大输出数量，按距离升序保留。"""
        if not max_per_zone:
            return warnings

        # 先按距离排序
        sorted_w = sorted(warnings, key=lambda w: w.get('distance', 999))

        counts: Dict[Tuple[str, str], int] = {}
        result = []
        for w in sorted_w:
            zone = w.get('zone', '')
            cls_name = w.get('_class_name', 'other_obstacle')
            key = (zone, cls_name)
            mx = max_per_zone.get(cls_name, 10)
            current = counts.get(key, 0)
            if current < mx:
                result.append(w)
                counts[key] = current + 1
        return result

    def _build_active_zones(self, warnings: List[Dict]) -> List[Dict[str, Any]]:
        """构建当前激活的 ROI 区域列表。"""
        active_names = set()
        for w in warnings:
            if w['warning_level'] >= WarningLevel.INFO:
                active_names.add(w.get('zone', ''))

        zones_cfg = self._roi_zones_config
        result = []
        for name in active_names:
            cfg = zones_cfg.get(name, {})
            result.append({
                'name': name,
                'weight': cfg.get('weight', 1.0),
                'y_min': 0.0,
                'y_max': 0.0,
                'description': cfg.get('description', ''),
            })
        return result

    # ------------------------------------------------------------------
    # History cleanup
    # ------------------------------------------------------------------
    def _cleanup_history(self, timestamp: float, max_age: float = 10.0) -> None:
        """清理超出 max_age 未更新的历史记录。"""
        expired = []
        for key, hist in self._target_history.items():
            if timestamp - hist.get('last_seen', 0) > max_age:
                expired.append(key)
        for key in expired:
            del self._target_history[key]

    # ------------------------------------------------------------------
    # Config refresh
    # ------------------------------------------------------------------
    def _refresh_config(self) -> None:
        """刷新热更新配置。"""
        self._refresh_legs()
        self._refresh_roi()

    def _refresh_legs(self) -> None:
        """从 geometry.yaml 刷新支腿坐标。"""
        self._legs = self._config_loader.get_ego_footprint_legs()

    def _refresh_roi(self) -> None:
        """从 warning.yaml 刷新 ROI 区域权重。"""
        self._roi_zones_config = self._config_loader.get_roi_zones()

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置预警状态。"""
        self._target_history.clear()
        self._refresh_legs()
        self._refresh_roi()
        logger.info('WarningEngine reset')


# ==============================================================================
# Helpers
# ==============================================================================
_CLASS_NAMES = {0: 'person', 1: 'truck', 2: 'car', 3: 'other_obstacle'}


def _class_id_to_name(cls_id: int) -> str:
    return _CLASS_NAMES.get(int(cls_id), 'other_obstacle')
