"""
分级预警模块 — warning_engine.py

核心设计: RTG 自车坐标驱动 (非固定车道)
  - 距离: 目标中心到最近 RTG 支腿的 BEV 平面距离
  - 方向: 运动方向扫掠走廊加权 (前方 1.0, 后方 0.5)
  - 静止: 默认不预警, 仅极近人员提示
  - site_layout: 可选场地语义, 用于降权/升权 (非硬依赖)

Usage:
    engine = WarningEngine(config_loader)
    result = engine.evaluate(tracks, ego_motion_state, timestamp)
"""

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import WarningLevel

logger = logging.getLogger(__name__)


# ==============================================================================
# WARNING ENGINE
# ==============================================================================
class WarningEngine:
    """RTG 自车坐标驱动的分级预警引擎。"""

    def __init__(self, config_loader):
        self._config_loader = config_loader
        self._target_history: Dict[str, Dict[str, Any]] = {}
        self._refresh_geometry()

    # ------------------------------------------------------------------
    # Evaluate (主入口)
    # ------------------------------------------------------------------
    def evaluate(
        self,
        tracks: List[Dict[str, Any]],
        ego_motion_state: int,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        if timestamp is None:
            timestamp = time.time()

        self._refresh_config()

        thresholds = self._config_loader.get_distance_thresholds()
        frame_conf = self._config_loader.get_frame_confirmation()
        ego_cfg = self._config_loader.get_ego_motion_config()
        static_policy = self._ego_geom.get('risk_model', {}).get('static_policy', {})
        dir_weight_cfg = self._ego_geom.get('risk_model', {}).get('motion_direction_weight', {})
        output_rules = self._config_loader.get_output_rules()
        min_conf = output_rules.get('min_confidence_for_warning', 0.3)

        warnings_out = []

        for track in tracks:
            if track.get('confidence', 0.0) < min_conf:
                continue

            cls_id = track.get('class_id', 3)
            class_name = _CLASS_NAMES.get(int(cls_id), 'other_obstacle')
            cls_thresholds = thresholds.get(class_name, thresholds.get('other_obstacle', {}))

            # 1) 距离
            tx, ty = track['x'], track['y']
            distance = self._min_leg_distance(tx, ty)

            # 2) 运动方向权重
            dir_weight = self._motion_direction_weight(
                tx, ty, ego_motion_state, dir_weight_cfg
            )

            # 3) 基础等级 (距离 → 等级)
            base_level = self._distance_to_level(distance, cls_thresholds)

            # 4) 运动方向修正
            weighted_level = self._apply_direction_weight(base_level, dir_weight)

            # 5) 场地语义修正 (可选 site_layout)
            weighted_level = self._apply_site_semantic(weighted_level, tx, ty, class_name)

            # 6) 帧确认
            confirmed_level = self._apply_frame_confirmation(
                track, weighted_level, timestamp, frame_conf
            )

            # 7) RTG 静止抑制
            confirmed_level = self._apply_static_policy(
                confirmed_level, ego_motion_state, distance, class_name, static_policy
            )

            if confirmed_level == WarningLevel.NONE:
                continue

            warnings_out.append({
                'track_id': track.get('track_id', -1),
                'warning_level': confirmed_level,
                'target_class': cls_id,
                'distance': round(distance, 2),
                'trigger_reason': 'distance',
                'trigger_time': timestamp,
                '_class_name': class_name,
            })

        # 后处理
        if output_rules.get('deduplicate_by_highest', True):
            warnings_out = self._deduplicate_highest(warnings_out)

        max_per_class = output_rules.get('max_targets_per_class', {})
        if max_per_class:
            warnings_out = self._limit_per_class(warnings_out, max_per_class)

        self._cleanup_history(timestamp)

        return {
            'warnings': warnings_out,
            'active_zones': [],
            'ego_motion_state': ego_motion_state,
        }

    # ==================================================================
    # 距离
    # ==================================================================
    def _min_leg_distance(self, tx: float, ty: float) -> float:
        min_d = float('inf')
        for leg in self._legs:
            d = np.hypot(tx - leg['x'], ty - leg['y'])
            if d < min_d:
                min_d = d
        return min_d

    # ==================================================================
    # 运动方向权重
    # ==================================================================
    def _motion_direction_weight(
        self, tx: float, ty: float, ego_state: int, cfg: Dict
    ) -> float:
        """计算目标相对于 RTG 运动方向的风险权重。

        ego_state: 0=STATIC, 1=MOVING_+X, 2=MOVING_-X, 3=UNKNOWN
        返回: 1.0 = 全风险, 0.5 = 半风险 (背向)
        """
        if ego_state in (0, 3):  # 静止或未知 → 等权
            return 1.0

        fwd = cfg.get('forward', 1.0)
        bwd = cfg.get('backward', 0.5)
        lat = cfg.get('lateral', 0.8)

        # 获取 RTG x 包络 (ego_geometry.truck_side_assembly)
        ego = self._ego_geom.get('ego_geometry', {})
        truck = ego.get('truck_side_assembly', {})
        x_min = truck.get('x_min', -12.5)
        x_max = truck.get('x_max', 0.5)

        # 判断目标在 RTG 包络外的前方还是后方
        # MOVING_+X: 包络前方 = tx > x_max, 包络后方 = tx < x_min
        # MOVING_-X: 包络前方 = tx < x_min, 包络后方 = tx > x_max
        if ego_state == 1:  # +x
            if tx > x_max:
                return fwd           # 前方
            elif tx < x_min:
                return bwd           # 后方
            else:
                return lat           # 在包络内部 → 侧向
        else:  # -x
            if tx < x_min:
                return fwd           # 前方
            elif tx > x_max:
                return bwd           # 后方
            else:
                return lat           # 在包络内部 → 侧向

    @staticmethod
    def _apply_direction_weight(level: int, weight: float) -> int:
        if level == WarningLevel.NONE:
            return WarningLevel.NONE
        if weight >= 1.0 or level == WarningLevel.DANGER:
            return level
        # 背向: 降一级
        return max(WarningLevel.INFO, level - 1)

    # ==================================================================
    # 场地语义 (可选 site_layout)
    # ==================================================================
    def _apply_site_semantic(
        self, level: int, tx: float, ty: float, class_name: str
    ) -> int:
        """用可选场地语义修正预警等级 (仅降权/升权, 不做硬决策)。"""
        if level == WarningLevel.NONE:
            return WarningLevel.NONE

        site = self._site_layout
        if not site or not site.get('enabled', False):
            return level

        # 集卡车道: 正常通行区, 降一级 (仅对 truck/car)
        truck_lane = site.get('truck_lane', {})
        truck_y = truck_lane.get('y_range')
        if truck_y and class_name in ('truck', 'car'):
            if truck_y[0] <= ty <= truck_y[1]:
                return max(WarningLevel.INFO, level - 1)

        # 禁行侧入侵: 升权 (保持原等级, 但 danger 不抑制)
        forb = site.get('main_lane_forbidden_side', {})
        forb_y = forb.get('y_range')
        if forb_y and ty < forb_y[0]:
            # 禁行侧外部 (比禁行侧大车道更远): 即使距离稍远也保持关注
            pass  # 不降权, 保持原等级

        return level

    # ==================================================================
    # 距离 → 等级
    # ==================================================================
    @staticmethod
    def _distance_to_level(distance: float, thresholds: Dict[str, float]) -> int:
        danger_d = thresholds.get('danger', 8.0)
        warning_d = thresholds.get('warning', 15.0)
        info_d = thresholds.get('info', 25.0)

        if distance <= danger_d:
            return WarningLevel.DANGER
        if distance <= warning_d:
            return WarningLevel.WARNING
        if distance <= info_d:
            return WarningLevel.INFO
        return WarningLevel.NONE

    # ==================================================================
    # RTG 静止抑制
    # ==================================================================
    @staticmethod
    def _apply_static_policy(
        level: int, ego_state: int, distance: float, class_name: str, cfg: Dict
    ) -> int:
        """RTG 静止时默认不预警, 仅极近人员提示。"""
        if ego_state != 0 or level == WarningLevel.NONE:
            return level

        warn_static = cfg.get('warn_when_static', False)
        if warn_static:
            return level

        close_enabled = cfg.get('close_proximity_enabled', True)
        close_dist = cfg.get('close_proximity_distance', 2.0)
        close_classes = cfg.get('close_proximity_classes', ['person'])

        if close_enabled and class_name in close_classes and distance <= close_dist:
            # RTG 静止时的极近提示: 仅 INFO 级别 (无碰撞风险, 仅提醒)
            return WarningLevel.INFO

        return WarningLevel.NONE

    # ==================================================================
    # 帧确认 (与之前一致)
    # ==================================================================
    def _apply_frame_confirmation(
        self, track: Dict[str, Any], level: int, timestamp: float,
        frame_conf: Dict[str, int],
    ) -> int:
        track_id = track.get('track_id', -1)
        key = str(track_id)

        danger_frames = frame_conf.get('danger_confirm_frames', 1)
        warn_frames = frame_conf.get('warning_confirm_frames', 3)
        release_delay = frame_conf.get('release_delay_frames', 3)

        if key not in self._target_history:
            self._target_history[key] = {
                'level': WarningLevel.NONE, 'frame_counter': 0,
                'release_counter': 0, 'last_seen': timestamp,
                'confirmed': False, 'confirmed_level': WarningLevel.NONE,
            }
        hist = self._target_history[key]
        hist['last_seen'] = timestamp

        # ---- 释放延迟: 当前 level=NONE 但之前已确认 ----
        if level == WarningLevel.NONE:
            if hist['confirmed']:
                hist['release_counter'] += 1
                if hist['release_counter'] < release_delay:
                    return hist['confirmed_level']
                # 超过释放帧数, 彻底解除
                hist['confirmed'] = False
                hist['confirmed_level'] = WarningLevel.NONE
                hist['release_counter'] = 0
            return WarningLevel.NONE

        hist['release_counter'] = 0

        # ---- Danger: 单帧触发 ----
        if level >= WarningLevel.DANGER:
            if danger_frames <= 1:
                hist['level'] = level
                hist['confirmed'] = True
                hist['confirmed_level'] = level
                hist['frame_counter'] = 1
                return WarningLevel.DANGER

        # ---- Warning/Info: 多帧确认 ----
        required = warn_frames if level == WarningLevel.WARNING else frame_conf.get('info_confirm_frames', 3)

        if hist['level'] == level:
            hist['frame_counter'] += 1
        else:
            # 等级变化: 如果是降级 (DANGER→WARNING), 保留已确认状态
            # 输出当前 (较低) 等级，不重置 confirmed= False
            was_confirmed = hist['confirmed']
            hist['level'] = level
            hist['frame_counter'] = 1
            if not was_confirmed:
                hist['confirmed'] = False

        if hist['frame_counter'] >= required:
            hist['confirmed'] = True
            hist['confirmed_level'] = level
            return level

        # 已确认但当前帧等级不同 (降级过渡): 输出较低等级, 同步更新 confirmed_level
        # 避免 release hold 阶段复活旧的高等级
        if hist['confirmed']:
            out = min(level, hist['confirmed_level'])
            hist['confirmed_level'] = out
            return out

        return WarningLevel.NONE

    # ==================================================================
    # Config / Geometry refresh
    # ==================================================================
    def _refresh_config(self) -> None:
        self._refresh_geometry()

    def _refresh_geometry(self) -> None:
        geom = self._config_loader.geometry
        ego = geom.get('ego_geometry', {})

        # 支腿坐标 (兼容旧 ego_footprint 和新的 ego_geometry.legs)
        legs = ego.get('legs', [])
        if not legs:
            # fallback: 从旧的 ego_footprint 或 rtg_dimensions 构建
            fp = geom.get('ego_footprint', {})
            dim = geom.get('rtg_dimensions', {})
            span = ego.get('span_y', dim.get('span_width', 23.5))
            wb = ego.get('wheelbase_x', dim.get('wheelbase', 12.0))
            truck_y = ego.get('truck_side_y', 0.0)
            forb_y = ego.get('forbidden_side_y', -span)
            legs = [
                {'x': 0.0, 'y': truck_y},
                {'x': -wb, 'y': truck_y},
                {'x': 0.0, 'y': forb_y},
                {'x': -wb, 'y': forb_y},
            ]
        self._legs = legs

        # site_layout (可选)
        self._site_layout = geom.get('site_layout', {})

        # risk_model (合并 geometry + warning 配置)
        self._ego_geom = geom

    # ==================================================================
    # Helpers
    # ==================================================================
    @staticmethod
    def _deduplicate_highest(warnings: List[Dict]) -> List[Dict]:
        by_id: Dict[int, Dict] = {}
        for w in warnings:
            tid = w.get('track_id', -1)
            if tid not in by_id or w['warning_level'] > by_id[tid]['warning_level']:
                by_id[tid] = w
        return list(by_id.values())

    @staticmethod
    def _limit_per_class(warnings: List[Dict], max_per_class: Dict) -> List[Dict]:
        """按类别限制最大输出数量，按距离升序保留最近的目标。"""
        # 按类别分组
        by_class: Dict[str, List[Dict]] = {}
        for w in warnings:
            cls = w.get('_class_name', 'other_obstacle')
            if cls not in by_class:
                by_class[cls] = []
            by_class[cls].append(w)

        result = []
        for cls, ws in by_class.items():
            limit = max_per_class.get(cls, max_per_class.get('other_obstacle', 5))
            # 按距离升序，保留最近的 limit 个
            ws.sort(key=lambda w: w.get('distance', 999))
            result.extend(ws[:limit])
        return result

    def _cleanup_history(self, timestamp: float, max_age: float = 10.0) -> None:
        expired = [k for k, h in self._target_history.items()
                   if timestamp - h.get('last_seen', 0) > max_age]
        for k in expired:
            del self._target_history[k]

    def reset(self) -> None:
        self._target_history.clear()
        self._refresh_geometry()
        logger.info('WarningEngine reset')


# ==============================================================================
# Helpers
# ==============================================================================
_CLASS_NAMES = {0: 'person', 1: 'truck', 2: 'car', 3: 'other_obstacle'}
