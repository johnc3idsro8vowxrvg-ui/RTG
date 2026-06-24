"""
短时目标跟踪模块 — tracker.py

基于 IoU + Kalman Filter 的轻量跟踪器：
  - 状态机: candidate → confirmed → lost → deleted
  - 内部稳定 ID
  - track_id 管理、年龄(age)计数
  - 速度估计（由连续帧位置差分）
  - 所有阈值从配置读取

依赖: numpy, filterpy (可选, 用于 KalmanFilter; 不安装时使用纯 numpy 降级)
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入 filterpy，不可用时降级为纯 numpy 线性运动模型
try:
    from filterpy.kalman import KalmanFilter as _FKF
    HAS_FILTERPY = True
except ImportError:
    HAS_FILTERPY = False
    logger.info('filterpy not installed; using simple motion-model fallback')


from .constants import TrackerState as _StateConst

_STATE_NAMES = {0: 'candidate', 1: 'confirmed', 2: 'lost', 3: 'deleted'}

# Kalman 状态向量: [x, y, z, w, l, h, yaw, vx, vy, vz]
_DIM_X = 10
_DIM_Z = 7  # 观测: [x, y, z, w, l, h, yaw]


# ==============================================================================
# KALMAN BOX TRACKER (filterpy wrapper)
# ==============================================================================
class KalmanBoxTracker:
    """基于 Kalman Filter 的单个目标跟踪状态。

    使用恒定速度模型（CV），状态向量包含位置、尺寸、朝向和速度。
    """

    def __init__(
        self,
        bbox: np.ndarray,
        track_id: int,
        init_velocity: Tuple[float, float] = (0.0, 0.0),
        init_confidence: float = 0.0,
    ):
        """
        Parameters
        ----------
        bbox : np.ndarray (7,)  [x, y, z, w, l, h, yaw]
        track_id : int
        init_velocity : (vx, vy)
        init_confidence : float
        """
        self.track_id = track_id
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = _StateConst.CANDIDATE
        self.class_id = 0

        self.kf = self._make_kf() if HAS_FILTERPY else None
        self._x_hat = self._init_state(bbox, init_velocity)

        # 置信度历史 (最近 N 帧检测置信度)
        self._confidence_history: List[float] = [init_confidence] if init_confidence > 0 else []

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------
    def predict_with_dt(self, dt: float) -> np.ndarray:
        """使用指定时间步长预测。"""
        self.age += 1
        self.time_since_update += 1

        dt = max(dt, 1e-4)
        if self.kf is not None:
            # 按实际 dt 更新状态转移矩阵中的速度项
            self.kf.F[0, 7] = dt
            self.kf.F[1, 8] = dt
            self.kf.F[2, 9] = dt
            self.kf.predict()
            self._x_hat = self.kf.x.copy()
        else:
            # 降级: 恒定速度外推
            self._x_hat[0] += self._x_hat[7] * dt
            self._x_hat[1] += self._x_hat[8] * dt
            self._x_hat[2] += self._x_hat[9] * dt
        return self._x_hat

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self, bbox: np.ndarray, class_id: int = 0,
               confidence: float = 0.0, min_hits_confirm: int = 3) -> None:
        """用观测更新卡尔曼状态。

        Parameters
        ----------
        bbox : np.ndarray (7,)  [x, y, z, w, l, h, yaw]
        class_id : int
        confidence : float
            当前帧检测置信度 (0~1)
        min_hits_confirm : int
            Hits required to promote a candidate track to confirmed.
        """
        self.hits += 1
        self.time_since_update = 0
        self.class_id = class_id

        # 保存置信度历史 (最近 10 帧)
        if confidence > 0:
            self._confidence_history.append(confidence)
            if len(self._confidence_history) > 10:
                self._confidence_history.pop(0)

        if self.kf is not None:
            self.kf.update(bbox.reshape(-1, 1))
            self._x_hat = self.kf.x.copy()
        else:
            # 降级: 简单指数平滑 + 有限差速度估计
            # 注意: 必须在覆盖位置前保存旧值，否则速度永远≈0
            old_x, old_y = self._x_hat[0], self._x_hat[1]
            alpha = 0.7
            self._x_hat[:7] = alpha * bbox + (1 - alpha) * self._x_hat[:7]
            dt_v = max(self.time_since_update or 1, 1)
            self._x_hat[7] = (bbox[0] - old_x) / dt_v
            self._x_hat[8] = (bbox[1] - old_y) / dt_v

        # 状态机晋升
        if self.state == _StateConst.CANDIDATE and self.hits >= min_hits_confirm:
            self.state = _StateConst.CONFIRMED

    def mark_missed(self) -> None:
        """标记本帧未匹配。"""
        if self.time_since_update > 5 and self.state == _StateConst.CONFIRMED:
            self.state = _StateConst.LOST

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------
    def get_state(self) -> np.ndarray:
        """返回当前估计状态 [x,y,z,w,l,h,yaw,vx,vy,vz] (10,)。"""
        return self._x_hat.copy()

    def get_position(self) -> np.ndarray:
        """返回当前估计位置 [x, y, z] (3,)。"""
        return self._x_hat[:3].copy()

    def get_bbox(self) -> np.ndarray:
        """返回当前估计 bbox [x, y, z, w, l, h, yaw] (7,)。"""
        return self._x_hat[:7].copy()

    def get_velocity(self) -> np.ndarray:
        """返回估计速度 [vx, vy] (2,)。"""
        return self._x_hat[7:9].copy()

    def get_confidence(self) -> float:
        """返回平均检测置信度。

        如果没有置信度历史，返回 0.0。
        """
        if not self._confidence_history:
            return 0.0
        return float(np.mean(self._confidence_history))

    def is_deleted(self) -> bool:
        return self.state == _StateConst.DELETED

    def mark_deleted(self) -> None:
        self.state = _StateConst.DELETED

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    @staticmethod
    def _init_state(bbox: np.ndarray, init_velocity: Tuple[float, float]) -> np.ndarray:
        x_hat = np.zeros(_DIM_X, dtype=np.float64)
        x_hat[:7] = bbox
        x_hat[7] = init_velocity[0]
        x_hat[8] = init_velocity[1]
        return x_hat

    @staticmethod
    def _make_kf():
        """构造恒定速度 Kalman Filter (状态10维, 观测7维)。"""
        kf = _FKF(dim_x=_DIM_X, dim_z=_DIM_Z)

        # 状态转移矩阵 (dt 在 predict_with_dt 中按实际帧间隔更新)
        kf.F = np.eye(_DIM_X)

        # 观测矩阵: 只观测 [x,y,z,w,l,h,yaw]
        kf.H = np.zeros((_DIM_Z, _DIM_X))
        for i in range(_DIM_Z):
            kf.H[i, i] = 1.0

        # 过程噪声 (位置噪声小, 速度噪声较大)
        kf.Q = np.eye(_DIM_X) * 0.01
        kf.Q[7, 7] = 0.1
        kf.Q[8, 8] = 0.1
        kf.Q[9, 9] = 0.01

        # 观测噪声
        kf.R = np.eye(_DIM_Z) * 0.1
        kf.R[0, 0] = 0.2   # x
        kf.R[1, 1] = 0.2   # y
        kf.R[2, 2] = 0.05  # z
        kf.R[3, 3] = 0.1   # w
        kf.R[4, 4] = 0.1   # l
        kf.R[5, 5] = 0.05  # h
        kf.R[6, 6] = 0.05  # yaw

        # 初始协方差
        kf.P = np.eye(_DIM_X) * 1.0
        kf.P[7, 7] = 10.0
        kf.P[8, 8] = 10.0
        kf.P[9, 9] = 10.0

        return kf


# ==============================================================================
# IoU UTILITIES
# ==============================================================================
def bbox_bev_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """计算两个 BEV 框的 2D IoU。

    将 3D 框投影到 BEV 平面 (忽略 z 和 h)，使用旋转框 polygon 计算。

    Parameters
    ----------
    box1, box2 : np.ndarray (7,) [x, y, z, w, l, h, yaw]
    """
    corners1 = _bbox_bev_corners(box1)
    corners2 = _bbox_bev_corners(box2)
    return _polygon_iou(corners1, corners2)


def _bbox_bev_corners(box: np.ndarray) -> np.ndarray:
    """将 [x, y, z, w, l, h, yaw] 转为 BEV 平面 4 个角点 (4×2)。"""
    x, y = float(box[0]), float(box[1])
    w, l = float(box[3]) / 2.0, float(box[4]) / 2.0
    yaw = float(box[6])

    cos_yaw = float(np.cos(yaw))
    sin_yaw = float(np.sin(yaw))

    # 角偏移 (物体坐标系)
    corners_local = np.array([
        [-l, -w],
        [l, -w],
        [l, w],
        [-l, w],
    ], dtype=np.float64)

    # 旋转并平移
    rot = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    corners = corners_local @ rot.T + np.array([x, y], dtype=np.float64)
    return corners


def _polygon_iou(poly1: np.ndarray, poly2: np.ndarray) -> float:
    """计算两个凸多边形的 IoU (使用 Sutherland-Hodgman 裁剪)。"""
    inter = _polygon_intersection_area(poly1, poly2)
    area1 = _polygon_area(poly1)
    area2 = _polygon_area(poly2)
    union = area1 + area2 - inter
    if union < 1e-12:
        return 0.0
    return inter / union


def _polygon_area(poly: np.ndarray) -> float:
    """Shoelace 公式计算多边形面积。"""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def _polygon_intersection_area(subject: np.ndarray, clip: np.ndarray) -> float:
    """Sutherland-Hodgman 多边形裁剪相交面积。"""
    output = list(subject)
    clip_edges = [(clip[i], clip[(i + 1) % len(clip)]) for i in range(len(clip))]
    for edge_start, edge_end in clip_edges:
        if not output:
            break
        input_list = output
        output = []
        for i, current in enumerate(input_list):
            prev = input_list[i - 1]
            inside_current = _is_inside(current, edge_start, edge_end)
            inside_prev = _is_inside(prev, edge_start, edge_end)

            if inside_current:
                if not inside_prev:
                    output.append(_line_intersection(prev, current, edge_start, edge_end))
                output.append(current)
            elif inside_prev:
                output.append(_line_intersection(prev, current, edge_start, edge_end))

    return _polygon_area(np.array(output)) if len(output) >= 3 else 0.0


def _is_inside(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> bool:
    """判断点 p 是否在直线 ab 的左侧 (给定逆时针多边形)。"""
    return np.cross(b - a, p - a) >= -1e-12


def _line_intersection(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """两条线段 (p1p2, p3p4) 的交点。"""
    d1 = np.cross(p2 - p1, p3 - p1)
    d2 = np.cross(p2 - p1, p4 - p1)
    if abs(d1 - d2) < 1e-12:
        return p3
    t = d1 / (d1 - d2)
    return p3 + t * (p4 - p3)


# ==============================================================================
# TRACKER
# ==============================================================================
class Tracker:
    """基于 IoU + Kalman Filter 的轻量跟踪器。

    状态机:
      candidate (hits<3) → confirmed (hits>=3) → lost (unmatched>5) → deleted

    Usage:
        tracker = Tracker(config)
        tracks = tracker.update(detections, timestamp)
    """

    # 默认参数 (可被 config 覆盖)
    DEFAULT_CONFIG = {
        'iou_threshold': 0.1,
        'max_age_lost': 5,         # lost 状态最大存活帧数
        'min_hits_confirm': 3,     # 确认所需的最小命中数
        'min_confidence': 0.2,     # 最低检测置信度
        'publish_internal_tracks': False,  # 第一版不对外发布 track_id
    }

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Parameters
        ----------
        config : dict or None
            跟踪器配置。为 None 时使用 DEFAULT_CONFIG。
        """
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self._tracks: Dict[int, KalmanBoxTracker] = {}
        self._next_id = 0
        self._prev_timestamp: Optional[float] = None

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(
        self,
        detections: List[Dict[str, Any]],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """对一帧检测结果执行跟踪。

        Parameters
        ----------
        detections : list of dict
            每个 dict: {x, y, z, w, l, h, yaw, class_id, confidence, ...}
        timestamp : float
            当前帧时间戳 (秒, 用于 dt 计算)

        Returns
        -------
        tracks : list of dict
            每个 dict: {track_id, class_id, age, x, y, z, vx, vy, w, l, h, yaw, state}
        """
        dt = 0.05
        if self._prev_timestamp is not None:
            dt = max(timestamp - self._prev_timestamp, 0.01)
        self._prev_timestamp = timestamp

        # 置信度过滤
        min_conf = self.config.get('min_confidence', 0.2)
        dets = [d for d in detections if d.get('confidence', 0.0) >= min_conf]

        det_boxes = self._detections_to_boxes(dets)
        num_det = len(det_boxes)

        # 1) 为所有活跃轨迹做 predict
        for trk in list(self._tracks.values()):
            trk.predict_with_dt(dt)

        # 2) 计算 IoU 成本矩阵
        cost = self._iou_cost_matrix(self._tracks, det_boxes, dets)

        # 3) 贪婪匹配
        iou_thresh = self.config.get('iou_threshold', 0.1)
        min_hits = self.config.get('min_hits_confirm', 3)
        matches, unmatched_tracks, unmatched_dets = self._associate(
            cost, iou_thresh
        )

        # 4) 更新匹配上的轨迹
        for trk_idx, det_idx in matches:
            trk = list(self._tracks.values())[trk_idx]
            bbox = det_boxes[det_idx]
            cls_id = dets[det_idx].get('class_id', 0)
            conf = dets[det_idx].get('confidence', 0.0)
            trk.update(
                bbox, class_id=cls_id, confidence=conf, min_hits_confirm=min_hits
            )

        # 5) 标记未匹配的轨迹
        for trk_idx in unmatched_tracks:
            trk = list(self._tracks.values())[trk_idx]
            trk.mark_missed()

        # 6) 为未匹配的检测创建新轨迹
        for det_idx in unmatched_dets:
            bbox = det_boxes[det_idx]
            cls_id = dets[det_idx].get('class_id', 0)
            conf = dets[det_idx].get('confidence', 0.0)
            new_trk = KalmanBoxTracker(bbox, self._next_id, init_confidence=conf)
            new_trk.class_id = cls_id
            new_trk.state = _StateConst.CANDIDATE
            if new_trk.hits >= min_hits:
                new_trk.state = _StateConst.CONFIRMED
            self._tracks[self._next_id] = new_trk
            self._next_id += 1

        # 7) 删除长期丢失的轨迹
        max_age = self.config.get('max_age_lost', 5)
        deleted_ids = []
        for tid, trk in self._tracks.items():
            if trk.time_since_update > max_age:
                trk.mark_deleted()
                deleted_ids.append(tid)
        for tid in deleted_ids:
            del self._tracks[tid]

        # 8) 构建输出，应用 min_hits 过滤
        publish_internal = self.config.get('publish_internal_tracks', False)

        tracks_out = []
        for trk in self._tracks.values():
            if trk.hits < min_hits and trk.state == _StateConst.CANDIDATE:
                # 首版: 内部跟踪但暂不对外输出 candidate
                if not publish_internal:
                    continue

            state = trk.state
            pos = trk.get_position()
            vel = trk.get_velocity()
            bbox = trk.get_bbox()

            # 第一版 track_id 对内不对外
            track_id = -1 if (state != _StateConst.CONFIRMED and not publish_internal) else trk.track_id

            tracks_out.append({
                'track_id': track_id,
                'class_id': trk.class_id,
                'age': trk.age,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'z': float(pos[2]),
                'vx': float(vel[0]),
                'vy': float(vel[1]),
                'w': float(bbox[3]),
                'l': float(bbox[4]),
                'h': float(bbox[5]),
                'yaw': float(bbox[6]),
                'state': state,
                'state_name': _STATE_NAMES.get(state, 'unknown'),
                'hits': trk.hits,
                'time_since_update': trk.time_since_update,
                'confidence': trk.get_confidence(),
            })

        logger.debug(
            'Tracker: %d dets → %d tracks (active=%d, dt=%.3f)',
            num_det, len(tracks_out), len(self._tracks), dt,
        )
        return tracks_out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _detections_to_boxes(self, detections: List[Dict]) -> List[np.ndarray]:
        """将检测字典列表转为 numpy bbox 数组列表。"""
        boxes = []
        for d in detections:
            box = np.array([
                d.get('x', 0.0), d.get('y', 0.0), d.get('z', 0.0),
                d.get('w', 0.0), d.get('l', 0.0), d.get('h', 0.0),
                d.get('yaw', 0.0),
            ], dtype=np.float64)
            boxes.append(box)
        return boxes

    def _iou_cost_matrix(
        self,
        tracks: Dict[int, KalmanBoxTracker],
        det_boxes: List[np.ndarray],
        detections: List[Dict],
    ) -> np.ndarray:
        """Calculate IoU cost matrix, disallowing cross-class matches."""
        num_tracks = len(tracks)
        num_dets = len(det_boxes)
        if num_tracks == 0 or num_dets == 0:
            return np.zeros((num_tracks, num_dets))

        cost = np.zeros((num_tracks, num_dets))
        for i, trk in enumerate(tracks.values()):
            pred_box = trk.get_bbox()
            for j, det_box in enumerate(det_boxes):
                det_class = int(detections[j].get('class_id', trk.class_id))
                if int(trk.class_id) != det_class:
                    cost[i, j] = np.inf
                    continue
                iou = bbox_bev_iou(pred_box, det_box)
                cost[i, j] = 1.0 - iou
        return cost

    @staticmethod
    def _associate(
        cost: np.ndarray,
        threshold: float,
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """基于成本矩阵的贪婪匹配。

        Returns
        -------
        matches : list of (track_idx, det_idx)
        unmatched_tracks : list of track indices
        unmatched_dets : list of det indices
        """
        if cost.size == 0:
            return (
                [],
                list(range(cost.shape[0])),
                list(range(cost.shape[1])),
            )

        # 按 (行,列) 展平并排序
        rows, cols = cost.shape
        indices = np.argsort(cost, axis=None)

        matched_rows = set()
        matched_cols = set()
        matches = []

        for idx in indices:
            r = int(idx // cols)
            c = int(idx % cols)
            if r in matched_rows or c in matched_cols:
                continue
            if cost[r, c] > (1.0 - threshold):
                # 成本太高，停止匹配
                break
            matches.append((r, c))
            matched_rows.add(r)
            matched_cols.add(c)

        unmatched_rows = [r for r in range(rows) if r not in matched_rows]
        unmatched_cols = [c for c in range(cols) if c not in matched_cols]

        return matches, unmatched_rows, unmatched_cols

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置跟踪器状态。"""
        self._tracks.clear()
        self._next_id = 0
        self._prev_timestamp = None
        logger.info('Tracker reset')
