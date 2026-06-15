"""Shared constants and enums for RTG BEV postprocessing."""


class WarningLevel:
    """预警等级。"""
    NONE = 0      # 无风险
    INFO = 1      # 提示/关注
    WARNING = 2   # 警告
    DANGER = 3    # 危险 (单帧即时触发)

    _names = {0: 'none', 1: 'info', 2: 'warning', 3: 'danger'}

    @classmethod
    def name(cls, level: int) -> str:
        return cls._names.get(level, 'unknown')


class ROIZone:
    """预警 ROI 区域。"""
    LANE_CORE = 'lane_core'             # 大车道核心
    LANE_APPROACH = 'lane_approach'     # 大车道近邻
    TRUCK_LANE = 'truck_lane'           # 集卡车道
    SIDE_INTRUSION = 'side_intrusion'   # 禁行侧入侵


class EgoMotionState:
    """RTG 运动状态。"""
    STATIC = 0         # 静止
    MOVING_PLUS_X = 1  # 沿 +x 方向运动
    MOVING_MINUS_X = 2 # 沿 -x 方向运动
    UNKNOWN = 3        # 无法判断

    _names = {0: 'static', 1: 'moving_+x', 2: 'moving_-x', 3: 'unknown'}

    @classmethod
    def name(cls, state: int) -> str:
        return cls._names.get(state, 'unknown')


class TrackerState:
    """短时目标跟踪状态。"""
    CANDIDATE = 0    # 待确认 (新出现)
    CONFIRMED = 1    # 已确认 (连续匹配 ≥3 帧)
    LOST = 2         # 丢失 (超时未匹配)
    DELETED = 3      # 已删除
