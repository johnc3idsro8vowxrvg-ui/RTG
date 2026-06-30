"""
跟踪可视化工具模块 — _draw_tracking.py

提供可复用的跟踪结果可视化函数:
  - draw_track_box:        单个跟踪框（不同状态不同线型）
  - draw_track_trajectories: BEV 轨迹线 + track_id 标注
  - draw_track_timeline:    时间线图（位置/速度/状态随时间变化）
  - draw_multiframe_bev:    多帧 BEV 序列对比
  - create_track_summary:   生成统计字典

纯 matplotlib + numpy 实现，无 ROS1 依赖。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection

logger = logging.getLogger(__name__)

# ==============================================================================
# 颜色方案 — 与现有 _visualize_bev.py 保持一致
# ==============================================================================
CLASS_COLORS = {
    'person':          '#FF3333',  # 红
    'truck':           '#3366FF',  # 蓝
    'car':             '#33CC33',  # 绿
    'other_obstacle':  '#FFAA00',  # 橙
}

CLASS_NAMES_CN = {
    'person':          '人员',
    'truck':           '集卡',
    'car':             '乘用车',
    'other_obstacle':  '其他障碍物',
}

STATE_COLORS = {
    0: '#999999',  # candidate — 灰
    1: '#33CC33',  # confirmed — 绿
    2: '#FF9933',  # lost — 橙
    3: '#FF3333',  # deleted — 红
}

STATE_LINESTYLES = {
    0: '--',       # candidate — 虚线
    1: '-',        # confirmed — 实线
    2: ':',        # lost — 点线
    3: (0, (3, 5)),  # deleted — 长虚线
}

STATE_NAMES = {0: 'candidate', 1: 'confirmed', 2: 'lost', 3: 'deleted'}
STATE_NAMES_CN = {0: '候选', 1: '确认', 2: '丢失', 3: '删除'}

# 轨迹历史颜色（透明渐变）
TRAIL_ALPHA_START = 0.15
TRAIL_ALPHA_END = 0.85


# ==============================================================================
# BEV 场景布局 (复用现有布局)
# ==============================================================================
def draw_bev_scene_background(
    ax: plt.Axes,
    x_lim: Tuple[float, float] = (-50, 50),
    y_lim: Tuple[float, float] = (-30, 15),
    draw_rtg: bool = True,
    draw_lanes: bool = True,
    draw_labels: bool = True,
) -> None:
    """绘制 BEV 场景背景（大车道、集卡车道、集装箱区、RTG footprint）。

    坐标系: x=大车道方向, y=跨距方向(禁行侧→集卡侧)
    """
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)

    if draw_lanes:
        # 集卡侧大车道 (y≈0)
        ax.axhspan(-1.5, 1.5, alpha=0.06, color='gray', zorder=0)
        if draw_labels:
            ax.text(x_lim[0] + 2, 0, '大车道\n(集卡侧)', fontsize=6,
                    va='center', color='#666666', alpha=0.8)

        # 集卡车道 (geometry.yaml: y∈[-6,-1.5])
        ax.axhspan(-6.0, -1.5, alpha=0.04, color='blue', zorder=0)
        if draw_labels:
            ax.text(x_lim[0] + 2, -3.75, '集卡车道', fontsize=6,
                    va='center', color='#4477AA', alpha=0.8)

        # 禁行侧大车道 (geometry.yaml: y≈-23.5)
        ax.axhspan(-25.6, -22.6, alpha=0.06, color='gray', zorder=0)
        if draw_labels:
            ax.text(x_lim[0] + 2, -23.8, '大车道\n(禁行侧)', fontsize=6,
                    va='center', color='#666666', alpha=0.8)

        # 集装箱区 (geometry.yaml: y∈[-22.63,-6])
        for row_y in np.arange(-22.6, -6.0, 2.6):
            ax.axhline(row_y, color='#8B4513', linewidth=0.25, linestyle='--', alpha=0.3, zorder=0)
        if draw_labels:
            ax.text(x_lim[0] + 2, -14, '集装箱区', fontsize=6,
                    va='center', color='#8B4513', alpha=0.8)

        # 相邻箱区/中间车道 (+y)
        ax.axhspan(1.5, 7.0, alpha=0.03, color='cyan', zorder=0)

    if draw_rtg:
        # 集卡侧支腿
        ax.add_patch(Rectangle(
            (-12.5, -0.5), 13.0, 1.0,
            facecolor='black', alpha=0.30, edgecolor='#333333',
            linewidth=0.6, zorder=2,
        ))
        if draw_labels:
            ax.text(-6, -1.3, 'RTG 集卡侧', fontsize=5, ha='center', color='black', alpha=0.6)

        # 禁行侧支腿
        ax.add_patch(Rectangle(
            (-12.5, -24.0), 13.0, 1.0,
            facecolor='black', alpha=0.30, edgecolor='#333333',
            linewidth=0.6, zorder=2,
        ))
        if draw_labels:
            ax.text(-6, -24.8, 'RTG 禁行侧', fontsize=5, ha='center', color='black', alpha=0.6)

    # 传感器位置标记
    ax.plot(0, 0, 'r*', markersize=8, markeredgecolor='darkred', markeredgewidth=0.5, zorder=5)
    ax.plot(-12.0, 0, 'y*', markersize=8, markeredgecolor='darkorange', markeredgewidth=0.5, zorder=5)
    if draw_labels:
        ax.annotate('L1', (0.5, -0.5), fontsize=5, color='red', alpha=0.7)
        ax.annotate('L2', (-13.0, 0.5), fontsize=5, color='darkorange', alpha=0.7)


# ==============================================================================
# 跟踪框绘制
# ==============================================================================
def draw_track_box(
    ax: plt.Axes,
    track: Dict[str, Any],
    color: Optional[str] = None,
    show_label: bool = True,
    show_velocity: bool = True,
    alpha: float = 0.7,
    linewidth: float = 1.5,
) -> None:
    """在 BEV 平面上绘制单个跟踪框。

    Parameters
    ----------
    ax : matplotlib Axes
    track : dict
        跟踪目标 dict (来自 tracker.update() 输出)。
        包含: x, y, w, l, yaw, track_id, class_id, state, state_name, [vx, vy]
    color : str or None
        框颜色，None 时根据 class_id 自动选择。
    show_label : bool
        是否显示 track_id 和类别标签。
    show_velocity : bool
        是否绘制速度箭头。
    alpha : float
        透明度。
    linewidth : float
        线宽。
    """
    cx, cy = track['x'], track['y']
    w, l, yaw = track.get('w', 2.0), track.get('l', 4.0), track.get('yaw', 0.0)
    state = track.get('state', 1)
    cls_id = track.get('class_id', 3)

    # 颜色
    if color is None:
        cls_name = _class_id_to_name(cls_id)
        color = CLASS_COLORS.get(cls_name, '#888888')

    # 线型（按状态）
    ls = STATE_LINESTYLES.get(state, '-')
    lw = linewidth
    if state == 1:  # confirmed = 加粗
        lw = linewidth + 0.8

    # 计算旋转框角点
    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    corners_local = np.array([
        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
    ])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners_local @ rot.T + np.array([cx, cy])

    # 填充（按状态调整透明度）
    fill_alpha = alpha * 0.25
    if state == 1:
        fill_alpha = alpha * 0.35
    elif state in (2, 3):
        fill_alpha = alpha * 0.12

    ax.fill(corners[:, 0], corners[:, 1],
            alpha=fill_alpha, facecolor=color,
            edgecolor=color, linewidth=lw, linestyle=ls, zorder=3)

    # 朝向箭头
    arrow_len = l * 0.35
    dx = np.cos(yaw) * arrow_len
    dy = np.sin(yaw) * arrow_len
    ax.arrow(cx, cy, dx, dy,
             head_width=0.4, head_length=0.6,
             fc=color, ec=color, alpha=alpha * 0.8,
             linewidth=0.5, zorder=4)

    if show_label:
        label_parts = []
        track_id = track.get('track_id', -1)
        if track_id >= 0:
            label_parts.append(f'#{track_id}')
        cls_name = _class_id_to_name(cls_id)
        label_parts.append(f'{CLASS_NAMES_CN.get(cls_name, cls_name)}')

        # 速度标注
        if show_velocity:
            vx = track.get('vx', 0.0)
            vy = track.get('vy', 0.0)
            speed = np.sqrt(vx**2 + vy**2)
            if speed > 0.3:
                label_parts.append(f'{speed:.1f}m/s')

        label = ' '.join(label_parts)
        fontsize = 5.5 if state == 1 else 4.5
        fontweight = 'bold' if state == 1 else 'normal'
        ax.text(cx, cy + w/2 + 1.2, label,
                fontsize=fontsize, ha='center', va='bottom',
                color=color, fontweight=fontweight, alpha=alpha,
                zorder=6)


# ==============================================================================
# 轨迹线绘制
# ==============================================================================
def draw_track_trajectories(
    ax: plt.Axes,
    tracks_history: Dict[int, List[Dict[str, Any]]],
    show_arrows: bool = True,
    trail_length: int = 30,
    min_trail_points: int = 2,
) -> None:
    """在 BEV 平面上绘制所有 track 的历史轨迹线。

    Parameters
    ----------
    ax : matplotlib Axes
    tracks_history : dict  {track_id: [track_dict_at_frame_0, ...]}
        每个 track 的完整历史记录列表。
    show_arrows : bool
        是否在轨迹末端绘制方向箭头。
    trail_length : int
        最多显示最近 N 帧轨迹。
    min_trail_points : int
        最少需要多少帧才绘制轨迹。
    """
    for track_id, history in tracks_history.items():
        if len(history) < min_trail_points:
            continue

        # 取最近 N 帧
        recent = history[-trail_length:]

        cls_id = recent[-1].get('class_id', 3)
        cls_name = _class_id_to_name(cls_id)
        color = CLASS_COLORS.get(cls_name, '#888888')

        pts = np.array([[t['x'], t['y']] for t in recent])

        # 渐变透明度的轨迹线
        n = len(pts)
        for i in range(n - 1):
            frac = i / max(n - 1, 1)
            alpha = TRAIL_ALPHA_START + frac * (TRAIL_ALPHA_END - TRAIL_ALPHA_START)
            ax.plot(pts[i:i+2, 0], pts[i:i+2, 1],
                    color=color, linewidth=1.2, alpha=alpha,
                    solid_capstyle='round', zorder=1)

        # 末端箭头（运动方向指示）
        if show_arrows and n >= 2:
            last_pt = pts[-1]
            prev_pt = pts[-2]
            dx = last_pt[0] - prev_pt[0]
            dy = last_pt[1] - prev_pt[1]
            dist = np.sqrt(dx**2 + dy**2)
            if dist > 0.01:
                dx_n, dy_n = dx / dist * 0.8, dy / dist * 0.8
                ax.arrow(last_pt[0] - dx_n, last_pt[1] - dy_n,
                         dx_n, dy_n,
                         head_width=0.5, head_length=0.7,
                         fc=color, ec=color, alpha=0.9,
                         linewidth=0.3, zorder=5)

        # 起点圆点
        if len(pts) > 1:
            ax.scatter(pts[0, 0], pts[0, 1],
                       c=color, s=15, marker='o', alpha=0.5,
                       edgecolors='white', linewidth=0.3, zorder=2)

        # 终点标记
        ax.scatter(pts[-1, 0], pts[-1, 1],
                   c=color, s=25, marker='D', alpha=0.9,
                   edgecolors='white', linewidth=0.5, zorder=4)


# ==============================================================================
# 时间线图
# ==============================================================================
def draw_track_timeline(
    fig: Optional[plt.Figure] = None,
    tracks_history: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    detections_by_frame: Optional[List[List[Dict]]] = None,
    frame_timestamps: Optional[List[float]] = None,
    show_velocity: bool = True,
) -> plt.Figure:
    """绘制跟踪状态时间线图。

    包含子图:
      - 上方: 每个 track 的 x 位置随时间变化（折线）
      - 中间: 每帧检测数 vs 跟踪数（柱状图）
      - 下方: 每个 track 的状态机彩色条 (candidate/confirmed/lost)

    Parameters
    ----------
    fig : Figure or None
    tracks_history : dict
    detections_by_frame : list of list of dict or None
    frame_timestamps : list of float or None
    show_velocity : bool

    Returns
    -------
    fig : matplotlib Figure
    """
    if tracks_history is None:
        tracks_history = {}

    if fig is None:
        n_rows = 3 if show_velocity else 2
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 3 * n_rows),
                                  sharex=True, gridspec_kw={'hspace': 0.08})
    else:
        axes = fig.subplots(3 if show_velocity else 2, 1, sharex=True,
                            gridspec_kw={'hspace': 0.08})

    if not isinstance(axes, np.ndarray):
        axes = [axes]
    axes = list(axes)

    # 计算帧范围
    all_frames = set()
    for history in tracks_history.values():
        for t in history:
            all_frames.add(t.get('_frame_idx', 0))
    max_frame = max(all_frames) if all_frames else 100
    frame_range = np.arange(max_frame + 1)

    # ---- 子图 1: X 位置随时间变化 ----
    ax1 = axes[0]
    for track_id, history in tracks_history.items():
        if len(history) < 2:
            continue
        cls_id = history[-1].get('class_id', 3)
        cls_name = _class_id_to_name(cls_id)
        color = CLASS_COLORS.get(cls_name, '#888888')

        frames = [t.get('_frame_idx', 0) for t in history]
        xs = [t['x'] for t in history]
        states = [t.get('state', 1) for t in history]

        # 按状态分段绘制（不同线型）
        ax1.plot(frames, xs, color=color, linewidth=1.2, alpha=0.7,
                 label=f'Track #{track_id}' if track_id < 5 else None)

        # 在 confirmed 变为 lost 处标记
        for i in range(1, len(states)):
            if states[i] == 2 and states[i-1] == 1:
                ax1.axvline(x=frames[i], color='orange', linewidth=0.5,
                           alpha=0.3, linestyle=':')

    ax1.set_ylabel('x 位置 (m)', fontsize=9)
    ax1.grid(True, alpha=0.2, linewidth=0.4)
    if len(tracks_history) <= 10:
        ax1.legend(fontsize=6, ncol=2, loc='upper right')

    # ---- 子图 2: 每帧检测/跟踪数量 ----
    ax2 = axes[1]
    if detections_by_frame is not None:
        det_counts = [len(d) for d in detections_by_frame]
        frames_det = list(range(len(det_counts)))
        ax2.bar(frames_det, det_counts, color='#AAAAAA', alpha=0.5,
                label='detections', width=0.8)

    # 每帧活跃 track 数
    track_counts_by_frame = defaultdict(int)
    for history in tracks_history.values():
        for t in history:
            fi = t.get('_frame_idx', 0)
            if t.get('state', 1) != 3:  # not deleted
                track_counts_by_frame[fi] += 1
    if track_counts_by_frame:
        tc_frames = sorted(track_counts_by_frame.keys())
        tc_counts = [track_counts_by_frame[f] for f in tc_frames]
        ax2.plot(tc_frames, tc_counts, 'o-', color='#3366FF', linewidth=1.5,
                 markersize=3, label='active tracks', zorder=5)

    ax2.set_ylabel('数量', fontsize=9)
    ax2.grid(True, alpha=0.2, linewidth=0.4)
    ax2.legend(fontsize=7, loc='upper right')

    # ---- 子图 3: 状态机 ----
    if show_velocity and len(axes) > 2:
        ax3 = axes[2]
        sorted_tracks = sorted(tracks_history.items(),
                               key=lambda kv: len(kv[1]), reverse=True)
        # 只显示前 20 个
        for rank, (track_id, history) in enumerate(sorted_tracks[:20]):
            if len(history) < 1:
                continue
            frames = [t.get('_frame_idx', 0) for t in history]
            states = [t.get('state', 1) for t in history]

            for i in range(len(frames)):
                s = states[i]
                color = STATE_COLORS.get(s, '#888888')
                f_start = frames[i]
                f_end = frames[i+1] if i+1 < len(frames) else frames[i] + 1
                ax3.barh(rank, f_end - f_start, left=f_start, height=0.8,
                         color=color, alpha=0.7, edgecolor='none')

        ax3.set_yticks(range(len(sorted_tracks[:20])))
        ax3.set_yticklabels([f'#{tid}' for tid, _ in sorted_tracks[:20]],
                            fontsize=6)
        ax3.set_ylabel('Track ID', fontsize=9)
        ax3.set_xlabel('帧序号', fontsize=9)
        ax3.grid(True, alpha=0.15, linewidth=0.4, axis='x')

        # 图例
        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=STATE_COLORS[s],
                          edgecolor='white', linewidth=0.3, alpha=0.7,
                          label=STATE_NAMES_CN.get(s, STATE_NAMES.get(s, '?')))
            for s in [0, 1, 2, 3]
        ]
        ax3.legend(handles=legend_elements, fontsize=6, ncol=4,
                   loc='upper right')

    ax1.set_xlim(0, max_frame + 1)
    if frame_timestamps is not None and len(frame_timestamps) > 1:
        # 添加时间戳二次轴
        sec_ax = ax1.twiny()
        sec_ax.set_xlim(ax1.get_xlim())
        tick_frames = np.linspace(0, max_frame, min(6, max_frame + 1), dtype=int)
        tick_times = []
        for f in tick_frames:
            if f < len(frame_timestamps):
                tick_times.append(f'{frame_timestamps[f]:.1f}s')
            else:
                tick_times.append('')
        sec_ax.set_xticks(tick_frames)
        sec_ax.set_xticklabels(tick_times, fontsize=7)
        sec_ax.set_xlabel('时间', fontsize=8)

    fig.suptitle('跟踪状态时间线', fontsize=11, fontweight='bold', y=0.98)
    return fig


# ==============================================================================
# 多帧 BEV 序列
# ==============================================================================
def draw_multiframe_bev(
    tracks_by_frame: Dict[int, List[Dict[str, Any]]],
    detections_by_frame: Optional[Dict[int, List[Dict]]] = None,
    tracks_history: Optional[Dict[int, List[Dict[str, Any]]]] = None,
    frame_indices: Optional[List[int]] = None,
    n_cols: int = 4,
    figsize_per_ax: Tuple[float, float] = (5.5, 5.0),
) -> plt.Figure:
    """绘制多帧 BEV 序列，每帧显示当前 track 框和历史尾迹。

    Parameters
    ----------
    tracks_by_frame : dict {frame_idx: [track_dicts]}
    detections_by_frame : dict or None
    tracks_history : dict or None  {track_id: [history]}
        用于绘制轨迹尾迹。如果提供，每帧只显示在当前帧活跃 track 的最近 N 帧历史。
    frame_indices : list of int or None
        要绘制的帧序号列表。None 时自动选择均匀分布。
    n_cols : int
    figsize_per_ax : (w, h)

    Returns
    -------
    fig : matplotlib Figure
    """
    if frame_indices is None:
        all_frames = sorted(tracks_by_frame.keys())
        if not all_frames:
            logger.warning('No frames to visualize')
            return plt.figure()
        # 均匀选择最多 n_cols*3 帧
        n_frames = min(len(all_frames), n_cols * 3)
        step = max(1, len(all_frames) // n_frames)
        frame_indices = all_frames[::step][:n_frames]

    n_rows = (len(frame_indices) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * figsize_per_ax[0], n_rows * figsize_per_ax[1]),
        squeeze=False,
    )

    for plot_idx, fi in enumerate(frame_indices):
        ax = axes[plot_idx // n_cols][plot_idx % n_cols]
        draw_bev_scene_background(ax)

        # 当前帧的 track 框
        tracks = tracks_by_frame.get(fi, [])
        for trk in tracks:
            draw_track_box(ax, trk, alpha=0.85)

        # 当前帧的检测（虚线框）
        if detections_by_frame is not None:
            dets = detections_by_frame.get(fi, [])
            for det in dets:
                _draw_detection_ghost(ax, det)

        # 轨迹尾迹
        if tracks_history is not None:
            active_ids = {trk.get('track_id') for trk in tracks if trk.get('track_id', -1) >= 0}
            active_history = {
                tid: hist for tid, hist in tracks_history.items()
                if tid in active_ids
            }
            draw_track_trajectories(ax, active_history, trail_length=15, show_arrows=False)

        ax.set_title(f'Frame {fi} | {len(tracks)} tracks',
                     fontsize=8, fontweight='bold')

    # 隐藏多余子图
    for plot_idx in range(len(frame_indices), n_rows * n_cols):
        axes[plot_idx // n_cols][plot_idx % n_cols].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    fig.suptitle('BEV 多帧跟踪序列', fontsize=12, fontweight='bold')
    return fig


def _draw_detection_ghost(ax: plt.Axes, det: Dict[str, Any]) -> None:
    """绘制检测"幽灵"框（灰色虚线，表示未被跟踪的检测）。"""
    cx, cy = det.get('x', 0), det.get('y', 0)
    w, l = det.get('w', 2.0), det.get('l', 4.0)
    yaw = det.get('yaw', 0.0)

    cos_a, sin_a = np.cos(yaw), np.sin(yaw)
    corners_local = np.array([
        [-l/2, -w/2], [l/2, -w/2], [l/2, w/2], [-l/2, w/2]
    ])
    rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    corners = corners_local @ rot.T + np.array([cx, cy])

    ax.fill(corners[:, 0], corners[:, 1],
            alpha=0.08, facecolor='gray',
            edgecolor='gray', linewidth=0.6, linestyle='--', zorder=1)


# ==============================================================================
# 统计报告
# ==============================================================================
def create_track_summary(
    tracks_history: Dict[int, List[Dict[str, Any]]],
    detections_by_frame: Optional[List[List[Dict]]] = None,
    warnings_by_frame: Optional[List[List[Dict]]] = None,
    latency_ms_per_frame: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """生成跟踪统计报告字典（用于 JSON 序列化）。

    Parameters
    ----------
    tracks_history : dict {track_id: [track_dicts]}
    detections_by_frame : list of list of dict or None
    warnings_by_frame : list of list of dict or None
    latency_ms_per_frame : list of float or None

    Returns
    -------
    summary : dict
    """
    summary: Dict[str, Any] = {}

    # 轨迹统计
    n_tracks = len(tracks_history)
    summary['total_tracks'] = n_tracks

    if n_tracks > 0:
        ages = [len(hist) for hist in tracks_history.values()]
        summary['track_age'] = {
            'min': int(np.min(ages)),
            'max': int(np.max(ages)),
            'mean': round(float(np.mean(ages)), 1),
            'median': int(np.median(ages)),
        }

        # 状态分布（最后一帧）
        state_count = defaultdict(int)
        for hist in tracks_history.values():
            if hist:
                s = hist[-1].get('state', 1)
                sn = STATE_NAMES.get(s, 'unknown')
                state_count[sn] += 1
        summary['final_state_distribution'] = dict(state_count)

        # 类别分布
        cls_count = defaultdict(int)
        for hist in tracks_history.values():
            if hist:
                cls_id = hist[-1].get('class_id', 3)
                cls_name = _class_id_to_name(cls_id)
                cls_count[cls_name] += 1
        summary['class_distribution'] = dict(cls_count)

        # 速度统计 (仅 confirmed)
        speeds = []
        for hist in tracks_history.values():
            for t in hist:
                if t.get('state', 0) == 1:  # confirmed
                    vx = t.get('vx', 0)
                    vy = t.get('vy', 0)
                    speeds.append(np.sqrt(vx**2 + vy**2))
        if speeds:
            summary['speed_mps'] = {
                'min': round(float(np.min(speeds)), 2),
                'max': round(float(np.max(speeds)), 2),
                'mean': round(float(np.mean(speeds)), 2),
                'median': round(float(np.median(speeds)), 2),
            }

        # 确认率
        confirmed_count = sum(
            1 for hist in tracks_history.values()
            if any(t.get('state', 0) == 1 for t in hist)
        )
        summary['confirmation_rate'] = round(confirmed_count / n_tracks, 3) if n_tracks else 0

    # 检测统计
    if detections_by_frame is not None:
        det_counts = [len(d) for d in detections_by_frame if d]
        if det_counts:
            summary['detections_per_frame'] = {
                'min': int(np.min(det_counts)),
                'max': int(np.max(det_counts)),
                'mean': round(float(np.mean(det_counts)), 1),
            }
        summary['total_detections'] = int(sum(det_counts))
        summary['frames_with_detections'] = int(sum(1 for c in det_counts if c > 0))

    # 预警统计
    if warnings_by_frame is not None:
        total_warns = sum(len(w) for w in warnings_by_frame if w)
        summary['total_warnings'] = total_warns

        warn_levels = defaultdict(int)
        for warns in warnings_by_frame:
            for w in (warns or []):
                level = w.get('warning_level', 0)
                warn_levels[level] += 1
        summary['warning_level_distribution'] = {
            f'level_{k}': v for k, v in sorted(warn_levels.items())
        }

    # 延迟统计
    if latency_ms_per_frame is not None:
        lats = [l for l in latency_ms_per_frame if l is not None]
        if lats:
            summary['latency_ms'] = {
                'min': round(float(np.min(lats)), 1),
                'max': round(float(np.max(lats)), 1),
                'mean': round(float(np.mean(lats)), 1),
                'median': round(float(np.median(lats)), 1),
            }

    return summary


# ==============================================================================
# 辅助函数
# ==============================================================================
_CLASS_ID_MAP = {0: 'person', 1: 'truck', 2: 'car', 3: 'other_obstacle'}


def _class_id_to_name(cls_id: int) -> str:
    return _CLASS_ID_MAP.get(int(cls_id), 'other_obstacle')
