"""RTG BEV FOV Top-Down — based on RViz alignment data."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Rectangle
from matplotlib.lines import Line2D
import math

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = '/media/shen/data/zk/Auto/fov_topdown_aligned.png'

fig, ax = plt.subplots(1, 1, figsize=(28, 16))
ax.set_xlim(-15, 45)
ax.set_ylim(-15, 15)
ax.set_aspect('equal')
ax.grid(True, alpha=0.15, linestyle=':', linewidth=0.4)
ax.set_xlabel('x (m)', fontsize=12)
ax.set_ylabel('y (m)', fontsize=12)
ax.set_title('RTG BEV 传感器 FOV 俯视图 (RViz 对齐数据)', fontsize=15, fontweight='bold')

# ═══ Sensor positions from RViz alignment / geometry.yaml ═══
# Plot convention: x = BEV y, y = BEV x
# L1: BEV (0, 0, 1.5)     → plot (0, 0)
# L2: BEV (-12.0, 0, 1.5) → plot (0, -12.0)
# L3: BEV (0, 23.5, 1.5)  → plot (23.5, 0)
# L4: BEV (-12.0, 23.5, 1.5) → plot (23.5, -12.0)

SENSORS = {
    'L1': {'pos': (0, 0), 'yaw_deg': 0, 'color': 'red', 'marker': 's', 'label': 'L1 OS1'},
    'L2': {'pos': (0, -12.0), 'yaw_deg': -1, 'color': 'green', 'marker': 's', 'label': 'L2 OS1'},
    'L3': {'pos': (23.5, 0), 'yaw_deg': 45, 'color': 'blue', 'marker': 's', 'label': 'L3 Helios32'},
    'L4': {'pos': (23.5, -12.0), 'yaw_deg': -135, 'color': 'orange', 'marker': 's', 'label': 'L4 Helios32'},
    'C1': {'pos': (0, 0), 'yaw_deg': 0, 'color': '#ff4444', 'marker': 'o', 'label': 'C1 camera'},
    'C2': {'pos': (0, -12.0), 'yaw_deg': 180, 'color': '#44ff44', 'marker': 'o', 'label': 'C2 camera'},
    'C3': {'pos': (23.5, 0), 'yaw_deg': 0, 'color': '#4444ff', 'marker': 'o', 'label': 'C3 camera'},
    'C4': {'pos': (23.5, -12.0), 'yaw_deg': 180, 'color': '#ffaa00', 'marker': 'o', 'label': 'C4 camera'},
}

# ═══ Footprints (two rectangles: wheel spacing=12.0m along y, span=23.5m along x) ═══
# Truck side: L1(0,0) to L2(0,-12.0) along y
fp_w = 1.0  # leg width
# Truck side footprint: connects L1(0,0) to L2(0,-12.0) along y
ax.add_patch(Rectangle((-0.6, -12.5), 1.2, 13.0, facecolor='black', alpha=0.4, zorder=5))
ax.text(1.2, -6.0, '集卡侧\nfootprint', fontsize=7, color='black', va='center', fontweight='bold')
# Forbidden side: L3(23.5,0) to L4(23.5,-12.0)
ax.add_patch(Rectangle((22.9, -12.5), 1.2, 13.0, facecolor='black', alpha=0.4, zorder=5))
ax.text(25, -6.0, '禁行侧\nfootprint', fontsize=7, color='black', va='center', fontweight='bold')

# ═══ Draw OS1 rings (L1, L2: 360°) ═══
OS1_R = 42
for name in ['L1', 'L2']:
    s = SENSORS[name]
    ax.add_patch(Wedge(s['pos'], OS1_R, 0, 360, facecolor=s['color'], alpha=0.06,
                       edgecolor=s['color'], linewidth=1.2, linestyle='--', zorder=1))
    ax.add_patch(plt.Circle(s['pos'], 3.5, fill=True, facecolor='white', alpha=0.15,
                            edgecolor=s['color'], linewidth=0.8, linestyle=':', zorder=2))

# ═══ Draw Helios32 FOV (L3, L4: 70° H) ═══
for name in ['L3', 'L4']:
    s = SENSORS[name]
    yaw = math.radians(s['yaw_deg'])
    hfov = 35  # half of 70°
    ax.add_patch(Wedge(s['pos'], 30, math.degrees(yaw)-hfov, math.degrees(yaw)+hfov,
                       facecolor=s['color'], alpha=0.12, edgecolor=s['color'], linewidth=1.5, zorder=1))
    # Arrow
    ax.annotate('', xy=(s['pos'][0]+15*math.cos(yaw), s['pos'][1]+15*math.sin(yaw)),
                xytext=s['pos'], arrowprops=dict(arrowstyle='->', color=s['color'], lw=2, alpha=0.7))

# ═══ Draw Camera FOV (C1-C4: 87.6° HFOV) ═══
CH = 43.8  # half of 87.6°
CR = 22
for name in ['C1', 'C2', 'C3', 'C4']:
    s = SENSORS[name]
    yaw = math.radians(s['yaw_deg'])
    ax.add_patch(Wedge(s['pos'], CR, math.degrees(yaw)-CH, math.degrees(yaw)+CH,
                       facecolor=s['color'], alpha=0.15, edgecolor=s['color'], linewidth=2, zorder=3))
    # Arrow
    ax.annotate('', xy=(s['pos'][0]+12*math.cos(yaw), s['pos'][1]+12*math.sin(yaw)),
                xytext=s['pos'], arrowprops=dict(arrowstyle='->', color=s['color'], lw=2.5, alpha=0.8))

# ═══ Draw sensor markers ═══
for name, s in SENSORS.items():
    ax.plot(s['pos'][0], s['pos'][1], marker=s['marker'], color=s['color'],
            markersize=12, markeredgecolor='white', markeredgewidth=1, zorder=10)
    label = name
    dx = 0.7 if s['yaw_deg'] >= 0 else -0.7
    dy = 0.7
    ax.annotate(label, s['pos'], textcoords='offset points', xytext=(dx*12, dy*12),
                fontsize=8.5, color=s['color'], fontweight='bold', zorder=11)

# ═══ Dimension lines ═══
# Wheelbase (前后支腿间距)
ax.annotate('', xy=(0, -12.0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, linestyle='--'))
ax.text(-1.2, -6.0, '12.0m\n(前后支腿间距)', fontsize=8, color='gray', ha='right', va='center')

# Span
ax.annotate('', xy=(23.5, 1.5), xytext=(0, 1.5),
            arrowprops=dict(arrowstyle='<->', color='gray', lw=1.5, linestyle='--'))
ax.text(11.8, 2.5, '23.5m (跨距, 推算)', fontsize=8, color='gray', ha='center')

# ═══ Legend ═══
legend_elements = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='L1 OS1 (集卡前)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='L2 OS1 (集卡后)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', markersize=10, label='L3 Helios32 (禁行前)'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='orange', markersize=10, label='L4 Helios32 (禁行后)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff4444', markersize=8, label='C1/C3 相机 (+x)'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#44ff44', markersize=8, label='C2/C4 相机 (-x)'),
    Rectangle((0,0),1,1, facecolor='black', alpha=0.4, label='RTG footprint'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=8, ncol=2, framealpha=0.9)

# ═══ Spec table ═══
spec_text = (
    "geometry.yaml 传感器位置 (plot: x=BEVy, y=BEVx)\n"
    "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
    "L1: BEV (0, 0)        → plot (0, 0)\n"
    "L2: BEV (-12, 0)      → plot (0, -12)\n"
    "L3: BEV (0, 23.5)     → plot (23.5, 0)\n"
    "L4: BEV (-12, 23.5)   → plot (23.5, -12)\n"
    "\n"
    "前后支腿间距 12.0m\n"
    "跨距 ~23.5m 沿 +x (推算)\n"
    "\n"
    "C1/C3: 87.6deg HFOV +x\n"
    "C2/C4: 87.6deg HFOV -x\n"
    "L1/L2: OS1 360deg\n"
    "L3/L4: Helios32 70deg H"
)
ax.text(0.01, 0.99, spec_text, transform=ax.transAxes, fontsize=7.5,
        va='top', fontfamily='monospace', color='#222222',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray', linewidth=0.8))

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved: {OUT}')
