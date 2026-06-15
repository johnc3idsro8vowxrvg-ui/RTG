"""RTG BEV Sensor FOV Top-Down Diagram — final version."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Wedge, Rectangle, FancyBboxPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = '/media/shen/data/zk/Auto/fov_topdown.png'

fig, ax = plt.subplots(1, 1, figsize=(28, 16))
# Origin shifted right: more balanced view
ax.set_xlim(14, -42)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')
ax.grid(True, alpha=0.10, linestyle='--', linewidth=0.4)
ax.set_xlabel('x — 大车道方向   +x (C1朝向)     -x →', fontsize=11, labelpad=8)
ax.set_ylabel('y — 跨距方向   集卡侧 → 禁行侧 (+y)', fontsize=11)
ax.set_title('RTG BEV 传感器 FOV 俯视图', fontsize=15, fontweight='bold')

# ═══ Lane fills ═══
ls = dict(alpha=0.10, linewidth=0.5, edgecolor='gray')
ax.axhspan(-1.5, 1.5, facecolor='gold', **ls)
ax.text(-30, 0, '大车道 (集卡侧·本RTG)', fontsize=6.5, va='center', color='darkorange', fontweight='bold')
ax.axhspan(1.5, 6.0, facecolor='lightblue', **ls)
ax.text(-30, 3.75, '集卡车道 4.5m', fontsize=6.5, va='center', color='blue')
for y0, y1 in [(6.,8.44),(8.84,11.28),(11.68,14.12),(14.52,16.95),(17.35,19.79),(20.19,22.63)]:
    ax.axhspan(y0, y1, facecolor='brown', alpha=0.06, linewidth=0.3, edgecolor='brown')
ax.text(-30, 14.3, '6列集装箱 16.63m', fontsize=6.5, va='center', color='brown')
ax.axhspan(22.63, 25.0, facecolor='gold', **ls)
ax.text(-30, 23.8, '大车道 (禁行侧·本RTG)', fontsize=6.5, va='center', color='darkorange', fontweight='bold')

ax.axhspan(-7.0, -1.5, facecolor='lightgreen', **ls)
ax.text(-30, -4.25, '中间车道 5.5m', fontsize=6.5, va='center', color='green')
ax.axhspan(-10.0, -7.0, facecolor='gold', **ls)
ax.text(-30, -8.5, '大车道 (集卡侧·相邻RTG)', fontsize=6.5, va='center', color='orange')
for y0, y1 in [(-14.5,-16.94),(-16.94,-19.38),(-19.38,-21.82),(-21.82,-24.26),(-24.26,-26.70),(-26.70,-29.14)]:
    ax.axhspan(y0, y1, facecolor='brown', alpha=0.05, linewidth=0.3, edgecolor='brown')
ax.axhspan(-33.5, -31.13, facecolor='gold', **ls)
ax.text(-30, -32.3, '大车道 (禁行侧·相邻RTG)', fontsize=6.5, va='center', color='orange')

# ═══ Footprints ═══
ax.add_patch(Rectangle((-7.4, -0.5), 7.9, 1.0, facecolor='black', alpha=0.45))
ax.add_patch(Rectangle((-7.4, 23.0), 7.9, 1.0, facecolor='black', alpha=0.45))
ax.add_patch(Rectangle((-7.4, -9.0), 7.9, 1.0, facecolor='gray', alpha=0.30))
ax.add_patch(Rectangle((-7.4, -32.5), 7.9, 1.0, facecolor='gray', alpha=0.30))

# ═══ Helpers ═══
def fov_wedge(x, y, angle, half, r, color, alpha=0.12, lw=1.5, ls='-'):
    ax.add_patch(Wedge((x,y), r, angle-half, angle+half,
                       facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw, linestyle=ls, zorder=1))

def arrow(x, y, angle, length, color, lw=2):
    rad = np.radians(angle)
    ax.annotate('', xy=(x+length*np.cos(rad), y+length*np.sin(rad)),
                xytext=(x+0.5*np.cos(rad), y+0.5*np.sin(rad)),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, alpha=0.85))

def sensor_mark(label, x, y, mk, col, ms, zo, dx, dy, fs=7.5):
    ax.plot(x, y, marker=mk, color=col, markersize=ms,
            markeredgecolor='white', markeredgewidth=0.8, zorder=zo)
    ax.annotate(label, (x,y), textcoords='offset points', xytext=(dx*12, dy*12),
                fontsize=fs, color=col, fontweight='bold', zorder=zo+1)

# ═══ 本RTG Sensors (simple labels) ═══
# BEV 坐标 per geometry.yaml
sensor_mark('C1', 0, 0, 'o', '#ff2222', 11, 12, 0.7, -0.8)
sensor_mark('C2', -12.0, 0, 'o', '#cc1111', 11, 12, -0.8, -0.8)
sensor_mark('C3', 0, 23.5, 'o', '#3366ff', 11, 12, 0.7, 0.8)
sensor_mark('C4', -12.0, 23.5, 'o', '#2244cc', 11, 12, -0.8, 0.8)
sensor_mark('L1', 0, 0, 's', 'red', 14, 11, 0.7, 1.0)
sensor_mark('L2', -12.0, 0, 's', 'darkred', 14, 11, -0.7, 1.0)
sensor_mark('L3', 0, 23.5, 's', 'blue', 14, 11, 0.7, 1.0)
sensor_mark('L4', -12.0, 23.5, 's', 'darkblue', 14, 11, -0.7, 1.0)

# ═══ 本RTG OS1 360° ring (smaller, more visible) ═══
OS1_R = 42
for pos, col in [((0,0), 'red'), ((-12.0,0), 'darkred')]:
    ax.add_patch(Wedge(pos, OS1_R, 0, 360, facecolor=col, alpha=0.04,
                       edgecolor=col, linewidth=1.2, linestyle='--', zorder=1))
    # ground blind ring (smaller, tighter)
    ax.add_patch(plt.Circle(pos, 3.5, fill=True, facecolor='white', alpha=0.55,
                            edgecolor=col, linewidth=0.8, linestyle=':', zorder=2))

# ═══ Helios32 70° (L3, L4) ═══
fov_wedge(0, 23.5, 0, 35, 40, 'blue', alpha=0.11, lw=1.5)
arrow(0, 23.5, 0, 20, 'blue')
fov_wedge(-12.0, 23.5, 180, 35, 40, 'darkblue', alpha=0.11, lw=1.5)
arrow(-12.0, 23.5, 180, 20, 'darkblue')

# ═══ Camera 87.6° (C1-C4) ═══
CH = 43.8
CR = 28
fov_wedge(0, 0, 0, CH, CR, '#ff2222', alpha=0.14, lw=2.2)
arrow(0, 0, 0, 16, '#ff2222')
fov_wedge(-12.0, 0, 180, CH, CR, '#cc1111', alpha=0.14, lw=2.2)
arrow(-12.0, 0, 180, 16, '#cc1111')
fov_wedge(0, 23.5, 0, CH, CR, '#3366ff', alpha=0.14, lw=2.2)
arrow(0, 23.5, 0, 16, '#3366ff')
fov_wedge(-12.0, 23.5, 180, CH, CR, '#2244cc', alpha=0.14, lw=2.2)
arrow(-12.0, 23.5, 180, 16, '#2244cc')

# ═══ 相邻RTG sensors (中心对称, 前后互换) ═══
OL = 'darkorange'
sensor_mark('L1\'', -12.0, -8.5, 's', OL, 11, 8, 0.7, -1.2, fs=6.5)
sensor_mark('L2\'', 0, -8.5, 's', OL, 11, 8, -0.7, -1.2, fs=6.5)
sensor_mark('L3\'', -12.0, -32.0, 's', OL, 11, 8, 0.7, 1.0, fs=6.5)
sensor_mark('L4\'', 0, -32.0, 's', OL, 11, 8, -0.7, 1.0, fs=6.5)
sensor_mark('C1\'', -12.0, -8.5, 'o', 'peru', 10, 9, 0.8, 0.7, fs=6)
sensor_mark('C2\'', 0, -8.5, 'o', 'peru', 10, 9, -0.8, 0.7, fs=6)
sensor_mark('C3\'', -12.0, -32.0, 'o', 'peru', 10, 9, 0.8, 0.7, fs=6)
sensor_mark('C4\'', 0, -32.0, 'o', 'peru', 10, 9, -0.8, 0.7, fs=6)

# Adj FOVs (dashed) — center-symmetric: adj faces -x, sensors point outward from legs
for pos in [(-12.0,-8.5), (0,-8.5)]:
    ax.add_patch(Wedge(pos, 38, 0, 360, facecolor=OL, alpha=0.02,
                       edgecolor=OL, linewidth=0.5, linestyle='--', zorder=1))
    ax.add_patch(plt.Circle(pos, 3.5, fill=False, color=OL, linewidth=0.4,
                            linestyle=':', alpha=0.25, zorder=2))
# Helios32: adj L3 (front, x=-12.0) → faces -x (angle=180); adj L4 (rear, x=0) → faces +x (angle=0)
fov_wedge(-12.0, -32.0, 180, 35, 30, OL, alpha=0.05, lw=0.7, ls=':')
fov_wedge(0, -32.0, 0, 35, 30, OL, alpha=0.05, lw=0.7, ls=':')
# Cameras: adj C1'/C3' (front) → face -x (angle=180); adj C2'/C4' (rear) → face +x (angle=0)
for pos, ang in [((-12.0,-8.5),180), ((0,-8.5),0), ((-12.0,-32.0),180), ((0,-32.0),0)]:
    fov_wedge(pos[0], pos[1], ang, CH, 20, 'peru', alpha=0.06, lw=0.7, ls=':')

# ═══ Lane mirror symmetry axis ═══
ax.axhline(y=-4.25, color='purple', linewidth=1.5, linestyle='--', alpha=0.6, zorder=30)
ax.annotate('车道镜像对称轴 y=-4.25\n(传感器位置为中心对称)', (14, -4.9),
            textcoords='data', fontsize=7.5, color='purple', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

# ═══ TOP-LEFT: sensor specification legend ═══
from matplotlib.lines import Line2D

# Top-left box: symbol meanings
tl_items = [
    ("■ LiDAR (方)", 's', 'black', 'black', 10),
    ("● Camera (圆)", 'o', 'black', 'black', 10),
    ("— 本RTG", '', 'black', 'black', 0),
    ("- - 相邻RTG", '', 'gray', 'gray', 0),
]
tl_text = "\n".join([t[0] for t in tl_items])

# Top-left spec table
spec_text = (
    "传感器参数\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "OS1 128线 (L1/L2)\n"
    "  360° HFOV, 42.4° VFOV\n"
    "  检测半径 ~55m\n"
    "  地面盲区 r≈3.9m\n"
    "  1.5m 安装高度\n"
    "\n"
    "Helios 32 (L3/L4)\n"
    "  70° HFOV (±35°)\n"
    "  360° VFOV (旋转90°)\n"
    "  沿±x大车道扫描\n"
    "  1.5m 安装高度\n"
    "\n"
    "海康相机 (C1~C4)\n"
    "  87.6° HFOV (沿y跨距)\n"
    "  46° VFOV (沿x纵深)\n"
    "  竖屏 1080×1920\n"
    "  4m 安装高度, 斜向下\n"
    "\n"
    "C1/C3 → +x (前)\n"
    "C2/C4 → -x (后)\n"
    "L3 → +x (前), L4 → -x (后)"
)
ax.text(0.01, 0.99, spec_text, transform=ax.transAxes, fontsize=7,
        va='top', fontfamily='monospace', color='#222222',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.8))

# ═══ BOTTOM-RIGHT: legend ═══
legend_elements = [
    Rectangle((0,0),1,1, facecolor='black', alpha=0.45, label='RTG自车 footprint'),
    Rectangle((0,0),1,1, facecolor='gray', alpha=0.30, label='相邻RTG footprint'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='red', markersize=8, label='L1/L2 OS1(360°全向)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='L3/L4 H32(70°±x)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff2222', markersize=8, label='C1/C3 相机(87.6°+x)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#cc1111', markersize=8, label='C2/C4 相机(87.6°-x)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor=OL, markersize=8, label='相邻RTG LiDAR'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='peru', markersize=8, label='相邻RTG 相机'),
    Line2D([0],[0], color='black', lw=1.5, alpha=0.4, linestyle='--', label='OS1检测环(R≈55m)'),
    Line2D([0],[0], color='#ff2222', lw=2, alpha=0.7, label='相机87.6°FOV'),
    Line2D([0],[0], color='blue', lw=1.5, alpha=0.7, label='Helios32 70°FOV'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2, framealpha=0.85)

plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches='tight')
print(f'Saved: {OUT}')
