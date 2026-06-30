"""RTG BEV + Sensor FOV animated top-down video."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np
from matplotlib.patches import Wedge, Rectangle, FancyBboxPatch

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['AR PL UMing CN', 'Noto Sans CJK JP', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT = '/media/shen/data/zk/Auto/fov_topdown_animation.mp4'
FPS = 15
DURATION = 40  # seconds
TOTAL_FRAMES = FPS * DURATION

fig, ax = plt.subplots(1, 1, figsize=(28, 16))
ax.set_xlim(14, -42)
ax.set_ylim(-40, 40)
ax.set_aspect('equal')
ax.grid(True, alpha=0.08, linestyle='--', linewidth=0.3)
ax.set_xlabel('x — 大车道方向   +x (C1朝向)     -x →', fontsize=11, labelpad=8)
ax.set_ylabel('y — 跨距方向   禁行侧 → 集卡侧 (+y)', fontsize=11)
ax.set_title('RTG BEV 传感器 FOV 俯视图 — 动态演示', fontsize=15, fontweight='bold')

# ═══ Static lane fills ═══
ls = dict(alpha=0.10, linewidth=0.5, edgecolor='gray')
ax.axhspan(-1.5, 1.5, facecolor='gold', **ls)
ax.text(-35, 0, '大车道 (集卡侧·本RTG)', fontsize=6.5, va='center', color='darkorange', fontweight='bold')
ax.axhspan(1.5, 6.0, facecolor='lightblue', **ls)
ax.text(-35, 3.75, '集卡车道 4.5m', fontsize=6.5, va='center', color='blue')
for y0, y1 in [(6.,8.44),(8.84,11.28),(11.68,14.12),(14.52,16.95),(17.35,19.79),(20.19,22.63)]:
    ax.axhspan(y0, y1, facecolor='brown', alpha=0.06, linewidth=0.3, edgecolor='brown')
ax.text(-35, 14.3, '6列集装箱 16.63m', fontsize=6.5, va='center', color='brown')
ax.axhspan(22.63, 25.0, facecolor='gold', **ls)
ax.text(-35, 23.8, '大车道 (禁行侧·本RTG)', fontsize=6.5, va='center', color='darkorange', fontweight='bold')

ax.axhspan(-7.0, -1.5, facecolor='lightgreen', **ls)
ax.text(-35, -4.25, '中间车道 5.5m', fontsize=6.5, va='center', color='green')
ax.axhspan(-10.0, -7.0, facecolor='gold', **ls)
ax.text(-35, -8.5, '大车道 (集卡侧·相邻RTG)', fontsize=6.5, va='center', color='orange')
for y0, y1 in [(-14.5,-16.94),(-16.94,-19.38),(-19.38,-21.82),(-21.82,-24.26),(-24.26,-26.70),(-26.70,-29.14)]:
    ax.axhspan(y0, y1, facecolor='brown', alpha=0.05, linewidth=0.3, edgecolor='brown')
ax.axhspan(-33.5, -31.13, facecolor='gold', **ls)
ax.text(-35, -32.3, '大车道 (禁行侧·相邻RTG)', fontsize=6.5, va='center', color='orange')

# Lane mirror symmetry axis
ax.axhline(y=-4.25, color='purple', linewidth=1.2, linestyle='--', alpha=0.5, zorder=30)
ax.annotate('车道镜像对称轴 y=-4.25', (14, -5.0), fontsize=7, color='purple', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8), zorder=31)

# ═══ Helpers ═══
CH = 43.8  # camera half-angle
CR = 28    # camera visual radius
OS1_R = 42

def fov_wedge(x, y, angle, half, r, color, alpha=0.12, lw=1.5, ls='-'):
    return ax.add_patch(Wedge((x,y), r, angle-half, angle+half,
                        facecolor=color, alpha=alpha, edgecolor=color, linewidth=lw, linestyle=ls, zorder=1))

def make_os1_ring(x, y, color, alpha=0.04, lw=1.2):
    return [ax.add_patch(Wedge((x,y), OS1_R, 0, 360, facecolor=color, alpha=alpha,
                                edgecolor=color, linewidth=lw, linestyle='--', zorder=1)),
            ax.add_patch(plt.Circle((x,y), 3.5, fill=True, facecolor='white', alpha=0.55,
                                     edgecolor=color, linewidth=0.8, linestyle=':', zorder=2))]

def make_rtg_marks(base_x):
    """Return list of artists for this RTG's sensors at given base_x (front leg x)."""
    marks = []
    # Positions relative to front leg at base_x
    rear_x = base_x - 12.0  # 前后支腿间距 12.0m (per geometry.yaml)
    # LiDAR marks
    L1, = ax.plot(base_x, 0, 's', color='red', markersize=14, markeredgecolor='white', markeredgewidth=0.8, zorder=11)
    L2, = ax.plot(rear_x, 0, 's', color='darkred', markersize=14, markeredgecolor='white', markeredgewidth=0.8, zorder=11)
    L3, = ax.plot(base_x, 23.5, 's', color='blue', markersize=14, markeredgecolor='white', markeredgewidth=0.8, zorder=11)
    L4, = ax.plot(rear_x, 23.5, 's', color='darkblue', markersize=14, markeredgecolor='white', markeredgewidth=0.8, zorder=11)
    # Camera marks
    C1, = ax.plot(base_x, 0, 'o', color='#ff2222', markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=12)
    C2, = ax.plot(rear_x, 0, 'o', color='#cc1111', markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=12)
    C3, = ax.plot(base_x, 23.5, 'o', color='#3366ff', markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=12)
    C4, = ax.plot(rear_x, 23.5, 'o', color='#2244cc', markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=12)
    marks.extend([L1, L2, L3, L4, C1, C2, C3, C4])
    # OS1 rings
    for cx, c in [(base_x, 'red'), (rear_x, 'darkred')]:
        marks.extend(make_os1_ring(cx, 0, c))
    # Helios32 FOVs (70°)
    marks.append(fov_wedge(base_x, 23.5, 0, 35, 40, 'blue', alpha=0.11, lw=1.5))
    marks.append(fov_wedge(rear_x, 23.5, 180, 35, 40, 'darkblue', alpha=0.11, lw=1.5))
    # Camera FOVs (87.6°)
    marks.append(fov_wedge(base_x, 0, 0, CH, CR, '#ff2222', alpha=0.14, lw=2.2))
    marks.append(fov_wedge(rear_x, 0, 180, CH, CR, '#cc1111', alpha=0.14, lw=2.2))
    marks.append(fov_wedge(base_x, 23.5, 0, CH, CR, '#3366ff', alpha=0.14, lw=2.2))
    marks.append(fov_wedge(rear_x, 23.5, 180, CH, CR, '#2244cc', alpha=0.14, lw=2.2))
    # Footprint
    fp1 = ax.add_patch(Rectangle((base_x - 7.4, -0.5), 7.9, 1.0, facecolor='black', alpha=0.45))
    fp2 = ax.add_patch(Rectangle((base_x - 7.4, 23.0), 7.9, 1.0, facecolor='black', alpha=0.45))
    marks.extend([fp1, fp2])
    # Labels
    t1 = ax.annotate('C1', (base_x, 0), textcoords='offset points', xytext=(8, -10),
                     fontsize=7.5, color='#ff2222', fontweight='bold', zorder=13)
    t2 = ax.annotate('C2', (rear_x, 0), textcoords='offset points', xytext=(-10, -10),
                     fontsize=7.5, color='#cc1111', fontweight='bold', zorder=13)
    t3 = ax.annotate('C3', (base_x, 23.5), textcoords='offset points', xytext=(8, 10),
                     fontsize=7.5, color='#3366ff', fontweight='bold', zorder=13)
    t4 = ax.annotate('C4', (rear_x, 23.5), textcoords='offset points', xytext=(-10, 10),
                     fontsize=7.5, color='#2244cc', fontweight='bold', zorder=13)
    t5 = ax.annotate('L1', (base_x, 0), textcoords='offset points', xytext=(8, 12),
                     fontsize=7.5, color='red', fontweight='bold', zorder=13)
    t6 = ax.annotate('L2', (rear_x, 0), textcoords='offset points', xytext=(-8, 12),
                     fontsize=7.5, color='darkred', fontweight='bold', zorder=13)
    t7 = ax.annotate('L3', (base_x, 23.5), textcoords='offset points', xytext=(8, 12),
                     fontsize=7.5, color='blue', fontweight='bold', zorder=13)
    t8 = ax.annotate('L4', (rear_x, 23.5), textcoords='offset points', xytext=(-8, 12),
                     fontsize=7.5, color='darkblue', fontweight='bold', zorder=13)
    marks.extend([t1, t2, t3, t4, t5, t6, t7, t8])
    return marks

def make_adj_marks(base_x):
    """Adjacent RTG: center-symmetric sensor positions (front/rear swapped)."""
    marks = []
    rear_x = base_x - 12.0  # adj's front leg = center-symmetry of ego rear leg
    OL = 'darkorange'
    # Adj LiDAR (swapped)
    L1, = ax.plot(base_x, -8.5, 's', color=OL, markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=8)
    L2, = ax.plot(rear_x, -8.5, 's', color=OL, markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=8)
    L3, = ax.plot(base_x, -32.0, 's', color=OL, markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=8)
    L4, = ax.plot(rear_x, -32.0, 's', color=OL, markersize=11, markeredgecolor='white', markeredgewidth=0.8, zorder=8)
    C1, = ax.plot(base_x, -8.5, 'o', color='peru', markersize=10, markeredgecolor='white', markeredgewidth=0.8, zorder=9)
    C2, = ax.plot(rear_x, -8.5, 'o', color='peru', markersize=10, markeredgecolor='white', markeredgewidth=0.8, zorder=9)
    C3, = ax.plot(base_x, -32.0, 'o', color='peru', markersize=10, markeredgecolor='white', markeredgewidth=0.8, zorder=9)
    C4, = ax.plot(rear_x, -32.0, 'o', color='peru', markersize=10, markeredgecolor='white', markeredgewidth=0.8, zorder=9)
    marks.extend([L1, L2, L3, L4, C1, C2, C3, C4])
    # Adj OS1 rings
    for cx in [base_x, rear_x]:
        marks.append(ax.add_patch(Wedge((cx, -8.5), 38, 0, 360, facecolor=OL, alpha=0.02,
                                         edgecolor=OL, linewidth=0.5, linestyle='--', zorder=1)))
        marks.append(ax.add_patch(plt.Circle((cx, -8.5), 3.5, fill=False, color=OL, linewidth=0.4,
                                              linestyle=':', alpha=0.25, zorder=2)))
    # Adj Helios32: adj faces -x (in our frame), sensors point outward from leg
    # L3' at base_x (adj front) → adj +x = our -x → angle=180
    marks.append(fov_wedge(base_x, -32.0, 180, 35, 30, OL, alpha=0.05, lw=0.7, ls=':'))
    # L4' at rear_x (adj rear) → adj -x = our +x → angle=0
    marks.append(fov_wedge(rear_x, -32.0, 0, 35, 30, OL, alpha=0.05, lw=0.7, ls=':'))
    # Adj Cameras: C1'/C3' (adj front) → face our -x (angle=180); C2'/C4' (adj rear) → face our +x (angle=0)
    marks.append(fov_wedge(base_x, -8.5, 180, CH, 20, 'peru', alpha=0.06, lw=0.7, ls=':'))
    marks.append(fov_wedge(rear_x, -8.5, 0, CH, 20, 'peru', alpha=0.06, lw=0.7, ls=':'))
    marks.append(fov_wedge(base_x, -32.0, 180, CH, 20, 'peru', alpha=0.06, lw=0.7, ls=':'))
    marks.append(fov_wedge(rear_x, -32.0, 0, CH, 20, 'peru', alpha=0.06, lw=0.7, ls=':'))
    # Adj footprint (center-symmetric: front/rear swapped)
    fp1 = ax.add_patch(Rectangle((rear_x - 0.5, -9.0), 7.9, 1.0, facecolor='gray', alpha=0.30))
    fp2 = ax.add_patch(Rectangle((rear_x - 0.5, -32.5), 7.9, 1.0, facecolor='gray', alpha=0.30))
    marks.extend([fp1, fp2])
    # Labels
    t1 = ax.annotate("C1'", (base_x, -8.5), textcoords='offset points', xytext=(8, -10),
                     fontsize=6.5, color='peru', fontweight='bold', zorder=10)
    t2 = ax.annotate("C2'", (rear_x, -8.5), textcoords='offset points', xytext=(-10, -10),
                     fontsize=6.5, color='peru', fontweight='bold', zorder=10)
    marks.extend([t1, t2])
    return marks

# ═══ Moving objects: truck, person ═══
def make_truck(x_pos):
    """Truck in truck lane (y≈3.75)."""
    truck = ax.add_patch(Rectangle((x_pos - 6, 2.3), 12, 2.9,
                           facecolor='green', alpha=0.6, edgecolor='darkgreen', linewidth=1.5))
    label = ax.annotate('集卡', (x_pos, 2.0), fontsize=7, color='darkgreen',
                        fontweight='bold', ha='center', zorder=20)
    return [truck, label]

def make_person(px, py):
    """Person at (px, py)."""
    p = ax.add_patch(plt.Circle((px, py), 0.3, facecolor='red', alpha=0.8, zorder=20))
    label = ax.annotate('人', (px, py - 0.6), fontsize=7, color='red',
                        fontweight='bold', ha='center', zorder=20)
    return [p, label]

def make_container(x_pos):
    """Container in column area."""
    c = ax.add_patch(Rectangle((x_pos - 3, 16.0), 6, 2.4,
                       facecolor='saddlebrown', alpha=0.7, edgecolor='darkred', linewidth=1))
    return [c]

# ═══ Legend ═══
from matplotlib.lines import Line2D
legend_elements = [
    Rectangle((0,0),1,1, facecolor='black', alpha=0.45, label='本RTG footprint'),
    Rectangle((0,0),1,1, facecolor='gray', alpha=0.30, label='相邻RTG footprint'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='red', markersize=8, label='L1/L2 OS1 (360°)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='blue', markersize=8, label='L3/L4 H32 (70°)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#ff2222', markersize=8, label='C1/C3 相机 (87.6°+x)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='#cc1111', markersize=8, label='C2/C4 相机 (87.6°-x)'),
    Line2D([0],[0], marker='s', color='w', markerfacecolor='darkorange', markersize=8, label='相邻RTG LiDAR'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='peru', markersize=8, label='相邻RTG 相机'),
    Rectangle((0,0),1,1, facecolor='green', alpha=0.6, label='集卡'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=7, ncol=2, framealpha=0.85)

# ═══ Spec table ═══
spec_text = (
    "传感器参数\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "本RTG: C1/C3朝+x, C2/C4朝-x\n"
    "相邻RTG: 传感器中心对称 (前后互换)\n"
    "车道布局: 关于y=-4.25 镜像对称\n"
    "━━━━━━━━━━━━━━━━━━\n"
    "OS1 128线 (L1/L2)\n"
    "  360° HFOV, 42.4° VFOV\n"
    "  1.5m 安装高度\n"
    "\n"
    "Helios 32 (L3/L4)\n"
    "  70° HFOV (±35°), 360° VFOV\n"
    "  旋转90°安装, 1.5m高\n"
    "\n"
    "海康相机 (C1~C4)\n"
    "  87.6° HFOV (沿y跨距)\n"
    "  46° VFOV (沿x纵深)\n"
    "  竖屏 1080×1920, 4m高"
)
ax.text(0.01, 0.99, spec_text, transform=ax.transAxes, fontsize=7,
        va='top', fontfamily='monospace', color='#222222',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=0.8))

# ═══ Time text ═══
time_text = ax.text(0.98, 0.01, '', transform=ax.transAxes, fontsize=9, ha='right',
                     bbox=dict(boxstyle='round', facecolor='black', alpha=0.6, edgecolor='none'), color='white')

plt.tight_layout()

# ═══ Animation state ═══
# This RTG: sinusoidal motion
# Adjacent RTG: independent motion (different period + linear drift)
this_rtg_x = 12.0       # starting x position of front leg
adj_rtg_x = -18.0       # starting x position of adj front leg (center-symmetry: this is the swapped leg)
truck_positions = []    # list of truck x positions (multiple trucks)
person_positions = []   # list of (x, y) for persons

# Spawn initial truck
truck_x = -25.0
truck_speed = 1.2  # m/s

# Spawn person
person_x = -15.0
person_y = -3.0  # in middle lane
person_walking = True

# Initialize artist containers
this_marks = []
adj_marks = []
truck_artists = []
person_artists = []

def init():
    global this_marks, adj_marks, truck_artists, person_artists
    this_marks = make_rtg_marks(this_rtg_x)
    adj_marks = make_adj_marks(adj_rtg_x)
    truck_artists = make_truck(truck_x)
    person_artists = make_person(person_x, person_y)
    return this_marks + adj_marks + truck_artists + person_artists + [time_text]

def update(frame):
    global this_rtg_x, adj_rtg_x, truck_x, person_x, person_y
    global this_marks, adj_marks, truck_artists, person_artists

    t = frame / FPS

    # Remove old artists
    for m in this_marks + adj_marks + truck_artists + person_artists:
        m.remove()
    this_marks.clear()
    adj_marks.clear()
    truck_artists.clear()
    person_artists.clear()

    # This RTG: slow sinusoidal motion ±16m
    this_rtg_x = 12.0 + 16.0 * np.sin(2 * np.pi * t / 22.0)

    # Adjacent RTG: independent motion (different period + slower drift)
    adj_rtg_x = -18.0 + 12.0 * np.sin(2 * np.pi * t / 31.0 + 1.2)

    # Truck: moves along truck lane
    truck_x += truck_speed / FPS
    if truck_x > 40:
        truck_x = -35.0
    if truck_x < -35:
        truck_x = 40.0

    # Person: walks in middle lane area
    person_x += 0.6 / FPS * np.cos(2 * np.pi * t / 12.0)
    person_y = -3.0 + 1.5 * np.sin(2 * np.pi * t / 15.0)

    # Redraw
    this_marks = make_rtg_marks(this_rtg_x)
    adj_marks = make_adj_marks(adj_rtg_x)
    truck_artists = make_truck(truck_x)
    person_artists = make_person(person_x, person_y)

    time_text.set_text(f' t = {t:.1f}s')

    return this_marks + adj_marks + truck_artists + person_artists + [time_text]

print('Rendering animation...')
ani = anim.FuncAnimation(fig, update, frames=TOTAL_FRAMES, init_func=init,
                          blit=False, interval=1000/FPS)
ani.save(OUT, fps=FPS, dpi=120, extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print(f'Saved: {OUT}')
