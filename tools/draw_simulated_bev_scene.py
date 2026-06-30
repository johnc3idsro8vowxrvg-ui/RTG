"""Generate a simulated RTG BEV scene from the documented geometry.

The diagram is deterministic and intended as a visual reference for lanes,
ego footprint, static container rows, and sample trucks in BEV coordinates.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np


X_MIN, X_MAX = -60.0, 60.0
Y_MIN, Y_MAX = -26.0, 25.0

CONTAINER_20FT = 6.058
CONTAINER_40FT = 12.192
CONTAINER_WIDTH = 2.438
CONTAINER_X_GAP = 0.25

LANES = [
    ("forbidden_side_main_lane", -25.6, -22.63, "#f7c948"),
    ("container_rows", -22.63, -6.0, "#a5663f"),
    ("truck_lane", -6.0, -1.5, "#6fb6ff"),
    ("ego_truck_side_main_lane", -1.5, 1.5, "#f7c948"),
    ("middle_lane", 1.5, 7.0, "#84d67b"),
    ("adjacent_truck_side_main_lane", 7.0, 10.0, "#f7c948"),
    ("adjacent_truck_lane", 10.0, 14.5, "#6fb6ff"),
    ("adjacent_container_rows", 14.5, 25.0, "#a5663f"),
]

CONTAINER_ROWS = [
    ("R1", -8.44, -6.00),
    ("R2", -11.28, -8.84),
    ("R3", -14.12, -11.68),
    ("R4", -16.95, -14.52),
    ("R5", -19.79, -17.35),
    ("R6", -22.63, -20.19),
    ("A1", 14.50, 16.94),
    ("A2", 16.94, 19.38),
    ("A3", 19.38, 21.82),
    ("A4", 21.82, 24.26),
]

EGO_FOOTPRINTS = [
    ("truck_side", -12.5, 0.5, -0.5, 0.5),
    ("forbidden_side", -12.5, 0.5, -24.0, -23.0),
]

LEGS = [(0.0, 0.0), (-12.0, 0.0), (0.0, -23.5), (-12.0, -23.5)]
SENSORS = {
    "L1/C1": (0.0, 0.0),
    "L2/C2": (-12.0, 0.0),
    "L3/C3": (0.0, -23.5),
    "L4/C4": (-12.0, -23.5),
}


def setup_fonts() -> None:
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Source Han Sans SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def rotated_box(cx: float, cy: float, length: float, width: float, yaw_deg: float) -> np.ndarray:
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    local = np.array(
        [
            [length / 2, width / 2],
            [length / 2, -width / 2],
            [-length / 2, -width / 2],
            [-length / 2, width / 2],
        ]
    )
    rot = np.array([[c, -s], [s, c]])
    return local @ rot.T + np.array([cx, cy])


def add_label(ax, x: float, y: float, text: str, **kwargs) -> None:
    defaults = {
        "fontsize": 8,
        "color": "#25313f",
        "ha": "center",
        "va": "center",
        "zorder": 20,
    }
    defaults.update(kwargs)
    ax.text(x, y, text, **defaults)


def draw_lanes(ax) -> None:
    for name, y0, y1, color in LANES:
        ax.axhspan(y0, y1, facecolor=color, alpha=0.16, edgecolor=color, linewidth=0.8)

    lane_labels = [
        ("禁行侧大车道", -24.1),
        ("本箱区 6 列集装箱", -14.3),
        ("集卡车道", -3.75),
        ("本 RTG 集卡侧大车道 / y=0 原点", 0.0),
        ("中间车道", 4.25),
        ("相邻 RTG 集卡侧大车道", 8.5),
        ("相邻箱区集卡车道", 12.25),
        ("相邻箱区集装箱区(截取)", 20.0),
    ]
    for text, y in lane_labels:
        add_label(ax, X_MIN + 7, y, text, ha="left", fontsize=8, color="#334155")

    for y in [-25.0, -22.63, -6.0, -1.5, 0.0, 1.5, 4.25, 7.0, 10.0, 14.5, 25.0]:
        style = "--" if y in [-25.0, 25.0, 4.25] else "-"
        color = "#7c3aed" if y == 4.25 else "#64748b"
        lw = 1.3 if y in [-25.0, 25.0, 4.25] else 0.55
        ax.axhline(y, color=color, linestyle=style, linewidth=lw, alpha=0.55, zorder=2)
    add_label(ax, 49, 4.9, "镜像轴 y=+4.25", color="#6d28d9", fontsize=8, ha="right")


def draw_container_rows(ax) -> None:
    palette = ["#b45309", "#c2410c", "#92400e", "#9a3412", "#854d0e"]
    for row_idx, (name, y0, y1) in enumerate(CONTAINER_ROWS):
        center_y = (y0 + y1) / 2
        y = center_y - CONTAINER_WIDTH / 2
        x = X_MIN + 1.0
        n = 0
        while x < X_MAX - 1.0:
            length = CONTAINER_20FT if (n + row_idx) % 3 == 0 else CONTAINER_40FT
            if x + length > X_MAX - 1.0:
                length = X_MAX - 1.0 - x
            color = palette[(n + row_idx) % len(palette)]
            ax.add_patch(
                Rectangle(
                    (x, y),
                    length,
                    CONTAINER_WIDTH,
                    facecolor=color,
                    edgecolor="#fef3c7",
                    linewidth=0.35,
                    alpha=0.78,
                    zorder=5,
                )
            )
            if length > 5.0:
                add_label(
                    ax,
                    x + length / 2,
                    center_y,
                    "20" if length < 8.0 else "40",
                    fontsize=5.5,
                    color="#fff7ed",
                )
            x += length + CONTAINER_X_GAP
            n += 1
        add_label(ax, X_MAX - 2.0, center_y, name, ha="right", fontsize=7, color="#78350f")


def draw_ego(ax) -> None:
    for _, x0, x1, y0, y1 in EGO_FOOTPRINTS:
        ax.add_patch(
            Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor="#111827",
                edgecolor="#111827",
                linewidth=1.2,
                alpha=0.62,
                zorder=11,
            )
        )

    for x, y in LEGS:
        ax.add_patch(
            Rectangle(
                (x - 0.5, y - 0.5),
                1.0,
                1.0,
                facecolor="#020617",
                edgecolor="white",
                linewidth=0.8,
                zorder=13,
            )
        )

    for x in [0.0, -12.0]:
        ax.plot([x, x], [0.0, -23.5], color="#111827", linewidth=1.4, alpha=0.42, zorder=10)
    ax.plot([0.0, -12.0], [0.0, 0.0], color="#111827", linewidth=1.2, alpha=0.5, zorder=10)
    ax.plot([0.0, -12.0], [-23.5, -23.5], color="#111827", linewidth=1.2, alpha=0.5, zorder=10)

    for label, (x, y) in SENSORS.items():
        ax.add_patch(Circle((x, y), 0.42, facecolor="#ef4444", edgecolor="white", linewidth=0.8, zorder=15))
        add_label(ax, x + 1.0, y + 0.85, label, fontsize=7.2, ha="left", color="#991b1b")

    add_label(ax, -6.0, 1.05, "自车 footprint x=[-12.5,0.5]", fontsize=7.5, color="#111827")
    add_label(ax, -6.0, -22.45, "禁行侧 footprint", fontsize=7.5, color="#111827")


def draw_truck(ax, cx: float, cy: float, yaw: float, label: str, body_color: str) -> None:
    trailer_len = 12.5
    cab_len = 3.5
    width = 2.7
    total_len = trailer_len + cab_len

    body = rotated_box(cx, cy, total_len, width, yaw)
    ax.add_patch(Polygon(body, closed=True, facecolor=body_color, edgecolor="#172554", linewidth=1.2, alpha=0.86, zorder=17))

    yaw_rad = np.deg2rad(yaw)
    ux, uy = np.cos(yaw_rad), np.sin(yaw_rad)
    cab_cx = cx + ux * (total_len / 2 - cab_len / 2)
    cab_cy = cy + uy * (total_len / 2 - cab_len / 2)
    trailer_cx = cx - ux * (cab_len / 2)
    trailer_cy = cy - uy * (cab_len / 2)

    trailer = rotated_box(trailer_cx, trailer_cy, trailer_len, width * 0.92, yaw)
    cab = rotated_box(cab_cx, cab_cy, cab_len, width, yaw)
    ax.add_patch(Polygon(trailer, closed=True, facecolor="#f97316", edgecolor="#7c2d12", linewidth=0.8, alpha=0.88, zorder=18))
    ax.add_patch(Polygon(cab, closed=True, facecolor="#0ea5e9", edgecolor="#075985", linewidth=0.8, alpha=0.95, zorder=19))

    front = (cx + ux * (total_len / 2 + 2.0), cy + uy * (total_len / 2 + 2.0))
    tail = (cx + ux * (total_len / 2 - 0.6), cy + uy * (total_len / 2 - 0.6))
    ax.add_patch(FancyArrowPatch(tail, front, arrowstyle="-|>", mutation_scale=12, color="#075985", linewidth=1.3, zorder=21))
    add_label(ax, cx, cy + 2.15, label, fontsize=8, color="#0f172a")


def draw_misc_targets(ax) -> None:
    car = rotated_box(-21.5, 11.9, 4.6, 1.9, 180)
    ax.add_patch(Polygon(car, closed=True, facecolor="#22c55e", edgecolor="#166534", linewidth=0.9, alpha=0.9, zorder=16))
    add_label(ax, -21.5, 13.35, "car", fontsize=7, color="#14532d")

    ax.add_patch(Circle((8.5, -2.2), 0.35, facecolor="#dc2626", edgecolor="white", linewidth=0.7, zorder=18))
    add_label(ax, 8.5, -1.45, "person", fontsize=7, color="#991b1b")


def draw_axes_and_legend(ax) -> None:
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#94a3b8", linestyle=":", linewidth=0.45, alpha=0.45)
    ax.set_xlabel("+x 大车道方向 / C1 朝向 (m)", fontsize=10)
    ax.set_ylabel("+y 跨距方向: 禁行侧 -> 集卡侧 (m)", fontsize=10)
    ax.set_title("RTG BEV 模拟场景: 车道 / 自车 footprint / 集装箱 / 集卡", fontsize=14, fontweight="bold")

    ax.axvline(0, color="#0f172a", linewidth=0.8, alpha=0.45)
    ax.axhline(0, color="#0f172a", linewidth=0.8, alpha=0.45)
    ax.add_patch(Rectangle((X_MIN, -25.0), X_MAX - X_MIN, 50.0, fill=False, edgecolor="#ef4444", linestyle="--", linewidth=1.2, zorder=30))
    add_label(ax, 47, -24.25, "BEV y范围边界 [-25,25]", ha="right", fontsize=7.5, color="#b91c1c")

    legend_text = (
        "文档几何:\n"
        "原点=L1地面投影; +x=C1朝向; +y=禁行侧->集卡侧\n"
        "RTG跨距=23.5m; 前后支腿距=12.0m\n"
        "集卡标注框约 16m x 2.7m, yaw沿 x 轴"
    )
    ax.text(
        0.012,
        0.985,
        legend_text,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="#1f2937",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#cbd5e1", alpha=0.9),
        zorder=40,
    )


def generate(out_png: Path, out_svg: Path | None = None) -> None:
    setup_fonts()
    fig, ax = plt.subplots(figsize=(24, 11.5))
    draw_lanes(ax)
    draw_container_rows(ax)
    draw_ego(ax)
    draw_truck(ax, 25.0, -3.75, 0.0, "truck: loaded container", "#1d4ed8")
    draw_truck(ax, -36.0, -3.75, 180.0, "truck: approaching from -x", "#2563eb")
    draw_misc_targets(ax)
    draw_axes_and_legend(ax)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    if out_svg is not None:
        out_svg.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_svg, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Draw a simulated RTG BEV scene.")
    parser.add_argument("--out", default="outputs/rtg_simulated_bev_scene.png", help="Output PNG path.")
    parser.add_argument("--svg", default="outputs/rtg_simulated_bev_scene.svg", help="Output SVG path.")
    args = parser.parse_args()
    generate(Path(args.out), Path(args.svg) if args.svg else None)
    print(f"Saved PNG: {args.out}")
    if args.svg:
        print(f"Saved SVG: {args.svg}")


if __name__ == "__main__":
    main()
