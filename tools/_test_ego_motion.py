#!/usr/bin/env python3
"""Smoke-test RTG ego-motion estimation on synchronized L1/L2 bag frames."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def iter_synced_lidar_pairs(
    bag_path: Path,
    topics: Dict[str, str],
    sync_window_sec: float,
    limit: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Yield synchronized lidar_01/lidar_02 message pairs from a ROS1 bag."""
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "The pure-Python 'rosbags' package is required outside ROS1. "
            "Install it with: python -m pip install rosbags"
        ) from exc

    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted = {
        topics["lidar_01"]: "lidar_01",
        topics["lidar_02"]: "lidar_02",
    }
    latest: Dict[str, Dict[str, Any]] = {}
    emitted = 0

    with Reader(bag_path) as reader:
        connection_by_topic = {conn.topic: conn for conn in reader.connections}
        missing = [topic for topic in wanted if topic not in connection_by_topic]
        if missing:
            raise RuntimeError(f"Missing lidar topic(s) in bag: {missing}")

        connections = [connection_by_topic[topic] for topic in wanted]
        for conn, timestamp, rawdata in reader.messages(connections=connections):
            sensor = wanted[conn.topic]
            bag_time = float(timestamp) / 1e9
            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            entry = {"msg": msg, "bag_time": bag_time}

            other = "lidar_02" if sensor == "lidar_01" else "lidar_01"
            if other in latest:
                delta = bag_time - latest[other]["bag_time"]
                if abs(delta) <= sync_window_sec:
                    pair = {
                        sensor: entry,
                        other: latest[other],
                        "sync_delta_sec": delta,
                    }
                    yield pair
                    emitted += 1
                    latest.clear()
                    if limit is not None and emitted >= limit:
                        return
                    continue

            latest[sensor] = entry


def run_bag_ego_motion(
    bag_path: Path,
    config_dir: Path,
    frames: int,
    sync_window_sec: float,
) -> List[Dict[str, Any]]:
    """Run RTGBEVNode on synchronized bag frames and return ego-motion rows."""
    from nodes.rtg_bev_node import RTGBEVNode
    from tools._analyze_new_bag import DEFAULT_TOPICS

    node = RTGBEVNode(config_dir=str(config_dir))
    rows: List[Dict[str, Any]] = []

    for index, pair in enumerate(
        iter_synced_lidar_pairs(bag_path, DEFAULT_TOPICS, sync_window_sec, frames)
    ):
        timestamp = pair["lidar_01"]["bag_time"]
        output = node.process_frame(
            {
                "timestamp": timestamp,
                "lidar_01": pair["lidar_01"]["msg"],
                "lidar_02": pair["lidar_02"]["msg"],
            }
        )
        ego = output["ego_motion"]
        rows.append(
            {
                "frame": index,
                "timestamp": round(float(timestamp), 6),
                "sync_delta_sec": round(float(pair["sync_delta_sec"]), 6),
                "state": ego["state_name"],
                "frame_displacement": round(float(ego["frame_displacement"]), 4),
                "fitness": round(float(ego["icp_fitness"]), 3),
                "num_static_points": int(ego["num_static_points"]),
                "valid": bool(ego["valid"]),
                "latency_ms": round(float(output["latency_ms"]), 2),
            }
        )

    return rows


def print_rows(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No synchronized lidar_01/lidar_02 frames found.")
        return

    print("frame  sync_ms  state     dx_m     fit  static_pts  latency_ms")
    for row in rows:
        print(
            f"{row['frame']:5d}  "
            f"{row['sync_delta_sec'] * 1000:7.2f}  "
            f"{row['state']:<8}  "
            f"{row['frame_displacement']:>7.4f}  "
            f"{row['fitness']:>4.2f}  "
            f"{row['num_static_points']:10d}  "
            f"{row['latency_ms']:10.2f}"
        )


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run RTG ego-motion smoke test on real L1/L2 bag frames."
    )
    parser.add_argument("bag_path", help="Path to ROS1 .bag file.")
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parents[1] / "config"),
        help="RTG config directory.",
    )
    parser.add_argument("--frames", type=int, default=10, help="Number of pairs to process.")
    parser.add_argument(
        "--sync-window-sec",
        type=float,
        default=0.05,
        help="Maximum L1/L2 pairing delta.",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    args = parser.parse_args(argv)

    bag_path = Path(args.bag_path)
    if not bag_path.is_file():
        raise SystemExit(f"Bag file not found: {bag_path}")

    rows = run_bag_ego_motion(
        bag_path=bag_path,
        config_dir=Path(args.config_dir),
        frames=max(args.frames, 1),
        sync_window_sec=max(args.sync_window_sec, 0.0),
    )
    print_rows(rows)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written to: {output}")


if __name__ == "__main__":
    main()
