#!/usr/bin/env python3
"""Analyze RTG rosbag point cloud and camera formats.

This script is intentionally lightweight: it reads bag metadata plus a small
number of messages per topic, then reports PointCloud2 field layouts, point
counts, intensity source, coordinate ranges, and timestamp mismatches.

It prefers the pure-Python ``rosbags`` package so it can run outside a ROS1
runtime. A ROS1 ``rosbag`` environment is still fine for extraction tools.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

DEFAULT_TOPICS = {
    "camera_01": "/Camera_Raw_Img_01",
    "camera_02": "/Camera_Raw_Img_02",
    "camera_03": "/Camera_Raw_Img_03",
    "camera_04": "/Camera_Raw_Img_04",
    "lidar_01": "/ouster1/points",
    "lidar_02": "/ouster2/points",
    "lidar_03": "/lidar3/rslidar3_points",
    "lidar_04": "/lidar4/rslidar4_points",
}

POINTFIELD_DTYPES = {
    1: "int8",
    2: "uint8",
    3: "int16",
    4: "uint16",
    5: "int32",
    6: "uint32",
    7: "float32",
    8: "float64",
}


def _numpy_dtype(datatype: int, is_bigendian: bool) -> np.dtype:
    endian = ">" if is_bigendian else "<"
    mapping = {
        1: np.dtype("i1"),
        2: np.dtype("u1"),
        3: np.dtype(endian + "i2"),
        4: np.dtype(endian + "u2"),
        5: np.dtype(endian + "i4"),
        6: np.dtype(endian + "u4"),
        7: np.dtype(endian + "f4"),
        8: np.dtype(endian + "f8"),
    }
    if datatype not in mapping:
        raise ValueError(f"Unsupported PointField datatype: {datatype}")
    return mapping[datatype]


def _field_array(msg: Any, field_name: str) -> np.ndarray:
    fields = {field.name: field for field in msg.fields}
    field = fields[field_name]
    width = int(msg.width)
    height = int(msg.height)
    row_step = int(getattr(msg, "row_step", width * int(msg.point_step)))
    data = msg.data.tobytes() if isinstance(msg.data, np.ndarray) else bytes(msg.data)
    return np.ndarray(
        shape=(height, width),
        dtype=_numpy_dtype(field.datatype, bool(msg.is_bigendian)),
        buffer=data,
        offset=int(field.offset),
        strides=(row_step, int(msg.point_step)),
    ).reshape(-1)


def _stamp_to_float(stamp: Any) -> float:
    sec = getattr(stamp, "sec", 0)
    nsec = getattr(stamp, "nanosec", getattr(stamp, "nsec", 0))
    return float(sec) + float(nsec) / 1e9


def _summarize_pointcloud(msg: Any, bag_time: float) -> Dict[str, Any]:
    fields = {field.name: field for field in msg.fields}
    field_summary = [
        {
            "name": field.name,
            "offset": int(field.offset),
            "datatype": POINTFIELD_DTYPES.get(field.datatype, str(field.datatype)),
            "count": int(field.count),
        }
        for field in msg.fields
    ]

    x = _field_array(msg, "x").astype(np.float32)
    y = _field_array(msg, "y").astype(np.float32)
    z = _field_array(msg, "z").astype(np.float32)
    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

    intensity_source = None
    intensity_stats: Dict[str, Optional[float]] = {"mean": None, "p95": None}
    if "intensity" in fields:
        intensity_source = "intensity"
    elif "reflectivity" in fields:
        intensity_source = "reflectivity"
    if intensity_source is not None and finite.any():
        intensity = _field_array(msg, intensity_source).astype(np.float32)
        intensity_stats = {
            "mean": round(float(np.nanmean(intensity[finite])), 3),
            "p95": round(float(np.nanpercentile(intensity[finite], 95)), 3),
        }

    timestamp_source = None
    if "timestamp" in fields:
        timestamp_source = "timestamp"
    elif "t" in fields:
        timestamp_source = "t"

    header_time = _stamp_to_float(msg.header.stamp)
    ranges = {}
    if finite.any():
        ranges = {
            "x": [round(float(np.nanmin(x[finite])), 3), round(float(np.nanmax(x[finite])), 3)],
            "y": [round(float(np.nanmin(y[finite])), 3), round(float(np.nanmax(y[finite])), 3)],
            "z": [round(float(np.nanmin(z[finite])), 3), round(float(np.nanmax(z[finite])), 3)],
        }

    return {
        "frame_id": str(msg.header.frame_id),
        "header_time": round(header_time, 6),
        "bag_time": round(float(bag_time), 6),
        "bag_minus_header_sec": round(float(bag_time - header_time), 6),
        "height": int(msg.height),
        "width": int(msg.width),
        "raw_points": int(msg.height) * int(msg.width),
        "finite_xyz_points": int(finite.sum()),
        "point_step": int(msg.point_step),
        "row_step": int(msg.row_step),
        "is_dense": bool(msg.is_dense),
        "fields": field_summary,
        "intensity_source": intensity_source,
        "timestamp_source": timestamp_source,
        "intensity": intensity_stats,
        "ranges": ranges,
    }


def _pointcloud_to_numpy(msg: Any) -> np.ndarray:
    fields = {field.name: field for field in msg.fields}
    if not all(name in fields for name in ("x", "y", "z")):
        return np.zeros((0, 5), dtype=np.float32)

    n_points = int(msg.height) * int(msg.width)
    points = np.zeros((n_points, 5), dtype=np.float32)
    points[:, 0] = _field_array(msg, "x").astype(np.float32)
    points[:, 1] = _field_array(msg, "y").astype(np.float32)
    points[:, 2] = _field_array(msg, "z").astype(np.float32)

    if "intensity" in fields:
        points[:, 3] = _field_array(msg, "intensity").astype(np.float32)
    elif "reflectivity" in fields:
        points[:, 3] = _field_array(msg, "reflectivity").astype(np.float32)

    if "timestamp" in fields:
        points[:, 4] = _field_array(msg, "timestamp").astype(np.float32)
    elif "t" in fields:
        points[:, 4] = _field_array(msg, "t").astype(np.float32)

    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1]) & np.isfinite(points[:, 2])
    return points[finite]


def _summarize_image(msg: Any, bag_time: float) -> Dict[str, Any]:
    header_time = _stamp_to_float(msg.header.stamp)
    return {
        "frame_id": str(msg.header.frame_id),
        "header_time": round(header_time, 6),
        "bag_time": round(float(bag_time), 6),
        "bag_minus_header_sec": round(float(bag_time - header_time), 6),
        "height": int(msg.height),
        "width": int(msg.width),
        "encoding": str(msg.encoding),
        "step": int(msg.step),
        "data_bytes": len(msg.data),
    }


def _mean(values: Iterable[float]) -> Optional[float]:
    values = list(values)
    if not values:
        return None
    return round(float(np.mean(values)), 3)


def analyze_with_rosbags(
    bag_path: Path,
    topics: Dict[str, str],
    sample_messages: int,
) -> Dict[str, Any]:
    try:
        from rosbags.rosbag1 import Reader
        from rosbags.typesys import Stores, get_typestore
    except ImportError as exc:
        raise RuntimeError(
            "The pure-Python 'rosbags' package is required outside ROS1. "
            "Install it with: python -m pip install rosbags"
        ) from exc

    typestore = get_typestore(Stores.ROS1_NOETIC)
    wanted_topics = set(topics.values())
    samples: Dict[str, List[Dict[str, Any]]] = {name: [] for name in topics}

    with Reader(bag_path) as reader:
        duration = (reader.end_time - reader.start_time) / 1e9
        connection_by_topic = {conn.topic: conn for conn in reader.connections}
        metadata = {
            "bag_path": str(bag_path),
            "duration_sec": round(float(duration), 3),
            "connections": [
                {
                    "topic": conn.topic,
                    "msgtype": conn.msgtype,
                    "message_count": int(conn.msgcount),
                    "frequency_hz": round(float(conn.msgcount) / max(duration, 1e-6), 3),
                }
                for conn in reader.connections
            ],
        }

        connections = [
            connection_by_topic[topic]
            for topic in wanted_topics
            if topic in connection_by_topic
        ]
        active_sensor_names = [
            name for name, topic in topics.items()
            if topic in connection_by_topic
        ]

        for conn, timestamp, rawdata in reader.messages(connections=connections):
            sensor_name = next(name for name, topic in topics.items() if topic == conn.topic)
            if len(samples[sensor_name]) >= sample_messages:
                continue

            msg = typestore.deserialize_ros1(rawdata, conn.msgtype)
            bag_time = float(timestamp) / 1e9
            if conn.msgtype.endswith("PointCloud2"):
                samples[sensor_name].append(_summarize_pointcloud(msg, bag_time))
            elif conn.msgtype.endswith("Image"):
                samples[sensor_name].append(_summarize_image(msg, bag_time))

            if all(len(samples[name]) >= sample_messages for name in active_sensor_names):
                break

    summary = _build_summary(metadata, samples)
    return {"metadata": metadata, "samples": samples, "summary": summary}


def _build_summary(metadata: Dict[str, Any], samples: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {}
    counts_by_topic = {item["topic"]: item["message_count"] for item in metadata["connections"]}

    summary["message_counts"] = counts_by_topic
    summary["pointclouds"] = {}
    summary["cameras"] = {}

    for sensor, items in samples.items():
        if not items:
            continue
        first = items[0]
        if "raw_points" in first:
            deltas = [abs(item["bag_minus_header_sec"]) for item in items]
            summary["pointclouds"][sensor] = {
                "topic_points": first["raw_points"],
                "finite_xyz_mean": _mean(item["finite_xyz_points"] for item in items),
                "intensity_source": first["intensity_source"],
                "timestamp_source": first["timestamp_source"],
                "bag_header_delta_first_sec": first["bag_minus_header_sec"],
                "uses_device_header_time": bool(max(deltas) > 1000),
                "fields": [field["name"] for field in first["fields"]],
                "ranges": first["ranges"],
            }
        elif "encoding" in first:
            summary["cameras"][sensor] = {
                "resolution": [first["height"], first["width"]],
                "encoding": first["encoding"],
                "bag_header_delta_first_sec": first["bag_minus_header_sec"],
            }

    return summary


def build_lidar_merge_sample(
    bag_path: Path,
    topics: Dict[str, str],
    config_dir: Path,
    bev_output: Path,
    sync_window_sec: float = 0.05,
    max_points_per_lidar: int = 40000,
) -> Dict[str, Any]:
    pair = _read_synced_lidar_pair(bag_path, topics, sync_window_sec)
    l1_points = _pointcloud_to_numpy(pair["lidar_01"]["msg"])
    l2_points = _pointcloud_to_numpy(pair["lidar_02"]["msg"])
    l2_to_l1 = _load_l2_to_l1_transform(config_dir)
    l2_bev = _transform_points(l2_points, l2_to_l1)
    merged = np.concatenate([l1_points, l2_bev], axis=0)

    _save_lidar_merge_bev(bev_output, l1_points, l2_bev, max_points_per_lidar)

    return {
        "bev_output": str(bev_output),
        "sync_window_sec": round(float(sync_window_sec), 6),
        "sync_delta_sec": round(float(pair["sync_delta_sec"]), 6),
        "synced": bool(abs(pair["sync_delta_sec"]) <= sync_window_sec),
        "l1_bag_time": round(float(pair["lidar_01"]["bag_time"]), 6),
        "l2_bag_time": round(float(pair["lidar_02"]["bag_time"]), 6),
        "l1": _summarize_points(l1_points),
        "l2_transformed_to_bev": _summarize_points(l2_bev),
        "merged": _summarize_points(merged),
        "l2_to_l1_translation": [round(float(v), 3) for v in l2_to_l1[:3, 3]],
    }


def _read_synced_lidar_pair(
    bag_path: Path,
    topics: Dict[str, str],
    sync_window_sec: float,
) -> Dict[str, Any]:
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
    first: Dict[str, Dict[str, Any]] = {}

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
            first.setdefault(sensor, entry)

            other = "lidar_02" if sensor == "lidar_01" else "lidar_01"
            if other in latest:
                delta = bag_time - latest[other]["bag_time"]
                if abs(delta) <= sync_window_sec:
                    return {
                        sensor: entry,
                        other: latest[other],
                        "sync_delta_sec": delta,
                    }
            latest[sensor] = entry

    if "lidar_01" in first and "lidar_02" in first:
        delta = first["lidar_01"]["bag_time"] - first["lidar_02"]["bag_time"]
        return {
            "lidar_01": first["lidar_01"],
            "lidar_02": first["lidar_02"],
            "sync_delta_sec": delta,
        }
    raise RuntimeError("Could not read a lidar_01/lidar_02 sample pair from the bag.")


def _load_l2_to_l1_transform(config_dir: Path) -> np.ndarray:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read RTG calibration files.") from exc

    candidates = [
        config_dir / "calib.yaml",
        config_dir / "calib_from_bag.yaml",
        config_dir / "calib_synthetic.yaml",
    ]
    calib_path = next((path for path in candidates if path.is_file()), None)
    if calib_path is None:
        raise RuntimeError(f"No calibration file found under: {config_dir}")

    calib = yaml.safe_load(calib_path.read_text(encoding="utf-8")) or {}
    extrinsics = calib.get("extrinsics", {})
    l2_to_l1 = extrinsics.get("LIDAR_RE_to_LIDAR_FR", {})

    transform = np.eye(4, dtype=np.float32)
    transform[:3, :3] = np.array(l2_to_l1.get("R", np.eye(3)), dtype=np.float32)
    transform[:3, 3] = np.array(l2_to_l1.get("T", [0.0, 0.0, 0.0]), dtype=np.float32)
    return transform


def _transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if points.shape[0] == 0:
        return points.copy()
    transformed = points.copy()
    homogeneous = np.ones((points.shape[0], 4), dtype=np.float32)
    homogeneous[:, :3] = points[:, :3]
    transformed[:, :3] = (homogeneous @ transform.T)[:, :3]
    return transformed


def _summarize_points(points: np.ndarray) -> Dict[str, Any]:
    if points.shape[0] == 0:
        return {"points": 0, "ranges": {}}
    return {
        "points": int(points.shape[0]),
        "ranges": {
            "x": [round(float(np.min(points[:, 0])), 3), round(float(np.max(points[:, 0])), 3)],
            "y": [round(float(np.min(points[:, 1])), 3), round(float(np.max(points[:, 1])), 3)],
            "z": [round(float(np.min(points[:, 2])), 3), round(float(np.max(points[:, 2])), 3)],
        },
        "intensity_mean": round(float(np.mean(points[:, 3])), 3),
    }


def _save_lidar_merge_bev(
    output_path: Path,
    l1_points: np.ndarray,
    l2_bev_points: np.ndarray,
    max_points_per_lidar: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    try:
        from tools._draw_tracking import draw_bev_scene_background

        draw_bev_scene_background(ax, draw_labels=False)
    except Exception:
        ax.set_xlim(-40, 80)
        ax.set_ylim(-30, 15)
        ax.grid(True, linewidth=0.3, alpha=0.4)

    for label, color, points in (
        ("L1", "#2563eb", l1_points),
        ("L2->BEV", "#dc2626", l2_bev_points),
    ):
        sampled = _sample_points_for_plot(points, max_points_per_lidar)
        if sampled.shape[0] == 0:
            continue
        ax.scatter(
            sampled[:, 0],
            sampled[:, 1],
            s=0.2,
            c=color,
            alpha=0.35,
            linewidths=0,
            label=label,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.legend(loc="upper right", markerscale=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _sample_points_for_plot(points: np.ndarray, max_points: int) -> np.ndarray:
    if points.shape[0] <= max_points:
        return points
    stride = max(1, int(np.ceil(points.shape[0] / max_points)))
    return points[::stride][:max_points]


def _parse_topic_override(values: Optional[List[str]]) -> Dict[str, str]:
    topics = dict(DEFAULT_TOPICS)
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid topic override '{value}', expected name=/topic")
        name, topic = value.split("=", 1)
        if name not in topics:
            raise ValueError(f"Unknown topic alias '{name}'. Valid aliases: {sorted(topics)}")
        topics[name] = topic
    return topics


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze RTG rosbag point cloud formats.")
    parser.add_argument("bag_path", help="Path to ROS1 .bag file.")
    parser.add_argument(
        "--sample-messages",
        type=int,
        default=5,
        help="Number of messages to sample per topic (default: 5).",
    )
    parser.add_argument(
        "--topic",
        action="append",
        help="Override topic alias, e.g. --topic lidar_01=/ouster1/points",
    )
    parser.add_argument("--output", help="Optional JSON report path.")
    parser.add_argument(
        "--bev-output",
        help="Optional PNG path for a real L1/L2 merged BEV sample.",
    )
    parser.add_argument(
        "--config-dir",
        default=str(Path(__file__).resolve().parents[1] / "config"),
        help="RTG config directory used for L2->L1 calibration.",
    )
    parser.add_argument(
        "--sync-window-sec",
        type=float,
        default=0.05,
        help="L1/L2 pairing window for the BEV sample (default: 0.05).",
    )
    parser.add_argument(
        "--max-points-per-lidar",
        type=int,
        default=40000,
        help="Maximum plotted points per lidar for --bev-output.",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_path)
    if not bag_path.is_file():
        raise SystemExit(f"Bag file not found: {bag_path}")

    topics = _parse_topic_override(args.topic)
    report = analyze_with_rosbags(bag_path, topics, max(args.sample_messages, 1))
    if args.bev_output:
        report["summary"]["lidar_pair_merge_sample"] = build_lidar_merge_sample(
            bag_path=bag_path,
            topics=topics,
            config_dir=Path(args.config_dir),
            bev_output=Path(args.bev_output),
            sync_window_sec=max(args.sync_window_sec, 0.0),
            max_points_per_lidar=max(args.max_points_per_lidar, 1),
        )

    text = json.dumps(report["summary"], ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nReport written to: {output}")


if __name__ == "__main__":
    main()
