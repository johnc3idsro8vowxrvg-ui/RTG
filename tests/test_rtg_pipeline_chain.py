import shutil
import struct
from pathlib import Path

import numpy as np
import pytest
import yaml


def _copy_runtime_configs(dst: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "config"
    for name in ("geometry.yaml", "system.yaml", "warning.yaml", "calib_from_bag.yaml"):
        shutil.copy(config_dir / name, dst / name)


def test_config_loader_uses_committed_calib_fallback(tmp_path):
    from postprocessing.config_loader import ConfigLoader

    _copy_runtime_configs(tmp_path)

    loader = ConfigLoader(str(tmp_path))
    loader.load_all()

    assert "extrinsics" in loader.calib
    assert "LIDAR_RE_to_LIDAR_FR" in loader.calib["extrinsics"]


def test_numpy_lidar_inputs_are_normalized_to_five_columns():
    from nodes.rtg_bev_node import _pointcloud2_to_numpy

    points = np.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=np.float32,
    )

    decoded = _pointcloud2_to_numpy(points)

    assert decoded.shape == (2, 5)
    np.testing.assert_allclose(decoded[:, :3], points)
    np.testing.assert_allclose(decoded[:, 3:], 0.0)


def test_pointcloud2_decode_respects_row_step_padding():
    from nodes.rtg_bev_node import _pointcloud2_to_numpy

    class Field:
        datatype = 7

        def __init__(self, name, offset):
            self.name = name
            self.offset = offset

    class Msg:
        width = 2
        height = 2
        point_step = 16
        row_step = 40
        is_bigendian = False
        fields = [
            Field("x", 0),
            Field("y", 4),
            Field("z", 8),
            Field("reflectivity", 12),
        ]

    msg = Msg()
    raw = bytearray(msg.height * msg.row_step)
    rows = [
        [(1.0, 2.0, 3.0, 10.0), (4.0, 5.0, 6.0, 11.0)],
        [(7.0, 8.0, 9.0, 12.0), (10.0, 11.0, 12.0, 13.0)],
    ]
    for row_idx, row in enumerate(rows):
        for col_idx, values in enumerate(row):
            base = row_idx * msg.row_step + col_idx * msg.point_step
            for field_idx, value in enumerate(values):
                struct.pack_into("<f", raw, base + field_idx * 4, value)
    msg.data = bytes(raw)

    decoded = _pointcloud2_to_numpy(msg)

    assert decoded.shape == (4, 5)
    np.testing.assert_allclose(
        decoded[:, :4],
        np.array([point for row in rows for point in row], dtype=np.float32),
    )
    np.testing.assert_allclose(decoded[:, 4], 0.0)


def test_sensor_buffer_sync_requires_l1_l2_not_camera():
    from nodes.rtg_bev_node import SensorBuffer

    buffer = SensorBuffer(maxlen=10)
    buffer.add("front", 10.00, "lidar_01")
    buffer.add("rear", 10.02, "lidar_02")

    frame = buffer.get_synced_frame(window=0.05)

    assert frame is not None
    assert frame["lidar_01"] == "front"
    assert frame["lidar_02"] == "rear"


def test_sensor_buffer_uses_lidar_pair_time_for_output_timestamp():
    from nodes.rtg_bev_node import SensorBuffer

    buffer = SensorBuffer(maxlen=10)
    buffer.add("front", 10.00, "lidar_01")
    buffer.add("rear", 10.02, "lidar_02")
    buffer.add("camera", 10.04, "camera_01")

    frame = buffer.get_synced_frame(window=0.05)

    assert frame is not None
    assert frame["timestamp"] == 10.02

def test_sensor_buffer_does_not_emit_same_lidar_pair_twice():
    from nodes.rtg_bev_node import SensorBuffer

    buffer = SensorBuffer(maxlen=10)
    buffer.add("front-1", 10.00, "lidar_01")
    buffer.add("rear-1", 10.02, "lidar_02")

    assert buffer.get_synced_frame(window=0.05) is not None
    assert buffer.get_synced_frame(window=0.05) is None

    buffer.add("camera-only", 10.03, "camera_01")
    assert buffer.get_synced_frame(window=0.05) is None

    buffer.add("front-2", 10.10, "lidar_01")
    buffer.add("rear-2", 10.12, "lidar_02")
    frame = buffer.get_synced_frame(window=0.05)

    assert frame is not None
    assert frame["lidar_01"] == "front-2"
    assert frame["lidar_02"] == "rear-2"


def test_extract_timestamp_rejects_device_internal_time(monkeypatch):
    import nodes.rtg_bev_node as node_mod

    class Stamp:
        def to_sec(self):
            return 3113627.857

    class Header:
        stamp = Stamp()

    class Msg:
        header = Header()

    monkeypatch.setattr(node_mod.time, "time", lambda: 1779176674.8)

    assert node_mod._extract_timestamp(Msg()) == 1779176674.8


def test_preprocess_lidar_merges_l1_l2_arrays_in_bev_frame(tmp_path, monkeypatch):
    from nodes.rtg_bev_node import RTGBEVNode

    _copy_runtime_configs(tmp_path)
    monkeypatch.setattr(RTGBEVNode, "_load_model", lambda self: None)

    node = RTGBEVNode(config_dir=str(tmp_path))
    frame = {
        "lidar_01": np.array([[20.0, 5.0, 0.2, 0.4, 0.01]], dtype=np.float32),
        "lidar_02": np.array([[5.0, 3.0, 0.3, 0.7, 0.02]], dtype=np.float32),
    }

    merged = node._preprocess_lidar(frame)

    assert merged.shape == (2, 5)
    np.testing.assert_allclose(merged[0], [20.0, 5.0, 0.2, 0.4, 0.01])
    np.testing.assert_allclose(merged[1], [-7.0, 3.0, 0.3, 0.7, 0.02])


def test_preprocess_lidar_does_not_mutate_source_arrays(tmp_path, monkeypatch):
    from nodes.rtg_bev_node import RTGBEVNode

    _copy_runtime_configs(tmp_path)
    monkeypatch.setattr(RTGBEVNode, "_load_model", lambda self: None)

    node = RTGBEVNode(config_dir=str(tmp_path))
    lidar_01 = np.array([[20.0, 5.0, 0.2, 0.4, 0.01]], dtype=np.float32)
    lidar_02 = np.array([[5.0, 3.0, 0.3, 0.7, 0.02]], dtype=np.float32)
    lidar_01_before = lidar_01.copy()
    lidar_02_before = lidar_02.copy()

    node._preprocess_lidar({"lidar_01": lidar_01, "lidar_02": lidar_02})

    np.testing.assert_allclose(lidar_01, lidar_01_before)
    np.testing.assert_allclose(lidar_02, lidar_02_before)

def test_preprocess_lidar_returns_empty_five_column_cloud(tmp_path, monkeypatch):
    from nodes.rtg_bev_node import RTGBEVNode

    _copy_runtime_configs(tmp_path)
    monkeypatch.setattr(RTGBEVNode, "_load_model", lambda self: None)

    node = RTGBEVNode(config_dir=str(tmp_path))

    merged = node._preprocess_lidar({})

    assert merged.shape == (0, 5)


def test_bev_tracking_background_uses_negative_forbidden_side():
    import matplotlib.pyplot as plt

    from tools._draw_tracking import draw_bev_scene_background

    fig, ax = plt.subplots()
    draw_bev_scene_background(ax, draw_labels=False)

    rectangle_y_values = [
        patch.get_y()
        for patch in ax.patches
        if hasattr(patch, "get_y") and hasattr(patch, "get_height")
    ]
    plt.close(fig)

    assert any(y <= -24.0 for y in rectangle_y_values)


def test_rosbag_extractor_uses_instance_sync_window():
    from tools.data_converter.rosbag_extract import RosbagExtractor

    extractor = object.__new__(RosbagExtractor)
    extractor.sync_window_sec = 0.2

    match = extractor._find_closest([(10.15, "msg")], 10.0, set())

    assert match == (10.15, "msg")


def test_bag_analysis_transform_points_keeps_extra_columns():
    from tools._analyze_new_bag import _transform_points

    points = np.array([[1.0, 2.0, 3.0, 9.0, 10.0]], dtype=np.float32)
    transform = np.eye(4, dtype=np.float32)
    transform[:3, 3] = [-12.0, 0.5, 1.0]

    transformed = _transform_points(points, transform)

    np.testing.assert_allclose(transformed[0], [-11.0, 2.5, 4.0, 9.0, 10.0])


def test_bag_analysis_pointcloud_decode_respects_row_step_padding():
    from tools._analyze_new_bag import _pointcloud_to_numpy

    class Field:
        datatype = 7

        def __init__(self, name, offset):
            self.name = name
            self.offset = offset
            self.count = 1

    class Msg:
        width = 2
        height = 2
        point_step = 16
        row_step = 40
        is_bigendian = False
        is_dense = True
        fields = [
            Field("x", 0),
            Field("y", 4),
            Field("z", 8),
            Field("intensity", 12),
        ]

    msg = Msg()
    raw = bytearray(msg.height * msg.row_step)
    rows = [
        [(1.0, 2.0, 3.0, 0.1), (4.0, 5.0, 6.0, 0.2)],
        [(7.0, 8.0, 9.0, 0.3), (10.0, 11.0, 12.0, 0.4)],
    ]
    for row_idx, row in enumerate(rows):
        for col_idx, values in enumerate(row):
            base = row_idx * msg.row_step + col_idx * msg.point_step
            for field_idx, value in enumerate(values):
                struct.pack_into("<f", raw, base + field_idx * 4, value)
    msg.data = bytes(raw)

    decoded = _pointcloud_to_numpy(msg)

    assert decoded.shape == (4, 5)
    np.testing.assert_allclose(
        decoded[:, :4],
        np.array([point for row in rows for point in row], dtype=np.float32),
    )
    np.testing.assert_allclose(decoded[:, 4], 0.0)


def test_tracker_empty_detection_update_counts_one_miss():
    from postprocessing.tracker import Tracker

    tracker = Tracker(
        {
            "min_hits_confirm": 1,
            "publish_internal_tracks": True,
            "max_age_lost": 5,
        }
    )
    det = {
        "class_id": 0,
        "confidence": 0.9,
        "x": 1.0,
        "y": 2.0,
        "z": 0.5,
        "w": 0.8,
        "l": 0.8,
        "h": 1.7,
        "yaw": 0.0,
    }

    tracker.update([det], timestamp=1.0)
    tracks = tracker.update([], timestamp=1.1)

    assert len(tracks) == 1
    assert tracks[0]["time_since_update"] == 1


def test_tracker_fallback_velocity_uses_timestamp_dt(monkeypatch):
    import postprocessing.tracker as tracker_mod
    from postprocessing.tracker import Tracker

    monkeypatch.setattr(tracker_mod, "HAS_FILTERPY", False)

    tracker = Tracker(
        {
            "min_hits_confirm": 1,
            "publish_internal_tracks": True,
            "iou_threshold": 0.1,
        }
    )
    first = {
        "class_id": 0,
        "confidence": 0.95,
        "x": 0.0,
        "y": 0.0,
        "z": 0.5,
        "w": 1.0,
        "l": 1.0,
        "h": 1.7,
        "yaw": 0.0,
    }
    second = {**first, "x": 0.2}

    tracker.update([first], timestamp=1.0)
    tracks = tracker.update([second], timestamp=1.1)

    assert len(tracks) == 1
    assert tracks[0]["vx"] == pytest.approx(2.0)

def test_tracker_min_hits_confirm_controls_track_state():
    from postprocessing.tracker import Tracker

    tracker = Tracker(
        {
            "min_hits_confirm": 1,
            "publish_internal_tracks": False,
            "max_age_lost": 5,
        }
    )
    det = {
        "class_id": 0,
        "confidence": 0.95,
        "x": 1.0,
        "y": 2.0,
        "z": 0.5,
        "w": 0.8,
        "l": 0.8,
        "h": 1.7,
        "yaw": 0.0,
    }

    tracks = tracker.update([det], timestamp=1.0)

    assert len(tracks) == 1
    assert tracks[0]["state_name"] == "confirmed"
    assert tracks[0]["track_id"] >= 0

def test_tracker_does_not_match_different_classes_at_same_location():
    from postprocessing.tracker import Tracker

    tracker = Tracker(
        {
            "min_hits_confirm": 1,
            "publish_internal_tracks": True,
            "iou_threshold": 0.1,
            "max_age_lost": 5,
        }
    )
    person = {
        "class_id": 0,
        "confidence": 0.95,
        "x": 1.0,
        "y": 2.0,
        "z": 0.5,
        "w": 0.8,
        "l": 0.8,
        "h": 1.7,
        "yaw": 0.0,
    }
    truck = {**person, "class_id": 1}

    first = tracker.update([person], timestamp=1.0)
    person_id = first[0]["track_id"]
    second = tracker.update([truck], timestamp=1.1)

    assert len(second) == 2
    by_class = {track["class_id"]: track for track in second}
    assert by_class[0]["track_id"] == person_id
    assert by_class[0]["time_since_update"] == 1
    assert by_class[1]["track_id"] != person_id
    assert by_class[1]["hits"] == 1

def test_generate_infos_concat_accepts_legacy_four_column_bins(tmp_path):
    from tools.data_converter.generate_rtg_infos import concat_lidar_points

    front = np.array([[1.0, 2.0, 3.0, 0.5]], dtype=np.float32)
    rear = np.array([[4.0, 5.0, 6.0, 0.7]], dtype=np.float32)
    front_path = tmp_path / "front.bin"
    rear_path = tmp_path / "rear.bin"
    front.tofile(front_path)
    rear.tofile(rear_path)

    merged = concat_lidar_points(str(front_path), str(rear_path))

    assert merged.shape == (2, 5)
    np.testing.assert_allclose(merged[:, :4], np.vstack([front, rear]))
    np.testing.assert_allclose(merged[:, 4], 0.0)


def test_rtg_dataset_pointcloud_io_loads_four_and_five_column_bins(tmp_path):
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    centerpoint_root = str(root / "CenterPoint")
    if centerpoint_root not in sys.path:
        sys.path.insert(0, centerpoint_root)

    from det3d.datasets.rtg.pointcloud_io import load_lidar_bin_5d

    four_col = np.array([[1.0, 2.0, 3.0, 0.4]], dtype=np.float32)
    five_col = np.array([[4.0, 5.0, 6.0, 0.7, 0.02]], dtype=np.float32)
    four_path = tmp_path / "four.bin"
    five_path = tmp_path / "five.bin"
    four_col.tofile(four_path)
    five_col.tofile(five_path)

    loaded_four = load_lidar_bin_5d(four_path)
    loaded_five = load_lidar_bin_5d(five_path)

    assert loaded_four.shape == (1, 5)
    np.testing.assert_allclose(loaded_four[0], [1.0, 2.0, 3.0, 0.4, 0.0])
    np.testing.assert_allclose(loaded_five, five_col)


def test_ego_motion_bag_tool_import_has_no_side_effects():
    import tools._test_ego_motion as ego_motion_tool

    assert callable(ego_motion_tool.main)


def test_centerpoint_result_parser_preserves_rtg_class_and_velocity_layout():
    from nodes.rtg_bev_node import _centerpoint_results_to_detections

    result = [
        {
            "box3d_lidar": np.array(
                [[1.0, 2.0, 0.5, 0.8, 1.2, 1.7, 0.4, -0.2, 1.57]],
                dtype=np.float32,
            ),
            "scores": np.array([0.9], dtype=np.float32),
            "label_preds": np.array([0], dtype=np.int64),
        }
    ]

    detections = _centerpoint_results_to_detections(
        result,
        score_thr=0.1,
        class_names=["person", "truck", "car", "other_obstacle"],
    )

    assert detections[0]["class_id"] == 0
    assert detections[0]["class_name"] == "person"
    assert detections[0]["yaw"] == np.float32(1.57)
    assert detections[0]["vx"] == np.float32(0.4)
    assert detections[0]["vy"] == np.float32(-0.2)


def test_centerpoint_result_parser_maps_nuscenes_pedestrian_label():
    from nodes.rtg_bev_node import NUSC_CLASSES, _centerpoint_results_to_detections

    result = [
        {
            "box3d_lidar": np.array(
                [[1.0, 2.0, 0.5, 0.8, 1.2, 1.7, 0.25]],
                dtype=np.float32,
            ),
            "scores": np.array([0.9], dtype=np.float32),
            "label_preds": np.array([8], dtype=np.int64),
        }
    ]

    detections = _centerpoint_results_to_detections(
        result,
        score_thr=0.1,
        class_names=NUSC_CLASSES,
    )

    assert detections[0]["class_id"] == 0
    assert detections[0]["class_name"] == "person"
    assert detections[0]["yaw"] == np.float32(0.25)
    assert detections[0]["vx"] == 0.0
    assert detections[0]["vy"] == 0.0


def test_centerpoint_result_parser_defaults_to_nuscenes_labels():
    from nodes.rtg_bev_node import _centerpoint_results_to_detections

    result = [
        {
            "box3d_lidar": np.array(
                [[1.0, 2.0, 0.5, 1.8, 4.2, 1.6, 0.0]],
                dtype=np.float32,
            ),
            "scores": np.array([0.9], dtype=np.float32),
            "label_preds": np.array([0], dtype=np.int64),
        }
    ]

    detections = _centerpoint_results_to_detections(result, score_thr=0.1)

    assert detections[0]["class_id"] == 2
    assert detections[0]["class_name"] == "car"

def test_centerpoint_coordinates_include_batch_index():
    from nodes.rtg_bev_node import _add_batch_index_to_coordinates

    coordinates = np.array(
        [
            [3, 2, 1],
            [0, 4, 5],
        ],
        dtype=np.int32,
    )

    batched = _add_batch_index_to_coordinates(coordinates)

    assert batched.dtype == np.int32
    np.testing.assert_array_equal(
        batched,
        np.array(
            [
                [0, 3, 2, 1],
                [0, 0, 4, 5],
            ],
            dtype=np.int32,
        ),
    )


def test_centerpoint_merge_offsets_multitask_labels_without_torch():
    import importlib.util

    root = Path(__file__).resolve().parents[1]
    merge_utils = root / "CenterPoint" / "det3d" / "models" / "bbox_heads" / "merge_utils.py"
    spec = importlib.util.spec_from_file_location("centerpoint_merge_utils", merge_utils)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    rets = [
        [
            {
                "box3d_lidar": np.array([[1.0, 0.0]], dtype=np.float32),
                "scores": np.array([0.8], dtype=np.float32),
                "label_preds": np.array([0], dtype=np.int64),
            }
        ],
        [
            {
                "box3d_lidar": np.array([[2.0, 0.0]], dtype=np.float32),
                "scores": np.array([0.9], dtype=np.float32),
                "label_preds": np.array([1], dtype=np.int64),
            }
        ],
    ]

    merged = module.merge_task_predictions(
        rets,
        metas=[["sample-meta"], ["ignored-meta"]],
        num_classes=[1, 2],
        cat=lambda values: np.concatenate(values, axis=0),
    )

    assert len(merged) == 1
    np.testing.assert_allclose(merged[0]["box3d_lidar"], [[1.0, 0.0], [2.0, 0.0]])
    np.testing.assert_allclose(merged[0]["scores"], [0.8, 0.9])
    np.testing.assert_array_equal(merged[0]["label_preds"], [0, 2])
    assert merged[0]["metadata"] == "sample-meta"

def test_centerpoint_device_selection_honors_cuda_id_and_cpu_fallback():
    from nodes.rtg_bev_node import _resolve_torch_device

    class FakeCuda:
        def __init__(self, available):
            self._available = available

        def is_available(self):
            return self._available

    class FakeTorch:
        def __init__(self, cuda_available):
            self.cuda = FakeCuda(cuda_available)

        def device(self, name):
            return name

    assert _resolve_torch_device({"device_id": 2}, FakeTorch(True)) == "cuda:2"
    assert (
        _resolve_torch_device(
            {"device_id": 2, "allow_fallback_to_cpu": True},
            FakeTorch(False),
        )
        == "cpu"
    )

    with pytest.raises(RuntimeError):
        _resolve_torch_device(
            {"device_id": 2, "allow_fallback_to_cpu": False},
            FakeTorch(False),
        )


def test_centerpoint_import_path_is_inserted_once(tmp_path):
    from nodes.rtg_bev_node import _ensure_import_path

    centerpoint_root = tmp_path / "CenterPoint"
    centerpoint_root.mkdir()
    path_list = [str(tmp_path / "other")]

    _ensure_import_path(str(centerpoint_root), path_list)
    _ensure_import_path(str(centerpoint_root), path_list)

    assert path_list[0] == str(centerpoint_root)
    assert path_list.count(str(centerpoint_root)) == 1


def test_centerpoint_datasets_package_imports():
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    centerpoint_root = str(root / "CenterPoint")
    if centerpoint_root not in sys.path:
        sys.path.insert(0, centerpoint_root)

    from det3d.datasets import DATASETS, build_dataset

    assert DATASETS is not None
    assert callable(build_dataset)


def test_ros_message_builders_cast_numpy_scalars_to_python_types():
    from nodes.rtg_bev_node import (
        _build_detection_array,
        _build_ego_motion_state,
        _build_track_array,
        _build_warning_array,
    )

    det_msg = _build_detection_array(
        [
            {
                "class_id": np.int64(2),
                "class_name": "car",
                "confidence": np.float32(0.9),
                "x": np.float32(1.0),
                "y": np.float32(2.0),
                "z": np.float32(0.5),
                "w": np.float32(1.8),
                "l": np.float32(4.2),
                "h": np.float32(1.6),
                "yaw": np.float32(0.1),
                "vx": np.float32(0.2),
                "vy": np.float32(-0.1),
                "distance": np.float32(3.5),
            }
        ],
        np.float64(123.5),
    )
    det = det_msg.detections[0]
    assert type(det_msg.header.stamp) is float
    assert type(det.id) is int
    assert type(det.class_id) is int
    assert type(det.confidence) is float
    assert type(det.x) is float

    trk_msg = _build_track_array(
        [
            {
                "track_id": np.int64(4),
                "class_id": np.int64(1),
                "age": np.int64(6),
                "x": np.float32(1.0),
                "y": np.float32(2.0),
                "z": np.float32(0.5),
                "vx": np.float32(0.2),
                "vy": np.float32(-0.1),
                "w": np.float32(2.5),
                "l": np.float32(6.0),
                "h": np.float32(2.8),
                "yaw": np.float32(0.2),
                "state": np.int64(1),
            }
        ],
        np.float64(124.0),
    )
    trk = trk_msg.tracks[0]
    assert type(trk.track_id) is int
    assert type(trk.x) is float
    assert type(trk.state) is int

    warning_msg = _build_warning_array(
        {
            "ego_motion": {"state": np.int64(3)},
            "warnings": [
                {
                    "track_id": np.int64(4),
                    "warning_level": np.int64(3),
                    "target_class": np.int64(1),
                    "distance": np.float32(3.0),
                    "zone": "nearest_leg",
                    "trigger_reason": "distance",
                    "trigger_time": np.float64(124.1),
                }
            ],
            "active_zones": [
                {
                    "name": "zone_a",
                    "weight": np.float32(1.0),
                    "y_min": np.float32(-2.0),
                    "y_max": np.float32(2.0),
                    "description": "test",
                }
            ],
        },
        np.float64(124.0),
    )
    warning = warning_msg.warnings[0]
    zone = warning_msg.active_zones[0]
    assert type(warning_msg.ego_motion_state) is int
    assert type(warning.track_id) is int
    assert type(warning.distance) is float
    assert type(warning.trigger_time) is float
    assert type(zone.weight) is float

    ego_msg = _build_ego_motion_state(
        {
            "state": np.int64(2),
            "confidence": np.float32(0.8),
            "displacement": np.float32(1.2),
            "velocity_estimate": np.float32(0.4),
        }
    )
    assert type(ego_msg.state) is int
    assert type(ego_msg.confidence) is float

def test_process_frame_runs_detection_tracking_warning_and_bev_debug(tmp_path, monkeypatch):
    from nodes.rtg_bev_node import RTGBEVNode

    _copy_runtime_configs(tmp_path)
    monkeypatch.setattr(RTGBEVNode, "_load_model", lambda self: None)

    node = RTGBEVNode(config_dir=str(tmp_path))
    node._run_inference = lambda points, images, timestamp: [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "x": 4.0,
            "y": 0.0,
            "z": 0.8,
            "w": 0.8,
            "l": 0.8,
            "h": 1.7,
            "yaw": 0.0,
            "vx": 0.0,
            "vy": 0.0,
        }
    ]
    node._system_cfg.setdefault("debug", {}).setdefault("bev_visualization", {})["enabled"] = True
    node._system_cfg["debug"]["bev_visualization"]["save_every_n_frames"] = 1
    node._system_cfg["debug"]["bev_visualization"]["save_dir"] = str(tmp_path / "bev")

    output = None
    for i in range(3):
        output = node.process_frame(
            {
                "timestamp": 100.0 + i * 0.1,
                "lidar_01": np.array(
                    [[20.0 + i, 5.0, 0.2, 0.5, 0.0], [30.0, -10.0, 1.0, 0.2, 0.0]],
                    dtype=np.float32,
                ),
                "lidar_02": np.array([[10.0, 4.0, 0.2, 0.7, 0.0]], dtype=np.float32),
            }
        )

    assert output is not None
    assert output["detections"][0]["distance"] == 4.0
    assert len(output["tracks"]) == 1
    assert output["warnings"][0]["warning_level"] == 3
    assert len(list((tmp_path / "bev").glob("*.png"))) == 3


def test_node_uses_tracking_config_from_system_yaml(tmp_path, monkeypatch):
    from nodes.rtg_bev_node import RTGBEVNode

    _copy_runtime_configs(tmp_path)
    system_path = tmp_path / "system.yaml"
    system_cfg = yaml.safe_load(system_path.read_text(encoding="utf-8"))
    system_cfg["tracking"] = {
        "iou_threshold": 0.1,
        "max_age_lost": 5,
        "min_hits_confirm": 1,
        "min_confidence": 0.2,
        "publish_internal_tracks": True,
    }
    system_path.write_text(yaml.safe_dump(system_cfg, allow_unicode=True), encoding="utf-8")

    monkeypatch.setattr(RTGBEVNode, "_load_model", lambda self: None)
    node = RTGBEVNode(config_dir=str(tmp_path))
    node._run_inference = lambda points, images, timestamp: [
        {
            "class_id": 0,
            "class_name": "person",
            "confidence": 0.95,
            "x": 4.0,
            "y": 0.0,
            "z": 0.8,
            "w": 0.8,
            "l": 0.8,
            "h": 1.7,
            "yaw": 0.0,
        }
    ]

    output = node.process_frame(
        {
            "timestamp": 100.0,
            "lidar_01": np.array([[20.0, 5.0, 0.2, 0.5, 0.0]], dtype=np.float32),
            "lidar_02": np.array([[10.0, 4.0, 0.2, 0.7, 0.0]], dtype=np.float32),
        }
    )

    assert len(output["tracks"]) == 1
    assert output["tracks"][0]["state_name"] == "confirmed"
    assert output["tracks"][0]["track_id"] >= 0

def test_warning_immediate_danger_overrides_confirmation(tmp_path):
    from postprocessing.config_loader import ConfigLoader
    from postprocessing.constants import EgoMotionState, WarningLevel
    from postprocessing.warning_engine import WarningEngine

    _copy_runtime_configs(tmp_path)
    warning_path = tmp_path / "warning.yaml"
    warning_cfg = yaml.safe_load(warning_path.read_text(encoding="utf-8"))
    warning_cfg["distance_thresholds"]["person"] = {
        "danger": 4.0,
        "warning": 10.0,
        "info": 20.0,
    }
    warning_cfg["immediate_danger"]["person"] = 8.0
    warning_cfg["frame_confirmation"]["warning_confirm_frames"] = 3
    warning_path.write_text(yaml.safe_dump(warning_cfg, allow_unicode=True), encoding="utf-8")

    loader = ConfigLoader(str(tmp_path))
    loader.load_all()
    engine = WarningEngine(loader)

    result = engine.evaluate(
        [
            {
                "track_id": 7,
                "class_id": 0,
                "confidence": 0.95,
                "x": 6.0,
                "y": 0.0,
            }
        ],
        EgoMotionState.UNKNOWN,
        timestamp=100.0,
    )

    assert len(result["warnings"]) == 1
    assert result["warnings"][0]["warning_level"] == WarningLevel.DANGER

def test_warning_ego_motion_config_disables_static_close_alert(tmp_path):
    from postprocessing.config_loader import ConfigLoader
    from postprocessing.constants import EgoMotionState
    from postprocessing.warning_engine import WarningEngine

    _copy_runtime_configs(tmp_path)
    warning_path = tmp_path / "warning.yaml"
    warning_cfg = yaml.safe_load(warning_path.read_text(encoding="utf-8"))
    warning_cfg["ego_motion"]["static_close_proximity_alert"]["enabled"] = False
    warning_path.write_text(yaml.safe_dump(warning_cfg, allow_unicode=True), encoding="utf-8")

    loader = ConfigLoader(str(tmp_path))
    loader.load_all()
    engine = WarningEngine(loader)

    result = engine.evaluate(
        [
            {
                "track_id": 1,
                "class_id": 0,
                "confidence": 0.95,
                "x": 0.5,
                "y": 0.0,
            }
        ],
        EgoMotionState.STATIC,
        timestamp=100.0,
    )

    assert result["warnings"] == []


def test_warning_ego_motion_config_enables_static_close_alert(tmp_path):
    from postprocessing.config_loader import ConfigLoader
    from postprocessing.constants import EgoMotionState, WarningLevel
    from postprocessing.warning_engine import WarningEngine

    _copy_runtime_configs(tmp_path)
    warning_path = tmp_path / "warning.yaml"
    warning_cfg = yaml.safe_load(warning_path.read_text(encoding="utf-8"))
    warning_cfg["ego_motion"]["static_close_proximity_alert"] = {
        "enabled": True,
        "distance_threshold": 2.0,
        "target_classes": ["person"],
    }
    warning_path.write_text(yaml.safe_dump(warning_cfg, allow_unicode=True), encoding="utf-8")

    loader = ConfigLoader(str(tmp_path))
    loader.load_all()
    engine = WarningEngine(loader)

    result = engine.evaluate(
        [
            {
                "track_id": 1,
                "class_id": 0,
                "confidence": 0.95,
                "x": 0.5,
                "y": 0.0,
            }
        ],
        EgoMotionState.STATIC,
        timestamp=100.0,
    )

    assert len(result["warnings"]) == 1
    assert result["warnings"][0]["warning_level"] == WarningLevel.INFO


def test_ego_motion_dynamic_removal_uses_configured_detection_margin(monkeypatch):
    import postprocessing.ego_motion as ego_motion

    monkeypatch.setattr(ego_motion, "HAS_OPEN3D", False)

    estimator = ego_motion.EgoMotionEstimator(
        {
            "ground_height_range": [-0.2, 0.2],
            "detection_box_margin": 0.0,
        }
    )
    points = np.array(
        [
            [0.4, 0.0, 1.0],
            [0.9, 0.0, 1.0],
            [2.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    detections = [
        {
            "x": 0.0,
            "y": 0.0,
            "w": 1.0,
            "l": 1.0,
            "yaw": 0.0,
        }
    ]

    static = estimator._extract_static_points(points, detections)

    np.testing.assert_allclose(static, [[0.9, 0.0, 1.0], [2.0, 0.0, 1.0]])

def test_ego_motion_numpy_fallback_detects_plus_x_without_open3d(monkeypatch):
    import postprocessing.ego_motion as ego_motion
    from postprocessing.constants import EgoMotionState

    monkeypatch.setattr(ego_motion, "HAS_OPEN3D", False)

    estimator = ego_motion.EgoMotionEstimator(
        {
            "min_points_for_icp": 5,
            "ground_height_range": [-0.2, 0.2],
            "static_displacement_threshold": 0.05,
            "min_fitness_for_valid": 0.3,
            "confirmation_window": 1,
        }
    )
    base = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, -0.5, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.5, 1.0],
            [4.0, 1.0, 1.0],
            [5.0, 1.5, 1.0],
        ],
        dtype=np.float32,
    )

    first = estimator.update(base, timestamp=10.0)
    shift = np.array([0.2, 0.0, 0.0], dtype=np.float32)
    estimator.update(base + shift, timestamp=10.1)
    third = estimator.update(base + shift * 2.0, timestamp=10.2)

    assert first["state"] == EgoMotionState.UNKNOWN
    assert third["valid"] is True
    assert third["state"] == EgoMotionState.MOVING_PLUS_X
    assert third["frame_displacement"] > 0.1
    assert third["icp_fitness"] >= 0.3


def test_ego_motion_numpy_fallback_keeps_small_shift_static(monkeypatch):
    import postprocessing.ego_motion as ego_motion
    from postprocessing.constants import EgoMotionState

    monkeypatch.setattr(ego_motion, "HAS_OPEN3D", False)

    estimator = ego_motion.EgoMotionEstimator(
        {
            "min_points_for_icp": 5,
            "ground_height_range": [-0.2, 0.2],
            "static_displacement_threshold": 0.05,
            "min_fitness_for_valid": 0.3,
            "confirmation_window": 1,
        }
    )
    base = np.array(
        [
            [0.0, -1.0, 1.0],
            [1.0, -0.5, 1.0],
            [2.0, 0.0, 1.0],
            [3.0, 0.5, 1.0],
            [4.0, 1.0, 1.0],
            [5.0, 1.5, 1.0],
        ],
        dtype=np.float32,
    )

    small_shift = np.array([0.01, 0.0, 0.0], dtype=np.float32)
    estimator.update(base, timestamp=10.0)
    estimator.update(base + small_shift, timestamp=10.1)
    third = estimator.update(base + small_shift * 2.0, timestamp=10.2)

    assert third["valid"] is True
    assert third["state"] == EgoMotionState.STATIC
    assert abs(third["frame_displacement"]) < 0.05
