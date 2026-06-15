"""
RTGDataset — RTG (Rubber-Tyred Gantry) port crane BEV anti-collision dataset.

Adapted for the det3d/CenterPoint framework.  Compatible with info.pkl format
produced by tools/data_converter/generate_rtg_infos.py.

Key differences from NuScenesDataset:
  - 4 classes: person, truck, car, other_obstacle
  - Single-frame mode (nsweeps=1, no multi-sweep aggregation)
  - Concatenated lidar file (front+rear already merged by generate_rtg_infos.py)
  - LiDAR-only (no camera branch)
"""

import pickle
from pathlib import Path

import numpy as np

from det3d.core import box_np_ops
from det3d.core.input.voxel_generator import VoxelGenerator
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS
from .rtg_common import RTG_CLASS_NAMES


@DATASETS.register_module
class RTGDataset(PointCloudDataset):
    NumPointFeatures = 5  # x, y, z, intensity, timestamp

    def __init__(
        self,
        info_path,
        root_path,
        nsweeps=1,
        cfg=None,
        pipeline=None,
        class_names=None,
        test_mode=False,
        load_interval=1,
        **kwargs,
    ):
        self.load_interval = load_interval
        self.nsweeps = nsweeps

        if class_names is None:
            class_names = list(RTG_CLASS_NAMES)

        super(RTGDataset, self).__init__(
            root_path, info_path, pipeline, test_mode=test_mode, class_names=class_names
        )

        self.load_infos(info_path)

    def load_infos(self, info_path):
        with open(info_path, "rb") as f:
            raw = pickle.load(f)

        # Handle format: {'infos': [...], 'metadata': {...}} (from generate_rtg_infos.py)
        if isinstance(raw, dict) and 'infos' in raw:
            self._rtg_infos = raw['infos']
            self._metadata = raw.get('metadata', {})
        else:
            self._rtg_infos = raw
            self._metadata = {}

        # Apply load_interval
        self._rtg_infos = self._rtg_infos[::self.load_interval]

    def __len__(self):
        return len(self._rtg_infos)

    def _build_gt_boxes_9d(self, gt_boxes_7d):
        """Convert 7-dim boxes [x,y,z,w,l,h,yaw] to 9-dim [x,y,z,w,l,h,vx,vy,yaw].

        RTG scenario is quasi-static (gantry moves slowly), so velocity is set to 0.
        """
        if gt_boxes_7d.shape[-1] == 9:
            return gt_boxes_7d.astype(np.float32)
        gt_boxes_9d = np.zeros((gt_boxes_7d.shape[0], 9), dtype=np.float32)
        gt_boxes_9d[:, :6] = gt_boxes_7d[:, :6]   # x, y, z, w, l, h
        gt_boxes_9d[:, 8] = gt_boxes_7d[:, 6]      # yaw → index 8
        # vx, vy (indices 6,7) remain 0
        return gt_boxes_9d

    def get_sensor_data(self, idx):
        info = self._rtg_infos[idx]

        # Load concatenated lidar point cloud (5-d: x,y,z,intensity,timestamp)
        lidar_path = Path(self._root_path) / info["lidar_path"]
        points = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 5)

        # Add timestamp column (zero for single-frame)
        times = np.zeros((points.shape[0], 1), dtype=np.float32)

        res = {
            "lidar": {
                "type": "lidar",
                "points": points,
                # combined uses first 4 dims (x,y,z,intensity) + timestamp column
                "combined": np.hstack([points[:, :4], times]),
                "nsweeps": self.nsweeps,
                "annotations": None,
            },
            "type": "NuScenesDataset",  # Compatible with existing pipeline (Preprocess/Voxelization/AssignLabel)
            "metadata": {
                "image_prefix": self._root_path,
                "num_point_features": self.NumPointFeatures,
                "token": info["token"],
            },
            "calib": None,
            "cam": {},
            "mode": "val" if self.test_mode else "train",
        }

        data, _ = self.pipeline(res, info)
        return data

    def __getitem__(self, idx):
        return self.get_sensor_data(idx)

    @property
    def ground_truth_annotations(self):
        if not self._rtg_infos or "gt_boxes" not in self._rtg_infos[0]:
            return None

        gt_annos = []
        for info in self._rtg_infos:
            gt_names = np.array(info.get("gt_names", []))
            gt_boxes = info.get("gt_boxes", np.zeros((0, 7), dtype=np.float32))

            if len(gt_names) == 0:
                gt_annos.append({
                    "bbox": np.zeros((0, 4), dtype=np.float32),
                    "alpha": np.zeros(0, dtype=np.float32),
                    "occluded": np.zeros(0, dtype=np.int32),
                    "truncated": np.zeros(0, dtype=np.int32),
                    "name": np.array([], dtype=str),
                    "location": np.zeros((0, 3), dtype=np.float32),
                    "dimensions": np.zeros((0, 3), dtype=np.float32),
                    "rotation_y": np.zeros(0, dtype=np.float32),
                    "token": info["token"],
                })
                continue

            mask = np.array([n in self._class_names for n in gt_names], dtype=bool)
            gt_names = gt_names[mask]
            gt_boxes = gt_boxes[mask]

            N = len(gt_boxes)
            gt_annos.append({
                "bbox": np.tile(np.array([[0, 0, 50, 50]]), [N, 1]),
                "alpha": np.full(N, -10, dtype=np.float32),
                "occluded": np.zeros(N, dtype=np.int32),
                "truncated": np.zeros(N, dtype=np.int32),
                "name": gt_names,
                "location": gt_boxes[:, :3],
                "dimensions": gt_boxes[:, 3:6],
                "rotation_y": gt_boxes[:, 6],
                "token": info["token"],
            })

        return gt_annos

    def evaluation(self, dt_annos, output_dir=None, testset=False):
        """RTG evaluation placeholder.

        For RTG, we use custom evaluation metrics (recall, precision, FPR)
        via tools/eval_rtg.py instead of NuScenes evaluation protocol.
        """
        from det3d.torchie.trainer.utils import get_root_logger
        logger = get_root_logger('debug')
        logger.info('RTG evaluation: using custom metrics (see tools/eval_rtg.py)')
        return {}, None


from det3d.datasets.registry import PIPELINES


@PIPELINES.register_module
class LoadRTGAnnotations(object):
    """Load RTG 3D annotations from info dict into res.

    Converts 7-dim gt_boxes to 9-dim (adding zero velocity).
    """

    def __init__(self, with_bbox=True, **kwargs):
        pass

    def __call__(self, res, info):
        if "gt_boxes" in info:
            gt_boxes = info["gt_boxes"].astype(np.float32)
            gt_boxes[np.isnan(gt_boxes)] = 0

            # Convert 7-d → 9-d: [x,y,z,w,l,h,yaw] → [x,y,z,w,l,h,vx,vy,yaw]
            if gt_boxes.ndim == 2 and gt_boxes.shape[-1] == 7:
                gt_boxes_9d = np.zeros((gt_boxes.shape[0], 9), dtype=np.float32)
                gt_boxes_9d[:, :6] = gt_boxes[:, :6]   # x, y, z, w, l, h
                gt_boxes_9d[:, 8] = gt_boxes[:, 6]      # yaw → index 8
                # vx, vy (indices 6,7) remain 0
                gt_boxes = gt_boxes_9d

            res["lidar"]["annotations"] = {
                "boxes": gt_boxes,
                "names": info["gt_names"],
                "tokens": info.get("gt_boxes_token",
                                   [f"{info['token']}_{i}" for i in range(len(gt_boxes))]),
                "velocities": info.get("gt_velocity",
                                       np.zeros((gt_boxes.shape[0], 2), dtype=np.float32)),
            }
        else:
            res["lidar"]["annotations"] = {
                "boxes": np.zeros((0, 9), dtype=np.float32),
                "names": np.array([], dtype=str),
                "tokens": [],
                "velocities": np.zeros((0, 2), dtype=np.float32),
            }

        return res, info
