from pathlib import Path

import numpy as np


def load_lidar_bin_5d(path):
    """Load 4-d or 5-d RTG lidar bin as [x,y,z,intensity,timestamp]."""
    raw = np.fromfile(str(Path(path)), dtype=np.float32)
    if raw.size == 0:
        return np.zeros((0, 5), dtype=np.float32)

    if raw.size % 5 == 0:
        return raw.reshape(-1, 5)
    if raw.size % 4 == 0:
        points_4d = raw.reshape(-1, 4)
        points_5d = np.zeros((points_4d.shape[0], 5), dtype=np.float32)
        points_5d[:, :4] = points_4d
        return points_5d

    raise ValueError(f"Cannot infer lidar bin point dimension for {path}: {raw.size} floats")
