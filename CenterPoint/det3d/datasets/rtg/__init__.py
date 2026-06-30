from .rtg_common import general_to_detection, cls_attr_dist

try:
    from .rtg_dataset import RTGDataset
except ModuleNotFoundError as exc:
    missing = getattr(exc, "name", "") or ""
    if missing in ("torch", "numba") or missing.startswith(("torch.", "numba.")):
        RTGDataset = None
    else:
        raise

__all__ = ['RTGDataset', 'general_to_detection', 'cls_attr_dist']
