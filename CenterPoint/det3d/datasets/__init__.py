from .builder import build_dataset

import importlib

from .registry import DATASETS


def _is_missing_optional_dependency(exc, allowed_names):
    missing = getattr(exc, "name", "") or ""
    return any(
        missing == allowed_name or missing.startswith(allowed_name + ".")
        for allowed_name in allowed_names
    )


def _optional_symbol(module_name, symbol_name, optional_dependencies):
    try:
        module = importlib.import_module(module_name, __name__)
    except ModuleNotFoundError as exc:
        if _is_missing_optional_dependency(exc, optional_dependencies):
            return None
        raise
    return getattr(module, symbol_name)


NuScenesDataset = _optional_symbol(
    ".nuscenes", "NuScenesDataset", ("torch", "nuscenes", "pyquaternion")
)
WaymoDataset = _optional_symbol(
    ".waymo",
    "WaymoDataset",
    ("torch", "numba", "tensorflow", "nuscenes", "pyquaternion", "waymo_open_dataset"),
)
RTGDataset = _optional_symbol(".rtg", "RTGDataset", ("torch", "numba"))

ConcatDataset = _optional_symbol(".dataset_wrappers", "ConcatDataset", ("torch",))
RepeatDataset = _optional_symbol(".dataset_wrappers", "RepeatDataset", ("torch",))
DistributedGroupSampler = _optional_symbol(
    ".loader", "DistributedGroupSampler", ("torch",)
)
GroupSampler = _optional_symbol(".loader", "GroupSampler", ("torch",))
build_dataloader = _optional_symbol(".loader", "build_dataloader", ("torch",))

# from .voc import VOCDataset
# from .wider_face import WIDERFaceDataset
# from .xml_style import XMLDataset
#
__all__ = [
    "CustomDataset",
    "GroupSampler",
    "DistributedGroupSampler",
    "build_dataloader",
    "ConcatDataset",
    "RepeatDataset",
    "DATASETS",
    "build_dataset",
]
