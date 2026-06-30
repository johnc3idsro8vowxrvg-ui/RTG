import copy

from det3d.utils.registry import build_from_cfg

from .registry import DATASETS


def _dataset_wrappers():
    from .dataset_wrappers import ConcatDataset, RepeatDataset

    return ConcatDataset, RepeatDataset


def _concat_dataset(cfg, default_args=None):
    ConcatDataset, _ = _dataset_wrappers()
    ann_files = cfg["ann_file"]
    img_prefixes = cfg.get("img_prefix", None)
    seg_prefixes = cfg.get("seg_prefixes", None)
    proposal_files = cfg.get("proposal_file", None)
    datasets = []

    for i, ann_file in enumerate(ann_files):
        data_cfg = copy.deepcopy(cfg)
        data_cfg["ann_file"] = ann_file
        if isinstance(img_prefixes, (list, tuple)):
            data_cfg["img_prefix"] = img_prefixes[i]
        if isinstance(seg_prefixes, (list, tuple)):
            data_cfg["seg_prefix"] = seg_prefixes[i]
        if isinstance(proposal_files, (list, tuple)):
            data_cfg["proposal_file"] = proposal_files[i]
        datasets.append(build_dataset(data_cfg, default_args))

    return ConcatDataset(datasets)


def build_dataset(cfg, default_args=None):
    ConcatDataset, RepeatDataset = _dataset_wrappers()
    if isinstance(cfg, (list, tuple)):
        return ConcatDataset([build_dataset(c, default_args) for c in cfg])
    if cfg["type"] == "RepeatDataset":
        return RepeatDataset(build_dataset(cfg["dataset"], default_args), cfg["times"])
    if isinstance(cfg.get("ann_file", None), (list, tuple)):
        return _concat_dataset(cfg, default_args)
    return build_from_cfg(cfg, DATASETS, default_args)
