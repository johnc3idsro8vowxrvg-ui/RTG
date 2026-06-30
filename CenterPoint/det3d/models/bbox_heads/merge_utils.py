def merge_task_predictions(rets, metas, num_classes, cat):
    """Merge CenterPoint task predictions into per-sample predictions."""
    if not rets:
        return []

    num_samples = len(rets[0])
    merged = []
    for sample_idx in range(num_samples):
        sample = {}
        for key in rets[0][sample_idx].keys():
            if key in ("box3d_lidar", "scores"):
                sample[key] = cat([task_ret[sample_idx][key] for task_ret in rets])
            elif key == "label_preds":
                labels = []
                offset = 0
                for task_idx, num_class in enumerate(num_classes):
                    labels.append(rets[task_idx][sample_idx][key] + offset)
                    offset += num_class
                sample[key] = cat(labels)

        sample["metadata"] = metas[0][sample_idx] if metas and metas[0] else None
        merged.append(sample)

    return merged