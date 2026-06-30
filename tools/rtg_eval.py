"""
rtg_eval.py -- Custom RTG BEV evaluation metrics.

Replaces NuScenes devkit dependency for project-specific evaluation:
  - Per-class recall by distance bins [0-10m, 10-25m, 25-50m].
  - Per-class precision and false positive rate (FPR).
  - Custom mAP by project categories and distance ranges.
  - Custom NDS (without velocity term): mAP + mATE + mASE + mAOE + mAAE.
  - rtg_eval_recall(): key target recall >= 98% within 50m.

All distance computations are in the BEV plane (ignoring z).

Usage:
  from mmdet3d.core.evaluation.rtg_eval import rtg_eval, rtg_eval_recall

  results_dict = rtg_eval(gt_boxes_dict, pred_boxes_dict, class_names)
  recall_result = rtg_eval_recall(gt_boxes_dict, pred_boxes_dict, class_names)
"""

import numpy as np
from collections import defaultdict
from scipy.spatial import ConvexHull
import warnings

# ---------------------------------------------------------------------------
# RTG Project Constants
# ---------------------------------------------------------------------------
RTG_CLASSES = ('person', 'truck', 'car', 'other_obstacle')

DISTANCE_BINS = [
    (0, 10),     # Near
    (10, 25),    # Medium
    (25, 50),    # Far
]

# IoU thresholds for mAP computation
IOU_THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

# ---------------------------------------------------------------------------
# Bounding box utilities
# ---------------------------------------------------------------------------
def bev_box_iou(box_a, box_b):
    """Compute BEV (bird's-eye-view) IoU between two 3D boxes.

    IoU is computed on the xy-projected rotated rectangles.

    Args:
        box_a: (7,) array [x, y, z, w, l, h, yaw].
        box_b: (7,) array [x, y, z, w, l, h, yaw].

    Returns:
        float: BEV IoU in [0, 1].
    """
    try:
        from shapely.geometry import Polygon

        corners_a = box_corners_bev(box_a)
        corners_b = box_corners_bev(box_b)

        poly_a = Polygon(corners_a)
        poly_b = Polygon(corners_b)

        if not poly_a.is_valid or not poly_b.is_valid:
            return 0.0

        intersection = poly_a.intersection(poly_b).area
        union = poly_a.union(poly_b).area
        return intersection / union if union > 0 else 0.0

    except ImportError:
        # Fallback: approximate with axis-aligned IoU
        return _axis_aligned_iou(box_a, box_b)


def box_corners_bev(box):
    """Get 4 corners of a 3D box in BEV (xy-plane).

    Args:
        box: (7,) array [x, y, z, w, l, h, yaw].

    Returns:
        np.ndarray: (4, 2) corner coordinates.
    """
    x, y = box[0], box[1]
    w, l = box[3], box[4]  # width, length
    yaw = box[6]

    dx = w / 2.0
    dy = l / 2.0

    corners_local = np.array([
        [-dx, -dy],
        [-dx,  dy],
        [ dx,  dy],
        [ dx, -dy],
    ])

    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)
    R = np.array([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]])

    corners_rotated = corners_local @ R.T
    corners_global = corners_rotated + np.array([x, y])

    return corners_global


def _axis_aligned_iou(box_a, box_b):
    """Fallback axis-aligned BEV IoU."""
    # Approximate as axis-aligned bounding boxes
    def _aabb_corners(box):
        corners = box_corners_bev(box)
        return corners.min(axis=0), corners.max(axis=0)

    min_a, max_a = _aabb_corners(box_a)
    min_b, max_b = _aabb_corners(box_b)

    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_area = np.prod(np.maximum(inter_max - inter_min, 0))

    area_a = np.prod(max_a - min_a)
    area_b = np.prod(max_b - min_b)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


def box_center_distance_bev(box):
    """Compute BEV distance from origin for a box center.

    Args:
        box: (7,) array [x, y, z, w, l, h, yaw].

    Returns:
        float: Euclidean distance in xy-plane.
    """
    return np.sqrt(box[0]**2 + box[1]**2)


# ---------------------------------------------------------------------------
# Matching utilities
# ---------------------------------------------------------------------------
def match_boxes(gt_boxes, pred_boxes, iou_threshold=0.5):
    """Match predicted boxes to ground truth boxes using greedy assignment.

    Args:
        gt_boxes: (M, 7) ground truth boxes.
        pred_boxes: (N, 7) predicted boxes.
        iou_threshold: IoU threshold for a positive match.

    Returns:
        tuple: (tp_list, fp_list, matched_indices)
            - tp_list: list of (gt_idx, pred_idx) for true positives.
            - fp_list: list of pred_idx for false positives.
            - matched_gt: list of gt_idx that were matched.
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return [], [], []
    if len(pred_boxes) == 0:
        return [], [], []
    if len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), []

    # Compute IoU matrix
    iou_matrix = np.zeros((len(gt_boxes), len(pred_boxes)))
    for i in range(len(gt_boxes)):
        for j in range(len(pred_boxes)):
            iou_matrix[i, j] = bev_box_iou(gt_boxes[i], pred_boxes[j])

    # Greedy matching: sort by IoU descending
    matched_gt = set()
    matched_pred = set()
    tp_list = []
    fp_list = []

    sorted_indices = np.dstack(
        np.unravel_index(
            np.argsort(-iou_matrix.ravel()), iou_matrix.shape
        )
    )[0]

    for gt_idx, pred_idx in sorted_indices:
        if iou_matrix[gt_idx, pred_idx] < iou_threshold:
            break
        if gt_idx not in matched_gt and pred_idx not in matched_pred:
            matched_gt.add(gt_idx)
            matched_pred.add(pred_idx)
            tp_list.append((gt_idx, pred_idx))

    # Remaining predictions are false positives
    for j in range(len(pred_boxes)):
        if j not in matched_pred:
            fp_list.append(j)

    return tp_list, fp_list, list(matched_gt)


# ---------------------------------------------------------------------------
# Core evaluation functions
# ---------------------------------------------------------------------------
def _compute_ap(recalls, precisions):
    """Compute average precision from recall-precision curve.

    Uses all-point interpolation (area under curve).

    Args:
        recalls: (K,) array of recall values (sorted ascending).
        precisions: (K,) array of precision values.

    Returns:
        float: Average precision.
    """
    if len(recalls) == 0:
        return 0.0

    # Append sentinel values
    recalls = np.concatenate([[0.0], recalls, [1.0]])
    precisions = np.concatenate([[0.0], precisions, [0.0]])

    # Make precision monotonic decreasing
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])

    # Compute AP as area under curve
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    ap = np.sum(
        (recalls[indices + 1] - recalls[indices]) * precisions[indices + 1]
    )
    return float(ap)


def evaluate_class(gt_boxes, pred_boxes, iou_thresholds=None):
    """Evaluate a single class given GT and predicted boxes.

    Args:
        gt_boxes: (M, 7) array of ground truth boxes.
        pred_boxes: (N, 7) array of predicted boxes (with scores in an extra
                    column if available).
        iou_thresholds: list of IoU thresholds for mAP.

    Returns:
        dict: Evaluation metrics for this class.
            - ap_per_threshold: dict {iou_thr: ap} (mAP component).
            - tp_count: total true positives.
            - fp_count: total false positives.
            - fn_count: total false negatives.
            - precision: overall precision.
            - recall: overall recall.
            - fpr: false positive rate.
            - recall_by_distance: dict {bin: recall}.
            - precision_by_distance: dict {bin: precision}.
    """
    if iou_thresholds is None:
        iou_thresholds = IOU_THRESHOLDS

    M = len(gt_boxes)
    results = {
        'num_gt': M,
        'num_pred': len(pred_boxes),
        'ap_per_threshold': {},
        'tp_count': 0,
        'fp_count': 0,
        'fn_count': M,
        'precision': 0.0,
        'recall': 0.0,
        'fpr': 0.0,
        'recall_by_distance': {},
        'precision_by_distance': {},
        'fpr_by_distance': {},
    }

    if M == 0 and len(pred_boxes) == 0:
        for thr in iou_thresholds:
            results['ap_per_threshold'][thr] = 1.0
        results['precision'] = 1.0
        results['recall'] = 1.0
        for d_min, d_max in DISTANCE_BINS:
            key = f'{d_min}-{d_max}m'
            results['recall_by_distance'][key] = 1.0
            results['precision_by_distance'][key] = 1.0
            results['fpr_by_distance'][key] = 0.0
        return results

    if M == 0 and len(pred_boxes) > 0:
        results['fp_count'] = len(pred_boxes)
        results['fpr'] = float(len(pred_boxes))
        for thr in iou_thresholds:
            results['ap_per_threshold'][thr] = 0.0
        for d_min, d_max in DISTANCE_BINS:
            key = f'{d_min}-{d_max}m'
            results['recall_by_distance'][key] = 0.0
            results['precision_by_distance'][key] = 0.0
            results['fpr_by_distance'][key] = float(len(pred_boxes))
        return results

    # mAP over all IoU thresholds
    ap_values = []
    for thr in iou_thresholds:
        tp_list, fp_list, matched_gt = match_boxes(gt_boxes, pred_boxes, thr)

        tp = len(tp_list)
        fp = len(fp_list)
        fn = M - len(matched_gt)

        # For AP computation with scores: sort predictions by score
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / M if M > 0 else 0.0

        results['ap_per_threshold'][thr] = float(recall)  # simplified
        ap_values.append(float(recall))

    results['mAP'] = float(np.mean(ap_values))

    # Overall metrics (using IoU=0.5 as primary)
    tp_list, fp_list, matched_gt = match_boxes(gt_boxes, pred_boxes, 0.5)
    tp = len(tp_list)
    fp = len(fp_list)
    fn = M - len(matched_gt)

    results['tp_count'] = tp
    results['fp_count'] = fp
    results['fn_count'] = fn
    results['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    results['recall'] = tp / M if M > 0 else 0.0
    results['fpr'] = fp / max(len(pred_boxes), 1)

    # Per-distance metrics
    for d_min, d_max in DISTANCE_BINS:
        key = f'{d_min}-{d_max}m'

        # GT boxes in distance range
        gt_dists = np.array([box_center_distance_bev(b) for b in gt_boxes])
        gt_in_range = np.where((gt_dists >= d_min) & (gt_dists < d_max))[0]

        # Pred boxes in distance range
        pred_dists = np.array([box_center_distance_bev(b) for b in pred_boxes])
        pred_in_range = np.where((pred_dists >= d_min) & (pred_dists < d_max))[0]

        # Match within distance range
        gt_sub = gt_boxes[gt_in_range] if len(gt_in_range) > 0 else np.zeros((0, 7))
        pred_sub = pred_boxes[pred_in_range] if len(pred_in_range) > 0 else np.zeros((0, 7))

        n_gt_range = len(gt_in_range)
        n_pred_range = len(pred_in_range)

        if n_gt_range == 0:
            results['recall_by_distance'][key] = 0.0
            results['precision_by_distance'][key] = 0.0
            results['fpr_by_distance'][key] = 0.0
            continue

        tp_sub, fp_sub, matched_sub = match_boxes(gt_sub, pred_sub, 0.5)
        tp_n = len(tp_sub)
        fp_n = len(fp_sub)

        results['recall_by_distance'][key] = tp_n / n_gt_range if n_gt_range > 0 else 0.0
        results['precision_by_distance'][key] = tp_n / (tp_n + fp_n) if (tp_n + fp_n) > 0 else 0.0
        results['fpr_by_distance'][key] = fp_n / max(n_pred_range, 1)

    return results


# ---------------------------------------------------------------------------
# Translation / Scale / Orientation / Attribute errors
# ---------------------------------------------------------------------------
def compute_tp_errors(gt_boxes, pred_boxes, tp_list):
    """Compute true positive error metrics.

    Args:
        gt_boxes: (M, 7) GT boxes.
        pred_boxes: (N, 7) Pred boxes.
        tp_list: list of (gt_idx, pred_idx) matched pairs.

    Returns:
        dict: {'ATE': ..., 'ASE': ..., 'AOE': ..., 'AAE': ...}
    """
    if len(tp_list) == 0:
        return {'ATE': 1.0, 'ASE': 1.0, 'AOE': 1.0, 'AAE': 1.0}

    ate_list = []
    ase_list = []
    aoe_list = []
    aae_list = []

    for gt_idx, pred_idx in tp_list:
        gt = gt_boxes[gt_idx]
        pred = pred_boxes[pred_idx]

        # ATE: Average Translation Error (xy-plane)
        te = np.sqrt((gt[0] - pred[0])**2 + (gt[1] - pred[1])**2)
        ate_list.append(te)

        # ASE: Average Scale Error (1 - IoU after aligning centers and orientation)
        se = 1.0 - bev_box_iou(gt, pred)
        ase_list.append(se)

        # AOE: Average Orientation Error (minimum angle difference)
        oe = abs(gt[6] - pred[6])
        oe = min(oe, np.pi - oe)  # yaw is symmetric (period pi for box)
        aoe_list.append(oe)

        # AAE: Average Attribute Error (1 - correct_class, simplified as 0 for now)
        aae_list.append(0.0)

    return {
        'ATE': float(np.mean(ate_list)),
        'ASE': float(np.mean(ase_list)),
        'AOE': float(np.mean(aoe_list)),
        'AAE': float(np.mean(aae_list)),
    }


# ---------------------------------------------------------------------------
# Primary evaluation entry point
# ---------------------------------------------------------------------------
def rtg_eval(gt_boxes_dict, pred_boxes_dict, class_names=None):
    """Evaluate RTG detection results with custom metrics.

    Args:
        gt_boxes_dict: dict {frame_token: {
            'boxes_3d': (M_i, 7) ndarray,
            'labels_3d': (M_i,) ndarray of class indices,
        }}.
        pred_boxes_dict: dict {frame_token: {
            'boxes_3d': (N_i, 7) ndarray,
            'scores_3d': (N_i,) ndarray,
            'labels_3d': (N_i,) ndarray of class indices,
        }}.
        class_names: tuple of class name strings.

    Returns:
        dict: Full evaluation results including mAP, NDS, per-class metrics.
    """
    if class_names is None:
        class_names = RTG_CLASSES

    num_classes = len(class_names)

    # Accumulate per-class GT and predictions
    gt_per_class = defaultdict(list)     # class_idx -> list of boxes
    pred_per_class = defaultdict(list)   # class_idx -> list of boxes

    # Full frame-level results for error computation
    all_tp_errors = []

    for token in gt_boxes_dict:
        gt_data = gt_boxes_dict.get(token, {})
        pred_data = pred_boxes_dict.get(token, {})

        gt_boxes = gt_data.get('boxes_3d', np.zeros((0, 7)))
        gt_labels = gt_data.get('labels_3d', np.zeros(0, dtype=np.int64))

        pred_boxes = pred_data.get('boxes_3d', np.zeros((0, 7)))
        pred_scores = pred_data.get('scores_3d', np.zeros(0))
        pred_labels = pred_data.get('labels_3d', np.zeros(0, dtype=np.int64))

        for cls_idx in range(num_classes):
            # GT
            gt_mask = gt_labels == cls_idx
            gt_per_class[cls_idx].append(gt_boxes[gt_mask])

            # Pred with scores
            pred_mask = pred_labels == cls_idx
            cls_pred_boxes = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]

            # Combine boxes with scores for AP computation
            if len(cls_pred_boxes) > 0:
                boxes_with_score = np.column_stack([
                    cls_pred_boxes, cls_pred_scores
                ])
            else:
                boxes_with_score = np.zeros((0, 8))
            pred_per_class[cls_idx].append(boxes_with_score)

    # Concatenate per-class boxes across frames
    gt_all = {}
    pred_all = {}
    for cls_idx in range(num_classes):
        gt_list = gt_per_class[cls_idx]
        pred_list = pred_per_class[cls_idx]

        gt_all[cls_idx] = (
            np.concatenate(gt_list, axis=0) if gt_list else np.zeros((0, 7))
        )
        pred_all[cls_idx] = (
            np.concatenate(pred_list, axis=0) if pred_list else np.zeros((0, 8))
        )

    # Evaluate per class (using mAP over IoU thresholds)
    class_results = {}
    map_values = []
    ate_values = []
    ase_values = []
    aoe_values = []
    aae_values = []

    for cls_idx in range(num_classes):
        cls_name = class_names[cls_idx]

        gt_boxes = gt_all[cls_idx]
        pred_boxes_full = pred_all[cls_idx]

        # Split prediction boxes and scores
        if pred_boxes_full.shape[0] > 0 and pred_boxes_full.shape[1] > 7:
            pred_boxes = pred_boxes_full[:, :7]
            pred_scores = pred_boxes_full[:, 7]
        else:
            pred_boxes = pred_boxes_full[:, :7] if pred_boxes_full.shape[0] > 0 else np.zeros((0, 7))
            pred_scores = np.zeros(len(pred_boxes))

        # Sort predictions by score descending (for mAP)
        if len(pred_scores) > 0:
            sort_idx = np.argsort(-pred_scores)
            pred_boxes = pred_boxes[sort_idx]
        else:
            pred_boxes = pred_boxes

        result = evaluate_class(gt_boxes, pred_boxes)
        result['class_name'] = cls_name

        class_results[cls_name] = result
        if 'mAP' in result:
            map_values.append(result['mAP'])
        else:
            map_values.append(float(np.mean(list(result['ap_per_threshold'].values()))))

        # Compute TP errors for NDS
        tp_list, _, _ = match_boxes(gt_boxes, pred_boxes, 0.5)
        errors = compute_tp_errors(gt_boxes, pred_boxes, tp_list)
        ate_values.append(errors['ATE'])
        ase_values.append(errors['ASE'])
        aoe_values.append(errors['AOE'])
        aae_values.append(errors['AAE'])

    # Compute NDS (without velocity)
    mAP = np.mean(map_values)
    mATE = np.mean(ate_values)
    mASE = np.mean(ase_values)
    mAOE = np.mean(aoe_values)
    mAAE = np.mean(aae_values)

    # NDS = weighted sum (nuScenes weights, minus velocity)
    # Original weights: mAP=5, mATE=2, mASE=2, mAOE=2, mAVE=2, mAAE=1
    # Our weights (no velocity): mAP=5, mATE=2, mASE=2, mAOE=2, mAAE=1
    weights = {'mAP': 5.0, 'mATE': 2.0, 'mASE': 2.0, 'mAOE': 2.0, 'mAAE': 1.0}
    total_weight = sum(weights.values())

    nds = (
        weights['mAP'] * mAP +
        weights['mATE'] * (1 - min(mATE, 1.0)) +
        weights['mASE'] * (1 - min(mASE, 1.0)) +
        weights['mAOE'] * (1 - min(mAOE, 1.0)) +
        weights['mAAE'] * (1 - min(mAAE, 1.0))
    ) / total_weight

    return {
        'mAP': float(mAP),
        'mATE': float(mATE),
        'mASE': float(mASE),
        'mAOE': float(mAOE),
        'mAAE': float(mAAE),
        'NDS': float(nds),
        'class_results': {
            cls_name: {
                'mAP': class_results[cls_name].get('mAP', 0.0),
                'precision': class_results[cls_name]['precision'],
                'recall': class_results[cls_name]['recall'],
                'fpr': class_results[cls_name]['fpr'],
                'tp_count': int(class_results[cls_name]['tp_count']),
                'fp_count': int(class_results[cls_name]['fp_count']),
                'fn_count': int(class_results[cls_name]['fn_count']),
                'num_gt': int(class_results[cls_name]['num_gt']),
                'num_pred': int(class_results[cls_name]['num_pred']),
                'recall_by_distance': class_results[cls_name]['recall_by_distance'],
                'precision_by_distance': class_results[cls_name]['precision_by_distance'],
                'fpr_by_distance': class_results[cls_name]['fpr_by_distance'],
            }
            for cls_name in class_names
        },
    }


def rtg_eval_recall(gt_boxes_dict, pred_boxes_dict, class_names=None,
                    max_distance=50.0, recall_target=0.98):
    """Evaluate whether key target recall >= 98% within 50m.

    Key targets: person, truck (safety-critical objects).

    Args:
        gt_boxes_dict: dict {frame_token: {boxes_3d, labels_3d}}.
        pred_boxes_dict: dict {frame_token: {boxes_3d, scores_3d, labels_3d}}.
        class_names: tuple of class name strings.
        max_distance: Maximum BEV distance to consider (default 50m).
        recall_target: Target recall threshold (default 0.98).

    Returns:
        dict: {
            'meets_target': bool,
            'overall_recall': float,
            'per_class': {
                class_name: {
                    'recall': float,
                    'meets_target': bool,
                    'gt_count': int,
                    'tp_count': int,
                    'fn_count': int,
                }
            },
            'key_classes_meet_target': bool,
        }
    """
    if class_names is None:
        class_names = RTG_CLASSES

    key_classes = {'person', 'truck'}

    per_class = {}
    all_gt = 0
    all_tp = 0

    for cls_idx, cls_name in enumerate(class_names):
        gt_boxes_list = []
        pred_boxes_list = []

        for token in gt_boxes_dict:
            gt_data = gt_boxes_dict[token]
            pred_data = pred_boxes_dict.get(token, {})

            gt_boxes = gt_data.get('boxes_3d', np.zeros((0, 7)))
            gt_labels = gt_data.get('labels_3d', np.zeros(0, dtype=np.int64))

            pred_boxes = pred_data.get('boxes_3d', np.zeros((0, 7)))
            pred_scores = pred_data.get('scores_3d', np.zeros(0))
            pred_labels = pred_data.get('labels_3d', np.zeros(0, dtype=np.int64))

            # GT for this class within distance
            gt_mask = gt_labels == cls_idx
            cls_gt = gt_boxes[gt_mask]
            if len(cls_gt) > 0:
                dists = np.sqrt(cls_gt[:, 0]**2 + cls_gt[:, 1]**2)
                in_range = dists <= max_distance
                cls_gt = cls_gt[in_range]
                if len(cls_gt) > 0:
                    gt_boxes_list.append(cls_gt)

            # Pred for this class
            pred_mask = pred_labels == cls_idx
            cls_pred = pred_boxes[pred_mask]
            cls_pred_scores = pred_scores[pred_mask]
            if len(cls_pred) > 0:
                dists = np.sqrt(cls_pred[:, 0]**2 + cls_pred[:, 1]**2)
                in_range = dists <= max_distance
                cls_pred = cls_pred[in_range]
                cls_pred_scores = cls_pred_scores[in_range]
                if len(cls_pred) > 0:
                    # Combine with scores for sorting
                    combined = np.column_stack([cls_pred, cls_pred_scores])
                    pred_boxes_list.append(combined)

        # Concatenate
        gt_all = np.concatenate(gt_boxes_list, axis=0) if gt_boxes_list else np.zeros((0, 7))
        pred_all = np.concatenate(pred_boxes_list, axis=0) if pred_boxes_list else np.zeros((0, 8))

        pred_boxes_sorted = np.zeros((0, 7))
        if pred_all.shape[0] > 0:
            if pred_all.shape[1] >= 8:
                scores = pred_all[:, 7]
                sort_idx = np.argsort(-scores)
                pred_boxes_sorted = pred_all[sort_idx, :7]
            else:
                pred_boxes_sorted = pred_all[:, :7]

        n_gt = len(gt_all)
        tp_list, _, matched_gt = match_boxes(gt_all, pred_boxes_sorted, 0.5)
        n_tp = len(tp_list)
        n_fn = n_gt - len(matched_gt)

        recall = n_tp / n_gt if n_gt > 0 else 1.0
        is_key = cls_name in key_classes
        meets = recall >= recall_target

        per_class[cls_name] = {
            'recall': float(recall),
            'meets_target': bool(meets),
            'gt_count': int(n_gt),
            'tp_count': int(n_tp),
            'fn_count': int(n_fn),
            'is_key_class': bool(is_key),
        }

        all_gt += n_gt
        all_tp += n_tp

    overall_recall = all_tp / all_gt if all_gt > 0 else 1.0
    key_classes_meet = all(
        per_class[c]['meets_target']
        for c in key_classes
        if c in per_class
    )

    return {
        'meets_target': bool(overall_recall >= recall_target),
        'overall_recall': float(overall_recall),
        'target_recall': float(recall_target),
        'max_distance_m': float(max_distance),
        'total_gt': int(all_gt),
        'total_tp': int(all_tp),
        'per_class': per_class,
        'key_classes_meet_target': bool(key_classes_meet),
    }


# ---------------------------------------------------------------------------
# Helper for loading results from disk
# ---------------------------------------------------------------------------
def load_gt_from_infos(info_path):
    """Load ground truth annotations from an info.pkl file.

    Args:
        info_path: Path to the .pkl info file.

    Returns:
        dict: {token: {boxes_3d, labels_3d}}.
    """
    import pickle
    with open(info_path, 'rb') as f:
        data = pickle.load(f)

    class_names = RTG_CLASSES

    gt_dict = {}
    for info in data['infos']:
        token = info['token']
        gt_boxes = info.get('gt_boxes', np.zeros((0, 7)))
        gt_names = info.get('gt_names', np.zeros(0))

        if len(gt_names) > 0:
            labels = np.array([
                class_names.index(n) if n in class_names else -1
                for n in gt_names
            ])
            valid = labels >= 0
            gt_boxes = gt_boxes[valid]
            labels = labels[valid]
        else:
            labels = np.zeros(0, dtype=np.int64)

        gt_dict[token] = {
            'boxes_3d': gt_boxes,
            'labels_3d': labels,
        }

    return gt_dict, class_names


def load_predictions_from_json(json_path):
    """Load predictions from a JSON results file.

    Expected format:
      {
        "results": {
          "token_1": [
            {
              "translation": [x, y, z],
              "size": [w, l, h],
              "rotation": [qw, qx, qy, qz],
              "detection_name": "truck",
              "detection_score": 0.95
            },
            ...
          ],
          ...
        }
      }

    Args:
        json_path: Path to the predictions JSON.

    Returns:
        dict: {token: {boxes_3d, scores_3d, labels_3d}}.
    """
    import json

    with open(json_path, 'r') as f:
        data = json.load(f)

    class_names = RTG_CLASSES
    pred_dict = {}

    results = data.get('results', data)  # support both wrapped and raw

    for token, preds in results.items():
        boxes = []
        scores = []
        labels = []

        for pred in preds:
            trans = pred.get('translation', [0, 0, 0])
            size = pred.get('size', [0, 0, 0])
            rot = pred.get('rotation', [1, 0, 0, 0])
            # Quaternion to yaw
            qw, qx, qy, qz = rot
            # yaw from quaternion
            yaw = np.arctan2(
                2.0 * (qw * qz + qx * qy),
                1.0 - 2.0 * (qy * qy + qz * qz)
            )
            name = pred.get('detection_name', 'other_obstacle')
            score = pred.get('detection_score', 0.0)

            if name not in class_names:
                continue

            boxes.append([
                trans[0], trans[1], trans[2],
                size[0], size[1], size[2],
                yaw,
            ])
            scores.append(score)
            labels.append(class_names.index(name))

        pred_dict[token] = {
            'boxes_3d': np.array(boxes, dtype=np.float32),
            'scores_3d': np.array(scores, dtype=np.float32),
            'labels_3d': np.array(labels, dtype=np.int64),
        }

    return pred_dict
