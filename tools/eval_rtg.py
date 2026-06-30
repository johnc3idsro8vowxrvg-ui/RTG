#!/usr/bin/env python3
"""
eval_rtg.py -- Offline evaluation script for RTG BEV detection results.

Loads ground truth from info.pkl and predictions from a JSON results file,
computes custom RTG evaluation metrics, and outputs a JSON report.

Usage:
  # Basic evaluation
  python tools/eval_rtg.py \\
      --gt data/rtg_infos_val.pkl \\
      --pred results/results_rtg.json \\
      --output eval_report.json

  # With recall check
  python tools/eval_rtg.py \\
      --gt data/rtg_infos_val.pkl \\
      --pred results/results_rtg.json \\
      --output eval_report.json \\
      --recall-check

  # Specify class names
  python tools/eval_rtg.py \\
      --gt data/rtg_infos_val.pkl \\
      --pred results/results_rtg.json \\
      --classes person,truck,car,other_obstacle \\
      --output eval_report.json
"""

import argparse
import json
import os
import sys
import time


def try_import_rtg_eval():
    """Import rtg_eval module, adding the project root to path.

    Returns:
        module: The rtg_eval module, or None on failure.
    """
    try:
        from mmdet3d.core.evaluation.rtg_eval import (
            rtg_eval,
            rtg_eval_recall,
            load_gt_from_infos,
            load_predictions_from_json,
            RTG_CLASSES,
        )
        return rtg_eval, rtg_eval_recall, load_gt_from_infos, \
            load_predictions_from_json, RTG_CLASSES
    except ImportError:
        # Add project root to path and retry
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)

        try:
            from mmdet3d.core.evaluation.rtg_eval import (
                rtg_eval as _rtg_eval,
                rtg_eval_recall as _rtg_eval_recall,
                load_gt_from_infos as _load_gt,
                load_predictions_from_json as _load_pred,
                RTG_CLASSES as _RTG_CLASSES,
            )
            return _rtg_eval, _rtg_eval_recall, _load_gt, _load_pred, _RTG_CLASSES
        except ImportError as e:
            print(f'[ERROR] Failed to import rtg_eval module: {e}')
            print(
                '[HINT] Make sure the project root is in PYTHONPATH, '
                'e.g.: export PYTHONPATH=/path/to/project:$PYTHONPATH'
            )
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Offline RTG BEV evaluation script.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/eval_rtg.py \\
      --gt data/rtg_infos_val.pkl \\
      --pred results/results_rtg.json \\
      --output eval_report.json

  python tools/eval_rtg.py \\
      --gt data/rtg_infos_val.pkl \\
      --pred results/results_rtg.json \\
      --output eval_report.json \\
      --recall-check \\
      --recall-distance 50 \\
      --recall-target 0.98
        """,
    )

    # Required arguments
    parser.add_argument(
        '--gt', required=True,
        help='Path to ground truth info.pkl file (e.g., rtg_infos_val.pkl).',
    )
    parser.add_argument(
        '--pred', required=True,
        help='Path to predictions JSON file.',
    )
    parser.add_argument(
        '--output', default='eval_report.json',
        help='Output path for the evaluation report JSON.',
    )

    # Optional: class names
    parser.add_argument(
        '--classes', default=None,
        help='Comma-separated class names (default: person,truck,car,other_obstacle).',
    )

    # Recall check options
    parser.add_argument(
        '--recall-check', action='store_true',
        help='Run the recall-target check (>= 98%% recall within 50m).',
    )
    parser.add_argument(
        '--recall-distance', type=float, default=50.0,
        help='Maximum distance for recall check in meters (default: 50).',
    )
    parser.add_argument(
        '--recall-target', type=float, default=0.98,
        help='Target recall threshold (default: 0.98 = 98%%).',
    )

    # Report format options
    parser.add_argument(
        '--pretty', action='store_true',
        help='Pretty-print the summary to stdout.',
    )

    args = parser.parse_args()

    # Resolve imports
    (
        rtg_eval_fn,
        rtg_eval_recall_fn,
        load_gt_from_infos,
        load_predictions_from_json,
        _default_classes,
    ) = try_import_rtg_eval()

    # Parse class names
    if args.classes:
        class_names = tuple(args.classes.split(','))
    else:
        class_names = _default_classes
    print(f'[INFO] Classes: {class_names}')

    # Load ground truth
    print(f'[INFO] Loading ground truth from: {args.gt}')
    if not os.path.isfile(args.gt):
        print(f'[ERROR] GT file not found: {args.gt}')
        sys.exit(1)
    gt_dict, gt_class_names = load_gt_from_infos(args.gt)
    print(f'[INFO] Loaded GT for {len(gt_dict)} frames.')

    # Count total GT boxes per class
    gt_class_counts = {name: 0 for name in class_names}
    for token, data in gt_dict.items():
        labels = data.get('labels_3d', [])
        for lbl in labels:
            if 0 <= lbl < len(class_names):
                gt_class_counts[class_names[lbl]] += 1
    print(f'[INFO] GT box counts: {gt_class_counts}')

    # Load predictions
    print(f'[INFO] Loading predictions from: {args.pred}')
    if not os.path.isfile(args.pred):
        print(f'[ERROR] Prediction file not found: {args.pred}')
        sys.exit(1)
    pred_dict = load_predictions_from_json(args.pred)
    print(f'[INFO] Loaded predictions for {len(pred_dict)} frames.')

    # Count total pred boxes per class
    pred_class_counts = {name: 0 for name in class_names}
    for token, data in pred_dict.items():
        labels = data.get('labels_3d', [])
        for lbl in labels:
            if 0 <= lbl < len(class_names):
                pred_class_counts[class_names[lbl]] += 1
    print(f'[INFO] Pred box counts: {pred_class_counts}')

    # Run evaluation
    print('\n[INFO] Running RTG evaluation...')
    t_start = time.time()

    eval_results = rtg_eval_fn(gt_dict, pred_dict, class_names)

    t_eval = time.time() - t_start
    print(f'[INFO] Evaluation completed in {t_eval:.2f}s')

    # Run recall check if requested
    recall_results = None
    if args.recall_check:
        print('\n[INFO] Running recall target check...')
        recall_results = rtg_eval_recall_fn(
            gt_dict,
            pred_dict,
            class_names,
            max_distance=args.recall_distance,
            recall_target=args.recall_target,
        )

    # Build report
    report = {
        'metadata': {
            'gt_file': os.path.abspath(args.gt),
            'pred_file': os.path.abspath(args.pred),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'evaluation_time_sec': round(t_eval, 3),
            'classes': list(class_names),
        },
        'summary': {
            'mAP': eval_results['mAP'],
            'NDS': eval_results['NDS'],
            'mATE': eval_results['mATE'],
            'mASE': eval_results['mASE'],
            'mAOE': eval_results['mAOE'],
            'mAAE': eval_results['mAAE'],
        },
        'per_class': eval_results['class_results'],
    }

    if recall_results:
        report['recall_check'] = {
            'meets_target': recall_results['meets_target'],
            'overall_recall': recall_results['overall_recall'],
            'target_recall': recall_results['target_recall'],
            'max_distance_m': recall_results['max_distance_m'],
            'total_gt': recall_results['total_gt'],
            'total_tp': recall_results['total_tp'],
            'key_classes_meet_target': recall_results['key_classes_meet_target'],
            'per_class': recall_results['per_class'],
        }

    # Save report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2)
    print(f'\n[DONE] Report saved to: {args.output}')

    # Print summary
    if args.pretty:
        print('\n' + '=' * 60)
        print('RTG BEV Evaluation Summary')
        print('=' * 60)
        print(f'  mAP : {report["summary"]["mAP"]:.4f}')
        print(f'  NDS : {report["summary"]["NDS"]:.4f}')
        print(f'  mATE: {report["summary"]["mATE"]:.4f}')
        print(f'  mASE: {report["summary"]["mASE"]:.4f}')
        print(f'  mAOE: {report["summary"]["mAOE"]:.4f}')
        print(f'  mAAE: {report["summary"]["mAAE"]:.4f}')
        print('-' * 60)
        print('Per-class results:')
        print(f'{"Class":<20} {"mAP":>8} {"Recall":>8} {"Prec":>8} {"FPR":>8}')
        print('-' * 60)
        for cls_name, cls_res in report['per_class'].items():
            mAP_val = cls_res.get('mAP', 0.0)
            print(
                f'{cls_name:<20} '
                f'{mAP_val:>8.4f} '
                f'{cls_res["recall"]:>8.4f} '
                f'{cls_res["precision"]:>8.4f} '
                f'{cls_res["fpr"]:>8.4f}'
            )
        print('-' * 60)
        for cls_name, cls_res in report['per_class'].items():
            print(f'\n{cls_name} - Recall by Distance:')
            for bin_key, val in cls_res.get('recall_by_distance', {}).items():
                print(f'  {bin_key}: {val:.4f}')
        print('=' * 60)

        if recall_results:
            print('\nRecall Target Check:')
            status = 'PASS' if recall_results['meets_target'] else 'FAIL'
            print(f'  Overall: {recall_results["overall_recall"]:.4f} '
                  f'(target: {recall_results["target_recall"]:.2f}) [{status}]')
            print(f'  Key classes meet target: '
                  f'{recall_results["key_classes_meet_target"]}')
            for cls_name, cls_res in recall_results['per_class'].items():
                status = 'PASS' if cls_res['meets_target'] else 'FAIL'
                print(f'  {cls_name:<20}: {cls_res["recall"]:.4f} '
                      f'(GT: {cls_res["gt_count"]}, TP: {cls_res["tp_count"]}) '
                      f'[{status}]')

    # Print compact summary always
    print(f'\n[RESULT] mAP={report["summary"]["mAP"]:.4f} '
          f'NDS={report["summary"]["NDS"]:.4f}')
    if recall_results:
        status = 'PASS' if recall_results['meets_target'] else 'FAIL'
        print(f'[RECALL] {recall_results["overall_recall"]:.4f} '
              f'(target={recall_results["target_recall"]}) [{status}]')


if __name__ == '__main__':
    main()
