"""
RTG-specific utilities and class mappings for CenterPoint/det3d integration.

Provides:
  - general_to_detection: class name mapping for evaluation output
  - cls_attr_dist: attribute distribution (placeholder, RTG has few attributes)
  - RTG_CLASS_NAMES: canonical 4-class list
"""

RTG_CLASS_NAMES = ('person', 'truck', 'car', 'other_obstacle')

# RTG 4-class name mapping for NuScenes-style evaluation
general_to_detection = {
    'person': 'pedestrian',
    'truck': 'truck',
    'car': 'car',
    'other_obstacle': 'barrier',
}

cls_attr_dist = {
    'pedestrian': {'pedestrian.moving': 1.0},
    'truck': {'vehicle.moving': 1.0},
    'car': {'vehicle.moving': 1.0},
    'barrier': {'cycle.with_rider': 1.0},
}

# Reverse mapping for internal use
DETECTION_TO_CLASS = {v: k for k, v in general_to_detection.items()}
