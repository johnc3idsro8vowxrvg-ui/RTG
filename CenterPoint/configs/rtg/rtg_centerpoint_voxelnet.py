"""
RTG CenterPoint VoxelNet configuration.

Based on CenterPoint/configs/nusc/voxelnet/nusc_centerpoint_voxelnet_0075voxel_fix_bn_z.py
adapted for RTG gantry crane BEV scenario.

Differences from NuScenes config:
  - 4 classes (single task): person, truck, car, other_obstacle
  - point_cloud_range: [-60.0, -25.2, -3.0, 60.0, 25.2, 8.0]
  - nsweeps=1 (single-frame mode)
  - No GT-AUG (DB sampler disabled, RTG has limited samples)
  - RTGDataset dataset type
"""

import logging

from det3d.utils.config_tool import get_downsample_factor

tasks = [
    dict(num_class=4, class_names=["person", "truck", "car", "other_obstacle"]),
]

class_names = list(tasks[0]["class_names"])

# training and testing settings
target_assigner = dict(
    tasks=tasks,
)

# Point cloud range: x=[-60,60], y=[-25.2,25.2], z=[-3,8]
# Voxel grid: [1600, 672, 55] for voxel size [0.075, 0.075, 0.2]
point_cloud_range = [-60.0, -25.2, -3.0, 60.0, 25.2, 8.0]
voxel_size = [0.075, 0.075, 0.2]

model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(
        type="VoxelFeatureExtractorV3",
        num_input_features=5,  # x, y, z, intensity, timestamp(zeros)
    ),
    backbone=dict(
        type="SpMiddleResNetFHD",
        num_input_features=5,
        ds_factor=8,
    ),
    neck=dict(
        type="RPN",
        layer_nums=[5, 5],
        ds_layer_strides=[1, 2],
        ds_num_filters=[128, 256],
        us_layer_strides=[1, 2],
        us_num_filters=[256, 256],
        num_input_features=256,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="CenterHead",
        in_channels=sum([256, 256]),
        tasks=tasks,
        dataset='nuscenes',  # Use nuscenes format (box 9-dim with vel)
        weight=0.25,
        code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2, 1.0, 1.0],
        common_heads={'reg': (2, 2), 'height': (1, 2), 'dim': (3, 2), 'rot': (2, 2), 'vel': (2, 2)},
        share_conv_channel=64,
        dcn_head=False,
    ),
)

assigner = dict(
    target_assigner=target_assigner,
    out_size_factor=get_downsample_factor(model),
    dense_reg=1,
    gaussian_overlap=0.1,
    max_objs=500,
    min_radius=2,
)

train_cfg = dict(assigner=assigner)

test_cfg = dict(
    post_center_limit_range=[-61.2, -26.4, -10.0, 61.2, 26.4, 10.0],
    max_per_img=500,
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=83,
        nms_iou_threshold=0.2,
    ),
    score_threshold=0.1,
    pc_range=point_cloud_range[:2],
    out_size_factor=get_downsample_factor(model),
    voxel_size=voxel_size[:2],
)

# dataset settings
dataset_type = "RTGDataset"
nsweeps = 1
data_root = "data/rtg/"

train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    global_rot_noise=[-0.3925 * 2, 0.3925 * 2],
    global_scale_noise=[0.9, 1.1],
    global_translate_std=0.5,
    db_sampler=None,
    class_names=class_names,
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
)

voxel_generator = dict(
    range=point_cloud_range,
    voxel_size=voxel_size,
    max_points_in_voxel=10,
    max_voxel_num=[120000, 160000],
)

# Note: Point cloud loading is handled directly in RTGDataset.get_sensor_data(),
# so no LoadPointCloudFromFile step is needed in the pipeline.
train_pipeline = [
    dict(type="LoadRTGAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
test_pipeline = [
    dict(type="LoadRTGAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignLabel", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]

train_anno = data_root + "rtg_infos_train.pkl"
val_anno = data_root + "rtg_infos_val.pkl"
test_anno = None

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        test_mode=True,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        test_mode=True,
        nsweeps=nsweeps,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
)

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
optimizer = dict(
    type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,
)
lr_config = dict(
    type="one_cycle", lr_max=0.001, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,
)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=5,
    hooks=[
        dict(type="TextLoggerHook"),
    ],
)
total_epochs = 20
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = './work_dirs/{}/'.format(__file__[__file__.rfind('/') + 1:-3])
load_from = None
resume_from = None
workflow = [('train', 1)]
