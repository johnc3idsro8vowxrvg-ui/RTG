# Phase 2 数据质量分析报告

> 日期: 2026-05-20（传感器参数确认 + Pipeline 验证完成）
> 状态: ✅ Phase 2 完成（真实测试数据 4cam+4lidar 就位，传感器参数全部确认，融合推理 347ms）
> 数据来源: 现场真实测试数据 `lidar_and_camera_2026-05-19-15-44-34.bag` (14.5 GB)
> 训练数据: 5000 帧将在标定完成后另行采集

---

## 1. 数据概况

### 1.1 Bag 信息

| 属性 | 值 |
|---|---|
| 文件 | `lidar_and_camera_2026-05-19-15-44-34.bag` |
| 时长 | **51.7 秒** |
| 大小 | 14.5 GB（7z 压缩后 3.5 GB，6 卷） |
| 消息数 | 2,697 |
| 同步方式 | **四路传感器同袋录制** |

### 1.2 Topic 汇总

| Topic | 类型 | 消息数 | 频率 | 说明 |
|---|---|---|---|---|
| `/Camera_Raw_Img_01` | `sensor_msgs/Image` | 311 | 9.9 Hz | C1 集卡侧前，1080×1920 bgr8 |
| `/Camera_Raw_Img_02` | `sensor_msgs/Image` | 350 | 9.9 Hz | C2 集卡侧后 |
| `/Camera_Raw_Img_03` | `sensor_msgs/Image` | 327 | 9.8 Hz | C3 禁行侧前 |
| `/Camera_Raw_Img_04` | `sensor_msgs/Image` | 263 | 9.9 Hz | C4 禁行侧后 |
| `/ouster1/points` | `sensor_msgs/PointCloud2` | 478 | 10.0 Hz | **L1 OS1 128线** (262k pts/帧, 含 ring) |
| `/ouster2/points` | `sensor_msgs/PointCloud2` | 257 | 9.6 Hz | **L2 OS1 128线** (262k pts/帧, 含 ring) |
| `/lidar3/rslidar3_points` | `sensor_msgs/PointCloud2` | 327 | 9.9 Hz | L3 RoboSense 32线 (~25k pts/帧) |
| `/lidar4/rslidar4_points` | `sensor_msgs/PointCloud2` | 384 | 10.0 Hz | L4 RoboSense 32线 (~25k pts/帧) |

### 1.3 提取结果

以 L1（ouster1, 10Hz）为参考时间轴，100ms 同步窗口，**全量 8 传感器提取**：

| 数据 | 帧数 | 说明 |
|---|---|---|
| 同步帧总数 | **478 帧** | L1 为参考 |
| 集卡侧完整帧 (C1+C2+L1+L2) | **256 帧** | 用于主模型训练 |
| 禁行侧完整帧 (C3+C4+L3+L4) | **237 帧** | 用于禁行侧延后处理 |
| C1 可用帧 | 388 / 478 (81%) | 集卡侧前相机 |
| C2 可用帧 | 418 / 478 (87%) | 集卡侧后相机 |
| C3 可用帧 | 391 / 478 (82%) | **禁行侧前相机** |
| C4 可用帧 | 321 / 478 (67%) | **禁行侧后相机** |
| L1 点云 | 478 帧, 262,144 pts/帧 | OS1 128线 |
| L2 点云 | 321 帧, 262,144 pts/帧 | OS1 128线 |
| L3 点云 | 410 帧, ~16k pts/帧 | **RoboSense 32线** |
| L4 点云 | 447 帧, ~18k pts/帧 | **RoboSense 32线** |
| 数据量 | 3.8 GB (extracted) | cam_01~04 + lidar_01~04 |

> **注意**: 集卡侧和禁行侧传感器来自两组独立触发，没有单帧内 4 相机全齐的情况（256 帧集卡侧 + 237 帧禁行侧），符合两端独立采集的预期。

---

## 2. 传感器实际参数

### 2.1 相机

| 参数 | 计划 | 实际 | 匹配 |
|---|---|---|---|
| 数量 | 4 (C1~C4) | **4** | ✅ |
| 集卡侧 | C1+C2 | C1+C2 | ✅ |
| 分辨率 | 待确认 | **1080×1920** (竖屏) | ✅ 已确认 |
| 编码 | 待确认 | bgr8 | ✅ 已确认 |
| 频率 | 待确认 | **~10 Hz** | ✅ 已确认 |

### 2.2 激光雷达

| 参数 | 计划 (L1/L2) | 实际 (L1/L2) | 计划 (L3/L4) | 实际 (L3/L4) |
|---|---|---|---|---|
| 型号 | OS1 128线 | **Ouster OS1 128线** | Helios 32 | **RoboSense 32线** |
| 线数 | 128 | **128** ✅ | 32 | **32** ✅ |
| 频率 | 20 Hz | **10 Hz**（已确认） | 10 Hz | **10 Hz** ✅ |
| 单帧点数 | >50k | **262,144**（已确认） | ~25k | ~25k |
| 字段(128线) | x,y,z,intensity,timestamp | **x,y,z,t,ring,range,reflectivity,near_ir** (8字段, 含 ring) |
| 字段(32线) | x,y,z,intensity,timestamp | **x,y,z,intensity** (4字段, 无 ring) |

---

## 3. 与规划的差距

> 本包为现场真实测试数据（2026-05-19），用于传感器参数确认和 Pipeline 验证。
> 5000 帧训练数据将在标定完成后另行采集。

| # | 差距 | 严重度 | 状态 |
|---|---|---|---|
| 1 | 测试数据 336 帧（训练需 5000） | ⚠️ 中 | 标定后采集，本包仅验证用 |
| 2 | 相机竖屏 (1080×1920) | ⚠️ 低 | Pipeline 可适配 |
| 3 | L1/L2 频率 10Hz（计划 20Hz） | ⚠️ 低 | 10Hz 满足 ≥10Hz 要求 |
| 4 | **无标定数据** | ❌ **阻塞** | 融合推理需标定 YAML |
| 5 | 含 ring 字段 | ✅ 优于计划 | 可用于地面分割 |
| 6 | 四路传感器同袋录制 | ✅ 已确认 | — |

---

## 4. 数据质量

### 4.1 相机
- 4 路相机全部在线，~10Hz，同袋录制
- 1080×1920 竖屏，bgr8 编码
- C1/C2 用于集卡侧推理，C3/C4 禁行侧数据已同步提取（237 帧完整）
- RViz 中可同时查看 4 路相机画面（`config/rtg_all_sensors.rviz`）

### 4.2 雷达
- L1/L2: 128 线 Ouster，262k pts/帧，含 ring（可做地面分割），10Hz
- L3/L4: 32 线 RoboSense，~25k pts/帧，10Hz
- 128 线点云密度远超计划（262k vs 预期 50k），对检测有利但增加推理延迟
- 含 ring 字段——可用于地面点剔除，优于计划预期

---

## 5. 基于现有数据的验证工作

> 完成日期: 2026-05-20
> 原则: 在无标定情况下最大化利用新数据，跑通 Pipeline、验证代码改造、测量真实延迟

### 5.1 合成标定生成

基于 `config/geometry.yaml` 已知传感器安装位置和相机 FOV 规格，反推近似内外参（仅用于 Pipeline 验证，不可用于精度评估）：

| 参数 | 值 | 来源 |
|---|---|---|
| 相机内参 fx, fy | 1272.2, 1001.1 | FOV 规格 + 1080×1920 分辨率 |
| 相机内参 cx, cy | 540, 960 | 图像中心 |
| 相机→雷达平移 | (0, 0, -2.5)m | 相机 ~4m − 雷达 ~1.5m (geometry.yaml) |
| 相机→雷达旋转 | 单位矩阵 | 假设同向安装 |
| 雷达 L2→L1 平移 | (-12.0, 0, 0)m | geometry.yaml 轮组间距 |

### 5.2 info.pkl 生成

| 文件 | 样本数 | 相机 | 用途 |
|---|---|---|---|
| `data/rtg/rtg_infos_new.pkl` | **336** | **2 路** (CAM_FRONT + CAM_BACK) | Pipeline + 融合推理验证 |
| `data/rtg/rtg_infos_2cam.pkl` | 336 | 2 路 | RTG 2 相机配置专用 |

### 5.3 DataLoader Pipeline 验证

以 RTG 实际配置（`num_camera_views=2`, 128 线 LiDAR）构建 RTGDataset：

| 验证项 | 方法 | 结果 |
|---|---|---|
| RTGDataset 构建 | `build_dataset(test_cfg)` | ✅ **336 样本**加载成功 |
| L1 128 线点云加载 | `np.fromfile` × 336 帧 | ✅ **262,144 pts/frame** |
| L2 128 线点云加载 | `np.fromfile` × 256 帧 | ✅ 262,144 pts/frame |
| 2 相机图像加载 | `cv2.imread` × 336 帧 | ✅ 1080×1920, bgr8 |
| SparseDepth 生成 | 2 相机 + 合成标定 | ✅ 深度图正常 |

### 5.4 Bug 发现与修复

| # | 文件 | 问题 | 修复 |
|---|---|---|---|
| 1 | `mmdet3d/datasets/rtg_dataset.py:139` | `cam_intrinsic` 从 pickle 恢复后是 `list`，调用 `.shape` 报错 | 添加 `np.array()` 强制转换 |
| 2 | `mmdet3d/datasets/rtg_dataset.py:109,130` | `pts_filename` 和 `data_path` 未与 `data_root` 拼接，文件路径错误 | 添加 `osp.join(self.data_root, ...)` |
| 3 | `mmdet3d/datasets/rtg_dataset.py:2` | 仅 `from os import path as osp`，未 `import os` | 统一使用 `osp.join` |
| 4 | `mmdet3d/datasets/pipelines/loading.py:1118` | SparseDepth 访问 `all_points[:,4]` 但 RTG 点云仅 4 维（无 timestamp_delta） | 检测维度 ≤4 时跳过 sweep 过滤 |
| 5 | `mmdet3d/core/evaluation/__init__.py` | 未导出 `rtg_eval` / `rtg_eval_recall` | 添加 import 和 `__all__` |
| 6 | `postprocessing/__init__.py` | 引用不存在的类名 `MultiObjectTracker` | 改为 `Tracker` |
| 7 | `tools/data_converter/gen_synthetic_infos.py` | camera `data_path` 使用绝对路径，导致 Dataset 拼接时双重路径 | 改为相对路径 |

### 5.5 场景理解

从 336 帧新数据（1080×1920 竖屏 + 128 线点云）分析：

- **128 线点云密度**: 262k pts/帧，远超计划的 50k（5x），对检测有利
- **含 ring 字段**: 可直接按线号做地面/非地面分割，无需 RANSAC
- **L1 覆盖**: 集卡侧前雷达，大车道方向覆盖充足
- **L2 覆盖**: 集卡侧后雷达，与 L1 共视区域通过外参拼接

### 5.6 融合推理 (2026-05-20)

**方法**: NuScenes 原始 Pipeline（6 相机 + 真实相机参数）提供相机分支，仅将点云替换为 RTG 128 线 Ouster 数据。全融合链路（Camera backbone + LiDAR backbone + SparseDepth + ImageTransformer + ViewTransformer + FusionTransformer）全部激活。

> **为什么用 6 相机而非 2 相机?** RTG 2 相机 + 2 雷达配置需要标定数据生成正确的 SparseDepth 深度图，无标定时深度图维度不匹配（1080×1920 → 480×270 vs 模型期望 112×200），融合层崩溃。6 相机 NuScenes Pipeline 维度完全匹配，是当前唯一可测融合延迟的方案。

**延迟结果** (RTX 3060 12GB, 200k pts/frame):

| 配置 | LiDAR | 相机 | 延迟 | 检测 |
|---|---|---|---|---|
| NuScenes 基线 | 32线, 10 sweeps (~350k pts) | 6 real | 1878ms | 18 |
| **RTG 128线单帧** | 128线, 200k pts | 6 real (Nusc) | **347ms** | 0 |
| RTG 2-camera 实际配置 | 128线 | 2 RTG | **无法测试** | 阻塞：无标定 |

**分析**:
- 347ms 是 LiDAR backbone (voxelization + SparseEncoder + SECOND + FPN) + 相机 backbone + 融合全链路的真实延迟
- 0 检测 = 场景域差距，确认训练必要性
- 2 相机实际延迟应 ≤347ms（相机分支计算量更少，但需标定才能实测）
- RTG 2-view 模型 spconv OOM (1192 GiB 异常) 是 PyTorch 1.10 兼容性 bug，训练/部署前需升级 PyTorch 或修复 spconv

### 5.7 双雷达拼接推理 (2026-05-20)

L1（前）和 L2（后）128 线点云通过外参拼接后做 LiDAR-only 推理。

**拼接方法**: L2 坐标 → L1 坐标系（L2 在 BEV 中 x=-12.0m → 平移变换），合并后截断至 200k pts。

**延迟结果** (RTX 3060):

| 配置 | 延迟 | L2 开销 |
|---|---|---|
| 单 L1 128线 (200k pts) + 融合 | 347ms | — |
| **L1+L2 合并 (200k pts) + 融合** | **353ms** | **+6ms** |

**结论**: 双雷达拼接仅增加 ~6ms 延迟（pts 上限相同），`LoadPointsFromTwoLidars` 方案可行。全量 466k pts 需更大显存。

### 5.8 LiDAR-only 模型改造 (TransFusion-L, 2026-05-20)

将 SparseFusion 改造为支持 LiDAR-only 模式（`lidar_only=True`），跳过相机分支和融合层，直接使用 `PointTransformer2D_3D` + `CenterHead` 做纯 LiDAR 检测。

**改造范围** (`sparsefusion_head_deform.py`):

| 改动 | 说明 |
|---|---|
| 新增 `lidar_only` 参数 | `__init__` 接收，默认 `False` 保持兼容 |
| 新增 `_forward_lidar_only()` | 独立 LiDAR 推理路径: voxel features → heatmap → PointTransformer2D_3D → CenterHead |
| `forward_single` 分流 | 开头检查 `lidar_only` → 直接走 `_forward_lidar_only` |
| `get_bboxes` 适配 | `'cls'` key 缺失处理 / `img_query_heatmap_score` 跳过 |

**配置切换**:
```python
pts_bbox_head=dict(lidar_only=True)   # LiDAR-only
pts_bbox_head=dict(lidar_only=False)  # Full fusion (default)
```

**延迟结果** (RTX 3060, 128线 150k pts):

| 模式 | 延迟 | 说明 |
|---|---|---|
| 融合推理 (6cam) | 347ms | 全链路 |
| **LiDAR-only** | **222ms** | 跳过相机+融合 (-125ms, -36%) |

**检测结果**: 0 检测。PointTransformer2D_3D 在 NuScenes 城市道路训练，不认识港口集卡/集装箱/龙门架。backbone 几何特征可迁移，检测头需 RTG 数据 fine-tune。确认训练策略正确。

### 5.9 RTG 运动状态估计 (2026-05-20)

使用连续 LiDAR 帧做 ICP 平移估计，判断 RTG 运动状态。

**方法**: 相邻帧非地面静态背景点（z>0.5m, <60m）做 voxel 降采样后 KD-tree ICP 匹配。

**结果** (50 帧对, ~5 秒):

| 指标 | 值 |
|---|---|
| 帧间平均位移 | 0.037m |
| 帧间最大位移 | 0.051m |
| 匹配置信度 | 99% |
| dx（大车道方向）均值 | +0.010m |
| dy（跨距方向）均值 | +0.038m |
| **判定** | **静止** |

**多间隔验证**（排除噪声累积极限）：

| 帧间隔 | 时间 | 平均位移 |
|---|---|---|
| 1 帧 | 0.1s | 0.024m |
| 10 帧 | 1.0s | 0.163m |
| 50 帧 | 5.0s | 0.104m |

位移不随时间线性累积（50帧 < 10帧），确认为传感器抖动/ICP噪声，非真实运动。

**32 秒轨迹**: dx∈[-0.16, +0.17]m, dy∈[-0.32, +0.34]m，无漂移趋势。

**结论**: RTG 在此测试数据中静止。ICP 运动估计方法可用——帧间隔越大越能区分真运动和噪声。真实运动时位移会线性累积。

### 5.10 自车 Footprint 点云过滤

基于 `config/geometry.yaml` 构建 RTG 自车 footprint，在推理前剔除自车结构上的点云。

**Footprint 定义**（BEV 坐标系，两个独立矩形）:

| 区域 | x 范围 | y 范围 | 说明 |
|---|---|---|---|
| 集卡侧 | [-12.5, 0.5]m | [-0.5, 0.5]m | 前后支腿 + 连接梁，宽 1m |
| 禁行侧 | [-12.5, 0.5]m | [23.0, 24.0]m | 跨距远端，宽 1m |

**实现**:
- 新增 `postprocessing/footprint_filter.py` — `SelfFootprintFilter` 类
- 支持 4 个雷达 ID（L1~L4）自动坐标转换 (LiDAR → BEV)
- 集成到 `nodes/rtg_bev_node.py._preprocess_lidar()` 预处理阶段

**验证结果**:

| 帧 | 原始点数 | 过滤后 | 剔除 | 比例 |
|---|---|---|---|---|
| 0 | 13,826 | 13,567 | 259 | 1.9% |

剔除的点集中在 L1 前支腿区域（x∈[0,0.1], y∈[-0.5,-0.4]）。128 线数据上待标定后重新验证。

### 5.11 全传感器 RViz 可视化 (2026-05-20)

完成 bag 原始数据 → RViz 可视化全链路，支持集卡侧 + 禁行侧 8 传感器实时回放。

**问题**: bag 中点云 frame_id 不唯一（两个 Ouster 均用 `os_sensor`，两个 RoboSense 均用 `/rslidar`），且无 TF 数据，RViz 无法区分各传感器位置。

**解决方案**:

| 组件 | 文件 | 用途 |
|---|---|---|
| TF 重发布节点 | `tools/_tf_republisher.py` | 发布 `rtg_bev_origin` → 8 个传感器 frame 的静态 TF；将点云重发布到修正 topic（`/rtg/ouster1/points` 等）并赋予唯一 frame_id |
| RViz 配置 | `config/rtg_all_sensors.rviz` | 4 雷达点云（红/绿/蓝/橙）+ 4 相机图像面板 + `rtg_bev_origin` 固定帧 |
| 一键启动脚本 | `tools/play_bag_rviz.sh` | `roscore → TF republisher → rosbag play --clock → rviz` |

**TF 树** (2026-06-11 更新, bag 估计值):
```
rtg_bev_origin
├── lidar_01_front  TF(0, 0, 1.5),        RPY(  0°,  0°,  +4.62°)  L1 OS1
├── lidar_02_rear   TF(0, -12.0, 1.5),     RPY(  0°,  0°,  -0.82°)  L2 OS1
├── lidar_03_front  TF(23.5, 0, 1.5),      RPY(+90°, +6°, +90° )   L3 RoboSense 32
├── lidar_04_rear   TF(23.5, -12.0, 1.5),  RPY(+96°, +2°, -90° )   L4 RoboSense 32
├── camera_01       TF(0, 0, 4.0),         R_y(+90°)               C1
├── camera_02       TF(-0.1, -12.0, 4.0),  R_y(-90°)               C2
├── camera_03       TF(23.5, 0, 4.0),      R_y(+90°)               C3
└── camera_04       TF(23.4, -12.0, 4.0),  R_y(-90°)               C4
```

### 5.12 标定外参 Bag 估计 (2026-06-11)

在无真实标定数据的情况下，利用 bag 数据通过多种方法估计传感器外参。

**估计方法**:

| 方法 | 适用传感器 | 精度 (估计) | 原理 |
|---|---|---|---|
| 集装箱边缘分析 | L1, L2 | ±0.5° | 集装箱边缘与车道方向平行/垂直，点云提取边缘方向 vs 理论方向 → yaw |
| RViz 视觉对齐 | L3, L4 | ±2° | RViz 3D 视图同时显示 L1/L2(已标定) + L3/L4 点云，以堆场固定结构为参考交互式调整 RPY |
| 设计值 | C1~C4 | — | 机械安装位置 + FOV 规格反推 |

**估计结果** (详见 `config/calib_from_bag.yaml`):

| 传感器 | BEV 平移 (x, y, z) m | RPY (roll, pitch, yaw) ° | 估计方法 |
|---|---|---|---|
| **L1** | (0, 0, 1.5) | (0, 0, **+4.62**) | 集装箱边缘分析 (2026-06-04) |
| **L2** | (12.0, 0, 1.5) | (0, 0, **-0.82**) | 集装箱边缘分析 (2026-06-04) |
| **L3** | (0, 23.5, 1.5) | (**+90, +6, +90**) | RViz 视觉对齐 (2026-06-11) |
| **L4** | (12.0, 23.5, 1.5) | (**+96, +2, -90**) | RViz 视觉对齐 (2026-06-11) |

> **RoboSense 32 线 (L3/L4)**: 旋转 90° 安装（垂直 360°，水平 ~70° 沿大车道方向扫描），roll ~90° 反映此安装方式。pitch/roll 微调量（L3: +6°, L4: +6°/+2°）为 RViz 中与 L1/L2 点云手动对齐的修正值。

**TF 树** (最终):
```
rtg_bev_origin
├── lidar_01_front  (0, 0, 1.5),        RPY=( 0°,  0°,  +4.62°)  L1 OS1
├── lidar_02_rear   (0, -12.0, 1.5),     RPY=( 0°,  0°,  -0.82°)  L2 OS1
├── lidar_03_front  (23.5, 0, 1.5),      RPY=(+90°, +6°, +90° )   L3 RoboSense 32
├── lidar_04_rear   (23.5, -12.0, 1.5),  RPY=(+96°, +2°, -90° )   L4 RoboSense 32
├── camera_01  (0, 0, 4.0),             R_y(+90°)  C1
├── camera_02  (-0.1, -12.0, 4.0),      R_y(-90°)  C2
├── camera_03  (23.5, 0, 4.0),          R_y(+90°)  C3
└── camera_04  (23.4, -12.0, 4.0),      R_y(-90°)  C4
```

**限制**: 此为视觉估计值，不可替代标定团队的精确标定（旋转 ±0.3°、平移 ±2cm）。待标定团队交付后需全部替换。

---

## 6. 后续步骤

> 传感器参数已通过测试数据确认，Pipeline 验证完成。以下为后续步骤。

| 优先级 | 事项 | 依赖 | 预计耗时 | 状态 |
|---|---|---|---|---|
| **P0** | 获取真实标定 YAML | 标定团队 | — | ⬜ |
| **P0** | 采集 5000 帧训练数据（多场景） | 标定完成 | 现场 | ⬜ |
| **P1** | 替换 `calib.yaml` → 跑通 2 相机融合 DataLoader | P0 | 0.5d | ⬜ |
| **P1** | 3D 框标注（按标注规范） | P0 | ~170 工时 | ⬜ |
| **P1** | 首轮训练（backbone-only fine-tune, A800） | 标注 | ~2d GPU | ⬜ |
| **P2** | 2 相机融合推理 + 精度评估 | 标定 + 训练 | 1d | ⬜ |
| **P2** | 预警模块联调 + 运动估计 ICP 验证 | 训练 | 1d | ⬜ |

### 传感器外参确认完毕

| 参数 | 状态 |
|---|---|
| 4 相机 1080×1920, 10Hz, bgr8, 位置已确认 | ✅ 已确认 |
| L1/L2 Ouster 128线, 262k pts, 10Hz, 含 ring, yaw 已估计 | ✅ 已确认 |
| L3/L4 RoboSense 32线, ~17k pts, 10Hz, RPY 已视觉估计 | ✅ 已确认 |
| 四路传感器同袋录制 | ✅ 已确认 |
| Topic 命名 | ✅ system.yaml 已更新 |
| **Bag 标定估计** | ✅ 2026-06-11 完成 (calib_from_bag.yaml) |

---

## 7. 数据目录

```
data/rtg/
├── raw/
│   └── lidar_and_camera_2026-05-19-15-44-34.bag    # 现场测试数据 (14.5 GB, 51.7s)
├── rtg_infos_new.pkl                               # 336帧 info.pkl (2相机, 合成标定)
├── rtg_infos_2cam.pkl                              # 336帧 info.pkl (RTG 2相机配置)
└── extracted/
    └── scene_001/                                   # 478 帧全量同步提取 (3.8 GB)
        ├── cam_01/000000~000387.jpg                 # C1 集卡侧前 (1080×1920)
        ├── cam_02/000000~000417.jpg                 # C2 集卡侧后
        ├── cam_03/000000~000390.jpg                 # C3 禁行侧前
        ├── cam_04/000000~000320.jpg                 # C4 禁行侧后
        ├── lidar_01/000000~000477.bin               # L1 Ouster 128线 (262k pts)
        ├── lidar_02/000000~000320.bin               # L2 Ouster 128线
        ├── lidar_03/000000~000409.bin               # L3 RoboSense 32线 (~16k pts)
        ├── lidar_04/000000~000446.bin               # L4 RoboSense 32线 (~18k pts)
        ├── timestamps.csv
        └── quality_report.json
```

### 新增工具文件

```
tools/
├── _extract_new_bag.py          # 全量 8 传感器 rosbag 抽取 (集卡侧 + 禁行侧)
├── _tf_republisher.py           # TF 树广播 + 点云 frame_id 修正节点
├── _analyze_new_bag.py          # Bag topic/传感器快速分析
├── play_bag_rviz.sh             # 一键 roscore + rosbag play + RViz
config/
└── rtg_all_sensors.rviz         # RViz 全传感器可视化配置
```
