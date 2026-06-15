# RTG BEV 大车防撞感知系统

基于 CenterPoint (CVPR 2021) 的轮胎吊（RTG）大车防撞 BEV LiDAR 感知系统。

**当前阶段**: Phase 2 完成（478帧 8传感器全量提取，集卡侧+禁行侧数据就位，LiDAR-only 222ms 推理通过，RViz 全传感器可视化可用），等待标定进入 Phase 3。

> **运行环境**: 本项目在 conda 环境 `auto` 中开发和运行。执行任何 Python 脚本或推理前，需先 `conda activate auto`。

---

## 项目概述

为港口轮胎式集装箱门式起重机（RTG）构建面向大车防撞的 BEV 感知系统。系统输入集卡车道侧 2 个 OS1 128 线激光雷达 + 2 个海康工业相机，输出 3D/BEV 障碍物检测结果和分级碰撞风险预警。

| 项目 | 说明 |
|---|---|
| 检测类别 | person / truck / car / other_obstacle（4 类） |
| BEV 范围 | x∈[-60,60]m, y∈[-25,25]m, z∈[-3,8]m |
| 感知距离 | ≥50m（沿大车道方向） |
| 推理延迟 | ≤150ms |
| 输出帧率 | ≥10Hz |
| 通信中间件 | ROS1 |
| 训练平台 | 远程 A800 GPU 服务器 |
| 开发平台 | Ubuntu 20.04 + RTX 3060 (12GB) |
| 部署平台 | 工控机（GPU 型号待定） |

---

## 快速开始

### 1. 环境安装

```bash
# 创建 conda 环境
conda create -n auto python=3.8 -y
conda activate auto
conda install cudatoolkit=11.3 -y

# 安装 PyTorch
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 安装 spconv
pip install spconv-cu113

# CenterPoint 依赖
cd CenterPoint
pip install -r requirements.txt

# 其他依赖
pip install nuscenes-devkit scikit-image plyfile trimesh networkx numba
```

### 2. 数据准备

```bash
# NuScenes mini（验证用）：放入 data/nuscenes/，包括 samples/sweeps/v1.0-mini/
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes --version v1.0-mini
python tools/combine_view_info.py

# RTG rosbag（训练用，待到位）:
# python tools/data_converter/rosbag_extract.py input.bag -o data/rtg/extracted/
# python tools/data_converter/generate_rtg_infos.py -d data/rtg/extracted/ -a annotations.json
```

### 3. 训练

```bash
# RTG CenterPoint 训练
bash train_rtg.sh
```

### 4. 评估

```bash
python tools/eval_rtg.py --gt data/rtg/rtg_infos_val.pkl --pred results.json \
    --class-names person truck car other_obstacle --distance-bins 10,25,50
```

### 5. 部署推理 (TODO: 待工控机就绪)

```bash
python nodes/rtg_bev_node.py --config config/system.yaml
```

---

## 目录结构

```
RTG_BEV/
├── README.md                                     # 本文件
├── CenterPoint/                                  # CenterPoint 训练框架 (det3d)
│   ├── det3d/
│   │   ├── models/
│   │   │   ├── detectors/voxelnet.py             # VoxelNet 检测器
│   │   │   ├── backbones/scn.py                  # SpMiddleResNetFHD
│   │   │   ├── necks/rpn.py                      # RPN 特征金字塔
│   │   │   ├── bbox_heads/center_head.py         # CenterHead 检测头
│   │   │   └── readers/voxel_encoder.py          # VFE
│   │   └── datasets/
│   │       ├── nuscenes/                         # NuScenes 数据集
│   │       ├── pipelines/                        # 数据处理管道
│   │       └── rtg/                              # [新建] RTG 数据集适配
│   │           ├── rtg_dataset.py                # RTGDataset 类
│   │           └── rtg_common.py                 # RTG 工具函数
│   ├── configs/
│   │   └── rtg/
│   │       └── rtg_centerpoint_voxelnet.py       # RTG 训练配置
│   └── tools/
│       └── train_rtg.py                          # RTG 训练入口
├── CUDA-CenterPoint/                             # CUDA/TensorRT 推理加速
├── config/                                       # 项目配置文件
│   ├── calib.yaml.example                        # 标定参数模板
│   ├── geometry.yaml                             # RTG 几何与车道配置
│   ├── warning.yaml                              # 预警参数（支持热更新）
│   ├── system.yaml                               # 系统/部署配置
│   └── rtg_all_sensors.rviz                      # RViz 全传感器可视化配置
├── tools/                                        # 数据与评估工具
│   ├── data_converter/
│   │   ├── rosbag_extract.py                     # ROS1 rosbag 抽取
│   │   ├── generate_rtg_infos.py                 # → info.pkl 格式转换
│   │   ├── project_3d_to_2d.py                   # 3D→2D 投影可视化
│   │   └── calib_validator.py                    # 标定验证
│   ├── rtg_eval.py                               # RTG 自定义评估
│   ├── eval_rtg.py                               # 离线评估 CLI
│   ├── _extract_new_bag.py                       # 新 bag 全量提取
│   ├── _tf_republisher.py                        # TF 树 + 点云 frame_id 修正
│   ├── _visualize_bev.py                         # BEV 鸟瞰图可视化
│   └── play_bag_rviz.sh                          # 一键 RViz 回放
├── postprocessing/                               # 后处理模块
│   ├── tracker.py                                # 短时目标跟踪
│   ├── ego_motion.py                             # ICP 运动估计
│   ├── warning_engine.py                         # 分级预警引擎
│   ├── footprint_filter.py                       # 自车点云过滤
│   └── config_loader.py                          # 配置加载+验证
├── nodes/                                        # ROS1 节点
│   └── rtg_bev_node.py                           # 主节点
├── rtg_bev_msgs/                                 # ROS1 自定义消息包
├── train_rtg.sh                                  # 训练脚本
└── doc/                                          # 方案文档
```

---

## 传感器配置

### 集卡车道侧（第一版主模型）

| 位置 | 传感器 | 型号 | 关键参数 |
|---|---|---|---|
| 前支腿 | 相机 C1 | 海康 DS-2XC64ZZY-SHZJ-4mm | ~4m高, 竖屏1080×1920, 俯视图锥角46° |
| 后支腿 | 相机 C2 | 同上 | 同上 |
| 前支腿 | 雷达 L1 | OS1 128线 | ~1.5m高, 10Hz, 360°HFOV |
| 后支腿 | 雷达 L2 | OS1 128线 | 同上 |

### 禁行侧（第一版仅录制，延后处理）

| 位置 | 传感器 | 型号 | 关键参数 |
|---|---|---|---|
| 前支腿 | 相机 C3 | 同上 | ~4m高 |
| 后支腿 | 相机 C4 | 同上 | 同上 |
| 前支腿 | 雷达 L3 | RoboSense 32线 | ~1.5m高, 旋转90°安装 |
| 后支腿 | 雷达 L4 | RoboSense 32线 | 同上 |

---

## BEV 坐标系

```
原点: L1（集卡侧前雷达）在地面的垂直投影
+x: 大车道方向（C1 朝向，RTG 可沿 ±x 双向行驶）
+y: 跨距方向（禁行侧 → 集卡侧），与 ROS 惯例一致
+z: 竖直向上
```

传感器 BEV 坐标详见 [几何空间与传感器FOV参考手册](doc/几何空间与传感器FOV参考手册.md)。

---

## 检测类别

| ID | 类别 | 说明 |
|---|---|---|
| 0 | `person` | 人员（行人、作业人员） |
| 1 | `truck` | 集卡（整体，含牵引车+挂车+载箱） |
| 2 | `car` | 乘用车/小型车辆 |
| 3 | `other_obstacle` | 其他近地障碍物（邻车RTG、AGV/IGV、叉车、锥桶等） |

---

## 预警等级

| 等级 | 含义 | 触发方式 |
|---|---|---|
| 0 | 无风险 | — |
| 1 | 提示/关注 | 连续 3 帧确认 |
| 2 | 警告 | 连续 3 帧确认 |
| 3 | 危险 | 单帧即时触发 |

RTG 静止时不预警（仅输出检测结果）。所有阈值配置化，详见 [config/warning.yaml](config/warning.yaml)。

---

## ROS1 接口

### 输入（订阅）

| Topic | 类型 | 说明 |
|---|---|---|
| `/Camera_Raw_Img_01` | `sensor_msgs/Image` | 集卡侧前相机 |
| `/Camera_Raw_Img_02` | `sensor_msgs/Image` | 集卡侧后相机 |
| `/ouster1/points` | `sensor_msgs/PointCloud2` | 集卡侧前雷达 |
| `/ouster2/points` | `sensor_msgs/PointCloud2` | 集卡侧后雷达 |

### 输出（发布）

| Topic | 类型 | 频率 |
|---|---|---|
| `/rtg_bev/detections` | `rtg_bev_msgs/DetectionArray` | ≥10Hz |
| `/rtg_bev/tracks` | `rtg_bev_msgs/TrackArray` | ≥10Hz |
| `/rtg_bev/warnings` | `rtg_bev_msgs/WarningArray` | ≥10Hz |
| `/rtg_bev/ego_motion` | `rtg_bev_msgs/EgoMotionState` | ≥10Hz |
| `/rtg_bev/diagnostics` | `diagnostic_msgs/DiagnosticArray` | 1Hz |

详见 [接口协议](doc/接口协议.md)。

---

## 开发阶段

| 阶段 | 内容 | 状态 |
|---|---|---|
| **Phase 1** | 环境搭建 + 代码改造 + 验证 | ✅ 全部通过（2026-05-18） |
| **Phase 2** | 数据提取 + 传感器验证 + Pipeline 验证 | ✅ 完成（2026-05-19 新数据就位, 336帧 4cam+4lidar） |
| **Phase 3** | 标注 (~5000 帧) | ⬜ 阻塞：标定 + 标注工具 |
| **Phase 4** | 训练 + 迭代优化 | ⬜ 阻塞：Phase 3 |
| **Phase 5** | 后处理 + ROS1 集成联调 | ⬜ 阻塞：Phase 4 |
| **Phase 6** | 现场部署 + 验收 | ⬜ 阻塞：Phase 5 |

详见 [开发计划](doc/开发计划.md)。

---

## 文档索引

所有方案文档统一存放在 [doc/](doc/) 目录下。总入口为 [doc/大车防撞项目知识文档.md](doc/大车防撞项目知识文档.md)，其中包含完整的子文档目录和引用关系。

| 文档 | 说明 |
|---|---|
| [大车防撞项目知识文档](doc/大车防撞项目知识文档.md) | **总入口** — 项目目标、关键决策摘要、文档索引 |
| [几何空间与传感器FOV参考手册](doc/几何空间与传感器FOV参考手册.md) | 场景布局、BEV坐标系、传感器位置/FOV、镜像对称关系（几何空间唯一权威来源） |
| [总体技术方案](doc/总体技术方案.md) | 架构总览、模块划分、数据流、部署架构、风险清单 |
| [数据方案](doc/数据方案.md) | 数据采集、rosbag 抽取、格式转换管线 |
| [标注规范](doc/标注规范.md) | 标注工具、3D 框规则、质检流程 |
| [标定需求](doc/标定需求.md) | 7 项标定要求、精度、验收方式 |
| [接口协议](doc/接口协议.md) | ROS1 topic 定义、消息格式 |
| [配置说明](doc/配置说明.md) | 5 份配置文件格式、加载、热更新策略 |
| [测试验收方案](doc/测试验收方案.md) | 8 个现场测试用例、离线评估方案 |
| [开发计划](doc/开发计划.md) | 6 阶段开发计划、里程碑 |
| [Phase2_数据分析报告](doc/Phase2_数据分析报告.md) | Phase 2 数据质量分析与 Pipeline 验证报告 |

---

## 模型

### 架构

基于 CenterPoint (CVPR 2021)，LiDAR-only 单分支：

```
雷达输入:  Voxelization → SpMiddleResNetFHD (SparseConv3D) → RPN (FPN)
检测头:    CenterHead (Heatmap + Regression) → 3D Boxes
```

### 权重

| 权重文件 | 用途 |
|---|---|
| `CenterPoint/work_dirs/rtg_centerpoint_voxelnet/epoch_20.pth` | (TODO) RTG 训练权重 |

### 评估指标 (TODO: 训练后填写)

| 指标 | 目标 | 实际 |
|---|---|---|
| mAP | — | — |
| NDS | — | — |
| 关键目标召回率@50m | ≥98% | — |
| 推理延迟 | ≤150ms | — |

---

## 配置

所有阈值配置化，不硬编码。详见 [配置说明](doc/配置说明.md)。

- **calib.yaml** — 标定参数（标定团队提供，不提交 git）
- **calib_from_bag.yaml** — Bag 估计标定（2026-06-11，4雷达RPY+8传感器位置，RViz视觉对齐，提交 git）
- **geometry.yaml** — RTG 几何与车道（现场测量，不提交 git）
- **warning.yaml** — 预警参数（算法配置，支持热更新，提交 git）
- **system.yaml** — 系统/部署（部署环境，不提交 git）

启动时自动验证：外参矩阵有效性、内参合理性、距离阈值单调性、模型文件存在性。

---

## 测试

### 离线评估

```bash
python tools/eval_rtg.py \
    --gt data/rtg/rtg_infos_val.pkl \
    --pred work_dirs/results.json \
    --class-names person truck car other_obstacle \
    --distance-bins 10,25,50 \
    --output eval_report.json
```

### 现场测试用例

| 编号 | 场景 |
|---|---|
| TC-01 | 集卡从前方正常驶近 → 检测+分级预警 |
| TC-02 | 集卡从后方驶近 → 正确识别预警方向 |
| TC-03 | 人员进入集卡车道 → 8m内单帧危险触发 |
| TC-04 | RTG 静止 → 检测但不预警 |
| TC-05 | 多目标同时在场 → 独立跟踪 |
| TC-06 | 运动状态推断 → 方向正确切换 |
| TC-07 | 静止目标 → 不误报 |
| TC-08 | 系统性能 → 延迟≤150ms, 帧率≥10Hz |

详见 [测试验收方案](doc/测试验收方案.md)。

---

## 部署 (TODO: Phase 6 后完善)

- 平台：工控机（GPU 型号待定）
- 系统：Ubuntu + ROS1
- 优化：优先 PyTorch 原生推理，按需评估 TensorRT

---

## RViz 全传感器可视化

支持直接回放 bag 并在 RViz 中查看 4 相机 + 4 雷达数据：

```bash
# 一键启动：roscore + rosbag play + TF republisher + RViz
bash tools/play_bag_rviz.sh

# 暂停模式（逐帧查看）
bash tools/play_bag_rviz.sh --pause

# 循环播放
bash tools/play_bag_rviz.sh --loop
```

**工作原理**:
- `tools/_tf_republisher.py` — 发布 `rtg_bev_origin` → 各传感器帧的静态 TF，并修正点云 frame_id 为唯一名称
- `config/rtg_all_sensors.rviz` — RViz 布局配置（3D 视图 4 雷达 + 侧面板 4 相机）
- 点云颜色区分：红=L1 (集卡侧前), 绿=L2 (集卡侧后), 蓝=L3 (禁行侧前), 橙=L4 (禁行侧后)

## 日志与回放

- 检测/跟踪/预警事件日志
- 预警前后 N 秒数据截取（用于复盘）
- 支持 ROS1 rosbag 回放模式（`replay.enabled: true`）

---

## 依赖

| 依赖 | 版本（开发环境） | 说明 |
|---|---|---|
| Python | 3.8 | conda env: auto |
| PyTorch | 1.10.0+cu113 | RTX 3060 (Ampere) |
| CUDA | 11.3 | Driver 535.183.01 |
| torchvision | 0.11.1+cu113 | |
| spconv | 2.3.6 (spconv-cu113) | |
| nuscenes-devkit | 1.1.11 | 数据转换 |
| numba | — | JIT 加速（数据预处理） |
| Open3D | — | 运动估计（有降级模式） |
| ROS1 | noetic | 部署时（代码有优雅降级） |

---

## 变更记录

| 日期 | 版本 | 变更 |
|---|---|---|
| 2026-05-21 | v0.6 | **FOV 重大修正**：相机俯视图锥角 87.6°→46°，HFOV/VFOV 方向交换；L3/L4 型号 Helios 32→RoboSense 32线；禁行侧大车道宽度 2.37m→3.0m；L1/L2 频率 20Hz→10Hz，高度 1.6m→1.5m |
| 2026-05-21 | v0.5 | 文档重构：知识文档精简为总入口，详细信息下沉子文档；合并 BEV_俯视图 入 FOV参考手册；全部文档移入 doc/；修正对称模型为镜像对称 |
| 2026-05-20 | v0.4 | 禁行侧传感器提取完成（478帧 8传感器），全传感器 RViz 可视化就绪（`tools/play_bag_rviz.sh`） |
| 2026-05-18 | v0.3 | 本地开发环境搭建完成（conda auto, PyTorch 1.10+cu113, mmcv 1.3.16, spconv 2.3.6），NuScenes mini 数据就绪，CenterPoint 管线验证通过 |
| 2026-05-18 | v0.2 | Phase 1 代码改造完成（32 文件, ~8000 行） |
| 2026-05-15 | v0.1 | 11 份方案文档完成，43 轮决策确认 |

---

## 团队

| 角色 | 负责范围 |
|---|---|
| 项目负责人/感知算法专家 | 总体架构、技术决策、代码审查 |
| Agent 1 | CenterPoint 模型核心代码改造 |
| Agent 2 | RTG 数据管线开发（Dataset + Pipeline） |
| Agent 3 | 工具脚本 + 自定义评估代码 |
| Agent 4 | 训练入口 + NuScenes 回归验证 |
| Agent 5 | ROS1 消息包 + 后处理模块（跟踪/运动/预警） |
| 环境搭建 Agent | ✅ CUDA/PyTorch/spconv/mmdet3d 全家桶安装完成 |

---

## 引用

本项目基于 CenterPoint (CVPR 2021):

```bibtex
@article{yin2021center,
  title={Center-based 3D Object Detection and Tracking},
  author={Yin, Tianwei and Zhou, Xingyi and Kr\"{a}henb\"{u}hl, Philipp},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2021}
}
```
