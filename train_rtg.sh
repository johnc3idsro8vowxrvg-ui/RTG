#!/usr/bin/env bash
# ===========================================================================
# train_rtg.sh — RTG CenterPoint Training Script
#
# 训练 CenterPoint VoxelNet on RTG BEV dataset.
#
# 用法:
#   # 单 GPU 训练
#   bash train_rtg.sh
#
#   # 多 GPU 分布式训练
#   bash train_rtg.sh --gpus 4
#
#   # 从断点恢复
#   bash train_rtg.sh --resume-from work_dirs/rtg_centerpoint_voxelnet/epoch_5.pth
# ===========================================================================

set -euo pipefail

# 项目根目录
cd "$(dirname "$(readlink -f "$0")")"
PROJECT_ROOT="$(pwd)"
CENTERPOINT="$PROJECT_ROOT/CenterPoint"

# 默认配置
CONFIG="configs/rtg/rtg_centerpoint_voxelnet.py"
GPUS=1

PYTHON="${PROJECT_ROOT}/.venv/bin/python3"
# 如果 conda auto 环境存在，使用它
if command -v conda &>/dev/null; then
    if conda env list | grep -q '^auto '; then
        eval "$(conda shell.bash hook)"
        conda activate auto
        PYTHON=$(which python)
    fi
fi

cd "$CENTERPOINT"

echo "=========================================="
echo "RTG CenterPoint Training"
echo "=========================================="
echo "  Config:     $CONFIG"
echo "  GPUs:       $GPUS"
echo "  Work dir:   $CENTERPOINT/work_dirs/rtg_centerpoint_voxelnet"
echo "  Python:     $PYTHON"
echo "=========================================="

$PYTHON tools/train_rtg.py --config "$CONFIG" --gpus "$GPUS" "$@"
