#!/bin/bash
# =============================================================================
# RTG 全传感器 rosbag 回放 + RViz 可视化
# =============================================================================
# 用法:
#   bash tools/play_bag_rviz.sh          # 实时播放
#   bash tools/play_bag_rviz.sh --pause  # 暂停模式开始
#   bash tools/play_bag_rviz.sh --loop   # 循环播放
# =============================================================================

set -e

source /opt/ros/noetic/setup.bash

BAG='/media/shen/data/zk/Auto/CenterPoint/data/rtg/raw/lidar_and_camera_2026-05-19-15-44-34.bag'
RVIZ_CFG='/media/shen/data/zk/Auto/config/rtg_all_sensors.rviz'

case "${1:-}" in
    --loop)  PLAY_FLAGS="--clock --loop" ;;
    --pause) PLAY_FLAGS="--clock --pause" ;;
    *)       PLAY_FLAGS="--clock" ;;
esac

echo "============================================"
echo "  RTG 全传感器 rosbag 回放"
echo "============================================"
echo "  Bag:  $(basename ${BAG})"
echo "  RViz config: ${RVIZ_CFG}"
echo "  Topics:"
echo "    Camera (raw): /Camera_Raw_Img_01 .. 04"
echo "    LiDAR (raw):  /ouster1/points, /ouster2/points"
echo "                  /lidar3/rslidar3_points, /lidar4/rslidar4_points"
echo "    LiDAR (rviz): /rtg/ouster1/points .. /rtg/lidar4/points"
echo "    TF tree: rtg_bev_origin → lidar_0{1,2,3,4}_{front,rear}"
echo "============================================"

# Kill background processes on exit
cleanup() {
    echo "Cleaning up..."
    kill %1 %2 %3 2>/dev/null || true
    exit 0
}
trap cleanup INT TERM

# 1. Start roscore
echo "[1/3] Starting roscore..."
roscore &
sleep 2

# Wait for roscore to be ready
until rostopic list > /dev/null 2>&1; do
    sleep 0.5
done
echo "  roscore ready."

# 2. Start TF republisher (fixes frame_ids for RViz)
echo "[2/4] Starting TF republisher..."
python3 /media/shen/data/zk/Auto/tools/_tf_republisher.py &
sleep 2

# 3. Start rosbag play (with --clock so RViz uses bag time)
echo "[3/4] Starting rosbag play..."
rosbag play ${PLAY_FLAGS} "${BAG}" &
sleep 1

# 4. Start RViz
echo "[4/4] Starting RViz..."
rviz -d "${RVIZ_CFG}" &
sleep 3

echo ""
echo "============================================"
echo "  RViz now playing bag data."
echo "  Press Ctrl+C to stop."
echo ""
echo "  RViz layout:"
echo "    3D View:   4 LiDAR point clouds"
echo "    Side panels: 4 Camera images"
echo "  Colors: Red=L1, Green=L2, Blue=L3, Orange=L4"
echo "============================================"

# Wait for user interrupt
wait
