#!/bin/bash
# Quick start: roscore + TF + rosbag + RViz
kill -9 $(pgrep -f "ros|rviz|_tf" 2>/dev/null) 2>/dev/null
sleep 4

source /opt/ros/noetic/setup.bash

nohup roscore > /tmp/roscore.log 2>&1 &
sleep 5

nohup python3 /media/shen/data/zk/Auto/tools/_tf_republisher.py > /tmp/tf_repub.log 2>&1 &
sleep 3

nohup rosbag play --clock --loop /media/shen/data/zk/Auto/CenterPoint/data/rtg/raw/lidar_and_camera_2026-05-19-15-44-34.bag > /tmp/rosbag.log 2>&1 &
sleep 2

nohup rviz -d /media/shen/data/zk/Auto/config/rtg_all_sensors.rviz > /tmp/rviz.log 2>&1 &

sleep 3
echo "All started. RViz on DISPLAY=:1"
