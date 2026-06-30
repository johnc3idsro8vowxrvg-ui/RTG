# EgoMotionState.msg — RTG 自身运动状态

uint8 state                       # 0=静止, 1=运动+x, 2=运动-x, 3=未知
float32 confidence                # 状态置信度 [0, 1]
float32 displacement              # 最近1秒估计位移 (米)
float32 velocity_estimate         # 估计速度 (m/s, 可选)
