# WarningArray.msg — 预警结果数组

Header header                     # ROS1 标准头
Warning[] warnings                # 预警列表
uint8 ego_motion_state            # 0=静止, 1=运动+x, 2=运动-x, 3=未知
rtg_bev_msgs/WarningZone[] active_zones  # [deprecated] always empty
