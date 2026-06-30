# Detection.msg — 单目标检测结果
# 坐标系: BEV 坐标系 (rtg_bev_origin)

int32 id                  # 检测目标序号 (逐帧递增，非跟踪 ID)
int32 class_id            # 类别: 0=person, 1=truck, 2=car, 3=other_obstacle
string class_name         # 类别名称
float32 confidence        # 置信度 [0, 1]
float32 x                 # 3D 框中心 x (BEV 坐标系，米)
float32 y                 # 3D 框中心 y (BEV 坐标系，米)
float32 z                 # 3D 框中心 z (BEV 坐标系，米)
float32 w                 # 3D 框宽度 (米)
float32 l                 # 3D 框长度 (米)
float32 h                 # 3D 框高度 (米)
float32 yaw               # 朝向角 (弧度)
float32 vx                # 速度估计 x (可选，由跟踪器填充，m/s)
float32 vy                # 速度估计 y (可选，由跟踪器填充，m/s)
float32 distance          # 目标到 RTG 最近支腿的距离 (BEV 平面投影，米)
