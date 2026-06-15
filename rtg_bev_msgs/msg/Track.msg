# Track.msg — 跟踪目标

int32 track_id            # 稳定跟踪 ID (跨帧一致)
int32 class_id            # 类别: 0=person, 1=truck, 2=car, 3=other_obstacle
int32 age                 # 跟踪持续帧数
float32 x                 # 当前估计位置 x (BEV 坐标系，米)
float32 y                 # 当前估计位置 y (BEV 坐标系，米)
float32 z                 # 当前估计位置 z (BEV 坐标系，米)
float32 vx                # 估计速度 x (m/s)
float32 vy                # 估计速度 y (m/s)
float32 w                 # 3D 框宽度 (米)
float32 l                 # 3D 框长度 (米)
float32 h                 # 3D 框高度 (米)
float32 yaw               # 朝向角 (弧度)
int32 state               # 跟踪状态: 0=候选(candidate), 1=确认(confirmed), 2=丢失(lost)
