# Warning.msg — 单条预警信息

int32 track_id                    # 关联的跟踪 ID
int8 warning_level                # 0=无风险, 1=提示(info), 2=警告(warning), 3=危险(danger)
int8 target_class                 # 0=person, 1=truck, 2=car, 3=other_obstacle
float32 distance                  # 目标到最近 RTG 支腿的距离 (米)
string zone                       # [deprecated] always empty — 预警改用距离+运动方向驱动
string trigger_reason             # 触发原因: "distance" / "release_hold"
time trigger_time                 # 触发时间戳
