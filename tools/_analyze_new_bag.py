"""Quick analysis of new RTG bag"""
import sys; sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import rosbag
from sensor_msgs import point_cloud2

bag_path = '/tmp/rtg_new/lidar_and_camera_2026-05-19-15-44-34.bag'
bag = rosbag.Bag(bag_path)

print("=== Camera format ===")
for topic, msg, t in bag.read_messages(topics=['/Camera_Raw_Img_01']):
    print(f"{topic}: {msg.width}x{msg.height}, encoding={msg.encoding}")
    break

print("\n=== LiDAR point cloud formats ===")
for topic in ['/ouster1/points', '/ouster2/points',
              '/lidar3/rslidar3_points', '/lidar4/rslidar4_points']:
    for _, msg, _ in bag.read_messages(topics=[topic]):
        fields = [(f.name, f.datatype) for f in msg.fields]
        n_pts = msg.width * msg.height
        print(f"{topic}: fields={fields}, height={msg.height}, "
              f"width={msg.width}, pts={n_pts:,.0f}")
        break

print("\n=== Topic statistics ===")
info = bag.get_type_and_topic_info()
for topic, meta in info.topics.items():
    print(f"{topic}: {meta.message_count:4d} msgs, "
          f"{meta.frequency:5.1f} Hz ({meta.msg_type})")

print(f"\nDuration: {bag.get_end_time() - bag.get_start_time():.1f}s")
print(f"Size: {bag.size / 1e9:.1f} GB")
bag.close()
