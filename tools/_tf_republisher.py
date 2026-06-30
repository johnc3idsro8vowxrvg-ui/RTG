"""
TF tree + point cloud republisher + CameraInfo for RTG bag RViz visualization.
2026-06-15: BEV +y flipped (禁行侧→集卡侧), aligned with ROS convention.
  BEV y axis now matches ROS +y (left). RViz top view: R_z(-90°) applied at
  rtg_bev_origin → rtg_rviz frame so the truck lane extends along screen.
2026-06-04: Container-edge analysis → L1/L2 yaw (+4.62°, -0.82°).
2026-06-11: RViz visual alignment → L3/L4 full RPY estimated from bag.
  L3: RPY = (+90°, +6°, +90°)      (RoboSense 32 旋转90°安装)
  L4: RPY = (+96°, +2°, -90°)      (RoboSense 32 旋转90°安装)
  Note: RoboSense 32线旋转90°安装 → 垂直360°/水平~70° 沿大车道扫描
  BEV 坐标直接用于 TF (与 ROS 一致): +y = 禁行侧→集卡侧 = ROS left
  L2→L1 merge: pure translation (-12.0, 0, 0) in BEV/TF frame (LIDAR_RE_to_LIDAR_FR)
  Camera orientations — R_y(±90°), HFOV 87.6°, VFOV 46°
"""
import sys, math
sys.path.insert(0, '/opt/ros/noetic/lib/python3/dist-packages')
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Quaternion
from sensor_msgs.msg import PointCloud2, CameraInfo

# Sensor positions in BEV (now aligned with ROS TF):
#   BEV +x = TF +x (forward, 大车道方向)
#   BEV +y = TF +y (left, 禁行侧→集卡侧)
#   BEV +z = TF +z (up)
#
# RViz: parent frame rtg_rviz has R_z(-90°) relative to rtg_bev_origin,
#       so the truck lane (x-axis) extends along the screen in top view.
#
# Camera orientations (R_y(±90°) rotation):
#   C1/C3 朝 +x: cam +z → world +x,  cam +y → world +y, cam +x → world -z
#   C2/C4 朝 -x: cam +z → world -x,  cam +y → world +y, cam +x → world +z

def q_from_rpy(roll, pitch, yaw):
    """quaternion from RPY (radians, XYZ convention)"""
    cy, sy = math.cos(yaw*0.5), math.sin(yaw*0.5)
    cp, sp = math.cos(pitch*0.5), math.sin(pitch*0.5)
    cr, sr = math.cos(roll*0.5), math.sin(roll*0.5)
    return Quaternion(
        w=cr*cp*cy + sr*sp*sy,
        x=sr*cp*cy - cr*sp*sy,
        y=cr*sp*cy + sr*cp*sy,
        z=cr*cp*sy - sr*sp*cy)

# BEV coordinate quaternions (aligned with ROS TF):
#   L1: yaw +4.62° (container-edge analysis)
#   L2: yaw -0.82° (container-edge analysis)
yaw1 = math.radians(4.62);  Q_L1 = Quaternion(w=math.cos(yaw1/2), x=0.0, y=0.0, z=math.sin(yaw1/2))
yaw2 = math.radians(-0.82); Q_L2 = Quaternion(w=math.cos(yaw2/2), x=0.0, y=0.0, z=math.sin(yaw2/2))
# L3: RPY(+90°,+6°,+90°) RoboSense 32, rotated 90° mount
Q_L3 = q_from_rpy(roll=-math.pi/2, pitch=-(math.pi + math.radians(6)), yaw=-math.pi/2)
# L4: RPY(+96°,+2°,-90°) RoboSense 32, rotated 90° mount
Q_L4 = q_from_rpy(roll=+math.pi/2 + math.radians(6), pitch=+math.radians(2), yaw=-math.pi/2)

# Camera R_y(±90°) quaternions
Q_CAM_plusX = Quaternion(w=0.7071068, x=0.0, y=0.7071068, z=0.0)
Q_CAM_minusX = Quaternion(w=0.7071068, x=0.0, y=-0.7071068, z=0.0)

# RViz view rotation: R_z(-90°) so truck lane extends along screen in top view
Q_RVIZ = Quaternion(w=0.7071068, x=0.0, y=0.0, z=-0.7071068)

SENSOR_TF = [
    # (child_frame, parent, x, y, z, quaternion) — BEV coords used directly as TF coords
    ('lidar_01_front', 'rtg_bev_origin', 0.0, 0.0, 1.5, Q_L1),              # L1 OS1 (origin)
    ('lidar_02_rear',  'rtg_bev_origin', -12.0, 0.0, 1.5, Q_L2),             # L2 OS1
    ('lidar_03_front', 'rtg_bev_origin', 0.0, -23.5, 1.5, Q_L3),             # L3 H32 (禁行侧, y=-23.5)
    ('lidar_04_rear',  'rtg_bev_origin', -12.0, -23.5, 1.5, Q_L4),           # L4 H32 (禁行侧, y=-23.5)
    # Cameras, z=4m
    ('camera_01', 'rtg_bev_origin', 0.0, 0.0, 4.0, Q_CAM_plusX),             # C1
    ('camera_02', 'rtg_bev_origin', -12.0, -0.1, 4.0, Q_CAM_minusX),         # C2
    ('camera_03', 'rtg_bev_origin', 0.0, -23.5, 4.0, Q_CAM_plusX),           # C3
    ('camera_04', 'rtg_bev_origin', -12.0, -23.4, 4.0, Q_CAM_minusX),        # C4
]

TOPIC_REMAP = {
    '/ouster1/points':            ('/rtg/lidar_01/points', 'lidar_01_front'),
    '/ouster2/points':            ('/rtg/lidar_02/points', 'lidar_02_rear'),
    '/lidar3/rslidar3_points':   ('/rtg/lidar3/points',   'lidar_03_front'),
    '/lidar4/rslidar4_points':   ('/rtg/lidar4/points',   'lidar_04_rear'),
}

# Synthetic CameraInfo for RViz FOV visualization
# 1080×1920竖屏, fx=1272.2 (46°VFOV along width), fy=1001.1 (87.6°HFOV along height)
CAM_INFO = {
    '/Camera_Raw_Img_01': 'camera_01',
    '/Camera_Raw_Img_02': 'camera_02',
    '/Camera_Raw_Img_03': 'camera_03',
    '/Camera_Raw_Img_04': 'camera_04',
}

def make_camera_info_msg(frame_id):
    msg = CameraInfo()
    msg.header.frame_id = frame_id
    msg.height = 1920
    msg.width = 1080
    msg.distortion_model = 'plumb_bob'
    msg.D = [0.0, 0.0, 0.0, 0.0, 0.0]
    msg.K = [1272.2, 0.0, 540.0,
             0.0, 1001.1, 960.0,
             0.0, 0.0, 1.0]
    msg.R = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    msg.P = [1272.2, 0.0, 540.0, 0.0,
             0.0, 1001.1, 960.0, 0.0,
             0.0, 0.0, 1.0, 0.0]
    return msg

class RTGTFRepublisher:
    def __init__(self):
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster()
        self._publish_tfs(rospy.Time.now())

        # RViz view transform: rtg_bev_origin → rtg_rviz (R_z(-90°))
        self._publish_rviz_tf(rospy.Time.now())

        # LiDAR point cloud remapping (immediate republish)
        for orig, (repub, frame) in TOPIC_REMAP.items():
            pub = rospy.Publisher(repub, PointCloud2, queue_size=10)
            rospy.Subscriber(orig, PointCloud2,
                             lambda m, f=frame, p=pub: self._repub(m, f, p),
                             queue_size=10)
            rospy.loginfo(f"  {orig} -> {repub} (frame: {frame})")

        # Synthetic CameraInfo publishers (for RViz camera FOV visualization)
        self.caminfo_pubs = {}
        for img_topic, frame_id in CAM_INFO.items():
            ci_topic = img_topic + '/camera_info'
            pub = rospy.Publisher(ci_topic, CameraInfo, queue_size=10, latch=True)
            msg = make_camera_info_msg(frame_id)
            pub.publish(msg)
            self.caminfo_pubs[img_topic] = pub
            rospy.loginfo(f"  CameraInfo: {ci_topic} (frame: {frame_id})")

        # Keep TF alive
        rospy.Timer(rospy.Duration(1.0), lambda e: self._publish_caminfo())
        rospy.Timer(rospy.Duration(1.0), lambda e: self._publish_all_tfs(rospy.Time.now()))
        rospy.loginfo(f"TF republisher ready ({len(SENSOR_TF)} frames, {len(CAM_INFO)} camera infos)")

    def _publish_caminfo(self):
        for img_topic in CAM_INFO:
            pub = self.caminfo_pubs.get(img_topic)
            if pub:
                msg = make_camera_info_msg(CAM_INFO[img_topic])
                msg.header.stamp = rospy.Time.now()
                pub.publish(msg)

    def _publish_rviz_tf(self, stamp):
        """R_z(-90°) parent rotation for RViz top view: rtg_bev_origin → rtg_rviz."""
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = 'rtg_bev_origin'
        t.child_frame_id = 'rtg_rviz'
        t.transform.translation.x = 0.0
        t.transform.translation.y = 0.0
        t.transform.translation.z = 0.0
        t.transform.rotation = Q_RVIZ
        self.tf_broadcaster.sendTransform(t)

    def _publish_all_tfs(self, stamp):
        self._publish_rviz_tf(stamp)
        tfs = []
        for child, parent, x, y, z, q in SENSOR_TF:
            t = TransformStamped()
            t.header.stamp = stamp
            t.header.frame_id = parent
            t.child_frame_id = child
            t.transform.translation.x = x
            t.transform.translation.y = y
            t.transform.translation.z = z
            t.transform.rotation = q
            tfs.append(t)
        self.tf_broadcaster.sendTransform(tfs)

    @staticmethod
    def _repub(msg, frame, pub):
        msg.header.frame_id = frame
        pub.publish(msg)

def main():
    rospy.init_node('rtg_tf_republisher')
    RTGTFRepublisher()
    rospy.spin()

if __name__ == '__main__':
    main()
