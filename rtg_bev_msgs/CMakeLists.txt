cmake_minimum_required(VERSION 3.0.2)
project(rtg_bev_msgs)

find_package(catkin REQUIRED COMPONENTS
  message_generation
  std_msgs
)

# Generate messages
add_message_files(
  FILES
  Detection.msg
  DetectionArray.msg
  Track.msg
  TrackArray.msg
  Warning.msg
  WarningArray.msg
  WarningZone.msg
  EgoMotionState.msg
)

generate_messages(
  DEPENDENCIES
  std_msgs
)

catkin_package(
  CATKIN_DEPENDS
  message_runtime
  std_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)
