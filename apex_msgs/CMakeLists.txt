cmake_minimum_required(VERSION 3.8)
project(apex_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

# generates C++ and Python code for the messages, for every node that uses them
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/StateEstimate.msg"
  "msg/Obstacle.msg"
  "msg/ObstacleArray.msg"
  DEPENDENCIES std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
