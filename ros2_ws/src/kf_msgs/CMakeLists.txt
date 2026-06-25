cmake_minimum_required(VERSION 3.20)
project(kf_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

# Generate the C++/Python bindings for the section 5.4 interfaces.
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/EgoState.msg"
  "msg/Detection.msg"
  "msg/DetectionArray.msg"
  "msg/Track.msg"
  "msg/TrackArray.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
