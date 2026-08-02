cmake_minimum_required(VERSION 3.20)
project(kf_tracker)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)        # C++17 per the tech stack
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
endif()
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(eigen3_cmake_module REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(kf_common REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(kf_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)

# Header-only tracking kernel: an INTERFACE target carrying the include dir + kf_common
# (which transitively brings Eigen). Host-compiles without ROS.
add_library(kf_tracker INTERFACE)
target_include_directories(kf_tracker INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(kf_tracker INTERFACE kf_common::kf_common Eigen3::Eigen)

install(DIRECTORY include/ DESTINATION include)
install(TARGETS kf_tracker EXPORT export_kf_tracker)
ament_export_targets(export_kf_tracker HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3 kf_common)

# TrackerNode: the ROS 2 wrapper around the header-only tracking kernel.
add_executable(tracker_node src/tracker_node.cpp)
target_include_directories(tracker_node PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(tracker_node kf_tracker kf_common::kf_common Eigen3::Eigen)
ament_target_dependencies(tracker_node rclcpp geometry_msgs std_msgs kf_msgs)
install(TARGETS tracker_node DESTINATION lib/${PROJECT_NAME})

# DetectionTransformNode: the ego/tracker coupling -- base_link detections + /ego/state -> map_bev.
add_executable(detection_transform_node src/detection_transform_node.cpp)
target_include_directories(detection_transform_node PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(detection_transform_node kf_tracker kf_common::kf_common Eigen3::Eigen)
# builtin_interfaces is listed explicitly: <builtin_interfaces/msg/time.hpp> is included
# directly by this node and otherwise resolves only transitively through kf_msgs.
ament_target_dependencies(detection_transform_node
  rclcpp geometry_msgs std_msgs kf_msgs builtin_interfaces)
install(TARGETS detection_transform_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_box3d test/test_box3d.cpp)      # oriented 3D-IoU
  target_link_libraries(test_box3d kf_tracker kf_common::kf_common Eigen3::Eigen)
  ament_add_gtest(test_models test/test_models.cpp)    # CV/CA/CT per-mode filters
  target_link_libraries(test_models kf_tracker kf_common::kf_common Eigen3::Eigen)
  ament_add_gtest(test_imm test/test_imm.cpp)          # IMM mixing / mode probabilities
  target_link_libraries(test_imm kf_tracker kf_common::kf_common Eigen3::Eigen)
  ament_add_gtest(test_tracker test/test_tracker.cpp)  # association + M-of-N lifecycle
  target_link_libraries(test_tracker kf_tracker kf_common::kf_common Eigen3::Eigen)
  ament_add_gtest(test_ego_transform test/test_ego_transform.cpp)  # base_link -> map_bev coupling
  target_link_libraries(test_ego_transform kf_tracker kf_common::kf_common Eigen3::Eigen)
  ament_add_gtest(test_pending_frames test/test_pending_frames.cpp)  # detection/ego arrival orders
  target_link_libraries(test_pending_frames kf_tracker kf_common::kf_common Eigen3::Eigen)
endif()

ament_package()
