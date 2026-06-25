cmake_minimum_required(VERSION 3.20)
project(kf_eskf)

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
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(kf_msgs REQUIRED)

# Header-only library: an INTERFACE target carrying the include dir + kf_common (which
# transitively brings Eigen).
add_library(kf_eskf INTERFACE)
target_include_directories(kf_eskf INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(kf_eskf INTERFACE kf_common::kf_common Eigen3::Eigen)

# Install the headers and export the target for downstream packages.
install(DIRECTORY include/ DESTINATION include)
install(TARGETS kf_eskf EXPORT export_kf_eskf)
ament_export_targets(export_kf_eskf HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3 kf_common)

# EskfNode: the ROS 2 wrapper around the header-only ESKF kernel.
add_executable(eskf_node src/eskf_node.cpp)
target_include_directories(eskf_node PRIVATE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>")
target_link_libraries(eskf_node kf_eskf kf_common::kf_common Eigen3::Eigen)
ament_target_dependencies(eskf_node
  rclcpp sensor_msgs geometry_msgs nav_msgs
  tf2 tf2_ros tf2_geometry_msgs kf_msgs)
install(TARGETS eskf_node DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_eskf test/test_eskf.cpp)     # predict-Jacobian, PSD, GPS, convergence
  target_link_libraries(test_eskf kf_eskf kf_common::kf_common Eigen3::Eigen)
endif()

ament_package()
