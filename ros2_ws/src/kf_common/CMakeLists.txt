cmake_minimum_required(VERSION 3.20)
project(kf_common)

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

# Header-only library: an INTERFACE target carrying the include dir + Eigen.
add_library(kf_common INTERFACE)
target_include_directories(kf_common INTERFACE
  "$<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>"
  "$<INSTALL_INTERFACE:include>")
target_link_libraries(kf_common INTERFACE Eigen3::Eigen)

# Install the headers and export the target so kf_eskf can depend on it.
install(DIRECTORY include/ DESTINATION include)
install(TARGETS kf_common EXPORT export_kf_common)
ament_export_targets(export_kf_common HAS_LIBRARY_TARGET)
ament_export_dependencies(eigen3_cmake_module Eigen3)

if(BUILD_TESTING)
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_so3 test/test_so3.cpp)        # DOD 2.1: quaternion-op tests
  target_link_libraries(test_so3 kf_common Eigen3::Eigen)
endif()

ament_package()
