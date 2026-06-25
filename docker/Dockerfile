# ROS2 Humble dev image for the C++ state-estimation nodes.
# ros-base (no GUI) is enough — Foxglove on the Mac host is the visualizer.
FROM ros:humble-ros-base

# Build toolchain + Eigen (header-only linear algebra, the robotics standard).
# gtest ships with ament_cmake_gtest, so no separate install is needed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libeigen3-dev \
    python3-colcon-common-extensions \
    python3-numpy \
    ros-humble-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# Workspace lives at /workspace/ros2_ws (mounted from the host repo by compose).
WORKDIR /workspace/ros2_ws

# Source ROS2 in every interactive shell so `ros2` / `colcon` are on PATH.
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc \
    && echo "test -f /workspace/ros2_ws/install/setup.bash && source /workspace/ros2_ws/install/setup.bash" >> /root/.bashrc

CMD ["bash"]
