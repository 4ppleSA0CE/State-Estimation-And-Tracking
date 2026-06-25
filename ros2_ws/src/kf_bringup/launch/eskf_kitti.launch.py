"""Launch the ESKF node and KITTI replay for the Stage 1.4 parity check.

Usage (in container):
    ros2 launch kf_bringup eskf_kitti.launch.py
Optional overrides:
    cache_path:=/path/to/oxts.npz reference_path:=/path/to/py_ref.npz
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(
        get_package_share_directory("kf_bringup"),
        "config",
        "eskf_kitti.yaml",
    )

    cache_arg = DeclareLaunchArgument(
        "cache_path",
        default_value=(
            "/workspace/data/cache/"
            "kitti_raw_2011_09_26_drive_0001_extract_oxts_v1.npz"
        ),
        description="Path to the OXTS cache .npz (mounted in container).",
    )
    ref_arg = DeclareLaunchArgument(
        "reference_path",
        default_value="/workspace/data/cache/eskf_py_ref.npz",
        description="Path to the Python ESKF reference trajectory .npz.",
    )

    eskf_node = Node(
        package="kf_eskf",
        executable="eskf_node",
        name="eskf_node",
        parameters=[config],
        output="screen",
    )

    kitti_replay_node = Node(
        package="kf_bringup",
        executable="kitti_replay",
        name="kitti_replay",
        parameters=[
            config,
            {
                "cache_path":      LaunchConfiguration("cache_path"),
                "reference_path":  LaunchConfiguration("reference_path"),
            },
        ],
        output="screen",
    )

    return LaunchDescription([
        cache_arg,
        ref_arg,
        eskf_node,
        kitti_replay_node,
    ])
