"""Bring up the C++ tracker node plus the synthetic detection replay and its parity gate.

Prerequisite — generate the reference on the HOST (it needs the venv's numpy/scipy), repo root:
    python3 scripts/write_py_tracker_refs.py
Then, in the container:
    ros2 launch kf_bringup tracker_synthetic.launch.py
Optional override:
    reference_path:=/workspace/data/cache/tracker_py_ref.npz

The verdict is the tracking_replay process exit code (0 = PARITY PASS, non-zero = FAIL); the
OnProcessExit handler below logs it verbatim and then tears the whole launch down.

Both this file and config/tracker.yaml reach share/ only via setup.py, which lists data files by
explicit filename — if get_package_share_directory cannot find config/tracker.yaml at launch time,
that edit was missed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, LogInfo, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config = os.path.join(
        get_package_share_directory("kf_bringup"),
        "config",
        "tracker.yaml",
    )

    ref_arg = DeclareLaunchArgument(
        "reference_path",
        default_value="/workspace/data/cache/tracker_py_ref.npz",
        description="Python reference .npz written by scripts/write_py_tracker_refs.py.",
    )

    tracker_node = Node(
        package="kf_tracker",
        executable="tracker_node",
        name="tracker_node",
        parameters=[config],
        output="screen",
    )

    tracking_replay_node = Node(
        package="kf_bringup",
        executable="tracking_replay",
        name="tracking_replay",
        parameters=[
            config,
            {"reference_path": LaunchConfiguration("reference_path")},
        ],
        output="screen",
    )

    # The replay owns the verdict. Its exit ends the launch; the tracker outliving it is normal,
    # so only flag a tracker exit that happens FIRST (that one is a crash, not a teardown).
    state = {"replay_done": False}

    def on_replay_exit(event, context):
        state["replay_done"] = True
        code = event.returncode
        verdict = "PARITY PASS" if code == 0 else "PARITY FAIL"
        return [
            LogInfo(msg=f"tracking_replay exited with code {code} -> {verdict}"),
            EmitEvent(event=Shutdown(reason=f"tracking_replay exited with code {code}")),
        ]

    def on_tracker_exit(event, context):
        if state["replay_done"]:
            return None     # ordinary teardown after the gate reported
        return [LogInfo(
            msg=f"tracker_node exited with code {event.returncode} BEFORE the replay finished — "
                f"the replay will time out; the real failure is in the tracker log above."
        )]

    return LaunchDescription([
        ref_arg,
        tracker_node,
        tracking_replay_node,
        RegisterEventHandler(
            OnProcessExit(target_action=tracking_replay_node, on_exit=on_replay_exit)),
        RegisterEventHandler(
            OnProcessExit(target_action=tracker_node, on_exit=on_tracker_exit)),
    ])
