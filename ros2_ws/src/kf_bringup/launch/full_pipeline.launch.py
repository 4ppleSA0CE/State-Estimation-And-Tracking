"""Stage 6 unified pipeline: KITTI OXTS replay -> ESKF -> detection transform -> IMM tracker.

The whole point of the stage is the middle arrow: detection_transform_node converts base_link
detections into map_bev with the ESKF's ESTIMATED ego pose, so localization error lands in the
track positions. tracker_node reaches that stream by the launch-level remap below and is
otherwise the same binary Stage 5B's parity gate validated.

Usage (in container):
    ros2 launch kf_bringup full_pipeline.launch.py                     # mode:=baseline
    ros2 launch kf_bringup full_pipeline.launch.py mode:=gps_dropout \\
        baseline_npz:=/workspace/data/cache/stage6_baseline.npz
    ros2 launch kf_bringup full_pipeline.launch.py mode:=baseline foxglove:=true

Run the whole suite with scripts/run_stage6.py rather than by hand — six of the seven gates
need the baseline npz on disk first.

The verdict is pipeline_replay's exit code (0 = the mode's gate passed). OnProcessExit logs it
and tears the launch down; the gate's own `STAGE6 <mode>: PASS|FAIL` line comes from the node,
never from this file, so a crash that prints no verdict is reported as MISSING by run_stage6.py
instead of being dressed up as a FAIL here.

Both this file and config/full_pipeline.yaml reach share/ only via setup.py, which lists data
files by explicit filename — if get_package_share_directory cannot find config/full_pipeline.yaml
at launch time, that edit was missed.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    EmitEvent,
    LogInfo,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

CONFIG_FILE = "full_pipeline.yaml"
DEFAULT_CACHE = ("/workspace/data/cache/"
                 "kitti_raw_2011_09_26_drive_0001_extract_oxts_v1.npz")
# Real Velodyne scans for the live view only (foxglove:=true). The `_sync` tree is the
# motion-compensated float32 .bin product; lidar_replay's docstring explains the choice and the
# fact that it is a DIFFERENT tree from the `_extract` OXTS the cache above was built from --
# same drive, same clock, and the node differences the two timestamp files to prove it.
DEFAULT_VELODYNE = ("/workspace/data/kitti_raw/2011_09_26/"
                    "2011_09_26_drive_0001_sync/velodyne_points")
# The real road video for the live view only (foxglove:=true). image_02 is the left colour
# camera; the `_extract` tree is the one that ships it for this drive. Same clock story as the
# Velodyne above -- image_replay differences this tree's timestamps.txt against oxts_timestamps.
DEFAULT_IMAGE = ("/workspace/data/kitti_raw/2011_09_26/"
                 "2011_09_26_drive_0001_extract/image_02")
# Anchors the Velodyne AND camera clocks to the replay's t=0, so it must track cache_path, not
# velodyne_dir or image_dir.
DEFAULT_OXTS_TIMESTAMPS = ("/workspace/data/kitti_raw/2011_09_26/"
                           "2011_09_26_drive_0001_extract/oxts/timestamps.txt")

# The seven failure modes, in one dict so a reader can see the whole suite at once. Each entry
# is a pipeline_replay parameter override layered on top of config/full_pipeline.yaml, which
# holds every baseline value. Design doc section 6; the mode names are stage6_gates.MODES.
PRESETS = {
    "baseline":          {},
    "gps_dropout":       {"gps_dropout_s": [4.0, 8.0]},
    "imu_bias":          {"imu_accel_bias_xyz": [0.1, 0.0, 0.0]},
    # Target 0 (the LEADING vehicle), not the crosser: measured 2026-07-28, targets 1/2/3 all
    # leave the sensor FOV within ~3 s on drive_0001, so a 5 s onset has no visible subject and
    # the gate correctly reports "never matched". Target 0 gives 50 pre-onset and 20 post-onset
    # visible frames -- the full MANEUVER_MAX_FRAMES window. See spec section 5.1.
    "maneuver":          {"maneuver_target": 0, "maneuver_start_s": 5.0,
                          "maneuver_omega": 0.4},
    "det_dropout_short": {"det_dropout_s": [6.0, 7.0]},          # tracker max_age stays 2
    "det_dropout_coast": {"det_dropout_s": [6.0, 7.0]},          # tracker max_age -> 15
    "clutter":           {"clutter_lambda": 2.0},
}

# The ONE tracker parameter a preset touches. max_age = 2 at 10 Hz is a 0.2 s coast budget, so
# the same 1 s detection gap kills the track under `short` and is survived under `coast` --
# that pair, not a single run, is what PRD 6.2.4 actually demonstrates (design doc section 6).
TRACKER_OVERRIDES = {"det_dropout_coast": {"max_age": 15}}

FOXGLOVE_PORT = 8765
FOXGLOVE_RATE_SCALE = 1.0     # a live view has to run at wall clock to be watchable


def _as_bool(value: str, name: str) -> bool:
    text = str(value).strip().lower()
    if text in ("true", "1", "yes"):
        return True
    if text in ("false", "0", "no", ""):
        return False
    raise RuntimeError(f"launch argument `{name}` must be true or false, got {value!r}")


def _pipeline(context, *_args, **_kwargs):
    """Resolve the launch arguments to real Python values, then build the four nodes.

    An OpaqueFunction rather than plain substitutions because the preset lookup needs the
    resolved `mode` string, and because every pipeline_replay parameter has a declared type:
    a LaunchConfiguration reaches launch_ros as a string, and rclpy rejects a string override on
    a double parameter. Resolving here keeps `rate_scale` a float and `maneuver_target` an int.
    """
    mode = LaunchConfiguration("mode").perform(context)
    if mode not in PRESETS:
        # FATAL at startup, never a silently-substituted preset -- the same discipline
        # pipeline_replay applies to `mode` and tracker_node applies to `cost`.
        raise RuntimeError(f"launch argument `mode` must be one of {sorted(PRESETS)}, "
                           f"got {mode!r}")

    foxglove = _as_bool(LaunchConfiguration("foxglove").perform(context), "foxglove")
    rate_scale = float(LaunchConfiguration("rate_scale").perform(context))
    # foxglove:=true supplies wall-clock pacing so the view is watchable, but only as a DEFAULT:
    # an explicit rate_scale must win, or there is no way to slow the ~12 s replay down enough to
    # connect a viewer to it. 0.0 is the "as fast as possible" default, i.e. "user said nothing".
    if foxglove and rate_scale == 0.0:
        rate_scale = FOXGLOVE_RATE_SCALE

    config = os.path.join(get_package_share_directory("kf_bringup"), "config", CONFIG_FILE)

    eskf_node = Node(
        package="kf_eskf",
        executable="eskf_node",
        name="eskf_node",
        parameters=[config],
        output="screen",
    )

    transform_node = Node(
        package="kf_tracker",
        executable="detection_transform_node",
        name="detection_transform_node",
        parameters=[config],
        output="screen",
    )

    # The remap IS the coupling: tracker_node knows nothing about /ego/state and is not edited,
    # so tracker_synthetic.launch.py and its 5B parity gate stay valid unchanged (spec D4).
    tracker_params = [config]
    if TRACKER_OVERRIDES.get(mode):
        tracker_params.append(dict(TRACKER_OVERRIDES[mode]))
    tracker_node = Node(
        package="kf_tracker",
        executable="tracker_node",
        name="tracker_node",
        parameters=tracker_params,
        remappings=[("/detections", "/detections_map")],
        output="screen",
    )

    replay_node = Node(
        package="kf_bringup",
        executable="pipeline_replay",
        name="pipeline_replay",
        parameters=[
            config,
            {
                "mode":         mode,
                "cache_path":   LaunchConfiguration("cache_path").perform(context),
                "output_npz":   LaunchConfiguration("output_npz").perform(context),
                # "" means "no baseline". The baseline run must OMIT the token entirely rather
                # than pass baseline_npz:="" -- ros2launch.api.parse_launch_arguments rejects
                # any token ending in ':=' with "malformed launch argument", which kills the
                # launch before a node starts. Hence default_value="" below, and hence
                # run_stage6.py appending the token only when it has a value.
                "baseline_npz": LaunchConfiguration("baseline_npz").perform(context),
                "rate_scale":   rate_scale,
                **PRESETS[mode],
            },
        ],
        output="screen",
    )

    nodes = [eskf_node, transform_node, tracker_node, replay_node]
    # (node, label) so the crash guards below can name them without touching Node.node_name,
    # which raises until the action has actually been executed. Neither node feeds a gate, so a
    # death here cannot change the verdict -- but an unguarded death is SILENT, and "the live
    # view was blank" then has no entry in the log to explain it. Diagnostics, not verdict.
    viz_nodes: list[tuple[Node, str]] = []

    if foxglove:
        # Task 7 ships viz_markers.py and adds ros-humble-foxglove-bridge to the image; this
        # branch is inert at the default foxglove:=false.
        viz_nodes.append((Node(
            package="foxglove_bridge",
            executable="foxglove_bridge",
            name="foxglove_bridge",
            parameters=[{"port": FOXGLOVE_PORT}],
            output="screen",
        ), "foxglove_bridge"))
        viz_nodes.append((Node(
            package="kf_bringup",
            executable="viz_markers",
            name="viz_markers",
            # The SAME cache pipeline_replay is replaying, so the amber ground-truth ego trail is
            # keyed to the very stamps the cyan estimate trail is drawn from. Pointing the two at
            # different drives would render a large, steady, entirely fictitious ESKF error.
            parameters=[{"cache_path": LaunchConfiguration("cache_path").perform(context)}],
            output="screen",
        ), "viz_markers"))
        viz_nodes.append((Node(
            package="kf_bringup",
            executable="lidar_replay",
            name="lidar_replay",
            parameters=[{
                "velodyne_dir": LaunchConfiguration("velodyne_dir").perform(context),
                "stride": int(LaunchConfiguration("lidar_stride").perform(context)),
                # Must name the OXTS timestamps of the drive `cache_path` was built from, NOT
                # the velodyne's own drive: the node needs the offset from the Velodyne clock to
                # the REPLAY's t=0, which is that file's first line. Left unset it defaulted to
                # drive_0001 -- replaying any other drive then aligned the cloud against a
                # foreign clock (drive_0001 vs drive_0009 are ~6 minutes apart, so every scan
                # fell outside the replay window and the cloud froze on scan 0).
                "oxts_timestamps": LaunchConfiguration("oxts_timestamps").perform(context),
            }],
            output="screen",
        ), "lidar_replay"))
        viz_nodes.append((Node(
            package="kf_bringup",
            executable="image_replay",
            name="image_replay",
            parameters=[{
                "image_dir": LaunchConfiguration("image_dir").perform(context),
                # The SAME anchor lidar_replay uses, and for the same reason: it names the OXTS
                # timestamps of the drive `cache_path` was built from, not the camera's own
                # drive. Deliberately not a second argument -- one clock, one knob, so the two
                # sensors can never be anchored to different drives.
                "oxts_timestamps": LaunchConfiguration("oxts_timestamps").perform(context),
            }],
            output="screen",
        ), "image_replay"))
        nodes.extend(node for node, _label in viz_nodes)

    # pipeline_replay owns the verdict. Its exit ends the launch; the other nodes outliving it
    # is normal, so only flag one that exits FIRST -- that one is a crash, not a teardown.
    state = {"replay_done": False}

    def on_replay_exit(event, _context):
        state["replay_done"] = True
        code = event.returncode
        # Deliberately NOT formatted as "STAGE6 <mode>: PASS|FAIL": that string is the gate's
        # own, and run_stage6.py greps for it. Re-emitting it here would turn a replay that
        # crashed before evaluating anything (MISSING) into a confident-looking FAIL.
        verdict = "gate PASSED" if code == 0 else "gate FAILED (or the replay aborted)"
        return [
            LogInfo(msg=f"pipeline_replay exited with code {code} -> {verdict}"),
            EmitEvent(event=Shutdown(reason=f"pipeline_replay exited with code {code}")),
        ]

    def _crash_guard(label: str, gated: bool = True):
        """Log a node dying before the replay reported. `gated=False` for the viz nodes: they
        are not in the measured path, so their death changes no number -- but it must still be
        visible, otherwise "Foxglove showed nothing" has no trace in the launch log at all."""
        consequence = ("the replay will report FAIL(no-peer) or stall; the real failure is in "
                       f"the {label} log above."
                       if gated else
                       "the gated path is unaffected (this node feeds no gate), but the live "
                       f"view is dead from here on; see the {label} log above.")

        def on_exit(event, _context):
            if state["replay_done"]:
                return None      # ordinary teardown after the gate reported
            return [LogInfo(
                msg=f"{label} exited with code {event.returncode} BEFORE pipeline_replay "
                    f"finished — {consequence}"
            )]
        return on_exit

    handlers = [
        RegisterEventHandler(OnProcessExit(target_action=replay_node, on_exit=on_replay_exit)),
        RegisterEventHandler(
            OnProcessExit(target_action=eskf_node, on_exit=_crash_guard("eskf_node"))),
        RegisterEventHandler(
            OnProcessExit(target_action=transform_node,
                          on_exit=_crash_guard("detection_transform_node"))),
        RegisterEventHandler(
            OnProcessExit(target_action=tracker_node, on_exit=_crash_guard("tracker_node"))),
    ]
    handlers += [
        RegisterEventHandler(
            OnProcessExit(target_action=viz, on_exit=_crash_guard(label, gated=False)))
        for viz, label in viz_nodes
    ]
    return nodes + handlers


def generate_launch_description() -> LaunchDescription:
    args = [
        DeclareLaunchArgument(
            "mode",
            default_value="baseline",
            description=f"Failure-mode preset: one of {sorted(PRESETS)}.",
        ),
        DeclareLaunchArgument(
            "foxglove",
            default_value="false",
            description="Also start foxglove_bridge, viz_markers, lidar_replay and "
                        "image_replay, and force rate_scale to 1.0.",
        ),
        DeclareLaunchArgument(
            "velodyne_dir",
            default_value=DEFAULT_VELODYNE,
            description="KITTI velodyne_points directory (holds data/ and timestamps.txt). "
                        "foxglove:=true only.",
        ),
        DeclareLaunchArgument(
            "lidar_stride",
            default_value="2",
            description="Keep every Nth lidar point. 1 = the full ~120k-point scan. "
                        "foxglove:=true only.",
        ),
        DeclareLaunchArgument(
            "image_dir",
            default_value=DEFAULT_IMAGE,
            description="KITTI image_02 directory (holds data/*.png and timestamps.txt). "
                        "foxglove:=true only.",
        ),
        DeclareLaunchArgument(
            "oxts_timestamps",
            default_value=DEFAULT_OXTS_TIMESTAMPS,
            description="OXTS timestamps.txt of the drive `cache_path` was built from — this "
                        "is what anchors the Velodyne and camera clocks to the replay's t=0. "
                        "Change it whenever you change cache_path, or the cloud and video align "
                        "to a foreign clock and freeze. foxglove:=true only.",
        ),
        DeclareLaunchArgument(
            "rate_scale",
            default_value="0.0",
            description="0 = replay as fast as possible; 1.0 = wall clock.",
        ),
        DeclareLaunchArgument(
            "cache_path",
            default_value=DEFAULT_CACHE,
            description="OXTS cache .npz (mounted in the container).",
        ),
        DeclareLaunchArgument(
            "output_npz",
            default_value=["/workspace/data/cache/stage6_", LaunchConfiguration("mode"), ".npz"],
            description="Where pipeline_replay records this run.",
        ),
        DeclareLaunchArgument(
            "baseline_npz",
            # "" = none. Do NOT invoke this as baseline_npz:="" -- an empty token is a launch
            # parse error, not "none". The baseline run simply omits the argument.
            default_value="",
            description="Baseline run npz for the ratio gates; empty for the baseline run "
                        "itself, which is the reference.",
        ),
    ]
    return LaunchDescription(args + [OpaqueFunction(function=_pipeline)])
