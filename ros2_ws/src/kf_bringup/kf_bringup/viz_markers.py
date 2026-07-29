"""Stage 6 live view: /tracks + /detections_map + /targets/truth -> /viz/markers, /ego/state
plus the OXTS cache -> /viz/trails and the two ego cars.

Started only by `full_pipeline.launch.py foxglove:=true`; nothing in the gated path consumes
this node's output, so it can never change a verdict. All three box inputs are in `map_bev` and
all three encode box yaw about the (downward) y axis, so one decoder serves the lot.

    /targets/truth   green LINE_LIST wireframe
    /detections_map  grey  LINE_LIST wireframe
    /tracks          CUBE coloured by id % 10, plus a TEXT_VIEW_FACING label carrying the id

ONE timer publishes ONE array holding all three layers, led by a DELETEALL. That is not a
stylistic choice: RViz and Foxglove both apply DELETEALL to the whole display rather than to the
sending namespace, so three independently published arrays would each erase the other two. The
0.5 s `lifetime` is the belt to DELETEALL's braces -- if this node or the pipeline dies, the
scene decays instead of freezing on the last frame.

--------------------------------------------------------------------------------------------
The two ego cars, and why the ego is a BOX and not an axis triad (section 5.5, extended)

    ego_est    translucent cyan  CUBE      in `base_link`, the ESKF ESTIMATE
    ego_truth  solid amber       LINE_LIST in `map`,       the OXTS TRUTH
    ego_error  white LINE_LIST + TEXT_VIEW_FACING spanning the gap between them

Before this the ego was only the `base_link` TF triad: three thin axis lines a couple of metres
long. Green target boxes 25 m away then float in a scene with no visible vehicle to be relative
to, which is exactly as confusing as it sounds. A car-shaped box fixes it, and drawing the car
TWICE -- once at the estimate, once at truth -- turns the localization error from a pair of
curves into two vehicles pulling apart, which is the thing this whole stage is about.

Geometry. The KITTI platform is a VW Passat B6 estate (4.77 x 1.82 x 1.52 m); this renders it as
4.5 x 1.8 x 1.5 m, deliberately a touch short so the box reads as "the ego car" without
overhanging the lane markings in the point cloud.

Mounting height, which is the part that is easy to get wrong. `base_link` IS the OXTS/IMU frame,
not the road: the IMU box sits 0.93 m above the ground (lidar_replay's docstring derives this
independently -- the Velodyne is 1.73 m up, `calib_imu_to_velo` puts it 0.80 m above the IMU, and
1.73 - 0.80 = 0.93). A CUBE is placed by its CENTRE, so a car box drawn at the `base_link` origin
would float with its wheels 0.93 m off the road and its roof 1.68 m up. The centre therefore goes
at z = h/2 - 0.93 = -0.18 m in `base_link`, which puts the wheels exactly on the road plane the
LiDAR sweep draws. `x = y = 0`: the OXTS box is roughly amidships and no calibration file in this
dataset gives a body-to-IMU longitudinal offset, so inventing one would be a guess dressed as a
measurement.

    NOTE, recorded rather than silently worked around: targets.py's `EGO_HEIGHT_M = 1.7` is
    labelled "OXTS/IMU origin above the ground" but is really the Velodyne's 1.73 m. The
    synthetic target boxes therefore sit ~0.77 m below the road the point cloud shows. That
    constant is frozen (it is baked into the recorded runs and pinned by test_targets.py), so
    this node does NOT copy the error across for cosmetic agreement -- the ego car is placed on
    the real road, and the discrepancy stays visible instead of being laundered.

Frames do the pose maths, so this node does none. The estimate car is published in `base_link`
and simply rides eskf_node's map -> base_link TF -- if the ESKF drifts, the car drifts, with the
full estimated attitude, and there is no second copy of the pose to fall out of sync. The truth
car is published in `map` (ENU) at the cached OXTS position and heading, so the gap on screen is
the estimate-vs-truth error and nothing else.

Telling them apart: SOLID versus WIREFRAME, not merely two colours. Two translucent cubes blend
into an indistinct mush whenever they overlap -- which is most of a baseline run, where the error
is a few tenths of a metre. A filled cyan car inside an open amber cage stays readable when the
two coincide AND when they separate, and the cyan/amber pair is already the established legend
for estimate/truth from the trails, so the cars need no new one. A white segment joins the two
box centres with the live horizontal error, in metres, printed at its midpoint: under
`mode:=gps_dropout` the ESKF drifts to 4.49 m, and the number next to a visibly 4.5 m-long
segment is what makes that read as a magnitude rather than as "a bit of a wobble". The error is
the HORIZONTAL one (East/North, height excluded), which is stage6_gates' definition verbatim --
a number on screen that disagreed with the number in the gate report would be worse than none.

Topic choice, deliberately: all three go on /viz/markers, WITH the DELETEALL and the 0.5 s
lifetime, NOT on /viz/trails. They are per-frame state, not history -- every one of them is
rebuilt from the newest /ego/state at 10 Hz, so DELETEALL costs them nothing and the lifetime
buys the same decay-on-death the box layers get. On /viz/trails (lifetime 0, no DELETEALL) the
final pair of cars would instead hang in the scene forever after the run ended, a ghost vehicle
parked at the last stamp. Each marker keeps a stable ns+id (ego_est/0, ego_truth/0, ego_error/0
and /1) so a republish REPLACES its predecessor -- belt and braces behind the DELETEALL, and the
thing that keeps this correct if the DELETEALL policy is ever revisited. /viz/markers is already
enabled in the Foxglove layout, so riding it also means no layout edit can leave the ego
invisible again.

--------------------------------------------------------------------------------------------
The ego trails, and why they are a SEPARATE topic (design doc section 5.5, extended)

    /viz/trails   cyan  LINE_STRIP -- the ESKF ESTIMATE's travelled path, from /ego/state
                  amber LINE_STRIP -- the OXTS TRUTH path over the same stamps

Two live curves, one `map`-frame scene, and the gap between them IS the localization error the
whole stage exists to make visible. Before this the driven path was only inferable from
Foxglove's TF parent-child connector line, which is a line between two frame ORIGINS, not a path.

The trails cannot ride in the /viz/markers array. That array is led by a DELETEALL, and a
DELETEALL applies to the whole receiving display -- which is precisely how the box layers avoid
erasing each other. A trail is by definition the one thing in the scene that must survive
between frames, so it goes on its own topic, with `lifetime = 0` (never expires) and no
DELETEALL. Same ns + same id every publish, so each republish REPLACES its predecessor rather
than stacking; the strip simply gets longer.

The cost is honest: the full strip is re-sent at PUBLISH_HZ. At 100 Hz ego and drive_0001's
11.65 s that peaks at ~1166 points (~28 kB) per trail, which is nothing over a local websocket.
`TRAIL_CAPACITY` bounds it anyway for any longer drive.

Where the TRUTH comes from: this node loads the SAME OXTS cache .npz that pipeline_replay
replays, and keys it by exact int64-nanosecond stamp. pipeline_replay is deliberately NOT taught
to publish a truth path -- it owns the gate, its recorded npz schema is frozen, and an extra
publisher plus an extra declared parameter (which lands in the recorded `params_json`) would put
viz-only work inside the measured path. Reading the cache here instead costs the gated runs
exactly nothing, because this node does not run in them at all. Keying by the same stamp that
carries the estimate is what makes the two trails advance in lockstep, so any visible offset
between them is real error and never a rendering artifact. The SAME lookup feeds the amber truth
car and the error readout below, so all three read from one source. A missing or unreadable cache
degrades to "estimate trail and estimate car only" with an error logged: killing the whole live
view over the optional half of one overlay would be the worse failure.

--------------------------------------------------------------------------------------------
The static map -> map_bev transform (design doc section 4.3)

    map      is ENU:                              x East, y North, z Up
    map_bev  is that same ENU frame written in
             the KITTI-camera convention:         x East, y Down,  z North

so a point's COORDINATES map as  p_bev = M p_map  with  M = [[1,0,0],[0,0,-1],[0,1,0]]
(x_bev = x_enu, y_bev = -z_enu, z_bev = y_enu; det M = +1, so it is a rotation, not a mirror).

A TransformStamped(frame_id=map, child_frame_id=map_bev) carries the rotation that takes CHILD
coordinates to PARENT coordinates -- p_map = R p_bev -- so R is the INVERSE of M:

    R = M^-1 = M^T = [[1, 0, 0],
                      [0, 0, 1],
                      [0,-1, 0]]  = R_x(-90 deg)

Read R by columns and it is self-checking: they are the map_bev axes written in ENU, namely
(1,0,0)=East, (0,0,-1)=Down, (0,1,0)=North. As a quaternion (w, x, y, z):

    (cos(-45 deg), sin(-45 deg), 0, 0) = (sqrt(0.5), -sqrt(0.5), 0, 0)

Sanity check -- the one that catches a mirrored or rolled scene. A target 25 m due North of an
ego at the ENU origin is at ENU (0, 25, 0), which this pipeline publishes in map_bev as
(0, 0, 25). Rendering it back into map gives R (0, 0, 25) = (0, 25, 0): 25 m North of the ego,
level with it. The opposite sign, R_x(+90 deg), would place it at (0, -25, 0) -- due SOUTH,
which is exactly the plausible-looking-but-wrong picture this derivation exists to rule out.

--------------------------------------------------------------------------------------------
Marker stamps are copied from the SOURCE message, never re-stamped from the wall clock. The
whole pipeline runs on the KITTI OXTS time base (t = 0 .. 11.65 s), including eskf_node's
map -> base_link TF, and re-stamping here would put the geometry on a different clock from the
ego pose it is meant to sit beside. It costs nothing: RViz and Foxglove both measure `lifetime`
from receipt, not from the header.
"""

from __future__ import annotations

import math
from collections import deque
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point, TransformStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import ColorRGBA
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray

from kf_bringup.kitti_replay import _stamp_from_secs
from kf_msgs.msg import DetectionArray, EgoState, TrackArray

MAP_FRAME = "map"
MAP_BEV_FRAME = "map_bev"
BASE_LINK_FRAME = "base_link"

PUBLISH_HZ = 10.0        # matches the 10 Hz detection/track cadence; no reason to render faster
LIFETIME_S = 0.5
LINE_WIDTH_M = 0.12      # LINE_LIST uses scale.x only
TEXT_HEIGHT_M = 1.2
LABEL_CLEARANCE_M = 0.6  # how far above the box roof the id label floats
MIN_EXTENT_M = 0.05      # floor on cube scale: a zero extent is a viewer warning, not a box

TRAILS_TOPIC = "/viz/trails"
TRAIL_WIDTH_M = 0.5      # LINE_STRIP uses scale.x only
TRAIL_CAPACITY = 20000   # bound on either trail; drive_0001 needs 1166, so this never bites here

# The ego car. (length, width, height) m -- a VW Passat B6 estate is 4.77 x 1.82 x 1.52.
EGO_DIMS_M = (4.5, 1.8, 1.5)
# base_link IS the OXTS/IMU frame, 0.93 m above the road (lidar_replay: velodyne 1.73 m up,
# 0.80 m above the IMU). Subtract it to get the road plane; see the module docstring.
EGO_GROUND_OFFSET_M = 0.93
EGO_WIRE_WIDTH_M = 0.14      # thicker than the target wireframes: this box is the reference
ERROR_LABEL_LIFT_M = 2.5     # above the ego roof (roof is 0.57 m above base_link)
DEFAULT_CACHE = ("/workspace/data/cache/"
                 "kitti_raw_2011_09_26_drive_0001_extract_oxts_v1.npz")

_SQRT_HALF = math.sqrt(0.5)

TRUTH_COLOR = (0.10, 0.85, 0.25, 0.95)
DET_COLOR = (0.65, 0.65, 0.65, 0.75)
LABEL_COLOR = (1.0, 1.0, 1.0, 0.95)
# Neither trail may collide with the green truth boxes, the grey detections or the translucent
# Tableau track cubes -- the trails are the one pair a viewer must tell apart at a glance.
EST_PATH_COLOR = (0.05, 0.95, 1.00, 1.0)     # cyan  -- the ESKF estimate
TRUTH_PATH_COLOR = (1.00, 0.65, 0.00, 1.0)   # amber -- OXTS truth
# The two ego cars reuse the trail legend, so cyan/amber already means estimate/truth by the time
# a viewer looks at them. alpha 0.40 on the solid one keeps the LiDAR sweep visible THROUGH the
# car -- an opaque box parked over the sensor origin hides the nearest few metres of the cloud.
EGO_EST_COLOR = EST_PATH_COLOR[:3] + (0.40,)
EGO_TRUTH_COLOR = TRUTH_PATH_COLOR           # a wireframe has nothing to see through; keep a=1
ERROR_COLOR = (1.0, 1.0, 1.0, 0.95)          # white = annotation, like the track id labels

# id % 10 -> RGB. Tableau-10, which stays distinguishable against both Foxglove themes.
TRACK_PALETTE = (
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17), (0.84, 0.15, 0.16),
    (0.58, 0.40, 0.74), (0.55, 0.34, 0.29), (0.89, 0.47, 0.76), (0.50, 0.50, 0.50),
    (0.74, 0.74, 0.13), (0.09, 0.75, 0.81),
)
TRACK_ALPHA = 0.55       # translucent so the green truth wireframe stays visible inside a cube

# BEV footprint edge loop; the 3D box is this ring at the bottom face, again at the roof, plus
# the four verticals.
_RING = ((0, 1), (1, 2), (2, 3), (3, 0))


def _yaw_about_y(q) -> float:
    """Decode yaw about the y axis: atan2(R(0,2), R(2,2)).

    The map_bev convention -- y is DOWN, so the ground-plane rotation is about y. This is the
    exact inverse of the encoding tracker_node, detection_transform_node and pipeline_replay all
    use (`w = cos(psi/2)`, `y = sin(psi/2)`). Decoding about z here would silently flatten every
    box yaw to zero and merely look like a badly tracked scene.
    """
    return math.atan2(2.0 * (q.w * q.y + q.x * q.z), 1.0 - 2.0 * (q.y * q.y + q.x * q.x))


def _box_of(obj) -> tuple[float, float, float, float, float, float, float]:
    """(x, y, z, yaw, l, w, h) from a Detection or a Track -- identical field layout in both.

    `y` is the box BOTTOM face (spec section 4.2): the box occupies [y - h, y] because y is down.
    """
    p = obj.pose.position
    d = obj.dimensions
    return (p.x, p.y, p.z, _yaw_about_y(obj.pose.orientation), d.x, d.y, d.z)


def _bev_corners(x: float, z: float, yaw: float, length: float, width: float):
    """Four ground-plane corners, mirroring kf_tracker box3d::bevCorners exactly.

    x' = c*xl + s*zl + x,  z' = -s*xl + c*zl + z -- the KITTI rotation_y sign convention. Using
    the textbook +sin here instead would draw every box mirrored about its own heading.
    """
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length * 0.5, width * 0.5
    return [(c * xl + s * zl + x, -s * xl + c * zl + z)
            for xl, zl in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]


def _pt(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x, p.y, p.z = float(x), float(y), float(z)
    return p


def _wire_points(box) -> list:
    """24 endpoints = the 12 edges of one box, as a LINE_LIST expects (pairs, not a strip)."""
    x, y, z, yaw, length, width, height = box
    ring = _bev_corners(x, z, yaw, length, width)
    roof = y - height          # y is down, so the roof is the SMALLER y
    pts: list = []
    for i, j in _RING:
        pts += [_pt(ring[i][0], y, ring[i][1]), _pt(ring[j][0], y, ring[j][1])]
        pts += [_pt(ring[i][0], roof, ring[i][1]), _pt(ring[j][0], roof, ring[j][1])]
    for i in range(4):
        pts += [_pt(ring[i][0], y, ring[i][1]), _pt(ring[i][0], roof, ring[i][1])]
    return pts


def _enu_wire(x: float, y: float, z_ground: float, yaw: float,
              length: float, width: float, height: float) -> list:
    """24 endpoints = the 12 edges of one box in ENU (x East, y North, z Up), for a LINE_LIST.

    Deliberately NOT _wire_points, which is a map_bev routine: there the ground plane is (x, z),
    "up" is -y and the corner rotation carries KITTI's rotation_y sign. Here the ground plane is
    (x, y), up is +z and yaw is the ordinary CCW-from-East heading, so the rotation is the
    textbook [[c, -s], [s, c]]. Reusing the other one would draw the truth car mirrored about its
    own heading -- invisible while driving straight, wrong on every turn.

    `z_ground` is the bottom face, so the box occupies [z_ground, z_ground + height].
    """
    c, s = math.cos(yaw), math.sin(yaw)
    hl, hw = length * 0.5, width * 0.5
    ring = [(x + c * xl - s * yl, y + s * xl + c * yl)
            for xl, yl in ((hl, hw), (hl, -hw), (-hl, -hw), (-hl, hw))]
    roof = z_ground + height
    pts: list = []
    for i, j in _RING:
        pts += [_pt(ring[i][0], ring[i][1], z_ground), _pt(ring[j][0], ring[j][1], z_ground)]
        pts += [_pt(ring[i][0], ring[i][1], roof), _pt(ring[j][0], ring[j][1], roof)]
    for i in range(4):
        pts += [_pt(ring[i][0], ring[i][1], z_ground), _pt(ring[i][0], ring[i][1], roof)]
    return pts


def _color(rgba) -> ColorRGBA:
    c = ColorRGBA()
    c.r, c.g, c.b, c.a = (float(v) for v in rgba)
    return c


def _ns(stamp) -> int:
    """int64 nanoseconds from a builtin_interfaces/Time -- pipeline_replay's exact keying."""
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


class VizMarkers(Node):
    """Latest-message-wins cache plus a 10 Hz renderer. Deliberately stateless beyond that."""

    def __init__(self) -> None:
        super().__init__("viz_markers")

        # Reliable to match every upstream publisher (all KeepLast(2000).reliable()); depth 10 is
        # plenty because only the newest message of each topic is ever drawn.
        qos = QoSProfile(
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=QoSReliabilityPolicy.RELIABLE,
        )

        self._tracks: TrackArray | None = None
        self._dets: DetectionArray | None = None
        self._truth: DetectionArray | None = None

        self.declare_parameter("cache_path", DEFAULT_CACHE)
        self._truth_by_ns = self._load_truth_path(str(self.get_parameter("cache_path").value))
        self._est_trail: deque = deque(maxlen=TRAIL_CAPACITY)
        self._truth_trail: deque = deque(maxlen=TRAIL_CAPACITY)
        self._last_ego_stamp = None
        self._truth_pose = None      # newest matched (x, y, z, yaw) ENU -- drives the truth car

        self._pub = self.create_publisher(MarkerArray, "/viz/markers", qos)
        self._trail_pub = self.create_publisher(MarkerArray, TRAILS_TOPIC, qos)
        self.create_subscription(TrackArray, "/tracks", self._on_tracks, qos)
        self.create_subscription(DetectionArray, "/detections_map", self._on_dets, qos)
        self.create_subscription(DetectionArray, "/targets/truth", self._on_truth, qos)
        self.create_subscription(EgoState, "/ego/state", self._on_ego, qos)

        # Static, so it is latched (transient_local) and a Foxglove session that connects late
        # still receives it. Broadcast once, in the constructor -- it never changes.
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform(self._map_to_map_bev())

        self._lifetime = Duration()
        self._lifetime.sec = int(LIFETIME_S)
        self._lifetime.nanosec = int(round((LIFETIME_S - int(LIFETIME_S)) * 1e9))

        self.create_timer(1.0 / PUBLISH_HZ, self._publish)
        self.get_logger().info(
            f"viz_markers up: /viz/markers at {PUBLISH_HZ:g} Hz in {MAP_BEV_FRAME}, "
            f"{TRAILS_TOPIC} in {MAP_FRAME} ({len(self._truth_by_ns)} truth samples), "
            f"ego car {EGO_DIMS_M[0]:g}x{EGO_DIMS_M[1]:g}x{EGO_DIMS_M[2]:g} m dropped "
            f"{EGO_GROUND_OFFSET_M:g} m to the road, "
            f"static {MAP_FRAME} -> {MAP_BEV_FRAME} broadcast"
        )

    # ------------------------------------------------------------------
    # Ground truth.
    # ------------------------------------------------------------------
    def _load_truth_path(self, cache_path: str) -> dict:
        """{stamp_ns: (x, y, z, yaw) ENU} from the OXTS cache pipeline_replay is replaying.

        The keys are built with the SAME `_stamp_from_secs` that stamps every published message,
        so an /ego/state stamp is looked up by exact int64 equality -- no nearest-match, matching
        the discipline the rest of the stage keys on.

        The yaw column feeds the amber TRUTH CAR (the trail only ever needed a position). It is
        `roll_pitch_yaw[:, 2]`, the same OXTS heading pipeline_replay turns into the initial
        attitude quaternion, so the truth car points where the ESKF was initialized to point.
        """
        try:
            with np.load(Path(cache_path), allow_pickle=False) as cache:
                ts = np.asarray(cache["timestamps"], dtype=float)
                enu = np.asarray(cache["enu_position_m"], dtype=float)
                # Fetched separately and defensively: an OXTS cache without a heading column
                # should cost the truth CAR its heading, not cost the truth TRAIL its existence.
                rpy = (np.asarray(cache["roll_pitch_yaw"], dtype=float)
                       if "roll_pitch_yaw" in cache.files else None)
        except Exception as exc:                # noqa: BLE001 -- viz degrades, never dies
            self.get_logger().error(
                f"could not read the OXTS cache {cache_path} ({type(exc).__name__}: {exc}); the "
                f"cyan ESTIMATE trail and car still render, the amber TRUTH trail and car will "
                f"stay empty")
            return {}
        if rpy is None or rpy.shape[0] != ts.size:
            self.get_logger().warn(
                f"{cache_path} has no usable roll_pitch_yaw column; the amber TRUTH CAR will be "
                f"drawn pointing due East. Its position, and both trails, are unaffected")
            yaw = np.zeros(ts.size)
        else:
            yaw = rpy[:, 2]
        return {_ns(_stamp_from_secs(float(t))): (float(enu[i][0]), float(enu[i][1]),
                                                  float(enu[i][2]), float(yaw[i]))
                for i, t in enumerate(ts)}

    # ------------------------------------------------------------------
    # Frames.
    # ------------------------------------------------------------------
    def _map_to_map_bev(self) -> TransformStamped:
        """R_x(-90 deg) -- see the module docstring for the derivation and the sanity check."""
        tf = TransformStamped()
        tf.header.stamp = self.get_clock().now().to_msg()
        tf.header.frame_id = MAP_FRAME          # parent
        tf.child_frame_id = MAP_BEV_FRAME       # child
        tf.transform.translation.x = 0.0        # same origin; the frames differ only in axes
        tf.transform.translation.y = 0.0
        tf.transform.translation.z = 0.0
        tf.transform.rotation.w = _SQRT_HALF
        tf.transform.rotation.x = -_SQRT_HALF
        tf.transform.rotation.y = 0.0
        tf.transform.rotation.z = 0.0
        return tf

    # ------------------------------------------------------------------
    # Subscriptions -- cache only; rendering is the timer's job.
    # ------------------------------------------------------------------
    def _on_tracks(self, msg: TrackArray) -> None:
        self._tracks = msg

    def _on_dets(self, msg: DetectionArray) -> None:
        self._dets = msg

    def _on_truth(self, msg: DetectionArray) -> None:
        self._truth = msg

    def _on_ego(self, msg: EgoState) -> None:
        """Append to both trails, and latch the matched truth pose for the truth car.

        Unlike the box layers the TRAILS are APPEND, not latest-wins; `_truth_pose` is the one
        latest-wins piece here because a car is a pose, not a history.
        """
        self._last_ego_stamp = msg.header.stamp
        p = msg.pose.position
        self._est_trail.append((p.x, p.y, p.z))
        truth = self._truth_by_ns.get(_ns(msg.header.stamp))
        if truth is not None:
            self._truth_trail.append((truth[0], truth[1], truth[2]))
            self._truth_pose = truth
        elif self._truth_by_ns:
            # A stamp with no exact truth match. pipeline_replay derives the /ego/state stamps
            # from these very OXTS timestamps, so this should never fire; if it does, HOLD the
            # last matched pose rather than dropping the car -- a truth car that blinks out for
            # one frame reads as a bug in the estimator, which is the opposite of the point. The
            # held pose is stale, hence the warning; the printed error goes stale with it.
            self.get_logger().warn(
                f"/ego/state stamp {_ns(msg.header.stamp)} ns has no exact match among the "
                f"{len(self._truth_by_ns)} OXTS truth stamps; the amber TRUTH CAR is holding its "
                f"last matched pose and the error readout is stale",
                throttle_duration_sec=5.0)

    # ------------------------------------------------------------------
    # Rendering.
    # ------------------------------------------------------------------
    def _marker(self, stamp, ns: str, marker_id: int, kind: int,
                frame: str = MAP_BEV_FRAME) -> Marker:
        """`frame` defaults to map_bev because every box layer lives there; the ego cars are the
        exception (base_link for the estimate, map for truth) and pass it explicitly."""
        m = Marker()
        m.header.stamp = stamp
        m.header.frame_id = frame
        m.ns = ns
        m.id = marker_id
        m.type = kind
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0      # identity unless the caller overrides it
        m.lifetime = self._lifetime
        return m

    def _wireframe(self, msg, ns: str, rgba) -> list:
        """One LINE_LIST carrying every box in the layer -- cheaper than a marker per box, and
        the whole layer then shares one id, so a shrinking detection count leaves nothing behind.
        """
        if msg is None or not msg.detections:
            return []
        m = self._marker(msg.header.stamp, ns, 0, Marker.LINE_LIST)
        m.scale.x = LINE_WIDTH_M
        m.color = _color(rgba)
        for det in msg.detections:
            m.points.extend(_wire_points(_box_of(det)))
        return [m]

    def _track_markers(self) -> list:
        msg = self._tracks
        if msg is None or not msg.tracks:
            return []
        out: list = []
        for track in msg.tracks:
            x, y, z, _yaw, length, width, height = _box_of(track)
            r, g, b = TRACK_PALETTE[track.id % len(TRACK_PALETTE)]

            cube = self._marker(msg.header.stamp, "tracks", track.id, Marker.CUBE)
            # Box position is the BOTTOM face; a CUBE is placed by its centre, so lift it by h/2
            # -- and "up" is -y here. Skipping this sinks every track half a box into the ground.
            cube.pose.position = _pt(x, y - height * 0.5, z)
            cube.pose.orientation = track.pose.orientation   # already yaw about y, reuse verbatim
            # Local axes after a yaw about y: x is the heading, y is down (height), z is lateral.
            cube.scale.x = max(length, MIN_EXTENT_M)
            cube.scale.y = max(height, MIN_EXTENT_M)
            cube.scale.z = max(width, MIN_EXTENT_M)
            cube.color = _color((r, g, b, TRACK_ALPHA))
            out.append(cube)

            label = self._marker(msg.header.stamp, "track_ids", track.id,
                                 Marker.TEXT_VIEW_FACING)
            label.pose.position = _pt(x, y - height - LABEL_CLEARANCE_M, z)   # above the roof
            label.scale.x = TEXT_HEIGHT_M     # unused for text, set so no scale reads as zero
            label.scale.y = TEXT_HEIGHT_M
            label.scale.z = TEXT_HEIGHT_M     # this is the one TEXT_VIEW_FACING actually uses
            label.color = _color(LABEL_COLOR)
            label.text = str(track.id)
            out.append(label)
        return out

    def _ego_markers(self) -> list:
        """The estimate car, the truth car, and the error annotation between them.

        Rides /viz/markers with the box layers: per-frame state, rebuilt from the newest
        /ego/state every tick, so the DELETEALL costs nothing and the 0.5 s lifetime makes the
        cars decay with the rest of the scene if the pipeline dies. See the module docstring.
        """
        if self._last_ego_stamp is None:
            return []                    # no ego pose yet -- nothing to draw, and no TF either
        stamp = self._last_ego_stamp
        length, width, height = EGO_DIMS_M
        out: list = []

        # 1. The ESTIMATE car, in base_link. No pose maths at all: eskf_node's map -> base_link TF
        # already carries the full estimated position AND attitude, so the box inherits both, and
        # there is no second copy of the estimate here to drift out of sync with the real one.
        car = self._marker(stamp, "ego_est", 0, Marker.CUBE, BASE_LINK_FRAME)
        # A CUBE is placed by its CENTRE and base_link is the IMU, 0.93 m up: h/2 - 0.93 = -0.18 m
        # puts the wheels on the road. Dropping this term floats the whole car by 0.93 m.
        car.pose.position = _pt(0.0, 0.0, height * 0.5 - EGO_GROUND_OFFSET_M)
        # base_link is x forward, y left, z up -- so scale reads (length, width, height) in order.
        car.scale.x, car.scale.y, car.scale.z = length, width, height
        car.color = _color(EGO_EST_COLOR)
        out.append(car)

        if self._truth_pose is None or not self._est_trail:
            return out                   # no cache, or no match yet: estimate-only, as promised

        tx, ty, tz, tyaw = self._truth_pose
        ex, ey, ez = self._est_trail[-1]     # newest /ego/state ENU position, same stamp

        # 2. The TRUTH car, in map (ENU), as a WIREFRAME: an open cage stays readable when it
        # overlaps the solid estimate car, which is most of a baseline run.
        wire = self._marker(stamp, "ego_truth", 0, Marker.LINE_LIST, MAP_FRAME)
        wire.scale.x = EGO_WIRE_WIDTH_M      # LINE_LIST uses scale.x only
        wire.color = _color(EGO_TRUTH_COLOR)
        wire.points = _enu_wire(tx, ty, tz - EGO_GROUND_OFFSET_M, tyaw, length, width, height)
        out.append(wire)

        # 3. The error, drawn AND printed. Horizontal (East/North) only, which is stage6_gates'
        # `_ego_series` definition verbatim -- an on-screen number that disagreed with the gate
        # report would be worse than no number. The segment joins the two ego ORIGINS, which sit
        # at the identical spot inside each car, so the drawn gap and the printed metres are the
        # same quantity and the label lands in the middle of the divergence rather than beside it.
        link = self._marker(stamp, "ego_error", 0, Marker.LINE_LIST, MAP_FRAME)
        link.scale.x = EGO_WIRE_WIDTH_M
        link.color = _color(ERROR_COLOR)
        link.points = [_pt(ex, ey, ez), _pt(tx, ty, tz)]
        out.append(link)

        label = self._marker(stamp, "ego_error", 1, Marker.TEXT_VIEW_FACING, MAP_FRAME)
        label.pose.position = _pt(0.5 * (ex + tx), 0.5 * (ey + ty),
                                  0.5 * (ez + tz) + ERROR_LABEL_LIFT_M)   # above both roofs
        label.scale.x = TEXT_HEIGHT_M        # unused for text, set so no scale reads as zero
        label.scale.y = TEXT_HEIGHT_M
        label.scale.z = TEXT_HEIGHT_M        # this is the one TEXT_VIEW_FACING actually uses
        label.color = _color(ERROR_COLOR)
        label.text = f"ESKF err {math.hypot(ex - tx, ey - ty):.2f} m"
        out.append(label)
        return out

    def _publish(self) -> None:
        array = MarkerArray()

        clear = Marker()
        clear.header.stamp = self.get_clock().now().to_msg()
        clear.header.frame_id = MAP_BEV_FRAME
        clear.action = Marker.DELETEALL
        array.markers.append(clear)

        array.markers.extend(self._wireframe(self._truth, "truth", TRUTH_COLOR))
        array.markers.extend(self._wireframe(self._dets, "detections", DET_COLOR))
        array.markers.extend(self._track_markers())
        array.markers.extend(self._ego_markers())
        self._pub.publish(array)
        self._publish_trails()

    def _trail(self, ns: str, points: deque, rgba) -> list:
        """One LINE_STRIP in `map`, lifetime 0, same ns+id every time -- see the docstring.

        Two points is the minimum a LINE_STRIP can draw; a 1-point strip is a viewer warning.
        """
        if len(points) < 2:
            return []
        m = Marker()
        m.header.stamp = self._last_ego_stamp     # source stamp, like every other marker here
        m.header.frame_id = MAP_FRAME             # ENU, NOT map_bev: /ego/state is an ENU pose
        m.ns = ns
        m.id = 0
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.lifetime = Duration()                   # 0 = never expires; the whole point of a trail
        m.scale.x = TRAIL_WIDTH_M
        m.color = _color(rgba)
        m.points = [_pt(*p) for p in points]
        return [m]

    def _publish_trails(self) -> None:
        """Separate topic, no DELETEALL. /viz/markers' DELETEALL clears its whole display, so a
        trail sharing that array would be wiped 10 times a second and never appear to grow."""
        if self._last_ego_stamp is None:
            return
        array = MarkerArray()
        array.markers.extend(self._trail("ego_path_est", self._est_trail, EST_PATH_COLOR))
        array.markers.extend(self._trail("ego_path_truth", self._truth_trail, TRUTH_PATH_COLOR))
        if array.markers:
            self._trail_pub.publish(array)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VizMarkers()
    try:
        rclpy.spin(node)
    # ExternalShutdownException is what rclpy raises when the launch SIGINTs this node during a
    # normal teardown. Letting it escape printed a 15-line traceback and exit code 1, which the
    # launch then reported as "process has died" — a viz node that shut down correctly looking
    # exactly like one that crashed, in the log a demo viewer reads.
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
