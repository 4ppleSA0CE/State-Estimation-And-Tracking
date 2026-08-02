"""Host-side tests for the synthetic target simulator. Run from the repo root:
    python3 -m pytest ros2_ws/src/kf_bringup/test -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kf_bringup.targets import (  # noqa: E402
    BOX_DIM, CAR_DIMS, EGO_HEIGHT_M, N_TARGETS, TargetConfig, X, Y, Z, YAW,
    detect, enu_to_base_link, enu_to_map_bev, map_bev_to_enu, target_truth,
)

# Table A from the plan — hand-computed, NOT derived from the C++ side.
TABLE_A = [
    ([10.0, 2.0, -1.7, 0.5],        [10.0, 1.7, 2.0, -0.5]),
    ([101.0, 200.0, 5.0, 0.0],      [101.0, -5.0, 200.0, 0.0]),
    ([0.0, 10.0, 0.0, math.pi / 2], [0.0, 0.0, 10.0, -math.pi / 2]),
    ([9.0, -5.0, 3.0, -math.pi / 4],[9.0, -3.0, -5.0, math.pi / 4]),
    ([-2.0, -3.0, 0.5, -1.0],       [-2.0, -0.5, -3.0, 1.0]),
]


def _box(xyzyaw):
    b = np.zeros(BOX_DIM)
    b[X], b[Y], b[Z], b[YAW] = xyzyaw
    b[4], b[5], b[6] = CAR_DIMS
    return b


@pytest.mark.parametrize("enu,bev", TABLE_A)
def test_enu_to_map_bev_matches_the_pinned_table(enu, bev):
    got = enu_to_map_bev(_box(enu))
    assert got[[X, Y, Z, YAW]] == pytest.approx(bev, abs=1e-12)
    assert got[4:7] == pytest.approx(CAR_DIMS)          # dims carried through untouched


@pytest.mark.parametrize("enu,bev", TABLE_A)
def test_map_bev_to_enu_is_the_exact_inverse(enu, bev):
    assert map_bev_to_enu(_box(bev))[[X, Y, Z, YAW]] == pytest.approx(enu, abs=1e-12)


def test_enu_to_map_bev_is_not_an_identity_on_y_and_z():
    """Sentinel: a no-op permutation would pass a symmetric fixture. This one cannot."""
    got = enu_to_map_bev(_box([1.0, 2.0, 3.0, 0.4]))
    assert got[Y] == pytest.approx(-3.0)
    assert got[Z] == pytest.approx(2.0)
    assert got[YAW] == pytest.approx(-0.4)


def _rot_z(psi):
    c, s = math.cos(psi), math.sin(psi)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def test_enu_to_base_link_round_trips_through_a_known_ego_pose():
    ego_p = np.array([100.0, -50.0, 3.0])
    ego_r = _rot_z(0.7)
    enu = _box([120.0, -40.0, 1.3, 0.2])
    b = enu_to_base_link(enu, ego_p, ego_r)
    back = ego_r @ b[[X, Y, Z]] + ego_p
    assert back == pytest.approx(enu[[X, Y, Z]], abs=1e-9)
    assert b[YAW] == pytest.approx(0.2 - 0.7, abs=1e-12)


def test_enu_to_base_link_puts_a_target_dead_ahead_on_the_x_axis():
    """Ego heading 90 deg (facing North); a target due North must land on +x, y=0."""
    ego_p = np.array([0.0, 0.0, 0.0])
    b = enu_to_base_link(_box([0.0, 25.0, 0.0, math.pi / 2]), ego_p, _rot_z(math.pi / 2))
    assert b[X] == pytest.approx(25.0, abs=1e-9)
    assert b[Y] == pytest.approx(0.0, abs=1e-9)
    assert b[YAW] == pytest.approx(0.0, abs=1e-12)


def _frames(n=60, dt=0.1):
    return np.arange(n) * dt


def test_target_truth_shape_and_ground_plane():
    t = _frames()
    out = target_truth(TargetConfig(), t, np.array([10.0, 20.0, 4.0]), 0.3, 12.0)
    assert out.shape == (len(t), N_TARGETS, BOX_DIM)
    assert np.allclose(out[..., Z], 4.0 - EGO_HEIGHT_M)      # bottom-center on the ground
    assert np.allclose(out[..., 4:7], np.asarray(CAR_DIMS))


def test_constant_velocity_targets_move_in_a_straight_line():
    t = _frames()
    out = target_truth(TargetConfig(), t, np.zeros(3), 0.0, 10.0)
    for i in range(N_TARGETS):
        step = np.diff(out[:, i, :2], axis=0)
        assert np.allclose(step, step[0], atol=1e-9), f"target {i} is not CV"


def test_maneuver_target_turns_only_after_the_onset_time():
    cfg = TargetConfig(maneuver_target=3, maneuver_start_s=2.0, maneuver_omega=0.4)
    t = _frames(80)
    out = target_truth(cfg, t, np.zeros(3), 0.0, 10.0)
    yaw = np.unwrap(out[:, 3, YAW])
    assert abs(yaw[19] - yaw[0]) < 1e-9          # t < 2.0 s: still CV
    assert abs(yaw[-1] - yaw[0]) > 0.5           # after onset: clearly turning
    # the other three are untouched
    for i in (0, 1, 2):
        assert np.allclose(np.unwrap(out[:, i, YAW]), out[0, i, YAW], atol=1e-9)


def test_the_maneuver_onset_frame_is_inclusive():
    """The onset test above pins only "not before" and "clearly after", so `>` instead of `>=`
    merely delays the turn by one frame and survives it. Pin the boundary frame itself.
    """
    cfg = TargetConfig(maneuver_target=3, maneuver_start_s=2.0, maneuver_omega=0.4)
    t = _frames(30)
    assert t[20] == 2.0                                    # the fixture really does hit the onset
    yaw = np.unwrap(target_truth(cfg, t, np.zeros(3), 0.0, 10.0)[:, 3, YAW])
    assert yaw[19] == pytest.approx(yaw[0], abs=1e-12)     # t = 1.9 s: not yet turning
    assert yaw[20] - yaw[19] == pytest.approx(0.4 * 0.1, abs=1e-9)   # t = 2.0 s: already turning


def test_maneuver_off_by_default_leaves_every_target_straight():
    t = _frames(80)
    out = target_truth(TargetConfig(), t, np.zeros(3), 0.0, 10.0)
    for i in range(N_TARGETS):
        assert np.allclose(np.unwrap(out[:, i, YAW]), out[0, i, YAW], atol=1e-9)


def test_ct_propagation_preserves_speed():
    cfg = TargetConfig(maneuver_target=3, maneuver_start_s=0.0, maneuver_omega=0.6)
    out = target_truth(cfg, _frames(100), np.zeros(3), 0.0, 10.0)
    v = np.diff(out[:, 3, :2], axis=0) / 0.1
    speed = np.hypot(v[:, 0], v[:, 1])
    assert speed.max() - speed.min() < 1e-6      # a turn changes heading, never speed


def test_ct_step_is_the_exact_arc_chord_not_a_euler_step():
    """Speed preservation alone cannot see a `p += v*dt` CT branch -- an Euler step keeps the
    speed constant too, so only the absolute step length distinguishes them. A coordinated turn
    advances by the arc CHORD, 2(s/w)sin(w dt/2), strictly shorter than the straight-line s*dt.
    The reference speed is read off the CV branch so this does not hard-code the layout.
    """
    dt, omega = 0.1, 0.6
    t = _frames(60, dt)
    straight = target_truth(TargetConfig(), t, np.zeros(3), 0.0, 10.0)
    speed = float(np.linalg.norm(straight[1, 3, :2] - straight[0, 3, :2])) / dt
    cfg = TargetConfig(maneuver_target=3, maneuver_start_s=0.0, maneuver_omega=omega)
    chord = np.linalg.norm(np.diff(target_truth(cfg, t, np.zeros(3), 0.0, 10.0)[:, 3, :2],
                                   axis=0), axis=1)
    assert chord == pytest.approx(2.0 * (speed / omega) * math.sin(omega * dt / 2.0), abs=1e-9)
    assert chord.max() < speed * dt          # strictly inside the straight-line step


def test_detector_is_deterministic_for_a_given_seed():
    cfg = TargetConfig(clutter_lambda=2.0)
    boxes = np.stack([_box([20.0, 1.0, -1.7, 0.0]), _box([30.0, -4.0, -1.7, 0.3])])
    a = detect(boxes, np.random.default_rng(7), cfg)
    b = detect(boxes, np.random.default_rng(7), cfg)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_invisible_targets_do_not_consume_the_rng_stream():
    """Same-seed determinism compares the code against itself, so it cannot see a reordering of
    the stream. This can: the visibility gate runs BEFORE the p_detect draw, so an out-of-view
    target must not shift the noise applied to a visible one.
    """
    cfg = TargetConfig(p_detect=1.0)          # default noise stds, so the stream is observable
    visible = _box([20.0, 1.0, -1.7, 0.0])
    alone, src_alone = detect(np.stack([visible]), np.random.default_rng(5), cfg)
    with_invisible, src_with = detect(
        np.stack([_box([-30.0, 0.0, -1.7, 0.0]), visible]), np.random.default_rng(5), cfg)
    assert src_alone.tolist() == [0] and src_with.tolist() == [1]
    assert with_invisible == pytest.approx(alone, abs=0.0)


def test_detector_shuffles_the_emitted_order_and_keeps_boxes_paired_with_ids():
    """A detector emits no ordering. Dropping the final permutation leaves src in input order,
    which the identity-permutation counter rejects; permuting boxes and ids with DIFFERENT
    draws would break the x-to-id pairing, which the per-seed check rejects.
    """
    cfg = TargetConfig(p_detect=1.0, det_pos_std=0.0, det_yaw_std=0.0)
    boxes = np.stack([_box([10.0 + 5.0 * i, 0.0, -1.7, 0.0]) for i in range(4)])
    shuffled = 0
    for seed in range(20):
        dets, src = detect(boxes, np.random.default_rng(seed), cfg)
        assert sorted(src.tolist()) == [0, 1, 2, 3]                 # nothing lost or duplicated
        assert dets[:, X] == pytest.approx(10.0 + 5.0 * src)        # each box kept its own id
        shuffled += int(src.tolist() != [0, 1, 2, 3])
    assert shuffled > 0, "the emitted order is never permuted"


def test_visibility_gate_boundaries_are_exclusive():
    """Off-by-one sentinels for all three gates, each hit exactly in floating point:
    x == 0 is beside the sensor, ‖p‖ == max_range_m is out, bearing == half-FOV is out.
    """
    cfg = TargetConfig(p_detect=1.0, det_pos_std=0.0, det_yaw_std=0.0,
                       max_range_m=50.0, fov_deg=90.0)
    for label, xy in (("x == 0", (0.0, 0.0)),
                      ("range == max", (50.0, 0.0)),
                      ("bearing == half FOV", (1.0, 1.0))):
        dets, src = detect(np.stack([_box([xy[0], xy[1], -1.7, 0.0])]),
                           np.random.default_rng(0), cfg)
        assert dets.shape == (0, BOX_DIM) and src.shape == (0,), f"{label} was not rejected"


def test_detector_drops_targets_behind_and_beyond_range():
    cfg = TargetConfig(p_detect=1.0, det_pos_std=0.0, det_yaw_std=0.0, max_range_m=50.0)
    boxes = np.stack([
        _box([20.0, 0.0, -1.7, 0.0]),     # visible
        _box([-20.0, 0.0, -1.7, 0.0]),    # behind
        _box([80.0, 0.0, -1.7, 0.0]),     # too far
        _box([10.0, 40.0, -1.7, 0.0]),    # outside the 90 deg FOV
    ])
    dets, src = detect(boxes, np.random.default_rng(0), cfg)
    assert sorted(src.tolist()) == [0]
    assert dets[0][X] == pytest.approx(20.0)


def test_detector_noise_is_zero_when_the_stds_are_zero():
    cfg = TargetConfig(p_detect=1.0, det_pos_std=0.0, det_yaw_std=0.0)
    boxes = np.stack([_box([20.0, 1.0, -1.7, 0.25])])
    dets, _ = detect(boxes, np.random.default_rng(3), cfg)
    assert dets[0][[X, Y, Z, YAW]] == pytest.approx([20.0, 1.0, -1.7, 0.25])


def test_clutter_count_matches_the_requested_poisson_mean():
    cfg = TargetConfig(p_detect=0.0, clutter_lambda=3.0)
    rng = np.random.default_rng(11)
    boxes = np.stack([_box([20.0, 0.0, -1.7, 0.0])])
    counts = [len(detect(boxes, rng, cfg)[1]) for _ in range(2000)]
    assert abs(float(np.mean(counts)) - 3.0) < 0.15
    assert all(s == -1 for s in np.concatenate([detect(boxes, rng, cfg)[1] for _ in range(20)]))


def test_clutter_is_off_by_default():
    cfg = TargetConfig(p_detect=1.0)
    boxes = np.stack([_box([20.0, 0.0, -1.7, 0.0])])
    for seed in range(50):
        _, src = detect(boxes, np.random.default_rng(seed), cfg)
        assert (src >= 0).all()


def test_p_detect_zero_yields_no_target_detections():
    cfg = TargetConfig(p_detect=0.0)
    boxes = np.stack([_box([20.0, 0.0, -1.7, 0.0])])
    dets, src = detect(boxes, np.random.default_rng(0), cfg)
    assert dets.shape == (0, BOX_DIM) and src.shape == (0,)


def test_config_rejects_out_of_range_values():
    for kwargs in ({"p_detect": 1.5}, {"p_detect": -0.1}, {"det_pos_std": -1.0},
                   {"clutter_lambda": -1.0}, {"max_range_m": 0.0}, {"fov_deg": 400.0}):
        with pytest.raises(ValueError):
            TargetConfig(**kwargs)


def test_config_rejects_a_bad_maneuver_target_and_a_negative_yaw_std():
    """The two fields the case list above never reached. `maneuver_target` indexes _LAYOUT, so
    an out-of-range value would silently select no target (or IndexError deeper in), and a
    negative `det_yaw_std` is rejected by numpy far from the call site.
    """
    for kwargs in ({"maneuver_target": -2}, {"maneuver_target": N_TARGETS},
                   {"maneuver_target": N_TARGETS + 5}, {"det_yaw_std": -1e-12},
                   {"det_yaw_std": -1.0}):
        with pytest.raises(ValueError):
            TargetConfig(**kwargs)
    TargetConfig(maneuver_target=-1)                # the documented "no maneuver" sentinel
    TargetConfig(maneuver_target=0)
    TargetConfig(maneuver_target=N_TARGETS - 1)     # the last valid index
    TargetConfig(det_yaw_std=0.0)


# --------------------------------------------------------------------------------------
# Scenario placement and speeds -- spec section 5.1, transcribed by hand from the design
# doc. Every fixture above passes ego_yaw0 = 0.0, where sin(0) == 0 erases both lateral
# terms of the ego-frame -> ENU rotation and cos(0) == 1 hides its scaling, so the
# placement math is exercised only by the tests below.
# --------------------------------------------------------------------------------------

def test_initial_placement_at_a_ninety_degree_ego_heading_is_hand_computed():
    """Ego facing North (yaw = pi/2): ego-frame "forward" becomes ENU +y and "left" becomes
    ENU -x. Offsets are spec 5.1's: (+25, 0), (+60, +3.5), (+80, +7.0), (+40, -12.0).
    """
    ego_p = np.array([10.0, 20.0, 4.0])
    out = target_truth(TargetConfig(), _frames(2), ego_p, math.pi / 2, 11.0)
    expected = [(10.0 - 0.0, 20.0 + 25.0),          # 0 leading
                (10.0 - 3.5, 20.0 + 60.0),          # 1 oncoming near lane
                (10.0 - 7.0, 20.0 + 80.0),          # 2 oncoming far lane
                (10.0 + 12.0, 20.0 + 40.0)]         # 3 crossing (left = -12 -> +x)
    for i, (ex, ey) in enumerate(expected):
        assert out[0, i, X] == pytest.approx(ex, abs=1e-9), f"target {i} initial ENU x"
        assert out[0, i, Y] == pytest.approx(ey, abs=1e-9), f"target {i} initial ENU y"


def test_initial_placement_at_a_forty_five_degree_ego_heading_is_hand_computed():
    """pi/2 leaves cos(yaw0) == 0, so the lateral contribution to ENU y is invisible there.
    At pi/4 both axes carry the same weight r = sqrt(2)/2 and each of the four signs in
    (x, y) = (c*fwd - s*left, s*fwd + c*left) is separately observable.
    """
    r = math.sqrt(2.0) / 2.0
    out = target_truth(TargetConfig(), _frames(2), np.zeros(3), math.pi / 4, 11.0)
    expected = [(r * (25.0 - 0.0), r * (25.0 + 0.0)),
                (r * (60.0 - 3.5), r * (60.0 + 3.5)),
                (r * (80.0 - 7.0), r * (80.0 + 7.0)),
                (r * (40.0 + 12.0), r * (40.0 - 12.0))]
    for i, (ex, ey) in enumerate(expected):
        assert out[0, i, X] == pytest.approx(ex, abs=1e-9), f"target {i} initial ENU x"
        assert out[0, i, Y] == pytest.approx(ey, abs=1e-9), f"target {i} initial ENU y"


def test_initial_forward_and_lateral_offsets_match_the_spec_scenario_table():
    """Same contract, read back in the ego frame so the numbers are literally spec 5.1's
    table: project the ENU placement onto the ego heading and its left-normal.
    """
    ego_p = np.array([-30.0, 7.0, 2.0])
    for ego_yaw0 in (0.0, math.pi / 2, math.pi / 4, -1.1, 2.9):
        c0, s0 = math.cos(ego_yaw0), math.sin(ego_yaw0)
        out = target_truth(TargetConfig(), _frames(2), ego_p, ego_yaw0, 11.0)
        d = out[0, :, [X, Y]].T - ego_p[:2]                  # ENU offset from the ego origin
        fwd = d[:, 0] * c0 + d[:, 1] * s0                    # along the heading
        left = -d[:, 0] * s0 + d[:, 1] * c0                  # 90 deg CCW from the heading
        assert fwd == pytest.approx([25.0, 60.0, 80.0, 40.0], abs=1e-9), f"yaw0={ego_yaw0}"
        assert left == pytest.approx([0.0, 3.5, 7.0, -12.0], abs=1e-9), f"yaw0={ego_yaw0}"


def test_initial_speeds_match_the_spec_scenario_table():
    """Spec 5.1: the leader runs at the ego's own initial speed, 1 and 2 are oncoming at 8
    and 10 m/s, 3 crosses at 6 m/s. Speeds are read off the CV step, not from _LAYOUT.
    """
    dt = 0.1
    for ego_speed0 in (11.0, 4.0):
        out = target_truth(TargetConfig(), _frames(3, dt), np.zeros(3), 0.0, ego_speed0)
        v = (out[1, :, :2] - out[0, :, :2]) / dt
        speed = np.hypot(v[:, 0], v[:, 1])
        assert speed == pytest.approx([ego_speed0, 8.0, 10.0, 6.0], abs=1e-9), \
            f"ego_speed0={ego_speed0}"
    # The leader must FOLLOW the ego speed, which the two ego_speed0 values above pin: a
    # constant would have to equal both 11.0 and 4.0.


def _ang_diff(a, b):
    return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)


@pytest.mark.parametrize("ego_yaw0", [0.0, math.pi / 2, math.pi / 4, -1.1, 2.9])
def test_initial_headings_are_offsets_from_the_ego_heading(ego_yaw0):
    """Spec 5.1: 0 travels with the ego, 1 and 2 are oncoming (pi), 3 crosses right-to-left
    (+pi/2). The offsets are relative to the ego heading, which only a nonzero ego_yaw0 shows.
    """
    dt = 0.1
    out = target_truth(TargetConfig(), _frames(3, dt), np.zeros(3), ego_yaw0, 11.0)
    v = (out[1, :, :2] - out[0, :, :2]) / dt
    for i, dpsi in enumerate([0.0, math.pi, math.pi, math.pi / 2]):
        assert _ang_diff(math.atan2(v[i, 1], v[i, 0]), ego_yaw0 + dpsi) < 1e-9, \
            f"target {i} velocity heading"
        assert _ang_diff(out[0, i, YAW], ego_yaw0 + dpsi) < 1e-9, f"target {i} box yaw"


def test_the_maneuver_onset_is_relative_to_the_first_frame_not_absolute_time():
    """`maneuver_start_s` is documented as seconds after the FIRST frame. Every other onset
    fixture starts its clock at t = 0, where "relative" and "absolute" are the same number.
    A KITTI frame clock does not start at zero, so use one that does not either.
    """
    cfg = TargetConfig(maneuver_target=3, maneuver_start_s=2.0, maneuver_omega=0.4)
    t = 100.0 + _frames(30)
    assert t[0] == 100.0 and t[20] == 102.0          # the fixture straddles t0 + 2.0 s exactly
    out = target_truth(cfg, t, np.zeros(3), 0.0, 10.0)
    yaw = np.unwrap(out[:, 3, YAW])
    assert yaw[19] == pytest.approx(yaw[0], abs=1e-12)               # t0 + 1.9 s: still CV
    assert yaw[20] - yaw[19] == pytest.approx(0.4 * 0.1, abs=1e-9)   # t0 + 2.0 s: turning
    # ...and the non-maneuvering targets stay straight on this clock too
    for i in (0, 1, 2):
        assert np.allclose(np.unwrap(out[:, i, YAW]), out[0, i, YAW], atol=1e-9)


def test_enu_to_base_link_wraps_the_relative_yaw_into_the_principal_branch():
    """A box at yaw 3.0 seen from an ego at -3.0 has a raw yaw difference of 6.0 rad, outside
    the principal branch. Every other yaw fixture here is already inside it, so dropping the
    wrap is invisible to them.
    """
    ego_p = np.zeros(3)
    b = enu_to_base_link(_box([10.0, 0.0, -1.7, 3.0]), ego_p, _rot_z(-3.0))
    assert b[YAW] == pytest.approx(6.0 - 2.0 * math.pi, abs=1e-9)
    assert -math.pi <= b[YAW] <= math.pi
    b2 = enu_to_base_link(_box([10.0, 0.0, -1.7, -3.0]), ego_p, _rot_z(3.0))
    assert b2[YAW] == pytest.approx(2.0 * math.pi - 6.0, abs=1e-9)
    # The branch convention itself: the wrap is half-open [-pi, pi), so exactly pi maps to -pi.
    b3 = enu_to_base_link(_box([10.0, 0.0, -1.7, math.pi]), ego_p, _rot_z(0.0))
    assert b3[YAW] == pytest.approx(-math.pi, abs=1e-12)


# --------------------------------------------------------------------------------------
# Detector model internals
# --------------------------------------------------------------------------------------

def test_position_noise_moves_x_y_z_and_never_the_box_dimensions_or_yaw():
    """`d[[X, Y, Z]] += ...` writing to any other columns -- [L, W, H] or [X, Y, YAW] -- keeps
    the detection count, the ids and the determinism intact, so only a per-column check sees it.
    """
    cfg = TargetConfig(p_detect=1.0, det_pos_std=0.5, det_yaw_std=0.0)
    truth = _box([20.0, 1.0, -1.7, 0.25])
    moved = np.zeros(3, dtype=bool)
    for seed in range(40):
        d = detect(np.stack([truth]), np.random.default_rng(seed), cfg)[0][0]
        assert d[4:7] == pytest.approx(CAR_DIMS, abs=0.0), f"seed {seed}: l/w/h perturbed"
        assert d[YAW] == pytest.approx(0.25, abs=0.0), f"seed {seed}: yaw moved at zero yaw std"
        moved |= np.abs(d[[X, Y, Z]] - truth[[X, Y, Z]]) > 1e-12
    assert moved.all(), f"position axes never perturbed: {np.flatnonzero(~moved).tolist()}"


def test_yaw_noise_moves_only_yaw():
    """The mirror of the check above: with the position std at zero, x/y/z must be untouched
    and yaw must actually move.
    """
    cfg = TargetConfig(p_detect=1.0, det_pos_std=0.0, det_yaw_std=0.05)
    truth = _box([20.0, 1.0, -1.7, 0.25])
    moved = 0
    for seed in range(40):
        d = detect(np.stack([truth]), np.random.default_rng(seed), cfg)[0][0]
        assert d[[X, Y, Z]] == pytest.approx(truth[[X, Y, Z]], abs=0.0), f"seed {seed}"
        assert d[4:7] == pytest.approx(CAR_DIMS, abs=0.0), f"seed {seed}"
        moved += int(abs(d[YAW] - 0.25) > 1e-12)
    assert moved == 40


def _residual_spread(std, seed, n=4000):
    cfg = TargetConfig(p_detect=1.0, det_pos_std=std, det_yaw_std=0.0)
    truth = _box([20.0, 1.0, -1.7, 0.0])
    boxes = np.stack([truth])
    rng = np.random.default_rng(seed)
    res = np.array([detect(boxes, rng, cfg)[0][0][[X, Y, Z]] - truth[[X, Y, Z]]
                    for _ in range(n)])
    return float(res.std())


def test_position_noise_spread_scales_with_det_pos_std():
    """`rng.normal(0.0, cfg.det_pos_std, 3)` with the config replaced by the default literal
    0.35 still produces plausible-looking noise and passes every qualitative check. Only the
    magnitude distinguishes them, so measure it. Two DIFFERENT seeds, so this cannot pass by
    the two draws sharing one scaled stream.

    12000 residual samples per spread: the relative error of a sample std is 1/sqrt(2N) ~=
    0.6%, and the worst deviation observed over 40 seed pairs was 2.3%, so 8% is ~10 sigma.
    """
    lo, hi = _residual_spread(0.5, 3), _residual_spread(2.0, 97)
    assert lo == pytest.approx(0.5, rel=0.08)
    assert hi == pytest.approx(2.0, rel=0.08)
    assert hi / lo == pytest.approx(4.0, rel=0.08)


def test_clutter_boxes_sit_on_the_ground_inside_the_range_gate_and_the_fov():
    """The Poisson COUNT is pinned above; the box CONTENT was not. Clutter that lands at the
    wrong height, past max_range_m, or behind the sensor still counts correctly.
    """
    max_range, fov = 45.0, 70.0
    cfg = TargetConfig(p_detect=0.0, clutter_lambda=4.0, max_range_m=max_range, fov_deg=fov)
    half_fov = math.radians(fov) * 0.5
    rng = np.random.default_rng(23)
    boxes = np.stack([_box([20.0, 0.0, -1.7, 0.0])])
    n = 0
    for _ in range(300):
        dets, src = detect(boxes, rng, cfg)
        assert (src == -1).all()
        for c in dets:
            n += 1
            assert c[Z] == pytest.approx(-EGO_HEIGHT_M, abs=0.0)     # exactly on the ground
            assert c[4:7] == pytest.approx(CAR_DIMS, abs=0.0)
            assert c[X] > 0.0                                        # in front of the sensor
            r = math.hypot(c[X], c[Y])
            assert 2.0 - 1e-9 <= r <= max_range + 1e-9, f"clutter range {r}"
            assert abs(math.atan2(c[Y], c[X])) <= half_fov + 1e-9, "clutter outside the wedge"
            assert -math.pi <= c[YAW] <= math.pi
    assert n > 500, f"only {n} clutter boxes drawn -- the fixture is not exercising anything"


def test_clutter_fills_the_visible_wedge_rather_than_hugging_the_axis():
    """Complement to the bound checks: a clutter model pinned to a narrow band would satisfy
    every inequality above. Pin that the draws actually span the range and the bearing.
    """
    cfg = TargetConfig(p_detect=0.0, clutter_lambda=6.0, max_range_m=45.0, fov_deg=70.0)
    half_fov = math.radians(70.0) * 0.5
    rng = np.random.default_rng(5)
    boxes = np.stack([_box([20.0, 0.0, -1.7, 0.0])])
    rs, bearings = [], []
    for _ in range(400):
        for c in detect(boxes, rng, cfg)[0]:
            rs.append(math.hypot(c[X], c[Y]))
            bearings.append(math.atan2(c[Y], c[X]))
    rs, bearings = np.asarray(rs), np.asarray(bearings)
    assert rs.min() < 5.0 and rs.max() > 42.0            # spans (2, max_range_m)
    assert bearings.min() < -0.9 * half_fov and bearings.max() > 0.9 * half_fov
    assert abs(float(np.mean(bearings))) < 0.05          # symmetric about boresight


def _wrap_ref(a):
    """Independent re-statement of the module's angle wrap, for the RNG-order replay below."""
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


def test_the_rng_draw_order_within_one_detection_is_position_then_yaw():
    """`detect`'s docstring makes the RNG call order part of the contract. Same-seed
    determinism compares the code against itself and is structurally blind to a swap, and both
    draws are Gaussian and zero-mean so no statistical check separates them either. Replay the
    stream independently instead.
    """
    cfg = TargetConfig(p_detect=0.9, det_pos_std=0.4, det_yaw_std=0.05)
    truth = _box([20.0, 1.0, -1.7, 0.25])
    dets, src = detect(np.stack([truth]), np.random.default_rng(19), cfg)

    ref = np.random.default_rng(19)
    assert ref.random() < 0.9                       # the detect draw comes first at this seed
    dpos = ref.normal(0.0, 0.4, 3)                  # then the 3-vector of position noise
    dyaw = ref.normal(0.0, 0.05)                    # then the scalar yaw noise
    assert src.tolist() == [0]
    assert dets[0][[X, Y, Z]] == pytest.approx(truth[[X, Y, Z]] + dpos, abs=1e-12)
    assert dets[0][YAW] == pytest.approx(_wrap_ref(0.25 + dyaw), abs=1e-12)
    assert abs(dpos).min() > 1e-6 and abs(dyaw) > 1e-6      # the fixture is not all zeros


def test_the_full_rng_call_order_is_pinned_end_to_end():
    """The whole contract in one replay: per visible target in index order (detect draw,
    position noise, yaw noise), then the clutter count, then per-clutter (range, bearing,
    yaw), then exactly one permutation. Invisible targets never touch the stream.
    """
    max_range, fov, lam, pd = 60.0, 90.0, 2.0, 0.6
    cfg = TargetConfig(p_detect=pd, det_pos_std=0.4, det_yaw_std=0.05,
                       clutter_lambda=lam, max_range_m=max_range, fov_deg=fov)
    half_fov = math.radians(fov) * 0.5
    truth = np.stack([_box([20.0, 1.0, -1.7, 0.25]),
                      _box([-5.0, 0.0, -1.7, 0.0]),      # behind: must not touch the stream
                      _box([30.0, -6.0, -1.7, -0.5])])
    dets, src = detect(truth, np.random.default_rng(18), cfg)

    ref = np.random.default_rng(18)
    expect, expect_src = [], []
    for i in (0, 2):                                     # visible targets, in index order
        if ref.random() >= pd:
            continue
        b = truth[i].copy()
        b[[X, Y, Z]] += ref.normal(0.0, 0.4, 3)
        b[YAW] = _wrap_ref(b[YAW] + ref.normal(0.0, 0.05))
        expect.append(b)
        expect_src.append(i)
    n_detected = len(expect)
    for _ in range(int(ref.poisson(lam))):
        r = ref.uniform(2.0, max_range)
        a = ref.uniform(-half_fov, half_fov)
        c = np.zeros(BOX_DIM)
        c[X], c[Y], c[Z] = r * math.cos(a), r * math.sin(a), -EGO_HEIGHT_M
        c[YAW] = ref.uniform(-math.pi, math.pi)
        c[4], c[5], c[6] = CAR_DIMS
        expect.append(c)
        expect_src.append(-1)
    perm = ref.permutation(len(expect))

    assert n_detected == 2 and len(expect) - n_detected >= 2, \
        "the seed must detect both visible targets and draw clutter, or this proves little"
    assert src.tolist() == [expect_src[j] for j in perm]
    assert dets == pytest.approx(np.asarray(expect)[perm], abs=1e-12)
