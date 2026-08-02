"""Dataset-free tests for the KITTI Car tracking evaluation driver.

Covers the four things that silently go wrong: hand-checkable MOTA, snapshot aliasing,
the val-split list, and pooling-vs-averaging the aggregate.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_tracking_eval import (  # noqa: E402
    AB3DMOT_VAL_SPLIT,
    IOU_THRESH,
    MIN_DETECTION_SCORE,
    load_detection_frames,
    parse_ab3dmot_detections,
    pooled_metrics,
    snapshot_tracks,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototypes" / "python"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "prototypes" / "python" / "tracking"))
from kitti_boxes import Box3D  # noqa: E402
from kitti_eval import accumulate, evaluate  # noqa: E402
from kitti_tracker import BoxTrack, KittiTrackerConfig  # noqa: E402
from kitti_tracking_loader import val_sequences  # noqa: E402


def _box(x=0.0, z=0.0, tid=-1):
    # KITTI camera frame: y is down (box bottom), z forward. 1.5 x 1.6 x 4.0 car.
    return Box3D(x=x, y=1.5, z=z, yaw=0.0, l=4.0, w=1.6, h=1.5, score=1.0, track_id=tid)


# --- the val split -----------------------------------------------------------------

def test_val_split_has_exactly_eleven_sequences():
    assert len(AB3DMOT_VAL_SPLIT) == 11


def test_val_split_matches_ab3dmot_utils_list():
    # AB3DMOT_libs/utils.py:39
    assert AB3DMOT_VAL_SPLIT == [
        "0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019",
    ]


def test_frozen_loader_val_sequences_is_byte_identical_to_the_split():
    assert val_sequences() == AB3DMOT_VAL_SPLIT


# --- hand-computable MOTA ----------------------------------------------------------

def _two_frame_case():
    """Frame 0: GT{1,2}, one hypothesis on GT 1  -> 1 miss.
    Frame 1: GT{1,2}; hyp 100 now sits on GT 2 and hyp 101 on GT 1 -> GT 1 switches 100->101.

    FN=1, FP=0, IDSW=1, GT total=4  =>  MOTA = 1 - (1+0+1)/4 = 0.5
    """
    gt_frames = [
        [_box(z=0.0, tid=1), _box(z=20.0, tid=2)],
        [_box(z=0.0, tid=1), _box(z=20.0, tid=2)],
    ]
    hyp_frames = [
        [(100, _box(z=0.0))],
        [(100, _box(z=20.0)), (101, _box(z=0.0))],
    ]
    return gt_frames, hyp_frames


def test_synthetic_two_frame_pair_gives_hand_computed_mota():
    result = evaluate(*_two_frame_case(), iou_thresh=0.25)
    assert result["mota"] == pytest.approx(0.5)


def test_synthetic_two_frame_pair_has_exactly_one_id_switch():
    result = evaluate(*_two_frame_case(), iou_thresh=0.25)
    assert int(result["num_switches"]) == 1


def test_identical_boxes_match_perfectly_so_motp_is_zero():
    result = evaluate(*_two_frame_case(), iou_thresh=0.25)
    assert result["motp"] == pytest.approx(0.0, abs=1e-12)


# --- the snapshot trap -------------------------------------------------------------

class _FakeTrack:
    """Mimics BoxTrack: mutable state, box() returns a fresh object."""

    def __init__(self, track_id, x, z):
        self.id = track_id
        self.x = x
        self.z = z

    def box(self):
        return _box(x=self.x, z=self.z)


def test_snapshot_does_not_alias_a_mutating_track():
    track = _FakeTrack(7, x=1.0, z=2.0)
    snap = snapshot_tracks([track])
    track.x, track.z = 99.0, 99.0
    assert snap[0][0] == 7
    assert (snap[0][1].x, snap[0][1].z) == (1.0, 2.0)


def test_snapshot_of_a_real_track_survives_a_later_update():
    cfg = KittiTrackerConfig()
    track = BoxTrack(3, _box(x=0.0, z=0.0), cfg)
    snap = snapshot_tracks([track])
    before = (snap[0][1].x, snap[0][1].z, snap[0][1].l)
    for _ in range(20):
        track.predict()
        track.update(Box3D(x=50.0, y=1.5, z=80.0, yaw=1.0, l=9.0, w=3.0, h=2.5))
    after = (snap[0][1].x, snap[0][1].z, snap[0][1].l)
    assert before == after
    assert track.box().z != snap[0][1].z   # the track really did move


def test_snapshot_returns_id_box_pairs_in_track_order():
    tracks = [_FakeTrack(2, 0.0, 0.0), _FakeTrack(5, 1.0, 1.0)]
    assert [tid for tid, _ in snapshot_tracks(tracks)] == [2, 5]


# --- pooling vs averaging ----------------------------------------------------------

def _lopsided_pair():
    """Sequence A: 1 GT object, perfectly tracked      -> MOTA 1.0
    Sequence B: 10 GT objects, none tracked            -> MOTA 0.0
    mean of the two          = 0.5
    pooled over 11 objects   = 1 - 10/11 = 0.0909...
    """
    a_gt = [[_box(tid=1)]]
    a_hyp = [[(1, _box())]]
    b_gt = [[_box(z=40.0 * i, tid=i) for i in range(10)]]
    b_hyp = [[]]
    return (a_gt, a_hyp), (b_gt, b_hyp)


def test_per_sequence_metrics_are_the_expected_extremes():
    (a_gt, a_hyp), (b_gt, b_hyp) = _lopsided_pair()
    assert evaluate(a_gt, a_hyp)["mota"] == pytest.approx(1.0)
    assert evaluate(b_gt, b_hyp)["mota"] == pytest.approx(0.0)


def test_aggregate_pools_accumulators_instead_of_averaging_mota():
    (a_gt, a_hyp), (b_gt, b_hyp) = _lopsided_pair()
    accs = [accumulate(a_gt, a_hyp), accumulate(b_gt, b_hyp)]
    summary = pooled_metrics(accs, ["A", "B"])
    pooled = summary.loc["OVERALL", "mota"]
    averaged = summary.loc[["A", "B"], "mota"].mean()

    assert pooled == pytest.approx(1.0 - 10.0 / 11.0)
    assert averaged == pytest.approx(0.5)
    assert pooled != pytest.approx(averaged)


def test_pooled_summary_carries_a_row_per_sequence_plus_overall():
    (a_gt, a_hyp), (b_gt, b_hyp) = _lopsided_pair()
    summary = pooled_metrics([accumulate(a_gt, a_hyp), accumulate(b_gt, b_hyp)], ["A", "B"])
    assert list(summary.index) == ["A", "B", "OVERALL"]


# --- AB3DMOT detection file layout -------------------------------------------------

AB3DMOT_SAMPLE = (
    # frame,type,x1,y1,x2,y2,score,h,w,l,x,y,z,rot_y,alpha
    "0,2,786.7492,180.1760,1241.0000,374.0000,12.2286,1.5206,1.6824,4.4501,2.9312,1.6089,6.4281,-1.5828,-2.0107\n"
    "0,2,718.1009,178.6554,858.6496,280.5958,11.7592,1.5622,1.6099,3.8266,3.0233,1.6841,13.1890,-1.5741,-1.7995\n"
    "3,2,384.3615,191.2274,463.4154,244.3732,10.3184,1.5171,1.5954,3.7611,-6.0804,2.1715,23.7919,1.5839,1.8341\n"
)


def test_parse_ab3dmot_detections_groups_by_frame(tmp_path):
    path = tmp_path / "0001.txt"
    path.write_text(AB3DMOT_SAMPLE)
    frames = parse_ab3dmot_detections(path)
    assert sorted(frames) == [0, 3]
    assert len(frames[0]) == 2
    assert len(frames[3]) == 1


def test_parse_ab3dmot_detections_maps_columns_to_box_fields(tmp_path):
    path = tmp_path / "0001.txt"
    path.write_text(AB3DMOT_SAMPLE)
    b = parse_ab3dmot_detections(path)[0][0]
    assert (b.h, b.w, b.l) == pytest.approx((1.5206, 1.6824, 4.4501))
    assert (b.x, b.y, b.z) == pytest.approx((2.9312, 1.6089, 6.4281))
    assert b.yaw == pytest.approx(-1.5828)
    assert b.score == pytest.approx(12.2286)
    assert b.track_id == -1


def test_parse_ab3dmot_detections_keeps_every_detection_no_score_gate(tmp_path):
    path = tmp_path / "0001.txt"
    path.write_text(AB3DMOT_SAMPLE + "4,2,0,0,1,1,0.0001,1.5,1.6,4.0,0,1.5,5.0,0.0,0.0\n")
    frames = parse_ab3dmot_detections(path)
    assert sum(len(v) for v in frames.values()) == 4
    assert frames[4][0].score == pytest.approx(0.0001)


def test_detection_frames_dispatch_reads_ab3dmot_files(tmp_path):
    det_dir = tmp_path / "dets"
    det_dir.mkdir()
    (det_dir / "0001.txt").write_text(AB3DMOT_SAMPLE)
    frames = load_detection_frames("0001", source="ab3dmot", ab3dmot_det_dir=det_dir)
    assert sum(len(v) for v in frames.values()) == 3


def test_detection_frames_dispatch_rejects_unknown_source(tmp_path):
    with pytest.raises(ValueError):
        load_detection_frames("0001", source="nope", ab3dmot_det_dir=tmp_path)


# --- the pinned score convention ---------------------------------------------------

def test_headline_run_applies_no_detection_score_gate():
    assert MIN_DETECTION_SCORE is None


def test_default_dispatch_keeps_the_low_scoring_tail(tmp_path):
    det_dir = tmp_path / "dets"
    det_dir.mkdir()
    (det_dir / "0001.txt").write_text(
        AB3DMOT_SAMPLE + "0,2,0,0,1,1,-0.8000,1.5,1.6,4.0,0,1.5,5.0,0.0,0.0\n"
    )
    frames = load_detection_frames("0001", source="ab3dmot", ab3dmot_det_dir=det_dir)
    assert sum(len(v) for v in frames.values()) == 4


def test_score_gate_is_available_for_the_sensitivity_sweep(tmp_path):
    det_dir = tmp_path / "dets"
    det_dir.mkdir()
    (det_dir / "0001.txt").write_text(AB3DMOT_SAMPLE)   # scores 12.23, 11.76, 10.32
    frames = load_detection_frames("0001", source="ab3dmot", ab3dmot_det_dir=det_dir,
                                   min_score=11.0)
    assert sum(len(v) for v in frames.values()) == 2
    assert 3 not in frames                              # frame emptied by the gate is dropped


def test_pinned_iou_threshold_is_the_ab3dmot_car_convention():
    assert IOU_THRESH == 0.25
