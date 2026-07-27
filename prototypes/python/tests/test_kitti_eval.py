"""Perfect tracker (hyp == GT boxes, stable ids) -> MOTA == 1.0, no switches."""
from kitti_boxes import Box3D
from kitti_eval import evaluate


def _car(x, tid):
    return Box3D(x=x, y=1.2, z=0.0, yaw=0.0, l=4.0, w=2.0, h=1.5, track_id=tid)


def test_perfect_tracker_mota_one():
    gt_frames = [[_car(1.0 * k, 0)] for k in range(5)]
    hyp_frames = [[(7, _car(1.0 * k, 7))] for k in range(5)]   # hyp id 7 == same object each frame
    m = evaluate(gt_frames, hyp_frames, iou_thresh=0.25)
    assert abs(m["mota"] - 1.0) < 1e-9
    assert int(m["num_switches"]) == 0


def test_missed_everything_mota_zero_or_negative():
    gt_frames = [[_car(1.0 * k, 0)] for k in range(5)]
    hyp_frames = [[] for _ in range(5)]
    m = evaluate(gt_frames, hyp_frames)
    assert m["mota"] <= 0.0
