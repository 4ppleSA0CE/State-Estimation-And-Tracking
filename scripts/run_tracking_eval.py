#!/usr/bin/env python3
"""Run the IMM Car tracker over the KITTI Tracking val split and write MOTA/MOTP/IDF1/ID-sw.

Grid: {AB3DMOT PointRCNN detections, GT-as-detections} x {3D-IoU association}, plus a
Mahalanobis-association ablation on a few sequences. Aggregates by pooling accumulators
across sequences (MOTA does not average).

Pinned evaluation conventions:
  * 3D-IoU match threshold        0.25   (AB3DMOT's Car convention; the frozen evaluate() default)
  * detection score threshold     none   (every detection is fed to the tracker; AB3DMOT gates at
                                          output, and GT-as-detections carry score 1.0, so any input
                                          gate would make the two sources incomparable)
  * DontCare / non-Car rows       dropped at parse time
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
# The tracking prototypes import each other by bare name, so both dirs go on the path.
for _p in (REPO_ROOT / "prototypes" / "python", REPO_ROOT / "prototypes" / "python" / "tracking"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import motmetrics as mm  # noqa: E402

from kitti_boxes import Box3D  # noqa: E402
from kitti_eval import accumulate  # noqa: E402
from kitti_tracker import KittiTracker, KittiTrackerConfig  # noqa: E402
from kitti_tracking_loader import KittiTrackingConfig, load_gt  # noqa: E402

# AB3DMOT's KITTI validation split, AB3DMOT_libs/utils.py:39.
AB3DMOT_VAL_SPLIT = ["0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019"]

IOU_THRESH = 0.25          # pinned: AB3DMOT's Car convention
MIN_DETECTION_SCORE = None  # pinned: no input-side score gate
METRICS = ["mota", "motp", "idf1", "num_switches",
           "num_objects", "num_detections", "num_false_positives", "num_misses"]

RESULTS_DIR = REPO_ROOT / "data" / "results"
PER_SEQUENCE_CSV = RESULTS_DIR / "tracking_per_sequence.csv"
SUMMARY_CSV = RESULTS_DIR / "tracking_summary.csv"

# AB3DMOT detection files are comma-separated with 15 columns:
#   0 frame, 1 type, 2-5 bbox2d, 6 score, 7 h, 8 w, 9 l, 10 x, 11 y, 12 z, 13 rot_y, 14 alpha
# (their loader reads dets = seq_dets[:, 7:14]). This is NOT the whitespace-separated 17-column
# label_02 layout that the frozen parse_label_file expects, hence the separate parser here.
_AB3DMOT_COLUMNS = 15


def parse_ab3dmot_detections(path) -> dict[int, list[Box3D]]:
    """AB3DMOT Car detection file -> {frame: [Box3D]}. Every detection is kept (no score gate)."""
    frames: dict[int, list[Box3D]] = {}
    with open(path) as f:
        for line in f:
            cols = line.strip().split(",")
            if len(cols) < _AB3DMOT_COLUMNS:
                continue
            frames.setdefault(int(float(cols[0])), []).append(
                Box3D(
                    x=float(cols[10]), y=float(cols[11]), z=float(cols[12]),
                    yaw=float(cols[13]),
                    l=float(cols[9]), w=float(cols[8]), h=float(cols[7]),
                    score=float(cols[6]),
                    track_id=-1,
                )
            )
    return frames


def load_detection_frames(seq: str, source: str, ab3dmot_det_dir=None,
                          data_cfg: KittiTrackingConfig | None = None,
                          min_score: float | None = MIN_DETECTION_SCORE) -> dict[int, list[Box3D]]:
    """Detections per frame. 'ab3dmot' parses their PointRCNN Car files; 'gt' uses the labels
    as perfect detections (the ceiling run), stripped of their track ids.

    min_score is None for the pinned headline run. It exists only for the sensitivity sweep, and
    never applies to 'gt' (those boxes are all score 1.0 by construction)."""
    data_cfg = data_cfg or KittiTrackingConfig()
    if source == "gt":
        return {f: [Box3D(b.x, b.y, b.z, b.yaw, b.l, b.w, b.h, 1.0, -1) for b in boxes]
                for f, boxes in load_gt(seq, data_cfg).items()}
    if source == "ab3dmot":
        det_dir = Path(ab3dmot_det_dir) if ab3dmot_det_dir is not None else data_cfg.ab3dmot_det_dir
        det_path = det_dir / f"{seq}.txt"
        if not det_path.exists():
            raise FileNotFoundError(f"AB3DMOT Car detections not found at {det_path}")
        frames = parse_ab3dmot_detections(det_path)
        if min_score is not None:
            frames = {f: kept for f, boxes in frames.items()
                      if (kept := [b for b in boxes if b.score >= min_score])}
        return frames
    raise ValueError(f"unknown detection source {source!r}; expected 'ab3dmot' or 'gt'")


def snapshot_tracks(tracks) -> list[tuple[int, Box3D]]:
    """Freeze (id, box) for this frame. Tracks and their filters mutate in place, so the box has
    to be copied out now; holding the track object would report its final state instead."""
    return [(t.id, t.box()) for t in tracks]


def run_sequence(seq: str, source: str, trk_cfg: KittiTrackerConfig,
                 data_cfg: KittiTrackingConfig | None = None,
                 min_score: float | None = MIN_DETECTION_SCORE):
    """Drive the tracker frame by frame. Returns (gt_frames, hyp_frames, counts)."""
    data_cfg = data_cfg or KittiTrackingConfig()
    gt = load_gt(seq, data_cfg)
    dets = load_detection_frames(seq, source, data_cfg=data_cfg, min_score=min_score)
    tracker = KittiTracker(trk_cfg)
    n_frames = max(max(gt, default=0), max(dets, default=0)) + 1
    gt_frames, hyp_frames = [], []
    for f in range(n_frames):
        confirmed = tracker.step(dets.get(f, []))
        hyp_frames.append(snapshot_tracks(confirmed))
        gt_frames.append(gt.get(f, []))
    counts = {
        "frames": n_frames,
        "gt_boxes": sum(len(v) for v in gt.values()),
        "det_boxes": sum(len(v) for v in dets.values()),
        "hyp_boxes": sum(len(v) for v in hyp_frames),
    }
    return gt_frames, hyp_frames, counts


def pooled_metrics(accumulators, names):
    """Per-sequence rows plus an OVERALL row computed from the pooled events.

    OVERALL is not the mean of the per-sequence MOTAs: MOTA is a ratio of pooled error counts to
    pooled object counts, so a short sequence must not weigh as much as a long one.
    """
    mh = mm.metrics.create()
    return mh.compute_many(list(accumulators), names=list(names), metrics=METRICS,
                           generate_overall=True)


def run_configuration(source: str, cost: str, sequences, data_cfg=None,
                      min_score: float | None = MIN_DETECTION_SCORE):
    """One (detection source, association cost) cell of the grid over the given sequences."""
    trk_cfg = KittiTrackerConfig(cost=cost)
    accs, names, rows, failures = [], [], [], []
    for seq in sequences:
        t0 = time.perf_counter()
        try:
            gt_frames, hyp_frames, counts = run_sequence(seq, source, trk_cfg, data_cfg, min_score)
            acc = accumulate(gt_frames, hyp_frames, IOU_THRESH)
        except Exception as exc:                                # keep going, name it later
            failures.append((seq, f"{type(exc).__name__}: {exc}"))
            print(f"  {seq}: FAILED {type(exc).__name__}: {exc}")
            continue
        elapsed = time.perf_counter() - t0
        accs.append(acc)
        names.append(seq)
        rows.append({"sequence": seq, "seconds": elapsed, **counts})
        print(f"  {seq}: {counts['frames']:4d} frames  gt={counts['gt_boxes']:5d} "
              f"det={counts['det_boxes']:5d}  {elapsed:6.1f}s")
    summary = pooled_metrics(accs, names) if accs else None
    return summary, rows, failures


def _metric_fields(summary, key):
    row = summary.loc[key]
    objects = int(row["num_objects"])
    matched = int(row["num_detections"])
    return {
        "mota": float(row["mota"]),
        # motmetrics MOTP is a mean distance (1 - IoU), so lower is better; motp_iou is the same
        # quantity as the mean 3D IoU that KITTI/AB3DMOT tables report, where higher is better.
        "motp_dist": float(row["motp"]),
        "motp_iou": 1.0 - float(row["motp"]),
        "idf1": float(row["idf1"]),
        "id_switches": int(row["num_switches"]),
        "gt_objects": objects,
        "matched": matched,
        "false_positives": int(row["num_false_positives"]),
        "misses": int(row["num_misses"]),
        "recall": matched / objects if objects else 0.0,
    }


_METRIC_COLS = ["mota", "motp_dist", "motp_iou", "idf1", "id_switches", "gt_objects", "matched",
                "false_positives", "misses", "recall"]
FIELDS = ["subset", "detections", "association", "sequence", "frames", "gt_boxes", "det_boxes",
          "hyp_boxes", *_METRIC_COLS, "seconds"]
SUMMARY_FIELDS = ["subset", "detections", "association", "sequences", "frames", "gt_boxes",
                  "det_boxes", "hyp_boxes", *_METRIC_COLS, "seconds"]


def _write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()})
    print(f"wrote {path}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sequences", nargs="*", default=None, help="override the val split")
    ap.add_argument("--sources", nargs="*", default=["ab3dmot", "gt"])
    ap.add_argument("--ablation-sequences", nargs="*", default=["0001", "0008", "0018"],
                    help="sequences for the association-cost ablation, the three densest in Car "
                         "ground truth ('none' to skip)")
    ap.add_argument("--score-sweep", nargs="*", type=float, default=[2.0, 4.0, 6.0],
                    help="diagnostic detection-score cutoffs; the headline run stays ungated")
    args = ap.parse_args(argv)

    sequences = args.sequences or AB3DMOT_VAL_SPLIT
    data_cfg = KittiTrackingConfig()
    print(f"IoU threshold {IOU_THRESH} | detection score gate: none | non-Car rows dropped at parse")

    per_seq_rows, summary_rows, all_failures = [], [], []
    label = f"val{len(sequences)}"
    cells = [(label, s, "iou", sequences, MIN_DETECTION_SCORE) for s in args.sources]
    ablation = [] if args.ablation_sequences == ["none"] else args.ablation_sequences
    if ablation:
        # Both costs on the same subset: the pooled IoU number over 11 sequences is not a
        # like-for-like baseline for a Mahalanobis run over 3.
        sub = f"ablation{len(ablation)}"
        cells += [(sub, "ab3dmot", "iou", ablation, MIN_DETECTION_SCORE),
                  (sub, "ab3dmot", "maha", ablation, MIN_DETECTION_SCORE)]
    # Diagnostic only: shows how much of the headline is the unfiltered detection tail.
    # The headline stays ungated; these rows are labelled so they cannot be mistaken for it.
    for thr in args.score_sweep:
        cells.append((f"sensitivity_score>={thr:g}", "ab3dmot", "iou", sequences, thr))

    for subset, source, cost, seqs, min_score in cells:
        print(f"\n=== {subset} detections={source} association={cost} ({len(seqs)} sequences) ===")
        t0 = time.perf_counter()
        summary, rows, failures = run_configuration(source, cost, seqs, data_cfg, min_score)
        wall = time.perf_counter() - t0
        all_failures += [(source, cost, seq, err) for seq, err in failures]
        if summary is None:
            print("  no sequence succeeded; nothing to aggregate")
            continue
        by_seq = {r["sequence"]: r for r in rows}
        for seq in by_seq:
            per_seq_rows.append({"subset": subset, "detections": source, "association": cost,
                                 **by_seq[seq], **_metric_fields(summary, seq)})
        overall = {"subset": subset, "detections": source, "association": cost,
                   "sequences": len(rows),
                   "frames": sum(r["frames"] for r in rows),
                   "gt_boxes": sum(r["gt_boxes"] for r in rows),
                   "det_boxes": sum(r["det_boxes"] for r in rows),
                   "hyp_boxes": sum(r["hyp_boxes"] for r in rows),
                   "seconds": wall, **_metric_fields(summary, "OVERALL")}
        summary_rows.append(overall)
        print(summary.to_string())
        print(f"  OVERALL MOTA={overall['mota']:.4f} mean-IoU={overall['motp_iou']:.4f} "
              f"IDF1={overall['idf1']:.4f} ID-sw={overall['id_switches']} "
              f"FP={overall['false_positives']} FN={overall['misses']} "
              f"recall={overall['recall']:.4f} [{wall:.1f}s]")

    _write_csv(PER_SEQUENCE_CSV, FIELDS, per_seq_rows)
    _write_csv(SUMMARY_CSV, SUMMARY_FIELDS, summary_rows)
    if all_failures:
        print("\nFAILED sequences:")
        for source, cost, seq, err in all_failures:
            print(f"  {source}/{cost} {seq}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
