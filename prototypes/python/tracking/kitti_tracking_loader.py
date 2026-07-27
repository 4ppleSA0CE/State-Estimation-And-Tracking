# prototypes/python/tracking/kitti_tracking_loader.py
"""Load KITTI Tracking Car boxes: GT from label_02, detections from AB3DMOT files, val split.
Box-level only (no images/velodyne). Raises a clear error when the dataset is absent."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kitti_boxes import Box3D

# AB3DMOT KITTI val split (training sequences held out for validation).
# PIN/VERIFY against AB3DMOT's repo (scripts/KITTI split) before trusting for the headline number.
VAL_SEQUENCES = ["0001", "0006", "0008", "0010", "0012", "0013", "0014", "0015", "0016", "0018", "0019"]

CAR = "Car"


def _car_box(cols, with_score: bool) -> Box3D:
    # cols indices per label_02: 2=type,10=h,11=w,12=l,13=x,14=y,15=z,16=ry,[17=score]
    return Box3D(
        x=float(cols[13]), y=float(cols[14]), z=float(cols[15]),
        yaw=float(cols[16]),
        l=float(cols[12]), w=float(cols[11]), h=float(cols[10]),
        score=float(cols[17]) if with_score and len(cols) > 17 else 1.0,
        track_id=int(cols[1]),
    )


def parse_label_file(path) -> dict[int, list[Box3D]]:
    """label_02/{seq}.txt -> {frame: [Box3D(Car), ...]} (DontCare/other types dropped)."""
    frames: dict[int, list[Box3D]] = {}
    with open(path) as f:
        for line in f:
            cols = line.split()
            if len(cols) < 17 or cols[2] != CAR:
                continue
            frame = int(cols[0])
            frames.setdefault(frame, []).append(_car_box(cols, with_score=True))
    return frames


@dataclass
class KittiTrackingConfig:
    root: Path = Path("data/kitti_tracking")
    detection_source: str = "ab3dmot"     # "ab3dmot" | "gt"
    ab3dmot_det_dir: Path = Path("data/kitti_tracking/ab3dmot_car_val")
    min_score: float = 0.0                 # PIN: detection score threshold for the headline run
                                           # (0.0 = NO filtering yet; the comparable MOTA needs this pinned)

    @property
    def label_dir(self) -> Path:
        return self.root / "training" / "label_02"


def require_kitti_tracking(cfg: KittiTrackingConfig) -> None:
    if not cfg.label_dir.exists():
        raise FileNotFoundError(
            f"KITTI Tracking labels not found at {cfg.label_dir}. Download the KITTI Tracking "
            f"training label_02 (+ calib) and AB3DMOT Car detections; see the Stage 5A spec."
        )


def load_gt(seq: str, cfg: KittiTrackingConfig | None = None) -> dict[int, list[Box3D]]:
    cfg = cfg or KittiTrackingConfig()
    require_kitti_tracking(cfg)
    return parse_label_file(cfg.label_dir / f"{seq}.txt")


def load_detections(seq: str, cfg: KittiTrackingConfig | None = None) -> dict[int, list[Box3D]]:
    """Detections per frame. source='gt' returns label boxes (score=1); 'ab3dmot' parses their
    detection files. NOTE: pin the AB3DMOT detection column layout from their repo before relying
    on the headline number; parse_label_file's column map matches label_02, adjust if theirs differs."""
    cfg = cfg or KittiTrackingConfig()
    if cfg.detection_source == "gt":
        return {f: [Box3D(b.x, b.y, b.z, b.yaw, b.l, b.w, b.h, 1.0, -1) for b in boxes]
                for f, boxes in load_gt(seq, cfg).items()}
    det_path = cfg.ab3dmot_det_dir / f"{seq}.txt"
    if not det_path.exists():
        raise FileNotFoundError(f"AB3DMOT Car detections not found at {det_path}. See Stage 5A spec.")
    frames = parse_label_file(det_path)   # adjust column map here if AB3DMOT format differs
    if cfg.min_score > 0.0:               # PIN: no filtering by default; set min_score for the headline run
        frames = {f: [b for b in boxes if b.score >= cfg.min_score] for f, boxes in frames.items()}
    return frames


def val_sequences() -> list[str]:
    return list(VAL_SEQUENCES)
