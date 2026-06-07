"""High-rate KITTI Raw OXTS setup guard for Stage 1 mechanization."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


KITTI_RAW_URL = "https://www.cvlibs.net/datasets/kitti/raw_data.php"


class HighRateOxtsSetupError(RuntimeError):
    """Raised when required high-rate KITTI Raw OXTS data is unavailable."""


@dataclass(frozen=True)
class HighRateOxtsConfig:
    root: Path = Path("data/kitti_raw")
    date: str = "2011_09_26"
    drive: str = "0001"

    def normalized_drive(self) -> str:
        return self.drive.zfill(4)


def expected_extract_drive_path(config: HighRateOxtsConfig) -> Path:
    return Path(config.root) / config.date / f"{config.date}_drive_{config.normalized_drive()}_extract"


def expected_sync_drive_path(config: HighRateOxtsConfig) -> Path:
    return Path(config.root) / config.date / f"{config.date}_drive_{config.normalized_drive()}_sync"


def _has_oxts_files(oxts_path: Path) -> bool:
    data_path = oxts_path / "data"
    return (
        (oxts_path / "timestamps.txt").is_file()
        and data_path.is_dir()
        and any(entry.is_file() for entry in data_path.glob("*.txt"))
    )


def require_highrate_oxts(config: HighRateOxtsConfig) -> Path:
    extract_drive_path = expected_extract_drive_path(config)
    extract_oxts_path = extract_drive_path / "oxts"
    if _has_oxts_files(extract_oxts_path):
        return extract_oxts_path

    sync_drive_path = expected_sync_drive_path(config)
    sync_oxts_path = sync_drive_path / "oxts"
    if _has_oxts_files(sync_oxts_path):
        raise HighRateOxtsSetupError(
            "High-rate KITTI Raw OXTS data is required for Stage 1 mechanization; "
            f"expected extract drive path: {extract_drive_path}. "
            f"Only synced KITTI data exists at {sync_drive_path}."
        )

    root = Path(config.root)
    if not root.exists():
        raise HighRateOxtsSetupError(
            f"KITTI Raw root does not exist: {root}. Download KITTI Raw data from {KITTI_RAW_URL}."
        )

    raise HighRateOxtsSetupError(
        f"Expected high-rate KITTI Raw OXTS files under {extract_oxts_path}: "
        "timestamps.txt and data/*.txt."
    )
