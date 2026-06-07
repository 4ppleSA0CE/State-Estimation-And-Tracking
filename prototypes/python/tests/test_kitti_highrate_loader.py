from __future__ import annotations

from pathlib import Path

import pytest

from kitti_highrate_loader import (
    HighRateOxtsConfig,
    HighRateOxtsSetupError,
    expected_extract_drive_path,
    expected_sync_drive_path,
    require_highrate_oxts,
)


def _write_oxts_files(oxts_path: Path) -> None:
    (oxts_path / "data").mkdir(parents=True)
    (oxts_path / "timestamps.txt").write_text("2011-09-26 13:02:25.100000000\n", encoding="utf-8")
    (oxts_path / "data" / "0000000000.txt").write_text("0 0 0\n", encoding="utf-8")


def test_expected_drive_paths_are_date_level_under_kitti_root(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="1")

    assert expected_extract_drive_path(config) == (
        tmp_path / "kitti_raw" / "2011_09_26" / "2011_09_26_drive_0001_extract"
    )
    assert expected_sync_drive_path(config) == (
        tmp_path / "kitti_raw" / "2011_09_26" / "2011_09_26_drive_0001_sync"
    )


def test_require_highrate_oxts_reports_sync_only_dataset(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="0001")
    sync_drive_path = expected_sync_drive_path(config)
    _write_oxts_files(sync_drive_path / "oxts")

    with pytest.raises(HighRateOxtsSetupError) as excinfo:
        require_highrate_oxts(config)

    message = str(excinfo.value)
    assert "High-rate KITTI Raw OXTS data is required" in message
    assert str(expected_extract_drive_path(config)) in message
    assert "synced KITTI data exists" in message


def test_require_highrate_oxts_returns_oxts_path_when_extract_data_exists(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="0001")
    extract_oxts_path = expected_extract_drive_path(config) / "oxts"
    _write_oxts_files(extract_oxts_path)

    assert require_highrate_oxts(config) == extract_oxts_path


def test_require_highrate_oxts_reports_missing_root(tmp_path: Path) -> None:
    missing_root = tmp_path / "does_not_exist"
    config = HighRateOxtsConfig(root=missing_root)

    with pytest.raises(HighRateOxtsSetupError) as excinfo:
        require_highrate_oxts(config)

    message = str(excinfo.value)
    assert "KITTI Raw root does not exist" in message
    assert str(missing_root) in message


def test_require_highrate_oxts_reports_missing_extract_layout(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="0001")
    (config.root / config.date).mkdir(parents=True)

    with pytest.raises(HighRateOxtsSetupError) as excinfo:
        require_highrate_oxts(config)

    message = str(excinfo.value)
    assert str(expected_extract_drive_path(config) / "oxts") in message
    assert "timestamps.txt" in message
    assert "data/*.txt" in message


def test_malformed_extract_directory_entries_do_not_count_as_oxts_files(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="0001")
    extract_oxts_path = expected_extract_drive_path(config) / "oxts"
    (extract_oxts_path / "timestamps.txt").mkdir(parents=True)
    (extract_oxts_path / "data" / "0000000000.txt").mkdir(parents=True)

    with pytest.raises(HighRateOxtsSetupError):
        require_highrate_oxts(config)


def test_local_workspace_currently_blocks_highrate_if_only_sync_is_present() -> None:
    config = HighRateOxtsConfig()
    sync_oxts_path = expected_sync_drive_path(config) / "oxts"

    if expected_extract_drive_path(config).exists():
        pytest.skip("Local high-rate extract data is present.")
    if not (
        (sync_oxts_path / "timestamps.txt").is_file()
        and (sync_oxts_path / "data").is_dir()
        and any(entry.is_file() for entry in (sync_oxts_path / "data").glob("*.txt"))
    ):
        pytest.skip("Local synced KITTI OXTS data is not present.")

    with pytest.raises(HighRateOxtsSetupError, match="synced KITTI data exists"):
        require_highrate_oxts(config)
