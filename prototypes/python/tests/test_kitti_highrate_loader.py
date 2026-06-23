from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from kitti_highrate_loader import (
    HighRateOxtsConfig,
    HighRateOxtsSetupError,
    cache_path_for,
    expected_extract_drive_path,
    expected_sync_drive_path,
    load_cache,
    load_highrate_oxts,
    require_highrate_oxts,
    save_cache,
)


_OXTS_FIELD_NAMES = (
    "lat",
    "lon",
    "alt",
    "roll",
    "pitch",
    "yaw",
    "vn",
    "ve",
    "vf",
    "vl",
    "vu",
    "ax",
    "ay",
    "az",
    "af",
    "al",
    "au",
    "wx",
    "wy",
    "wz",
    "wf",
    "wl",
    "wu",
    "pos_accuracy",
    "vel_accuracy",
    "navstat",
    "numsats",
    "posmode",
    "velmode",
    "orimode",
)


def _packet_values(**overrides: float) -> list[float]:
    values = {
        "lat": 49.0,
        "lon": 8.0,
        "alt": 100.0,
        "roll": 0.01,
        "pitch": -0.02,
        "yaw": 0.3,
        "vn": 0.0,
        "ve": 0.0,
        "vf": 5.0,
        "vl": 0.1,
        "vu": -0.2,
        "ax": 0.0,
        "ay": 0.0,
        "az": 0.0,
        "af": 0.5,
        "al": -0.1,
        "au": 9.7,
        "wx": 0.0,
        "wy": 0.0,
        "wz": 0.0,
        "wf": 0.01,
        "wl": -0.02,
        "wu": 0.03,
        "pos_accuracy": 0.5,
        "vel_accuracy": 0.1,
        "navstat": 4.0,
        "numsats": 10.0,
        "posmode": 4.0,
        "velmode": 4.0,
        "orimode": 4.0,
    }
    values.update(overrides)
    return [values[name] for name in _OXTS_FIELD_NAMES]


def _write_oxts_files(oxts_path: Path) -> None:
    (oxts_path / "data").mkdir(parents=True)
    (oxts_path / "timestamps.txt").write_text("2011-09-26 13:02:25.100000000\n", encoding="utf-8")
    (oxts_path / "data" / "0000000000.txt").write_text("0 0 0\n", encoding="utf-8")


def _write_extract_oxts(config: HighRateOxtsConfig, packets: list[list[float]] | None = None) -> Path:
    if packets is None:
        packets = [
            _packet_values(yaw=0.0, vf=1.0, vl=0.1, vu=-0.2),
            _packet_values(yaw=0.01, vf=1.1, vl=0.2, vu=-0.3),
            _packet_values(yaw=0.02, vf=1.2, vl=0.3, vu=-0.4),
        ]

    oxts_path = expected_extract_drive_path(config) / "oxts"
    data_path = oxts_path / "data"
    data_path.mkdir(parents=True)
    (oxts_path / "timestamps.txt").write_text(
        "\n".join(
            [
                "2011-09-26 13:02:25.000000000",
                "2011-09-26 13:02:25.010000000",
                "2011-09-26 13:02:25.020000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    for idx, packet in enumerate(packets):
        (data_path / f"{idx:010d}.txt").write_text(
            " ".join(str(value) for value in packet) + "\n",
            encoding="utf-8",
        )
    return oxts_path


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


def test_load_highrate_oxts_parses_extract_packets_and_enu(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="1",
        cache_root=tmp_path / "cache",
    )
    extract_oxts_path = _write_extract_oxts(config)

    sequence = load_highrate_oxts(config)

    assert sequence.date == "2011_09_26"
    assert sequence.drive == "0001"
    assert sequence.source_path == str(extract_oxts_path)
    assert sequence.sample_count == 3
    assert sequence.duration_s == pytest.approx(0.02)
    np.testing.assert_allclose(sequence.timestamps, np.array([0.0, 0.01, 0.02]))
    np.testing.assert_allclose(sequence.lat_lon_alt[0], np.array([49.0, 8.0, 100.0]))
    np.testing.assert_allclose(sequence.roll_pitch_yaw[:, 2], np.array([0.0, 0.01, 0.02]))
    np.testing.assert_allclose(
        sequence.velocity,
        np.array(
            [
                [1.0, 0.1, -0.2],
                [1.1, 0.2, -0.3],
                [1.2, 0.3, -0.4],
            ]
        ),
    )
    np.testing.assert_allclose(sequence.accel_body[0], np.array([0.5, -0.1, 9.7]))
    np.testing.assert_allclose(sequence.gyro_body[0], np.array([0.01, -0.02, 0.03]))
    np.testing.assert_allclose(sequence.origin_lat_lon_alt, np.array([49.0, 8.0, 100.0]))
    np.testing.assert_allclose(sequence.enu_position_m[0], np.zeros(3), rtol=0.0, atol=1e-9)
    assert cache_path_for(config).is_file()


def test_highrate_oxts_cache_round_trip(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="1",
        cache_root=tmp_path / "cache",
    )
    _write_extract_oxts(config)
    sequence = load_highrate_oxts(config)

    loaded = load_cache(cache_path_for(config))

    assert loaded.date == sequence.date
    assert loaded.drive == sequence.drive
    assert loaded.source_path == sequence.source_path
    assert loaded.cache_version == sequence.cache_version
    assert loaded.source_kind == sequence.source_kind
    np.testing.assert_allclose(loaded.timestamps, sequence.timestamps)
    np.testing.assert_allclose(loaded.lat_lon_alt, sequence.lat_lon_alt)
    np.testing.assert_allclose(loaded.enu_position_m, sequence.enu_position_m)
    np.testing.assert_allclose(loaded.roll_pitch_yaw, sequence.roll_pitch_yaw)
    np.testing.assert_allclose(loaded.velocity, sequence.velocity)
    np.testing.assert_allclose(loaded.accel_body, sequence.accel_body)
    np.testing.assert_allclose(loaded.gyro_body, sequence.gyro_body)
    np.testing.assert_allclose(loaded.origin_lat_lon_alt, sequence.origin_lat_lon_alt)


def test_load_highrate_oxts_rebuilds_cache_when_source_path_mismatches(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="1",
        cache_root=tmp_path / "cache",
    )
    extract_oxts_path = _write_extract_oxts(config)
    extract_sequence = load_highrate_oxts(config)
    stale_sequence = replace(
        extract_sequence,
        source_path=str(tmp_path / "stale_root" / "2011_09_26_drive_0001_extract" / "oxts"),
        velocity=np.full_like(extract_sequence.velocity, 99.0),
    )
    save_cache(stale_sequence, cache_path_for(config))

    sequence = load_highrate_oxts(config)

    assert sequence.source_path == str(extract_oxts_path)
    np.testing.assert_allclose(sequence.velocity, extract_sequence.velocity)

    refreshed_cache = load_cache(cache_path_for(config))
    assert refreshed_cache.source_path == str(extract_oxts_path)
    np.testing.assert_allclose(refreshed_cache.velocity, extract_sequence.velocity)


def test_load_highrate_oxts_rejects_timestamp_packet_count_mismatch(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    _write_extract_oxts(config, packets=[_packet_values(), _packet_values()])

    with pytest.raises(HighRateOxtsSetupError, match="timestamp count"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_non_monotonic_timestamps(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    oxts_path = _write_extract_oxts(config)
    (oxts_path / "timestamps.txt").write_text(
        "\n".join(
            [
                "2011-09-26 13:02:25.000000000",
                "2011-09-26 13:02:25.020000000",
                "2011-09-26 13:02:25.010000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HighRateOxtsSetupError, match="strictly increasing"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_malformed_timestamp_with_line_number(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    oxts_path = _write_extract_oxts(config)
    (oxts_path / "timestamps.txt").write_text(
        "\n".join(
            [
                "2011-09-26 13:02:25.000000000",
                "2011-09-26 13:02:25.010000abc",
                "2011-09-26 13:02:25.020000000",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(HighRateOxtsSetupError, match=r"Malformed.*timestamps\.txt:2"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_malformed_packet_values(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    oxts_path = _write_extract_oxts(config)
    bad_packet = oxts_path / "data" / "0000000001.txt"
    bad_packet.write_text("not-a-number\n", encoding="utf-8")

    with pytest.raises(HighRateOxtsSetupError, match=r"0000000001\.txt.*numeric"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_extra_packet_values(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    _write_extract_oxts(
        config,
        packets=[
            _packet_values(),
            _packet_values() + [99.0],
            _packet_values(),
        ],
    )

    with pytest.raises(HighRateOxtsSetupError, match="expected exactly|Malformed"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_non_finite_packet_values(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    _write_extract_oxts(
        config,
        packets=[
            _packet_values(),
            _packet_values(lat=float("nan")),
            _packet_values(),
        ],
    )

    with pytest.raises(HighRateOxtsSetupError, match="finite"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_metadata_valid_cache_with_bad_shapes(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    cache_path = cache_path_for(config)
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(
        cache_path,
        timestamps=np.array([0.0, 0.01]),
        lat_lon_alt=np.zeros((1, 3)),
        enu_position_m=np.zeros((2, 3)),
        roll_pitch_yaw=np.zeros((2, 3)),
        velocity=np.zeros((2, 3)),
        accel_body=np.zeros((2, 3)),
        gyro_body=np.zeros((2, 3)),
        origin_lat_lon_alt=np.zeros(3),
        date=np.array(config.date),
        drive=np.array(config.normalized_drive()),
        source_path=np.array(str(expected_extract_drive_path(config) / "oxts")),
        cache_version=np.array(1, dtype=np.int64),
        source_kind=np.array("extract"),
    )

    with pytest.raises(HighRateOxtsSetupError, match="Cache integrity|shape"):
        load_highrate_oxts(config)


def test_load_highrate_oxts_rejects_stale_cache_metadata(tmp_path: Path) -> None:
    config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    wrong_config = HighRateOxtsConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0002",
        cache_root=tmp_path / "cache",
    )
    _write_extract_oxts(config)
    sequence = load_highrate_oxts(config)
    save_cache(sequence, cache_path_for(wrong_config))

    with pytest.raises(HighRateOxtsSetupError, match="Cache metadata mismatch"):
        load_highrate_oxts(wrong_config)
