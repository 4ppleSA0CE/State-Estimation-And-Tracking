from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zipfile import BadZipFile

import numpy as np
import pytest

import kitti_loader
from kitti_loader import (
    CACHE_VERSION,
    KittiLoaderConfig,
    KittiSequence,
    KittiSetupError,
    _build_sequence_with_pykitti,
    cache_path_for,
    enu_from_wgs84,
    load_cache,
    load_kitti_sequence,
    save_cache,
)


def _synthetic_sequence(tmp_path: Path) -> KittiSequence:
    return KittiSequence(
        timestamps=np.array([0.0, 0.5, 1.0]),
        lat_lon_alt=np.array(
            [
                [49.0, 8.0, 110.0],
                [49.00001, 8.00002, 111.0],
                [49.00002, 8.00004, 112.0],
            ]
        ),
        enu_position_m=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.5, 1.0, 1.0],
                [3.0, 2.0, 2.0],
            ]
        ),
        roll_pitch_yaw=np.zeros((3, 3)),
        velocity=np.zeros((3, 3)),
        accel_body=np.zeros((3, 3)),
        gyro_body=np.zeros((3, 3)),
        gps_covariance=np.repeat(np.eye(3)[None, :, :], 3, axis=0),
        origin_lat_lon_alt=np.array([49.0, 8.0, 110.0]),
        date="2011_09_26",
        drive="0001",
        source_path=str(tmp_path / "raw" / "2011_09_26"),
        cache_version=CACHE_VERSION,
    )


def test_format_sequence_summary_contains_key_sequence_metadata(tmp_path: Path) -> None:
    sequence = _synthetic_sequence(tmp_path)
    cache_path = tmp_path / "cache" / "sequence.npz"

    summary = kitti_loader.format_sequence_summary(sequence, cache_path)

    assert "2011_09_26" in summary
    assert "0001" in summary
    assert "samples: 3" in summary
    assert "duration_s: 1.000" in summary
    assert "cache:" in summary


def test_plot_enu_trajectory_writes_nonempty_png(tmp_path: Path) -> None:
    sequence = _synthetic_sequence(tmp_path)
    output_path = tmp_path / "plots" / "trajectory.png"

    kitti_loader.plot_enu_trajectory(sequence, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert output_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_plot_enu_trajectory_closes_figure_when_save_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import matplotlib.figure
    import matplotlib.pyplot as plt

    sequence = _synthetic_sequence(tmp_path)
    output_path = tmp_path / "plots" / "trajectory.png"

    def fail_savefig(self: matplotlib.figure.Figure, *args: object, **kwargs: object) -> None:
        del self, args, kwargs
        raise RuntimeError("save failed")

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", fail_savefig)

    with pytest.raises(RuntimeError, match="save failed"):
        kitti_loader.plot_enu_trajectory(sequence, output_path)

    assert plt.get_fignums() == []


def test_enu_from_wgs84_maps_origin_to_zero() -> None:
    origin_lat = 49.0123
    origin_lon = 8.4012
    origin_alt = 113.4

    enu = enu_from_wgs84(
        np.array([origin_lat]),
        np.array([origin_lon]),
        np.array([origin_alt]),
        np.array([origin_lat, origin_lon, origin_alt]),
    )

    np.testing.assert_allclose(enu, np.zeros((1, 3)), rtol=0.0, atol=1e-6)


def test_enu_from_wgs84_small_offsets_have_expected_direction_and_magnitude() -> None:
    origin_lat = 49.0123
    origin_lon = 8.4012
    origin_alt = 113.4
    earth_radius_m = 6_378_137.0

    north_m = 10.0
    east_m = 5.0
    up_m = 2.0
    dlat_deg = np.rad2deg(north_m / earth_radius_m)
    dlon_deg = np.rad2deg(east_m / (earth_radius_m * np.cos(np.deg2rad(origin_lat))))

    enu = enu_from_wgs84(
        np.array([origin_lat + dlat_deg, origin_lat, origin_lat]),
        np.array([origin_lon, origin_lon + dlon_deg, origin_lon]),
        np.array([origin_alt, origin_alt, origin_alt + up_m]),
        np.array([origin_lat, origin_lon, origin_alt]),
    )

    expected = np.array(
        [
            [0.0, north_m, 0.0],
            [east_m, 0.0, 0.0],
            [0.0, 0.0, up_m],
        ]
    )
    np.testing.assert_allclose(enu, expected, rtol=0.0, atol=0.05)


def test_cache_path_for_is_deterministic_and_includes_cache_version(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )

    first = cache_path_for(cfg)
    second = cache_path_for(cfg)

    assert first == second
    assert str(CACHE_VERSION) in str(first)
    assert first.parent == tmp_path / "cache"


def test_save_cache_and_load_cache_round_trip_sequence_arrays_and_metadata(tmp_path: Path) -> None:
    sequence = KittiSequence(
        timestamps=np.array([0.0, 0.1, 0.2]),
        lat_lon_alt=np.array(
            [
                [49.0123, 8.4012, 113.4],
                [49.0124, 8.4013, 113.5],
                [49.0125, 8.4014, 113.6],
            ]
        ),
        enu_position_m=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 2.0, 0.1],
                [2.5, 4.0, 0.2],
            ]
        ),
        roll_pitch_yaw=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.01],
                [0.0, 0.0, 0.02],
            ]
        ),
        velocity=np.array(
            [
                [10.0, 0.0, 0.0],
                [10.1, 0.2, 0.0],
                [10.2, 0.4, 0.1],
            ]
        ),
        accel_body=np.array(
            [
                [0.1, 0.0, 9.8],
                [0.1, 0.1, 9.8],
                [0.2, 0.1, 9.7],
            ]
        ),
        gyro_body=np.array(
            [
                [0.0, 0.0, 0.001],
                [0.0, 0.0, 0.002],
                [0.0, 0.0, 0.003],
            ]
        ),
        gps_covariance=np.repeat(np.eye(3)[None, :, :] * 1.5**2, 3, axis=0),
        origin_lat_lon_alt=np.array([49.0123, 8.4012, 113.4]),
        date="2011_09_26",
        drive="0001",
        source_path=str(tmp_path / "raw" / "2011_09_26"),
        cache_version=CACHE_VERSION,
    )
    cache_path = tmp_path / "sequence_cache.npz"

    save_cache(sequence, cache_path)
    loaded = load_cache(cache_path)

    np.testing.assert_allclose(loaded.timestamps, sequence.timestamps, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.lat_lon_alt, sequence.lat_lon_alt, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.enu_position_m, sequence.enu_position_m, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.roll_pitch_yaw, sequence.roll_pitch_yaw, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.velocity, sequence.velocity, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.accel_body, sequence.accel_body, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.gyro_body, sequence.gyro_body, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(loaded.gps_covariance, sequence.gps_covariance, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(
        loaded.origin_lat_lon_alt,
        sequence.origin_lat_lon_alt,
        rtol=0.0,
        atol=0.0,
    )
    assert loaded.date == sequence.date
    assert loaded.drive == sequence.drive
    assert loaded.source_path == sequence.source_path
    assert loaded.cache_version == sequence.cache_version


def test_load_kitti_sequence_raises_setup_error_when_raw_root_is_missing(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "missing_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )

    with pytest.raises(KittiSetupError, match="KITTI Raw root does not exist"):
        load_kitti_sequence(cfg)


def test_load_kitti_sequence_raises_setup_error_when_drive_folder_is_missing(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    (cfg.root / cfg.date).mkdir(parents=True)

    with pytest.raises(KittiSetupError, match="KITTI Raw sequence does not exist") as exc_info:
        load_kitti_sequence(cfg)

    assert "2011_09_26_drive_0001_sync" in str(exc_info.value)


def test_load_kitti_sequence_raises_setup_error_when_oxts_files_are_missing(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    drive_path = cfg.root / cfg.date / "2011_09_26_drive_0001_sync"
    drive_path.mkdir(parents=True)

    with pytest.raises(KittiSetupError, match="KITTI Raw OXTS files are missing") as exc_info:
        load_kitti_sequence(cfg)

    assert "oxts/timestamps.txt" in str(exc_info.value)


def test_load_kitti_sequence_raises_setup_error_when_oxts_data_is_empty(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    oxts_path = cfg.root / cfg.date / "2011_09_26_drive_0001_sync" / "oxts"
    (oxts_path / "data").mkdir(parents=True)
    (oxts_path / "timestamps.txt").write_text("", encoding="utf-8")

    with pytest.raises(KittiSetupError, match="KITTI Raw OXTS files are missing") as exc_info:
        load_kitti_sequence(cfg)

    assert "oxts/data/*.txt" in str(exc_info.value)


def test_load_kitti_sequence_raises_setup_error_for_malformed_cache(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "missing_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    cache_path = cache_path_for(cfg)
    cache_path.parent.mkdir(parents=True)
    np.savez_compressed(cache_path, timestamps=np.array([0.0]))

    with pytest.raises(KittiSetupError) as exc_info:
        load_kitti_sequence(cfg)

    message = str(exc_info.value)
    assert "Cache could not be loaded" in message
    assert str(cache_path) in message
    assert "KITTI Raw root does not exist" not in message


def test_load_kitti_sequence_raises_setup_error_for_corrupt_cache_container(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "missing_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    cache_path = cache_path_for(cfg)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"not a valid npz")

    with pytest.raises(KittiSetupError) as exc_info:
        load_kitti_sequence(cfg)

    message = str(exc_info.value)
    assert "Cache could not be loaded" in message
    assert str(cache_path) in message
    assert "KITTI Raw root does not exist" not in message


def test_load_kitti_sequence_raises_setup_error_for_corrupt_npz_container(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "missing_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    cache_path = cache_path_for(cfg)
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"PK\x03\x04truncated")

    with pytest.raises(KittiSetupError) as exc_info:
        load_kitti_sequence(cfg)

    message = str(exc_info.value)
    assert "Cache could not be loaded" in message
    assert str(cache_path) in message
    assert isinstance(exc_info.value.__cause__, BadZipFile)


def test_load_kitti_sequence_reports_stale_cache_metadata_when_source_is_absent(
    tmp_path: Path,
) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "missing_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    stale_sequence = KittiSequence(
        timestamps=np.array([0.0]),
        lat_lon_alt=np.array([[49.0, 8.0, 110.0]]),
        enu_position_m=np.zeros((1, 3)),
        roll_pitch_yaw=np.zeros((1, 3)),
        velocity=np.zeros((1, 3)),
        accel_body=np.zeros((1, 3)),
        gyro_body=np.zeros((1, 3)),
        gps_covariance=np.repeat(np.eye(3)[None, :, :], 1, axis=0),
        origin_lat_lon_alt=np.array([49.0, 8.0, 110.0]),
        date="2011_09_27",
        drive="0001",
        source_path=str(tmp_path / "raw" / "2011_09_27"),
        cache_version=CACHE_VERSION,
    )
    cache_path = cache_path_for(cfg)
    save_cache(stale_sequence, cache_path)

    with pytest.raises(KittiSetupError) as exc_info:
        load_kitti_sequence(cfg)

    message = str(exc_info.value)
    assert "Cache metadata mismatch" in message
    assert str(cache_path) in message
    assert "KITTI Raw root does not exist" not in message


def test_load_kitti_sequence_reports_stale_cache_metadata_when_source_exists(
    tmp_path: Path,
) -> None:
    cfg = KittiLoaderConfig(
        root=tmp_path / "kitti_raw",
        date="2011_09_26",
        drive="0001",
        cache_root=tmp_path / "cache",
    )
    (cfg.root / cfg.date).mkdir(parents=True)
    stale_sequence = KittiSequence(
        timestamps=np.array([0.0]),
        lat_lon_alt=np.array([[49.0, 8.0, 110.0]]),
        enu_position_m=np.zeros((1, 3)),
        roll_pitch_yaw=np.zeros((1, 3)),
        velocity=np.zeros((1, 3)),
        accel_body=np.zeros((1, 3)),
        gyro_body=np.zeros((1, 3)),
        gps_covariance=np.repeat(np.eye(3)[None, :, :], 1, axis=0),
        origin_lat_lon_alt=np.array([49.0, 8.0, 110.0]),
        date=cfg.date,
        drive="0002",
        source_path=str(cfg.root / cfg.date),
        cache_version=CACHE_VERSION,
    )
    cache_path = cache_path_for(cfg)
    save_cache(stale_sequence, cache_path)

    with pytest.raises(KittiSetupError) as exc_info:
        load_kitti_sequence(cfg)

    message = str(exc_info.value)
    assert "Cache metadata mismatch" in message
    assert str(cache_path) in message
    assert "pykitti is required" not in message


def _fake_pykitti(
    *,
    timestamps: list[object] | None = None,
    packets: list[SimpleNamespace] | None = None,
) -> object:
    class FakePykitti:
        @staticmethod
        def raw(root: str, date: str, drive: str) -> SimpleNamespace:
            del root, date, drive
            return SimpleNamespace(
                timestamps=(
                    timestamps
                    if timestamps is not None
                    else [datetime(2011, 9, 26) + timedelta(seconds=0.1 * idx) for idx in range(3)]
                ),
                oxts=packets if packets is not None else _fake_oxts_packets(),
            )

    return FakePykitti()


def _fake_oxts_packets() -> list[SimpleNamespace]:
    packets = []
    for idx in range(3):
        packet = SimpleNamespace(
            lat=49.0 + idx * 1e-5,
            lon=8.0,
            alt=110.0 + idx,
            roll=0.01 * idx,
            pitch=0.02 * idx,
            yaw=0.03 * idx,
            vf=1.0,
            vl=0.0,
            vu=0.0,
            af=0.1,
            al=0.2,
            au=0.3,
            wf=0.01,
            wl=0.02,
            wu=0.03,
        )
        packets.append(SimpleNamespace(packet=packet))
    return packets


def test_build_sequence_with_fake_pykitti_dataset(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="1")
    sequence = _build_sequence_with_pykitti(cfg, _fake_pykitti())

    assert sequence.timestamps.tolist() == [0.0, 0.1, 0.2]
    assert sequence.drive == "0001"
    assert sequence.lat_lon_alt.shape == (3, 3)
    assert sequence.enu_position_m.shape == (3, 3)
    assert sequence.gps_covariance.shape == (3, 3, 3)
    np.testing.assert_allclose(sequence.enu_position_m[0], np.zeros(3), atol=1e-9)


def test_build_sequence_with_fake_pykitti_rejects_invalid_timestamp(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="1")

    with pytest.raises(KittiSetupError, match="timestamp at index 1"):
        _build_sequence_with_pykitti(
            cfg,
            _fake_pykitti(
                timestamps=[
                    datetime(2011, 9, 26),
                    "bad timestamp",
                    datetime(2011, 9, 26) + timedelta(seconds=0.2),
                ]
            ),
        )


def test_build_sequence_with_fake_pykitti_rejects_missing_packet_field(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="1")
    packets = _fake_oxts_packets()
    delattr(packets[1].packet, "wu")

    with pytest.raises(KittiSetupError, match="index 1: missing field wu"):
        _build_sequence_with_pykitti(cfg, _fake_pykitti(packets=packets))


def test_build_sequence_with_fake_pykitti_rejects_non_numeric_packet_field(tmp_path: Path) -> None:
    cfg = KittiLoaderConfig(root=tmp_path / "kitti_raw", date="2011_09_26", drive="1")
    packets = _fake_oxts_packets()
    packets[2].packet.lat = "not numeric"

    with pytest.raises(KittiSetupError, match="index 2: field lat is not numeric"):
        _build_sequence_with_pykitti(cfg, _fake_pykitti(packets=packets))
