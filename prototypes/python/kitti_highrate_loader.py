"""High-rate KITTI Raw OXTS setup guard for Stage 1 mechanization."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from kitti_loader import enu_from_wgs84

KITTI_RAW_URL = "https://www.cvlibs.net/datasets/kitti/raw_data.php"  # download page
CACHE_VERSION = 1  # bump when cache schema changes; stale caches are rejected on load
SOURCE_KIND = "extract"  # tag distinguishing extract (high-rate) from sync (10 Hz) data

# 30 raw OXTS packet columns, in order, as they appear in each data/*.txt file.
# columns 0-2: position (lat/lon/alt), 3-5: euler angles, 6-10: velocities,
# 11-16: accelerations (world then body), 17-22: angular rates (world then body),
# 23-24: accuracy estimates, 25-29: navigation status flags.
OXTS_FIELD_NAMES = (
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


class HighRateOxtsSetupError(RuntimeError):
    """Raised when required high-rate KITTI Raw OXTS data is unavailable."""


@dataclass(frozen=True)
class HighRateOxtsConfig:
    root: Path = Path("data/kitti_raw")  # root path to KITTI raw dataset, contains date folders
    date: str = "2011_09_26"  # capture date in YYYY_MM_DD format
    drive: str = "0001"  # 4-digit drive identifier (zero-padded)
    cache_root: Path = Path("data/cache")  # directory for .npz cache files
    force_refresh: bool = False  # skip cache and reload from raw files when True

    def normalized_drive(self) -> str:
        return self.drive.zfill(4)


@dataclass(frozen=True)
class HighRateOxtsSequence:
    timestamps: np.ndarray  # (N,) time since first sample, s; ~100 Hz for extract
    lat_lon_alt: np.ndarray  # (N, 3) WGS-84 lat deg, lon deg, alt m
    enu_position_m: np.ndarray  # (N, 3) ENU position relative to origin, m
    roll_pitch_yaw: np.ndarray  # (N, 3) OXTS roll, pitch, yaw, rad
    velocity: np.ndarray  # (N, 3) body-frame forward/left/up velocity, m/s
    accel_body: np.ndarray  # (N, 3) body-frame specific force, m/s^2
    gyro_body: np.ndarray  # (N, 3) body-frame angular rate, rad/s
    origin_lat_lon_alt: np.ndarray  # (3,) WGS-84 origin for ENU conversion
    date: str  # date string YYYY_MM_DD
    drive: str  # 4-digit drive string
    source_path: str  # path to the oxts folder loaded from
    cache_version: int = CACHE_VERSION  # schema version, must match CACHE_VERSION to use
    source_kind: str = SOURCE_KIND  # "extract" or "sync", identifies the data source type

    @property
    def sample_count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration_s(self) -> float:
        if self.sample_count < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


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


def cache_path_for(config: HighRateOxtsConfig) -> Path:
    return (
        Path(config.cache_root)
        / f"kitti_raw_{config.date}_drive_{config.normalized_drive()}_extract_oxts_v{CACHE_VERSION}.npz"
    )


def _parse_timestamp(line: str, path: Path, line_number: int) -> dt.datetime:
    row = line.strip()
    if not row:
        raise HighRateOxtsSetupError(f"Empty KITTI OXTS timestamp at {path}:{line_number}.")
    try:
        whole, fractional = row.split(".", maxsplit=1)
        if not fractional.isdigit():
            raise ValueError("timestamp fractional seconds must be numeric")
        normalized = f"{whole}.{(fractional + '000000')[:6]}"
        return dt.datetime.strptime(normalized, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError as exc:
        raise HighRateOxtsSetupError(
            f"Malformed KITTI OXTS timestamp at {path}:{line_number}: {row!r}."
        ) from exc


def _timestamp_seconds(path: Path) -> np.ndarray:
    timestamps = [
        _parse_timestamp(line, path, line_number)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
    ]
    if not timestamps:
        raise HighRateOxtsSetupError(f"No KITTI OXTS timestamps found in {path}.")

    first = timestamps[0]
    seconds = np.asarray([(timestamp - first).total_seconds() for timestamp in timestamps], dtype=float)
    if np.any(np.diff(seconds) <= 0.0):
        raise HighRateOxtsSetupError(f"KITTI OXTS timestamps in {path} must be strictly increasing.")
    return seconds


def _packet_files(oxts_path: Path) -> list[Path]:
    data_path = oxts_path / "data"
    packet_paths = sorted(path for path in data_path.glob("*.txt") if path.is_file())
    if not packet_paths:
        raise HighRateOxtsSetupError(f"No KITTI OXTS packet files found under {data_path}.")
    return packet_paths


def _parse_packet(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    raw_values = text.split()
    expected_count = len(OXTS_FIELD_NAMES)
    if len(raw_values) != expected_count:
        raise HighRateOxtsSetupError(
            f"Malformed KITTI OXTS packet {path}: expected exactly "
            f"{expected_count} numeric values, found {len(raw_values)}."
        )

    values: list[float] = []
    for idx, raw_value in enumerate(raw_values):
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise HighRateOxtsSetupError(
                f"Malformed KITTI OXTS packet {path}: value {idx} is not numeric."
            ) from exc
        if not np.isfinite(value):
            raise HighRateOxtsSetupError(
                f"Malformed KITTI OXTS packet {path}: value {idx} must be finite."
            )
        values.append(value)
    return dict(zip(OXTS_FIELD_NAMES, values, strict=True))


def save_cache(sequence: HighRateOxtsSequence, path: Path | str) -> None:
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        timestamps=sequence.timestamps,
        lat_lon_alt=sequence.lat_lon_alt,
        enu_position_m=sequence.enu_position_m,
        roll_pitch_yaw=sequence.roll_pitch_yaw,
        velocity=sequence.velocity,
        accel_body=sequence.accel_body,
        gyro_body=sequence.gyro_body,
        origin_lat_lon_alt=sequence.origin_lat_lon_alt,
        date=np.array(sequence.date),
        drive=np.array(sequence.drive),
        source_path=np.array(sequence.source_path),
        cache_version=np.array(sequence.cache_version, dtype=np.int64),
        source_kind=np.array(sequence.source_kind),
    )


def load_cache(path: Path | str) -> HighRateOxtsSequence:
    with np.load(Path(path), allow_pickle=False) as data:
        return HighRateOxtsSequence(
            timestamps=data["timestamps"],
            lat_lon_alt=data["lat_lon_alt"],
            enu_position_m=data["enu_position_m"],
            roll_pitch_yaw=data["roll_pitch_yaw"],
            velocity=data["velocity"],
            accel_body=data["accel_body"],
            gyro_body=data["gyro_body"],
            origin_lat_lon_alt=data["origin_lat_lon_alt"],
            date=str(data["date"].item()),
            drive=str(data["drive"].item()),
            source_path=str(data["source_path"].item()),
            cache_version=int(data["cache_version"].item()),
            source_kind=str(data["source_kind"].item()),
        )


def _expected_cache_source_path(config: HighRateOxtsConfig) -> str:
    return str(expected_extract_drive_path(config) / "oxts")


def _validate_cache(sequence: HighRateOxtsSequence, config: HighRateOxtsConfig) -> bool:
    return (
        sequence.cache_version == CACHE_VERSION
        and sequence.source_kind == SOURCE_KIND
        and sequence.date == config.date
        and sequence.drive == config.normalized_drive()
        and sequence.source_path == _expected_cache_source_path(config)
    )


def _cache_metadata_mismatch_error(
    path: Path,
    sequence: HighRateOxtsSequence,
    config: HighRateOxtsConfig,
) -> HighRateOxtsSetupError:
    return HighRateOxtsSetupError(
        f"Cache metadata mismatch: {path}. Expected date={config.date}, "
        f"drive={config.normalized_drive()}, version={CACHE_VERSION}, "
        f"source_kind={SOURCE_KIND}, source_path={_expected_cache_source_path(config)}; "
        f"found date={sequence.date}, drive={sequence.drive}, "
        f"version={sequence.cache_version}, source_kind={sequence.source_kind}, "
        f"source_path={sequence.source_path}."
    )


def _require_finite(name: str, array: np.ndarray, context: str) -> None:
    try:
        has_only_finite_values = bool(np.all(np.isfinite(array)))
    except TypeError as exc:
        raise HighRateOxtsSetupError(
            f"{context}: {name} values must be finite numeric values."
        ) from exc
    if not has_only_finite_values:
        raise HighRateOxtsSetupError(f"{context}: {name} values must be finite.")


def _validate_sequence_integrity(sequence: HighRateOxtsSequence, context: str) -> None:
    timestamps = np.asarray(sequence.timestamps)
    if timestamps.ndim != 1:
        raise HighRateOxtsSetupError(
            f"{context}: timestamps shape must be 1D, got {timestamps.shape}."
        )
    if timestamps.shape[0] == 0:
        raise HighRateOxtsSetupError(f"{context}: timestamps must not be empty.")
    _require_finite("timestamps", timestamps, context)
    if timestamps.shape[0] > 1 and np.any(np.diff(timestamps) <= 0.0):
        raise HighRateOxtsSetupError(f"{context}: timestamps must be strictly increasing.")

    sample_count = timestamps.shape[0]
    sample_shape = (sample_count, 3)
    for name in (
        "lat_lon_alt",
        "enu_position_m",
        "roll_pitch_yaw",
        "velocity",
        "accel_body",
        "gyro_body",
    ):
        array = np.asarray(getattr(sequence, name))
        if array.shape != sample_shape:
            raise HighRateOxtsSetupError(
                f"{context}: {name} shape must be {sample_shape}, got {array.shape}."
            )
        _require_finite(name, array, context)

    origin = np.asarray(sequence.origin_lat_lon_alt)
    if origin.shape != (3,):
        raise HighRateOxtsSetupError(
            f"{context}: origin_lat_lon_alt shape must be (3,), got {origin.shape}."
        )
    _require_finite("origin_lat_lon_alt", origin, context)


def _require_shape(name: str, array: np.ndarray, shape: tuple[int, ...], config: HighRateOxtsConfig) -> None:
    if array.shape != shape:
        raise HighRateOxtsSetupError(
            f"Invalid high-rate KITTI OXTS {name} shape for {config.date} "
            f"drive {config.normalized_drive()}: expected {shape}, got {array.shape}."
        )


def _packet_array(packets: list[dict[str, float]], fields: tuple[str, str, str]) -> np.ndarray:
    return np.asarray([[packet[field] for field in fields] for packet in packets], dtype=float)


def _build_sequence_from_extract(config: HighRateOxtsConfig, oxts_path: Path) -> HighRateOxtsSequence:
    timestamps = _timestamp_seconds(oxts_path / "timestamps.txt")
    packet_paths = _packet_files(oxts_path)
    if timestamps.shape[0] != len(packet_paths):
        raise HighRateOxtsSetupError(
            f"KITTI OXTS timestamp count mismatch for {config.date} drive {config.normalized_drive()}: "
            f"{timestamps.shape[0]} timestamps vs {len(packet_paths)} packet files."
        )

    packets = [_parse_packet(path) for path in packet_paths]
    lat_lon_alt = _packet_array(packets, ("lat", "lon", "alt"))
    roll_pitch_yaw = _packet_array(packets, ("roll", "pitch", "yaw"))
    velocity = _packet_array(packets, ("vf", "vl", "vu"))
    accel_body = _packet_array(packets, ("af", "al", "au"))
    gyro_body = _packet_array(packets, ("wf", "wl", "wu"))

    sample_shape = (timestamps.shape[0], 3)
    _require_shape("lat_lon_alt", lat_lon_alt, sample_shape, config)
    _require_shape("roll_pitch_yaw", roll_pitch_yaw, sample_shape, config)
    _require_shape("velocity", velocity, sample_shape, config)
    _require_shape("accel_body", accel_body, sample_shape, config)
    _require_shape("gyro_body", gyro_body, sample_shape, config)

    origin = lat_lon_alt[0].copy()
    enu = enu_from_wgs84(
        lat_deg=lat_lon_alt[:, 0],
        lon_deg=lat_lon_alt[:, 1],
        alt_m=lat_lon_alt[:, 2],
        origin_lat_lon_alt=origin,
    )
    _require_shape("enu_position_m", enu, sample_shape, config)

    sequence = HighRateOxtsSequence(
        timestamps=timestamps,
        lat_lon_alt=lat_lon_alt,
        enu_position_m=enu,
        roll_pitch_yaw=roll_pitch_yaw,
        velocity=velocity,
        accel_body=accel_body,
        gyro_body=gyro_body,
        origin_lat_lon_alt=origin,
        date=config.date,
        drive=config.normalized_drive(),
        source_path=str(oxts_path),
        cache_version=CACHE_VERSION,
        source_kind=SOURCE_KIND,
    )
    _validate_sequence_integrity(
        sequence,
        f"Extract sequence integrity error for {config.date} drive {config.normalized_drive()}",
    )
    return sequence


def load_highrate_oxts(config: HighRateOxtsConfig) -> HighRateOxtsSequence:
    path = cache_path_for(config)
    cache_metadata_error: HighRateOxtsSetupError | None = None
    if path.exists() and not config.force_refresh:
        try:
            sequence = load_cache(path)
            if _validate_cache(sequence, config):
                _validate_sequence_integrity(sequence, f"Cache integrity error: {path}")
                return sequence
            cache_metadata_error = _cache_metadata_mismatch_error(path, sequence, config)
        except (OSError, ValueError, KeyError, BadZipFile) as exc:
            raise HighRateOxtsSetupError(f"Cache could not be loaded: {path}") from exc

    try:
        oxts_path = require_highrate_oxts(config)
    except HighRateOxtsSetupError as exc:
        if cache_metadata_error is not None:
            raise cache_metadata_error from exc
        raise
    sequence = _build_sequence_from_extract(config, oxts_path)
    save_cache(sequence, path)
    return sequence
