"""KITTI raw OXTS loading helpers and cache utilities."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

CACHE_VERSION = 1  # bump when cache schema changes; stale caches are rejected on load
EARTH_RADIUS_M = 6_378_137.0  # WGS-84 equatorial radius, m; used for flat-earth ENU approx
KITTI_RAW_URL = "https://www.cvlibs.net/datasets/kitti/raw_data.php"  # download page
DEFAULT_GPS_STD_M = 1.5  # default isotropic GPS position noise std, m


class KittiSetupError(RuntimeError):
    """Raised when KITTI raw data or optional loader dependencies are missing."""


@dataclass(frozen=True)
class KittiLoaderConfig:
    root: Path = Path("data/kitti_raw")  # root path to KITTI raw dataset, contains date folders
    date: str = "2011_09_26"  # capture date in YYYY_MM_DD format
    drive: str = "0001"  # 4-digit drive identifier (zero-padded)
    cache_root: Path = Path("data/cache")  # directory for .npz cache files
    force_refresh: bool = False  # skip cache and reload from raw files when True
    gps_std_m: float = DEFAULT_GPS_STD_M  # isotropic GPS position noise std, m

    def normalized_drive(self) -> str:
        return self.drive.zfill(4)


@dataclass(frozen=True)
class KittiSequence:
    timestamps: np.ndarray  # (N,) time since first sample, s
    lat_lon_alt: np.ndarray  # (N, 3) WGS-84 lat deg, lon deg, alt m
    enu_position_m: np.ndarray  # (N, 3) ENU position relative to first sample, m
    roll_pitch_yaw: np.ndarray  # (N, 3) OXTS roll, pitch, yaw, rad
    velocity: np.ndarray  # (N, 3) body-frame forward/left/up velocity, m/s
    accel_body: np.ndarray  # (N, 3) body-frame specific force (accel - gravity), m/s^2
    gyro_body: np.ndarray  # (N, 3) body-frame angular rate, rad/s
    gps_covariance: np.ndarray  # (N, 3, 3) ENU position covariance from gps_std_m, m^2
    origin_lat_lon_alt: np.ndarray  # (3,) WGS-84 origin used for ENU conversion
    date: str  # date string YYYY_MM_DD
    drive: str  # 4-digit drive string
    source_path: str  # path to the raw data date folder loaded from
    cache_version: int = CACHE_VERSION  # schema version, must match CACHE_VERSION to use

    @property
    def sample_count(self) -> int:
        return int(self.timestamps.shape[0])

    @property
    def duration_s(self) -> float:
        if self.sample_count < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])


def enu_from_wgs84(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    alt_m: np.ndarray,
    origin_lat_lon_alt: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    origin = np.asarray(origin_lat_lon_alt, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    alt = np.asarray(alt_m, dtype=float)

    east = np.deg2rad(lon - origin[1]) * EARTH_RADIUS_M * np.cos(np.deg2rad(origin[0]))  # longitude arc scaled by cos(lat), m
    north = np.deg2rad(lat - origin[0]) * EARTH_RADIUS_M  # latitude arc, m
    up = alt - origin[2]  # altitude difference, m
    return np.column_stack([east, north, up])


def cache_path_for(config: KittiLoaderConfig) -> Path:
    return (
        config.cache_root
        / f"kitti_raw_{config.date}_drive_{config.normalized_drive()}_oxts_v{CACHE_VERSION}.npz"
    )


def save_cache(sequence: KittiSequence, path: Path | str) -> None:
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
        gps_covariance=sequence.gps_covariance,
        origin_lat_lon_alt=sequence.origin_lat_lon_alt,
        date=np.array(sequence.date),
        drive=np.array(sequence.drive),
        source_path=np.array(sequence.source_path),
        cache_version=np.array(sequence.cache_version, dtype=np.int64),
    )


def load_cache(path: Path | str) -> KittiSequence:
    with np.load(Path(path), allow_pickle=False) as data:
        return KittiSequence(
            timestamps=data["timestamps"],
            lat_lon_alt=data["lat_lon_alt"],
            enu_position_m=data["enu_position_m"],
            roll_pitch_yaw=data["roll_pitch_yaw"],
            velocity=data["velocity"],
            accel_body=data["accel_body"],
            gyro_body=data["gyro_body"],
            gps_covariance=data["gps_covariance"],
            origin_lat_lon_alt=data["origin_lat_lon_alt"],
            date=str(data["date"].item()),
            drive=str(data["drive"].item()),
            source_path=str(data["source_path"].item()),
            cache_version=int(data["cache_version"].item()),
        )


def _sequence_date_path(config: KittiLoaderConfig) -> Path:
    return config.root / config.date


def _sequence_drive_path(config: KittiLoaderConfig) -> Path:
    return _sequence_date_path(config) / f"{config.date}_drive_{config.normalized_drive()}_sync"


def _validate_cache(sequence: KittiSequence, config: KittiLoaderConfig) -> bool:
    return (
        sequence.cache_version == CACHE_VERSION
        and sequence.date == config.date
        and sequence.drive == config.normalized_drive()
    )


def _timestamp_seconds(timestamps: list[object]) -> np.ndarray:
    if not timestamps:
        return np.empty((0,), dtype=float)
    first = timestamps[0]
    seconds: list[float] = []
    for idx, timestamp in enumerate(timestamps):
        try:
            delta = timestamp - first
            seconds.append(float(delta.total_seconds()))
        except (AttributeError, TypeError, ValueError) as exc:
            raise KittiSetupError(f"Invalid KITTI OXTS timestamp at index {idx}: {timestamp!r}.") from exc
    return np.asarray(seconds, dtype=float)


def _oxts_packet_arrays(dataset: object) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_lon_alt: list[list[float]] = []
    roll_pitch_yaw: list[list[float]] = []
    velocity: list[list[float]] = []
    accel_body: list[list[float]] = []
    gyro_body: list[list[float]] = []

    for idx, packet in enumerate(dataset.oxts):
        try:
            data = packet.packet
        except AttributeError as exc:
            raise KittiSetupError(f"Invalid KITTI OXTS packet at index {idx}: missing packet data.") from exc

        lat_lon_alt.append(_packet_values(data, idx, ("lat", "lon", "alt")))
        roll_pitch_yaw.append(_packet_values(data, idx, ("roll", "pitch", "yaw")))
        velocity.append(_packet_values(data, idx, ("vf", "vl", "vu")))
        accel_body.append(_packet_values(data, idx, ("af", "al", "au")))
        gyro_body.append(_packet_values(data, idx, ("wf", "wl", "wu")))

    return (
        np.asarray(lat_lon_alt, dtype=float),
        np.asarray(roll_pitch_yaw, dtype=float),
        np.asarray(velocity, dtype=float),
        np.asarray(accel_body, dtype=float),
        np.asarray(gyro_body, dtype=float),
    )


def _packet_values(data: object, packet_idx: int, fields: tuple[str, str, str]) -> list[float]:
    values: list[float] = []
    for field in fields:
        try:
            value = getattr(data, field)
        except AttributeError as exc:
            raise KittiSetupError(
                f"Invalid KITTI OXTS packet at index {packet_idx}: missing field {field}."
            ) from exc
        try:
            values.append(float(value))
        except (TypeError, ValueError) as exc:
            raise KittiSetupError(
                f"Invalid KITTI OXTS packet at index {packet_idx}: field {field} is not numeric."
            ) from exc
    return values


def _require_shape(name: str, array: np.ndarray, shape: tuple[int, ...], config: KittiLoaderConfig) -> None:
    if array.shape != shape:
        raise KittiSetupError(
            f"Invalid KITTI OXTS {name} shape for {config.date} drive {config.normalized_drive()}: "
            f"expected {shape}, got {array.shape}."
        )


def _build_sequence_with_pykitti(config: KittiLoaderConfig, pykitti_module: object) -> KittiSequence:
    drive = config.normalized_drive()
    dataset = pykitti_module.raw(str(config.root), config.date, drive)

    timestamps = _timestamp_seconds(list(dataset.timestamps))
    lat_lon_alt, roll_pitch_yaw, velocity, accel_body, gyro_body = _oxts_packet_arrays(dataset)
    if timestamps.shape[0] == 0 or lat_lon_alt.shape[0] == 0:
        raise KittiSetupError(f"No OXTS samples found for KITTI {config.date} drive {drive}.")
    if timestamps.shape[0] != lat_lon_alt.shape[0]:
        raise KittiSetupError(
            f"Timestamp/OXTS length mismatch for KITTI {config.date} drive {drive}: "
            f"{timestamps.shape[0]} timestamps vs {lat_lon_alt.shape[0]} OXTS packets."
        )
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
    gps_covariance = np.repeat(
        (config.gps_std_m**2) * np.eye(3)[None, :, :],
        timestamps.shape[0],
        axis=0,
    )
    _require_shape("gps_covariance", gps_covariance, (timestamps.shape[0], 3, 3), config)

    return KittiSequence(
        timestamps=timestamps,
        lat_lon_alt=lat_lon_alt,
        enu_position_m=enu,
        roll_pitch_yaw=roll_pitch_yaw,
        velocity=velocity,
        accel_body=accel_body,
        gyro_body=gyro_body,
        gps_covariance=gps_covariance,
        origin_lat_lon_alt=origin,
        date=config.date,
        drive=drive,
        source_path=str(_sequence_date_path(config)),
        cache_version=CACHE_VERSION,
    )


def _load_from_pykitti(config: KittiLoaderConfig) -> KittiSequence:
    if not config.root.exists():
        raise KittiSetupError(
            f"KITTI Raw root does not exist: {config.root}. "
            f"Download KITTI raw data from {KITTI_RAW_URL} or pass --root."
        )
    date_path = _sequence_date_path(config)
    if not date_path.exists():
        raise KittiSetupError(
            f"KITTI Raw date folder does not exist: {date_path}. "
            f"Download KITTI raw data from {KITTI_RAW_URL} or pass --root."
        )
    drive_path = _sequence_drive_path(config)
    oxts_path = drive_path / "oxts"
    oxts_data_path = oxts_path / "data"
    if not drive_path.exists():
        raise KittiSetupError(
            f"KITTI Raw sequence does not exist: {drive_path}. "
            f"Expected drive {config.normalized_drive()} under {date_path}; "
            f"download it from {KITTI_RAW_URL}."
        )
    if not (
        (oxts_path / "timestamps.txt").exists()
        and oxts_data_path.is_dir()
        and any(oxts_data_path.glob("*.txt"))
    ):
        raise KittiSetupError(
            f"KITTI Raw OXTS files are missing for {drive_path}. "
            "Expected oxts/timestamps.txt and oxts/data/*.txt."
        )
    try:
        import pykitti
    except ImportError as exc:
        raise KittiSetupError(
            "pykitti is required to load KITTI raw OXTS data. "
            "Install it with `python3 -m pip install pykitti`."
        ) from exc
    return _build_sequence_with_pykitti(config, pykitti)


def load_kitti_sequence(config: KittiLoaderConfig) -> KittiSequence:
    path = cache_path_for(config)
    if path.exists() and not config.force_refresh:
        try:
            sequence = load_cache(path)
            if _validate_cache(sequence, config):
                return sequence
            raise KittiSetupError(
                f"Cache metadata mismatch: {path}. "
                f"Expected date={config.date}, drive={config.normalized_drive()}, "
                f"version={CACHE_VERSION}; found date={sequence.date}, "
                f"drive={sequence.drive}, version={sequence.cache_version}."
            )
        except (OSError, ValueError, KeyError, BadZipFile) as exc:
            raise KittiSetupError(f"Cache could not be loaded: {path}") from exc

    sequence = _load_from_pykitti(config)
    save_cache(sequence, path)
    return sequence


def format_sequence_summary(sequence: KittiSequence, cache_path: Path | str) -> str:
    origin = sequence.origin_lat_lon_alt
    enu = sequence.enu_position_m
    if sequence.sample_count == 0:
        span = np.zeros(3)
    else:
        span = np.ptp(enu, axis=0)

    return "\n".join(
        [
            f"sequence: {sequence.date} drive {sequence.drive}",
            f"samples: {sequence.sample_count}",
            f"duration_s: {sequence.duration_s:.3f}",
            f"origin_lat_lon_alt: {origin[0]:.8f}, {origin[1]:.8f}, {origin[2]:.3f}",
            f"enu_span_m: east={span[0]:.3f}, north={span[1]:.3f}, up={span[2]:.3f}",
            f"cache: {Path(cache_path)}",
        ]
    )


def plot_enu_trajectory(sequence: KittiSequence, output_path: Path | str) -> Path:
    import matplotlib.pyplot as plt

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    east = sequence.enu_position_m[:, 0]
    north = sequence.enu_position_m[:, 1]

    fig, ax = plt.subplots()
    try:
        ax.plot(east, north, label="ENU trajectory")
        if sequence.sample_count > 0:
            ax.scatter(east[0], north[0], marker="o", label="start")
            ax.scatter(east[-1], north[-1], marker="x", label="end")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(path, dpi=150)
    finally:
        plt.close(fig)
    return path


def default_plot_path(date: str, drive: str) -> Path:
    return Path("prototypes/output") / f"kitti_{date}_{drive.zfill(4)}_enu.png"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load and summarize a KITTI raw OXTS sequence.")
    parser.add_argument("--root", type=Path, default=KittiLoaderConfig.root)
    parser.add_argument("--date", default=KittiLoaderConfig.date)
    parser.add_argument("--drive", default=KittiLoaderConfig.drive)
    parser.add_argument("--cache-root", type=Path, default=KittiLoaderConfig.cache_root)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--plot", nargs="?", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = KittiLoaderConfig(
        root=args.root,
        date=args.date,
        drive=args.drive,
        cache_root=args.cache_root,
        force_refresh=args.force_refresh,
    )
    cache_path = cache_path_for(config)
    plot_path = args.plot if args.plot is not None else default_plot_path(config.date, config.normalized_drive())

    try:
        sequence = load_kitti_sequence(config)
    except KittiSetupError as exc:
        print(f"error: {exc}")
        return 2

    plot_enu_trajectory(sequence, plot_path)
    print(format_sequence_summary(sequence, cache_path))
    print(f"plot: {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
