from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from frames import (
    BASE_LINK,
    GPS_LINK,
    IMU_LINK,
    MAP,
    VELO_LINK,
    FrameCovariance,
    FrameError,
    FramePoint,
    FrameVector,
    Frames,
    RigidTransform,
    default_stage1_frames,
    kitti_imu_to_velo_transform,
    parse_kitti_rt_file,
)


def rot_z(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_x(theta_rad: float) -> np.ndarray:
    c = np.cos(theta_rad)
    s = np.sin(theta_rad)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def assert_identity_transform(transform: RigidTransform, atol: float = 1e-12) -> None:
    np.testing.assert_allclose(transform.rotation, np.eye(3), atol=atol, rtol=0.0)
    np.testing.assert_allclose(transform.translation, np.zeros(3), atol=atol, rtol=0.0)


def test_rigid_transform_inverse_and_compose_are_identity() -> None:
    t_map_base = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([10.0, -2.0, 1.5]),
    )

    identity = t_map_base.compose(t_map_base.inverse())

    assert identity.target == MAP
    assert identity.source == MAP
    assert_identity_transform(identity)


def test_rigid_transform_compose_matches_sequential_point_transforms() -> None:
    t_map_base = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([10.0, 0.0, 0.0]),
    )
    t_base_imu = RigidTransform(
        target=BASE_LINK,
        source="imu_link",
        rotation=rot_z(-np.pi / 2.0),
        translation=np.array([0.0, 2.0, 0.0]),
    )

    point_imu = np.array([1.0, 0.0, 0.0])
    composed = t_map_base.compose(t_base_imu)
    sequential = t_map_base.transform_points(t_base_imu.transform_points(point_imu))

    assert composed.target == MAP
    assert composed.source == "imu_link"
    np.testing.assert_allclose(composed.transform_points(point_imu), sequential, atol=1e-12, rtol=0.0)


def test_rigid_transform_points_and_vectors_handle_translation_differently() -> None:
    transform = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([10.0, 0.0, 0.0]),
    )

    point = transform.transform_points(np.array([1.0, 0.0, 0.0]))
    vector = transform.transform_vectors(np.array([1.0, 0.0, 0.0]))

    np.testing.assert_allclose(point, np.array([10.0, 1.0, 0.0]), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(vector, np.array([0.0, 1.0, 0.0]), atol=1e-12, rtol=0.0)


def test_rigid_transform_rejects_invalid_rotation() -> None:
    with pytest.raises(FrameError, match="orthonormal"):
        RigidTransform(
            target=MAP,
            source=BASE_LINK,
            rotation=np.diag([2.0, 1.0, 1.0]),
            translation=np.zeros(3),
        )

    with pytest.raises(FrameError, match="determinant"):
        RigidTransform(
            target=MAP,
            source=BASE_LINK,
            rotation=np.diag([1.0, 1.0, -1.0]),
            translation=np.zeros(3),
        )


def test_rigid_transform_rejects_malformed_numeric_inputs_as_frame_error() -> None:
    transform = RigidTransform.identity(MAP)

    with pytest.raises(FrameError, match="rotation"):
        RigidTransform(
            target=MAP,
            source=BASE_LINK,
            rotation=[[1.0, 0.0, 0.0], [0.0, 1.0]],
            translation=np.zeros(3),
        )

    with pytest.raises(FrameError, match="translation"):
        RigidTransform(
            target=MAP,
            source=BASE_LINK,
            rotation=np.eye(3),
            translation=["not-a-number", 0.0, 0.0],
        )

    with pytest.raises(FrameError, match="points"):
        transform.transform_points([["not-a-number", 0.0, 0.0]])

    with pytest.raises(FrameError, match="vectors"):
        transform.transform_vectors([[1.0, 2.0], [3.0]])

    with pytest.raises(FrameError, match="covariance"):
        transform.transform_covariance(
            [["not-a-number", 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        )


def test_rigid_transform_compose_rejects_non_transform() -> None:
    with pytest.raises(FrameError, match="RigidTransform"):
        RigidTransform.identity(MAP).compose(object())


def test_covariance_inputs_reject_non_symmetric_matrices() -> None:
    covariance = np.array(
        [
            [1.0, 0.5, 0.0],
            [0.25, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ]
    )

    with pytest.raises(FrameError, match="symmetric"):
        FrameCovariance(BASE_LINK, covariance)

    with pytest.raises(FrameError, match="symmetric"):
        RigidTransform.identity(BASE_LINK).transform_covariance(covariance)


def test_frames_resolves_indirect_transform_and_round_trips_point() -> None:
    t_map_base = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([10.0, 0.0, 0.0]),
    )
    t_base_gps = RigidTransform(
        target=BASE_LINK,
        source=GPS_LINK,
        rotation=rot_z(-np.pi / 2.0),
        translation=np.array([0.0, 2.0, 0.0]),
    )
    frames = Frames([t_map_base, t_base_gps])

    point_gps = FramePoint(GPS_LINK, np.array([1.0, 2.0, 3.0]))
    point_map = frames.transform_point(MAP, point_gps)
    expected_map = t_map_base.transform_points(t_base_gps.transform_points(point_gps.xyz))
    point_round_trip = frames.transform_point(GPS_LINK, point_map)

    assert point_map.frame == MAP
    np.testing.assert_allclose(point_map.xyz, expected_map, atol=1e-12, rtol=0.0)
    assert point_round_trip.frame == GPS_LINK
    np.testing.assert_allclose(point_round_trip.xyz, point_gps.xyz, atol=1e-12, rtol=0.0)


def test_frames_transform_vector_uses_rotation_without_translation() -> None:
    frames = Frames(
        [
            RigidTransform(
                target=MAP,
                source=BASE_LINK,
                rotation=rot_z(np.pi / 2.0),
                translation=np.array([10.0, -3.0, 2.0]),
            )
        ]
    )

    vector_base = FrameVector(BASE_LINK, np.array([1.0, 0.0, 0.0]))
    vector_map = frames.transform_vector(MAP, vector_base)

    assert vector_map.frame == MAP
    np.testing.assert_allclose(vector_map.xyz, np.array([0.0, 1.0, 0.0]), atol=1e-12, rtol=0.0)


def test_frames_transform_covariance_round_trip() -> None:
    t_map_base = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 3.0),
        translation=np.array([4.0, -2.0, 1.0]),
    )
    frames = Frames([t_map_base])
    covariance_base = FrameCovariance(
        BASE_LINK,
        np.array(
            [
                [4.0, 0.5, 0.0],
                [0.5, 2.0, 0.25],
                [0.0, 0.25, 1.0],
            ]
        ),
    )

    covariance_map = frames.transform_covariance(MAP, covariance_base)
    covariance_round_trip = frames.transform_covariance(BASE_LINK, covariance_map)

    assert covariance_map.frame == MAP
    np.testing.assert_allclose(
        covariance_map.covariance,
        t_map_base.rotation @ covariance_base.covariance @ t_map_base.rotation.T,
        atol=1e-12,
        rtol=0.0,
    )
    assert covariance_round_trip.frame == BASE_LINK
    np.testing.assert_allclose(
        covariance_round_trip.covariance,
        covariance_base.covariance,
        atol=1e-12,
        rtol=0.0,
    )


def test_frames_transform_covariance_accepts_large_scale_roundoff_asymmetry() -> None:
    rotation = rot_z(0.31) @ rot_y(-0.27) @ rot_x(0.19)
    frames = Frames(
        [
            RigidTransform(
                target=MAP,
                source=BASE_LINK,
                rotation=rotation,
                translation=np.array([4.0, -2.0, 1.0]),
            )
        ]
    )
    covariance_base = FrameCovariance(
        BASE_LINK,
        1e12
        * np.array(
            [
                [4.0, 0.7, -0.2],
                [0.7, 2.0, 0.4],
                [-0.2, 0.4, 1.0],
            ]
        ),
    )

    covariance_map = frames.transform_covariance(MAP, covariance_base)

    assert covariance_map.frame == MAP
    np.testing.assert_allclose(
        covariance_map.covariance,
        covariance_map.covariance.T,
        atol=1e-9,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        covariance_map.covariance,
        rotation @ covariance_base.covariance @ rotation.T,
        atol=1e-9,
        rtol=1e-12,
    )


def test_frames_rejects_unreachable_transform() -> None:
    frames = Frames([RigidTransform.identity(MAP)])

    with pytest.raises(FrameError, match=f"No transform path from {BASE_LINK} to {MAP}"):
        frames.transform(MAP, BASE_LINK)


def test_frames_accepts_equivalent_duplicate_transform() -> None:
    transform = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([1.0, 2.0, 3.0]),
    )
    frames = Frames([transform])

    frames.add(
        RigidTransform(
            target=MAP,
            source=BASE_LINK,
            rotation=rot_z(np.pi / 2.0),
            translation=np.array([1.0, 2.0, 3.0]),
        )
    )

    resolved = frames.transform(MAP, BASE_LINK)
    np.testing.assert_allclose(resolved.rotation, transform.rotation, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        resolved.translation,
        transform.translation,
        atol=1e-12,
        rtol=0.0,
    )


def test_frames_rejects_conflicting_duplicate_transform() -> None:
    transform = RigidTransform(
        target=MAP,
        source=BASE_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([1.0, 2.0, 3.0]),
    )

    frames = Frames([transform])
    with pytest.raises(FrameError, match="conflicting transform"):
        frames.add(
            RigidTransform(
                target=MAP,
                source=BASE_LINK,
                rotation=rot_z(np.pi / 2.0),
                translation=np.array([1.0, 2.0, 4.0]),
            )
        )

    frames = Frames([transform])
    with pytest.raises(FrameError, match="conflicting transform"):
        frames.add(
            RigidTransform(
                target=BASE_LINK,
                source=MAP,
                rotation=rot_z(np.pi / 2.0),
                translation=np.zeros(3),
            )
        )


def test_parse_kitti_rt_file_reads_rotation_and_translation(tmp_path: Path) -> None:
    calib_path = tmp_path / "calib_imu_to_velo.txt"
    rotation = np.array(
        [
            [9.999976e-01, 7.553071e-04, -2.035826e-03],
            [-7.854027e-04, 9.998898e-01, -1.482298e-02],
            [2.024406e-03, 1.482454e-02, 9.998881e-01],
        ]
    )
    calib_path.write_text(
        "\n".join(
            [
                "calib_time: 15-Mar-2012 11:37:16",
                (
                    "R: "
                    "9.999976e-01 7.553071e-04 -2.035826e-03 "
                    "-7.854027e-04 9.998898e-01 -1.482298e-02 "
                    "2.024406e-03 1.482454e-02 9.998881e-01"
                ),
                "T: 0.5 -0.25 1.25",
            ]
        ),
        encoding="utf-8",
    )

    parsed_rotation, translation = parse_kitti_rt_file(calib_path)

    np.testing.assert_allclose(parsed_rotation, rotation, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(translation, np.array([0.5, -0.25, 1.25]), atol=1e-12, rtol=0.0)


def test_parse_kitti_rt_file_rejects_missing_translation(tmp_path: Path) -> None:
    calib_path = tmp_path / "calib_imu_to_velo.txt"
    calib_path.write_text(
        "\n".join(
            [
                "calib_time: 15-Mar-2012 11:37:16",
                "R: 1 0 0 0 1 0 0 0 1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(FrameError, match="missing required key T"):
        parse_kitti_rt_file(calib_path)


def test_kitti_imu_to_velo_transform_uses_expected_direction(tmp_path: Path) -> None:
    calib_path = tmp_path / "calib_imu_to_velo.txt"
    calib_path.write_text(
        "\n".join(
            [
                "R: 0 -1 0 1 0 0 0 0 1",
                "T: 0.5 -0.25 1.25",
            ]
        ),
        encoding="utf-8",
    )

    transform = kitti_imu_to_velo_transform(calib_path)

    assert transform.target == VELO_LINK
    assert transform.source == IMU_LINK
    np.testing.assert_allclose(transform.rotation, rot_z(np.pi / 2.0), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(transform.translation, np.array([0.5, -0.25, 1.25]), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        transform.transform_points(np.array([1.0, 0.0, 0.0])),
        np.array([0.5, 0.75, 1.25]),
        atol=1e-12,
        rtol=0.0,
    )


def test_default_stage1_frames_use_colocated_base_imu_gps_and_configurable_lever_arm() -> None:
    frames = default_stage1_frames()

    assert_identity_transform(frames.transform(BASE_LINK, IMU_LINK))
    assert_identity_transform(frames.transform(BASE_LINK, GPS_LINK))

    t_velo_imu = RigidTransform(
        target=VELO_LINK,
        source=IMU_LINK,
        rotation=rot_z(np.pi / 2.0),
        translation=np.array([0.5, -0.25, 1.25]),
    )
    frames = default_stage1_frames(
        p_base_imu=(1.0, 2.0, 3.0),
        p_base_gps=(-1.0, 0.5, 0.25),
        t_velo_imu=t_velo_imu,
    )

    t_base_imu = frames.transform(BASE_LINK, IMU_LINK)
    t_base_gps = frames.transform(BASE_LINK, GPS_LINK)
    resolved_t_velo_imu = frames.transform(VELO_LINK, IMU_LINK)

    np.testing.assert_allclose(t_base_imu.rotation, np.eye(3), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(t_base_imu.translation, np.array([1.0, 2.0, 3.0]), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(t_base_gps.rotation, np.eye(3), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(t_base_gps.translation, np.array([-1.0, 0.5, 0.25]), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(resolved_t_velo_imu.rotation, t_velo_imu.rotation, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        resolved_t_velo_imu.translation,
        t_velo_imu.translation,
        atol=1e-12,
        rtol=0.0,
    )

    with pytest.raises(FrameError, match="t_velo_imu"):
        default_stage1_frames(
            t_velo_imu=RigidTransform(
                target=IMU_LINK,
                source=VELO_LINK,
                rotation=np.eye(3),
                translation=np.zeros(3),
            )
        )


def test_local_kitti_imu_to_velo_calibration_if_present() -> None:
    calib_path = Path("data/kitti_raw/2011_09_26/calib_imu_to_velo.txt")
    if not calib_path.exists():
        pytest.skip("local KITTI Raw date-level calibration is not installed")

    transform = kitti_imu_to_velo_transform(calib_path)

    assert transform.target == VELO_LINK
    assert transform.source == IMU_LINK
    np.testing.assert_allclose(transform.rotation.T @ transform.rotation, np.eye(3), atol=1e-6, rtol=0.0)
    assert np.isclose(np.linalg.det(transform.rotation), 1.0, atol=1e-6, rtol=0.0)
    assert transform.translation.shape == (3,)
    assert np.isfinite(transform.translation).all()
