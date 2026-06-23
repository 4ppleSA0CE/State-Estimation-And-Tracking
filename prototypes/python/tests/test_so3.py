import numpy as np
import pytest

from so3 import (
    SO3Error,
    euler_to_quat,
    quat_inverse,
    quat_multiply,
    quat_normalize,
    quat_to_euler,
    quat_to_rotmat,
    rotvec_to_quat,
)


def test_quat_normalize_returns_unit_scalar_first_quaternion():
    q = quat_normalize([2, 0, 0, 0])

    np.testing.assert_allclose(q, np.array([1, 0, 0, 0], dtype=float))
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_quat_normalize_handles_large_finite_values_without_overflow():
    q = quat_normalize([1e308, 0, 0, 0])

    np.testing.assert_allclose(q, np.array([1, 0, 0, 0], dtype=float))
    assert np.linalg.norm(q) == pytest.approx(1.0)


def test_quat_normalize_canonicalizes_negative_scalar_without_zeroing_tiny_terms():
    q = quat_normalize([-1.0, -1e-16, 0, 0])

    np.testing.assert_allclose(q, np.array([1.0, 1e-16, 0, 0], dtype=float), atol=1e-30)


def test_quat_normalize_rejects_invalid_inputs():
    with pytest.raises(SO3Error, match="shape"):
        quat_normalize([1, 0, 0])

    with pytest.raises(SO3Error, match="nonzero"):
        quat_normalize([0, 0, 0, 0])

    with pytest.raises(SO3Error, match="finite"):
        quat_normalize([np.nan, 0, 0, 0])


def test_quat_multiply_and_inverse_round_trip():
    q = rotvec_to_quat([0.1, -0.2, 0.3])

    identity = quat_multiply(q, quat_inverse(q))

    np.testing.assert_allclose(identity, np.array([1, 0, 0, 0], dtype=float), atol=1e-15)


def test_quat_multiply_matches_hamilton_product_for_non_unit_inputs():
    q = quat_multiply([2, 1, 0, 0], [2, 0, 1, 0])

    np.testing.assert_allclose(q, np.array([4 / 5, 2 / 5, 2 / 5, 1 / 5], dtype=float))


def test_rotvec_to_quat_small_and_finite_rotation():
    small = rotvec_to_quat([1e-16, 0, 0])
    finite = rotvec_to_quat([0, 0, np.pi / 2])

    np.testing.assert_allclose(small, np.array([1, 5e-17, 0, 0], dtype=float), atol=1e-30)
    np.testing.assert_allclose(
        finite,
        np.array([np.sqrt(0.5), 0, 0, np.sqrt(0.5)], dtype=float),
    )


def test_rotvec_to_quat_rejects_invalid_inputs():
    with pytest.raises(SO3Error, match="shape"):
        rotvec_to_quat([0, 0, 0, 0])

    with pytest.raises(SO3Error, match="finite"):
        rotvec_to_quat([np.nan, 0, 0])


def test_quat_to_rotmat_is_valid_rotation_matrix():
    q = euler_to_quat(0.1, -0.2, 0.3)

    R = quat_to_rotmat(q)

    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-15)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_yaw_rotation_matches_expected_matrix_and_euler_round_trip():
    q = euler_to_quat(0, 0, np.pi / 2)

    R = quat_to_rotmat(q)
    roll, pitch, yaw = quat_to_euler(q)

    np.testing.assert_allclose(
        R,
        np.array(
            [
                [0, -1, 0],
                [1, 0, 0],
                [0, 0, 1],
            ],
            dtype=float,
        ),
        atol=1e-15,
    )
    assert roll == pytest.approx(0.0)
    assert pitch == pytest.approx(0.0)
    assert yaw == pytest.approx(np.pi / 2)


def test_euler_nontrivial_round_trip():
    angles = np.array([0.31, -0.27, 0.42], dtype=float)

    q = euler_to_quat(*angles)
    result = quat_to_euler(q)

    np.testing.assert_allclose(result, angles, atol=1e-12)


def test_quat_to_euler_reconstructs_near_gimbal_without_collapsing_to_singular_branch():
    angles = np.array([0.8, np.pi / 2.0 - 5e-8, -0.4])

    q = euler_to_quat(*angles)
    recovered = quat_to_euler(q)
    reconstructed = euler_to_quat(*recovered)

    np.testing.assert_allclose(
        quat_to_rotmat(reconstructed),
        quat_to_rotmat(q),
        atol=1e-12,
    )
    assert abs(recovered[1]) < np.pi / 2.0


def test_quat_to_euler_reconstructs_very_near_gimbal_without_collapsing_to_singular_branch():
    angles = np.array([0.8, np.pi / 2.0 - 1e-13, -0.4])

    q = euler_to_quat(*angles)
    recovered = quat_to_euler(q)
    reconstructed = euler_to_quat(*recovered)

    np.testing.assert_allclose(
        quat_to_rotmat(reconstructed),
        quat_to_rotmat(q),
        atol=1e-9,
    )
    assert abs(recovered[1]) < np.pi / 2.0


def test_quat_to_euler_reconstructs_rotation_at_positive_pitch_gimbal_lock():
    original = euler_to_quat(0.3, np.pi / 2, 0.7)

    recovered = quat_to_euler(original)
    reconstructed = euler_to_quat(*recovered)

    np.testing.assert_allclose(
        quat_to_rotmat(reconstructed),
        quat_to_rotmat(original),
        atol=1e-12,
    )


def test_quat_to_euler_reconstructs_rotation_at_negative_pitch_gimbal_lock():
    original = euler_to_quat(-0.4, -np.pi / 2, 0.2)

    recovered = quat_to_euler(original)
    reconstructed = euler_to_quat(*recovered)

    np.testing.assert_allclose(
        quat_to_rotmat(reconstructed),
        quat_to_rotmat(original),
        atol=1e-12,
    )
