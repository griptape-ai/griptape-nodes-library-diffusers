import numpy as np

from modular_diffusion_nodes_library.utils.hdr_video_utils import (
    inverse_log_gamma,
    linear_to_log_gamma,
    srgb_to_linear,
)


def test_srgb_to_linear_and_log_gamma_round_trip_endpoints() -> None:
    srgb = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)

    encoded = linear_to_log_gamma(srgb_to_linear(srgb))

    assert np.allclose(encoded[0, 0], 0.0)
    assert np.allclose(encoded[0, 1], 1.0)


def test_srgb_to_linear_uses_srgb_transfer_curve() -> None:
    srgb = np.array([[[0.04045, 0.5, 1.0]]], dtype=np.float32)

    linear = srgb_to_linear(srgb)

    assert np.isclose(linear[0, 0, 0], 0.04045 / 12.92, atol=1e-6)
    assert np.isclose(linear[0, 0, 2], 1.0)


def test_inverse_log_gamma_round_trips_linear_radiance() -> None:
    linear = np.array([0.0, 0.18, 1.0, 10.0, 100.0], dtype=np.float32).reshape(1, 1, 5, 1)
    rgb = np.repeat(linear, 3, axis=-1)

    restored = inverse_log_gamma(linear_to_log_gamma(rgb))

    assert np.allclose(restored, rgb, rtol=1e-5, atol=1e-5)
