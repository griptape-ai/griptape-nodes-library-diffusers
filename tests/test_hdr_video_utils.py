from pathlib import Path

import pytest

pytest.importorskip("numpy", reason="Builds real NumPy arrays; run `make test/exec` for the execution environment.")
pytest.importorskip("OpenEXR", reason="Writes real EXR files; run `make test/exec` for the execution environment.")

import numpy as np  # noqa: E402
import OpenEXR  # noqa: E402  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.utils.hdr_video_utils import (  # noqa: E402
    LOG_GAMMA_MAX_VALUE,
    encode_float_rgb_exr_frame,
    inverse_log_gamma,
    linear_to_log_gamma,
    linear_to_srgb,
    srgb_to_linear,
)


def test_srgb_to_linear_and_log_gamma_round_trip_endpoints() -> None:
    srgb = np.array([[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]], dtype=np.float32)

    encoded = linear_to_log_gamma(srgb_to_linear(srgb))

    assert np.allclose(encoded[0, 0], 0.0)
    # sRGB white is linear 1.0, which sits well below the 65536 log-gamma ceiling.
    assert np.allclose(encoded[0, 1], 0.3478, atol=1e-4)
    assert np.allclose(linear_to_log_gamma(np.full((1, 1, 3), LOG_GAMMA_MAX_VALUE)), 1.0)


def test_srgb_to_linear_uses_srgb_transfer_curve() -> None:
    srgb = np.array([[[0.04045, 0.5, 1.0]]], dtype=np.float32)

    linear = srgb_to_linear(srgb)

    assert np.isclose(linear[0, 0, 0], 0.04045 / 12.92, atol=1e-6)
    assert np.isclose(linear[0, 0, 2], 1.0)


def test_linear_to_srgb_uses_srgb_transfer_curve_and_clips_hdr() -> None:
    linear = np.array([[[0.0, 0.0031308, 0.5], [1.0, 2.0, -1.0]]], dtype=np.float32)

    srgb = linear_to_srgb(linear)

    assert np.allclose(srgb[0, 0], [0.0, 0.0404499, 0.7353569], atol=1e-6)
    assert np.allclose(srgb[0, 1], [1.0, 1.0, 0.0])


def test_inverse_log_gamma_round_trips_linear_radiance() -> None:
    linear = np.array([0.0, 0.18, 1.0, 10.0, 100.0], dtype=np.float32).reshape(1, 1, 5, 1)
    rgb = np.repeat(linear, 3, axis=-1)

    restored = inverse_log_gamma(linear_to_log_gamma(rgb))

    assert np.allclose(restored, rgb, rtol=1e-5, atol=1e-5)


def test_encode_float_rgb_exr_frame_returns_openexr_bytes(tmp_path: Path) -> None:
    frame = np.full((4, 6, 3), 0.5, dtype=np.float32)

    encoded = encode_float_rgb_exr_frame(frame, save_as_half_float=False)

    exr_path = tmp_path / "frame.exr"
    exr_path.write_bytes(encoded)
    data_window = OpenEXR.InputFile(str(exr_path)).header()["dataWindow"]
    assert encoded[:4] == b"\x76\x2f\x31\x01"
    assert (data_window.max.x + 1, data_window.max.y + 1) == (6, 4)
