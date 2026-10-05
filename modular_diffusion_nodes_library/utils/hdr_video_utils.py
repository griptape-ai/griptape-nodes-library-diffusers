from __future__ import annotations

import math
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np  # type: ignore[reportMissingImports]

LOG_GAMMA = 2.2
LOG_GAMMA_MAX_VALUE = 65536.0
LOG_GAMMA_DENOMINATOR = math.log(LOG_GAMMA * LOG_GAMMA_MAX_VALUE + 1.0)


def srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
    """Convert normalized sRGB values in [0, 1] to linear light."""
    import numpy as np

    if srgb.ndim < 1 or srgb.shape[-1] != 3:
        raise ValueError(f"Expected an RGB array ending in 3 channels, got shape {srgb.shape}.")

    srgb = np.clip(srgb.astype(np.float32, copy=False), 0.0, 1.0)
    return np.where(srgb <= 0.04045, srgb / 12.92, ((srgb + 0.055) / 1.055) ** 2.4)


def linear_to_log_gamma(linear: np.ndarray) -> np.ndarray:
    """Apply upstream linear-radiance to Log-Gamma transform."""
    import numpy as np

    if linear.ndim < 1 or linear.shape[-1] != 3:
        raise ValueError(f"Expected an RGB array ending in 3 channels, got shape {linear.shape}.")

    linear = np.clip(linear.astype(np.float32, copy=False), 0.0, LOG_GAMMA_MAX_VALUE)
    normalized_log = np.log(LOG_GAMMA * linear + 1.0) / LOG_GAMMA_DENOMINATOR
    return np.power(normalized_log, 1.0 / LOG_GAMMA)


def inverse_log_gamma(log_gamma: np.ndarray) -> np.ndarray:
    """Apply upstream inverse Log-Gamma transform to linear radiance."""
    import numpy as np

    if log_gamma.ndim < 1 or log_gamma.shape[-1] != 3:
        raise ValueError(f"Expected an RGB array ending in 3 channels, got shape {log_gamma.shape}.")

    values = log_gamma.astype(np.float32, copy=False)
    return (np.exp(np.power(values, LOG_GAMMA) * LOG_GAMMA_DENOMINATOR) - 1.0) / LOG_GAMMA


def inverse_logc3(logc: np.ndarray) -> np.ndarray:
    """Convert ARRI LogC3-encoded RGB values in [0, 1] to linear HDR."""
    import numpy as np

    if logc.ndim < 1 or logc.shape[-1] != 3:
        raise ValueError(f"Expected an RGB array ending in 3 channels, got shape {logc.shape}.")

    logc = np.clip(logc.astype(np.float32, copy=False), 0.0, 1.0)
    cut_log = 5.367655 * 0.010591 + 0.092809
    linear_from_log = (np.power(10.0, (logc - 0.385537) / 0.247190) - 0.052272) / 5.555556
    linear_from_linear = (logc - 0.092809) / 5.367655
    return np.where(logc >= cut_log, linear_from_log, linear_from_linear)


def encode_float_rgb_exr_frame(frame_rgb: np.ndarray, *, save_as_half_float: bool = True) -> bytes:
    """Encode one float RGB frame as the bytes of a single-part OpenEXR file.

    Args:
        frame_rgb: Shape ``(H, W, 3)`` floating-point RGB values.
        save_as_half_float: Write channels as float16 when True, else float32.

    Returns:
        Bytes of the encoded ``.exr`` file.
    """
    import Imath  # type: ignore[reportMissingImports]
    import numpy as np
    import OpenEXR  # type: ignore[reportMissingImports]

    pixel_type = Imath.PixelType(Imath.PixelType.HALF if save_as_half_float else Imath.PixelType.FLOAT)
    exr_channel = Imath.Channel(pixel_type)
    target_dtype = np.float16 if save_as_half_float else np.float32
    frame_data = frame_rgb.astype(target_dtype)
    h, w = frame_data.shape[:2]

    header = OpenEXR.Header(w, h)
    header["channels"] = {"R": exr_channel, "G": exr_channel, "B": exr_channel}
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "frame.exr"
        exr = OpenEXR.OutputFile(str(path), header)
        try:
            exr.writePixels(
                {
                    "R": np.ascontiguousarray(frame_data[:, :, 0]).tobytes(),
                    "G": np.ascontiguousarray(frame_data[:, :, 1]).tobytes(),
                    "B": np.ascontiguousarray(frame_data[:, :, 2]).tobytes(),
                }
            )
        finally:
            exr.close()
        return path.read_bytes()
