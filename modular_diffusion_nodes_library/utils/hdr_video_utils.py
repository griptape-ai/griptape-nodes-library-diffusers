from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import Imath  # type: ignore[reportMissingImports]
import numpy as np
import OpenEXR  # type: ignore[reportMissingImports]


@dataclass(frozen=True)
class ExrFrameWriteEvent:
    """Payload reported via ``progress_callback`` after each EXR frame is written."""

    current_frame: int
    total_frames: int
    output_path: str
    frame_bytes: int
    total_bytes_written: int


def encode_linear_hdr_exr_sequence(
    frames: np.ndarray,
    output_dir: str,
    *,
    stem: str = "frame",
    save_as_half_float: bool = True,
    progress_callback: Callable[[ExrFrameWriteEvent], None] | None = None,
) -> list[str]:
    """Write linear HDR frames as an OpenEXR sequence.

    Each frame is written as a separate ``.exr`` file.

    Args:
        frames: Shape ``(F, H, W, 3)`` float32 linear HDR in ``[0, \u221e)``.
        output_dir: Directory to write the EXR files into (created if absent).
        stem: Filename stem; files are written as ``<stem>.0001.exr``, etc.
        save_as_half_float: Write channels as float16 when True, else float32.
        progress_callback: Optional callback invoked after each frame write with an
            :class:`ExrFrameWriteEvent`.

    Returns:
        List of absolute paths to the written EXR files, in frame order.
    """

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pixel_type = Imath.PixelType(Imath.PixelType.HALF if save_as_half_float else Imath.PixelType.FLOAT)
    exr_channel = Imath.Channel(pixel_type)
    header_channels = {"R": exr_channel, "G": exr_channel, "B": exr_channel}
    target_dtype = np.float16 if save_as_half_float else np.float32

    paths: list[str] = []
    total_bytes_written = 0
    total_frames = len(frames)
    for i, frame_rgb in enumerate(frames):
        frame_data = frame_rgb.astype(target_dtype)
        h, w = frame_data.shape[:2]

        header = OpenEXR.Header(w, h)
        header["channels"] = header_channels
        # BT.2020 primaries + D65 white point
        header["chromaticities"] = Imath.Chromaticities(
            Imath.V2f(0.708, 0.292),  # R
            Imath.V2f(0.170, 0.797),  # G
            Imath.V2f(0.131, 0.046),  # B
            Imath.V2f(0.3127, 0.3290),  # white
        )

        path = out_dir / f"{stem}.{i + 1:04d}.exr"
        exr = OpenEXR.OutputFile(str(path), header)
        exr.writePixels(
            {
                "R": np.ascontiguousarray(frame_data[:, :, 0]).tobytes(),
                "G": np.ascontiguousarray(frame_data[:, :, 1]).tobytes(),
                "B": np.ascontiguousarray(frame_data[:, :, 2]).tobytes(),
            }
        )
        exr.close()
        path_str = str(path)
        paths.append(path_str)
        frame_bytes = path.stat().st_size
        total_bytes_written += frame_bytes
        if progress_callback is not None:
            progress_callback(
                ExrFrameWriteEvent(
                    current_frame=i + 1,
                    total_frames=total_frames,
                    output_path=path_str,
                    frame_bytes=frame_bytes,
                    total_bytes_written=total_bytes_written,
                )
            )

    return paths
