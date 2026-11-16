from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast
from urllib.parse import unquote, urlparse

from griptape.artifacts.video_url_artifact import VideoUrlArtifact
from griptape_nodes.files.file import File

if TYPE_CHECKING:
    import numpy as np  # type: ignore[reportMissingImports]
    import torch  # type: ignore[reportMissingImports]
    from PIL import Image  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import VideoMedia


def resize_video_frames_torch(frames: np.ndarray, height: int, width: int) -> np.ndarray:
    """Resize an FHWC float video array with antialiased bicubic interpolation.

    Interpolation is computed in float32 and the input floating dtype is
    restored on return. Values are not clipped or quantized.
    """
    import numpy as np  # type: ignore[reportMissingImports]
    import torch  # type: ignore[reportMissingImports]
    import torch.nn.functional as F  # type: ignore[reportMissingImports]  # noqa: N812

    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"Expected RGB video frames with shape (F, H, W, 3), got {frames.shape}.")
    if frames.dtype not in (np.dtype(np.float16), np.dtype(np.float32)):
        raise TypeError(f"Expected float16 or float32 video frames, got {frames.dtype}.")
    if height <= 0 or width <= 0:
        raise ValueError(f"Resize dimensions must be positive, got height={height}, width={width}.")

    input_tensor = torch.from_numpy(np.array(frames, copy=True, order="C"))
    input_tensor = input_tensor.permute(0, 3, 1, 2).to(dtype=torch.float32)
    resized_tensor = F.interpolate(input_tensor, size=(height, width), mode="bicubic", antialias=True)
    resized_frames = resized_tensor.permute(0, 2, 3, 1).contiguous().numpy()
    return resized_frames.astype(frames.dtype, copy=False)


def download_video_to_temp_file(video_url_artifact: VideoUrlArtifact) -> Path:
    """Download a video from a VideoUrlArtifact to a temporary file.

    Args:
        video_url_artifact: The VideoUrlArtifact containing the video URL

    Returns:
        Path to the temporary file containing the downloaded video.
        The caller is responsible for cleaning up this file.

    Raises:
        ValueError: If video download fails with descriptive error message
    """
    url = video_url_artifact.value

    # Extract suffix from URL, defaulting to .mp4
    parsed_url = urlparse(url)
    url_path = Path(unquote(parsed_url.path))
    suffix = url_path.suffix or ".mp4"

    # TODO(#60): Use ProjectFileDestination with temp files situation
    fd, temp_path_str = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    temp_path = Path(temp_path_str)

    try:
        video_bytes = File(url).read_bytes()
        temp_path.write_bytes(video_bytes)
    except Exception as err:
        temp_path.unlink(missing_ok=True)
        details = f"Failed to download video at '{url}'.\nError: {err}"
        raise ValueError(details) from err

    return temp_path


def load_video_media_from_url_artifact(video_url_artifact: VideoUrlArtifact) -> VideoMedia:
    """Load a video artifact as frames, frame rate and soundtrack.

    Frames and fps come from the same decoder as `load_video_frames_from_url_artifact`, so
    orientation and frame content match what the frame-only path produces. The soundtrack is
    decoded separately, and is None when the file has no audio stream. Drivers that do not
    use audio simply ignore it.
    """
    import diffusers.utils  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import VideoMedia

    temp_path = download_video_to_temp_file(video_url_artifact)
    try:
        # load_video returns a bare list unless return_fps is set, so the tuple is narrowed explicitly.
        frames, fps = cast(
            "tuple[list[Image.Image], float]",
            diffusers.utils.load_video(str(temp_path), return_fps=True),  # type: ignore[reportPrivateImportUsage]
        )
        audio, sample_rate = _load_soundtrack(temp_path)
    finally:
        temp_path.unlink(missing_ok=True)

    if not frames:
        return VideoMedia(
            frames=frames, source_shape=(1, 3, 0, 0, 0), audio=None, audio_sample_rate=None, fps=float(fps)
        )
    height, width = frames[0].height, frames[0].width
    return VideoMedia(
        frames=frames,
        source_shape=(1, 3, len(frames), height, width),
        audio=audio,
        audio_sample_rate=sample_rate,
        fps=float(fps),
    )


def _load_soundtrack(video_path: Path) -> tuple[torch.Tensor | None, int | None]:
    """Decode a video's soundtrack as a float32 `(channels, samples)` waveform and its sample rate."""
    import av
    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3AudioReference

    with av.open(str(video_path)) as container:
        has_audio = bool(container.streams.audio)
    if not has_audio:
        return None, None

    reference = MiniMaxH3AudioReference.from_file(video_path)
    return reference.audio.to(dtype=torch.float32).contiguous(), reference.sample_rate


def load_video_frames_from_url_artifact(video_url_artifact: VideoUrlArtifact) -> list:
    """Load a VideoUrlArtifact as a list of PIL frames without decoding its soundtrack.

    Resolves Griptape path macros (e.g. `{inputs}/foo.mp4`) and HTTP URLs via
    `File(...)`, then decodes frames with `diffusers.utils.load_video`.
    """
    import diffusers.utils  # type: ignore[reportMissingImports]

    temp_path = download_video_to_temp_file(video_url_artifact)
    try:
        return diffusers.utils.load_video(str(temp_path))  # type: ignore[reportPrivateImportUsage]
    finally:
        temp_path.unlink(missing_ok=True)


def get_video_fps(video_path: Path, default_fps: float = 30.0) -> float:
    """Get the FPS (frames per second) of a video file using OpenCV.

    Args:
        video_path: Path to the video file
        default_fps: Default FPS to return if unable to determine from video

    Returns:
        The video's FPS, or default_fps if unable to determine
    """
    import cv2  # type: ignore[reportMissingImports]

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return default_fps

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            return default_fps
        return fps
    finally:
        cap.release()
