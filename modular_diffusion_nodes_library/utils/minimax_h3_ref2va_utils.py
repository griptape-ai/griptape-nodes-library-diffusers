"""Lightweight helpers shared by MiniMax-H3 ref2va validation and execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from modular_diffusion_nodes_library.utils.video_utils import download_video_to_temp_file


@dataclass(frozen=True)
class VideoTrimPlan:
    """Per-video kept durations; ``None`` means the video is dropped."""

    kept_seconds: tuple[float | None, ...]
    warning_message: str | None


@dataclass(frozen=True)
class LoadedReferenceVideo:
    """A decoded reference video with its source duration."""

    reference: Any
    duration_seconds: float


def plan_reference_video_trim(durations: list[float]) -> VideoTrimPlan:
    """Trim reference videos to a 15-second total, dropping clips from the end when needed."""
    for index, duration in enumerate(durations, start=1):
        if duration < 2.0:
            raise ValueError(
                f"Attempted to validate MiniMax-H3 ref2va references. Failed because Video {index} is "
                f"{duration:.1f} s; each reference video must be at least 2.0 s."
            )

    kept_seconds: list[float | None] = list(durations)
    total_duration = sum(durations)
    if total_duration <= 15.0:
        return VideoTrimPlan(kept_seconds=tuple(kept_seconds), warning_message=None)

    actions: list[str] = []
    while total_duration > 15.0:
        overflow = total_duration - 15.0
        last_index = next(index for index in range(len(kept_seconds) - 1, -1, -1) if kept_seconds[index] is not None)
        last_duration = kept_seconds[last_index]
        assert last_duration is not None

        if last_duration - overflow >= 2.0:
            trimmed_duration = last_duration - overflow
            kept_seconds[last_index] = trimmed_duration
            actions.append(f"Video {last_index + 1} trimmed from {last_duration:.1f} s to {trimmed_duration:.1f} s.")
            total_duration = 15.0
        else:
            kept_seconds[last_index] = None
            actions.append(f"Video {last_index + 1} ({last_duration:.1f} s) dropped.")
            total_duration -= last_duration

    warning_message = f"Total reference video duration is {sum(durations):.1f} s, over the 15 s limit. " + " ".join(
        actions
    )
    return VideoTrimPlan(kept_seconds=tuple(kept_seconds), warning_message=warning_message)


def get_reference_video_duration_seconds(video_artifact: VideoUrlArtifact) -> float:
    """Read a conditioning video's duration from video-stream metadata only."""
    import av

    temporary_path: Path = download_video_to_temp_file(video_artifact)
    try:
        with av.open(str(temporary_path)) as container:
            if not container.streams.video:
                raise ValueError(
                    f"Attempted to read MiniMax-H3 reference video duration. Failed because "
                    f"'{video_artifact.value}' has no video stream."
                )
            stream = container.streams.video[0]
            frame_rate = stream.average_rate or stream.guessed_rate
            if stream.frames > 0 and frame_rate is not None and float(frame_rate) > 0:
                duration = stream.frames / float(frame_rate)
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            else:
                raise ValueError(
                    f"Attempted to read MiniMax-H3 reference video duration. Failed because "
                    f"'{video_artifact.value}' has no usable duration metadata."
                )
    finally:
        temporary_path.unlink(missing_ok=True)

    return duration


def load_reference_video(video_artifact: VideoUrlArtifact) -> LoadedReferenceVideo:
    """Decode a reference video from the resolved artifact, returning its Diffusers reference and duration."""
    import av
    from diffusers.modular_pipelines.minimax_h3.references import MiniMaxH3VideoReference

    temporary_path: Path = download_video_to_temp_file(video_artifact)
    try:
        with av.open(str(temporary_path)) as container:
            if not container.streams.video:
                raise ValueError(
                    f"Attempted to load MiniMax-H3 reference video. Failed because "
                    f"'{video_artifact.value}' has no video stream."
                )

            stream = container.streams.video[0]
            frame_rate = stream.average_rate or stream.guessed_rate
            if stream.frames > 0 and frame_rate is not None and float(frame_rate) > 0:
                duration = stream.frames / float(frame_rate)
            elif stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            else:
                raise ValueError(
                    f"Attempted to load MiniMax-H3 reference video. Failed because "
                    f"'{video_artifact.value}' has no usable duration metadata."
                )

        reference = MiniMaxH3VideoReference.from_file(temporary_path)
        reference.audio = None
        reference.sample_rate = None
    finally:
        temporary_path.unlink(missing_ok=True)

    return LoadedReferenceVideo(reference=reference, duration_seconds=duration)
