from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

pytest.importorskip("torch", reason="Uses the MiniMax-H3 media reference; run `make test/exec`.")
pytest.importorskip("diffusers", reason="Uses the MiniMax-H3 media reference; run `make test/exec`.")
pytest.importorskip("av", reason="Uses the MiniMax-H3 media reference; run `make test/exec`.")

import torch  # noqa: E402
from PIL import Image  # noqa: E402

from modular_diffusion_nodes_library.utils import video_utils  # noqa: E402


class _FakeContainer:
    """Stands in for an `av` container, exposing only the audio streams the loader inspects."""

    def __init__(self, audio_streams: list[object]) -> None:
        self.streams = SimpleNamespace(audio=audio_streams)

    def __enter__(self) -> _FakeContainer:
        return self

    def __exit__(self, *_exc_info: object) -> None:
        return None


def _frames(count: int) -> list[Image.Image]:
    return [Image.new("RGB", (10, 8)) for _ in range(count)]


def test_video_media_loader_returns_frames_fps_and_optional_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    waveform = torch.ones((2, 3200), dtype=torch.float32)
    temp_path = Path("video-test.mp4")
    deleted: list[Path] = []
    monkeypatch.setattr(video_utils, "download_video_to_temp_file", lambda _artifact: temp_path)
    monkeypatch.setattr(Path, "unlink", lambda self, **_kwargs: deleted.append(self))
    monkeypatch.setattr("diffusers.utils.load_video", lambda _path, return_fps: (_frames(3), 24.0))
    monkeypatch.setattr("av.open", lambda _path: _FakeContainer(audio_streams=[object()]))
    monkeypatch.setattr(
        "diffusers.modular_pipelines.minimax_h3.references.MiniMaxH3AudioReference.from_file",
        lambda _path: SimpleNamespace(audio=waveform, sample_rate=32000),
    )

    media = video_utils.load_video_media_from_url_artifact(VideoUrlArtifact(value="video.mp4"))

    assert media.source_shape == (1, 3, 3, 8, 10)
    assert len(media.frames) == 3
    assert all(isinstance(frame, Image.Image) and frame.mode == "RGB" for frame in media.frames)
    assert media.audio is not None and torch.equal(media.audio, waveform)
    assert media.audio_sample_rate == 32000
    assert media.fps == 24.0
    assert deleted == [temp_path]


def test_video_media_loader_accepts_missing_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(video_utils, "download_video_to_temp_file", lambda _artifact: Path("silent.mp4"))
    monkeypatch.setattr(Path, "unlink", lambda self, **_kwargs: None)
    monkeypatch.setattr("diffusers.utils.load_video", lambda _path, return_fps: (_frames(2), 15.0))
    monkeypatch.setattr("av.open", lambda _path: _FakeContainer(audio_streams=[]))

    media = video_utils.load_video_media_from_url_artifact(VideoUrlArtifact(value="silent.mp4"))

    assert media.audio is None
    assert media.audio_sample_rate is None
    assert media.fps == 15.0
