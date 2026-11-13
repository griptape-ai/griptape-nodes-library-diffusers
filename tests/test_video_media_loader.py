from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

pytest.importorskip("torch", reason="Uses the MiniMax-H3 media reference; run `make test/exec`.")
pytest.importorskip("diffusers", reason="Uses the MiniMax-H3 media reference; run `make test/exec`.")
pytest.importorskip("av", reason="Uses the MiniMax-H3 media reference; run `make test/exec`.")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from modular_diffusion_nodes_library.utils import video_utils  # noqa: E402


def test_video_media_loader_returns_frames_fps_and_optional_audio(monkeypatch: pytest.MonkeyPatch) -> None:
    frames = np.zeros((3, 8, 10, 3), dtype=np.uint8)
    waveform = torch.ones((2, 3200), dtype=torch.float32)
    reference = SimpleNamespace(frames=frames, fps=24.0, audio=waveform, sample_rate=32000)
    temp_path = Path("video-test.mp4")
    deleted: list[Path] = []
    monkeypatch.setattr(video_utils, "download_video_to_temp_file", lambda _artifact: temp_path)
    monkeypatch.setattr(Path, "unlink", lambda self, **_kwargs: deleted.append(self))
    monkeypatch.setattr(
        "diffusers.modular_pipelines.minimax_h3.references.MiniMaxH3VideoReference.from_file",
        lambda _path: reference,
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
    reference = SimpleNamespace(frames=np.zeros((2, 8, 10, 3), dtype=np.uint8), fps=15.0, audio=None, sample_rate=None)
    monkeypatch.setattr(video_utils, "download_video_to_temp_file", lambda _artifact: Path("silent.mp4"))
    monkeypatch.setattr(Path, "unlink", lambda self, **_kwargs: None)
    monkeypatch.setattr(
        "diffusers.modular_pipelines.minimax_h3.references.MiniMaxH3VideoReference.from_file",
        lambda _path: reference,
    )

    media = video_utils.load_video_media_from_url_artifact(VideoUrlArtifact(value="silent.mp4"))

    assert media.audio is None
    assert media.audio_sample_rate is None
    assert media.fps == 15.0
