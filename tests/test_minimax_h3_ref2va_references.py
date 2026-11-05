from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

pytest.importorskip("torch", reason="Uses the real MiniMax-H3 reference dataclasses; run `make test/exec`.")
pytest.importorskip("diffusers", reason="Uses the real MiniMax-H3 reference dataclasses; run `make test/exec`.")

from diffusers.modular_pipelines.minimax_h3.references import (  # noqa: E402
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)
from PIL import Image  # noqa: E402

from modular_diffusion_nodes_library.latent_pipeline_drivers.minimax_h3_ref2va import (  # noqa: E402
    MiniMaxH3Ref2VALatentPipelineDriver,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (  # noqa: E402
    ConditioningInputValue,
    MediaGenConditioningPayload,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode  # noqa: E402
from modular_diffusion_nodes_library.utils.minimax_h3_ref2va_utils import LoadedReferenceVideo  # noqa: E402


@pytest.fixture
def fake_video_loader(monkeypatch: pytest.MonkeyPatch) -> list[Path]:
    import diffusers.modular_pipelines.minimax_h3.references as references_module

    loaded_paths: list[Path] = []

    def from_file(cls: type[MiniMaxH3VideoReference], media: str | Path) -> MiniMaxH3VideoReference:
        path = Path(media)
        loaded_paths.append(path)
        frame_count = 200 if path.name == "long.mp4" else 30
        return MiniMaxH3VideoReference(
            frames=[Image.new("RGB", (2, 2), (index, 0, 0)) for index in range(frame_count)],
            fps=10.0,
            audio=None,
        )

    monkeypatch.setattr(references_module.MiniMaxH3VideoReference, "from_file", classmethod(from_file))
    return loaded_paths


def _image_payload(path: str) -> MediaGenConditioningPayload:
    return MediaGenConditioningPayload(
        mode=ConditioningMode.IMAGE,
        entries=(
            ConditioningInputValue(
                artifact=ImageUrlArtifact(value=path),
                frame_index=0,
                strength=1.0,
                kind="image",
            ),
        ),
    )


def _video_payload(path: str) -> MediaGenConditioningPayload:
    return MediaGenConditioningPayload(
        mode=ConditioningMode.VIDEO,
        entries=(
            ConditioningInputValue(
                artifact=VideoUrlArtifact(value=path),
                frame_index=0,
                strength=1.0,
                kind="video",
            ),
        ),
    )


def test_reference_builder_orders_images_first_then_videos(
    monkeypatch: pytest.MonkeyPatch,
    fake_video_loader: list[Path],
) -> None:
    from modular_diffusion_nodes_library.latent_pipeline_drivers import minimax_h3_ref2va as driver_module

    video_paths: list[str] = []

    def load_video(artifact: Any) -> LoadedReferenceVideo:
        video_paths.append(artifact.value)
        frames = [Image.new("RGB", (2, 2)) for _ in range(30)]
        reference = MiniMaxH3VideoReference(frames=frames, fps=10.0, audio=torch.ones(1), sample_rate=44100)
        return LoadedReferenceVideo(reference=reference, duration_seconds=3.0)

    monkeypatch.setattr(driver_module, "get_reference_video_duration_seconds", lambda artifact: 3.0)
    monkeypatch.setattr(driver_module, "load_reference_video", load_video)
    monkeypatch.setattr(driver_module, "resolve_conditioning_image", lambda artifact: Image.new("RGB", (2, 2)))

    references = MiniMaxH3Ref2VALatentPipelineDriver._build_references(
        [
            _video_payload("video-a.mp4"),
            _image_payload("image.png"),
            _video_payload("video-b.mp4"),
            _video_payload("video-c.mp4"),
        ]
    )

    assert isinstance(references[0], MiniMaxH3ImageReference)
    assert all(isinstance(reference, MiniMaxH3VideoReference) for reference in references[1:])
    assert video_paths == ["video-a.mp4", "video-b.mp4", "video-c.mp4"]
    assert fake_video_loader == []
    assert [reference.fps for reference in references[1:]] == [10.0, 10.0, 10.0]
    assert all(reference.audio is None for reference in references[1:])
    assert all(reference.sample_rate is None for reference in references[1:])


def test_reference_builder_slices_trimmed_frames(
    monkeypatch: pytest.MonkeyPatch,
    fake_video_loader: list[Path],
) -> None:
    from modular_diffusion_nodes_library.latent_pipeline_drivers import minimax_h3_ref2va as driver_module

    def load_video(artifact: Any) -> LoadedReferenceVideo:
        frames = [Image.new("RGB", (2, 2)) for _ in range(200)]
        reference = MiniMaxH3VideoReference(frames=frames, fps=10.0, audio=torch.ones(1), sample_rate=44100)
        return LoadedReferenceVideo(reference=reference, duration_seconds=20.0)

    monkeypatch.setattr(driver_module, "get_reference_video_duration_seconds", lambda artifact: 20.0)
    monkeypatch.setattr(driver_module, "load_reference_video", load_video)

    references = MiniMaxH3Ref2VALatentPipelineDriver._build_references([_video_payload("long.mp4")])

    assert len(references) == 1
    assert len(references[0].frames) == 150
    assert references[0].fps == 10.0
    assert references[0].audio is None
    assert references[0].sample_rate is None
    assert fake_video_loader == []
