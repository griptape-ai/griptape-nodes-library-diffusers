from types import SimpleNamespace

import pytest

from modular_diffusion_nodes_library.latent_pipeline_drivers import ltx2
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    ConditioningInputValue,
    MediaGenConditioningPayload,
)
from modular_diffusion_nodes_library.runtime_parameters.ltx2_runtime_parameters import LTX2PipelineRuntimeParameters
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode


def _payload(mode: ConditioningMode, artifact: object, strength: float = 1.0) -> MediaGenConditioningPayload:
    return MediaGenConditioningPayload(
        mode=mode,
        entries=(
            ConditioningInputValue(
                artifact=artifact,  # type: ignore[arg-type]
                frame_index=0,
                strength=strength,
                kind=mode.value,
            ),
        ),
    )


def test_ltx2_reference_conditions_socket_accepts_images_and_videos() -> None:
    parameters = LTX2PipelineRuntimeParameters(object(), {})  # type: ignore[arg-type]

    assert parameters._reference_conditions_param._get_input_types() == [
        "media_gen_conditioning",
        "ImageUrlArtifact",
        "VideoUrlArtifact",
    ]


@pytest.mark.parametrize(
    ("mode", "resolver_name", "resolved_frames"),
    [
        (ConditioningMode.IMAGE, "resolve_conditioning_image", "image-frame"),
        (ConditioningMode.VIDEO, "resolve_conditioning_video", ["video-frame"]),
    ],
)
def test_ic_lora_reference_conditions_accept_image_and_video(
    monkeypatch: pytest.MonkeyPatch,
    mode: ConditioningMode,
    resolver_name: str,
    resolved_frames: str | list[str],
) -> None:
    artifact = object()
    monkeypatch.setattr(ltx2, resolver_name, lambda received: resolved_frames)

    conditions = ltx2.LTX2PipelineDriver._build_ic_reference_conditions(None, [_payload(mode, artifact, 0.4)])

    assert len(conditions) == 1
    assert conditions[0].frames == resolved_frames
    assert conditions[0].strength == 0.4


@pytest.mark.parametrize(
    ("mode", "resolver_name", "resolved_frames", "expected_frames"),
    [
        (ConditioningMode.IMAGE, "resolve_conditioning_image", "image-frame", "resized-image-frame"),
        (ConditioningMode.VIDEO, "resolve_conditioning_video", ["video-frame"], ["resized-video-frame"]),
    ],
)
def test_hdr_reference_conditions_accept_image_and_video(
    monkeypatch: pytest.MonkeyPatch,
    mode: ConditioningMode,
    resolver_name: str,
    resolved_frames: str | list[str],
    expected_frames: str | list[str],
) -> None:
    artifact = object()
    monkeypatch.setattr(ltx2, resolver_name, lambda received: resolved_frames)
    monkeypatch.setattr(
        ltx2,
        "resize_frames_scale_to_fill",
        lambda frames, target_height, target_width: [f"resized-{frame}" for frame in frames],
    )

    conditions = ltx2.LTX2PipelineDriver._build_hdr_reference_conditions(
        None, [_payload(mode, artifact, 0.7)], 720, 1280
    )

    assert len(conditions) == 1
    assert conditions[0].frames == expected_frames
    assert conditions[0].strength == 0.7


@pytest.mark.parametrize(
    "builder",
    [
        ltx2.LTX2PipelineDriver._build_ic_reference_conditions,
        ltx2.LTX2PipelineDriver._build_hdr_reference_conditions,
    ],
)
def test_reference_conditions_reject_unsupported_modes(builder: object) -> None:
    payload = SimpleNamespace(mode=SimpleNamespace(value="audio"), entries=())

    with pytest.raises(ValueError, match="mode 'audio' is unsupported"):
        if builder is ltx2.LTX2PipelineDriver._build_hdr_reference_conditions:
            builder(None, [payload], 720, 1280)  # type: ignore[operator]
        else:
            builder(None, [payload])  # type: ignore[operator]
