from __future__ import annotations

from typing import Any, cast

import pytest
from griptape.artifacts import ImageUrlArtifact
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    ConditioningInputValue,
    MediaGenConditioningPayload,
)
from modular_diffusion_nodes_library.runtime_parameters.minimax_h3_ref2va_runtime_parameters import (
    MiniMaxH3Ref2VAPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode


class _ConditioningInput:
    def __init__(self, payloads: list[MediaGenConditioningPayload]):
        self._payloads = payloads

    def validate_in_execution_environment(self) -> None:
        pass

    def get_payloads(self) -> list[MediaGenConditioningPayload]:
        return self._payloads


class _RuntimeParameters:
    def __new__(cls, payloads: list[MediaGenConditioningPayload]) -> MiniMaxH3Ref2VAPipelineRuntimeParameters:
        instance = object.__new__(MiniMaxH3Ref2VAPipelineRuntimeParameters)
        instance._node = cast(Any, type("Node", (), {"name": "Generate Media Latents"})())
        instance._references_param = cast(Any, _ConditioningInput(payloads))
        instance._duration_warning = cast(Any, type("Warning", (), {"value": "", "hide": True})())
        return instance


def _image_entry(index: int) -> ConditioningInputValue:
    return ConditioningInputValue(
        artifact=ImageUrlArtifact(value=f"image-{index}.png"),
        frame_index=index,
        strength=1.0,
        kind="image",
    )


def _video_entry(index: int) -> ConditioningInputValue:
    return ConditioningInputValue(
        artifact=VideoUrlArtifact(value=f"video-{index}.mp4"),
        frame_index=0,
        strength=1.0,
        kind="video",
    )


def _image_payload(count: int) -> MediaGenConditioningPayload:
    return MediaGenConditioningPayload(
        mode=ConditioningMode.IMAGE,
        entries=tuple(_image_entry(index) for index in range(count)),
    )


def _video_payload(index: int) -> MediaGenConditioningPayload:
    return MediaGenConditioningPayload(mode=ConditioningMode.VIDEO, entries=(_video_entry(index),))


def _install_duration_lookup(monkeypatch: pytest.MonkeyPatch, durations: dict[str, float]) -> None:
    from modular_diffusion_nodes_library.runtime_parameters import minimax_h3_ref2va_runtime_parameters as module

    monkeypatch.setattr(module, "get_reference_video_duration_seconds", lambda artifact: durations[artifact.value])


@pytest.mark.parametrize(
    ("payloads", "durations", "error"),
    [
        ([], {}, "at least one image or video reference is required"),
        ([_image_payload(10)], {}, "10 images exceed the limit of 9"),
        (
            [_video_payload(1), _video_payload(2), _video_payload(3), _video_payload(4)],
            {},
            "4 videos exceed the limit of 3",
        ),
        ([_video_payload(1)], {"video-1.mp4": 1.5}, "Video 1 is 1.5 s"),
    ],
)
def test_ref2va_validation_rejects_invalid_reference_sets(
    monkeypatch: pytest.MonkeyPatch,
    payloads: list[MediaGenConditioningPayload],
    durations: dict[str, float],
    error: str,
) -> None:
    _install_duration_lookup(monkeypatch, durations)
    params = _RuntimeParameters(payloads)

    errors = params.validate_in_execution_environment()

    assert errors is not None
    assert error in str(errors[0])
    assert params._duration_warning.hide


def test_ref2va_validation_sets_and_clears_duration_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    durations: dict[str, float] = {"video-1.mp4": 10.0, "video-2.mp4": 10.0}
    _install_duration_lookup(monkeypatch, durations)
    params = _RuntimeParameters([_video_payload(1), _video_payload(2)])

    assert params.validate_in_execution_environment() is None
    assert not params._duration_warning.hide
    assert "Video 2 trimmed from 10.0 s to 5.0 s" in params._duration_warning.value

    durations.update({"video-1.mp4": 5.0, "video-2.mp4": 5.0})
    assert params.validate_in_execution_environment() is None
    assert params._duration_warning.hide
    assert params._duration_warning.value == ""


def test_ref2va_runtime_parameters_configures_reference_surface() -> None:
    from modular_diffusion_nodes_library.runtime_parameters.minimax_h3_ref2va_runtime_parameters import (
        MiniMaxH3Ref2VAPipelineRuntimeParameters,
    )

    conditioning_config = MiniMaxH3Ref2VAPipelineRuntimeParameters.CONDITIONING_CONFIG

    assert conditioning_config is not None
    assert conditioning_config.image is not None
    image_config = cast(Any, conditioning_config.image)
    assert image_config.min_count == 1
    assert image_config.max_count == 9
    assert not image_config.expose_strength
    assert not image_config.expose_frame_index
    assert conditioning_config.video is not None
