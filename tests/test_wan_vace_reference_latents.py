"""Reference images shift the VACE latent frame count, and the driver has to track it.

VACE prepends one latent frame per reference image to the control branch. The pipeline
compensates when it sizes the noise latent itself, but this driver always hands it an
explicit ``latents`` tensor, which short-circuits that path — so the compensation has to
happen here or the transformer computes a negative control-padding width and the run dies
with ``Trying to create tensor with negative dimension``.

``_prepend_reference_noise`` is a module-level pure function (no ``self.pipe`` dependency),
so it's tested directly without constructing a driver. ``prepare_output_latent`` reads only
driver state (``_num_reference_latent_frames``), so it's tested against a bare, un-``__init__``'d
driver instance.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch", reason="Reads real torch tensors; run `make test/exec` for the execution environment.")

import PIL.Image  # noqa: E402
import torch  # noqa: E402
from griptape.artifacts import ImageUrlArtifact  # noqa: E402

from modular_diffusion_nodes_library.latent_pipeline_drivers.wan_vace import (
    WanVaceLatentPipelineDriver,
    _derive_mask_from_source_media,
    _payload_to_frames,
    _payload_to_reference_images,
    _prepend_reference_noise,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    ConditioningInputValue,
    MediaGenConditioningPayload,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode

pytest.importorskip(
    "torch",
    reason="Reads real diffusers classes; run `make test/exec` for the execution environment.",
)


# 49 frames at 1280x720 through a WAN VAE: 13 latent frames, 45x80 latent grid.
_LATENT_FRAMES = 13
_LATENT_HEIGHT = 45
_LATENT_WIDTH = 80
_SOURCE_SHAPE = (1, 3, 49, 720, 1280)

_RED = (255, 0, 0)
_BLACK = (0, 0, 0)


def _driver() -> WanVaceLatentPipelineDriver:
    """Build a driver without loading pipeline weights.

    The latent-frame bookkeeping under test reads only driver state, so the pipe is
    never touched.
    """
    return WanVaceLatentPipelineDriver.__new__(WanVaceLatentPipelineDriver)


def _noise_latent(num_frames: int = _LATENT_FRAMES) -> torch.Tensor:
    return torch.zeros(1, 16, num_frames, _LATENT_HEIGHT, _LATENT_WIDTH)


def _image_payload(frame_index: int, image: PIL.Image.Image, tmp_path: Path) -> MediaGenConditioningPayload:
    """Build an IMAGE-mode payload using a real ImageUrlArtifact backed by a temp file."""
    image_path = tmp_path / f"marker_{frame_index}.png"
    image.save(image_path)
    artifact = ImageUrlArtifact(str(image_path))
    entry = ConditioningInputValue(artifact=artifact, frame_index=frame_index, strength=1.0, kind="image")
    return MediaGenConditioningPayload(mode=ConditioningMode.IMAGE, entries=(entry,))


def test_prepend_reference_noise_unchanged_without_reference_images() -> None:
    latents = _noise_latent()

    prepended = _prepend_reference_noise(latents, 0, torch.Generator())

    assert prepended.shape == latents.shape


def test_prepend_reference_noise_adds_one_frame_per_reference_image() -> None:
    prepended = _prepend_reference_noise(_noise_latent(), 1, torch.Generator())

    # Without the extra frame the control branch is 3600 tokens longer than the noise
    # latent, which is what produced the reported -3600 crash.
    assert prepended.shape[2] == _LATENT_FRAMES + 1


def test_prepend_reference_noise_scales_with_reference_count() -> None:
    prepended = _prepend_reference_noise(_noise_latent(), 3, torch.Generator())

    assert prepended.shape[2] == _LATENT_FRAMES + 3


def test_prepend_reference_noise_keeps_base_latents_after_the_prefix() -> None:
    latents = _noise_latent()

    prepended = _prepend_reference_noise(latents, 2, torch.Generator())

    assert torch.equal(prepended[:, :, 2:], latents)


def test_output_latent_unchanged_without_reference_images() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 0

    latents = _noise_latent()

    assert driver.prepare_output_latent(latents, _SOURCE_SHAPE) is latents


def test_output_latent_drops_the_reference_frames() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 2

    trimmed = driver.prepare_output_latent(_noise_latent(_LATENT_FRAMES + 2), _SOURCE_SHAPE)

    # Untrimmed, the decoded video carries junk leading frames and runs long.
    assert trimmed.shape[2] == _LATENT_FRAMES


def test_output_latent_scales_with_reference_count() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 3

    trimmed = driver.prepare_output_latent(_noise_latent(_LATENT_FRAMES + 3), _SOURCE_SHAPE)

    assert trimmed.shape[2] == _LATENT_FRAMES


def test_reference_frames_round_trip_to_the_source_frame_count() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 1

    latents = _noise_latent()
    prepended = _prepend_reference_noise(latents, 1, torch.Generator())
    trimmed = driver.prepare_output_latent(prepended, _SOURCE_SHAPE)

    assert trimmed.shape == latents.shape
    # The trim must return the original frames, not the prepended noise.
    assert torch.equal(trimmed, latents)


def test_payload_to_frames_places_image_at_frame_index_and_fills_the_rest(tmp_path: Path) -> None:
    marker = PIL.Image.new("RGB", (10, 10), _RED)

    frames = _payload_to_frames(_image_payload(3, marker, tmp_path), num_frames=5, height=10, width=10, fill=_BLACK)

    assert len(frames) == 5
    assert frames[3].getpixel((0, 0)) == _RED
    assert frames[0].getpixel((0, 0)) == _BLACK


def test_payload_to_frames_rejects_out_of_range_frame_index(tmp_path: Path) -> None:
    marker = PIL.Image.new("RGB", (10, 10), _RED)

    with pytest.raises(ValueError, match="out of range"):
        _payload_to_frames(_image_payload(10, marker, tmp_path), num_frames=5, height=10, width=10, fill=_BLACK)


def test_derive_mask_from_source_media_marks_placed_frame_black(tmp_path: Path) -> None:
    marker = PIL.Image.new("RGB", (10, 10), _RED)

    mask = _derive_mask_from_source_media(_image_payload(2, marker, tmp_path), num_frames=4, height=10, width=10)

    assert mask is not None
    # Black (0) marks the frame to preserve; white (255) marks frames to generate.
    assert mask[2].getpixel((0, 0)) == 0
    assert mask[0].getpixel((0, 0)) == 255


def test_payload_to_reference_images_returns_image_entries_directly(tmp_path: Path) -> None:
    marker = PIL.Image.new("RGB", (10, 10), _RED)

    reference_images = _payload_to_reference_images(_image_payload(0, marker, tmp_path))

    assert len(reference_images) == 1
    assert reference_images[0].getpixel((0, 0)) == _RED
