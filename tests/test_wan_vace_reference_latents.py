"""Reference images shift the VACE latent frame count, and the driver has to track it.

VACE prepends one latent frame per reference image to the control branch. The pipeline
compensates when it sizes the noise latent itself, but this driver always hands it an
explicit ``latents`` tensor, which short-circuits that path — so the compensation has to
happen here or the transformer computes a negative control-padding width and the run dies
with ``Trying to create tensor with negative dimension``.
"""

from __future__ import annotations

import torch

from modular_diffusion_nodes_library.latent_pipeline_drivers.wan_vace import WanVaceLatentPipelineDriver

# 49 frames at 1280x720 through a WAN VAE: 13 latent frames, 45x80 latent grid.
_LATENT_FRAMES = 13
_LATENT_HEIGHT = 45
_LATENT_WIDTH = 80
_SOURCE_SHAPE = (1, 3, 49, 720, 1280)


def _driver() -> WanVaceLatentPipelineDriver:
    """Build a driver without loading pipeline weights.

    The latent-frame bookkeeping under test reads only driver state, so the pipe is
    never touched.
    """
    return WanVaceLatentPipelineDriver.__new__(WanVaceLatentPipelineDriver)


def _noise_latent(num_frames: int = _LATENT_FRAMES) -> torch.Tensor:
    return torch.zeros(1, 16, num_frames, _LATENT_HEIGHT, _LATENT_WIDTH)


def test_input_latent_unchanged_without_reference_images() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 0
    driver._reference_generator_state = None

    latents = _noise_latent()
    prepared = driver.prepare_input_latent(latents, _SOURCE_SHAPE)

    assert prepared is latents


def test_input_latent_gains_one_frame_per_reference_image() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 1
    driver._reference_generator_state = None

    prepared = driver.prepare_input_latent(_noise_latent(), _SOURCE_SHAPE)

    # Without the extra frame the control branch is 3600 tokens longer than the noise
    # latent, which is what produced the reported -3600 crash.
    assert prepared.shape[2] == _LATENT_FRAMES + 1


def test_input_latent_scales_with_reference_count() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 3
    driver._reference_generator_state = None

    prepared = driver.prepare_input_latent(_noise_latent(), _SOURCE_SHAPE)

    assert prepared.shape[2] == _LATENT_FRAMES + 3


def test_output_latent_drops_the_reference_frames() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 2
    driver._reference_generator_state = None

    trimmed = driver.prepare_output_latent(_noise_latent(_LATENT_FRAMES + 2), _SOURCE_SHAPE)

    # Untrimmed, the decoded video carries junk leading frames and runs long.
    assert trimmed.shape[2] == _LATENT_FRAMES


def test_reference_frames_round_trip_to_the_source_frame_count() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 1
    driver._reference_generator_state = None

    latents = _noise_latent()
    prepared = driver.prepare_input_latent(latents, _SOURCE_SHAPE)
    trimmed = driver.prepare_output_latent(prepared, _SOURCE_SHAPE)

    assert trimmed.shape == latents.shape
    # The trim must return the original frames, not the prepended noise.
    assert torch.equal(trimmed, latents)


def test_output_latent_unchanged_without_reference_images() -> None:
    driver = _driver()
    driver._num_reference_latent_frames = 0
    driver._reference_generator_state = None

    latents = _noise_latent()

    assert driver.prepare_output_latent(latents, _SOURCE_SHAPE) is latents
