"""Core maths for masked denoising: redo only part of an image.

A mask says how strongly each part of the image may change. While the image is being
generated, parts the mask protects are reset to the original image (with a little noise
added to match the current step), so they stay close to it. Parts with a higher mask value
are released earlier and get more time to change, so they are redone more freely.

This is the Differential Diffusion technique (https://arxiv.org/abs/2306.00950).

This module holds the per-step blend and the checks that reject settings it cannot honour.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]


def validate_masked_denoising_options(*, scheduler: object, kwargs: Mapping[str, object]) -> None:
    """Reject unsupported options, including schedulers with order other than 1."""
    unsupported_options = [
        name for name in ("timesteps", "sigmas", "denoising_start", "denoising_end") if name in kwargs
    ]
    if unsupported_options:
        raise ValueError(
            f"Attempted to prepare masked denoising. Failed with options {unsupported_options} because "
            "masked-latent denoising requires the pipeline's standard inference timestep schedule."
        )
    strength = kwargs.get("strength", 1.0)
    if strength != 1.0:
        raise ValueError(
            f"Attempted to prepare masked denoising. Failed with strength={strength!r} because masked-latent "
            "denoising requires the full inference schedule (strength=1.0). Use start_step to select the "
            "schedule window instead."
        )
    if "generator" in kwargs:
        raise ValueError(
            "Attempted to prepare masked denoising. Failed with a separate 'generator' option because "
            "masked-latent denoising uses generator_state."
        )

    scheduler_order = getattr(scheduler, "order", 1)
    if scheduler_order != 1:
        raise NotImplementedError(
            f"Attempted to prepare masked denoising. Failed with scheduler '{type(scheduler).__name__}' of "
            f"order={scheduler_order} because masked-latent denoising requires a scheduler with order=1."
        )

    if "callback_on_step_end_tensor_inputs" in kwargs:
        tensor_inputs = kwargs["callback_on_step_end_tensor_inputs"]
        if not isinstance(tensor_inputs, (list, tuple)) or "latents" not in tensor_inputs:
            raise ValueError(
                f"Attempted to prepare masked denoising. Failed with callback_on_step_end_tensor_inputs="
                f"{tensor_inputs!r} because it must include 'latents'."
            )


def validate_schedule_length(*, scheduler: object, num_inference_steps: int) -> None:
    """Reject schedulers whose schedule does not hold exactly one timestep per inference step."""
    config = getattr(scheduler, "config", None)
    if config is not None and config.get("use_dynamic_shifting", False):
        return

    scheduler.set_timesteps(num_inference_steps=num_inference_steps, device="cpu")  # type: ignore[attr-defined]
    schedule_length = len(scheduler.timesteps)  # type: ignore[attr-defined]
    if schedule_length != num_inference_steps:
        raise ValueError(
            f"Attempted to prepare masked denoising. Failed with scheduler '{type(scheduler).__name__}' producing "
            f"{schedule_length} timesteps for num_inference_steps={num_inference_steps} because masked-latent "
            "denoising requires exactly one timestep per inference step."
        )


def _validate_matching_shapes(*, latents: torch.Tensor, source: torch.Tensor, mask: torch.Tensor) -> None:
    if source.shape != latents.shape or mask.shape != latents.shape:
        raise ValueError(
            f"Attempted to apply a masked denoising update. Failed with latents={tuple(latents.shape)}, "
            f"source={tuple(source.shape)}, mask={tuple(mask.shape)} because the shapes must match."
        )


def blend_differential_diffusion_latents(
    *,
    latents: torch.Tensor,
    source: torch.Tensor,
    mask: torch.Tensor,
    absolute_step: int,
    num_inference_steps: int,
) -> torch.Tensor:
    """Apply one Differential Diffusion mask update to the denoising latents.

    At each schedule step, mask values at or above the step's release threshold
    keep the model's denoised latent values. Other values are re-anchored to the
    source latent noised for the next schedule step. The mask and latent tensors
    must have matching shapes.

    This follows *Differential Diffusion: Giving Each Pixel Its Strength* by
    Eran Levin and Ohad Fried: https://arxiv.org/abs/2306.00950.

    Args:
        latents: Latents produced by the current denoising step.
        source: Source latent prepared for the next schedule step.
        mask: Per-latent release strengths, in the same layout as ``latents``.
        absolute_step: Zero-based index of the current step in the full schedule.
        num_inference_steps: Number of steps in the full inference schedule.

    Returns:
        The latents with unreleased regions restored to their scheduled source.

    Raises:
        ValueError: If the schedule indices are invalid or tensor shapes differ.
    """
    import torch  # type: ignore[reportMissingImports]

    if num_inference_steps < 1:
        raise ValueError(
            f"Attempted to apply a masked denoising update. Failed with num_inference_steps={num_inference_steps} "
            "because it must be a positive integer."
        )
    if absolute_step < 0 or absolute_step >= num_inference_steps:
        raise ValueError(
            f"Attempted to apply a masked denoising update. Failed with absolute_step={absolute_step} "
            f"because it must be in [0, {num_inference_steps})."
        )

    _validate_matching_shapes(latents=latents, source=source, mask=mask)
    source = source.to(device=latents.device, dtype=latents.dtype)

    release_threshold = 1.0 - ((absolute_step + 1) / num_inference_steps)
    released = mask.to(device=latents.device) >= release_threshold
    return torch.where(released, latents, source)
