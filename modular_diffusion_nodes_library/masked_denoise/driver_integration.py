"""Driver-side integration for Differential Diffusion masked denoising.

This module wires the blend math in ``misc/masked_denoising.py`` into the two
denoising paths drivers use:

- ``make_masked_denoising_callback``: a step-end callback for classic ``DiffusionPipeline`` drivers.
- ``make_masked_denoise_block``: a modular block for modular-pipeline drivers such as FLUX.3.

``PreparedMaskedDenoising`` carries what a driver's ``_prepare_masked_denoising`` produces
(initial latents, mask, per-step noised sources, generator state) into either path.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.masked_latent_artifact import MaskedLatentArtifact
from modular_diffusion_nodes_library.misc.masked_denoising import (
    blend_differential_diffusion_latents,
    validate_masked_denoising_options,
)
from modular_diffusion_nodes_library.utils.torch_utils import no_grad

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
        ModularPipeline,
        ModularPipelineBlocks,
    )
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState

type StepEndCallback = Callable[[DiffusionPipeline, int, torch.Tensor, dict[str, Any]], dict[str, Any]]


@dataclass(frozen=True)
class PreparedMaskedDenoising:
    """Inputs prepared by a driver for the shared denoising call."""

    latents: torch.Tensor
    mask: torch.Tensor
    noised_sources: Mapping[int, torch.Tensor]
    generator_state: GeneratorState


def validate_masked_denoising_inputs(
    *,
    latent: LatentArtifact | InpaintMaskArtifact,
    masked_latent: MaskedLatentArtifact,
    scheduler: object,
    kwargs: Mapping[str, object],
) -> LatentArtifact:
    """Check the artifacts and pipeline options a masked denoise run needs, and return the encoded source latent."""
    if not isinstance(masked_latent, MaskedLatentArtifact):
        raise TypeError(
            f"Attempted to prepare masked denoising. Failed with _masked_latent_artifact of type "
            f"'{type(masked_latent).__name__}' because it must be a MaskedLatentArtifact."
        )
    if isinstance(latent, InpaintMaskArtifact):
        raise ValueError(
            "Attempted to prepare masked denoising. Failed with an InpaintMaskArtifact as the input latent "
            "because masked-latent denoising cannot be combined with inpainting."
        )
    validate_masked_denoising_options(scheduler=scheduler, kwargs=kwargs)

    source_latent = masked_latent.source_latent
    if source_latent is None:
        raise ValueError(
            "Attempted to prepare masked denoising. Failed with a MaskedLatentArtifact that has no source_latent "
            "because Set Latent Mask must be given an encoded latent."
        )
    if source_latent.source_shape != latent.source_shape:
        raise ValueError(
            f"Attempted to prepare masked denoising. Failed with masked source_shape {source_latent.source_shape} "
            f"and input latent source_shape {latent.source_shape} because the two must match."
        )

    return source_latent


def make_masked_denoising_callback(
    *,
    mask: torch.Tensor,
    noised_sources: Mapping[int, torch.Tensor],
    start_step: int,
    num_inference_steps: int,
    callback: StepEndCallback | None,
) -> StepEndCallback:
    """Build a step-end callback that applies the masked denoising update.

    Blends each step's latents with its prepared source using the mask, then
    invokes the optional callback. Requires ``latents`` in callback kwargs and a source
    for every step except the last of the full schedule, which overwrites nothing.

    Args:
        mask: Per-latent release strengths.
        noised_sources: Sources keyed by absolute step.
        start_step: Absolute index of callback step zero.
        num_inference_steps: Full schedule length.
        callback: Optional callback invoked after blending.

    Returns:
        A pipeline-compatible step-end callback.
    """
    if num_inference_steps < 1:
        raise ValueError(
            f"Attempted to build the masked denoising callback. Failed with num_inference_steps={num_inference_steps} "
            "because it must be a positive integer."
        )
    if start_step < 0:
        raise ValueError(
            f"Attempted to build the masked denoising callback. Failed with start_step={start_step} "
            "because it must be a non-negative integer."
        )

    import torch  # type: ignore[reportMissingImports]

    def masked_denoising_callback(
        pipe: DiffusionPipeline, step_index: int, timestep: torch.Tensor, callback_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        absolute_step = start_step + step_index
        # Nothing is overwritten after the final step, so it has no source.
        is_final_step = absolute_step == num_inference_steps - 1
        if not is_final_step and absolute_step not in noised_sources:
            raise ValueError(
                f"Attempted to apply masked denoising at step {absolute_step}. Failed with prepared sources for "
                f"steps {sorted(noised_sources)} because no source was prepared for this step."
            )

        latents = callback_kwargs.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                f"Attempted to apply masked denoising at step {absolute_step}. Failed with callback_kwargs['latents'] "
                f"of type '{type(latents).__name__}' because it must be a torch.Tensor."
            )

        if is_final_step:
            blended_latents = latents
        else:
            blended_latents = blend_differential_diffusion_latents(
                latents=latents,
                source=noised_sources[absolute_step],
                mask=mask,
                absolute_step=absolute_step,
                num_inference_steps=num_inference_steps,
            )

        updated_kwargs = dict(callback_kwargs)
        updated_kwargs["latents"] = blended_latents
        result = {}
        if callback is not None:
            callback_outputs = callback(pipe, step_index, timestep, updated_kwargs)
            if callback_outputs is not None:
                result = dict(callback_outputs)
        result.setdefault("latents", blended_latents)
        return result

    return masked_denoising_callback


def make_masked_denoise_block(
    *, model_name: str, masked_denoising: PreparedMaskedDenoising
) -> type[ModularPipelineBlocks]:
    """Build a modular block that applies one masked denoising update per step.

    The block reads ``latents`` and ``num_inference_steps`` from the enclosing loop's block state
    and the step's source from ``masked_denoising.noised_sources``, then blends them in place.
    Drivers list it after their denoiser and after-denoiser blocks.

    Args:
        model_name: Label used in error messages, e.g. ``"flux3"``.
        masked_denoising: The mask and per-step sources a driver prepared.

    Returns:
        A ``ModularPipelineBlocks`` subclass.
    """

    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
        BlockState,
        ModularPipelineBlocks,
    )
    from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
        InputParam,
        OutputParam,
    )

    block_model_name = model_name

    class MaskedDenoiseBlock(ModularPipelineBlocks):
        model_name = block_model_name

        @property
        def description(self) -> str:
            return f"Blend masked {model_name} latent tokens with their scheduled source after each denoising step."

        @property
        def inputs(self) -> list[InputParam]:
            return [
                InputParam("latents", required=True, type_hint=torch.Tensor),
                InputParam("num_inference_steps", required=True, type_hint=int),
            ]

        @property
        def intermediate_outputs(self) -> list[OutputParam]:
            return [OutputParam("latents", type_hint=torch.Tensor)]

        @no_grad
        def __call__(  # noqa: ARG002
            self, components: ModularPipeline, block_state: BlockState, i: int, t: torch.Tensor
        ) -> tuple[ModularPipeline, BlockState]:
            # Nothing is overwritten after the final step, so it has no source.
            num_inference_steps = block_state.num_inference_steps  # pyright: ignore[reportAttributeAccessIssue]
            is_final_step = i == num_inference_steps - 1
            source = masked_denoising.noised_sources.get(i)
            if source is None and not is_final_step:
                raise ValueError(
                    f"Attempted to apply {model_name} masked denoising at step {i}. Failed with prepared sources for "
                    f"steps {sorted(masked_denoising.noised_sources)} because no source was prepared for this step."
                )
            latents = block_state.latents  # pyright: ignore[reportAttributeAccessIssue]
            if not isinstance(latents, torch.Tensor) or not isinstance(num_inference_steps, int):
                raise TypeError(
                    f"Attempted to apply {model_name} masked denoising at step {i}. Failed with latents of type "
                    f"'{type(latents).__name__}' and num_inference_steps of type '{type(num_inference_steps).__name__}' "
                    "because they must be a torch.Tensor and an int."
                )
            if source is not None:
                block_state.latents = blend_differential_diffusion_latents(  # pyright: ignore[reportAttributeAccessIssue]
                    latents=latents,
                    source=source,
                    mask=masked_denoising.mask,
                    absolute_step=i,
                    num_inference_steps=num_inference_steps,
                )
            return components, block_state

    return MaskedDenoiseBlock
