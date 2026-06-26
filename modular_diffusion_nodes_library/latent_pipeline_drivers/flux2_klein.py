import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import Flux2KleinInpaintPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.encoders import Flux2VaeEncoderStep
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.latent_pipeline_drivers.flux2_base import Flux2BaseLatentPipelineDriver
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class Flux2KleinLatentPipelineDriver(Flux2BaseLatentPipelineDriver):
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = Flux2KleinInpaintPipeline

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    # standard pipeline denoise performs de-normalize before returning and denoise block de-normalizes before decoding
    # so we need to re-normalize before returning the latents.
    def _normalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize unnormalized patchified latents with BN."""
        latents_bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return (latents - latents_bn_mean) / latents_bn_std

    def _denormalize_latent(self, latents: torch.Tensor) -> torch.Tensor:
        """Reverse BN normalization on patchified latents."""
        latents_bn_mean = self.pipe.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
        latents_bn_std = torch.sqrt(
            self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        return latents * latents_bn_std + latents_bn_mean

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """Flux2Klein inpaint pipeline handles encoding/noise/masking internally.

        Pass the denormalized image latent and the pixel-space mask.
        The pipeline will BN-normalize image latent internally, so we must denormalize first.
        """
        device, dtype = self._get_device_and_type()
        source_latent = self._denormalize_latent(artifact.source_latent).to(device=device, dtype=dtype)
        return {
            "image": source_latent,
            "mask_image": artifact.mask_image,
            "strength": artifact.strength,
        }

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Packed means prepare for preview, otherwise normalize."""
        if latents_from_pipe.ndim == 3:
            return self._unpack_latents(latents_from_pipe, latents_source_shape[-2], latents_source_shape[-1])
        patchified = Flux2VaeEncoderStep._patchify_latents(latents_from_pipe)
        return self._normalize_latent(patchified)

    @torch.inference_mode()
    @override
    def denoise_latent(
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        payloads = normalize_to_payloads(kwargs.pop(MediaGenConditioningKey.OUTPUT, None))
        if payloads is not None:
            # Flux2Klein only routes conditioning into the inpaint variant
            if not isinstance(latent, InpaintMaskArtifact):
                msg = "Failed to apply Flux2Klein conditioning because it requires an inpaint mask input."
                raise ValueError(msg)
            image_reference = self._build_image_reference(payloads)
            if image_reference:
                kwargs["image_reference"] = image_reference

        return super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state,
            callback,
            start_step,
            end_step,
            return_fully_denoised,
            **kwargs,
        )

    @staticmethod
    def _build_image_reference(
        payloads: list[MediaGenConditioningPayload],
    ) -> list[Any] | None:
        """Convert typed conditioning payloads into a list of PIL images for image_reference."""
        reference_images: list[Any] = []
        for payload in payloads:
            if payload.mode is not ConditioningMode.IMAGE:
                msg = f"Failed to build Flux2Klein conditioning because mode '{payload.mode.value}' is unsupported."
                raise ValueError(msg)
            for entry in payload.entries:
                reference_images.append(resolve_conditioning_image(entry.artifact))

        return reference_images or None
