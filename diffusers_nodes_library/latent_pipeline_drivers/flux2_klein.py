import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import Flux2KleinInpaintPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.encoders import Flux2VaeEncoderStep
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from diffusers_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from diffusers_nodes_library.latent_pipeline_drivers.flux2_base import Flux2BaseLatentPipelineDriver

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
        latents_bn_std = torch.sqrt(self.pipe.vae.bn.running_var.view(1, -1, 1, 1) + self.pipe.vae.config.batch_norm_eps).to(
            latents.device, latents.dtype)
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
            "inpaint_strength": artifact.strength,
        }

    @torch.inference_mode()
    @override
    def denoise_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        num_inference_steps: int,
        seed: int = 0,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        denoised = super().denoise_latent(
            latents,
            latents_source_shape,
            num_inference_steps,
            seed,
            callback,
            start_step,
            end_step,
            return_fully_denoised,
            **kwargs,
        )
        # Klein denoise unpacks + BN-denormalises + unpatchifies before returning, so we must
        # reverse that: patchify then BN-normalise to get back to [B, C, H/2, W/2].
        output_latent = Flux2VaeEncoderStep._patchify_latents(denoised)
        output_latent = self._normalize_latent(output_latent)
        return output_latent
