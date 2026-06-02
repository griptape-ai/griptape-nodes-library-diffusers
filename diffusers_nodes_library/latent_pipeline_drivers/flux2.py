import logging
from typing import Any, override

import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.before_denoise import (
    Flux2PrepareLatentsStep,  # type: ignore[reportMissingImports]
)
from diffusers.modular_pipelines.flux2.decoders import Flux2UnpackLatentsStep  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from diffusers_nodes_library.latent_pipeline_drivers.flux2_base import Flux2BaseLatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class Flux2LatentPipelineDriver(Flux2BaseLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    def _unpack_latents(self, latents: torch.Tensor, height: int, width: int) -> torch.Tensor:
        device, dtype = self._get_device_and_type()
        latents = latents.to(device=device, dtype=dtype)

        prepare_latents = Flux2PrepareLatentsStep()
        id_state = self._call_block(
            prepare_latents,
            height=height,
            width=width,
            num_images_per_prompt=1,
            generator=None,
            batch_size=1,
            dtype=dtype,
        )
        latent_ids = id_state.get("latent_ids")

        unpack_latents = Flux2UnpackLatentsStep()
        unpack_state = self._call_block(unpack_latents, latents=latents, latent_ids=latent_ids)
        return unpack_state.get("latents")

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
        # Packed latents [B, seq, C] — e.g. flux2 output_type="latent".
        # The denoising loop never unpacks, so we unpack here to get [B, C, H/2, W/2].
        unpacked_latents = self._unpack_latents(denoised, latents_source_shape[-2], latents_source_shape[-1])
        return unpacked_latents
