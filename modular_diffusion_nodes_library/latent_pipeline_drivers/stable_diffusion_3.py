import logging
from typing import Any, override

import torch  # type: ignore[reportMissingImports]
from diffusers import StableDiffusion3InpaintPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.stable_diffusion_3.before_denoise import (  # type: ignore[reportMissingImports]
    StableDiffusion3Img2ImgPrepareLatentsStep,
    StableDiffusion3Img2ImgSetTimestepsStep,
    StableDiffusion3PrepareLatentsStep,
    StableDiffusion3SetTimestepsStep,
)
from diffusers.modular_pipelines.stable_diffusion_3.modular_blocks_stable_diffusion_3 import (  # type: ignore[reportMissingImports]
    StableDiffusion3AutoBlocks,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


# ------------------------------------------------------------------
# Composite blocks for standalone noise / add-noise operations
# ------------------------------------------------------------------


class _SD3PrepareNoiseLatentStep(SequentialPipelineBlocks):
    """``set_timesteps`` → ``prepare_latents`` so ``init_noise_sigma`` is valid."""

    model_name = "stable-diffusion-3"
    block_classes = [StableDiffusion3SetTimestepsStep, StableDiffusion3PrepareLatentsStep]
    block_names = ["set_timesteps", "prepare_latents"]


class _SD3AddNoiseStep(SequentialPipelineBlocks):
    """``Img2ImgSetTimesteps`` → ``Img2ImgPrepareLatents`` for img2img noise."""

    model_name = "stable-diffusion-3"
    block_classes = [
        StableDiffusion3Img2ImgSetTimestepsStep,
        StableDiffusion3Img2ImgPrepareLatentsStep,
    ]
    block_names = ["set_timesteps", "prepare_latents"]


class StableDiffusion3LatentPipelineDriver(LatentPipelineDriver):
    """Hybrid driver for StableDiffusion3Pipeline.

    Uses modular blocks for encode, decode, noise latent creation, and
    add-noise, but delegates denoise to the classic
    ``DiffusionPipeline.__call__`` (via the base class ``denoise_latent``).
    """

    _inpaint_pipeline_class = StableDiffusion3InpaintPipeline

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return StableDiffusion3AutoBlocks().init_pipeline()

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    # ------------------------------------------------------------------
    # Modular blocks for encode / decode / noise latent
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(
        self,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int = 20,
    ) -> torch.Tensor:
        output_state = self._call_block(
            _SD3PrepareNoiseLatentStep(),
            height=latents_source_shape[-2],
            width=latents_source_shape[-1],
            batch_size=1,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(seed),
        )
        return output_state.get("latents")

    @override
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        """Add noise to image latents via modular pipeline blocks.

        Returns the noised latent at the scheduler's sigma scale for the
        requested strength.
        """
        device, dtype = self._get_device_and_type()

        # Generate noise via the modular path
        noise_latent = self.create_noise_latent(latents_source_shape, seed, num_inference_steps)

        output_state = self._call_block(
            _SD3AddNoiseStep(),
            latents=noise_latent.to(device=device, dtype=dtype),
            image_latents=latents.to(device=device, dtype=dtype),
            num_inference_steps=num_inference_steps,
            strength=strength,
            height=latents_source_shape[-2],
            width=latents_source_shape[-1],
        )
        return output_state.get("latents")

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
        return super().denoise_latent(
            latents,
            latents_source_shape,
            num_inference_steps,
            seed=seed,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> Image:
        decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_block, latents=latents, output_type="pil")
        images = output_state.get("images")
        return images[0]

    @override
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=image)
        return output_state.get("image_latents")
