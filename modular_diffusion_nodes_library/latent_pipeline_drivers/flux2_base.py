import logging
from typing import override

import PIL.Image
import torch  # type: ignore[reportMissingImports]
import torchvision.transforms.functional as TF  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.before_denoise import (  # type: ignore[reportMissingImports]
    Flux2PrepareLatentsStep,
    Flux2SetTimestepsStep,
)
from diffusers.modular_pipelines.flux2.decoders import Flux2UnpackLatentsStep  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.encoders import Flux2VaeEncoderStep  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.inputs import Flux2ProcessImagesInputStep  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.flux2.modular_blocks_flux2 import Flux2AutoBlocks  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class Flux2BaseLatentPipelineDriver(LatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return Flux2AutoBlocks().init_pipeline()

    @override
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:  # noqa: ARG002
        device, dtype = self._get_device_and_type()
        height, width = latents_source_shape[-2], latents_source_shape[-1]

        generator = torch.Generator().manual_seed(seed)
        prepare_latents = Flux2PrepareLatentsStep()
        output_state = self._call_block(
            prepare_latents,
            height=height,
            width=width,
            num_images_per_prompt=1,
            generator=generator,
            batch_size=1,
            dtype=dtype,
        )
        packed_latents = output_state.get("latents")
        latent_ids = output_state.get("latent_ids")

        unpack_latents = Flux2UnpackLatentsStep()
        latents_state = self._call_block(unpack_latents, latents=packed_latents, latent_ids=latent_ids)
        latents = latents_state.get("latents")
        latents = latents.to(device)
        return latents

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> Image:  # noqa: ARG002
        device, dtype = self._get_device_and_type()
        latents = latents.to(device=device, dtype=dtype)

        decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_block, latents=latents, output_type="pil")

        return output_state.get("images")[0]

    @override
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            # Flux2 modular VAE encoder only accepts PIL images
            img_np = image.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            image = PIL.Image.fromarray(img_np)
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=image, height=image.height, width=image.width)
        return output_state.get("image_latents")[0]

    @override
    def encode_masked_image(self, image: Image, mask: Image) -> torch.Tensor:
        # PIL -> aligned, normalized tensor (multiple-of-32 alignment + <=1024^2 cap).
        pre = self._call_block(Flux2ProcessImagesInputStep(), image=image)
        source_t = pre.get("condition_images")[0]

        # Resize mask to match the (possibly aligned/cropped) tensor dims.
        aligned_h, aligned_w = source_t.shape[-2:]
        mask_resized = mask.convert("L").resize((aligned_w, aligned_h), PIL.Image.NEAREST)
        mask_t = TF.to_tensor(mask_resized)[None]  # (1, 1, H, W), float32 in [0, 1]

        masked_t = source_t * (mask_t < 0.5)

        out = self._call_block(Flux2VaeEncoderStep(), condition_images=[masked_t])
        return out.get("image_latents")[0]

    @override
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        # scale_noise is called manually because diffusers has no modular block for
        # strength-based noise addition. Flux2PrepareLatentsStep only generates or
        # passes through latents without blending noise at a specific timestep.
        # Flux2SetTimestepsStep to handle sigmas, dynamic shifting, and mu computation
        set_timesteps = Flux2SetTimestepsStep()
        height, width = latents_source_shape[-2], latents_source_shape[-1]
        timesteps_state = self._call_block(
            set_timesteps, num_inference_steps=num_inference_steps, height=height, width=width
        )
        self.pipe.scheduler._begin_index = None
        timesteps = timesteps_state.get("timesteps")

        if timesteps is None or len(timesteps) == 0:
            raise ValueError("Scheduler timesteps are not set. Cannot add noise to latent.")

        noise = self.create_noise_latent(latents_source_shape, seed)

        # Compute the timestep at which to add noise based on strength
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        latent_timestep = timesteps[t_start * self.pipe.scheduler.order :][:1]

        noisy_latent = self.pipe.scheduler.scale_noise(latents, latent_timestep, noise)
        return noisy_latent
