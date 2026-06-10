import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import (  # type: ignore[reportMissingImports]
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    StableDiffusionXLControlNetInpaintPipeline,
    StableDiffusionXLInpaintPipeline,
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.stable_diffusion_xl.before_denoise import (  # type: ignore[reportMissingImports]
    StableDiffusionXLImg2ImgPrepareLatentsStep,
    StableDiffusionXLImg2ImgSetTimestepsStep,
    StableDiffusionXLPrepareLatentsStep,
    StableDiffusionXLSetTimestepsStep,
)
from diffusers.modular_pipelines.stable_diffusion_xl.modular_blocks_stable_diffusion_xl import (  # type: ignore[reportMissingImports]
    StableDiffusionXLAutoBlocks,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


# ------------------------------------------------------------------
# Composite blocks for standalone noise / add-noise operations
# ------------------------------------------------------------------


class _SDXLPrepareNoiseLatentStep(SequentialPipelineBlocks):
    """``set_timesteps`` → ``prepare_latents`` so ``init_noise_sigma`` is valid."""

    model_name = "stable-diffusion-xl"
    block_classes = [StableDiffusionXLSetTimestepsStep, StableDiffusionXLPrepareLatentsStep]
    block_names = ["set_timesteps", "prepare_latents"]


class _SDXLAddNoiseStep(SequentialPipelineBlocks):
    """``Img2ImgSetTimesteps`` → ``Img2ImgPrepareLatents`` for img2img noise."""

    model_name = "stable-diffusion-xl"
    block_classes = [
        StableDiffusionXLImg2ImgSetTimestepsStep,
        StableDiffusionXLImg2ImgPrepareLatentsStep,
    ]
    block_names = ["set_timesteps", "prepare_latents"]


class StableDiffusionXLLatentPipelineDriver(LatentPipelineDriver):
    """Hybrid driver for classic StableDiffusionXLPipeline.

    Uses modular blocks for encode, decode, noise latent creation, and
    add-noise, but delegates denoise to the classic
    ``DiffusionPipeline.__call__`` (via the base class ``denoise_latent``).
    """

    _inpaint_pipeline_class = StableDiffusionXLInpaintPipeline
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = (
        StableDiffusionXLControlNetInpaintPipeline
    )

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return StableDiffusionXLAutoBlocks().init_pipeline()

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls,
        pipe: ModularPipeline | DiffusionPipeline,
        control_net_model_lists: list[str] | str | None,
    ) -> ModularPipeline | DiffusionPipeline:
        if not control_net_model_lists:
            return pipe

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        controlnet_torch_dtype = cls._get_torch_type(pipe)
        from_pretrained_kwargs: dict[str, Any] = {}
        if controlnet_torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = controlnet_torch_dtype

        control_net_models = [
            ControlNetModel.from_pretrained(cn, **from_pretrained_kwargs) for cn in control_net_model_lists
        ]
        controlnet = control_net_models[0] if len(control_net_models) == 1 else control_net_models

        return StableDiffusionXLControlNetImg2ImgPipeline(controlnet=controlnet, **pipe.components)

    # ------------------------------------------------------------------
    # Modular blocks for encode / decode / noise latent
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(
        self,
        source_shape: tuple[int, ...],
        seed: int,
        *,
        num_inference_steps: int = 20,
    ) -> LatentArtifact:
        output_state = self._call_block(
            _SDXLPrepareNoiseLatentStep(),
            height=source_shape[-2],
            width=source_shape[-1],
            batch_size=1,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator().manual_seed(seed),
        )
        latents = output_state.get("latents")

        return self._make_latent_artifact(latents, source_shape=source_shape)

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        """Add noise to image latents via modular pipeline blocks.

        Returns the noised latent at the scheduler's sigma scale for the
        requested strength.
        """
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        output_state = self._call_block(
            _SDXLAddNoiseStep(),
            image_latents=latents,
            num_inference_steps=num_inference_steps,
            strength=strength,
            batch_size=latents.shape[0],
            num_images_per_prompt=1,
            generator=torch.Generator(device=device).manual_seed(seed),
            dtype=dtype,
        )
        return self._make_latent_artifact(
            output_state.get("latents"), source_shape=latent.source_shape, upstream=latent
        )

    @override
    def denoise_latent(
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        seed: int = 0,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        # Img2Img requires ``image`` and ``strength``. Pass the latent
        # as ``image`` — preprocess detects 4-channel input and returns
        # it as-is; ``prepare_latents`` is skipped because ``latents``
        # is also provided.
        # Skip if inpainting — the inpaint path provides its own ``image``.
        if not isinstance(latent, InpaintMaskArtifact) or self.pipe.__class__ == StableDiffusionXLInpaintPipeline:
            if not isinstance(latent, InpaintMaskArtifact):
                device, dtype = self._get_device_and_type()
                latents_on_device = latent.to_torch(device=device, dtype=dtype)
                kwargs.setdefault("image", latents_on_device)
            kwargs.setdefault("strength", 1.0)

        return super().denoise_latent(
            latent,
            num_inference_steps,
            seed=seed,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> Image:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        decode_block = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_block, latents=latents, output_type="pil")
        images = output_state.get("images")
        return images[0]

    @override
    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """SDXL ControlNet Inpaint Pipeline always VAE-encodes image (no latent passthrough)."""
        if self._is_controlnet_pipe():
            source_pil = artifact.source_image_pil()
            if source_pil is None:
                raise ValueError(
                    f"{type(self).__name__} ControlNet+Inpaint requires source_image in the InpaintMaskArtifact."
                )
            return {
                "image": source_pil,
                "mask_image": artifact.mask_image,
                "strength": artifact.strength,
            }
        return super()._get_inpaint_kwargs(artifact)

    @override
    def encode_image(self, image: Image | torch.Tensor, source_shape: tuple[int, ...]) -> LatentArtifact:
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=image)
        result = output_state.get("image_latents")
        return self._make_latent_artifact(result, source_shape=source_shape)
