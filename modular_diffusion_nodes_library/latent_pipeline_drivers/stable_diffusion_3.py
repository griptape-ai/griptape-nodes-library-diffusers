import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import (  # type: ignore[reportMissingImports]
    StableDiffusion3ControlNetInpaintingPipeline,
    StableDiffusion3ControlNetPipeline,
    StableDiffusion3InpaintPipeline,
)
from diffusers.models.controlnets.controlnet_sd3 import SD3ControlNetModel  # type: ignore[reportMissingImports]
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

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import (
    GeneratorState,
    LatentPipelineDriver,
)

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
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = (
        StableDiffusion3ControlNetInpaintingPipeline
    )

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return StableDiffusion3AutoBlocks().init_pipeline()

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
            SD3ControlNetModel.from_pretrained(cn, **from_pretrained_kwargs) for cn in control_net_model_lists
        ]
        controlnet = control_net_models[0] if len(control_net_models) == 1 else control_net_models

        return StableDiffusion3ControlNetPipeline(controlnet=controlnet, **pipe.components)

    # ------------------------------------------------------------------
    # Modular blocks for encode / decode / noise latent
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(
        self,
        source_shape: tuple[int, ...],
        generator_state: GeneratorState,
        *,
        num_inference_steps: int = 20,
    ) -> LatentArtifact:
        generator = generator_state.to_generator()
        output_state = self._call_block(
            _SD3PrepareNoiseLatentStep(),
            height=source_shape[-2],
            width=source_shape[-1],
            batch_size=1,
            num_images_per_prompt=1,
            num_inference_steps=num_inference_steps,
            generator=generator,
        )
        latents = output_state.get("latents")
        return self._make_latent_artifact(
            latents, source_shape=source_shape, meta=GeneratorState.from_generator(generator).as_meta()
        )

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        """Add noise to image latents via modular pipeline blocks.

        Returns the noised latent at the scheduler's sigma scale for the
        requested strength.
        """
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        generator = generator_state.to_generator()

        # Generate noise via the modular path
        noise_latent_artifact = self.create_noise_latent(
            latent.source_shape, generator_state, num_inference_steps=num_inference_steps
        )
        noise_latent = noise_latent_artifact.to_torch(device=device, dtype=dtype)

        output_state = self._call_block(
            _SD3AddNoiseStep(),
            latents=noise_latent,
            image_latents=latents,
            num_inference_steps=num_inference_steps,
            strength=strength,
            height=latent.source_shape[-2],
            width=latent.source_shape[-1],
        )
        return self._make_latent_artifact(
            output_state.get("latents"), source_shape=latent.source_shape, upstream=latent,
            meta=GeneratorState.from_artifact(noise_latent_artifact).as_meta(),
        )

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
        device, dtype = self._get_device_and_type()

        # Img2Img requires ``image`` and ``strength``.  Pass the latent
        # as ``image`` — prepare_latents detects 16-channel input and
        # uses it directly; strength=1.0 gives full t2i behavior.
        # Skip if inpainting or ControlNet — those paths provide their own
        # ``image`` / do not accept ``image``+``strength`` kwargs.
        if not isinstance(latent, InpaintMaskArtifact) and not self._is_controlnet_pipe():
            latents_on_device = latent.to_torch(device=device, dtype=dtype)
            kwargs.setdefault("image", latents_on_device)
            kwargs.setdefault("strength", 1.0)

        return super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
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
        if self._is_controlnet_pipe():
            source_pil = artifact.source_image_pil()
            if source_pil is None:
                raise ValueError(
                    f"{type(self).__name__} ControlNet+Inpaint requires source_image in the InpaintMaskArtifact."
                )
            return {
                "control_image": source_pil,
                "control_mask": artifact.mask_image,
            }

        device, dtype = self._get_device_and_type()
        source_latent = artifact.to_torch(device=device, dtype=dtype)
        return {
            "image": source_latent,
            "mask_image": artifact.mask_image,
            "masked_image_latents": source_latent,
            "strength": artifact.strength,
        }

    @override
    def encode_image(self, image: Image | torch.Tensor, source_shape: tuple[int, ...]) -> LatentArtifact:
        encode_block = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        output_state = self._call_block(encode_block, image=image)
        result = output_state.get("image_latents")
        return self._make_latent_artifact(result, source_shape=source_shape)
