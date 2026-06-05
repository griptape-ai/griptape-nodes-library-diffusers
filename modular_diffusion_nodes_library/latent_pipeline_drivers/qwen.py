import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers import (
    QwenImageControlNetModel,  # type: ignore[reportMissingImports]
    QwenImageControlNetPipeline,  # type: ignore[reportMissingImports]
    QwenImageInpaintPipeline,  # type: ignore[reportMissingImports]
    QwenImageMultiControlNetModel,  # type: ignore[reportMissingImports]
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.qwenimage.before_denoise import (  # type: ignore[reportMissingImports]
    QwenImagePrepareLatentsStep,
    QwenImagePrepareLatentsWithStrengthStep,
    QwenImageSetTimestepsWithStrengthStep,
)
from diffusers.modular_pipelines.qwenimage.modular_blocks_qwenimage import (
    QwenImageAutoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.utils.torch_utils import randn_tensor  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class _QwenImageNoiseWithStrengthSequence(SequentialPipelineBlocks):
    """Add noise to the image latents according to the strength value."""

    block_classes = [
        QwenImagePrepareLatentsStep,
        QwenImageSetTimestepsWithStrengthStep,
        QwenImagePrepareLatentsWithStrengthStep,
    ]
    block_names = ["prepare_latents", "set_timesteps", "prepare_img2img_latents"]


class QwenLatentPipelineDriver(LatentPipelineDriver):
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = QwenImageInpaintPipeline

    def __init__(self, pipe: DiffusionPipeline) -> None:
        super().__init__(pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return True

    @classmethod
    @override
    def control_pipe_from_standard(
        cls, pipe: ModularPipeline | DiffusionPipeline, control_net_model_lists: list[str] | str | None
    ):

        if not control_net_model_lists:
            return pipe

        if not isinstance(control_net_model_lists, list):
            control_net_model_lists = [control_net_model_lists]

        controlnet_torch_dtype = cls._get_torch_type(pipe)
        from_pretrained_kwargs: dict[str, Any] = {}
        if controlnet_torch_dtype is not None:
            from_pretrained_kwargs["torch_dtype"] = controlnet_torch_dtype

        control_net_models = [
            QwenImageControlNetModel.from_pretrained(model, **from_pretrained_kwargs)
            for model in control_net_model_lists
        ]

        if len(control_net_models) == 1:
            controlnet = control_net_models[0]
        else:
            controlnet = QwenImageMultiControlNetModel(control_net_models)

        return QwenImageControlNetPipeline(controlnet=controlnet, **pipe.components)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return QwenImageAutoBlocks().init_pipeline()

    def _get_unpacked_image_latents(self, output_state: dict[str, Any], height: int, width: int) -> torch.Tensor:
        latents = output_state.get("latents")
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"Expected latents to be a torch.Tensor but got {type(latents)}")
        return self.prepare_output_latent(latents, (1, -1, height, width))

    @override
    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        """Return latents ready to be passed into the pipeline, which may involve packing or other preprocessing."""
        packed_latents = self.modular_pipe.pachifier.pack_latents(latents)
        return packed_latents

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Return latents ready to be process further e.g. with mask, which may involve unpacking or other postprocessing."""
        unpack_latents = self.modular_pipe.pachifier.unpack_latents(
            latents_from_pipe, latents_source_shape[-2], latents_source_shape[-1], self.modular_pipe.vae_scale_factor
        )
        image_latents = unpack_latents.squeeze(2)  # [B,z,1,H',W'] → [B,z,H',W']
        return image_latents

    @override
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:  # noqa: ARG002
        _, dtype = self._get_device_and_type()
        prepare_latents_pipeline = QwenImagePrepareLatentsStep()
        height, width = latents_source_shape[-2], latents_source_shape[-1]

        output_state = self._call_block(
            prepare_latents_pipeline,
            height=height,
            width=width,
            batch_size=1,
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
        )

        output_latents = self._get_unpacked_image_latents(output_state, height, width)
        return output_latents

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> Image:  # noqa: ARG002
        device, dtype = self._get_device_and_type()
        latents = latents.to(device=device, dtype=dtype)
        latents = latents.unsqueeze(2)

        decode_pipeline = self.modular_pipe.blocks.sub_blocks["decode"]
        output_state = self._call_block(decode_pipeline, latents=latents, output_type="pil")

        output_image = self._get_required(output_state, "images", list)
        return output_image[0]

    @override
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        noise_with_strength_pipeline = _QwenImageNoiseWithStrengthSequence()
        _, dtype = self._get_device_and_type()
        height, width = latents_source_shape[-2], latents_source_shape[-1]

        noisy_state = self._call_block(
            noise_with_strength_pipeline,
            height=height,
            width=width,
            batch_size=latents.shape[0],
            num_images_per_prompt=1,
            generator=torch.Generator().manual_seed(seed),
            dtype=dtype,
            latents=None,
            image_latents=self.modular_pipe.pachifier.pack_latents(latents),
            num_inference_steps=num_inference_steps,
            strength=strength,
        )

        output_latents = self._get_unpacked_image_latents(noisy_state, height, width)
        return output_latents

    @override
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:  # noqa: ARG002
        encode_pipeline = self.modular_pipe.blocks.sub_blocks["vae_encoder"]
        if isinstance(image, torch.Tensor):
            height = image.shape[-2]
            width = image.shape[-1]
        else:
            height = image.height
            width = image.width
        output_state = self._call_block(encode_pipeline, image=image, height=height, width=width)
        latents = self._get_required(output_state, "image_latents", torch.Tensor)
        return latents.squeeze(
            2
        )  # [B,z,1,H',W'] → [B,z,H',W'] - remove temporal dimension (the same VAE is shared between video and image pipelines)

    def _advance_generator(self, gen: torch.Generator, latents: torch.Tensor) -> None:
        """Consume one randn draw from ``gen`` to mimic the skipped ``_encode_vae_image`` sample."""
        _, dtype = self._get_device_and_type()
        randn_tensor(latents.shape, generator=gen, device=gen.device, dtype=dtype)

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
        """Align init-noise RNG with the standalone (PIL-image) inpaint path."""
        if kwargs.get("inpaint_mask_artifact") is not None:
            gen = torch.Generator().manual_seed(seed)
            self._advance_generator(gen, latents)
            kwargs["generator"] = gen
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
