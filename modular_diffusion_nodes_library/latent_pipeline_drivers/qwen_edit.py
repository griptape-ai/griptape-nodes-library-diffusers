import logging
from typing import Any, override

import torch  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    calculate_dimensions,  # type: ignore[reportMissingImports]
)
from PIL import Image

from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import TextEncodings
from modular_diffusion_nodes_library.latent_pipeline_drivers.qwen import QwenLatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")
_QWEN_EDIT_TARGET_AREA = 1024 * 1024


class QwenEditLatentPipelineDriver(QwenLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline) -> None:
        super().__init__(pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @staticmethod
    def _calculate_edit_latent_dims(height: int, width: int) -> tuple[int, int]:
        """Calculate the latent spatial dimensions for a given input image size.
        QwenImageEditResizeStep (e.g. used in encode block) resizes the input image to a fixed target_area of
        1024*1024 while preserving aspect ratio. We must use the same dims in other operations (such as create_noise)
        so that the latents from all operations share the same spatial resolution.
        """
        calc_width, calc_height, _ = calculate_dimensions(_QWEN_EDIT_TARGET_AREA, width / height)
        return int(calc_height), int(calc_width)

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        height, width = self._calculate_edit_latent_dims(latents_source_shape[-2], latents_source_shape[-1])
        latents_source_shape = (*latents_source_shape[:-2], height, width)
        return super().prepare_output_latent(latents_from_pipe, latents_source_shape)

    @override
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:  # noqa: ARG002
        height, width = self._calculate_edit_latent_dims(latents_source_shape[-2], latents_source_shape[-1])
        latents_source_shape = (*latents_source_shape[:-2], height, width)
        return super().create_noise_latent(latents_source_shape, seed)

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> Image.Image:
        # The pipeline snaps input dims to a fixed target_area (e.g. 512×512 → 1024×1024),
        # so the decoded image may be larger than what the user requested.
        # Resize back to the originally-requested dimensions.
        decoded = super().decode_latent(latents, latents_source_shape)
        height, width = latents_source_shape[-2], latents_source_shape[-1]
        if decoded.height != height or decoded.width != width:
            decoded = decoded.resize((width, height), Image.Resampling.LANCZOS)
        return decoded

    @override
    def encode_image(self, image: Image.Image | torch.Tensor) -> torch.Tensor:
        if isinstance(image, torch.Tensor):
            img_h, img_w = image.shape[-2], image.shape[-1]
        else:
            img_h, img_w = image.height, image.width
        height, width = self._calculate_edit_latent_dims(img_h, img_w)
        encode_pipeline = self.modular_pipe.blocks.sub_blocks["vae_encoder"]

        output_state = self._call_block(encode_pipeline, image=image, height=height, width=width)

        latents = output_state.get("image_latents")
        if isinstance(latents, list):
            latents = latents[0]
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"Expected Tensor for image_latents, got {type(latents).__name__}.")
        latents = latents.squeeze(2)
        return latents

    @override
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        height, width = self._calculate_edit_latent_dims(latents_source_shape[-2], latents_source_shape[-1])
        latents_source_shape = (*latents_source_shape[:-2], height, width)
        return super().add_noise_to_latent(latents, latents_source_shape, seed, num_inference_steps, strength)

    @override
    def encode_prompt(self, prompt: str, negative_prompt: str, **kwargs: Any) -> TextEncodings:
        text_encoder_pipe = self.modular_pipe.blocks.sub_blocks["text_encoder"]
        call_kwargs: dict[str, Any] = {"prompt": prompt}
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        image = kwargs.get("image")
        if image is not None:
            call_kwargs["image"] = image
        return self._call_block(text_encoder_pipe, **call_kwargs)

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
        height, width = self._calculate_edit_latent_dims(latents_source_shape[-2], latents_source_shape[-1])
        latents_source_shape = (*latents_source_shape[:-2], height, width)
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
