import logging
from typing import Any, override

import torch  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    calculate_dimensions,  # type: ignore[reportMissingImports]
)
from PIL import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    GeneratorState,
    ImageMedia,
    TextEncodings,
    VideoMedia,
)
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

    @classmethod
    def _snap_source_shape(cls, source_shape: tuple[int, ...]) -> tuple[int, ...]:
        height, width = cls._calculate_edit_latent_dims(source_shape[-2], source_shape[-1])
        return (*source_shape[:-2], height, width)

    def _with_snapped_source_shape(self, latent: LatentArtifact) -> LatentArtifact:
        """Return a copy of ``latent`` whose ``source_shape`` is snapped to QwenEdit's target area."""
        snapped = self._snap_source_shape(latent.source_shape)
        if snapped == latent.source_shape:
            return latent
        device, dtype = self._get_device_and_type()
        return LatentArtifact.from_torch(
            latent.to_torch(device=device, dtype=dtype), source_shape=snapped, meta=latent.meta
        )

    def _with_original_source_shape(
        self, snapped_output: LatentArtifact, original_source_shape: tuple[int, ...]
    ) -> LatentArtifact:
        """Return a copy of ``snapped_output`` whose ``source_shape`` is restored to the caller's original shape."""
        if snapped_output.source_shape == original_source_shape:
            return snapped_output
        return self._make_latent_artifact(
            snapped_output.to_torch(),
            source_shape=original_source_shape,
            upstream=snapped_output,
        )

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        return super().prepare_output_latent(latents_from_pipe, self._snap_source_shape(latents_source_shape))

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        snapped_output = super().create_noise_latent(self._snap_source_shape(source_shape), generator_state)
        return self._with_original_source_shape(snapped_output, source_shape)

    @override
    def decode_latent(self, latent: LatentArtifact) -> Image.Image:
        # The pipeline snaps input dims to a fixed target_area (e.g. 512×512 → 1024×1024),
        # so the decoded image may be larger than what the user requested.
        # Resize back to the originally-requested dimensions.
        height, width = latent.source_shape[-2], latent.source_shape[-1]
        decoded = super().decode_latent(self._with_snapped_source_shape(latent))
        if decoded.height != height or decoded.width != width:
            decoded = decoded.resize((width, height), Image.Resampling.LANCZOS)
        return decoded

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, VideoMedia):
            raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video.")
        image = media.image
        if isinstance(image, torch.Tensor):
            img_h, img_w = image.shape[-2], image.shape[-1]
        else:
            img_h, img_w = image.height, image.width
        height, width = self._calculate_edit_latent_dims(img_h, img_w)
        encode_pipeline = self.modular_pipe.blocks.sub_blocks["vae_encoder"]

        generator = generator_state.to_generator()
        output_state = self._call_block(encode_pipeline, image=image, height=height, width=width, generator=generator)

        latents = output_state.get("image_latents")
        if isinstance(latents, list):
            latents = latents[0]
        if not isinstance(latents, torch.Tensor):
            raise ValueError(f"Expected Tensor for image_latents, got {type(latents).__name__}.")
        latents = latents.squeeze(2)
        return self._make_latent_artifact(latents, source_shape=media.source_shape)

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        snapped_output = super().add_noise_to_latent(
            self._with_snapped_source_shape(latent), generator_state, num_inference_steps, strength
        )
        return self._with_original_source_shape(snapped_output, latent.source_shape)

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
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        original_source_shape = latent.source_shape
        if isinstance(latent, LatentArtifact):
            latent = self._with_snapped_source_shape(latent)
        snapped_output = super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )
        return self._with_original_source_shape(snapped_output, original_source_shape)
