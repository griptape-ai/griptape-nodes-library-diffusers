import logging
from typing import Any, override

import torch  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    calculate_dimensions,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (  # type: ignore[reportMissingImports]
    QwenImageEditPlusPipeline,
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
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    MediaGenConditioningKey,
    resolve_conditioning_image,
)
from modular_diffusion_nodes_library.utils.dimension_alignment import DimensionAlignmentResult
from modular_diffusion_nodes_library.utils.pipeline_utils import create_pipe_variant

logger = logging.getLogger("modular_diffusers_nodes_library")
_QWEN_EDIT_TARGET_AREA = 1024 * 1024


class QwenEditLatentPipelineDriver(QwenLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline) -> None:
        super().__init__(pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def align_dimensions(self, height: int, width: int, num_frames: int | None = None) -> DimensionAlignmentResult:
        calc_width, calc_height, _ = calculate_dimensions(_QWEN_EDIT_TARGET_AREA, width / height)
        return DimensionAlignmentResult(int(calc_height), int(calc_width), num_frames, None)

    @override
    def validate_dimensions(self, height: int, width: int, num_frames: int | None = None) -> list[str]:
        messages: list[str] = []
        aligned = self.align_dimensions(height, width)
        if aligned.height != height or aligned.width != width:
            messages.append(
                f"height={height}, width={width} do not fill the required target area of "
                f"{_QWEN_EDIT_TARGET_AREA} px. "
                f"Suggested values: height={aligned.height}, width={aligned.width}."
            )
        return messages

    @staticmethod
    def _images_from_conditioning(payload: Any) -> list[Image.Image]:
        payloads = normalize_to_payloads(payload)
        if payloads is None:
            return []
        images: list[Image.Image] = []
        for p in payloads:
            for entry in p.entries:
                images.append(resolve_conditioning_image(entry.artifact))
        return images

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, VideoMedia):
            raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video.")
        image = media.image
        if isinstance(image, torch.Tensor):
            height, width = image.shape[-2], image.shape[-1]
        else:
            height, width = image.height, image.width
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
    def encode_prompt(self, prompt: str, negative_prompt: str, **kwargs: Any) -> TextEncodings:
        text_encoder_pipe = self.modular_pipe.blocks.sub_blocks["text_encoder"]
        call_kwargs: dict[str, Any] = {"prompt": prompt}
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt

        payload = kwargs.get(MediaGenConditioningKey.OUTPUT)
        if payload is not None:
            images = self._images_from_conditioning(payload)
            if images:
                call_kwargs["image"] = images[0]
        else:
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
        original_pipe = self._pipe

        payload = kwargs.pop(MediaGenConditioningKey.OUTPUT, None)
        if payload is not None:
            images = self._images_from_conditioning(payload)
            if len(images) > 1:
                # Plus variant: swap pipeline, strip pre-computed embeddings so the Plus pipeline
                # re-encodes with all conditioning images via its own VLM call.
                torch_dtype = self._get_torch_type(self._pipe)
                self._pipe = create_pipe_variant(original_pipe, QwenImageEditPlusPipeline, torch_dtype=torch_dtype)
                kwargs["image"] = images
                for key in (
                    "prompt_embeds",
                    "prompt_embeds_mask",
                    "negative_prompt_embeds",
                    "negative_prompt_embeds_mask",
                ):
                    kwargs.pop(key, None)
            elif images:
                kwargs["image"] = images[0]

        try:
            result = super().denoise_latent(
                latent,
                num_inference_steps,
                generator_state=generator_state,
                callback=callback,
                start_step=start_step,
                end_step=end_step,
                return_fully_denoised=return_fully_denoised,
                **kwargs,
            )
        finally:
            self._pipe = original_pipe

        return result
