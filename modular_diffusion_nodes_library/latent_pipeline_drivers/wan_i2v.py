import logging
from typing import Any, override

import numpy as np
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_blocks_wan22_i2v import (
    Wan22Image2VideoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.modular_pipelines.wan.modular_blocks_wan_i2v import (
    WanImage2VideoAutoBlocks,  # type: ignore[reportMissingImports]
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image, Resampling

from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    DecodeResult,
    GeneratorState,
    ImageMedia,
    VideoMedia,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.wan import WanTextToVideoLatentPipelineDriver
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import normalize_to_payloads
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
    resolve_frame_index,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class WanImageToVideoLatentPipelineDriver(WanTextToVideoLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        if getattr(self.pipe, "transformer_2", None) is not None:
            return Wan22Image2VideoBlocks().init_pipeline()
        return WanImage2VideoAutoBlocks().init_pipeline()

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        width, height = self.get_resize_dimensions(source_shape[-1], source_shape[-2])
        resized_source_shape = (*source_shape[:-2], height, width)
        resized_output = super().create_noise_latent(resized_source_shape, generator_state)
        return self._make_latent_artifact(
            resized_output.to_torch(),
            source_shape=source_shape,
            upstream=resized_output,
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        frames = super().decode_latent(latent)
        source_shape = latent.source_shape
        output_frames = [frame.resize((source_shape[-1], source_shape[-2]), Resampling.LANCZOS) for frame in frames]  # type: ignore[reportGeneralTypeIssues]
        return output_frames

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, ImageMedia):
            return super().encode_media(media, generator_state)
        preprocessed = VideoMedia(
            frames=[self.preprocess_image(frame) for frame in media.frames],
            source_shape=media.source_shape,
        )
        return super().encode_media(preprocessed, generator_state)

    @override
    def denoise_latent(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        latent: LatentArtifact,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        """Denoise a WAN i2v video latent."""

        update_kwargs = kwargs.copy()
        payloads = normalize_to_payloads(update_kwargs.pop(MediaGenConditioningKey.OUTPUT, None))

        # Resize conditioning images to the latent's bin dims and pin height/width on
        # the pipe so prepare_latents matches the latent tensor's shape.
        width, height = self.get_resize_dimensions(latent.source_shape[-1], latent.source_shape[-2])

        if payloads is not None:
            num_frames = latent.source_shape[-3]
            for payload in payloads:
                if payload.mode is ConditioningMode.IMAGE:
                    for entry in payload.entries:
                        image = resolve_conditioning_image(entry.artifact)
                        frame_index = resolve_frame_index(entry.frame_index, num_frames)
                        output_image = self.preprocess_image(image)
                        if frame_index == 0:
                            update_kwargs["image"] = output_image
                        elif frame_index == -1 or frame_index == num_frames - 1:
                            update_kwargs["last_image"] = output_image
                        else:
                            msg = (
                                f"Attempted to build WAN i2v conditioning. "
                                f"Failed with frame_index={frame_index} because only 0 and -1/{num_frames - 1} "
                                f"are supported."
                            )
                            raise ValueError(msg)
                elif payload.mode is ConditioningMode.VIDEO:
                    logger.warning("Unsupported media_gen_conditioning mode 'video' for WAN i2v; ignoring.")
                else:
                    msg = f"Failed to build WAN video conditioning because mode '{payload.mode.value}' is unsupported."
                    raise ValueError(msg)

        if "image" not in update_kwargs:
            raise ValueError(
                f"{self.driver_namespace}: WAN i2v requires a first-frame image (frame_index=0) "
                "via media_gen_conditioning."
            )

        update_kwargs["height"] = height
        update_kwargs["width"] = width

        return super().denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **update_kwargs,
        )

    def get_resize_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculate the resize dimensions for a given width and height."""
        repo_id = self.pipe.config._name_or_path  # type: ignore[reportAttributeAccessIssue]
        match repo_id:
            case "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
                max_area = 832 * 480  # I2V 480P model
            case "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers":
                max_area = 1280 * 720  # I2V 720P model
            case "Wan-AI/Wan2.2-I2V-A14B-Diffusers":
                max_area = 480 * 832
            case _:
                msg = f"Unsupported model repo_id: {repo_id}."
                raise ValueError(msg)
        aspect_ratio = height / width
        mod_value = self.pipe.vae_scale_factor_spatial * self.pipe.transformer.config.patch_size[1]
        height = round(np.sqrt(max_area * aspect_ratio)) // mod_value * mod_value
        width = round(np.sqrt(max_area / aspect_ratio)) // mod_value * mod_value
        return width, height

    def preprocess_image(self, image: Image) -> Image:
        """Preprocess a PIL image for WAN i2v encoding."""
        # Automatically resize image based on model capabilities
        width, height = self.get_resize_dimensions(image.width, image.height)
        image = image.resize((width, height))
        return image
