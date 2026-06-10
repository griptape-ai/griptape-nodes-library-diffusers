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
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import DecodeResult
from modular_diffusion_nodes_library.latent_pipeline_drivers.wan import WanTextToVideoLatentPipelineDriver
from modular_diffusion_nodes_library.utils.conditioning_utils import resolve_conditioning_image

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
    def create_noise_latent(self, source_shape: tuple[int, ...], seed: int) -> LatentArtifact:
        width, height = self.get_resize_dimensions(source_shape[-1], source_shape[-2])
        resized_source_shape = (*source_shape[:-2], height, width)
        resized_output = super().create_noise_latent(resized_source_shape, seed)
        return self._make_latent_artifact(
            resized_output.to_torch(), source_shape=source_shape, upstream=resized_output,
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        frames = super().decode_latent(latent)
        source_shape = latent.source_shape
        output_frames = [
            frame.resize((source_shape[-1], source_shape[-2]), Resampling.LANCZOS) for frame in frames
        ]
        return output_frames

    @override
    def encode_video(self, frames: list[Image], source_shape: tuple[int, ...]) -> LatentArtifact:
        input_frames = [self.preprocess_image(frame) for frame in frames]
        return super().encode_video(input_frames, source_shape)

    @override
    def denoise_latent(
        self,
        latent: LatentArtifact,
        num_inference_steps: int,
        seed: int = 0,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        """Denoise a WAN i2v video latent."""

        update_kwargs = kwargs.copy()
        original_source_shape = latent.source_shape

        media_gen_conditioning = update_kwargs.pop("media_gen_conditioning", None)
        media_gen_conditioning_list = self._ensure_media_gen_conditioning_list(media_gen_conditioning)

        # Extract image/last_image from media_gen_conditioning entries for the i2v pipeline.
        # If first-frame conditioning is provided, the source_shape passed to super() needs
        # to match the conditioning image dimensions so the underlying pipe runs at those
        # dims. The returned artifact is restored to the caller's original source_shape so
        # downstream nodes (decode) see the requested dimensions.
        if media_gen_conditioning_list is not None:
            num_frames = latent.source_shape[-3]
            source_shape = latent.source_shape
            rebuilt = False
            for media_gen_conditioning in media_gen_conditioning_list:
                mode = media_gen_conditioning.get("mode")
                if mode == "image":
                    for image_item in media_gen_conditioning.get("images", []):
                        image = resolve_conditioning_image(image_item.get("image"))
                        frame_index = image_item.get("frame_index", None)
                        output_image = self.preprocess_image(image)
                        if frame_index == 0:
                            update_kwargs["image"] = output_image
                            source_shape = (*source_shape[:-2], output_image.height, output_image.width)
                            rebuilt = True
                        elif frame_index == -1 or frame_index == num_frames - 1:
                            update_kwargs["last_image"] = output_image
                        else:
                            raise ValueError(
                                f"Unsupported frame_index {frame_index} for WAN i2v conditioning. Only 0 and -1/{num_frames - 1} are supported."
                            )
                elif mode == "video":
                    logger.warning("Unsupported media_gen_conditioning mode '%s' for WAN i2v; ignoring.", mode)
            if rebuilt:
                device, dtype = self._get_device_and_type()
                latent = LatentArtifact.from_torch(
                    latent.to_torch(device=device, dtype=dtype), source_shape=source_shape, meta=latent.meta
                )

        denoised = super().denoise_latent(
            latent,
            num_inference_steps,
            seed=seed,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **update_kwargs,
        )
        return self._make_latent_artifact(
            denoised.to_torch(),
            source_shape=original_source_shape,
            upstream=denoised,
        )

    def get_resize_dimensions(self, width: int, height: int) -> tuple[int, int]:
        """Calculate the resize dimensions for a given width and height."""
        repo_id = self.pipe.config._name_or_path
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
