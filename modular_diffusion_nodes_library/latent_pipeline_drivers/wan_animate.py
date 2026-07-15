import logging
from typing import Any, override

import PIL.Image
import torch
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_blocks_wan22 import Wan22Blocks  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    DecodeResult,
    GeneratorState,
    ImageMedia,
    VideoMedia,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.wan import WanTextToVideoLatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


def _video_to_mask_frames(
    frames: list[PIL.Image.Image],
    num_frames: int,
    height: int,
    width: int,
) -> list[PIL.Image.Image]:
    """Convert resolved video frames to grayscale mask frames; unspecified positions default to white."""
    white = PIL.Image.new("L", (width, height), 255)
    result: list[PIL.Image.Image] = [white for _ in range(num_frames)]
    for i, frame in enumerate(frames):
        if i >= num_frames:
            break
        result[i] = frame.convert("L").resize((width, height))
    return result


class WanAnimateLatentPipelineDriver(WanTextToVideoLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        # WanAnimatePipeline is a Wan2.2-family model. Wan22Blocks VAE blocks are
        # compatible because the VAE type (AutoencoderKLWan) is unchanged; only the
        # transformer architecture differs (WanAnimateTransformer3DModel).
        return Wan22Blocks().init_pipeline()

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        extended_shape = (
            *source_shape[:-3],
            source_shape[-3] + self._pipe.vae_scale_factor_temporal,
            *source_shape[-2:],
        )
        noise = super().create_noise_latent(extended_shape, generator_state)
        noise_generator_state = GeneratorState.from_artifact(noise) or generator_state
        return self._make_latent_artifact(
            noise.to_torch(),
            source_shape=source_shape,
            meta=noise_generator_state.as_meta(),
        )

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        base_artifact = super().encode_media(media, generator_state)
        base_tensor = base_artifact.to_torch()
        # Prepend a copy of the first encoded latent frame as the reference conditioning slot.
        # All public latents for this driver have T_lat + 1 temporal frames; pos 0 is
        # stripped in decode_latent before VAE decoding.
        extended = torch.cat([base_tensor[:, :, :1, :, :], base_tensor], dim=2)
        return self._make_latent_artifact(extended, source_shape=base_artifact.source_shape, upstream=base_artifact)

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        device, dtype = self._get_device_and_type()
        tensor = latent.to_torch(device=device, dtype=dtype)
        # Strip the reference conditioning slot at pos 0 before VAE decoding.
        stripped = tensor[:, :, 1:, :, :]
        stripped_artifact = self._make_latent_artifact(stripped, source_shape=latent.source_shape)
        return super().decode_latent(stripped_artifact)

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
        mask_video_frames = kwargs.pop("mask_video", None)

        num_frames = latent.source_shape[-3]
        height = latent.source_shape[-2]
        width = latent.source_shape[-1]

        if mask_video_frames is not None:
            kwargs["mask_video"] = _video_to_mask_frames(mask_video_frames, num_frames, height, width)

        kwargs.pop("media_gen_conditioning", None)
        kwargs["height"] = height
        kwargs["width"] = width

        # Force single-segment mode: segment_frame_length equals the full video length so the
        # pipeline runs exactly one denoising pass and returns a latent via output_type="latent".
        # This keeps WAN Animate consistent with other drivers (one denoise call → one latent).
        # Future: to support longer videos across multiple segments, use callback_on_step_end to
        # capture each segment's normalized latent at i == pipe._num_timesteps - 1, concatenate
        # with pixel-level overlap trimming (prev_segment_conditioning_frames frames per boundary),
        # and re-normalize before storing in the artifact.
        kwargs["segment_frame_length"] = num_frames

        # Skip WanTextToVideoLatentPipelineDriver.denoise_latent — it injects num_frames which
        # WanAnimatePipeline does not accept.
        result = super(WanTextToVideoLatentPipelineDriver, self).denoise_latent(
            latent,
            num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **kwargs,
        )

        # (z_raw = z_norm * std + mean from WanAnimatePipeline's output path).
        # Reverse de-whitening to restore ~N(0, 1) for the public latent contract.
        device, dtype = self._get_device_and_type()
        vae = self._pipe.vae
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1).to(device, dtype)
        recip_std = (1.0 / torch.tensor(vae.config.latents_std)).view(1, -1, 1, 1, 1).to(device, dtype)
        normalized = (result.to_torch(device=device, dtype=dtype) - mean) * recip_std

        result_generator_state = GeneratorState.from_artifact(result) or generator_state
        return self._make_latent_artifact(
            normalized,
            source_shape=latent.source_shape,
            upstream=result,
            meta=result_generator_state.as_meta(),
        )
