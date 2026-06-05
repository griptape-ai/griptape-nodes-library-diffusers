import logging
from typing import Any, ClassVar, override

import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
    InputParam,
    OutputParam,
)
from diffusers.modular_pipelines.wan.before_denoise import (  # type: ignore[reportMissingImports]
    WanPrepareLatentsStep,
    WanSetTimestepsStep,
)
from diffusers.modular_pipelines.wan.decoders import WanVaeDecoderStep  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.encoders import encode_vae_image  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_blocks_wan import WanBlocks  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_blocks_wan22 import Wan22Blocks  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_pipeline import WanModularPipeline  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import DecodeResult, LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class _WanEncodeVideoStep(ModularPipelineBlocks):
    """Encode a list of PIL frames into a normalised WAN video latent ``[B, C, T, H/vsf, W/vsf]``."""

    model_name = "wan"

    @property
    def inputs(self) -> list[InputParam]:
        return [InputParam("frames", required=True), InputParam("generator")]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "video_latents",
                type_hint=torch.Tensor,
                description="Normalised video latent in the WAN scheduler's input space.",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: WanModularPipeline, state: PipelineState
    ) -> tuple[WanModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        device = components._execution_device
        vae_dtype = components.vae.dtype

        # video_processor.preprocess returns [T, C, H, W]; rearrange to [B, C, T, H, W].
        video_tensor = components.video_processor.preprocess(block_state.frames)
        video_tensor = video_tensor.permute(1, 0, 2, 3).unsqueeze(0).to(device=device, dtype=torch.float32)

        block_state.video_latents = encode_vae_image(
            video_tensor=video_tensor,
            vae=components.vae,
            generator=block_state.generator,
            device=device,
            dtype=vae_dtype,
            latent_channels=components.num_channels_latents,
        )

        self.set_block_state(state, block_state)
        return components, state


class WanTextToVideoLatentPipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True
    video_fps: ClassVar[int] = 16

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @staticmethod
    def _ensure_media_gen_conditioning_list(media_gen_conditioning: Any) -> list[dict[str, Any]] | None:
        if media_gen_conditioning is None:
            return None
        if not isinstance(media_gen_conditioning, list):
            return [media_gen_conditioning]
        return media_gen_conditioning

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        if getattr(self.pipe, "transformer_2", None) is not None:
            return Wan22Blocks().init_pipeline()
        return WanBlocks().init_pipeline()

    @override
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:
        _, dtype = self._get_device_and_type()
        generator = torch.Generator().manual_seed(seed)
        prepare_latents = WanPrepareLatentsStep()
        num_frames, height, width = latents_source_shape[-3], latents_source_shape[-2], latents_source_shape[-1]
        output_state = self._call_block(
            prepare_latents,
            height=height,
            width=width,
            num_frames=num_frames,
            num_videos_per_prompt=1,
            generator=generator,
            batch_size=1,
            dtype=dtype,
        )
        latents = output_state.get("latents")
        return latents

    @override
    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """WAN pipelines return video frames under ``.frames`` instead of ``.images``."""
        return pipe_output.frames

    @override
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> DecodeResult:
        """Decode a 5-D WAN video latent and return the video."""
        device, dtype = self._get_device_and_type()
        latents = latents.to(device=device, dtype=dtype)
        vae_decoder_step = WanVaeDecoderStep()
        output_state = self._call_block(
            vae_decoder_step,
            latents=latents,
            output_type="pil",
        )
        video_frames = output_state.get("videos")[0]
        return video_frames

    @override
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:
        # Currently VAE encoder switches input to video for WAN pipelines so this method is not implemented.
        raise NotImplementedError(
            f"Pipeline '{self.pipe.__class__.__name__}' does not support image encoding. Use a video input instead."
        )

    @override
    def encode_video(self, frames: list[Image]) -> torch.Tensor:
        """Encode a list of PIL images (video frames) as a normalised WAN video latent (5-D ``[B, C, T, H/vsf, W/vsf]``)."""
        vae_encoder = _WanEncodeVideoStep()
        output = self._call_block(vae_encoder, frames=frames, generator=None)
        return self._get_required(output, "video_latents", torch.Tensor)

    @override
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        """Add noise to a WAN video latent using modular blocks."""
        set_timesteps = WanSetTimestepsStep()
        timesteps_state = self._call_block(set_timesteps, num_inference_steps=num_inference_steps)
        timesteps = timesteps_state.get("timesteps")

        if timesteps is None or len(timesteps) == 0:
            raise ValueError("WANSetTimestepsStep did not return valid timesteps.")

        # Generate noise matching the latent shape directly
        noise = self.create_noise_latent(latents_source_shape, seed)

        # Compute timestep based on strength
        init_timestep = min(num_inference_steps * strength, num_inference_steps)
        t_start = int(max(num_inference_steps - init_timestep, 0))
        latent_timestep = timesteps[t_start * self.pipe.scheduler.order :][:1]

        # Scale noise at the target timestep
        with torch.no_grad():
            result = self.pipe.scheduler.add_noise(latents, noise, latent_timestep)
        return result

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
        """Denoise a WAN video latent"""

        update_kwargs = kwargs.copy()
        update_kwargs["num_frames"] = latents_source_shape[-3]

        update_kwargs.pop(
            "media_gen_conditioning", None
        )  # WAN t2v does not use media_gen_conditioning, so we pop it from kwargs to avoid passing unexpected arguments to the pipeline.
        output_latents = super().denoise_latent(
            latents,
            latents_source_shape,
            num_inference_steps,
            seed=seed,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
            **update_kwargs,
        )
        return output_latents
