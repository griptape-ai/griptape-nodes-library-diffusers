from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, cast, override

from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.qwenimage21.before_denoise import (  # type: ignore[reportMissingImports]
    QwenImage21PrepareLatentsStep,
    QwenImage21SetTimestepsStep,
)
from diffusers.modular_pipelines.qwenimage21.decoders import unpack_latents  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.qwenimage21.encoders import (  # type: ignore[reportMissingImports]
    QwenImage21ProcessImagesInputStep,
    QwenImage21VaeEncoderStep,
)
from diffusers.modular_pipelines.qwenimage21.inputs import pack_latents  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.qwenimage21.modular_blocks_qwenimage21 import (  # type: ignore[reportMissingImports]
    QwenImage21AutoBlocks,
)
from diffusers.modular_pipelines.qwenimage21.modular_blocks_qwenimage21_turbo import (  # type: ignore[reportMissingImports]
    QwenImage21TurboAutoBlocks,
)

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    DecodeOutputType,
    DecodeResult,
    GeneratorState,
    ImageMedia,
    VideoMedia,
)
from modular_diffusion_nodes_library.memory_family import MemoryFamily
from modular_diffusion_nodes_library.misc.schedule_utils import preserved_scheduler_state
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import normalize_to_payloads
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
)

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
        InputParam,
        OutputParam,
    )
    from PIL.Image import Image


class _QwenImage21EncodeImageStep(SequentialPipelineBlocks):
    """RGBA preprocessing followed by VAE encode for a PIL condition image."""

    block_classes = [QwenImage21ProcessImagesInputStep, QwenImage21VaeEncoderStep]
    block_names = ["preprocess", "encode"]


class _QwenImage21AddNoiseStep(ModularPipelineBlocks):
    """Scale packed noise to the sigma at ``start_step`` on the Qwen Image 2.1 sampling schedule."""

    @property
    def inputs(self) -> list[InputParam]:
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            InputParam,
        )

        return [
            InputParam("latents", required=True),
            InputParam("noise", required=True),
            InputParam("num_inference_steps", required=True),
            InputParam("start_step", required=True),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            OutputParam,
        )

        return [OutputParam("noisy_latents", type_hint=torch.Tensor, description="Packed latents with noise added.")]

    def __call__(self, components: Any, state: PipelineState) -> tuple[Any, PipelineState]:
        block_state = cast(Any, self.get_block_state(state))
        # The set-timesteps step applies the scheduler shift and falls back to the linear grid when no
        # sigmas are given, so only a configured grid (Turbo) needs passing in.
        configured_sigmas = getattr(components.config, "sample_sigmas", None)
        if configured_sigmas is not None:
            state.set("sigmas", list(configured_sigmas))
        _, state = QwenImage21SetTimestepsStep()(components, state)  # type: ignore[reportOperatorIssue]
        timesteps = state.get("timesteps")
        start_step = block_state.start_step
        if timesteps is None:
            raise ValueError(
                f"Attempted to add noise for Qwen Image 2.1. Failed with start_step={start_step} because "
                "the set-timesteps step produced no timesteps."
            )
        if start_step < 0 or start_step >= len(timesteps):
            raise ValueError(
                f"Attempted to add noise for Qwen Image 2.1. Failed with start_step={start_step} and "
                f"schedule_length={len(timesteps)} because the step index is outside the sampling schedule."
            )

        components.scheduler.set_begin_index(start_step)
        timestep = timesteps[start_step].expand(block_state.latents.shape[0])
        block_state.noisy_latents = components.scheduler.scale_noise(block_state.latents, timestep, block_state.noise)

        self.set_block_state(state, block_state)
        return components, state


class QwenImage21LatentPipelineDriver(LatentPipelineDriver):
    memory_family: ClassVar[MemoryFamily | None] = MemoryFamily.IMAGE_DIT_SDPA
    _DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 40

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _get_spatial_alignment(self) -> int:
        # Qwen Image 2.1 groups 2x2 VAE latents into one transformer token.
        return self.pipe.vae_scale_factor * 2

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        sample_sigmas = getattr(self.pipe.config, "sample_sigmas", None)
        if sample_sigmas is not None:
            modular_pipe = QwenImage21TurboAutoBlocks().init_pipeline()
            modular_pipe.update_components(sample_sigmas=sample_sigmas)
            return modular_pipe
        return QwenImage21AutoBlocks().init_pipeline()

    @override
    def _prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        return pack_latents(latents.unsqueeze(2))

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        unpacked = unpack_latents(
            latents_from_pipe,
            latents_source_shape[-2],
            latents_source_shape[-1],
            self.pipe.vae_scale_factor,
        )
        return unpacked.squeeze(2)

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]

        _, dtype = self._get_device_and_type()
        height, width = source_shape[-2:]
        generator = generator_state.to_generator()
        output_state = self._call_block(
            QwenImage21PrepareLatentsStep(),
            height=height,
            width=width,
            batch_size=1,
            num_images_per_prompt=1,
            generator=generator,
            dtype=dtype,
        )
        packed = self._get_required(output_state, "latents", torch.Tensor)
        latents = self.prepare_output_latent(packed, source_shape)
        return self._make_latent_artifact(
            latents,
            source_shape=source_shape,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def decode_latent(self, latent: LatentArtifact, output_type: DecodeOutputType = "pil") -> DecodeResult:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        output_state = self._call_block(
            self.modular_pipe.blocks.sub_blocks["decode"], latents=latents, output_type=output_type
        )
        return self._get_decoded_media(output_state, "images", output_type, is_video=False, expected_channels=4)

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        start_step: int,
    ) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]

        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        noise_artifact = self.create_noise_latent(source_shape, generator_state)
        clean_latents = self._prepare_input_latent(latent.to_torch(device=device, dtype=dtype), source_shape)
        noise = self._prepare_input_latent(noise_artifact.to_torch(device=device, dtype=dtype), source_shape)

        with preserved_scheduler_state(self.pipe.scheduler):
            output_state = self._call_block(
                _QwenImage21AddNoiseStep(),
                latents=clean_latents,
                noise=noise,
                num_inference_steps=num_inference_steps,
                start_step=start_step,
            )
        noisy_latents = self._get_required(output_state, "noisy_latents", torch.Tensor)
        noise_state = GeneratorState.from_artifact(noise_artifact) or generator_state
        return self._make_latent_artifact(
            self.prepare_output_latent(noisy_latents, source_shape),
            source_shape=source_shape,
            upstream=latent,
            meta=noise_state.as_meta(),
        )

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]

        if isinstance(media, VideoMedia):
            raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video.")
        image = media.image
        generator = generator_state.to_generator()
        if isinstance(image, torch.Tensor):
            encoded_state = self._call_block(
                QwenImage21VaeEncoderStep(),
                processed_image=[image],
                generator=generator,
            )
        else:
            # The VAE reads four channels; an RGB image gets a solid alpha.
            encoded_state = self._call_block(
                _QwenImage21EncodeImageStep(),
                resized_image=[image.convert("RGBA")],
                generator=generator,
            )
        image_latents = encoded_state.get("image_latents")
        if (
            not isinstance(image_latents, list)
            or len(image_latents) != 1
            or not isinstance(image_latents[0], torch.Tensor)
        ):
            raise ValueError(
                f"Attempted to encode media with {self.driver_namespace}. Failed because the Qwen Image 2.1 "
                f"VAE encoder returned image_latents={type(image_latents).__name__} instead of one tensor."
            )
        return self._make_latent_artifact(image_latents[0].squeeze(2), source_shape=media.source_shape)

    @staticmethod
    def _images_from_conditioning(payload: Any) -> list[Image]:
        payloads = normalize_to_payloads(payload)
        if payloads is None:
            return []
        images: list[Image] = []
        for item in payloads:
            if item.mode is not ConditioningMode.IMAGE:
                raise ValueError(
                    f"Attempted to resolve Qwen Image 2.1 conditioning. Failed with mode='{item.mode.value}' "
                    "because only image conditioning is supported."
                )
            images.extend(resolve_conditioning_image(entry.artifact, preserve_alpha=True) for entry in item.entries)
        if len(images) > 10:
            raise ValueError(
                f"Attempted to resolve Qwen Image 2.1 conditioning. Failed with image_count={len(images)} "
                "because the pipeline accepts at most 10 reference images."
            )
        return images

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
        payload = kwargs.pop(MediaGenConditioningKey.OUTPUT, None)
        if payload is not None:
            images = self._images_from_conditioning(payload)
            if images:
                kwargs["image"] = images
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
