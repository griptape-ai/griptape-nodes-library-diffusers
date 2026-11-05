"""MiniMax-H3 ref2va driver for joint video + audio generation from image/video references."""

from __future__ import annotations

from typing import Any, ClassVar, cast, override

from diffusers.modular_pipelines.minimax_h3.denoise import (  # type: ignore[reportMissingImports]
    MiniMaxH3LoopSchedulerStep,
    MiniMaxH3Ref2VALoopDenoiser,
)
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (  # type: ignore[reportMissingImports]
    MiniMaxH3Ref2VACoreDenoiseStep,
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    SequentialPipelineBlocks,
)
from griptape.artifacts.video_url_artifact import VideoUrlArtifact

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.latent_pipeline_drivers.minimax_h3 import (
    AUDIO_LATENTS_META_KEY,
    MiniMaxH3LatentPipelineDriver,
    _MiniMaxH3CallbackDenoiseStep,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import ConditioningMode, resolve_conditioning_image
from modular_diffusion_nodes_library.utils.minimax_h3_ref2va_utils import (
    get_reference_video_duration_seconds,
    load_reference_video,
    plan_reference_video_trim,
)


class _MiniMaxH3Ref2VACallbackDenoiseStep(_MiniMaxH3CallbackDenoiseStep):
    """MiniMax-H3 callback loop targeting the ref2va transformer partition."""

    block_classes: ClassVar[list[type]] = [MiniMaxH3Ref2VALoopDenoiser, MiniMaxH3LoopSchedulerStep]


class MiniMaxH3Ref2VALatentPipelineDriver(MiniMaxH3LatentPipelineDriver):
    """MiniMax-H3 ref2va driver using the dedicated reference transformer weights."""

    @staticmethod
    def _build_references(payloads: list[MediaGenConditioningPayload] | None) -> list[Any]:
        from diffusers.modular_pipelines.minimax_h3.references import (
            MiniMaxH3ImageReference,
            MiniMaxH3VideoReference,
        )

        if not payloads:
            raise ValueError(
                "Attempted to build MiniMax-H3 ref2va references. Failed because no references were given."
            )

        image_entries = [
            entry for payload in payloads if payload.mode is ConditioningMode.IMAGE for entry in payload.entries
        ]
        video_entries = [
            entry for payload in payloads if payload.mode is ConditioningMode.VIDEO for entry in payload.entries
        ]
        references: list[Any] = [
            MiniMaxH3ImageReference(image=resolve_conditioning_image(entry.artifact)) for entry in image_entries
        ]

        durations = [
            get_reference_video_duration_seconds(cast(VideoUrlArtifact, entry.artifact)) for entry in video_entries
        ]
        trim_plan = plan_reference_video_trim(durations)
        for entry, kept_seconds in zip(video_entries, trim_plan.kept_seconds, strict=True):
            if kept_seconds is None:
                continue
            loaded_video = load_reference_video(cast(VideoUrlArtifact, entry.artifact))
            fps = loaded_video.reference.fps
            assert fps is not None
            frame_count = round(kept_seconds * fps)
            references.append(
                MiniMaxH3VideoReference(frames=loaded_video.reference.frames[:frame_count], fps=fps, audio=None)
            )

        return references

    @property
    @override
    def driver_namespace(self) -> str:
        # fl2va and ref2va share the public video/audio latent metadata contract.
        return MiniMaxH3LatentPipelineDriver.__name__

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
        import torch  # type: ignore[reportMissingImports]

        if isinstance(latent, InpaintMaskArtifact):
            raise NotImplementedError(f"{self.driver_namespace} does not support inpainting.")
        if return_fully_denoised:
            raise NotImplementedError(
                f"{self.driver_namespace} does not support 'return_fully_denoised' because "
                "MiniMax-H3's two schedulers cannot express a non-contiguous schedule."
            )

        payloads = normalize_to_payloads(kwargs.pop("references", None))
        if not payloads:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to run ref2va denoising. Failed because at least one "
                "reference is required."
            )

        references = self._build_references(payloads)
        pipe = cast(Any, self.modular_pipe)
        device, _ = self._get_device_and_type()
        source_shape = latent.source_shape
        geometry = self._latent_geometry(source_shape)

        update_kwargs = kwargs.copy()
        update_kwargs["references"] = references
        update_kwargs["num_frames"] = geometry["num_frames"]
        update_kwargs["height"] = source_shape[-2]
        update_kwargs["width"] = source_shape[-1]

        generator = update_kwargs.pop("generator", generator_state.to_generator())
        video_latents_in = latent.to_torch(device=device, dtype=torch.float32)
        audio_latents = self._read_audio_latents(latent)
        if audio_latents is None and start_step > 0:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to resume a MiniMax-H3 denoise at step "
                f"{start_step}. Failed because the input latent carries no audio latent, so the "
                "soundtrack would restart from pure noise mid-schedule. Chain partial denoise "
                "directly from a previous Generate Media Latents run."
            )
        if audio_latents is not None:
            audio_latents = audio_latents.to(device=device, dtype=torch.float32)

        inner_blocks = dict(MiniMaxH3Ref2VACoreDenoiseStep().sub_blocks)
        denoise_step = _MiniMaxH3Ref2VACallbackDenoiseStep()
        denoise_step.callback = callback
        denoise_step.start_step = start_step
        denoise_step.end_step = end_step
        inner_blocks["denoise"] = denoise_step
        denoise_bundle = SequentialPipelineBlocks.from_blocks_dict(inner_blocks)

        outer_blocks = {
            "before_encode": pipe.blocks.sub_blocks["before_encode"],
            "text_encoder": pipe.blocks.sub_blocks["text_encoder"],
            "vae_encoder": pipe.blocks.sub_blocks["vae_encoder"],
            "denoise": denoise_bundle,
        }
        blocks = SequentialPipelineBlocks.from_blocks_dict(outer_blocks)

        pipe._interrupt = False
        try:
            state = self._run_blocks(
                blocks,
                latents=video_latents_in,
                audio_latents=audio_latents,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **update_kwargs,
            )
        finally:
            pipe._interrupt = False

        video_latents = self._get_required(state.values, "latents", torch.Tensor)
        audio_out = self._get_required(state.values, "audio_latents", torch.Tensor)

        return self._make_latent_artifact(
            video_latents,
            source_shape=source_shape,
            upstream=latent,
            meta={
                AUDIO_LATENTS_META_KEY: audio_out,
                **GeneratorState.from_generator(generator).as_meta(),
            },
        )
