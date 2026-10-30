from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast, override

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    GeneratorState,
    ImageMedia,
    VideoMedia,
    read_driver_meta,
)
from modular_diffusion_nodes_library.memory_family import MemoryFamily
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
)
from modular_diffusion_nodes_library.utils.dimension_alignment import DimensionAlignmentResult
from modular_diffusion_nodes_library.utils.flux3_layout import _LAYOUT_CANVAS
from modular_diffusion_nodes_library.utils.torch_utils import no_grad

if TYPE_CHECKING:
    import numpy as np
    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.flux3.denoise import (  # type: ignore[reportMissingImports]
        Flux3DenoiseLoopWrapper,
    )
    from diffusers.modular_pipelines.flux3.modular_pipeline import (  # type: ignore[reportMissingImports]
        Flux3ModularPipeline,
    )
    from diffusers.modular_pipelines.modular_pipeline import (
        ModularPipeline,  # type: ignore[reportMissingImports]
        PipelineState,  # type: ignore[reportMissingImports]
        SequentialPipelineBlocks,  # type: ignore[reportMissingImports]
    )
    from PIL.Image import Image


logger = logging.getLogger("modular_diffusers_nodes_library")
_REFERENCE_CROP_META_KEY = "reference_crop"
_LAYOUT_BOX_KEYS = ("bbox", "src_bbox", "tgt_bbox")


class _ResolvedLayout(NamedTuple):
    prompt: str
    boxes: list[dict] | None


def _unpatchify_latents(latents: torch.Tensor) -> torch.Tensor:
    from diffusers.modular_pipelines.flux3.decoders import (  # type: ignore[reportMissingImports]
        Flux3DecodeStep,
    )

    return Flux3DecodeStep._unpatchify_latents(latents)


def _patchify_latents(latents: torch.Tensor) -> torch.Tensor:
    from diffusers.modular_pipelines.flux3.encoders import (  # type: ignore[reportMissingImports]
        Flux3VaeEncoderStep,
    )

    return Flux3VaeEncoderStep._patchify_latents(latents)


def _make_callback_denoise_step(
    *,
    distilled: bool,
    callback: Callable[..., object] | None,
    start_step: int,
    end_step: int,
    height: int,
    width: int,
) -> Flux3DenoiseLoopWrapper:
    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.flux3.decoders import (  # type: ignore[reportMissingImports]
        Flux3UnpackLatentsStep,
    )
    from diffusers.modular_pipelines.flux3.denoise import (  # type: ignore[reportMissingImports]
        Flux3DenoiseLoopWrapper,
        Flux3DistilledLoopDenoiser,
        Flux3LoopAfterDenoiser,
        Flux3LoopDenoiser,
    )
    from diffusers.modular_pipelines.flux3.modular_pipeline import (  # type: ignore[reportMissingImports]
        Flux3ModularPipeline,
    )
    from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
        PipelineState,
    )

    if distilled:
        denoiser_class = Flux3DistilledLoopDenoiser
        model_name = "flux3-distilled"
    else:
        denoiser_class = Flux3LoopDenoiser
        model_name = "flux3"
    callback_model_name = model_name

    class CallbackDenoiseStep(Flux3DenoiseLoopWrapper):
        model_name = callback_model_name
        block_classes: ClassVar[list[type]] = [denoiser_class, Flux3LoopAfterDenoiser]
        block_names: ClassVar[list[str]] = ["denoiser", "after_denoiser"]

        def __init__(self) -> None:
            super().__init__()
            self.callback = callback
            self.start_step = start_step
            self.end_step = end_step

        @no_grad
        def __call__(self, components: Flux3ModularPipeline, state: PipelineState) -> PipelineState:
            block_state = cast(Any, self.get_block_state(state))
            num_steps = len(block_state.timesteps)
            if num_steps == 0:
                raise ValueError("FLUX.3 has no denoising timesteps.")
            if self.start_step < 0:
                raise ValueError("start_step must be a non-negative integer.")
            if self.end_step != -1 and self.start_step >= self.end_step:
                raise ValueError("start_step must be less than end_step.")

            begin = min(self.start_step, num_steps - 1)
            if self.end_step == -1:
                end = num_steps
            else:
                end = min(max(self.end_step, begin + 1), num_steps)

            components.scheduler.set_begin_index(begin)
            unpack_block = Flux3UnpackLatentsStep()
            with self.progress_bar(total=end - begin) as progress_bar:
                for step_index in range(begin, end):
                    timestep = block_state.timesteps[step_index]
                    components, block_state = cast(
                        tuple[Flux3ModularPipeline, Any],
                        self.loop_step(components, block_state, i=step_index, t=timestep),
                    )
                    progress_bar.update()

                    if self.callback is not None:
                        preview_state = PipelineState()
                        preview_state.set("latents", block_state.latents)
                        preview_state.set("height", height)
                        preview_state.set("width", width)
                        _, preview_state = cast(
                            tuple[Flux3ModularPipeline, PipelineState], unpack_block(components, preview_state)
                        )
                        preview_latents = preview_state.values.get("latents")
                        if not isinstance(preview_latents, torch.Tensor):
                            raise ValueError("FLUX.3 failed to unpack preview latents.")
                        self.callback(
                            components,
                            step_index,
                            timestep,
                            {"latents": preview_latents},
                        )

                    if getattr(components, "_interrupt", False):
                        break

            self.set_block_state(state, block_state)
            return cast(Any, (components, state))

    return CallbackDenoiseStep()


class Flux3LatentPipelineDriver(LatentPipelineDriver):
    memory_family: ClassVar[MemoryFamily | None] = MemoryFamily.IMAGE_DIT_SDPA
    _DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 50
    _is_distilled: ClassVar[bool] = False

    @property
    @override
    def modular_pipe(self) -> ModularPipeline:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]

        return cast(ModularPipeline, self._pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]

        return cast(ModularPipeline, self._pipe)

    @override
    def _prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        return _patchify_latents(latents)

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        if latents_from_pipe.ndim == 3:
            import torch  # type: ignore[reportMissingImports]
            from diffusers.modular_pipelines.flux3.decoders import (  # type: ignore[reportMissingImports]
                Flux3UnpackLatentsStep,
            )

            unpacked = self._call_block(
                Flux3UnpackLatentsStep(),
                latents=latents_from_pipe,
                height=latents_source_shape[-2],
                width=latents_source_shape[-1],
            )
            latents_from_pipe = self._get_required(unpacked, "latents", torch.Tensor)
        return _unpatchify_latents(latents_from_pipe)

    @override
    def _get_spatial_alignment(self) -> int:
        return self.modular_pipe.vae_scale_factor * 2

    @override
    def align_dimensions(self, height: int, width: int, num_frames: int | None = None) -> DimensionAlignmentResult:
        alignment = self._get_spatial_alignment()
        return DimensionAlignmentResult(
            self._ceil_to_alignment(height, alignment),
            self._ceil_to_alignment(width, alignment),
            num_frames,
            None,
        )

    @override
    def validate_dimensions(self, height: int, width: int, num_frames: int | None = None) -> list[str]:
        alignment = self._get_spatial_alignment()
        messages = []
        if height % alignment:
            messages.append(f"height={height} must be divisible by {alignment}.")
        if width % alignment:
            messages.append(f"width={width} must be divisible by {alignment}.")
        return messages

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    def _run_blocks(self, blocks: Any, **kwargs: Any) -> PipelineState:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
            PipelineState,
        )

        state = PipelineState()
        for param in blocks.inputs:
            if param.name in kwargs:
                state.set(param.name, kwargs[param.name], param.kwargs_type)
        with torch.no_grad():
            _, state = blocks(self.modular_pipe, state)
        return state

    @staticmethod
    def _to_pil_image(image: Image | torch.Tensor | np.ndarray) -> Image | np.ndarray:
        import PIL.Image
        import torch  # type: ignore[reportMissingImports]

        if isinstance(image, torch.Tensor):
            image_array = image.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
            return PIL.Image.fromarray(image_array)
        return image

    @classmethod
    def _build_image_reference(cls, payloads: list[MediaGenConditioningPayload] | None) -> list[Image] | None:
        if payloads is None:
            return None

        reference_images: list[Image] = []
        for payload in payloads:
            if payload.mode is not ConditioningMode.IMAGE:
                raise ValueError(f"FLUX.3 reference conditioning does not support '{payload.mode.value}' payloads.")
            for entry in payload.entries:
                reference_images.append(resolve_conditioning_image(entry.artifact))
        return reference_images or None

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.flux3.before_denoise import (  # type: ignore[reportMissingImports]
            Flux3PrepareLatentsStep,
        )
        from diffusers.modular_pipelines.flux3.decoders import (  # type: ignore[reportMissingImports]
            Flux3UnpackLatentsStep,
        )

        height, width = source_shape[-2:]
        messages = self.validate_dimensions(height, width)
        if messages:
            raise ValueError(f"Attempted to create FLUX.3 noise. Failed because {' '.join(messages)}")

        generator = generator_state.to_generator()
        packed_state = self._call_block(
            Flux3PrepareLatentsStep(),
            height=height,
            width=width,
            num_images_per_prompt=1,
            generator=generator,
            batch_size=1,
            dtype=self.modular_pipe.transformer.dtype,
        )
        unpacked_state = self._call_block(
            Flux3UnpackLatentsStep(),
            latents=self._get_required(packed_state, "latents", torch.Tensor),
            height=height,
            width=width,
        )
        latents = _unpatchify_latents(self._get_required(unpacked_state, "latents", torch.Tensor))
        return self._make_latent_artifact(
            latents,
            source_shape=source_shape,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> Image:
        device, dtype = self._get_device_and_type()
        latents = latent.to_torch(device=device, dtype=dtype)
        latents = _patchify_latents(latents)
        reference_crop = read_driver_meta(latent, _REFERENCE_CROP_META_KEY, self.driver_namespace)
        output_state = self._call_block(
            self.modular_pipe.blocks.sub_blocks["decode"],
            latents=latents,
            output_type="pil",
            reference_crop=reference_crop,
        )
        return self._get_required(output_state, "images", list)[0]

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        if isinstance(media, VideoMedia):
            raise NotImplementedError(f"'{self.pipe.__class__.__name__}' does not support video encoding.")

        image = self._to_pil_image(media.image)
        output_state = self._call_block(self.modular_pipe.blocks.sub_blocks["vae_encoder"], image=image)
        image_latents = self._get_required(output_state, "image_cond_latents", list)[0]
        reference_crop = self._get_required(output_state, "reference_crop", list)
        return self._make_latent_artifact(
            _unpatchify_latents(image_latents),
            source_shape=media.source_shape,
            meta={_REFERENCE_CROP_META_KEY: reference_crop},
        )

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.flux3.before_denoise import (  # type: ignore[reportMissingImports]
            Flux3SetTimestepsStep,
        )

        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        height, width = source_shape[-2:]
        timesteps_state = self._call_block(
            Flux3SetTimestepsStep(), num_inference_steps=num_inference_steps, height=height, width=width
        )
        timesteps = self._get_required(timesteps_state, "timesteps", torch.Tensor)
        if len(timesteps) == 0:
            raise ValueError("Attempted to add noise to a FLUX.3 latent. Failed because no timesteps were set.")

        noise_artifact = self.create_noise_latent(source_shape, generator_state)
        latents = latent.to_torch(device=device, dtype=dtype)
        latents = _patchify_latents(latents)
        noise = _patchify_latents(noise_artifact.to_torch(device=device, dtype=dtype))
        init_timestep = min(round(num_inference_steps * strength), num_inference_steps)
        start = max(num_inference_steps - init_timestep, 0)
        latent_timestep = timesteps[start : start + 1]
        # scale_noise reads sigmas from begin_index, not the timestep argument, so point it at the start step.
        self.modular_pipe.scheduler.set_begin_index(start)
        noisy_latents = self.modular_pipe.scheduler.scale_noise(latents, latent_timestep, noise)
        noise_state = GeneratorState.from_artifact(noise_artifact) or generator_state
        return self._make_latent_artifact(
            _unpatchify_latents(noisy_latents),
            source_shape=source_shape,
            upstream=latent,
            meta=noise_state.as_meta(),
        )

    def _resolve_reference_kwargs(self, kwargs: dict[str, Any]) -> Sequence[Image | np.ndarray] | None:
        payloads = normalize_to_payloads(kwargs.pop(MediaGenConditioningKey.OUTPUT, None))
        references = self._build_image_reference(payloads)
        if references:
            kwargs["image"] = references
        elif "image" in kwargs:
            image = kwargs["image"]
            if isinstance(image, list):
                references = [self._to_pil_image(item) for item in image]
            else:
                references = [self._to_pil_image(image)]
            kwargs["image"] = references
        return references

    def _has_prompt_upsampler(self) -> bool:
        components = self.modular_pipe.components
        return (
            components.get("prompt_upsampler") is not None and components.get("prompt_upsampler_processor") is not None
        )

    def _select_core_step(self, has_references: bool) -> type[SequentialPipelineBlocks]:
        from diffusers.modular_pipelines.flux3.modular_blocks_flux3 import (  # type: ignore[reportMissingImports]
            Flux3CoreDenoiseStep,
            Flux3ImageConditionedCoreDenoiseStep,
        )
        from diffusers.modular_pipelines.flux3.modular_blocks_flux3_distilled import (  # type: ignore[reportMissingImports]
            Flux3DistilledCoreDenoiseStep,
            Flux3DistilledImageConditionedCoreDenoiseStep,
        )

        if self._is_distilled:
            if has_references:
                return Flux3DistilledImageConditionedCoreDenoiseStep
            return Flux3DistilledCoreDenoiseStep

        if has_references:
            return Flux3ImageConditionedCoreDenoiseStep
        return Flux3CoreDenoiseStep

    def _build_generation_blocks(
        self,
        core_step: type[SequentialPipelineBlocks],
        callback: Callable[..., object] | None,
        start_step: int,
        end_step: int,
        height: int,
        width: int,
    ) -> SequentialPipelineBlocks:
        from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
            SequentialPipelineBlocks,
        )

        pipe = self.modular_pipe
        inner_blocks = dict(core_step().sub_blocks)
        inner_blocks["denoise"] = _make_callback_denoise_step(
            distilled=self._is_distilled,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            height=height,
            width=width,
        )
        denoise_blocks = SequentialPipelineBlocks.from_blocks_dict(inner_blocks)

        generation_blocks = {}
        if self._has_prompt_upsampler():
            generation_blocks["prompt_upsampler"] = pipe.blocks.sub_blocks["prompt_upsampler"]
        generation_blocks.update(
            {
                "text_encoder": pipe.blocks.sub_blocks["text_encoder"],
                "vae_encoder": pipe.blocks.sub_blocks["vae_encoder"],
                "denoise": denoise_blocks,
            }
        )
        return SequentialPipelineBlocks.from_blocks_dict(generation_blocks)

    @staticmethod
    def _scale_layout_row(row: dict[str, Any]) -> dict[str, Any]:
        from diffusers.modular_pipelines.flux3.prompt_upsampler import (  # type: ignore[reportMissingImports]
            valid_canvas_box,
        )

        scaled = dict(row)
        for key in _LAYOUT_BOX_KEYS:
            box = row.get(key)
            if box is None:
                continue
            valid_box = valid_canvas_box(box)
            if valid_box is None:
                raise ValueError(
                    f"Attempted to read FLUX.3 layout box '{row.get('id')}'. Failed because {key}={box} is not "
                    f"a valid [y_min, x_min, y_max, x_max] box on the 0-{_LAYOUT_CANVAS} canvas."
                )
            scaled[key] = [float(value) / _LAYOUT_CANVAS for value in valid_box]
        return scaled

    @classmethod
    def _parse_layout_prompt(cls, prompt: str) -> _ResolvedLayout:
        from diffusers.modular_pipelines.flux3.prompt_upsampler import (  # type: ignore[reportMissingImports]
            split_caption,
        )

        caption, rows = split_caption(prompt)
        if not rows:
            return _ResolvedLayout(prompt, None)
        is_layout = all(
            isinstance(row, dict) and any(row.get(key) is not None for key in _LAYOUT_BOX_KEYS) for row in rows
        )
        if not is_layout:
            return _ResolvedLayout(prompt, None)
        return _ResolvedLayout(caption, [cls._scale_layout_row(row) for row in rows])

    @classmethod
    def _resolve_layout(cls, prompt: str, boxes: list[dict] | None, upsampler_enabled: bool) -> _ResolvedLayout:
        if not upsampler_enabled:
            if boxes is not None:
                logger.warning(
                    "Attempted to use FLUX.3 layout boxes %s. Failed because the prompt upsampler is not loaded, so "
                    "the boxes are ignored and the prompt is encoded as given. Select a prompt upsampler to use boxes.",
                    [box.get("id") for box in boxes],
                )
            return _ResolvedLayout(prompt, None)
        if boxes is None:
            parsed = cls._parse_layout_prompt(prompt)
        else:
            parsed = _ResolvedLayout(prompt, [cls._scale_layout_row(box) for box in boxes])
        if parsed.boxes is None:
            return _ResolvedLayout(prompt, None)
        return parsed

    @contextmanager
    def _guidance_override(self, guidance_scale: float) -> Iterator[None]:
        pipe = cast(Any, self.modular_pipe)
        guider_to_restore = None
        if not self._is_distilled:
            guider_to_restore = pipe.guider
            pipe.guider = guider_to_restore.new(guidance_scale=float(guidance_scale))
        pipe._interrupt = False
        try:
            yield
        finally:
            pipe._interrupt = False
            if guider_to_restore is not None:
                pipe.guider = guider_to_restore

    @no_grad
    @override
    def denoise_latent(
        self,
        latent: LatentArtifact | InpaintMaskArtifact,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Callable[..., object] | None = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]

        if isinstance(latent, InpaintMaskArtifact):
            raise NotImplementedError(f"{self.driver_namespace} does not support inpainting.")

        source_shape = latent.source_shape
        height, width = source_shape[-2:]
        messages = self.validate_dimensions(height, width)
        if messages:
            raise ValueError(f"Attempted to denoise a FLUX.3 latent. Failed because {' '.join(messages)}")

        references = self._resolve_reference_kwargs(kwargs)
        guidance_scale = kwargs.pop("guidance_scale", 3.5)
        generator = kwargs.pop("generator", generator_state.to_generator())
        prompt = kwargs.pop("prompt", "")
        upsampler_enabled = self._has_prompt_upsampler()
        layout = self._resolve_layout(prompt, kwargs.pop("boxes", None), upsampler_enabled)
        prompt = layout.prompt
        device, dtype = self._get_device_and_type()
        latents = self._prepare_input_latent(latent.to_torch(device=device, dtype=dtype), source_shape)

        core_step = self._select_core_step(has_references=bool(references))
        blocks = self._build_generation_blocks(
            core_step=core_step,
            callback=callback,
            start_step=start_step,
            end_step=num_inference_steps if return_fully_denoised else end_step,
            height=height,
            width=width,
        )

        run_kwargs = {
            **kwargs,
            "prompt": prompt,
            "num_inference_steps": num_inference_steps,
            "num_images_per_prompt": 1,
            "height": height,
            "width": width,
            "latents": latents,
            "generator": generator,
        }
        if layout.boxes is not None:
            run_kwargs["boxes"] = layout.boxes

        with self._guidance_override(guidance_scale):
            state = self._run_blocks(blocks, **run_kwargs)

        output_latents = self._get_required(state.values, "latents", torch.Tensor)
        return self._make_latent_artifact(
            _unpatchify_latents(output_latents),
            source_shape=source_shape,
            upstream=latent,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )


class Flux3DistilledLatentPipelineDriver(Flux3LatentPipelineDriver):
    _DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 16
    _is_distilled: ClassVar[bool] = True
