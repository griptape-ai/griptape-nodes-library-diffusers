"""MiniMax-H3 driver: joint video + audio generation from a Modular-Diffusers-only integration.

MiniMax-H3 is the first model in this library with no ``DiffusionPipeline`` half, so this driver
owns the denoise loop instead of delegating to ``super().denoise_latent()``. That is what
``base_driver.denoise_latent`` asks for when it sees a ``ModularPipeline``.

Owning the loop is what makes partial denoise, live preview and mid-run cancellation available here
even though HunyuanVideo 1.5 (whose loop is sealed inside diffusers) cannot offer the latter two.

Two latent streams share one public artifact: the video latent is the artifact's tensor, and the
audio latent rides in this driver's namespaced ``meta`` sub-bag. That channel is only safe on a
direct Generate -> Decode edge. Nodes that rebuild meta from scratch (Empty Latent, save/load) drop
the audio, and latent math is worse: it sums the video latent while keeping the left operand's
*unsummed* audio, so the audio is present but no longer matches the picture. ``decode_latent``
therefore checks a fingerprint of the video latent the audio was paired with, and refuses rather
than muxing a desynchronised soundtrack.
"""

import logging
import math
from typing import Any, ClassVar, cast, override

import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.minimax_h3.denoise import (  # type: ignore[reportMissingImports]
    MiniMaxH3DenoiseLoopWrapper,
    MiniMaxH3LoopDenoiser,
    MiniMaxH3LoopSchedulerStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # type: ignore[reportMissingImports]
    MiniMaxH3ModularPipeline,
)
from diffusers.modular_pipelines.minimax_h3.packing import (  # type: ignore[reportMissingImports]
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    align_num_frames,
    audio_latent_num_frames,
    patchify_video_latents,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
    InputParam,
    OutputParam,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from diffusers.utils.torch_utils import randn_tensor  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    DecodeResult,
    GeneratorState,
    ImageMedia,
    VideoMedia,
    read_driver_meta,
)
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import (
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    resolve_conditioning_image,
    resolve_frame_index,
)

logger = logging.getLogger("modular_diffusers_nodes_library")

#: Key under which the audio latent rides in this driver's namespaced ``meta`` sub-bag.
AUDIO_LATENTS_META_KEY = "audio_latents"

#: Fingerprint of the video latent the audio latent was denoised with. Latent math merges meta
#: left-operand-wins over a *shallow* copy, so a summed video latent keeps the left operand's
#: unsummed audio: the audio is still present but no longer corresponds to the video. Comparing
#: fingerprints at decode time turns that from a silently desynchronised soundtrack into an error.
AUDIO_PAIRED_WITH_META_KEY = "audio_paired_with"


def _video_fingerprint(tensor: torch.Tensor) -> tuple[tuple[int, ...], float, float]:
    """Cheap value-sensitive fingerprint of a video latent.

    Shape alone would not do: latent math preserves shape and changes only values.
    """
    flat = tensor.detach().to(device="cpu", dtype=torch.float64)
    return (tuple(tensor.shape), float(flat.sum()), float(flat.square().sum()))


def _fingerprints_match(
    left: tuple[tuple[int, ...], float, float] | None,
    right: tuple[tuple[int, ...], float, float],
) -> bool:
    if left is None:
        return False
    if tuple(left[0]) != tuple(right[0]):
        return False
    return math.isclose(left[1], right[1], rel_tol=1e-9, abs_tol=1e-6) and math.isclose(
        left[2], right[2], rel_tol=1e-9, abs_tol=1e-6
    )


class _MiniMaxH3PrepareNoiseStep(ModularPipelineBlocks):
    """Draw the video and audio noise, leaving the video latent unpacked.

    Upstream's ``MiniMaxH3PrepareLatentsStep`` returns patchified rows, which the public latent
    surface forbids. This draws the same two tensors in the same order — video then audio, which is
    what the request generator reproduces — and returns the video one as a 5-D latent.
    """

    model_name = "minimax-h3"

    @property
    def inputs(self) -> list[InputParam]:
        return [
            InputParam("num_latent_frames", required=True),
            InputParam("latent_height", required=True),
            InputParam("latent_width", required=True),
            InputParam("num_audio_latents", required=True),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Unpacked video noise of shape (1, C, num_latent_frames, latent_height, latent_width).",
            ),
            OutputParam(
                "audio_latents",
                type_hint=torch.Tensor,
                description="Audio noise of shape (2, audio_latent_channels, num_audio_latents).",
            ),
        ]

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
        block_state = cast(Any, self.get_block_state(state))
        device = components._execution_device

        block_state.latents = randn_tensor(
            (
                1,
                components.vae_latent_channels,
                block_state.num_latent_frames,
                block_state.latent_height,
                block_state.latent_width,
            ),
            generator=block_state.generator,
            device=device,
            dtype=torch.float32,
        )
        # Drawn in row layout upstream, then reshaped to the (2, C, N) shape `prepare_latents`
        # accepts back, so the generator sees the same draw either way.
        audio_rows = randn_tensor(
            (block_state.num_audio_latents * MINIMAX_H3_AUDIO_CHANNELS, components.audio_latent_channels),
            generator=block_state.generator,
            device=device,
            dtype=torch.float32,
        )
        block_state.audio_latents = audio_rows.reshape(
            MINIMAX_H3_AUDIO_CHANNELS, block_state.num_audio_latents, components.audio_latent_channels
        ).permute(0, 2, 1)

        self.set_block_state(state, block_state)
        return components, state


class _MiniMaxH3CallbackDenoiseStep(MiniMaxH3DenoiseLoopWrapper):
    """``MiniMaxH3DenoiseStep`` plus partial denoise, a step-end callback and an interrupt break.

    ``block_classes`` / ``block_names`` are re-declared rather than inherited because
    ``LoopSequentialPipelineBlocks.__init__`` builds ``sub_blocks`` from them.
    """

    block_classes: ClassVar[list[type]] = [MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep]
    block_names: ClassVar[list[str]] = ["denoiser", "update"]

    #: Assigned per-instance after construction. A block's ``__init__`` takes no arguments and its
    #: signature is introspected for config, so these must not become constructor kwargs.
    callback: Any = None
    start_step: int = 0
    end_step: int = -1

    @property
    def description(self) -> str:
        return (
            "Runs the MiniMax-H3 denoising loop over an optional step window, with a step-end "
            "callback and cancellation support."
        )

    def _resolve_window(self, num_steps: int) -> tuple[int, int]:
        """Clamp the requested step window against the schedule's real length.

        ``num_inference_steps`` counts sigma grid points including the terminal zero, so the
        schedule drives one fewer model evaluation than the UI number. ``end_step`` equal to the UI
        value is therefore normal and must clamp rather than raise.
        """
        if self.start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {self.start_step}.")
        if self.end_step != -1 and self.start_step >= self.end_step:
            raise ValueError(f"start_step must be less than end_step, got {self.start_step} and {self.end_step}.")

        begin = min(self.start_step, num_steps - 1)
        if self.end_step == -1:
            end = num_steps
        else:
            end = min(max(self.end_step, begin + 1), num_steps)
        return begin, end

    @torch.no_grad()
    def __call__(
        self, components: MiniMaxH3ModularPipeline, state: PipelineState
    ) -> tuple[MiniMaxH3ModularPipeline, PipelineState]:
        block_state = self.get_block_state(state)

        if len(block_state.audio_timesteps) != len(block_state.timesteps):
            raise ValueError(
                f"MiniMax-H3's video and audio schedules must have equal length, got "
                f"{len(block_state.timesteps)} and {len(block_state.audio_timesteps)}."
            )

        begin, end = self._resolve_window(len(block_state.timesteps))
        # Slice all three per-step arrays by the identical window. The scheduler step block indexes
        # audio_timesteps[i] and row_timestep_plan[i] with the loop counter over the sliced
        # timesteps, so a shared window is what keeps video and audio in lockstep.
        block_state.timesteps = block_state.timesteps[begin:end]
        block_state.audio_timesteps = block_state.audio_timesteps[begin:end]
        block_state.row_timestep_plan = block_state.row_timestep_plan[begin:end]

        # The schedulers' own sigma grids stay whole: step() needs sigmas[i + 1], and
        # index_for_timestep searches the unsliced timesteps, so slicing either would desynchronise
        # them. set_begin_index pins where in the whole grid this window starts, and MUST run after
        # the set_timesteps block, which resets it to None.
        components.scheduler.set_begin_index(begin)
        components.audio_scheduler.set_begin_index(begin)

        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:
            for i, t in enumerate(block_state.timesteps):
                components, block_state = self.loop_step(components, block_state, i=i, t=t)
                progress_bar.update()
                if self.callback is not None:
                    # The return value is deliberately discarded: the framework's callback returns
                    # {} on its normal path, and merging that would clobber the loop's latents.
                    self.callback(components, i, t, {"latents": block_state.latents})
                if getattr(components, "_interrupt", False):
                    break
        self.set_block_state(state, block_state)
        return components, state


class MiniMaxH3LatentPipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True
    # MiniMax-H3 generates at a fixed 24 fps; any other rate desynchronises the soundtrack.
    video_fps: ClassVar[int] = MINIMAX_H3_FPS

    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)
        # Set by decode_latent, read by the VAE Decode node in the same _decode call so the
        # soundtrack can be muxed into the video that decode_latent returns.
        self.last_audio: torch.Tensor | None = None
        self.last_sampling_rate: int | None = None

    @property
    @override
    def modular_pipe(self) -> ModularPipeline:
        # self._pipe already IS the modular pipeline. The base implementation would round-trip
        # `update_components(**pipe.components)`, which drops the load spec of any component lacking
        # a _diffusers_load_id and re-registers everything with the ComponentsManager that owns this
        # model's CPU-offload hooks.
        return cast(ModularPipeline, self._pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return cast(ModularPipeline, self._pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _latent_geometry(self, source_shape: tuple[int, ...]) -> dict[str, int]:
        """Latent-space dimensions for a pixel-space ``source_shape`` of (..., T, H, W)."""
        pipe = cast(MiniMaxH3ModularPipeline, self.modular_pipe)
        num_frames = align_num_frames(source_shape[-3])
        compression = pipe.vae_spatial_compression_ratio
        return {
            "num_frames": num_frames,
            "num_latent_frames": video_latent_num_frames(num_frames),
            "latent_height": source_shape[-2] // compression,
            "latent_width": source_shape[-1] // compression,
            "num_audio_latents": audio_latent_num_frames(num_frames),
        }

    def _run_blocks(self, blocks: Any, **kwargs: Any) -> PipelineState:
        """Run ``blocks`` over one shared ``PipelineState`` and return it.

        Deliberately not ``_call_block``: that builds a fresh state per call (so a multi-block
        prefix would lose every intermediate) and runs under ``inference_mode``, whose tensors
        cannot be mutated in place — which the scheduler step does.
        """
        state = PipelineState()
        for param in blocks.inputs:
            if param.name in kwargs:
                state.set(param.name, kwargs[param.name], param.kwargs_type)
        with torch.no_grad():
            _, state = blocks(self.modular_pipe, state)
        return state

    # ------------------------------------------------------------------
    # Public latent surface
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        generator = generator_state.to_generator()
        geometry = self._latent_geometry(source_shape)
        state = self._run_blocks(
            _MiniMaxH3PrepareNoiseStep(),
            num_latent_frames=geometry["num_latent_frames"],
            latent_height=geometry["latent_height"],
            latent_width=geometry["latent_width"],
            num_audio_latents=geometry["num_audio_latents"],
            generator=generator,
        )
        latents = self._get_required(state.values, "latents", torch.Tensor)
        audio_latents = self._get_required(state.values, "audio_latents", torch.Tensor)
        return self._make_latent_artifact(
            latents,
            source_shape=source_shape,
            meta={
                AUDIO_LATENTS_META_KEY: audio_latents,
                AUDIO_PAIRED_WITH_META_KEY: _video_fingerprint(latents),
                **GeneratorState.from_generator(generator).as_meta(),
            },
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        """Decode the video, and its soundtrack when the artifact still carries the audio latent."""
        pipe = cast(MiniMaxH3ModularPipeline, self.modular_pipe)
        device, _ = self._get_device_and_type()
        geometry = self._latent_geometry(latent.source_shape)

        # The decode blocks take packed rows and unpatchify internally.
        latents = latent.to_torch(device=device, dtype=torch.float32)
        video_rows = patchify_video_latents(latents, pipe.patch_size)

        audio_latents = read_driver_meta(latent, AUDIO_LATENTS_META_KEY, self.driver_namespace)
        self.last_audio = None
        self.last_sampling_rate = None

        # A present-but-stale audio latent is the dangerous case: latent math sums the video latent
        # while carrying the left operand's unsummed audio, so the soundtrack would no longer match
        # the picture. Refuse instead of muxing a desynchronised track.
        if audio_latents is not None:
            paired_with = read_driver_meta(latent, AUDIO_PAIRED_WITH_META_KEY, self.driver_namespace)
            if not _fingerprints_match(paired_with, _video_fingerprint(latents)):
                raise ValueError(
                    f"{self.driver_namespace}: Attempted to decode a MiniMax-H3 latent. Failed "
                    f"because its audio latent belongs to a different video latent, so the "
                    f"soundtrack would not match the picture. MiniMax-H3 generates video and audio "
                    f"jointly and the audio travels in the latent's metadata, which latent math, "
                    f"composite and upsampler nodes do not recompute. Connect Generate Media "
                    f"Latents directly to Decode Media Latent."
                )

        video_state = self._run_blocks(
            pipe.blocks.sub_blocks["decode"].sub_blocks["video"],
            latents=video_rows,
            num_latent_frames=geometry["num_latent_frames"],
            latent_height=geometry["latent_height"],
            latent_width=geometry["latent_width"],
            output_type="pil",
        )
        video_frames = self._get_required(video_state.values, "videos", list)[0]

        # The live-preview path decodes a latent it rebuilt without meta, so a missing soundtrack is
        # normal there and must not raise. Callers that need the audio check `last_audio`.
        if audio_latents is None:
            logger.warning(
                "%s: decoding video only because the latent carries no audio latents in driver meta.",
                self.driver_namespace,
            )
            return video_frames

        audio_rows = (
            audio_latents.to(device=device, dtype=torch.float32)
            .permute(0, 2, 1)
            .reshape(-1, pipe.audio_latent_channels)
        )
        audio_state = self._run_blocks(
            pipe.blocks.sub_blocks["decode"].sub_blocks["audio"],
            audio_latents=audio_rows,
            num_audio_latents=geometry["num_audio_latents"],
            output_type="pil",
        )
        self.last_audio = self._get_required(audio_state.values, "audio", torch.Tensor)
        self.last_sampling_rate = self._get_required(audio_state.values, "sampling_rate", int)
        return video_frames

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        # MiniMax-H3's only VAE-encode block is keyframe-specific: it seeds the posterior with a
        # fixed seed, rounds to fp16 and emits noise-augmented conditioning rows rather than a
        # public latent. Keyframes reach the model through `conditioning_images` on the Generate
        # node instead. There is no general image or video encode in the blockset.
        raise NotImplementedError(
            f"{self.driver_namespace} does not support encoding media into a latent. MiniMax-H3 has "
            f"no general VAE-encode path; supply keyframes via the Media Gen Conditioning input on "
            f"Generate Media Latents instead."
        )

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        # Video-to-video would need to resume a partially-noised latent mid-schedule, but
        # MiniMaxH3Scheduler.set_timesteps always rebuilds the full linspace(1, 0, n) grid and the
        # blockset exposes no encode path to noise from in the first place.
        raise NotImplementedError(
            f"{self.driver_namespace} does not support adding noise to an existing latent "
            f"(no video-to-video path). Use Create Noise Latents for a fresh latent."
        )

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------

    def _resolve_keyframes(self, kwargs: dict[str, Any], num_frames: int) -> None:
        """Translate a Media Gen Conditioning payload into ``image`` / ``last_image`` kwargs."""
        payloads = normalize_to_payloads(kwargs.pop(MediaGenConditioningKey.OUTPUT, None))
        if payloads is None:
            return

        for payload in payloads:
            if payload.mode is not ConditioningMode.IMAGE:
                raise ValueError(
                    f"Attempted to build MiniMax-H3 keyframe conditioning. Failed with mode "
                    f"'{payload.mode.value}' because only image payloads are supported."
                )
            for entry in payload.entries:
                image = resolve_conditioning_image(entry.artifact)
                frame_index = resolve_frame_index(entry.frame_index, num_frames)
                if frame_index == 0:
                    kwargs["image"] = image
                elif frame_index in (-1, num_frames - 1):
                    kwargs["last_image"] = image
                else:
                    raise ValueError(
                        f"Attempted to build MiniMax-H3 keyframe conditioning. Failed with "
                        f"frame_index={frame_index} because only the first frame (0) and the last "
                        f"({num_frames - 1} or -1) are supported."
                    )

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
        if isinstance(latent, InpaintMaskArtifact):
            raise NotImplementedError(f"{self.driver_namespace} does not support inpainting.")
        if return_fully_denoised:
            # Reaching the terminal sigma from a truncated window would step the schedulers past
            # the window while their sigma grids stay whole, pairing each step with the wrong sigma.
            raise NotImplementedError(
                f"{self.driver_namespace} does not support 'return_fully_denoised' because "
                f"MiniMax-H3's two schedulers cannot express a non-contiguous schedule."
            )

        pipe = cast(MiniMaxH3ModularPipeline, self.modular_pipe)
        device, _ = self._get_device_and_type()
        source_shape = latent.source_shape
        geometry = self._latent_geometry(source_shape)

        update_kwargs = kwargs.copy()
        requested_num_frames = update_kwargs.pop("num_frames", None)
        if requested_num_frames is not None and align_num_frames(int(requested_num_frames)) != geometry["num_frames"]:
            logger.warning(
                "%s: ignoring num_frames=%s because the input latent holds %d frames.",
                self.driver_namespace,
                requested_num_frames,
                geometry["num_frames"],
            )
        # The latent's own shape is what gets denoised, so source_shape is the single source of truth.
        update_kwargs["num_frames"] = geometry["num_frames"]
        update_kwargs.setdefault("height", source_shape[-2])
        update_kwargs.setdefault("width", source_shape[-1])
        self._resolve_keyframes(update_kwargs, geometry["num_frames"])

        generator = update_kwargs.pop("generator", generator_state.to_generator())
        audio_latents = read_driver_meta(latent, AUDIO_LATENTS_META_KEY, self.driver_namespace)
        if audio_latents is not None:
            audio_latents = audio_latents.to(device=device, dtype=torch.float32)

        prefix_blocks = dict(zip(pipe.blocks.block_names, pipe.blocks.sub_blocks.values(), strict=True))
        prefix_blocks.pop("decode")
        denoise_step = _MiniMaxH3CallbackDenoiseStep()
        denoise_step.callback = callback
        denoise_step.start_step = start_step
        denoise_step.end_step = end_step
        prefix_blocks["denoise"] = denoise_step
        blocks = SequentialPipelineBlocks.from_blocks_dict(prefix_blocks)

        # The framework signals cancellation by setting `_interrupt` on the pipe, but only when the
        # attribute already exists — and ModularPipeline has none. The pipe is cached across runs,
        # so a leaked True would break the next run at step 0.
        pipe._interrupt = False
        try:
            state = self._run_blocks(
                blocks,
                latents=latent.to_torch(device=device, dtype=torch.float32),
                audio_latents=audio_latents,
                num_inference_steps=num_inference_steps,
                generator=generator,
                **update_kwargs,
            )
        finally:
            pipe._interrupt = False

        denoised_video_rows = self._get_required(state.values, "latents", torch.Tensor)
        denoised_audio_rows = self._get_required(state.values, "audio_latents", torch.Tensor)
        num_condition_video_rows = state.values.get("num_condition_video_rows", 0)
        num_condition_audio_rows = state.values.get("num_condition_audio_rows", 0)

        video_latents = unpatchify_video_tokens(
            denoised_video_rows[num_condition_video_rows:],
            geometry["num_latent_frames"],
            geometry["latent_height"],
            geometry["latent_width"],
            pipe.vae_latent_channels,
            pipe.patch_size,
        )
        audio_out = unpack_audio_tokens(denoised_audio_rows[num_condition_audio_rows:], geometry["num_audio_latents"])

        return self._make_latent_artifact(
            video_latents,
            source_shape=source_shape,
            upstream=latent,
            meta={
                AUDIO_LATENTS_META_KEY: audio_out,
                AUDIO_PAIRED_WITH_META_KEY: _video_fingerprint(video_latents),
                **GeneratorState.from_generator(generator).as_meta(),
            },
        )
