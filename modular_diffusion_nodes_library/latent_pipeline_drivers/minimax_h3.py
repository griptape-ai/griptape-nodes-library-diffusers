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
from diffusers.modular_pipelines.minimax_h3.modular_blocks_minimax_h3 import (  # type: ignore[reportMissingImports]
    MiniMaxH3CoreDenoiseStep,
    MiniMaxH3FL2VACoreDenoiseStep,
)
from diffusers.modular_pipelines.minimax_h3.modular_pipeline import (  # type: ignore[reportMissingImports]  # type: ignore[reportMissingImports]
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_FPS,
    MINIMAX_H3_MAX_ASPECT_RATIO,
    MINIMAX_H3_MIN_ASPECT_RATIO,
    MiniMaxH3ModularPipeline,
    align_num_frames,
    audio_latent_num_frames,
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
from modular_diffusion_nodes_library.utils.dimension_alignment import DimensionAlignmentResult

logger = logging.getLogger("modular_diffusers_nodes_library")

#: Key under which the audio latent rides in this driver's namespaced ``meta`` sub-bag.
AUDIO_LATENTS_META_KEY = "audio_latents"

#: Valid ``num_frames`` values must satisfy two independent constraints:
#:   1. Chunk alignment: the video VAE only encodes counts of the form ``17 * n + 5``.
#:   2. Duration window: at MiniMax-H3's fixed 24 fps, the resulting duration must fall within
#:      its 5-15 s generation window.
#: 124 (5.167 s, n=7) is the smallest ``17 * n + 5`` value whose duration is >= 5 s; 345
#: (14.375 s, n=20) is the largest whose duration is <= 15 s. The next aligned value up, 362
#: (n=21), is 15.083 s and falls outside the window, which upstream rejects.
MIN_REQUESTABLE_NUM_FRAMES = 124
MAX_REQUESTABLE_NUM_FRAMES = 345

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


def _unpack_video_rows(
    rows: torch.Tensor,
    components: MiniMaxH3ModularPipeline,
    *,
    num_condition_video_rows: int,
    num_latent_frames: int,
    latent_height: int,
    latent_width: int,
) -> torch.Tensor:
    """Turn denoised video rows back into an unpacked 5-D latent, dropping conditioning rows.

    Used only for the step-end preview: it needs the public 5-D shape mid-loop, before
    ``MiniMaxH3AfterDenoiseStep`` (which does this same reshape for the final output) has run.
    Reimplements the video half of ``MiniMaxH3AfterDenoiseStep.__call__``
    (``diffusers/modular_pipelines/minimax_h3/decoders.py``) rather than calling it directly, since
    that block also reshapes the audio stream and mutates ``block_state`` in place, which the
    mid-loop preview must not touch.
    """
    patch_t, patch_h, patch_w = components.patch_size
    channels = components.vae_latent_channels
    kept = rows[num_condition_video_rows:]
    kept = kept.reshape(
        -1,
        num_latent_frames // patch_t,
        latent_height // patch_h,
        latent_width // patch_w,
        channels,
        patch_t,
        patch_h,
        patch_w,
    )
    kept = kept.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return kept.reshape(-1, channels, num_latent_frames, latent_height, latent_width).contiguous()


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

    @property
    def loop_inputs(self) -> list[InputParam]:
        # The video geometry is not part of upstream's loop contract, but the step-end preview needs
        # it to unpack the in-flight rows into the public latent shape.
        return [
            *super().loop_inputs,
            InputParam("num_latent_frames", required=True),
            InputParam("latent_height", required=True),
            InputParam("latent_width", required=True),
        ]

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
    def __call__(self, components: MiniMaxH3ModularPipeline, state: PipelineState) -> PipelineState:
        block_state = cast(Any, self.get_block_state(state))

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
                components, block_state = cast(Any, self.loop_step(components, block_state, i=i, t=t))
                progress_bar.update()
                if self.callback is not None:
                    # The loop works in packed rows, but the framework feeds this straight back into
                    # decode_latent for the live preview, which expects the public 5-D shape.
                    # The return value is deliberately discarded: the callback returns {} on its
                    # normal path, and merging that would clobber the loop's latents.
                    preview_latents = _unpack_video_rows(
                        block_state.latents,
                        components,
                        num_condition_video_rows=block_state.num_condition_video_rows,
                        num_latent_frames=block_state.num_latent_frames,
                        latent_height=block_state.latent_height,
                        latent_width=block_state.latent_width,
                    )
                    self.callback(components, i, t, {"latents": preview_latents})
                if getattr(components, "_interrupt", False):
                    break
        self.set_block_state(state, block_state)
        return components, state  # type: ignore[reportReturnType]


class MiniMaxH3LatentPipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True
    # MiniMax-H3 generates at a fixed 24 fps; any other rate desynchronises the soundtrack.
    video_fps: ClassVar[int] = MINIMAX_H3_FPS

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
    # Dimension-alignment validation
    # ------------------------------------------------------------------

    @override
    def _get_spatial_alignment(self) -> int:
        return cast(MiniMaxH3ModularPipeline, self.modular_pipe).canvas_multiple

    @override
    def align_dimensions(self, height: int, width: int, num_frames: int | None = None) -> DimensionAlignmentResult:
        pipe = cast(MiniMaxH3ModularPipeline, self.modular_pipe)
        spatial_alignment = self._get_spatial_alignment()
        aligned_h = self._ceil_to_alignment(height, spatial_alignment)
        aligned_w = self._ceil_to_alignment(width, spatial_alignment)

        # Aspect ratio next: grow whichever axis is deficient (never shrink) so the suggestion stays
        # the largest canvas MiniMax-H3 will accept for the other, user-requested axis.
        ratio = aligned_w / aligned_h
        if ratio > MINIMAX_H3_MAX_ASPECT_RATIO:
            aligned_h = self._ceil_to_alignment(math.ceil(aligned_w / MINIMAX_H3_MAX_ASPECT_RATIO), spatial_alignment)
        elif ratio < MINIMAX_H3_MIN_ASPECT_RATIO:
            aligned_w = self._ceil_to_alignment(math.ceil(aligned_h * MINIMAX_H3_MIN_ASPECT_RATIO), spatial_alignment)

        # Pixel budget last: this is the one axis where "the maximum of what the pipeline suggests"
        # means shrinking, since there is no larger canvas that still fits the budget. Scale both
        # axes down together to preserve the now-valid ratio, then floor (not ceil) to the alignment
        # so the result never creeps back over the budget.
        max_pixels = pipe.config.canvas_max_pixels  # type: ignore[reportAttributeAccessIssue]
        if aligned_h * aligned_w > max_pixels:
            scale = (max_pixels / (aligned_h * aligned_w)) ** 0.5
            aligned_h = max(spatial_alignment, int(aligned_h * scale) // spatial_alignment * spatial_alignment)
            aligned_w = max(spatial_alignment, int(aligned_w * scale) // spatial_alignment * spatial_alignment)

        aligned_frames = num_frames
        if num_frames is not None:
            # Clamp into the requestable range first so the aligned value this returns is always
            # itself a valid frame count. `align_num_frames` only rounds up to the next 17n+5 value
            # with no notion of the 5-15s window, so an out-of-range input (e.g. 346, which rounds to
            # 362) would otherwise "align" to a value that's still invalid.
            clamped_frames = min(max(num_frames, MIN_REQUESTABLE_NUM_FRAMES), MAX_REQUESTABLE_NUM_FRAMES)
            aligned_frames = align_num_frames(clamped_frames, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)

        return DimensionAlignmentResult(aligned_h, aligned_w, aligned_frames, None)

    @override
    def validate_dimensions(self, height: int, width: int, num_frames: int | None = None) -> list[str]:
        messages: list[str] = []
        pipe = cast(MiniMaxH3ModularPipeline, self.modular_pipe)
        spatial_alignment = self._get_spatial_alignment()
        aligned = self.align_dimensions(height, width, num_frames)

        if num_frames is not None:
            # Range check first: only once num_frames is inside the requestable range is a
            # 17n+5 rounding suggestion meaningful. Checking chunk-alignment first can suggest an
            # equally out-of-range value (346 "rounds to" 362, which is also invalid) instead of
            # telling the user the actual problem.
            if num_frames < MIN_REQUESTABLE_NUM_FRAMES or num_frames > MAX_REQUESTABLE_NUM_FRAMES:
                messages.append(
                    f"num_frames={num_frames} is invalid: MiniMax-H3 generates between "
                    f"{pipe.min_duration:g} and {pipe.max_duration:g} seconds at {MINIMAX_H3_FPS} fps. "
                    f"Set num_frames between {MIN_REQUESTABLE_NUM_FRAMES} and {MAX_REQUESTABLE_NUM_FRAMES}."
                )
            else:
                aligned_frames = align_num_frames(num_frames, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)
                if num_frames != aligned_frames:
                    messages.append(
                        f"num_frames={num_frames} is invalid: must be of the form "
                        f"{pipe.vae_frames_per_chunk} * n + {pipe.vae_latents_per_chunk}. "
                        f"Suggested value: {aligned_frames}."
                    )

        # Ratio and pixel-budget checks only make sense once height/width are already spatially
        # aligned, mirroring the order `align_dimensions` applies its own corrections in.
        height_aligned = height % spatial_alignment == 0
        width_aligned = width % spatial_alignment == 0
        if not height_aligned:
            messages.append(
                f"height={height} is invalid: must be divisible by {spatial_alignment}. "
                f"Suggested value: {aligned.height}."
            )
        if not width_aligned:
            messages.append(
                f"width={width} is invalid: must be divisible by {spatial_alignment}. Suggested value: {aligned.width}."
            )

        max_pixels = pipe.config.canvas_max_pixels  # type: ignore[reportAttributeAccessIssue]
        if height_aligned and width_aligned:
            ratio = width / height
            if not MINIMAX_H3_MIN_ASPECT_RATIO <= ratio <= MINIMAX_H3_MAX_ASPECT_RATIO:
                messages.append(
                    f"{width}x{height} (ratio {ratio:g}) is invalid: MiniMax-H3 supports aspect ratios from "
                    f"1:4 to 4:1. Suggested value: {aligned.width}x{aligned.height}."
                )
            elif height * width > max_pixels:
                messages.append(
                    f"{width}x{height} ({height * width} pixels) is invalid: MiniMax-H3 generates at most "
                    f"{max_pixels} pixels. Suggested value: {aligned.width}x{aligned.height}."
                )

        return messages

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _latent_geometry(self, source_shape: tuple[int, ...]) -> dict[str, int]:
        """Latent-space dimensions for a pixel-space ``source_shape`` of (..., T, H, W).

        ``source_shape`` comes from the latent, which is the single source of truth for what gets
        denoised, so it is validated here rather than in the runtime parameters: the Create Noise
        Latents node owns these dimensions and has no MiniMax-H3-specific validation hook.
        """
        pipe = cast(MiniMaxH3ModularPipeline, self.modular_pipe)
        height, width = source_shape[-2], source_shape[-1]
        requested_num_frames = source_shape[-3]

        # height/width/num_frames are validated up front by validate_dimensions/align_dimensions
        # (multiple-of-32, aspect ratio, pixel budget, and 17n+5 frame alignment). This is a
        # defensive backstop for latents that reached denoise without going through that path.
        messages = self.validate_dimensions(height, width, requested_num_frames)
        if messages:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to size a MiniMax-H3 request. Failed because " + " ".join(messages)
            )

        num_frames = align_num_frames(requested_num_frames, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk)
        duration = num_frames / MINIMAX_H3_FPS
        if not self.modular_pipe.min_duration <= duration <= self.modular_pipe.max_duration:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to size a MiniMax-H3 request. Failed with "
                f"num_frames={requested_num_frames} because it snaps up to {num_frames} frames "
                f"({duration:.3f} s at {MINIMAX_H3_FPS} fps), outside the "
                f"{self.modular_pipe.min_duration:g}-{self.modular_pipe.max_duration:g} s window MiniMax-H3 "
                f"generates. Set num_frames between {MIN_REQUESTABLE_NUM_FRAMES} and "
                f"{MAX_REQUESTABLE_NUM_FRAMES} on the node that created this latent."
            )

        compression = pipe.vae_spatial_compression_ratio
        return {
            "num_frames": num_frames,
            "num_latent_frames": video_latent_num_frames(
                num_frames, pipe.vae_frames_per_chunk, pipe.vae_latents_per_chunk
            ),
            "latent_height": height // compression,
            "latent_width": width // compression,
            "num_audio_latents": audio_latent_num_frames(num_frames),
        }

    def _read_paired_audio_latents(
        self, latent: LatentArtifact, video_latents: torch.Tensor, *, action: str
    ) -> torch.Tensor | None:
        """Return the artifact's audio latent, or ``None`` when it carries none.

        Raises when an audio latent is present but was paired with a *different* video latent. That
        is the dangerous case: latent math shallow-merges meta left-operand-wins, so a summed video
        latent keeps the left operand's unsummed audio. Both the denoise and the decode entry points
        check this, so a stale pairing cannot be laundered by passing through a second denoise.
        """
        audio_latents = read_driver_meta(latent, AUDIO_LATENTS_META_KEY, self.driver_namespace)
        if audio_latents is None:
            return None

        paired_with = read_driver_meta(latent, AUDIO_PAIRED_WITH_META_KEY, self.driver_namespace)
        if not _fingerprints_match(paired_with, _video_fingerprint(video_latents)):
            raise ValueError(
                f"{self.driver_namespace}: Attempted to {action} a MiniMax-H3 latent. Failed "
                f"because its audio latent belongs to a different video latent, so the soundtrack "
                f"would not match the picture. MiniMax-H3 generates video and audio jointly and the "
                f"audio travels in the latent's metadata, which latent math, composite and upsampler "
                f"nodes do not recompute. Connect Generate Media Latents directly to Decode Media "
                f"Latent."
            )
        return audio_latents

    def _run_blocks(self, blocks: Any, **kwargs: Any) -> PipelineState:
        """Run ``blocks`` over one shared ``PipelineState`` and return it.

        Deliberately not ``_call_block``, for two reasons. It builds a fresh state per call, so a
        multi-block prefix would lose every intermediate. And it runs under ``inference_mode``,
        which mints inference tensors: legal to mutate in place *within* that scope, but the
        scheduler step mutates ``latents`` in a later call than the one that created them, and
        in-place mutation of an inference tensor outside ``inference_mode`` raises. ``no_grad``
        produces ordinary tensors and avoids the whole class of problem.
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
        self._latent_geometry(latent.source_shape)  # validates the latent's shape before decoding it

        latents = latent.to_torch(device=device, dtype=torch.float32)

        self.last_audio = None
        self.last_sampling_rate = None
        audio_latents = self._read_paired_audio_latents(latent, latents, action="decode")

        video_state = self._run_blocks(
            pipe.blocks.sub_blocks["decode"].sub_blocks["video"],
            latents=latents,
            output_type="pil",
        )
        video_frames = self._get_required(video_state.values, "videos", list)[0]

        # The live-preview path decodes a latent it rebuilt without meta, so a missing soundtrack is
        # normal there and must not raise. Debug rather than warning because that path decodes once
        # per denoise step. Callers that need the audio check `last_audio`.
        if audio_latents is None:
            logger.debug(
                "%s: decoding video only because the latent carries no audio latents in driver meta.",
                self.driver_namespace,
            )
            return video_frames

        audio_state = self._run_blocks(
            pipe.blocks.sub_blocks["decode"].sub_blocks["audio"],
            audio_latents=audio_latents.to(device=device, dtype=torch.float32),
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
        # The latent's own shape is what gets denoised, so it dictates the geometry outright. These
        # are assigned rather than merged: a layout built for different dimensions than the latent
        # tensor either crashes deep in attention or, when the row counts happen to collide,
        # silently produces garbage. Nothing upstream of here is allowed to disagree.
        update_kwargs["num_frames"] = geometry["num_frames"]
        update_kwargs["height"] = source_shape[-2]
        update_kwargs["width"] = source_shape[-1]
        self._resolve_keyframes(update_kwargs, geometry["num_frames"])

        generator = update_kwargs.pop("generator", generator_state.to_generator())
        video_latents_in = latent.to_torch(device=device, dtype=torch.float32)
        audio_latents = self._read_paired_audio_latents(latent, video_latents_in, action="denoise")
        if audio_latents is None and start_step > 0:
            # Upstream draws fresh audio noise when none is supplied, but with a begin index set
            # both schedulers would step that pure noise as if it were already partly denoised,
            # yielding a garbled soundtrack that this run would then stamp as validly paired.
            raise ValueError(
                f"{self.driver_namespace}: Attempted to resume a MiniMax-H3 denoise at step "
                f"{start_step}. Failed because the input latent carries no audio latent, so the "
                f"soundtrack would restart from pure noise mid-schedule. Chain partial denoise "
                f"directly from a previous Generate Media Latents run."
            )
        if audio_latents is not None:
            audio_latents = audio_latents.to(device=device, dtype=torch.float32)

        is_fl2va = "image" in update_kwargs or "last_image" in update_kwargs
        core_denoise_cls = MiniMaxH3FL2VACoreDenoiseStep if is_fl2va else MiniMaxH3CoreDenoiseStep
        inner_blocks = dict(core_denoise_cls().sub_blocks)
        denoise_step = _MiniMaxH3CallbackDenoiseStep()
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

        # The framework signals cancellation by setting `_interrupt` on the pipe, but only when the
        # attribute already exists — and ModularPipeline has none. The pipe is cached across runs,
        # so a leaked True would break the next run at step 0.
        pipe._interrupt = False  # type: ignore[reportAttributeAccessIssue]
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
            pipe._interrupt = False  # type: ignore[reportAttributeAccessIssue]

        video_latents = self._get_required(state.values, "latents", torch.Tensor)
        audio_out = self._get_required(state.values, "audio_latents", torch.Tensor)

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
