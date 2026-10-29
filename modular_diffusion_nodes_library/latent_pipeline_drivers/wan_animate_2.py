"""WAN Animate 2 driver: reference-character animation driven by a video, from a Modular-Diffusers-only integration.

Like MiniMax-H3, WAN Animate 2 has no ``DiffusionPipeline`` half, so this driver owns the denoise loop
instead of ending with ``super().denoise_latent()``. ``.claude/rules/latent-drivers.md`` asks for the
latter, but ``base_driver.denoise_latent`` raises for a ``ModularPipeline``; this follows the MiniMax-H3
precedent instead.

Upstream denoises the driving video segment by segment, and each segment conditions on the *decoded
pixels* of the previous one, so its loop only ever produces pixels. This driver swaps a few blocks in that
loop so it also yields the public latent contract:

- **Public latent.** One normalised Wan video latent ``[1, 16, T_lat, H/8, W/8]`` for the requested
  frames, with no reference slot. Each segment writes its kept latents into it.
- **Driver meta.** Each segment's reference-slot and overlap latents (the "prefix") ride in meta, so
  ``decode_latent`` can rebuild every segment and decode seams exactly as the loop did. Without meta the
  tensor decodes as one plain Wan latent, and frames near a seam can differ slightly. The padding latents
  past ``T_lat`` (the "tail") ride along so a single-segment run can resume mid-schedule.
- **Input noise is used.** Each segment's content noise is taken from the input latent; only the prefix
  and tail are drawn from the generator.
- **Partial denoise works for one segment only.** See ``TODO(wan-animate-2 Q5)`` in ``denoise_latent``.

The transformer's generation attention runs on flex attention, which only fits in memory under
``torch.compile``; ``_ensure_transformer_compiled`` handles that on the first denoise.
"""

from __future__ import annotations

import importlib.util
import logging
import math
import sys
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, cast, override

from diffusers.models.transformers.transformer_wan_animate_2 import (  # type: ignore[reportMissingImports]
    WanAnimate2KVCache,
)
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    BlockState,
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.modular_pipelines.wan_animate_2.denoise import (  # type: ignore[reportMissingImports]
    WanAnimate2RefExtractStep,
    WanAnimate2SegmentDecodeStep,
    WanAnimate2SegmentDenoiseInner,
    WanAnimate2SegmentLoopWrapper,
    WanAnimate2SegmentPrepareStep,
    WanAnimate2SegmentPrevFramesStep,
    WanAnimate2SegmentSchedulerResetStep,
    WanAnimate2SegmentVaeEncoderStep,
)
from diffusers.modular_pipelines.wan_animate_2.encoders import (  # type: ignore[reportMissingImports]
    WanAnimate2ProcessImagesInputStep,
)
from diffusers.modular_pipelines.wan_animate_2.modular_pipeline import (  # type: ignore[reportMissingImports]
    WanAnimate2ModularPipeline,
)

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
from modular_diffusion_nodes_library.utils.conditioning_utils import MediaGenConditioningKey
from modular_diffusion_nodes_library.utils.torch_utils import no_grad

if TYPE_CHECKING:
    import torch  # type: ignore[reportMissingImports]
    from diffusers.modular_pipelines.modular_pipeline_utils import (
        InputParam,  # type: ignore[reportMissingImports]
        OutputParam,  # type: ignore[reportMissingImports]
    )

logger = logging.getLogger("modular_diffusers_nodes_library")

#: The frame rate the blocks generate at (`fps` default in `wan_animate_2/encoders.py`). The driver never
#: passes another, so the output always plays at this rate.
WAN_ANIMATE_2_FPS = 24

#: Wan VAE temporal compression: latent frame 0 decodes to 1 pixel frame, every later one to 4. Upstream's
#: `get_i2v_mask` hard-codes the same factor.
VAE_TEMPORAL_FACTOR = 4

DEFAULT_SEGMENT_FRAME_LENGTH = 81
DEFAULT_PREV_SEGMENT_CONDITIONING_FRAMES = 1

#: Keys in this driver's namespaced ``meta`` sub-bag.
SEGMENT_FRAME_LENGTH_META_KEY = "segment_frame_length"
PREV_SEGMENT_CONDITIONING_FRAMES_META_KEY = "prev_segment_conditioning_frames"
NUM_SEGMENTS_META_KEY = "num_segments"
#: One ``[16, 1 + n_overlap, h, w]`` tensor per completed segment: the reference slot, then the overlap.
SEGMENT_PREFIX_LATENTS_META_KEY = "segment_prefix_latents"
#: One ``[16, n_padding, h, w]`` tensor per completed segment: the latents past ``T_lat`` (often empty).
SEGMENT_TAIL_LATENTS_META_KEY = "segment_tail_latents"

#: Set on the transformer once `compile_repeated_blocks()` has run, so a cached pipe compiles only once.
_COMPILED_FLAG = "_gtn_repeated_blocks_compiled"


# ----------------------------------------------------------------------
# Segment geometry (pure, no torch)
# ----------------------------------------------------------------------


class SegmentSlice(NamedTuple):
    """Where one segment's kept latents sit in the padded latent sequence."""

    start: int
    """First padded latent index this segment keeps."""
    stop: int
    """Padded latent index (exclusive) where this segment's kept latents end."""
    public_stop: int
    """``min(stop, T_lat)``: the part read from and written to the public latent."""
    prefix_len: int
    """``1 + n_overlap``: the reference slot plus the overlap latents in front of the kept ones."""


def segment_count(num_frames: int, segment_frame_length: int, prev_segment_conditioning_frames: int) -> int:
    """Number of segments upstream runs for ``num_frames`` driving frames. Needs ``num_frames > prev``."""
    effective_segment = segment_frame_length - prev_segment_conditioning_frames
    return math.ceil((num_frames - prev_segment_conditioning_frames) / effective_segment)


def padded_frame_count(num_frames: int, segment_frame_length: int, prev_segment_conditioning_frames: int) -> int:
    """Driving-frame count after padding to a whole number of segments."""
    effective_segment = segment_frame_length - prev_segment_conditioning_frames
    num_segments = segment_count(num_frames, segment_frame_length, prev_segment_conditioning_frames)
    return prev_segment_conditioning_frames + num_segments * effective_segment


def reflect_indices(num_frames: int, count: int) -> list[int]:
    """Indices that extend ``num_frames`` frames to ``count`` by symmetric reflection.

    ``0 1 … n-1 n-1 … 1 0 0 1 …``. For a pad no longer than the clip this matches upstream's zigzag pad
    (``[0 1 2 3 4 | 4 3 2]``); unlike upstream it keeps reflecting when the pad is longer than the clip,
    where upstream's negative slice start silently returns too few frames.
    """
    indices: list[int] = []
    for i in range(count):
        position = i % (2 * num_frames)
        if position < num_frames:
            indices.append(position)
        else:
            indices.append(2 * num_frames - 1 - position)
    return indices


def segment_slices(
    segment_frame_length: int,
    prev_segment_conditioning_frames: int,
    num_segments: int,
    num_latent_frames: int,
) -> list[SegmentSlice]:
    """Per-segment slices of the padded latent sequence, clipped to the public ``num_latent_frames``.

    Segment 0 keeps all ``L`` latents of its segment. Every later segment's first ``n_overlap`` latents
    re-generate the previous segment's tail (``prev = 4N + 1`` pixel frames decode from exactly
    ``n_overlap`` latents), so it keeps only the ``L - n_overlap`` after them.
    """
    segment_latents = (segment_frame_length - 1) // VAE_TEMPORAL_FACTOR + 1
    overlap_latents = (prev_segment_conditioning_frames - 1) // VAE_TEMPORAL_FACTOR + 1
    advance = segment_latents - overlap_latents

    slices: list[SegmentSlice] = []
    for k in range(num_segments):
        if k == 0:
            start = 0
            stop = segment_latents
            prefix_len = 1
        else:
            start = segment_latents + (k - 1) * advance
            stop = start + advance
            prefix_len = 1 + overlap_latents
        slices.append(SegmentSlice(start, stop, min(stop, num_latent_frames), prefix_len))
    return slices


class StepWindow(NamedTuple):
    begin: int
    end: int


def resolve_step_window(num_steps: int, start_step: int, end_step: int) -> StepWindow:
    """Clamp a requested ``[start_step, end_step)`` window against the schedule's real length.

    The same clamp as MiniMax-H3: ``end_step`` past the schedule clamps rather than raises. The caller
    validates ``start_step >= 0`` and ``start_step < end_step``.
    """
    begin = min(start_step, num_steps - 1)
    if end_step == -1:
        end = num_steps
    else:
        end = min(max(end_step, begin + 1), num_steps)
    return StepWindow(begin, end)


# ----------------------------------------------------------------------
# Blocks
# ----------------------------------------------------------------------


class _WanAnimate2PrepareNoiseStep(ModularPipelineBlocks):
    """Draw pure noise in the public latent shape ``[1, C, T_lat, h, w]``."""

    model_name = "wan-animate-2"

    @property
    def inputs(self) -> list[InputParam]:
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            InputParam,
        )

        return [
            InputParam("num_latent_frames", required=True),
            InputParam("latent_height", required=True),
            InputParam("latent_width", required=True),
            InputParam("generator"),
        ]

    @property
    def intermediate_outputs(self) -> list[OutputParam]:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            OutputParam,
        )

        return [
            OutputParam(
                "latents",
                type_hint=torch.Tensor,
                description="Normalised noise of shape (1, C, num_latent_frames, latent_height, latent_width).",
            ),
        ]

    @no_grad
    def __call__(
        self, components: WanAnimate2ModularPipeline, state: PipelineState
    ) -> tuple[WanAnimate2ModularPipeline, PipelineState]:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.utils.torch_utils import randn_tensor  # type: ignore[reportMissingImports]

        block_state = cast(Any, self.get_block_state(state))
        # float32 like upstream's per-segment noise (`WanAnimate2SegmentPrepareStep`).
        block_state.latents = randn_tensor(
            (
                1,
                components.num_channels_latents,
                block_state.num_latent_frames,
                block_state.latent_height,
                block_state.latent_width,
            ),
            generator=block_state.generator,
            device=components._execution_device,
            dtype=torch.float32,
        )

        self.set_block_state(state, block_state)
        return components, state


class _WanAnimate2ExactImageInputStep(WanAnimate2ProcessImagesInputStep):
    """Upstream's reference-image preprocess, at exactly the requested ``height`` x ``width``.

    Upstream treats ``height * width`` as a target *area* and rederives both sides from the image's aspect
    ratio, which cannot reproduce an exact size (e.g. 16x1680 becomes 16x1664). Here the latent owns the
    geometry and the driver has already stretched the image to it, so this only converts it to pixels and
    reports a crop region with no bars.
    """

    @property
    def description(self) -> str:
        return "Converts a reference image already sized to `height` x `width` into pixels, with no letterbox."

    @no_grad
    def __call__(
        self, components: WanAnimate2ModularPipeline, state: PipelineState
    ) -> tuple[WanAnimate2ModularPipeline, PipelineState]:
        import torch  # type: ignore[reportMissingImports]

        block_state = cast(Any, self.get_block_state(state))
        height = block_state.height
        width = block_state.width

        block_state.crop_region = (0, 0, height, width)
        # `resize_mode="fill"` like upstream. With an image already at the target size it adds no bars.
        block_state.image_pixels = components.image_processor.preprocess(
            block_state.image, height=height, width=width, resize_mode="fill"
        ).to(components._execution_device, dtype=torch.float32)

        self.set_block_state(state, block_state)
        return components, state


class _WanAnimate2SegmentNoiseStep(WanAnimate2SegmentPrepareStep):
    """Upstream's per-segment prepare step, with the content noise taken from the input latent.

    Upstream draws every segment's noise from the generator and ignores any input latent. Here segment
    ``k`` is ``[prefix | content | tail]``: the content is the input latent's slice for this segment, and
    the prefix (reference slot + overlap) and tail (padding past ``T_lat``) come from the previous run's
    meta when resuming, or from the generator otherwise, always drawn prefix first.
    """

    @property
    def description(self) -> str:
        return (
            "Builds this segment's starting latents from the input latent plus generated prefix and tail "
            "latents, and allocates a fresh KV cache."
        )

    @property
    def inputs(self) -> list[InputParam]:
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            InputParam,
        )

        return [
            InputParam.template("generator"),
            InputParam("input_latents", required=True, description="The public latent `[1, 16, T_lat, h, w]`"),
            InputParam("segment_slices", required=True, description="One `SegmentSlice` per segment"),
            InputParam("prefix_latents_in", description="Per-segment prefixes to resume from, or None"),
            InputParam("tail_latents_in", description="Per-segment tails to resume from, or None"),
        ]

    @no_grad
    def __call__(
        self, components: WanAnimate2ModularPipeline, block_state: BlockState, k: int
    ) -> tuple[WanAnimate2ModularPipeline, BlockState]:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.utils.torch_utils import randn_tensor  # type: ignore[reportMissingImports]

        state = cast(Any, block_state)
        device = components._execution_device
        segment: SegmentSlice = state.segment_slices[k]
        channels = state.input_latents.shape[1]
        latent_height = state.input_latents.shape[-2]
        latent_width = state.input_latents.shape[-1]
        tail_len = segment.stop - segment.public_stop

        if state.prefix_latents_in is not None:
            prefix = state.prefix_latents_in[k].to(device=device, dtype=torch.float32)
        else:
            prefix = randn_tensor(
                (channels, segment.prefix_len, latent_height, latent_width),
                generator=state.generator,
                device=device,
                dtype=torch.float32,
            )

        content = state.input_latents[0, :, segment.start : segment.public_stop].to(device=device, dtype=torch.float32)

        if state.tail_latents_in is not None:
            tail = state.tail_latents_in[k].to(device=device, dtype=torch.float32)
        elif tail_len > 0:
            tail = randn_tensor(
                (channels, tail_len, latent_height, latent_width),
                generator=state.generator,
                device=device,
                dtype=torch.float32,
            )
        else:
            tail = content.new_zeros((channels, 0, latent_height, latent_width))

        state.latents = torch.cat([prefix, content, tail], dim=1)
        state.kv_cache = WanAnimate2KVCache(components.transformer.config.num_layers)

        return components, block_state


class _WanAnimate2CallbackDenoiseInner(WanAnimate2SegmentDenoiseInner):
    """Upstream's inner timestep loop plus a step window, a step-end callback and an interrupt break.

    The body is upstream's ``WanAnimate2SegmentDenoiseInner.__call__`` with ``tqdm`` removed. It serves the
    distilled pipeline too: ``WanAnimate2DistilledSegmentDenoiseInner`` inherits ``__call__`` unchanged and
    only changes the guider's default config, and the pipe's own guider is used as is.

    The window lives here rather than in a step before ``ref_extract``, because that step takes ``t_ref``
    from the unsliced ``timesteps[0]``, which must stay the same in every window.
    """

    #: Assigned per-instance after construction. A block's ``__init__`` takes no arguments and its
    #: signature is introspected for config, so these must not become constructor kwargs.
    callback: Any = None
    start_step: int = 0
    end_step: int = -1

    @no_grad
    def __call__(
        self, components: WanAnimate2ModularPipeline, block_state: BlockState, k: int
    ) -> tuple[WanAnimate2ModularPipeline, BlockState]:
        import inspect

        import torch  # type: ignore[reportMissingImports]

        state = cast(Any, block_state)
        transformer_dtype = components.transformer.dtype

        guider_inputs = {
            "encoder_hidden_states": (state.prompt_embeds, state.negative_prompt_embeds),
            "is_uncondtion": (False, True),
        }

        transformer_args = set(inspect.signature(components.transformer.forward).parameters.keys())
        shared_kwargs = {
            name: value.to(transformer_dtype) if isinstance(value, torch.Tensor) else value
            for name, value in state.denoiser_input_fields.items()
            if name in transformer_args and name not in guider_inputs
        }

        window = resolve_step_window(len(state.timesteps), self.start_step, self.end_step)
        window_len = window.end - window.begin
        # The scheduler's own timestep grid stays whole: step() looks up its position in it. MUST run after
        # the scheduler-reset step, whose set_timesteps resets the begin index to None.
        components.scheduler.set_begin_index(window.begin)

        for i, t in enumerate(state.timesteps[window.begin : window.end]):
            timestep = torch.stack([t])

            # The global step, so the guider's step-range logic sees where this window sits in the schedule.
            components.guider.set_state(
                step=window.begin + i, num_inference_steps=state.num_inference_steps, timestep=t
            )
            guider_state = components.guider.prepare_inputs(guider_inputs)

            for guider_state_batch in guider_state:
                components.guider.prepare_models(components.transformer)

                guider_state_batch.noise_pred = components.transformer(
                    [state.latents.to(transformer_dtype)],
                    timestep=timestep,
                    encoder_hidden_states=[guider_state_batch.encoder_hidden_states[0].to(transformer_dtype)],
                    condition_latents=[state.reference_latents.to(transformer_dtype)],
                    kv_cache=state.kv_cache,
                    kv_cache_mode="cached",
                    seq_len=state.max_seq_len,
                    reference_grid_sizes=state.grid_sizes_ref,
                    origin_len=state.segment_frame_length,
                    origin_area=[state.height, state.width],
                    is_uncondtion=guider_state_batch.is_uncondtion,
                    **shared_kwargs,
                ).sample[0]

                components.guider.cleanup_models(components.transformer)

            noise_pred = components.guider(guider_state)[0]

            latents = components.scheduler.step(
                noise_pred.unsqueeze(0),
                t,
                state.latents.unsqueeze(0),
                return_dict=False,
                generator=state.generator,
            )[0]
            state.latents = latents.squeeze(0)

            if self.callback is not None:
                # The step index runs across segments (`k * window_len + i`), but the Generate node sizes its
                # progress bar to one window (`get_strength_affected_steps`). With more than one segment the
                # bar is therefore full after segment 1 and the engine warns on each later step. Accepted for
                # now (plan Q3).
                # The preview gets this segment without its reference slot, so it decodes as a plain Wan
                # latent. The return value is deliberately discarded: the callback returns {} on its normal
                # path, and merging that would clobber the loop's latents.
                self.callback(components, k * window_len + i, t, {"latents": state.latents[None, :, 1:]})
            if getattr(components, "_interrupt", False):
                break

        return components, block_state


class _WanAnimate2SegmentCollectStep(ModularPipelineBlocks):
    """Copy this segment's kept latents into the public latent, and keep its prefix and tail for meta.

    Must run before ``WanAnimate2SegmentDecodeStep``, which sets ``latents`` to None. The output buffers
    are seeded by the loop wrapper, as upstream seeds ``segment_frames``.
    """

    model_name = "wan-animate-2"

    @property
    def description(self) -> str:
        return "Writes this segment's kept latents into the public latent and records its prefix and tail."

    @property
    def inputs(self) -> list[InputParam]:
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            InputParam,
        )

        return [InputParam("segment_slices", required=True, description="One `SegmentSlice` per segment")]

    @no_grad
    def __call__(
        self, components: WanAnimate2ModularPipeline, block_state: BlockState, k: int
    ) -> tuple[WanAnimate2ModularPipeline, BlockState]:
        state = cast(Any, block_state)
        segment: SegmentSlice = state.segment_slices[k]
        kept_stop = segment.prefix_len + (segment.public_stop - segment.start)

        state.output_latents[0, :, segment.start : segment.public_stop] = state.latents[
            :, segment.prefix_len : kept_stop
        ]
        state.segment_prefix_latents.append(state.latents[:, : segment.prefix_len].clone())
        state.segment_tail_latents.append(state.latents[:, kept_stop:].clone())

        return components, block_state


class _WanAnimate2CallbackSegmentLoop(WanAnimate2SegmentLoopWrapper):
    """Upstream's segment loop with the noise step swapped, a collect step added, and an interrupt break.

    ``block_classes`` / ``block_names`` are re-declared rather than inherited because
    ``LoopSequentialPipelineBlocks.__init__`` builds ``sub_blocks`` from them.
    """

    block_classes: ClassVar[list[type]] = [
        WanAnimate2SegmentVaeEncoderStep,
        WanAnimate2SegmentPrevFramesStep,
        _WanAnimate2SegmentNoiseStep,
        WanAnimate2SegmentSchedulerResetStep,
        WanAnimate2RefExtractStep,
        _WanAnimate2CallbackDenoiseInner,
        _WanAnimate2SegmentCollectStep,
        WanAnimate2SegmentDecodeStep,
    ]
    block_names: ClassVar[list[str]] = [
        "vae_encoder",
        "prev_frames",
        "noise",
        "scheduler_reset",
        "ref_extract",
        "denoise_inner",
        "collect",
        "decode",
    ]

    @property
    def description(self) -> str:
        return (
            "Iterates over the driving video's segments, collecting each segment's latents into one public "
            "latent, with a step-end callback and cancellation support."
        )

    @property
    def loop_intermediate_outputs(self) -> list[OutputParam]:
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.modular_pipeline_utils import (  # type: ignore[reportMissingImports]
            OutputParam,
        )

        return [
            *super().loop_intermediate_outputs,
            OutputParam("output_latents", type_hint=torch.Tensor, description="The public latent, filled in"),
            OutputParam("segment_prefix_latents", type_hint=list, description="One prefix per completed segment"),
            OutputParam("segment_tail_latents", type_hint=list, description="One tail per completed segment"),
        ]

    @no_grad
    def __call__(
        self, components: WanAnimate2ModularPipeline, state: PipelineState
    ) -> tuple[WanAnimate2ModularPipeline, PipelineState]:
        block_state = cast(Any, self.get_block_state(state))

        # `out_frames` is the previous segment's decoded pixels, read by the next prev-frames step.
        block_state.segment_frames = []
        block_state.out_frames = None
        # Segments that never run (after a cancel) keep their input noise, so the shape never changes.
        block_state.output_latents = block_state.input_latents.clone()
        block_state.segment_prefix_latents = []
        block_state.segment_tail_latents = []

        for k in range(block_state.num_segments):
            components, block_state = cast(Any, self.loop_step(components, block_state, k=k))
            # The decode step appends each segment's pixels for upstream's final assembly. This driver
            # returns latents and decodes later, so holding them would only grow CPU memory per segment.
            block_state.segment_frames.clear()
            if getattr(components, "_interrupt", False):
                break

        self.set_block_state(state, block_state)
        return components, state


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------


class WanAnimate2LatentPipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True
    video_fps: ClassVar[int] = WAN_ANIMATE_2_FPS

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
        # Same `mod_value` upstream floors each side to: VAE factor x the transformer's 2x2 patch.
        return cast(WanAnimate2ModularPipeline, self.modular_pipe).vae_scale_factor_spatial * 2

    @override
    def _get_temporal_alignment(self) -> int | None:
        return cast(WanAnimate2ModularPipeline, self.modular_pipe).vae_scale_factor_temporal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _run_blocks(self, blocks: Any, state: PipelineState | None = None, **kwargs: Any) -> PipelineState:
        """Run ``blocks`` over one shared ``PipelineState`` and return it.

        Deliberately not ``_call_block``, for two reasons. It builds a fresh state per call, so the
        segment loop could not continue from the prefix run's state. And it runs under
        ``inference_mode``, which mints inference tensors: the loop mutates ``output_latents`` in place,
        and in-place mutation of an inference tensor outside ``inference_mode`` raises. ``no_grad``
        produces ordinary tensors and avoids the whole class of problem.
        """
        import torch  # type: ignore[reportMissingImports]

        if state is None:
            state = PipelineState()
        for param in blocks.inputs:
            if param.name in kwargs:
                state.set(param.name, kwargs[param.name], param.kwargs_type)
        with torch.no_grad():
            _, result = blocks(self.modular_pipe, state)
        return result

    def _ensure_transformer_compiled(self) -> None:
        """Compile the transformer's repeated blocks the first time this pipe denoises.

        The generation attention runs on flex attention, which only fits in memory compiled (see the
        `WanAnimate2AttnProcessor` docstring in diffusers). Compiling here rather than at build keeps
        encode, decode and noise free of it. ``.compile()`` is lazy, so the real cost falls on the first
        transformer forward.
        """
        transformer = cast(WanAnimate2ModularPipeline, self.modular_pipe).transformer
        if getattr(transformer, _COMPILED_FLAG, False):
            return
        if sys.platform == "win32" and importlib.util.find_spec("triton") is None:
            raise RuntimeError(
                f"{self.driver_namespace}: Attempted to compile the WAN Animate 2 transformer. Failed because "
                f"Triton is not installed. WAN Animate 2 needs torch.compile for its flex attention, and "
                f"torch.compile needs Triton on CUDA. On Windows, install 'triton-windows<3.4' into the "
                f"library's environment."
            )

        transformer.compile_repeated_blocks()
        setattr(transformer, _COMPILED_FLAG, True)

    # ------------------------------------------------------------------
    # Public latent surface
    # ------------------------------------------------------------------

    @override
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        import torch  # type: ignore[reportMissingImports]

        pipe = cast(WanAnimate2ModularPipeline, self.modular_pipe)
        generator = generator_state.to_generator()
        num_frames = source_shape[-3]
        height = source_shape[-2]
        width = source_shape[-1]

        state = self._run_blocks(
            _WanAnimate2PrepareNoiseStep(),
            num_latent_frames=(num_frames - 1) // pipe.vae_scale_factor_temporal + 1,
            latent_height=height // pipe.vae_scale_factor_spatial,
            latent_width=width // pipe.vae_scale_factor_spatial,
            generator=generator,
        )
        latents = self._get_required(state.values, "latents", torch.Tensor)
        return self._make_latent_artifact(
            latents,
            source_shape=source_shape,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        """Decode per segment when the meta allows it, else decode the tensor as one plain Wan latent."""
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.wan_animate_2.denoise import (  # type: ignore[reportMissingImports]
            decode_vae,
        )

        pipe = cast(WanAnimate2ModularPipeline, self.modular_pipe)
        device, _ = self._get_device_and_type()
        num_frames = latent.source_shape[-3]
        height = latent.source_shape[-2]
        width = latent.source_shape[-1]
        tensor = latent.to_torch(device=device, dtype=torch.float32)

        prefixes = read_driver_meta(latent, SEGMENT_PREFIX_LATENTS_META_KEY, self.driver_namespace)
        num_segments = read_driver_meta(latent, NUM_SEGMENTS_META_KEY, self.driver_namespace)

        segment_frames: list[torch.Tensor] = []
        with torch.no_grad():
            if prefixes is None or num_segments is None or len(prefixes) != num_segments:
                # The live preview, Save/Load and Empty Latent all reach here without meta, and so does a
                # cancelled run. Debug rather than warning because the preview path decodes once per step.
                logger.debug(
                    "%s: decoding as one segment because the latent carries no complete segment meta.",
                    self.driver_namespace,
                )
                segment_frames.append(decode_vae(pipe.vae, tensor).cpu())
            else:
                prev_segment_conditioning_frames = read_driver_meta(
                    latent, PREV_SEGMENT_CONDITIONING_FRAMES_META_KEY, self.driver_namespace
                )
                slices = segment_slices(
                    read_driver_meta(latent, SEGMENT_FRAME_LENGTH_META_KEY, self.driver_namespace),
                    prev_segment_conditioning_frames,
                    num_segments,
                    tensor.shape[2],
                )
                for k, segment in enumerate(slices):
                    # Rebuild the segment as the loop decoded it, minus the reference slot and the tail: the
                    # Wan VAE is causal in time, so dropping trailing latents leaves earlier frames unchanged.
                    overlap = prefixes[k][:, 1:].to(device=device, dtype=torch.float32)
                    content = tensor[0, :, segment.start : segment.public_stop]
                    frames = decode_vae(pipe.vae, torch.cat([overlap, content], dim=1)[None])
                    if k > 0:
                        # Same as `WanAnimate2SegmentDecodeStep`: the overlap re-generates frames the previous
                        # segment already produced.
                        frames = frames[:, :, prev_segment_conditioning_frames:]
                    segment_frames.append(frames.cpu())

        state = self._run_blocks(
            pipe.blocks.sub_blocks["decode"],
            segment_frames=segment_frames,
            real_frame_len=num_frames,
            # The driver stretches its inputs to the latent's size, so there are never letterbox bars.
            crop_region=(0, 0, height, width),
            output_type="pil",
        )
        return self._get_required(state.values, "videos", list)[0]

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        """Encode video frames as a normalised Wan video latent (5-D ``[B, C, T, H/8, W/8]``)."""
        import torch  # type: ignore[reportMissingImports]
        from diffusers.modular_pipelines.wan_animate_2.encoders import (  # type: ignore[reportMissingImports]
            encode_vae,
        )

        if isinstance(media, ImageMedia):
            # The reference image is conditioning, not a latent: it reaches the model through the `image`
            # input on Generate Media Latents.
            raise NotImplementedError(
                f"{self.driver_namespace} does not support image encoding. Use a video input instead, and "
                f"connect the reference image to the 'image' input of Generate Media Latents."
            )

        pipe = cast(WanAnimate2ModularPipeline, self.modular_pipe)
        device, _ = self._get_device_and_type()
        height = media.source_shape[-2]
        width = media.source_shape[-1]

        # The Encode node has already snapped the frame count and resized the frames to these values.
        pixels = pipe.video_processor.preprocess_video(media.frames, height=height, width=width)
        with torch.no_grad():
            latents = encode_vae(pipe.vae, pixels.to(device=device, dtype=torch.float32))
        return self._make_latent_artifact(latents, source_shape=media.source_shape)

    @override
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        # A noised video cannot become segment conditioning: each segment conditions on the previous
        # segment's fully denoised pixels, and there is no block path for starting mid-schedule from one.
        raise NotImplementedError(
            f"{self.driver_namespace} does not support adding noise to an existing latent "
            f"(no video-to-video path). Use Create Noise Latents for a fresh latent."
        )

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------

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
        from diffusers.modular_pipelines.wan_animate_2.encoders import (  # type: ignore[reportMissingImports]
            get_frame_indices,
        )

        pipe = cast(WanAnimate2ModularPipeline, self.modular_pipe)
        update_kwargs = kwargs.copy()
        update_kwargs.pop(MediaGenConditioningKey.OUTPUT, None)
        guidance_scale = update_kwargs.pop("guidance_scale", None)
        image = update_kwargs.pop("image", None)
        driving_video = update_kwargs.pop("driving_video", None)
        driving_video_fps = update_kwargs.pop("driving_video_fps", None)
        segment_frame_length = update_kwargs.pop("segment_frame_length", DEFAULT_SEGMENT_FRAME_LENGTH)
        prev_segment_conditioning_frames = update_kwargs.pop(
            "prev_segment_conditioning_frames", DEFAULT_PREV_SEGMENT_CONDITIONING_FRAMES
        )
        generator = update_kwargs.pop("generator", generator_state.to_generator())

        # --- Failure checks -------------------------------------------------
        if isinstance(latent, InpaintMaskArtifact):
            raise NotImplementedError(f"{self.driver_namespace} does not support inpainting.")
        if return_fully_denoised:
            raise NotImplementedError(
                f"{self.driver_namespace} does not support 'return_fully_denoised' because each segment "
                f"conditions on the previous segment's fully denoised pixels."
            )
        if start_step < 0:
            raise ValueError(f"start_step must be non-negative, got {start_step}.")
        if end_step != -1 and start_step >= end_step:
            raise ValueError(f"start_step must be less than end_step, got {start_step} and {end_step}.")
        if not hasattr(pipe.scheduler, "set_begin_index"):
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed because the "
                f"scheduler '{type(pipe.scheduler).__name__}' has no set_begin_index, which the step window needs."
            )
        if image is None:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed because no "
                f"reference 'image' was given."
            )
        if driving_video is None:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed because no "
                f"'driving_video' was given."
            )
        # The latent contract depends on these: `prev = 4N + 1` pixel frames decode from whole latents.
        if (
            segment_frame_length % VAE_TEMPORAL_FACTOR != 1
            or prev_segment_conditioning_frames % VAE_TEMPORAL_FACTOR != 1
        ):
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed with "
                f"segment_frame_length={segment_frame_length} and prev_segment_conditioning_frames="
                f"{prev_segment_conditioning_frames} because both must be of the form 4N + 1."
            )
        if prev_segment_conditioning_frames >= segment_frame_length:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed with "
                f"prev_segment_conditioning_frames={prev_segment_conditioning_frames} because it must be less "
                f"than segment_frame_length={segment_frame_length}."
            )

        source_shape = latent.source_shape
        num_frames = source_shape[-3]
        height = source_shape[-2]
        width = source_shape[-1]
        if num_frames <= prev_segment_conditioning_frames:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed with "
                f"num_frames={num_frames} because it must be greater than prev_segment_conditioning_frames="
                f"{prev_segment_conditioning_frames}. Raise num_frames on Create Noise Latents."
            )

        frames = list(driving_video)
        if driving_video_fps is not None and driving_video_fps > 0:
            frames = [frames[i] for i in get_frame_indices(len(frames), driving_video_fps, WAN_ANIMATE_2_FPS)]
        if len(frames) < num_frames:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to denoise a WAN Animate 2 latent. Failed with "
                f"num_frames={num_frames} because the driving video has only {len(frames)} frames at "
                f"{WAN_ANIMATE_2_FPS} fps. Lower num_frames on Create Noise Latents or use a longer driving video."
            )

        num_segments = segment_count(num_frames, segment_frame_length, prev_segment_conditioning_frames)
        is_partial_window = start_step > 0 or end_step != -1
        if is_partial_window and num_segments > 1:
            # TODO(wan-animate-2 Q5): temporary. Segment k+1 conditions on segment k's *fully denoised*
            # pixels, so a window cannot stop partway through a multi-segment run. See the plan's Q5 for
            # the options.
            raise NotImplementedError(
                f"{self.driver_namespace}: Attempted a partial denoise (start_step={start_step}, "
                f"end_step={end_step}). Failed because this clip needs {num_segments} segments, and partial "
                f"denoise works only for a single segment. Lower num_frames to at most {segment_frame_length} "
                f"or denoise in one pass."
            )

        prefix_latents_in = None
        tail_latents_in = None
        if start_step > 0:
            prefix_latents_in = read_driver_meta(latent, SEGMENT_PREFIX_LATENTS_META_KEY, self.driver_namespace)
            tail_latents_in = read_driver_meta(latent, SEGMENT_TAIL_LATENTS_META_KEY, self.driver_namespace)
            has_segment_meta = (
                prefix_latents_in is not None
                and tail_latents_in is not None
                and len(prefix_latents_in) == num_segments
                and len(tail_latents_in) == num_segments
            )
            if not has_segment_meta:
                # Fresh prefix and tail noise next to half-denoised content would be stepped as if it were
                # already partly denoised, which damages the result.
                raise ValueError(
                    f"{self.driver_namespace}: Attempted to resume a WAN Animate 2 denoise at step "
                    f"{start_step}. Failed because the input latent carries no segment meta, so the reference "
                    f"slot and padding would restart from pure noise mid-schedule. Chain partial denoise "
                    f"directly from a previous Generate Media Latents run."
                )

        # --- Fit the inputs to the latent ---------------------------------
        if len(frames) > num_frames:
            logger.info("%s: using the first %d of %d driving frames.", self.driver_namespace, num_frames, len(frames))
        fitted_frames = [frame.resize((width, height)) for frame in frames[:num_frames]]
        # Pad here rather than in upstream's video step, whose zigzag pad returns too few frames when the
        # pad is longer than the clip. With a whole number of segments its own pad then does nothing.
        target_frames = padded_frame_count(num_frames, segment_frame_length, prev_segment_conditioning_frames)
        padded_frames = [fitted_frames[i] for i in reflect_indices(num_frames, target_frames)]
        stretched_image = image.resize((width, height))

        # Before the text encoder runs: `requires_unconditional_embeds` reads the guider.
        if guidance_scale is not None:
            pipe.guider.guidance_scale = guidance_scale

        self._ensure_transformer_compiled()

        # --- Prefix run: encoders and segment geometry -----------------------
        device, _ = self._get_device_and_type()
        latents_in = latent.to_torch(device=device, dtype=torch.float32)
        prefix_blocks = SequentialPipelineBlocks.from_blocks_dict(
            {
                "text_encoder": pipe.blocks.sub_blocks["text_encoder"],
                "image_preprocess": _WanAnimate2ExactImageInputStep(),
                "image_clip": pipe.blocks.sub_blocks["image_encoder"].sub_blocks["encode"],
                "video_encoder": pipe.blocks.sub_blocks["video_encoder"],
                "vae_encoder": pipe.blocks.sub_blocks["vae_encoder"],
                "prepare_segments": pipe.blocks.sub_blocks["denoise"].sub_blocks["prepare_segments"],
            }
        )
        state = self._run_blocks(
            prefix_blocks,
            image=stretched_image,
            driving_video=padded_frames,
            # Already resampled above, so the video step must not resample again.
            driving_video_fps=None,
            height=height,
            width=width,
            segment_frame_length=segment_frame_length,
            prev_segment_conditioning_frames=prev_segment_conditioning_frames,
            **update_kwargs,
        )

        # --- Segment loop ----------------------------------------------------
        loop = _WanAnimate2CallbackSegmentLoop()
        denoise_inner = cast(_WanAnimate2CallbackDenoiseInner, loop.sub_blocks["denoise_inner"])
        denoise_inner.callback = callback
        denoise_inner.start_step = start_step
        denoise_inner.end_step = end_step

        # The framework signals cancellation by setting `_interrupt` on the pipe, but only when the
        # attribute already exists — and ModularPipeline has none. The pipe is cached across runs,
        # so a leaked True would break the next run at step 0.
        pipe._interrupt = False  # type: ignore[reportAttributeAccessIssue]
        try:
            state = self._run_blocks(
                loop,
                state,
                input_latents=latents_in,
                segment_slices=segment_slices(
                    segment_frame_length, prev_segment_conditioning_frames, num_segments, latents_in.shape[2]
                ),
                prefix_latents_in=prefix_latents_in,
                tail_latents_in=tail_latents_in,
                generator=generator,
                num_inference_steps=num_inference_steps,
            )
        finally:
            pipe._interrupt = False  # type: ignore[reportAttributeAccessIssue]

        output_latents = self._get_required(state.values, "output_latents", torch.Tensor)
        segment_prefix_latents = self._get_required(state.values, "segment_prefix_latents", list)
        segment_tail_latents = self._get_required(state.values, "segment_tail_latents", list)
        return self._make_latent_artifact(
            output_latents,
            source_shape=source_shape,
            upstream=latent,
            meta={
                SEGMENT_FRAME_LENGTH_META_KEY: segment_frame_length,
                PREV_SEGMENT_CONDITIONING_FRAMES_META_KEY: prev_segment_conditioning_frames,
                NUM_SEGMENTS_META_KEY: num_segments,
                SEGMENT_PREFIX_LATENTS_META_KEY: segment_prefix_latents,
                SEGMENT_TAIL_LATENTS_META_KEY: segment_tail_latents,
                **GeneratorState.from_generator(generator).as_meta(),
            },
        )
