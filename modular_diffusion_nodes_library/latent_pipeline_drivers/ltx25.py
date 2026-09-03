"""LTX-2.5 driver: fully modular (Modular Diffusers) integration.

Unlike ``ltx2.py`` (LTX-2.3, fixed ``DiffusionPipeline``), ``self._pipe`` here already IS the
``LTX25ModularPipeline``, so this driver owns the denoise loop instead of delegating to
``super().denoise_latent()`` (which raises for any ``ModularPipeline``, see ``base_driver.py``).
Owning the loop is what makes partial denoise, live preview and mid-run cancellation available.

Modelled on ``minimax_h3.py`` — the only other driver whose ``self._pipe`` is a ``ModularPipeline`` —
for the ``modular_pipe`` override, the driver-owned ``_run_blocks`` helper, and the meta-based
audio side-channel pattern (video and audio are denoised jointly; only the video latent is the
public artifact, audio rides in this driver's namespaced ``meta`` sub-bag, paired by fingerprint).

One exception is kept on the fixed pipeline: **HDR**. diffusers ships no modular block coverage for
``LTX2HDRPipeline`` (no "hdr" workflow, no HDR ``ModularPipeline`` subclass) as of diffusers 0.40.0.
When an HDR IC-LoRA adapter is active, this driver hand-builds an ``LTX2HDRPipeline`` from components
already loaded on the modular pipe (every component it needs — ``scheduler``, ``vae``, ``audio_vae``,
``text_encoder``, ``tokenizer``, ``connectors``, ``transformer``, ``vocoder`` — is already present),
temporarily swaps ``self._pipe`` to it, and delegates to ``super().denoise_latent()`` (the base
class's fixed-``DiffusionPipeline`` fallback), exactly like the Pattern-C pipe-swap ``ltx2.py`` uses
for its variants, except the swap target is hand-built rather than produced via ``.from_pipe()``
(``ModularPipeline`` has no such constructor).

Ordinary (non-HDR) media-gen conditioning (first frame, multi-keyframe) and IC-LoRA reference-video
conditioning both migrate to native modular blocks (the ``condition`` / ``in_context`` workflows of
``LTX25AutoBlocks``), eliminating the fixed-pipeline swaps ``ltx2.py`` needs for those cases.
"""

import logging
import math
from typing import Any, ClassVar, cast, override

import numpy as np
import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.ltx2.denoise import (  # type: ignore[reportMissingImports]
    LTX2ConditionLoopAfterDenoiser,
    LTX2ConditionLoopBeforeDenoiser,
    LTX2DenoiseLoopWrapper,
    LTX2LoopAfterDenoiser,
    LTX2LoopBeforeDenoiser,
    LTX2LoopDenoiser,
)
from diffusers.modular_pipelines.ltx2.guider import LTX2Guidance  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.ltx2.modular_blocks_ltx2 import (  # type: ignore[reportMissingImports]
    LTX2ConditionCoreDenoiseStep,
    LTX2CoreDenoiseStep,
    LTX2InContextCoreDenoiseStep,
)
from diffusers.modular_pipelines.ltx2.modular_pipeline import LTX2ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    PipelineState,
    SequentialPipelineBlocks,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.pipeline_ltx2_hdr_lora import (  # type: ignore[reportMissingImports]
    LTX2HDRPipeline,
    LTX2HDRReferenceCondition,
)
from diffusers.pipelines.ltx2.pipeline_ltx2_ic_lora import LTX2ReferenceCondition  # type: ignore[reportMissingImports]
from diffusers.pipelines.ltx2.utils import (  # type: ignore[reportMissingImports]
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
)
from diffusers.utils.torch_utils import randn_tensor  # type: ignore[reportMissingImports]
from griptape_nodes.files.path_utils import canonicalize_for_io
from PIL.Image import Image
from safetensors import safe_open  # type: ignore[reportMissingImports]

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
    MediaGenConditioningPayload,
    normalize_to_payloads,
)
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    MediaGenConditioningKey,
    pixel_frame_index_to_latent_index,
    resize_frames_scale_to_fill,
    resolve_conditioning_image,
    resolve_conditioning_video,
    resolve_frame_index,
)

logger = logging.getLogger("modular_diffusers_nodes_library")

#: Keys under which LTX-2.5-specific values ride in this driver's namespaced ``meta`` sub-bag.
AUDIO_LATENTS_META_KEY = "audio_latents"
AUDIO_NUM_FRAMES_META_KEY = "audio_num_frames"
#: Number of generated-video tokens before any appended keyframe/reference tokens. Present only for
#: the ``condition`` / ``in_context`` workflows; read back at decode time to trim those tokens before
#: unpacking (mirrors ``LTX2TrimConditionTokensStep``).
BASE_TOKEN_COUNT_META_KEY = "base_token_count"
#: Fingerprint of the video latent the audio latent was denoised with — same rationale as
#: MiniMax-H3's identically-named key: latent math shallow-merges meta left-operand-wins, so a summed
#: video latent keeps the left operand's unsummed audio. See ``_video_fingerprint``/``_fingerprints_match``.
AUDIO_PAIRED_WITH_META_KEY = "audio_paired_with"
#: Records which concrete pipeline produced the latent (``LTX25ModularPipeline`` or the hand-built
#: ``LTX2HDRPipeline``), so ``decode_latent`` can route HDR output through the HDR postprocessing path
#: without re-inspecting live adapter state (mirrors ``ltx2.py``'s identically-named mechanism).
_PIPELINE_CLASS_META_KEY = "pipeline_class"


def _pack_latents(latents: torch.Tensor, patch_size: int = 1, patch_size_t: int = 1) -> torch.Tensor:
    """Pack a 5-D video latent into transformer token rows.

    Mirrors ``diffusers.modular_pipelines.ltx2.denoise._pack_latents`` (also redefined identically in
    ``decoders.py``); redefined here rather than imported since it is a private module-level helper.
    """
    batch_size, num_channels, num_frames, height, width = latents.shape
    latents = latents.reshape(
        batch_size,
        -1,
        num_frames // patch_size_t,
        patch_size_t,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    )
    latents = latents.permute(0, 2, 4, 6, 1, 3, 5, 7).flatten(4, 7).flatten(1, 3)
    return latents


def _unpack_latents(
    latents: torch.Tensor, num_frames: int, height: int, width: int, patch_size: int = 1, patch_size_t: int = 1
) -> torch.Tensor:
    """Inverse of :func:`_pack_latents`. Mirrors the diffusers ``ltx2`` helper of the same name."""
    batch_size = latents.size(0)
    latents = latents.reshape(batch_size, num_frames, height, width, -1, patch_size_t, patch_size, patch_size)
    latents = latents.permute(0, 4, 1, 5, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(2, 3)
    return latents


def _pack_audio_latents(latents: torch.Tensor) -> torch.Tensor:
    """Pack a ``[B, C, L, M]`` audio latent into ``[B, L, C * M]`` rows (implicit patch sizes of 1/M).

    Mirrors ``diffusers.modular_pipelines.ltx2.before_denoise._pack_audio_latents`` called with no
    explicit patch sizes.
    """
    return latents.transpose(1, 2).flatten(2, 3)


def _video_fingerprint(tensor: torch.Tensor) -> tuple[tuple[int, ...], float, float]:
    """Cheap value-sensitive fingerprint of a video latent. Shape alone would not do: latent math
    preserves shape and changes only values.
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


class _LTX25CallbackMixin:
    """Adds partial denoise, a step-end callback and an interrupt break to an ``LTX2DenoiseLoopWrapper``.

    Mixed into three concrete subclasses below, one per ``block_classes``/``block_names`` triple LTX-2.5
    uses (text-to-video, condition, in-context reuses the condition triple unchanged) — re-declared per
    subclass rather than inherited because ``LoopSequentialPipelineBlocks.__init__`` builds ``sub_blocks``
    from them (mirrors ``minimax_h3.py``'s identically-shaped ``_MiniMaxH3CallbackDenoiseStep``).
    """

    #: Assigned per-instance after construction. A block's ``__init__`` takes no arguments and its
    #: signature is introspected for config, so these must not become constructor kwargs.
    callback: Any = None
    start_step: int = 0
    end_step: int = -1

    @property
    def loop_inputs(self) -> list[Any]:
        # `InputParam` is resolved lazily here (see subclasses) to avoid a second import block; declared
        # in addition to the base class's `timesteps`/`num_inference_steps` because the step-end preview
        # needs the video geometry, `audio_scheduler`'s begin-index needs resetting alongside the video
        # scheduler's, and `base_token_count` (condition/in-context only) is needed to trim appended
        # tokens before unpacking the preview.
        from diffusers.modular_pipelines.modular_pipeline_utils import InputParam  # noqa: PLC0415

        return [
            *super().loop_inputs,  # type: ignore[misc]
            InputParam("audio_scheduler", required=True),
            InputParam("height", default=None),
            InputParam("width", default=None),
            InputParam("num_frames", default=None),
            InputParam("base_token_count", default=None),
        ]

    def _resolve_window(self, num_steps: int) -> tuple[int, int]:
        """Clamp the requested step window against the schedule's real length.

        ``num_inference_steps`` counts the sigma grid including the terminal zero, so the schedule
        drives one fewer model evaluation than the UI number. ``end_step`` equal to the UI value is
        therefore normal and must clamp rather than raise.
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

    def _unpack_preview(self, components: Any, block_state: Any) -> torch.Tensor | None:
        if block_state.height is None or block_state.width is None or block_state.num_frames is None:
            return None
        latent_num_frames = (block_state.num_frames - 1) // components.vae_temporal_compression_ratio + 1
        latent_height = block_state.height // components.vae_spatial_compression_ratio
        latent_width = block_state.width // components.vae_spatial_compression_ratio
        rows = block_state.latents
        if block_state.base_token_count is not None:
            rows = rows[:, : block_state.base_token_count]
        return _unpack_latents(
            rows,
            latent_num_frames,
            latent_height,
            latent_width,
            components.transformer_spatial_patch_size,
            components.transformer_temporal_patch_size,
        )

    @torch.no_grad()
    def __call__(self, components: Any, state: PipelineState) -> PipelineState:
        block_state = cast(Any, self.get_block_state(state))  # type: ignore[attr-defined]

        begin, end = self._resolve_window(len(block_state.timesteps))
        # Slice both schedules' `timesteps` arrays by the identical window, keeping video and audio in
        # lockstep. The sigma grids themselves stay whole: `step()` needs `sigmas[i + 1]`, and
        # `index_for_timestep` searches the unsliced timesteps.
        block_state.timesteps = block_state.timesteps[begin:end]
        components.scheduler.set_begin_index(begin)
        block_state.audio_scheduler.set_begin_index(begin)

        with self.progress_bar(total=len(block_state.timesteps)) as progress_bar:  # type: ignore[attr-defined]
            for i, t in enumerate(block_state.timesteps):
                components, block_state = cast(Any, self.loop_step(components, block_state, i=i, t=t))  # type: ignore[attr-defined]
                progress_bar.update()
                if self.callback is not None:
                    preview = self._unpack_preview(components, block_state)
                    if preview is not None:
                        # Return value deliberately discarded: the callback returns {} on its normal
                        # path, and merging that would clobber the loop's own `latents`.
                        self.callback(components, i, t, {"latents": preview})
                if getattr(components, "_interrupt", False):
                    break

        self.set_block_state(state, block_state)  # type: ignore[attr-defined]
        return components, state  # type: ignore[return-value]


class _LTX25TextToVideoCallbackDenoiseStep(_LTX25CallbackMixin, LTX2DenoiseLoopWrapper):
    block_classes: ClassVar[list[type]] = [LTX2LoopBeforeDenoiser, LTX2LoopDenoiser, LTX2LoopAfterDenoiser]
    block_names: ClassVar[list[str]] = ["before_denoiser", "denoiser", "after_denoiser"]


class _LTX25ConditionCallbackDenoiseStep(_LTX25CallbackMixin, LTX2DenoiseLoopWrapper):
    block_classes: ClassVar[list[type]] = [
        LTX2ConditionLoopBeforeDenoiser,
        LTX2LoopDenoiser,
        LTX2ConditionLoopAfterDenoiser,
    ]
    block_names: ClassVar[list[str]] = ["before_denoiser", "denoiser", "after_denoiser"]


#: Workflow name -> (diffusers' own composite denoise-block class to mirror, our callback-enabled
#: replacement for its "denoise" sub-block). "condition" and "in_context" share the same inner triple
#: (see `LTX2InContextCoreDenoiseStep`'s docstring: "Reuses the condition denoise step unchanged").
_WORKFLOW_TABLE: dict[str, tuple[type[SequentialPipelineBlocks], type[LTX2DenoiseLoopWrapper]]] = {
    "text2video": (LTX2CoreDenoiseStep, _LTX25TextToVideoCallbackDenoiseStep),
    "condition": (LTX2ConditionCoreDenoiseStep, _LTX25ConditionCallbackDenoiseStep),
    "in_context": (LTX2InContextCoreDenoiseStep, _LTX25ConditionCallbackDenoiseStep),
}


class LTX25LatentPipelineDriver(LatentPipelineDriver):
    produces_video: ClassVar[bool] = True
    video_fps: ClassVar[int] = 24

    _HDR_LORA_ADAPTER_TOKEN: ClassVar[str] = "ic-lora-hdr"
    _IC_LORA_REFERENCE_KEY: ClassVar[str] = "ltx2_ic_lora_reference"

    def __init__(self, pipe: ModularPipeline):
        super().__init__(pipe)
        self._hdr_pipe: LTX2HDRPipeline | None = None

    @property
    @override
    def modular_pipe(self) -> ModularPipeline:
        # self._pipe already IS the modular pipeline; the base implementation would round-trip
        # `update_components(**pipe.components)`, which drops the load spec of components lacking a
        # `_diffusers_load_id` and re-registers everything with the ComponentsManager that owns this
        # model's CPU-offload hooks.
        return cast(ModularPipeline, self._pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        return cast(ModularPipeline, self._pipe)

    @classmethod
    @override
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        return False

    @override
    def _get_temporal_alignment(self) -> int | None:
        return cast(LTX2ModularPipeline, self.modular_pipe).vae_temporal_compression_ratio

    @override
    def _get_spatial_alignment(self) -> int:
        return cast(LTX2ModularPipeline, self.modular_pipe).vae_spatial_compression_ratio

    # ------------------------------------------------------------------
    # HDR IC-LoRA adapter detection (shared by the fixed-pipeline HDR path and decode routing)
    # ------------------------------------------------------------------

    @property
    def is_hdr_lora_active(self) -> bool:
        """Whether an LTX-2 HDR IC-LoRA adapter is currently loaded on the modular pipe."""
        token = self._HDR_LORA_ADAPTER_TOKEN
        return any(token in name.lower() for name in self._get_loaded_adapter_names())

    def _latent_was_produced_for_hdr(self, latent: LatentArtifact) -> bool:
        stamped = read_driver_meta(latent, _PIPELINE_CLASS_META_KEY, self.driver_namespace)
        if stamped is not None:
            return stamped == LTX2HDRPipeline.__name__
        return self.is_hdr_lora_active

    def _get_loaded_adapter_names(self) -> list[str]:
        get_list_adapters = getattr(self.modular_pipe, "get_list_adapters", None)
        if get_list_adapters is None:
            return []
        adapters_by_component = get_list_adapters()
        return [name for names in adapters_by_component.values() for name in names]

    def _infer_reference_downscale_factor_from_loras(self) -> int | None:
        """Read `reference_downscale_factor` from any loaded IC-LoRA's safetensors metadata.

        Mirrors ``ltx2.py``'s identically-named method: the LoRA loaders stash safetensors header
        metadata on the pipe as ``pipe._gtn_lora_metadata[adapter_name] = {"path": ..., "metadata": {...}}``.
        """
        lora_metadata = getattr(self.modular_pipe, "_gtn_lora_metadata", None)
        loaded_adapter_names = set(self._get_loaded_adapter_names())
        if not lora_metadata or not loaded_adapter_names:
            return None

        for adapter_name, entry in lora_metadata.items():
            if adapter_name in loaded_adapter_names:
                metadata = entry.get("metadata") or {}
                raw_value = metadata.get("reference_downscale_factor")
                if raw_value is not None:
                    try:
                        factor = float(raw_value)
                    except (TypeError, ValueError):
                        logger.warning(
                            "LTX25: ignoring non-numeric reference_downscale_factor=%r in %s",
                            raw_value,
                            entry.get("path"),
                        )
                    else:
                        if factor < 1:
                            logger.warning(
                                "LTX25: ignoring invalid reference_downscale_factor=%f in %s",
                                factor,
                                entry.get("path"),
                            )
                        else:
                            return int(factor)
        return None

    def _stamp_pipeline_class(self, artifact: LatentArtifact, pipeline_class_name: str) -> LatentArtifact:
        return self._make_latent_artifact(
            artifact.to_torch(),
            source_shape=artifact.source_shape,
            upstream=artifact,
            meta={_PIPELINE_CLASS_META_KEY: pipeline_class_name},
        )

    # ------------------------------------------------------------------
    # Block-running helper
    # ------------------------------------------------------------------

    def _run_blocks(self, blocks: Any, **kwargs: Any) -> PipelineState:
        """Run ``blocks`` over one shared ``PipelineState`` and return it.

        Deliberately not ``_call_block`` (see ``base_driver.py``): that builds a fresh state per call
        (unsuitable for our multi-block outer sequence, which must share state end to end) and runs
        under ``inference_mode`` (the scheduler step mutates ``latents`` in a later call than the one
        that created them, which ``inference_mode`` forbids outside its own scope). Mirrors
        ``minimax_h3.py``'s identically-named helper.
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
        """Return joint video + audio pure-noise latents, paired so a later denoise or resume is reproducible."""
        pipe = cast(LTX2ModularPipeline, self.modular_pipe)
        device, _ = self._get_device_and_type()
        generator = generator_state.to_generator()

        num_frames, height, width = source_shape[-3], source_shape[-2], source_shape[-1]
        latent_height = height // pipe.vae_spatial_compression_ratio
        latent_width = width // pipe.vae_spatial_compression_ratio
        latent_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
        num_channels_latents = pipe.transformer.config.in_channels

        video_shape = (1, num_channels_latents, latent_frames, latent_height, latent_width)
        video_latents = randn_tensor(video_shape, generator=generator, device=device, dtype=torch.float32)

        # Matches `LTX2PrepareAudioLatentsStep`'s own math for the "no audio_latents supplied" branch,
        # so create_noise_latent's audio pairing is reproducible when this same generator is reused.
        frame_rate = 24.0
        duration_s = num_frames / frame_rate
        audio_latents_per_second = (
            pipe.audio_sampling_rate / pipe.audio_hop_length / float(pipe.audio_vae_temporal_compression_ratio)
        )
        audio_num_frames = round(duration_s * audio_latents_per_second)
        num_mel_bins = pipe.audio_vae.config.mel_bins
        latent_mel_bins = num_mel_bins // pipe.audio_vae_mel_compression_ratio
        audio_channels = pipe.audio_vae.config.latent_channels
        audio_shape = (1, audio_channels, audio_num_frames, latent_mel_bins)
        audio_noise = randn_tensor(audio_shape, generator=generator, device=device, dtype=torch.float32)
        audio_latents = _pack_audio_latents(audio_noise)

        return self._make_latent_artifact(
            video_latents,
            source_shape=source_shape,
            meta={
                AUDIO_LATENTS_META_KEY: audio_latents,
                AUDIO_NUM_FRAMES_META_KEY: audio_num_frames,
                AUDIO_PAIRED_WITH_META_KEY: _video_fingerprint(video_latents),
                **GeneratorState.from_generator(generator).as_meta(),
            },
        )

    @override
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        """Encode a single reference image via the modular VAE-encoder block.

        LTX-2.5's only VAE-encode block (`LTX2VaeEncoderStep`) encodes exactly one reference image into
        a single normalized latent frame for image conditioning; there is no general multi-frame video
        encode path (confirmed: no other VAE-encode block exists under
        `diffusers.modular_pipelines.ltx2`), so `VideoMedia` is not supported.
        """
        if isinstance(media, VideoMedia):
            raise NotImplementedError(
                f"{self.driver_namespace} does not support encoding video into a latent. LTX-2.5's modular VAE "
                f"encoder only supports a single reference image."
            )
        if not isinstance(media.image, Image):
            raise TypeError(f"{self.driver_namespace}: Expected a PIL Image, got {type(media.image).__name__}.")

        height, width = media.source_shape[-2], media.source_shape[-1]
        generator = generator_state.to_generator()
        state = self._run_blocks(
            self.modular_pipe.blocks.sub_blocks["vae_encoder"],
            image=media.image,
            height=height,
            width=width,
            generator=generator,
        )
        image_latents = self._get_required(state.values, "image_latents", torch.Tensor)
        return self._make_latent_artifact(image_latents, source_shape=media.source_shape)

    @override
    def add_noise_to_latent(
        self, latent: LatentArtifact, generator_state: GeneratorState, num_inference_steps: int, strength: float
    ) -> LatentArtifact:
        # No image-to-image / video-to-video scale-noise block exists under
        # `diffusers.modular_pipelines.ltx2` (confirmed: no `scale_noise` call and no
        # `Img2Video`/`Image2Video`-style prepare-latents class in `before_denoise.py`).
        raise NotImplementedError(
            f"{self.driver_namespace} does not support adding noise to an existing latent "
            f"(no image-to-image / video-to-video path). Use Create Noise Latents for a fresh latent."
        )

    # ------------------------------------------------------------------
    # Decode
    # ------------------------------------------------------------------

    def _decode_branch(self, pipe: LTX2ModularPipeline, *, base_token_count: int | None) -> Any:
        """Return the decode sub-block tree for the pipe's condition/default branch.

        Both Distilled and Full (SFT) decode through `LTX25AutoDecoderStep` (the diffusion decoder,
        `LTX25AutoBlocks`'s own default — `standard_parameters` doesn't swap it), which selects on the
        `base_token_count` trigger with `block_names = ["condition", "default"]`.
        """
        decode_top = pipe.blocks.sub_blocks["decode"]
        branch_name = "condition" if base_token_count is not None else "default"
        return decode_top.sub_blocks[branch_name]

    @override
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        pipe = cast(LTX2ModularPipeline, self.modular_pipe)
        device, dtype = self._get_device_and_type()

        self.last_audio = None
        self.last_sampling_rate = None

        if self._latent_was_produced_for_hdr(latent):
            return self._decode_hdr(latent, pipe, device, dtype)

        num_frames, height, width = latent.source_shape[-3], latent.source_shape[-2], latent.source_shape[-1]
        video_latents = latent.to_torch(device=device, dtype=torch.float32)
        packed = _pack_latents(video_latents, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size)

        base_token_count = read_driver_meta(latent, BASE_TOKEN_COUNT_META_KEY, self.driver_namespace)
        branch = self._decode_branch(pipe, base_token_count=base_token_count)

        if base_token_count is not None:
            trim_state = self._run_blocks(
                branch.sub_blocks["trim_condition_tokens"], latents=packed, base_token_count=base_token_count
            )
            packed = self._get_required(trim_state.values, "latents", torch.Tensor)

        video_state = self._run_blocks(
            branch.sub_blocks["video_decode"],
            latents=packed,
            height=height,
            width=width,
            num_frames=num_frames,
            dtype=dtype,
            output_type="pil",
        )
        video_frames = self._get_required(video_state.values, "videos", list)[0]

        audio_latents = read_driver_meta(latent, AUDIO_LATENTS_META_KEY, self.driver_namespace)
        audio_num_frames = read_driver_meta(latent, AUDIO_NUM_FRAMES_META_KEY, self.driver_namespace)
        if audio_latents is None or audio_num_frames is None:
            # Normal for a latent rebuilt without meta (e.g. Empty Latent, save/load, latent math) —
            # mirrors `minimax_h3.py`'s identical accommodation. Debug rather than warning since this
            # is also the shape of a mid-loop preview decode were one ever routed here.
            logger.debug(
                "%s: decoding video only because the latent carries no audio latents in driver meta.",
                self.driver_namespace,
            )
            return video_frames

        paired_with = read_driver_meta(latent, AUDIO_PAIRED_WITH_META_KEY, self.driver_namespace)
        if not _fingerprints_match(paired_with, _video_fingerprint(video_latents)):
            raise ValueError(
                f"{self.driver_namespace}: Attempted to decode an LTX-2.5 latent. Failed because its audio "
                f"latent belongs to a different video latent, so the soundtrack would not match the picture. "
                f"LTX-2.5 generates video and audio jointly and the audio travels in the latent's metadata, "
                f"which latent math, composite and upsampler nodes do not recompute. Connect Generate Media "
                f"Latents directly to Decode Media Latent."
            )

        audio_state = self._run_blocks(
            branch.sub_blocks["audio_decode"],
            audio_latents=audio_latents.to(device=device, dtype=torch.float32),
            audio_num_frames=audio_num_frames,
            output_type="pil",
        )
        self.last_audio = self._get_required(audio_state.values, "audio", torch.Tensor)
        self.last_sampling_rate = pipe.vocoder.config.output_sampling_rate
        return video_frames

    def _decode_hdr(
        self, latent: LatentArtifact, pipe: LTX2ModularPipeline, device: torch.device, dtype: torch.dtype
    ) -> np.ndarray:
        """HDR decode: no audio, and postprocesses through `LTX2VideoHDRProcessor` instead of the
        standard video processor. Mirrors `ltx2.py`'s `decode_latent`/`_decode_hdr_to_linear_np` exactly,
        reading `vae`/`diffusion_decoder` off the modular pipe instead of a fixed pipe.
        """
        from diffusers.pipelines.ltx2.image_processor import LTX2VideoHDRProcessor  # noqa: PLC0415

        video_latents = latent.to_torch(device=device, dtype=torch.float32)
        diffusion_decoder = getattr(pipe, "diffusion_decoder", None)

        if pipe.vae.config.timestep_conditioning:
            timestep = torch.zeros(video_latents.shape[0], device=device, dtype=dtype)
        else:
            timestep = None

        latents = _denormalize_public_latents(pipe, video_latents).to(device=device, dtype=dtype)
        with torch.no_grad():
            if diffusion_decoder is not None:
                generator_state = GeneratorState.from_artifact(latent)
                generator = generator_state.to_generator() if generator_state is not None else None
                video = diffusion_decoder.decode(latents, generator=generator, return_dict=False)[0]
            else:
                video = pipe.vae.decode(latents, timestep, return_dict=False)[0]

        vae_spatial_ratio = pipe.vae_spatial_compression_ratio
        hdr_processor = LTX2VideoHDRProcessor(vae_scale_factor=vae_spatial_ratio, hdr_transform="logc3")
        return hdr_processor.postprocess_hdr_video(video, output_type="np")

    # ------------------------------------------------------------------
    # Denoise
    # ------------------------------------------------------------------

    @staticmethod
    def _update_args_for_distilled_pipeline(original_kwargs: dict[str, Any]) -> dict[str, Any]:
        kwargs = original_kwargs.copy()
        if "use_stage_2" in kwargs:
            use_stage_2 = kwargs.pop("use_stage_2")
            if not use_stage_2:
                kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
            else:
                kwargs["sigmas"] = STAGE_2_DISTILLED_SIGMA_VALUES
                kwargs["noise_scale"] = STAGE_2_DISTILLED_SIGMA_VALUES[0]
            kwargs["num_inference_steps"] = len(kwargs["sigmas"])
        return kwargs

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

        kwargs = self._update_args_for_distilled_pipeline(kwargs)
        num_inference_steps = kwargs.pop("num_inference_steps", num_inference_steps)
        kwargs.update(
            num_inference_steps=num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
        )

        if self.is_hdr_lora_active:
            return self._denoise_with_hdr_lora(latent, **kwargs)

        return self._denoise_modular(latent, **kwargs)

    # -- Modular (native) denoise path: text-to-video, condition, in-context -----------------------

    def _select_workflow(self, kwargs: dict[str, Any]) -> str:
        """Mirrors `LTX2AutoCoreDenoiseStep.select_block`'s exact trigger precedence."""
        if kwargs.get("reference_conditions"):
            return "in_context"
        if kwargs.get("conditions"):
            return "condition"
        return "text2video"

    def _configure_guidance(self, pipe: LTX2ModularPipeline, kwargs: dict[str, Any]) -> None:
        """Translate the runtime UI's scalar guidance kwargs into `LTX2Guidance` guider components.

        Replaces the fixed-pipeline driver's scalar-kwarg forwarding: the modular denoise blocks read
        guidance from the `guider`/`audio_guider` components, not from per-call kwargs.
        """
        guidance_scale = kwargs.pop("guidance_scale", 3.0)
        stg_scale = kwargs.pop("stg_scale", 1.0)
        modality_scale = kwargs.pop("modality_scale", 3.0)
        guidance_rescale = kwargs.pop("guidance_rescale", 0.7)
        audio_guidance_scale = kwargs.pop("audio_guidance_scale", 7.0)
        audio_stg_scale = kwargs.pop("audio_stg_scale", 1.0)
        audio_modality_scale = kwargs.pop("audio_modality_scale", 3.0)
        audio_guidance_rescale = kwargs.pop("audio_guidance_rescale", 0.7)
        pipe.update_components(
            guider=LTX2Guidance(
                guidance_scale=guidance_scale,
                stg_scale=stg_scale,
                modality_scale=modality_scale,
                guidance_rescale=guidance_rescale,
                # Matches `LTX2LoopDenoiser.expected_components`' shipped default for `guider`.
                spatio_temporal_guidance_blocks=[28],
            ),
            audio_guider=LTX2Guidance(
                guidance_scale=audio_guidance_scale,
                stg_scale=audio_stg_scale,
                modality_scale=audio_modality_scale,
                guidance_rescale=audio_guidance_rescale,
            ),
        )

    def _build_video_conditions_from_payloads(
        self,
        payloads: list[MediaGenConditioningPayload] | None,
        pixel_num_frames: int,
        temporal_ratio: int,
    ) -> list[LTX2VideoCondition]:
        """Build `LTX2VideoCondition` objects from conditioning payloads. Mirrors `ltx2.py`'s
        identically-named method (same dataclass, same driver-agnostic frame-index math).
        """
        conditions: list[LTX2VideoCondition] = []
        if payloads is None:
            return conditions
        for payload in payloads:
            if payload.mode is ConditioningMode.VIDEO:
                entry = payload.entries[0]
                frames = resolve_conditioning_video(entry.artifact)
                pixel_frame_index = resolve_frame_index(entry.frame_index, pixel_num_frames)
                index = pixel_frame_index_to_latent_index(pixel_frame_index, temporal_ratio, pixel_num_frames)
                conditions.append(LTX2VideoCondition(frames=frames, index=index, strength=entry.strength))
            elif payload.mode is ConditioningMode.IMAGE:
                if not payload.entries:
                    msg = "Failed to build LTX-2.5 video conditioning because the images list is empty."
                    raise ValueError(msg)
                for entry in payload.entries:
                    pixel_frame_index = resolve_frame_index(entry.frame_index, pixel_num_frames)
                    latent_index = pixel_frame_index_to_latent_index(
                        pixel_frame_index, temporal_ratio, pixel_num_frames
                    )
                    image = resolve_conditioning_image(entry.artifact)
                    conditions.append(LTX2VideoCondition(frames=image, index=latent_index, strength=entry.strength))
            else:
                msg = f"Failed to build LTX-2.5 video conditioning because mode '{payload.mode.value}' is unsupported."
                raise ValueError(msg)
        return conditions

    def _build_ic_reference_conditions(
        self, payloads: list[MediaGenConditioningPayload] | None
    ) -> list[LTX2ReferenceCondition]:
        """Mirrors `ltx2.py`'s identically-named method (same dataclass)."""
        if payloads is None:
            return []
        reference_conditions: list[LTX2ReferenceCondition] = []
        for payload in payloads:
            if payload.mode is not ConditioningMode.VIDEO:
                msg = (
                    f"Failed to build LTX-2.5 IC-LoRA conditioning because mode '{payload.mode.value}' is unsupported."
                )
                raise ValueError(msg)
            entry = payload.entries[0]
            frames = resolve_conditioning_video(entry.artifact)
            reference_conditions.append(LTX2ReferenceCondition(frames=frames, strength=entry.strength))
        return reference_conditions

    def _denoise_modular(
        self,
        latent: LatentArtifact,
        *,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        if return_fully_denoised:
            # Same limitation as MiniMax-H3: a truncated window's scheduler can't express jumping to the
            # terminal sigma from a non-contiguous schedule.
            raise NotImplementedError(f"{self.driver_namespace} does not support 'return_fully_denoised'.")

        pipe = cast(LTX2ModularPipeline, self.modular_pipe)
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape
        num_frames, height, width = source_shape[-3], source_shape[-2], source_shape[-1]

        self._configure_guidance(pipe, kwargs)

        kwargs.pop("text_embeddings_path", None)
        kwargs["num_frames"] = num_frames
        kwargs["height"] = height
        kwargs["width"] = width
        kwargs.setdefault("frame_rate", 24.0)
        kwargs.setdefault("batch_size", 1)
        kwargs.setdefault("dtype", dtype)

        ic_lora_payloads = normalize_to_payloads(kwargs.pop(self._IC_LORA_REFERENCE_KEY, None))
        reference_conditions = self._build_ic_reference_conditions(ic_lora_payloads)
        if reference_conditions:
            kwargs["reference_conditions"] = reference_conditions
            if "reference_downscale_factor" not in kwargs:
                inferred = self._infer_reference_downscale_factor_from_loras()
                if inferred is not None:
                    logger.info(
                        "LTX25: reference_downscale_factor not provided; using value %d from LoRA safetensors metadata.",
                        inferred,
                    )
                    kwargs["reference_downscale_factor"] = inferred

        media_gen_conditioning_payloads = normalize_to_payloads(kwargs.pop(MediaGenConditioningKey.OUTPUT, None))
        if media_gen_conditioning_payloads:
            conditions = self._build_video_conditions_from_payloads(
                media_gen_conditioning_payloads, num_frames, pipe.vae_temporal_compression_ratio
            )
            if conditions:
                kwargs["conditions"] = conditions

        workflow = self._select_workflow(kwargs)
        core_cls, denoise_cls = _WORKFLOW_TABLE[workflow]

        generator = kwargs.pop("generator", generator_state.to_generator())
        video_latents_in = latent.to_torch(device=device, dtype=torch.float32)
        kwargs["latents"] = _pack_latents(
            video_latents_in, pipe.transformer_spatial_patch_size, pipe.transformer_temporal_patch_size
        )

        audio_latents_in = read_driver_meta(latent, AUDIO_LATENTS_META_KEY, self.driver_namespace)
        if audio_latents_in is not None:
            paired_with = read_driver_meta(latent, AUDIO_PAIRED_WITH_META_KEY, self.driver_namespace)
            if not _fingerprints_match(paired_with, _video_fingerprint(video_latents_in)):
                raise ValueError(
                    f"{self.driver_namespace}: Attempted to denoise an LTX-2.5 latent. Failed because its audio "
                    f"latent belongs to a different video latent. Connect Generate Media Latents directly to a "
                    f"previous Generate Media Latents run rather than rebuilding the latent in between."
                )
            kwargs["audio_latents"] = audio_latents_in.to(device=device, dtype=torch.float32)
        elif start_step > 0:
            raise ValueError(
                f"{self.driver_namespace}: Attempted to resume an LTX-2.5 denoise at step {start_step}. Failed "
                f"because the input latent carries no audio latent, so the soundtrack would restart from pure "
                f"noise mid-schedule. Chain partial denoise directly from a previous Generate Media Latents run."
            )

        inner_blocks = dict(core_cls().sub_blocks)
        denoise_step = denoise_cls()
        denoise_step.callback = callback
        denoise_step.start_step = start_step
        denoise_step.end_step = end_step
        inner_blocks["denoise"] = denoise_step
        denoise_bundle = SequentialPipelineBlocks.from_blocks_dict(inner_blocks)

        outer_blocks: dict[str, Any] = {"text_encoder": pipe.blocks.sub_blocks["text_encoder"]}
        if workflow in ("condition", "in_context"):
            outer_blocks["condition_encoder"] = pipe.blocks.sub_blocks["condition_encoder"]
        if workflow == "in_context":
            outer_blocks["reference_encoder"] = pipe.blocks.sub_blocks["reference_encoder"]
        outer_blocks["denoise"] = denoise_bundle
        blocks = SequentialPipelineBlocks.from_blocks_dict(outer_blocks)

        # Cancellation is signalled by setting `_interrupt` on the pipe, but `ModularPipeline` has no
        # such attribute by default, and the pipe is cached across runs, so a leaked True would break
        # the next run at step 0.
        pipe._interrupt = False  # type: ignore[attr-defined]
        try:
            state = self._run_blocks(blocks, num_inference_steps=num_inference_steps, generator=generator, **kwargs)
        finally:
            pipe._interrupt = False  # type: ignore[attr-defined]

        packed_out = self._get_required(state.values, "latents", torch.Tensor)
        audio_out = self._get_required(state.values, "audio_latents", torch.Tensor)
        audio_num_frames_out = state.values.get("audio_num_frames")
        base_token_count = state.values.get("base_token_count")

        latent_num_frames = (num_frames - 1) // pipe.vae_temporal_compression_ratio + 1
        latent_height = height // pipe.vae_spatial_compression_ratio
        latent_width = width // pipe.vae_spatial_compression_ratio
        rows = packed_out[:, :base_token_count] if base_token_count is not None else packed_out
        video_out = _unpack_latents(
            rows,
            latent_num_frames,
            latent_height,
            latent_width,
            pipe.transformer_spatial_patch_size,
            pipe.transformer_temporal_patch_size,
        )

        meta: dict[str, Any] = {
            AUDIO_LATENTS_META_KEY: audio_out,
            AUDIO_NUM_FRAMES_META_KEY: audio_num_frames_out,
            AUDIO_PAIRED_WITH_META_KEY: _video_fingerprint(video_out),
            **GeneratorState.from_generator(generator).as_meta(),
        }
        if base_token_count is not None:
            meta[BASE_TOKEN_COUNT_META_KEY] = base_token_count

        return self._make_latent_artifact(video_out, source_shape=source_shape, upstream=latent, meta=meta)

    # -- HDR (fixed pipeline) denoise path -----------------------------------------------------------

    def _get_or_build_hdr_pipeline(self) -> LTX2HDRPipeline:
        """Hand-build `LTX2HDRPipeline` from components already loaded on the modular pipe.

        Every component `LTX2HDRPipeline.__init__` needs (`scheduler`, `vae`, `audio_vae`,
        `text_encoder`, `tokenizer`, `connectors`, `transformer`, `vocoder`) is already a genuine
        component of `LTX25ModularPipeline` — shared by reference, no fresh loading. Because LoRA
        adapters are activated in place on `self.modular_pipe.transformer`/`.connectors`, the HDR
        adapter is visible the moment it's active, with no weight duplication. `ModularPipeline` has no
        `.from_pipe()` (unlike `DiffusionPipeline`), which is why this must be built by hand rather
        than via `create_pipe_variant`. Cached: constructing it is cheap (no new tensors), but
        re-registering modules on every call is unnecessary churn. Deliberately never calls
        `enable_model_cpu_offload`/`enable_sequential_cpu_offload` on the result — its components are
        the exact objects the `ComponentsManager` already manages placement for; a second `accelerate`
        hook chain on top would conflict.
        """
        if self._hdr_pipe is None:
            pipe = self.modular_pipe
            self._hdr_pipe = LTX2HDRPipeline(
                scheduler=pipe.scheduler,
                vae=pipe.vae,
                audio_vae=pipe.audio_vae,
                text_encoder=pipe.text_encoder,
                tokenizer=pipe.tokenizer,
                connectors=pipe.connectors,
                transformer=pipe.transformer,
                vocoder=pipe.vocoder,
            )
        return self._hdr_pipe

    @staticmethod
    def _set_default_kwargs_hdr(original_kwargs: dict[str, Any]) -> dict[str, Any]:
        """Defaults for the `LTX2HDRPipeline` code path. Mirrors `ltx2.py`'s identically-named method:
        the HDR pipeline has no audio inputs and its example uses `guidance_scale=1.0`/`stg_scale=0.0`.
        """
        kwargs = {k: v for k, v in original_kwargs.items() if not k.startswith("audio_")}
        kwargs.setdefault("guidance_scale", 1.0)
        kwargs.setdefault("stg_scale", 0.0)
        kwargs.setdefault("spatio_temporal_guidance_blocks", [28])
        kwargs.setdefault("modality_scale", 1.0)
        kwargs.setdefault("guidance_rescale", 0.0)
        return kwargs

    def _load_hdr_text_embeddings(self, text_embeddings_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        canonical_path = canonicalize_for_io(text_embeddings_path)
        if not canonical_path.exists():
            raise ValueError(
                "Attempted to load HDR text embeddings. Failed because the embedding file "
                f"'{canonical_path}' does not exist."
            )

        device, _ = self._get_device_and_type()
        try:
            with safe_open(canonical_path, framework="pt", device=str(device)) as embedding_file:
                video_context = embedding_file.get_tensor("video_context")
                audio_context = embedding_file.get_tensor("audio_context")
        except KeyError as err:
            raise ValueError(
                "Attempted to load HDR text embeddings. Failed because the safetensors file "
                f"'{canonical_path}' is missing required tensor '{err.args[0]}'. Expected 'video_context' and 'audio_context'."
            ) from err

        return video_context.to(device=device), audio_context.to(device=device)

    def _build_hdr_reference_conditions(
        self,
        media_gen_conditioning_payloads: list[MediaGenConditioningPayload] | None,
        target_height: int,
        target_width: int,
    ) -> list[LTX2HDRReferenceCondition]:
        if not media_gen_conditioning_payloads:
            msg = "Failed to build LTX-2.5 HDR conditioning because no conditioning was provided."
            raise ValueError(msg)

        reference_conditions: list[LTX2HDRReferenceCondition] = []
        for payload in media_gen_conditioning_payloads:
            if payload.mode is not ConditioningMode.VIDEO:
                msg = f"Failed to build LTX-2.5 HDR conditioning because mode '{payload.mode.value}' is unsupported."
                raise ValueError(msg)
            entry = payload.entries[0]
            frames = resolve_conditioning_video(entry.artifact)
            frames = resize_frames_scale_to_fill(frames, target_height, target_width)
            reference_conditions.append(LTX2HDRReferenceCondition(frames=frames, strength=entry.strength))
        return reference_conditions

    def _denoise_with_hdr_lora(
        self,
        latent: LatentArtifact,
        *,
        num_inference_steps: int,
        generator_state: GeneratorState,
        callback: Any = None,
        start_step: int = 0,
        end_step: int = -1,
        return_fully_denoised: bool = False,
        **kwargs: Any,
    ) -> LatentArtifact:
        logger.info("LTX25: HDR IC-LoRA path active — denoising with a hand-built LTX2HDRPipeline.")
        media_gen_conditioning_payloads = normalize_to_payloads(kwargs.pop(self._IC_LORA_REFERENCE_KEY, None))
        kwargs.pop(MediaGenConditioningKey.OUTPUT, None)
        text_embeddings_path = kwargs.pop("text_embeddings_path", "")
        target_height = latent.source_shape[-2]
        target_width = latent.source_shape[-1]
        reference_conditions = self._build_hdr_reference_conditions(
            media_gen_conditioning_payloads, target_height, target_width
        )
        kwargs["num_frames"] = latent.source_shape[-3]
        kwargs = self._set_default_kwargs_hdr(kwargs)

        if text_embeddings_path:
            logger.info("LTX25: loading HDR text embeddings from '%s' for conditioning.", text_embeddings_path)
            video_context, audio_context = self._load_hdr_text_embeddings(str(text_embeddings_path))
            kwargs["prompt"] = None
            kwargs["negative_prompt"] = None
            kwargs["connector_video_embeds"] = video_context
            kwargs["connector_audio_embeds"] = audio_context

        kwargs["reference_conditions"] = reference_conditions
        kwargs.update(
            num_inference_steps=num_inference_steps,
            generator_state=generator_state,
            callback=callback,
            start_step=start_step,
            end_step=end_step,
            return_fully_denoised=return_fully_denoised,
        )

        hdr_pipe = self._get_or_build_hdr_pipeline()
        original_pipe = self._pipe
        try:
            self._pipe = hdr_pipe
            result = super().denoise_latent(latent, **kwargs)
        finally:
            self._pipe = original_pipe
        return self._stamp_pipeline_class(result, LTX2HDRPipeline.__name__)

    # ------------------------------------------------------------------
    # Fixed-pipeline fallback plumbing (used only by the HDR path via super().denoise_latent())
    # ------------------------------------------------------------------

    @override
    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        # Only reached from the HDR path (self.pipe temporarily swapped to the hand-built
        # LTX2HDRPipeline). LTX2HDRPipeline's `latents` kwarg expects raw VAE space, matching
        # `ltx2.py`'s identical convention for the fixed LTX2 pipelines.
        device, _ = self._get_device_and_type()
        latents = latents.to(device=device, dtype=torch.float32)
        return _denormalize_public_latents(cast(LTX2HDRPipeline, self.pipe), latents)

    @override
    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        device, _ = self._get_device_and_type()
        latents_from_pipe = latents_from_pipe.to(device=device, dtype=torch.float32)
        pipe = cast(LTX2HDRPipeline, self.pipe)
        return _normalize_public_latents(pipe, latents_from_pipe)

    @override
    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """LTX2HDRPipeline returns video frames under ``.frames`` instead of ``.images``."""
        return pipe_output.frames


def _denormalize_public_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    """Inverse of `_normalize_public_latents`. Used only on the HDR fixed-pipeline path, where
    `LTX2HDRPipeline.__call__`'s `latents` kwarg expects raw (un-whitened) VAE space.
    """
    latents_mean = pipe.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = pipe.vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    scaling_factor = pipe.vae.config.scaling_factor
    return latents * latents_std / scaling_factor + latents_mean


def _normalize_public_latents(pipe: Any, latents: torch.Tensor) -> torch.Tensor:
    """Whitens raw VAE-space latents into this driver's public (~N(0,1)) space. Used only on the HDR
    fixed-pipeline path; the native modular path never leaves the whitened space (see module docstring).
    """
    latents_mean = pipe.vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents_std = pipe.vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    scaling_factor = pipe.vae.config.scaling_factor
    return (latents - latents_mean) * scaling_factor / latents_std
