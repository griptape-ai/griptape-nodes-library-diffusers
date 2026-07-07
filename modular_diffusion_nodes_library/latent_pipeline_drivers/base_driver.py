from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import (
    ControlNetDiffusionPipelineArtifact,
    DiffusionPipelineArtifact,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers._base_driver_forwardable_signature import (
    FORWARDABLE_METHODS,
    validate_forwardable_signature,
)
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import (
    META_DRIVER_KEY,
    DecodeResult,
    GeneratorState,
    ImageMedia,
    MaskMedia,
    TextEncodings,
    VideoMedia,
)
from modular_diffusion_nodes_library.misc.partial_denoise import (
    PartialDenoisePipelineRunner,
    PartialDenoiseSchedulerProxy,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import create_pipe_variant

_T = TypeVar("_T")


class LatentPipelineDriver(ABC):
    """Abstract base class for latent pipeline drivers.

    Latent shape & space contract
    -----------------------------
    All latents flowing across the public driver surface
    (``create_noise_latent``, ``encode_media``, ``add_noise_to_latent``, the
    output of ``denoise_latent``, and the input of ``decode_latent``) share a
    single canonical shape **and** statistical space so they can be freely
    composited, masked, added, or otherwise manipulated by downstream nodes.

    Public latents are exchanged as :class:`LatentArtifact` instances. The
    artifact carries:

    - The unpacked tensor (4-D image or 5-D video, see below).
    - ``source_shape`` — the pixel-space shape of the original media before
      VAE encoding. Drivers translate between pixel-space dimensions and the
      divided latent-space tensor dimensions internally.
    - ``meta`` — free-form upstream entries (e.g. user-set provenance) plus a
      driver-namespaced sub-bag at ``meta[<driver_namespace>]`` for
      driver-internal invariants. See :meth:`_make_latent_artifact`.

    Shape
    ^^^^^
    Public latent tensors are **unpacked** (no model-specific patchifying /
    sequence packing). Their shape is:

    - 4-D for image pipelines: ``[B, C, H // vae_scale_factor, W // vae_scale_factor]``
    - 5-D for video pipelines: ``[B, C, T_latent, H // vae_scale_factor, W // vae_scale_factor]``

    where ``C`` (``num_channels_latents``) is inferred from the pipeline and
    ``T_latent = (T_video - 1) // vae_scale_factor_temporal + 1`` for video.

    The ``source_shape`` on the artifact is in **pixel space**, **not**
    latent space:

    - 4-D for image: ``[B, C_image, H_pixel, W_pixel]``
    - 5-D for video: ``[B, C_image, T_video, H_pixel, W_pixel]``

    Space
    ^^^^^
    Latent tensors are **normalised**: each channel is ~N(0, 1), matching the
    distribution of ``torch.randn``. For VAEs whose raw latents are not
    unit-variance (e.g. WAN, Flux2), drivers must apply the per-channel
    whitening ``(z - latents_mean) / latents_std`` inside ``encode_media``
    and the inverse inside ``decode_latent``.

    Any model-specific *packing* (e.g. Flux, Qwen) is applied transiently
    inside ``prepare_input_latent`` / ``prepare_output_latent`` and never
    appears on the public surface.
    """

    produces_video: ClassVar[bool] = False
    video_fps: ClassVar[int] = 16
    _partial_denoise_proxy_class: ClassVar[type[PartialDenoiseSchedulerProxy]] = PartialDenoiseSchedulerProxy
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = None
    _inpaint_controlnet_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = None
    _DEFAULT_NUM_INFERENCE_STEPS: ClassVar[int] = 20

    # ------------------------------------------------------------------
    # Public driver surface — cross-driver signature contract
    # ------------------------------------------------------------------
    # The methods listed in ``FORWARDABLE_METHODS`` must share an identical
    # positional signature across all drivers so callers can invoke them
    # uniformly; any driver-specific tunable beyond the contract must be
    # keyword-only. Enforced in ``__init_subclass__``.

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        for method_name in FORWARDABLE_METHODS:
            method = cls.__dict__.get(method_name)
            if method is None:
                continue
            validate_forwardable_signature(cls.__name__, method_name, method)

    def __init__(self, pipe: DiffusionPipeline):
        self._pipe = pipe
        self._modular_pipe: ModularPipeline | None = None

    @property
    def pipe(self) -> DiffusionPipeline:
        return self._pipe

    @property
    def driver_namespace(self) -> str:
        """Stable per-driver identifier used as the key into ``LatentArtifact.meta``."""
        return type(self).__name__

    @property
    def modular_pipe(self) -> ModularPipeline:
        if self._modular_pipe is None:
            modular_pipe = self._create_modular_pipe()
            components_dct = {k: v for k, v in self.pipe.components.items() if v is not None}
            modular_pipe.update_components(**components_dct)
            self._modular_pipe = modular_pipe
        return self._modular_pipe

    @abstractmethod
    def _create_modular_pipe(self) -> ModularPipeline:
        """Create and return a ModularPipeline for this driver.

        Subclasses must override this method.
        """
        ...

    @torch.inference_mode()
    def _call_block(self, block: ModularPipelineBlocks, **kwargs: Any) -> dict[str, Any]:
        state = PipelineState()
        for param in block.inputs:
            if param.name in kwargs:
                state.set(param.name, kwargs[param.name], param.kwargs_type)
        _, state = block(self.modular_pipe, state)  # type: ignore[reportOperatorIssue]
        return state.values

    @staticmethod
    def _get_required(state: dict[str, Any], key: str, type_: type[_T]) -> _T:
        value = state.get(key)
        if not isinstance(value, type_):
            raise ValueError(f"Expected {type_.__name__} for state key '{key}', got {type(value).__name__}.")
        return value

    @staticmethod
    def _get_torch_type(pipe: ModularPipeline | DiffusionPipeline) -> torch.dtype | None:
        for module_name in ["transformer", "unet", "vae", "text_encoder"]:
            module = getattr(pipe, module_name, None)
            if module is None:
                continue
            module_dtype = getattr(module, "dtype", None)
            if module_dtype is not None:
                return module_dtype
        return None

    @classmethod
    @abstractmethod
    def can_make_control_pipe_from_standard(cls, control_net_model_lists: list[str] | str | None) -> bool:
        """Return true if this driver can create a control net pipe from the list of models."""
        ...

    @classmethod
    def control_pipe_from_standard(
        cls, pipe: ModularPipeline | DiffusionPipeline, control_net_model_lists: list[str] | str | None
    ) -> DiffusionPipeline | ModularPipeline:
        """Given a standard pipeline, return a version of the pipeline that can be used for control net generation."""
        raise NotImplementedError("Subclasses should implement this method")

    def _get_device_and_type(self) -> tuple[torch.device, torch.dtype]:
        device = getattr(self.pipe, "_execution_device", None) or self.pipe.device
        dtype = getattr(self.pipe.vae, "dtype", None) or torch.float32
        return device, dtype

    def _make_latent_artifact(
        self,
        tensor: torch.Tensor,
        *,
        source_shape: tuple[int, ...],
        upstream: LatentArtifact | InpaintMaskArtifact | None = None,
        meta: dict[str, Any] | None = None,
    ) -> LatentArtifact:
        """Build a :class:`LatentArtifact` carrying this driver's namespaced meta.

        ``upstream`` is the input artifact this call was derived from, if any.
        Its non-namespace meta (e.g. user-set provenance) is preserved, and any
        same-driver namespaced meta is merged with ``meta`` (new values win).
        """

        base: dict[str, Any] = {}
        if upstream is not None:
            base = upstream.metadata

        upstream_driver_meta: dict[str, Any] = {}
        if base.get(META_DRIVER_KEY) == self.driver_namespace:
            existing = base.get(self.driver_namespace)
            if isinstance(existing, dict):
                upstream_driver_meta = dict(existing)

        namespaced_meta = {**upstream_driver_meta, **(meta or {})}
        base[META_DRIVER_KEY] = self.driver_namespace
        base[self.driver_namespace] = namespaced_meta

        return LatentArtifact.from_torch(tensor, source_shape=source_shape, meta=base)

    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        """Return latents ready to be passed into the pipeline, which may involve packing or other preprocessing."""
        return latents

    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Return latents ready to be process further e.g. with mask, which may involve unpacking or other postprocessing."""
        return latents_from_pipe

    @abstractmethod
    def create_noise_latent(self, source_shape: tuple[int, ...], generator_state: GeneratorState) -> LatentArtifact:
        """Return pure noise latent. See class docstring for the latent shape contract."""
        ...

    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """Extract the raw latent tensor from a pipeline output object.
        Image pipelines expose the result as `images`
        override for video pipelines with `frames`.
        """
        return pipe_output.images

    @abstractmethod
    def decode_latent(self, latent: LatentArtifact) -> DecodeResult:
        """Return decoded image. See class docstring for the input latent shape contract."""
        ...

    @abstractmethod
    def encode_media(self, media: ImageMedia | VideoMedia, generator_state: GeneratorState) -> LatentArtifact:
        """Return encoded latent for an :class:`ImageMedia` or :class:`VideoMedia` input.

        Image-only drivers should raise :class:`NotImplementedError` for
        :class:`VideoMedia` and vice versa. See class docstring for the output
        latent shape contract.
        """
        ...

    def encode_masked_image(
        self, image: ImageMedia, mask: MaskMedia, generator_state: GeneratorState
    ) -> LatentArtifact:
        """Encode the source image with the masked region zeroed out."""
        image_processor = self.modular_pipe.image_processor
        if isinstance(image.image, torch.Tensor):
            height = image.image.shape[-2]
            width = image.image.shape[-1]
        else:
            height = image.image.height
            width = image.image.width
        source_t = image_processor.preprocess(image.image, height=height, width=width)
        mask_t = torch.from_numpy(np.array(mask.mask, dtype="float32") / 255.0)[None, None]
        masked_t = source_t * (mask_t < 0.5)
        return self.encode_media(ImageMedia(image=masked_t, source_shape=image.source_shape), generator_state)

    @abstractmethod
    def add_noise_to_latent(
        self,
        latent: LatentArtifact,
        generator_state: GeneratorState,
        num_inference_steps: int,
        strength: float,
    ) -> LatentArtifact:
        """Return noised latent. See class docstring for the latent shape contract."""
        ...

    def encode_prompt(self, prompt: str, negative_prompt: str, **kwargs: Any) -> TextEncodings:  # noqa: ARG002
        """Encode prompt text into embeddings via the modular pipeline's ``text_encoder`` sub-block.

        Works for any driver whose modular pipeline exposes a ``text_encoder`` sub-block that
        accepts a ``prompt`` input (and optionally ``negative_prompt``). Override per model when
        extra inputs are needed.
        """
        text_encoder_pipe = self.modular_pipe.blocks.sub_blocks["text_encoder"]
        call_kwargs: dict[str, Any] = {"prompt": prompt}
        if negative_prompt:
            call_kwargs["negative_prompt"] = negative_prompt
        return self._call_block(text_encoder_pipe, **call_kwargs)

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
        """Run the denoising loop. See class docstring for the latent shape contract.

        ``latent`` is either a :class:`LatentArtifact` (standard denoise) or an
        :class:`InpaintMaskArtifact` (inpaint denoise).

        Subclasses can override behaviour by passing extra ``kwargs`` when
        re-implementing this method. E.g. ``height`` / ``width`` default to
        ``latent.source_shape`` but can be overridden by setting kwargs; useful when the
        dimensions differ from the source (e.g. pipeline expects specific resolution).
        """
        if isinstance(self.pipe, ModularPipeline):
            raise NotImplementedError(
                "denoise_latent is not implemented for ModularPipelines. Subclasses should implement this method for modular pipeline."
            )

        pipe = self.pipe
        device, dtype = self._get_device_and_type()
        source_shape = latent.source_shape

        if isinstance(latent, InpaintMaskArtifact):
            pipe = self._get_inpaint_pipe()
            if pipe is None:
                raise NotImplementedError(f"Inpainting is not supported by {self.driver_namespace}. ")

            inpaint_kwargs = self._get_inpaint_kwargs(latent)
            kwargs.update(inpaint_kwargs)
        elif "latents" not in kwargs:
            latents = latent.to_torch(device=device, dtype=dtype)
            latents = self.prepare_input_latent(latents, source_shape)
            kwargs["latents"] = latents

        kwargs.setdefault("height", source_shape[-2])
        kwargs.setdefault("width", source_shape[-1])
        generator = kwargs.pop("generator", generator_state.to_generator())

        pipe_kwargs: dict[str, Any] = {
            **kwargs,
            "output_type": "latent",
            "num_inference_steps": num_inference_steps,
            "callback_on_step_end": callback,
            "generator": generator,
        }

        is_partial_denoise = start_step > 0 or end_step >= 0
        if is_partial_denoise:
            if start_step < 0:
                raise ValueError("start_step must be a non-negative integer.")
            if start_step >= end_step and end_step != -1:
                raise ValueError("start_step must be less than end_step.")

            denoise_begin = start_step / num_inference_steps
            denoise_end = end_step / num_inference_steps if end_step != -1 else 1.0
            partial_denoise_pipe = PartialDenoisePipelineRunner(pipe, proxy_class=self._partial_denoise_proxy_class)
            pipe_output = partial_denoise_pipe(  # type: ignore[reportCallIssue]
                denoise_begin,
                denoise_end,
                return_fully_denoised,
                **pipe_kwargs,
            )
        else:
            pipe_output = pipe(**pipe_kwargs)  # type: ignore[reportCallIssue]

        output_tensor = self.prepare_output_latent(self._extract_latents_from_output(pipe_output), source_shape)
        return self._make_latent_artifact(
            output_tensor,
            source_shape=source_shape,
            upstream=latent,
            meta=GeneratorState.from_generator(generator).as_meta(),
        )

    # ------------------------------------------------------------------
    # Inpainting hooks
    # ------------------------------------------------------------------

    @classmethod
    def validate_run_configuration(
        cls,
        pipeline_artifact: DiffusionPipelineArtifact,
        input_latent: LatentArtifact | None,
    ) -> list[Exception] | None:
        """Validate that the requested run configuration is supported by this driver.

        Called from the Generate Latent node before execution.

        Default implementation surfaces driver-agnostic compatibility problems:
        when ``input_latent`` is an :class:`InpaintMaskArtifact`, the
        driver must declare an inpaint pipeline class for the current variant.
        """
        if not isinstance(input_latent, InpaintMaskArtifact):
            return None

        is_controlnet = isinstance(pipeline_artifact, ControlNetDiffusionPipelineArtifact)
        if is_controlnet:
            required_class = cls._inpaint_controlnet_pipeline_class
            mode = "ControlNet+Inpaint"
        else:
            required_class = cls._inpaint_pipeline_class
            mode = "Inpaint"

        if required_class is None:
            return [
                ValueError(
                    f"Pipeline '{pipeline_artifact.pipeline_name}' does not support {mode} mode "
                    f"(driver {cls.__name__} declares no inpaint pipeline class for this variant)."
                )
            ]
        return None

    def _is_controlnet_pipe(self) -> bool:
        """True if the current pipe is a ControlNet variant (heuristic by class name)."""
        return "ControlNet" in type(self.pipe).__name__

    def _get_inpaint_pipe(self) -> DiffusionPipeline | None:
        """Return an inpaint pipeline for this driver.

        If the current pipe is a ControlNet variant, swap to the combined
        ControlNet+Inpaint class so the ControlNet conditioning is preserved.
        """
        is_control_net = self._is_controlnet_pipe()
        if is_control_net:
            cls = self._inpaint_controlnet_pipeline_class
        else:
            cls = self._inpaint_pipeline_class
        if cls is None:
            return None
        if isinstance(self.pipe, cls):
            return self.pipe

        dtype = getattr(self.pipe, "dtype", None)
        return create_pipe_variant(self.pipe, cls, torch_dtype=dtype)

    def _get_inpaint_kwargs(self, artifact: InpaintMaskArtifact) -> dict[str, Any]:
        """Map an InpaintMaskArtifact to pipeline call kwargs."""
        device, dtype = self._get_device_and_type()
        result: dict[str, Any] = {
            "mask_image": artifact.mask_image,
            "strength": artifact.strength,
        }
        if artifact.source_latent is not None:
            result["image"] = artifact.source_latent.to(device=device, dtype=dtype)
        if artifact.masked_latent is not None:
            result["masked_image_latents"] = artifact.masked_latent.to(device=device, dtype=dtype)
        return result
