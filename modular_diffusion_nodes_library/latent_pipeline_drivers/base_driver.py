from abc import ABC, abstractmethod
from typing import Any, ClassVar, TypeVar

_T = TypeVar("_T")

import numpy as np
import torch  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.modular_pipeline import (  # type: ignore[reportMissingImports]
    ModularPipeline,
    ModularPipelineBlocks,
    PipelineState,
)
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]
from PIL.Image import Image

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.misc.partial_denoise import (
    PartialDenoisePipelineRunner,
    PartialDenoiseSchedulerProxy,
)
from modular_diffusion_nodes_library.utils.pipeline_utils import create_pipe_variant

TextEncodings = dict[str, Any]
DecodeResult = Image | list[Image]


class LatentPipelineDriver(ABC):
    """Abstract base class for latent pipeline drivers.

    Latent shape & space contract
    -----------------------------
    All latents flowing across the public driver surface
    (``create_noise_latent``, ``encode_image``, ``encode_video``,
    ``add_noise_to_latent``, the output of ``denoise_latent``, and the input
    of ``decode_latent``) share a single canonical shape **and** statistical
    space so they can be freely composited, masked, added, or otherwise
    manipulated by downstream nodes.

    Shape
    ^^^^^
    Public latents are **unpacked** tensors (no model-specific patchifying /
    sequence packing). Their shape is:

    - 4-D for image pipelines: ``[B, C, H // vae_scale_factor, W // vae_scale_factor]``
    - 5-D for video pipelines: ``[B, C, T_latent, H // vae_scale_factor, W // vae_scale_factor]``

    where ``C`` (``num_channels_latents``) is inferred from the pipeline and
    ``T_latent = (T_video - 1) // vae_scale_factor_temporal + 1`` for video.

    The ``latents_source_shape`` argument passed to driver methods is in
    **pixel space**, **not** latent space:

    - 4-D for image: ``[B, C_image, H_pixel, W_pixel]``
    - 5-D for video: ``[B, C_image, T_video, H_pixel, W_pixel]``

    Drivers are responsible for translating between pixel-space dimensions
    (``latents_source_shape``) and the divided latent-space tensor dimensions
    when calling underlying pipelines.

    Space
    ^^^^^
    Latents are **normalised**: each channel is ~N(0, 1), matching the
    distribution of ``torch.randn``. For VAEs whose raw latents are not
    unit-variance (e.g. WAN, Flux2), drivers must apply the per-channel
    whitening ``(z - latents_mean) / latents_std`` inside ``encode_image`` /
    ``encode_video`` and the inverse inside ``decode_latent``.

    Any model-specific *packing* (e.g. Flux, Qwen) is applied transiently
    inside ``prepare_input_latent`` / ``prepare_output_latent`` and never
    appears on the public surface.
    """

    produces_video: ClassVar[bool] = False
    video_fps: ClassVar[int] = 16
    _partial_denoise_proxy_class: ClassVar[type[PartialDenoiseSchedulerProxy]] = PartialDenoiseSchedulerProxy
    _inpaint_pipeline_class: ClassVar[type[DiffusionPipeline] | None] = None

    def __init__(self, pipe: DiffusionPipeline):
        self._pipe = pipe
        self._modular_pipe: ModularPipeline | None = None

    @property
    def pipe(self) -> DiffusionPipeline:
        return self._pipe

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
        _, state = block(self.modular_pipe, state)
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

    def prepare_input_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> torch.Tensor:
        """Return latents ready to be passed into the pipeline, which may involve packing or other preprocessing."""
        return latents

    def prepare_output_latent(
        self, latents_from_pipe: torch.Tensor, latents_source_shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Return latents ready to be process further e.g. with mask, which may involve unpacking or other postprocessing."""
        return latents_from_pipe

    @abstractmethod
    def create_noise_latent(self, latents_source_shape: tuple[int, ...], seed: int) -> torch.Tensor:
        """Return pure noise latent. See class docstring for the latent shape contract."""
        ...

    def _extract_latents_from_output(self, pipe_output: Any) -> torch.Tensor:
        """Extract the raw latent tensor from a pipeline output object.
        Image pipelines expose the result as `images`
        override for video pipelines with `frames`.
        """
        return pipe_output.images

    @abstractmethod
    def decode_latent(self, latents: torch.Tensor, latents_source_shape: tuple[int, ...]) -> DecodeResult:
        """Return decoded image. See class docstring for the input latent shape contract."""
        ...

    @abstractmethod
    def encode_image(self, image: Image | torch.Tensor) -> torch.Tensor:
        """Return encoded latent. See class docstring for the output latent shape contract."""
        ...

    def encode_masked_image(self, image: Image, mask: Image) -> torch.Tensor:
        """Encode the source image with the masked region zeroed out."""
        image_processor = self.modular_pipe.image_processor
        source_t = image_processor.preprocess(image, height=image.height, width=image.width)
        mask_t = torch.from_numpy(np.array(mask, dtype="float32") / 255.0)[None, None]
        masked_t = source_t * (mask_t < 0.5)
        return self.encode_image(masked_t)

    @abstractmethod
    def add_noise_to_latent(
        self,
        latents: torch.Tensor,
        latents_source_shape: tuple[int, ...],
        seed: int,
        num_inference_steps: int,
        strength: float,
    ) -> torch.Tensor:
        """Return noised latent. See class docstring for the latent shape contract."""
        ...

    def encode_video(self, frames: list[Image]) -> torch.Tensor:
        """Encode a video file into a latent tensor. Override for pipelines that support video."""
        raise NotImplementedError(f"Pipeline '{self.pipe.__class__.__name__}' does not support video encoding.")

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
        """Run the denoising loop. See class docstring for the latent shape contract."""
        if isinstance(self.pipe, ModularPipeline):
            raise NotImplementedError(
                "denoise_latent is not implemented for ModularPipelines. Subclasses should implement this method for modular pipeline."
            )

        pipe = self.pipe

        device, dtype = self._get_device_and_type()
        latents = latents.to(device=device, dtype=dtype)

        inpaint_mask_artifact: InpaintMaskArtifact | None = kwargs.pop("inpaint_mask_artifact", None)
        if inpaint_mask_artifact is not None:
            pipe = self._get_inpaint_pipe()
            if pipe is None:
                raise NotImplementedError(f"Inpainting is not supported by {type(self).__name__}. ")

            inpaint_kwargs = self._get_inpaint_kwargs(inpaint_mask_artifact)
            kwargs.update(inpaint_kwargs)

            inpaint_strength: float = kwargs.pop("inpaint_strength", 1.0)
            kwargs["strength"] = inpaint_strength
        else:
            latents = self.prepare_input_latent(latents, latents_source_shape)

        height = latents_source_shape[-2]
        width = latents_source_shape[-1]

        pipe_kwargs: dict[str, Any] = {
            **kwargs,
            "width": width,
            "height": height,
            "output_type": "latent",
            "num_inference_steps": num_inference_steps,
            "callback_on_step_end": callback,
            "generator": kwargs.pop("generator", None) or torch.Generator().manual_seed(seed),
        }
        # Only pass latents when not inpainting — inpaint pipelines
        # handle their own latent preparation from image + mask.
        if inpaint_mask_artifact is None:
            pipe_kwargs["latents"] = latents

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

        return self.prepare_output_latent(self._extract_latents_from_output(pipe_output), latents_source_shape)

    # ------------------------------------------------------------------
    # Inpainting hooks
    # ------------------------------------------------------------------

    def _get_inpaint_pipe(self) -> DiffusionPipeline | None:
        """Return an inpaint pipeline for this driver."""
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
            "image": artifact.source_latent.to(device=device, dtype=dtype),
            "mask_image": artifact.mask_image,
        }
        if artifact.masked_latent is not None:
            result["masked_image_latents"] = artifact.masked_latent.to(device=device, dtype=dtype)
        return result
