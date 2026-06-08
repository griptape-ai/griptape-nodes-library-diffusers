import logging
from typing import Any, override

import PIL.Image
from diffusers.modular_pipelines.modular_pipeline import ModularPipeline  # type: ignore[reportMissingImports]
from diffusers.modular_pipelines.wan.modular_blocks_wan import WanBlocks  # type: ignore[reportMissingImports]
from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.artifact_utils.inpaint_mask_artifact import InpaintMaskArtifact
from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_types import GeneratorState
from modular_diffusion_nodes_library.latent_pipeline_drivers.wan import WanTextToVideoLatentPipelineDriver
from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_payload import normalize_to_payloads
from modular_diffusion_nodes_library.utils.conditioning_utils import (
    ConditioningMode,
    resolve_conditioning_image,
    resolve_conditioning_video,
    resolve_frame_index,
)

logger = logging.getLogger("modular_diffusers_nodes_library")

_SOURCE_VIDEO_KEY = "vace_source_video"
_MASK_KEY = "vace_mask"
_REFERENCE_IMAGES_KEY = "vace_reference_images"


def _payload_to_frames(
    payload_value: Any,
    num_frames: int,
    height: int,
    width: int,
    fill: tuple[int, int, int] | int,
    image_mode: str = "RGB",
) -> list[PIL.Image.Image]:
    """Convert a conditioning payload to a list of PIL frames.

    VIDEO payload: place video frames starting at frame_index; truncate if the video extends
    past num_frames. Frames before frame_index and after the video end are filled with fill.
    IMAGE payload: build a num_frames list pre-filled with fill, place each image at
    its resolved frame position.

    image_mode controls the PIL mode of every frame in the returned list ("RGB" for video /
    source frames, "L" for mask frames). Placed images are converted to image_mode.
    """
    payloads = normalize_to_payloads(payload_value)
    if payloads is None:
        return []

    payload = payloads[0]

    if payload.mode is ConditioningMode.VIDEO:
        entry = payload.entries[0]
        frames = resolve_conditioning_video(entry.artifact)
        start = resolve_frame_index(entry.frame_index, num_frames)
        if start < 0:
            start = num_frames + start
        result = [PIL.Image.new(image_mode, (width, height), fill) for _ in range(num_frames)]
        for i, frame in enumerate(frames):
            target = start + i
            if target >= num_frames:
                break
            result[target] = frame.resize((width, height)).convert(image_mode)
        return result

    result: list[PIL.Image.Image] = [PIL.Image.new(image_mode, (width, height), fill) for _ in range(num_frames)]
    for entry in payload.entries:
        image = resolve_conditioning_image(entry.artifact)
        frame_index = resolve_frame_index(entry.frame_index, num_frames)
        if frame_index < 0:
            frame_index = num_frames + frame_index
        if not (0 <= frame_index < num_frames):
            raise ValueError(
                f"Attempted to build VACE conditioning frames. "
                f"Failed because frame_index={frame_index} is out of range for num_frames={num_frames}. "
                f"Valid range is [0, {num_frames - 1}] or negative indices down to {-num_frames}."
            )
        result[frame_index] = image.resize((width, height)).convert(image_mode)
    return result


def _derive_mask_from_source_media(
    payload_value: Any,
    num_frames: int,
    height: int,
    width: int,
) -> list[PIL.Image.Image] | None:
    """Auto-derive a binary mask from the source video payload.

    VIDEO payload: black (preserve) for the covered frame range, white (generate) elsewhere.
    IMAGE payload: black at explicitly placed frame positions, white elsewhere.
    """
    payloads = normalize_to_payloads(payload_value)
    if payloads is None:
        return None

    payload = payloads[0]

    mask_black = PIL.Image.new("L", (width, height), 0)
    mask_white = PIL.Image.new("L", (width, height), 255)
    result: list[PIL.Image.Image] = [mask_white for _ in range(num_frames)]

    if payload.mode is ConditioningMode.VIDEO:
        entry = payload.entries[0]
        frames = resolve_conditioning_video(entry.artifact)
        start = resolve_frame_index(entry.frame_index, num_frames)
        if start < 0:
            start = num_frames + start
        count = min(len(frames), max(0, num_frames - start))
        for i in range(count):
            result[start + i] = mask_black
        return result

    for entry in payload.entries:
        frame_index = resolve_frame_index(entry.frame_index, num_frames)
        if frame_index < 0:
            frame_index = num_frames + frame_index
        result[frame_index] = mask_black
    return result


def _payload_to_reference_images(payload_value: Any) -> list[PIL.Image.Image]:
    """Convert conditioning payload(s) to a flat list of PIL reference images.

    IMAGE payload: each image is used directly.
    VIDEO payload: the video is broken into individual frames.
    """
    payloads = normalize_to_payloads(payload_value)
    if payloads is None:
        return []

    reference_images: list[PIL.Image.Image] = []
    for payload in payloads:
        if payload.mode is ConditioningMode.VIDEO:
            entry = payload.entries[0]
            frames = resolve_conditioning_video(entry.artifact)
            reference_images.extend(frames)
        else:
            for entry in payload.entries:
                image = resolve_conditioning_image(entry.artifact)
                reference_images.append(image)
    return reference_images


class WanVaceLatentPipelineDriver(WanTextToVideoLatentPipelineDriver):
    def __init__(self, pipe: DiffusionPipeline):
        super().__init__(pipe)

    @override
    def _create_modular_pipe(self) -> ModularPipeline:
        # VACE always uses WAN 2.1 block architecture. transformer_2 on a VACE pipeline
        # is an optional second VACE transformer — not a WAN 2.2 transformer — so we
        # must not use Wan22Blocks regardless of its presence.
        return WanBlocks().init_pipeline()

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
        """Denoise a WAN VACE video latent, converting conditioning payloads to pipeline kwargs."""
        source_media_payload = kwargs.pop(_SOURCE_VIDEO_KEY, None)
        mask_payload = kwargs.pop(_MASK_KEY, None)
        reference_payload = kwargs.pop(_REFERENCE_IMAGES_KEY, None)

        if mask_payload is not None and source_media_payload is None:
            raise ValueError(
                f"Attempted to denoise with WAN VACE (node '{self.driver_namespace}'). "
                "Failed because `mask` was provided without `source_media`. "
                "Connect a source media conditioning or disconnect the mask."
            )
        if reference_payload is not None and source_media_payload is None:
            raise ValueError(
                f"Attempted to denoise with WAN VACE (node '{self.driver_namespace}'). "
                "Failed because `reference_images` were provided without `source_media`. "
                "Connect a source media conditioning or disconnect the reference images."
            )

        if source_media_payload is not None:
            num_frames = latent.source_shape[-3]
            height = latent.source_shape[-2]
            width = latent.source_shape[-1]

            kwargs["height"] = height
            kwargs["width"] = width
            kwargs["video"] = _payload_to_frames(source_media_payload, num_frames, height, width, (128, 128, 128))

            if mask_payload is not None:
                kwargs["mask"] = _payload_to_frames(mask_payload, num_frames, height, width, 255, "L")
            else:
                # Derive mask from source video: black (preserve) source frames, white (generate).
                kwargs["mask"] = _derive_mask_from_source_media(source_media_payload, num_frames, height, width)

            if reference_payload is not None:
                kwargs["reference_images"] = _payload_to_reference_images(reference_payload)

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
