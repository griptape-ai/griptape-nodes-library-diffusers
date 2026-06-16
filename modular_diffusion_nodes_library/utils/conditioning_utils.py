from enum import StrEnum
from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from PIL import Image as PILImage
from PIL.Image import Image

from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil
from modular_diffusion_nodes_library.utils.video_utils import load_video_frames_from_url_artifact


class ConditioningMode(StrEnum):
    IMAGE = "image"
    VIDEO = "video"


class MediaGenConditioningKey:
    """Payload dict keys for the media-gen conditioning surface.

    The conditioning node emits

        {OUTPUT: {MODE: "image" | "video", ...mode-specific...}}

    Video-mode entry shape:
        {MODE: "video", VIDEO: <artifact>, FRAME_INDEX: int, STRENGTH: float}

    Image-mode entry shape:
        {MODE: "image", IMAGES: [{IMAGE: <artifact>, FRAME_INDEX: int, STRENGTH: float}, ...]}

    Both the parameter component that produces the payload and the drivers
    that consume it MUST reference these constants.
    """

    OUTPUT = "media_gen_conditioning"
    MODE = "mode"
    VIDEO = "video"
    IMAGES = "images"
    IMAGE = "image"
    FRAME_INDEX = "frame_index"
    STRENGTH = "strength"


def resolve_conditioning_video(media_gen_conditioning: dict[str, Any]) -> list[Image]:
    video_artifact = media_gen_conditioning.get(MediaGenConditioningKey.VIDEO)
    if video_artifact is None:
        msg = (
            f"Attempted to build video conditioning. "
            f"Failed because '{MediaGenConditioningKey.VIDEO}' was missing."
        )
        raise ValueError(msg)

    frames = load_video_frames_from_url_artifact(video_artifact)
    if not frames:
        raise ValueError("Failed to load any frames from video conditioning artifact.")

    return [frame.convert("RGB") for frame in frames]


def resolve_conditioning_image(image_value: Any) -> Image:
    if isinstance(image_value, PILImage.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, ImageUrlArtifact):
        image_value = load_image_from_url_artifact(image_value)

    if isinstance(image_value, ImageArtifact):
        return image_artifact_to_pil(image_value).convert("RGB")

    raise ValueError(
        f"Attempted to build image conditioning. Failed with image value type '{type(image_value).__name__}'."
    )


def pixel_frame_index_to_latent_index(pixel_frame_index: int, temporal_ratio: int, pixel_num_frames: int) -> int:
    """Map a pixel-space frame index to its VAE-latent-space index.

    Negative or zero indices wrap modulo `pixel_num_frames`; positive indices map
    via the VAE temporal compression ratio (frame 1 is the first compressed group).
    """
    if pixel_frame_index <= 0:
        return pixel_frame_index % pixel_num_frames

    return 1 + (pixel_frame_index - 1) // temporal_ratio


def resize_frames_scale_to_fill(
    frames: list[Image],
    target_height: int,
    target_width: int,
) -> list[Image]:
    """Scale-to-fill then center-crop frames to exact target dimensions."""
    if not frames or not isinstance(frames[0], PILImage.Image):
        return frames

    src_w, src_h = frames[0].size
    if src_h == target_height and src_w == target_width:
        return frames

    scale = max(target_height / src_h, target_width / src_w)
    new_h = round(src_h * scale)
    new_w = round(src_w * scale)
    left = (new_w - target_width) // 2
    top = (new_h - target_height) // 2
    resized_frames = []
    for frame in frames:
        resized = frame.resize((new_w, new_h), PILImage.LANCZOS)
        cropped = resized.crop((left, top, left + target_width, top + target_height))
        resized_frames.append(cropped)
    return resized_frames
