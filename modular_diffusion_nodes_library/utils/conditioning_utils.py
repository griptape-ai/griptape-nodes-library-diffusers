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


class FramePosition(StrEnum):
    """Symbolic frame positions used by preset-based conditioning.

    Serialized into the output dict's `frame_index` field as a plain string
    (the StrEnum value). Drivers call `resolve_frame_index(value, num_frames)`
    to turn it into a concrete int at runtime, when `num_frames` is known.
    """

    FIRST = "first"
    MIDDLE = "middle"
    LAST = "last"


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


def resolve_conditioning_video(video_artifact: Any) -> list[Image]:
    """Decode a video artifact into RGB PIL frames."""
    if video_artifact is None:
        msg = "Attempted to build video conditioning. Failed because the video artifact was None."
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


def resolve_frame_index(value: int | str, num_frames: int) -> int:
    """Resolve a `frame_index` value from the conditioning payload to a concrete int.

    Presets emit symbolic frame positions (`"first"`, `"middle"`, `"last"`); the
    flexible image config emits plain ints. Drivers call this once per slot when
    they know `num_frames`, so the producer never has to know runtime sizing.
    """
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        msg = (
            f"Attempted to resolve frame_index. "
            f"Failed with value={value!r} (type {type(value).__name__}) because it is neither int nor str."
        )
        raise ValueError(msg)

    match value:
        case FramePosition.FIRST.value:
            return 0
        case FramePosition.MIDDLE.value:
            return num_frames // 2
        case FramePosition.LAST.value:
            return num_frames - 1
        case _:
            msg = (
                f"Attempted to resolve frame_index. "
                f"Failed with value={value!r} because it is not a known FramePosition "
                f"({[p.value for p in FramePosition]})."
            )
            raise ValueError(msg)


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
        resized = frame.resize((new_w, new_h), PILImage.Resampling.LANCZOS)
        cropped = resized.crop((left, top, left + target_width, top + target_height))
        resized_frames.append(cropped)
    return resized_frames
