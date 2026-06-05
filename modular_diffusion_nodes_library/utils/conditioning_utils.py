from typing import Any

from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from PIL import Image as PILImage
from PIL.Image import Image

from modular_diffusion_nodes_library.utils.image_utils import load_image_from_url_artifact
from modular_diffusion_nodes_library.utils.pillow_utils import image_artifact_to_pil
from modular_diffusion_nodes_library.utils.video_utils import load_video_frames_from_url_artifact


def resolve_conditioning_video(media_gen_conditioning: dict[str, Any]) -> list[Image]:
    video_artifact = media_gen_conditioning.get("video")
    if video_artifact is None:
        msg = "Attempted to build video conditioning. Failed because 'video' was missing."
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
