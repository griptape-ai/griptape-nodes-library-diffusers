from __future__ import annotations

import io
from typing import TYPE_CHECKING

from griptape.artifacts import ImageArtifact, ImageUrlArtifact

from modular_diffusion_nodes_library.utils.config_utils import get_config_value, get_workspace_path

if TYPE_CHECKING:
    from PIL.Image import Image


def image_artifact_to_pil(image_artifact: ImageArtifact) -> Image:
    """Converts Griptape ImageArtifact to Pillow Image."""
    import PIL.Image
    import PIL.ImageOps

    return PIL.Image.open(io.BytesIO(image_artifact.value))


def pil_to_image_artifact(pil_image: Image, directory_path: str = "") -> ImageUrlArtifact:
    """Converts Pillow Image to Griptape ImageArtifact."""
    from griptape_nodes.retained_mode.managers.os_manager import OSManager

    image_io = io.BytesIO()
    pil_image.save(image_io, "PNG")
    image_bytes = image_io.getvalue()

    from griptape_nodes.files.project_file import ProjectFileDestination

    if directory_path:
        # Perform cleanup if needed before saving new file
        cleanup_enabled = get_config_value("modular_diffusion_library.enable_directory_cleanup")
        if cleanup_enabled:
            static_files_directory = get_config_value("static_files_directory", default="staticfiles")
            path = get_workspace_path() / static_files_directory / directory_path

            max_size_gb = get_config_value("modular_diffusion_library.max_directory_size_gb")
            # `OSManager.cleanup_directory_if_needed` is a static method: it touches only the
            # filesystem, which a worker shares, and holds none of the manager state the facade guard
            # exists to protect. Imported directly because reaching it through the facade raises in a
            # worker, and no request wraps it.
            OSManager.cleanup_directory_if_needed(full_directory_path=path, max_size_gb=max_size_gb)

        dest = ProjectFileDestination.from_situation(
            filename=f"{directory_path}/image.png", situation="save_node_output"
        )
    else:
        # No directory prefix - direct storage
        dest = ProjectFileDestination.from_situation(filename="image.png", situation="save_node_output")

    saved = dest.write_bytes(image_bytes)
    return ImageUrlArtifact(saved.location)


def pad_mirror(image: Image, target_size: tuple[int, int]) -> Image:
    """Expand an image to the target size using repeated mirrored tiling.

    Parameters:
    - image: Input Pillow Image
    - target_size: (new_width, new_height)

    Returns:
    - A new Image of size target_size, filled with mirrored tiles of the original
    """
    import PIL.Image
    import PIL.ImageOps

    orig_w, orig_h = image.size
    target_w, target_h = target_size

    # Create the 2x2 mirrored variants
    tiles = [
        [image, PIL.ImageOps.mirror(image)],
        [PIL.ImageOps.flip(image), PIL.ImageOps.mirror(PIL.ImageOps.flip(image))],
    ]

    # Compute how many tiles are needed horizontally and vertically
    tiles_x = (target_w + orig_w - 1) // orig_w
    tiles_y = (target_h + orig_h - 1) // orig_h

    # Create blank output canvas
    new_img = PIL.Image.new(image.mode, (target_w, target_h))

    for y in range(tiles_y):
        for x in range(tiles_x):
            tile = tiles[y % 2][x % 2]
            new_img.paste(tile, (x * orig_w, y * orig_h))

    # Crop to exact target size (if overshot)
    return new_img.crop((0, 0, target_w, target_h))
