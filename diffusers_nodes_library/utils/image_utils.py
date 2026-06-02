import logging
from urllib.error import URLError

import numpy as np
from griptape.artifacts import ImageArtifact, ImageUrlArtifact
from griptape.loaders import ImageLoader
from griptape_nodes.files.file import File
from PIL import Image, ImageFilter
from requests.exceptions import RequestException

logger = logging.getLogger("modular_diffusers_nodes_library")

# OpenCV is faster for morphological/blur operations, so we use it if available.
# Fall back to PIL's MinFilter/MaxFilter/GaussianBlur otherwise.
try:
    import cv2  # type: ignore[reportMissingImports]

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None  # type: ignore[assignment]


def load_image_from_url_artifact(image_url_artifact: ImageUrlArtifact) -> ImageArtifact:
    """Load an ImageArtifact from an ImageUrlArtifact with proper error handling.

    Uses the engine's File class to resolve macro paths (e.g. {inputs}/...) and
    load from both local filesystem and remote URLs.

    Args:
        image_url_artifact: The ImageUrlArtifact to load

    Returns:
        ImageArtifact: The loaded image artifact

    Raises:
        ValueError: If image download fails with descriptive error message
    """
    try:
        image_bytes = File(image_url_artifact.value).read_bytes()
    except (URLError, RequestException, ConnectionError, TimeoutError, OSError) as err:
        details = (
            f"Failed to download image at '{image_url_artifact.value}'.\n"
            f"If this workflow was shared from another engine installation, "
            f"that image file will need to be regenerated.\n"
            f"Error: {err}"
        )
        raise ValueError(details) from err

    return ImageLoader().parse(image_bytes)


def _extract_from_rgb(image: Image.Image, channel: str) -> Image.Image:
    """Extract channel from RGB image."""
    red, green, blue = image.split()
    if channel == "red":
        return red
    if channel == "green":
        return green
    if channel == "blue":
        return blue
    # alpha not available in RGB, use red as fallback
    return red


def _extract_from_rgba(image: Image.Image, channel: str) -> Image.Image:
    """Extract channel from RGBA image."""
    red, green, blue, alpha = image.split()
    if channel == "red":
        return red
    if channel == "green":
        return green
    if channel == "blue":
        return blue
    if channel == "alpha":
        return alpha
    # Fallback to red channel
    return red


def _extract_from_la(image: Image.Image, channel: str) -> Image.Image:
    """Extract channel from LA image."""
    if channel == "alpha":
        _, alpha = image.split()
        return alpha
    gray, _ = image.split()
    return gray


def extract_channel_from_image(image: Image.Image, channel: str, context_name: str = "image") -> Image.Image:
    """Extract the specified channel from an image.

    Args:
        image: PIL Image to extract channel from
        channel: Channel to extract ("red", "green", "blue", "alpha")
        context_name: Name for error messages (e.g., "mask", "image")

    Returns:
        PIL Image containing the extracted channel

    Raises:
        ValueError: If the image mode is not supported
    """
    if image.mode == "L":
        return image
    if image.mode == "LA":
        return _extract_from_la(image, channel)
    if image.mode == "RGB":
        return _extract_from_rgb(image, channel)
    if image.mode == "RGBA":
        return _extract_from_rgba(image, channel)

    msg = f"Unsupported {context_name} mode: {image.mode}"
    raise ValueError(msg)


def apply_grow_shrink_to_mask(alpha: Image.Image, grow_shrink: float, context_name: str = "mask") -> Image.Image:
    """Apply grow/shrink morphological operation to mask using the fastest available method.

    Args:
        alpha: PIL Image (grayscale) representing the alpha channel/mask
        grow_shrink: Positive values shrink (erode), negative values grow (dilate)
        context_name: Name for debug logging (e.g., "mask", "Paint Mask")

    Returns:
        Transformed PIL Image
    """
    iterations = int(abs(grow_shrink))
    if iterations == 0:
        return alpha

    # Prefer OpenCV (fastest), then PIL iterations as fallback
    if OPENCV_AVAILABLE and cv2 is not None:
        msg = f"{context_name}: Using OpenCV for grow/shrink operation (iterations={iterations})"
        logger.debug(msg)
        # Use OpenCV for fastest morphological operations
        # Use a 3x3 kernel (same as PIL) and let OpenCV handle iterations
        alpha_array = np.array(alpha, dtype=np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        if grow_shrink > 0:
            alpha_array = cv2.erode(alpha_array, kernel, iterations=iterations)
        else:
            alpha_array = cv2.dilate(alpha_array, kernel, iterations=iterations)
        return Image.fromarray(alpha_array, mode="L")

    msg = f"{context_name}: Using PIL iterations for grow/shrink operation (iterations={iterations})"
    logger.debug(msg)
    # Fallback: PIL's MinFilter/MaxFilter only support size=3, so we must use iterations.
    # Each iteration processes the entire image, so large values (e.g., 100) can be slow.
    if grow_shrink > 0:
        for _ in range(iterations):
            alpha = alpha.filter(ImageFilter.MinFilter(size=3))
    else:
        for _ in range(iterations):
            alpha = alpha.filter(ImageFilter.MaxFilter(size=3))
    return alpha


def apply_blur_to_mask(alpha: Image.Image, blur_radius: float, context_name: str = "mask") -> Image.Image:
    """Apply blur to mask using the fastest available method.

    Args:
        alpha: PIL Image (grayscale) representing the alpha channel/mask
        blur_radius: Blur radius (0 = no blur)
        context_name: Name for debug logging (e.g., "mask", "Paint Mask")

    Returns:
        Blurred PIL Image
    """
    if blur_radius == 0:
        return alpha

    # Prefer OpenCV (faster), then PIL as fallback
    if OPENCV_AVAILABLE and cv2 is not None:
        msg = f"{context_name}: Using OpenCV for blur operation (radius={blur_radius})"
        logger.debug(msg)
        alpha_array = np.array(alpha, dtype=np.uint8)
        # OpenCV GaussianBlur requires kernel size to be odd
        kernel_size = int(blur_radius * 2 + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        alpha_array = cv2.GaussianBlur(alpha_array, (kernel_size, kernel_size), blur_radius)
        return Image.fromarray(alpha_array, mode="L")

    msg = f"{context_name}: Using PIL for blur operation (radius={blur_radius})"
    logger.debug(msg)
    return alpha.filter(ImageFilter.GaussianBlur(blur_radius))


def apply_mask_transformations(
    alpha: Image.Image,
    *,
    grow_shrink: float = 0,
    invert: bool = False,
    blur_radius: float = 0,
    context_name: str = "mask",
) -> Image.Image:
    """Apply all mask transformations in the correct order.

    Args:
        alpha: PIL Image (grayscale) representing the alpha channel/mask
        grow_shrink: Positive values shrink (erode), negative values grow (dilate)
        invert: Whether to invert the mask
        blur_radius: Blur radius (0 = no blur)
        context_name: Name for debug logging (e.g., "mask", "Paint Mask")

    Returns:
        Transformed PIL Image

    Order of operations: grow/shrink first (modify mask shape), then invert, then blur.
    """
    # Order: grow/shrink first (modify mask shape), then invert, then blur
    if grow_shrink != 0:
        alpha = apply_grow_shrink_to_mask(alpha, grow_shrink, context_name)

    if invert:
        alpha = Image.eval(alpha, lambda x: 255 - x)

    if blur_radius != 0:
        alpha = apply_blur_to_mask(alpha, blur_radius, context_name)

    return alpha
