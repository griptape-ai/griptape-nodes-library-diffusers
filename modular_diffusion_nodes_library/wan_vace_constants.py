"""Shared internal constants for WAN VACE runtime parameters and driver."""

SOURCE_VIDEO_KEY = "vace_source_video"
MASK_KEY = "vace_mask"
REFERENCE_IMAGES_KEY = "vace_reference_images"
GRAY_MASKED_SOURCE_KEY = "vace_gray_masked_source"

#: Each reference image costs a latent frame of noise and control tokens, so a video
#: connected to ``reference_images`` (one reference per frame) would balloon the latent.
MAX_REFERENCE_IMAGES = 8
