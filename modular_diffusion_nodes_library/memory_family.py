"""Shared memory-estimation family metadata."""

from enum import Enum


class MemoryFamily(Enum):
    """Architecture family used to pick a denoiser activation-memory formula."""

    UNET_SDPA = "unet_sdpa"
    IMAGE_DIT_SDPA = "image_dit_sdpa"
    VIDEO_DIT_JOINT_SDPA = "video_dit_joint_sdpa"
