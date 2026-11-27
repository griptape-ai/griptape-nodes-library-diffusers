import logging
from enum import StrEnum
from typing import TYPE_CHECKING, NamedTuple

from modular_diffusion_nodes_library.utils.config_utils import get_config_value

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class DimensionAlignmentStatus(StrEnum):
    PASS = "pass"  # Dimensions are valid and recommended.
    ADVISORY = "advisory"  # Dimensions are valid but outside the model's recommended sizes; never blocks a run.
    INVALID = "invalid"  # The requested dimensions are invalid and had to be (or must be) adjusted.


class DimensionAlignmentResult(NamedTuple):
    height: int
    width: int
    num_frames: int | None
    message: str | None  # None = status PASS; str = formatted for the status and the current auto_resize mode
    status: DimensionAlignmentStatus = DimensionAlignmentStatus.PASS


def snap_dimensions(
    driver: "LatentPipelineDriver",
    height: int,
    width: int,
    num_frames: int | None = None,
) -> DimensionAlignmentResult:
    """Return ceiling-snapped dimension values, a status, and a formatted message if there is anything to report.

    Always computes valid values via driver.align_dimensions. If nothing changed, the driver's
    result is returned as is (PASS, or ADVISORY if the driver flagged a non-recommended size).
    If any value was adjusted, status is INVALID and message is built from
    driver.validate_dimensions and formatted based on the enable_auto_resize config:
      auto_resize=True  -> warning message confirming resize happened
      auto_resize=False -> error message with hint to enable auto-resize

    Callers use result.message in logger.warning, and raise ValueError only for status INVALID.
    """
    aligned = driver.align_dimensions(height, width, num_frames)
    if aligned.height == height and aligned.width == width and aligned.num_frames == num_frames:
        return aligned
    detail = " ".join(driver.validate_dimensions(height, width, num_frames))
    auto_resize = get_config_value("modular_diffusion_library.enable_auto_resize")
    if auto_resize:
        message = (
            f"Resized automatically to the suggested value to match pipeline requirements (Auto Resize is on). {detail}"
        )
    else:
        message = f'{detail} To resize automatically, enable "Auto Resize" under Library Settings > Diffusion Library.'
    return DimensionAlignmentResult(
        aligned.height, aligned.width, aligned.num_frames, message, DimensionAlignmentStatus.INVALID
    )
