import logging
from typing import TYPE_CHECKING, NamedTuple

from griptape_nodes.retained_mode.griptape_nodes import GriptapeNodes

if TYPE_CHECKING:
    from modular_diffusion_nodes_library.latent_pipeline_drivers.base_driver import LatentPipelineDriver

logger = logging.getLogger("modular_diffusers_nodes_library")


class DimensionAlignmentResult(NamedTuple):
    height: int
    width: int
    num_frames: int | None
    message: str | None  # None = already valid; str = formatted for the current auto_resize mode


def snap_dimensions(
    driver: "LatentPipelineDriver",
    height: int,
    width: int,
    num_frames: int | None = None,
) -> DimensionAlignmentResult:
    """Return ceiling-snapped dimension values and a formatted message if any adjustment was needed.

    Always computes valid values via driver.align_dimensions. If nothing changed,
    message is None. If any value was adjusted, message is built from
    driver.validate_dimensions and formatted based on the enable_auto_resize config:
      auto_resize=True  -> warning message confirming resize happened
      auto_resize=False -> error message with hint to enable auto-resize

    Callers use result.message directly in logger.warning or raise ValueError.
    """
    aligned = driver.align_dimensions(height, width, num_frames)
    if aligned.height == height and aligned.width == width and aligned.num_frames == num_frames:
        return aligned
    detail = " ".join(driver.validate_dimensions(height, width, num_frames))
    auto_resize = GriptapeNodes.ConfigManager().get_config_value("modular_diffusion_library.enable_auto_resize")
    if auto_resize:
        message = (
            f"Resized automatically to the suggested value to match pipeline requirements (Auto Resize is on). {detail}"
        )
    else:
        message = f'{detail} To resize automatically, enable "Auto Resize" under Library Settings > Modular Diffusion Library.'
    return DimensionAlignmentResult(aligned.height, aligned.width, aligned.num_frames, message)
