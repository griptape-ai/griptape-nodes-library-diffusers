from typing import ClassVar

from modular_diffusion_nodes_library.parameters.media_gen_conditioning.conditioning_layout import (
    FlexibleImageConfig,
    MediaGenConditioningConfig,
)
from modular_diffusion_nodes_library.runtime_parameters.flux2_runtime_parameters import (
    Flux2PipelineRuntimeParameters,
)


class Flux2KleinPipelineRuntimeParameters(Flux2PipelineRuntimeParameters):
    """Flux2 Klein variant \u2014 same runtime params, with media-gen conditioning enabled.

    The Klein pipeline accepts up to 8 reference images as additional conditioning.
    Strength and frame-index controls are not exposed because Klein treats the
    reference images uniformly.
    """

    CONDITIONING_CONFIG: ClassVar[MediaGenConditioningConfig | None] = MediaGenConditioningConfig(
        image=FlexibleImageConfig(
            min_count=1,
            max_count=8,
            expose_strength=False,
            expose_frame_index=False,
        ),
    )
