"""Registry mapping pipeline class name to runtime-params class.

TODO: Merge with `_DRIVER_REGISTRY` in
`modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory`.
Both registries key on the same `pipeline_class: str` and every new pipeline
must be added to both. Until they are unified, the import-time guard below
asserts the two key sets stay aligned.
"""

from __future__ import annotations

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY
from modular_diffusion_nodes_library.runtime_parameters.flux2_klein_runtime_parameters import (
    Flux2KleinPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.flux2_runtime_parameters import (
    Flux2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.flux_fill_runtime_parameters import (
    FluxFillPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.flux_kontext_runtime_parameters import (
    FluxKontextPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.flux_runtime_parameters import (
    FluxPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.ltx2_runtime_parameters import (
    LTX2PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.ltx_runtime_parameters import (
    LTXPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.qwen_edit_runtime_parameters import (
    QwenEditPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.qwen_runtime_parameters import (
    QwenPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.runtime_parameters import (
    DiffusionPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.stable_diffusion_3_runtime_parameters import (
    StableDiffusion3PipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.stable_diffusion_xl_runtime_parameters import (
    StableDiffusionXLPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.wan_animate_runtime_parameters import (
    WanAnimatePipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.wan_i2v_runtime_parameters import (
    WanImageToVideoPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.wan_runtime_parameters import (
    WanPipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.wan_vace_runtime_parameters import (
    WanVacePipelineRuntimeParameters,
)
from modular_diffusion_nodes_library.runtime_parameters.z_image_runtime_parameters import (
    ZImagePipelineRuntimeParameters,
)

_RUNTIME_PARAMS_REGISTRY: dict[str, type[DiffusionPipelineRuntimeParameters]] = {
    "FluxPipeline": FluxPipelineRuntimeParameters,
    "FluxFillPipeline": FluxFillPipelineRuntimeParameters,
    "FluxKontextPipeline": FluxKontextPipelineRuntimeParameters,
    "Flux2Pipeline": Flux2PipelineRuntimeParameters,
    "Flux2KleinPipeline": Flux2KleinPipelineRuntimeParameters,
    "LTXPipeline": LTXPipelineRuntimeParameters,
    "LTX2Pipeline": LTX2PipelineRuntimeParameters,
    "QwenImagePipeline": QwenPipelineRuntimeParameters,
    "QwenImageEditPipeline": QwenEditPipelineRuntimeParameters,
    "StableDiffusion3Pipeline": StableDiffusion3PipelineRuntimeParameters,
    "StableDiffusionXLPipeline": StableDiffusionXLPipelineRuntimeParameters,
    "WanPipeline": WanPipelineRuntimeParameters,
    "WanAnimatePipeline": WanAnimatePipelineRuntimeParameters,
    "WanImageToVideoPipeline": WanImageToVideoPipelineRuntimeParameters,
    "WanVACEPipeline": WanVacePipelineRuntimeParameters,
    "ZImagePipeline": ZImagePipelineRuntimeParameters,
}


# Import-time invariant: keep the two pipeline-class registries aligned until
# they are merged (see module-level TODO).
_driver_keys = set(_DRIVER_REGISTRY)
_runtime_keys = set(_RUNTIME_PARAMS_REGISTRY)
if _driver_keys != _runtime_keys:
    _missing_runtime = sorted(_driver_keys - _runtime_keys)
    _missing_driver = sorted(_runtime_keys - _driver_keys)
    _msg = (
        f"Attempted to load _RUNTIME_PARAMS_REGISTRY. "
        f"Failed because its keys diverge from _DRIVER_REGISTRY. "
        f"Missing from _RUNTIME_PARAMS_REGISTRY: {_missing_runtime}. "
        f"Missing from _DRIVER_REGISTRY: {_missing_driver}."
    )
    raise RuntimeError(_msg)


def get_runtime_params_class(pipeline_class: str | None) -> type[DiffusionPipelineRuntimeParameters] | None:
    """Return the runtime-params class for ``pipeline_class``, or ``None`` if unsupported."""
    if pipeline_class is None:
        return None
    return _RUNTIME_PARAMS_REGISTRY.get(pipeline_class)
