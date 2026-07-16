"""Registry mapping diffusers ``model_type`` labels to pipeline class names.

``model_type`` is the string produced by
``diffusers.loaders.single_file_utils.infer_diffusers_model_type`` when it
inspects a single-file checkpoint. ``pipeline_type`` is the diffusers pipeline
class name (e.g. ``"FluxPipeline"``) that owns the resulting component.

The map is intentionally small: only entries we actively support in the
single-file loader belong here.
"""

from __future__ import annotations

import inspect

from diffusers.loaders.single_file_utils import DIFFUSERS_DEFAULT_PIPELINE_PATHS  # type: ignore[reportMissingImports]

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY

# ``model_type`` -> pipeline class name owning the component. Every value here
# must also be a registered driver in
# ``latent_pipeline_drivers.driver_factory._DRIVER_REGISTRY``, otherwise the
# resulting artifact cannot be wired into a pipeline builder. The invariant
# is enforced by the import-time guard below.
#
# Entries are sourced from ``diffusers.loaders.single_file_utils`` -
# specifically the branches inside ``infer_diffusers_model_type`` and the
# ``DIFFUSERS_DEFAULT_PIPELINE_PATHS`` table.
MODEL_TYPE_TO_PIPELINE_TYPE: dict[str, str] = {
    # Flux family (Black Forest Labs).
    "flux-dev": "FluxPipeline",
    "flux-schnell": "FluxPipeline",
    "flux-fill": "FluxFillPipeline",
    "flux-2-dev": "Flux2Pipeline",
    # LTX family (Lightricks).
    "ltx-video": "LTXPipeline",
    "ltx-video-0.9.1": "LTXPipeline",
    "ltx-video-0.9.5": "LTXPipeline",
    "ltx-video-0.9.7": "LTXPipeline",
    "ltx2-dev": "LTX2Pipeline",
    # Stable Diffusion 3 / 3.5 family (Stability AI).
    "sd3": "StableDiffusion3Pipeline",
    "sd35_medium": "StableDiffusion3Pipeline",
    "sd35_large": "StableDiffusion3Pipeline",
    # Stable Diffusion XL base (Stability AI). ``xl_refiner`` and
    # ``xl_inpaint`` map to Img2Img / Inpaint pipelines that are not in
    # ``_DRIVER_REGISTRY`` and are therefore omitted.
    "xl_base": "StableDiffusionXLPipeline",
    # Z-Image (Tongyi-MAI). ``z-image-turbo-controlnet*`` entries describe
    # controlnet checkpoints, not the base transformer, and are omitted.
    "z-image-turbo": "ZImagePipeline",
    # WAN family (Alibaba).
    "wan-t2v-1.3B": "WanPipeline",
    "wan-t2v-14B": "WanPipeline",
    "wan-i2v-14B": "WanImageToVideoPipeline",
    "wan-animate-14B": "WanAnimatePipeline",
    "wan-vace-1.3B": "WanVACEPipeline",
    "wan-vace-14B": "WanVACEPipeline",
}


# Import-time invariant: every pipeline referenced above must have a driver
# registered, otherwise a ``ComponentArtifact`` produced from that model_type
# would materialise successfully but fail to wire into any builder.
_unregistered_pipelines = sorted(set(MODEL_TYPE_TO_PIPELINE_TYPE.values()) - set(_DRIVER_REGISTRY))
if _unregistered_pipelines:
    _offending = {
        model_type: pipeline_type
        for model_type, pipeline_type in MODEL_TYPE_TO_PIPELINE_TYPE.items()
        if pipeline_type in _unregistered_pipelines
    }
    _msg = (
        f"Attempted to load MODEL_TYPE_TO_PIPELINE_TYPE. "
        f"Failed because these pipeline classes are not registered in "
        f"latent_pipeline_drivers.driver_factory._DRIVER_REGISTRY: "
        f"{_unregistered_pipelines}. Offending entries: {_offending}."
    )
    raise RuntimeError(_msg)

# Import-time invariant: every model_type key must also exist in the diffusers
# ``DIFFUSERS_DEFAULT_PIPELINE_PATHS`` table. Catches typos when diffusers is
# upgraded and a model_type is renamed or removed upstream.
_unknown_model_types = sorted(set(MODEL_TYPE_TO_PIPELINE_TYPE) - set(DIFFUSERS_DEFAULT_PIPELINE_PATHS))
if _unknown_model_types:
    _msg = (
        f"Attempted to load MODEL_TYPE_TO_PIPELINE_TYPE. "
        f"Failed because these model_type keys are not present in "
        f"diffusers.loaders.single_file_utils.DIFFUSERS_DEFAULT_PIPELINE_PATHS: "
        f"{_unknown_model_types}."
    )
    raise RuntimeError(_msg)


def get_component_class(pipeline_cls: type, component: str) -> type:
    """Return the component class for a pipeline slot.

    Example: ``get_component_class(FluxPipeline, "transformer")`` returns ``FluxTransformer2DModel``.
    """
    sig_types = pipeline_cls._get_signature_types()
    type_tuple = sig_types.get(component)
    if type_tuple is None:
        msg = (
            f"Attempted to look up component class. Failed with pipeline_cls='{pipeline_cls.__name__}' "
            f"component='{component}' because '{component}' is not a typed __init__ parameter "
            f"on {pipeline_cls.__name__}."
        )
        raise ValueError(msg)

    component_cls = next(
        (t for t in type_tuple if t is not type(None) and t is not inspect.Parameter.empty),
        None,
    )
    if component_cls is None:
        msg = (
            f"Attempted to look up component class. Failed with pipeline_cls='{pipeline_cls.__name__}' "
            f"component='{component}' because no concrete type could be resolved from "
            f"the annotation on {pipeline_cls.__name__}."
        )
        raise ValueError(msg)

    return component_cls
