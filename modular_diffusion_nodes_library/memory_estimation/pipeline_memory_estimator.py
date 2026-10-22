"""Orchestrates a per-component memory estimate for an already-loaded diffusion pipeline.

This is the only module in `memory_estimation` that touches the real `pipe` object.
Everything it calls (activation_formulas, vae_trace, text_encoder_formula,
family_registry) is pure/testable with synthetic inputs.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, cast

import torch  # type: ignore[reportMissingImports]
from huggingface_hub import try_to_load_from_cache

from modular_diffusion_nodes_library.artifact_utils.component_artifact import (
    ComponentArtifact,
    ComponentSourceType,
    ModelComponentArtifact,
)
from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import ControlNetDiffusionPipelineArtifact
from modular_diffusion_nodes_library.component_loading.component_slots import slot_component_kind
from modular_diffusion_nodes_library.component_loading.config_resolver import try_load_json_dict
from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class
from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import get_driver_class
from modular_diffusion_nodes_library.memory_estimation.activation_formulas import (
    estimate_image_dit_sdpa_activation_bytes,
    estimate_unet_sdpa_activation_bytes,
    estimate_video_dit_joint_sdpa_activation_bytes,
)
from modular_diffusion_nodes_library.memory_estimation.controlnet_registry import (
    get_controlnet_denoiser_config_fields,
)
from modular_diffusion_nodes_library.memory_estimation.estimate_types import (
    ComponentMemoryEstimate,
    ComponentRole,
    PipelineMemoryEstimate,
)
from modular_diffusion_nodes_library.memory_estimation.family_registry import (
    UnetLevelConfig,
    get_denoiser_config_fields,
    get_sdxl_unet_levels,
)
from modular_diffusion_nodes_library.memory_estimation.lora_memory import (
    LoraAdapterMemoryEstimate,
    collect_runtime_lora_memory,
)
from modular_diffusion_nodes_library.memory_estimation.meta_device_builder import (
    ComponentConfigNotCachedError,
    build_component_on_meta_device,
    resolve_base_component_config,
)
from modular_diffusion_nodes_library.memory_estimation.quantization_bytes import (
    QUANTIZATION_MODE_BYTES_PER_ELEMENT,
    estimate_component_weight_bytes,
    resolve_effective_bytes_per_element,
)
from modular_diffusion_nodes_library.memory_estimation.text_encoder_formula import (
    estimate_text_encoder_activation_bytes,
)
from modular_diffusion_nodes_library.memory_estimation.vae_trace import (
    estimate_vae_activation_bytes,
    vae_estimate_warning,
)
from modular_diffusion_nodes_library.memory_family import MemoryFamily
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.parameters.pipelinetype_parameters import MODULAR_PIPELINE_TYPE_PROVIDER_MAP
from modular_diffusion_nodes_library.utils.huggingface_utils import model_cache
from modular_diffusion_nodes_library.utils.pipeline_utils import (
    MEMORY_HEADROOM_FACTOR,
    apply_vae_memory_options,
    detect_offload_method,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_model_memory, to_human_readable_size

if TYPE_CHECKING:
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline  # type: ignore[reportMissingImports]

    from modular_diffusion_nodes_library.artifact_utils.latent_artifact import LatentArtifact
    from modular_diffusion_nodes_library.artifact_utils.pipeline_artifact import DiffusionPipelineArtifact

logger = logging.getLogger("modular_diffusers_nodes_library")

# Base-pipeline components are loaded at a hardcoded dtype today (e.g.
# flux_parameters.py:65) -- no per-pipeline dtype selection exists yet to read from the
# artifact, so the pre-load estimator assumes the same hardcoded value.
_BASE_PIPELINE_TORCH_DTYPE = torch.bfloat16

# Activations are computed in the model's compute dtype, never narrower than 16-bit, even
# when the weights are stored quantized. Reading the activation width off a quantized
# parameter reported int8/uint8 (halving every activation term) and, in the pre-load path
# where int4 resolves to 0.5 bytes, `int(0.5)` zeroed them out entirely with no warning.
_MIN_ACTIVATION_BYTES_PER_ELEMENT = 2

_TRUSTED_BUILDER_CLASSES: dict[tuple[str, str], type[ModularDiffusionPipelineTypePipelineParameters]] = {
    (parameters_class.__module__, parameters_class.__name__): parameters_class
    for provider_parameters_class in MODULAR_PIPELINE_TYPE_PROVIDER_MAP.values()
    for parameters_class in provider_parameters_class.get_pipeline_type_dict().values()
}


def _classify_role(component_name: str) -> ComponentRole:
    component_kind = slot_component_kind(component_name)
    if component_kind in {"transformer", "unet"}:
        return ComponentRole.DENOISER
    if component_kind in {"text_encoder", "vae", "controlnet"}:
        return ComponentRole(component_kind)
    return ComponentRole.OTHER


def _element_size(component: torch.nn.Module) -> int:
    try:
        return next(component.parameters()).element_size()
    except StopIteration:
        return 4  # no parameters found; fall back to float32 width


def _activation_element_size(component: torch.nn.Module) -> int:
    """Return the bytes-per-element activations will actually be computed at.

    Takes the widest parameter dtype rather than the first: a quantized module stores its
    linear weights at int8/uint8 but keeps norms and biases in the compute dtype, so the
    maximum recovers the compute width while `next(parameters())` may land on a quantized
    tensor and report 1.
    """
    element_sizes = [parameter.element_size() for parameter in component.parameters()]
    if not element_sizes:
        return _MIN_ACTIVATION_BYTES_PER_ELEMENT
    return max(max(element_sizes), _MIN_ACTIVATION_BYTES_PER_ELEMENT)


def _estimate_denoiser_activation_bytes(
    component: torch.nn.Module,
    pipe: DiffusionPipeline,
    pipeline_name: str,
    latent: LatentArtifact,
    element_size: int,
) -> tuple[int, str | None]:
    driver_class = get_driver_class(pipeline_name)
    family = driver_class.memory_family if driver_class is not None else None
    if family is None:
        return 0, f"No memory-estimation family registered for pipeline '{pipeline_name}'; showing weights only."

    latent_shape = latent.shape
    height_latent, width_latent = latent_shape[-2], latent_shape[-1]

    try:
        if family == MemoryFamily.UNET_SDPA:
            levels = get_sdxl_unet_levels(pipe)
            activation_bytes = estimate_unet_sdpa_activation_bytes(levels, height_latent, width_latent, element_size)
            return activation_bytes, None

        fields = get_denoiser_config_fields(pipeline_name, pipe)
        if fields is None:
            return 0, f"No denoiser config adapter registered for pipeline '{pipeline_name}'; showing weights only."

        if family == MemoryFamily.IMAGE_DIT_SDPA:
            activation_bytes = estimate_image_dit_sdpa_activation_bytes(
                height_latent,
                width_latent,
                fields.hidden_dim,
                fields.patch_size_spatial,
                element_size,
            )
            return activation_bytes, None

        num_frames_latent = latent_shape[-3] if len(latent_shape) >= 5 else 1  # noqa: PLR2004
        activation_bytes = estimate_video_dit_joint_sdpa_activation_bytes(
            num_frames_latent,
            height_latent,
            width_latent,
            fields.hidden_dim,
            fields.patch_size_spatial,
            fields.patch_size_temporal,
            element_size,
        )
        return activation_bytes, None
    except (AttributeError, KeyError, TypeError, IndexError) as e:
        return 0, f"Failed to read denoiser config for pipeline '{pipeline_name}': {e}. Showing weights only."


def _get_controlnet_unet_levels(controlnet_config: Any) -> list[UnetLevelConfig]:
    """Read a ControlNetModel's own config as UNet-SDPA per-level shapes.

    Mirrors get_sdxl_unet_levels() but reads the ControlNet's own
    block_out_channels/transformer_layers_per_block, not the base UNet's --
    ControlNetModel has no up-blocks, so this overestimates somewhat by
    reusing the full-UNet level-sum formula against a down-only architecture.
    """
    block_out_channels = controlnet_config.block_out_channels
    transformer_layers_per_block = controlnet_config.transformer_layers_per_block
    if isinstance(transformer_layers_per_block, int):
        transformer_layers_per_block = [transformer_layers_per_block] * len(block_out_channels)

    levels = []
    for level_index, (channels, num_transformer_layers) in enumerate(
        zip(block_out_channels, transformer_layers_per_block, strict=True)
    ):
        levels.append(
            UnetLevelConfig(
                hidden_dim=channels,
                num_transformer_layers=num_transformer_layers,
                downsample_factor=2**level_index,
            )
        )
    return levels


def _estimate_controlnet_activation_bytes(
    controlnet_component: torch.nn.Module,
    pipeline_name: str,
    latent: LatentArtifact,
    element_size: int,
) -> tuple[int, str | None]:
    """Estimate a single ControlNet's own activation cost.

    Reuses the same per-family formulas as the base denoiser
    (activation_formulas.py), parameterized by the ControlNet's own config --
    it runs its own forward pass each denoise step at the same spatial
    resolution as the base denoiser, so the same per-token cost model
    applies, just with the ControlNet's (usually smaller) layer count.
    """
    driver_class = get_driver_class(pipeline_name)
    family = driver_class.memory_family if driver_class is not None else None
    if family is None:
        return 0, f"No memory-estimation family registered for '{pipeline_name}' ControlNet; showing weights only."

    latent_shape = latent.shape
    height_latent, width_latent = latent_shape[-2], latent_shape[-1]

    try:
        if family == MemoryFamily.UNET_SDPA:
            levels = _get_controlnet_unet_levels(controlnet_component.config)
            activation_bytes = estimate_unet_sdpa_activation_bytes(levels, height_latent, width_latent, element_size)
            warning = (
                "ControlNet activation memory is an approximation: it reuses the full UNet-SDPA formula "
                "against the ControlNet's own config, which overestimates since the real ControlNetModel "
                "has no up-blocks."
            )
            return activation_bytes, warning

        fields = get_controlnet_denoiser_config_fields(pipeline_name, controlnet_component.config)
        if fields is None:
            return 0, f"No ControlNet config adapter registered for pipeline '{pipeline_name}'; showing weights only."

        activation_bytes = estimate_image_dit_sdpa_activation_bytes(
            height_latent,
            width_latent,
            fields.hidden_dim,
            fields.patch_size_spatial,
            element_size,
        )
        warning = (
            "ControlNet activation memory is an approximation: it reuses the base denoiser's per-token "
            "formula against the ControlNet's own (usually smaller) layer count."
        )
        return activation_bytes, warning
    except (AttributeError, KeyError, TypeError, IndexError) as e:
        return 0, f"Failed to read ControlNet config for pipeline '{pipeline_name}': {e}. Showing weights only."


def _estimate_controlnet_component(
    component_name: str,
    controlnet_component: torch.nn.Module,
    pipeline_name: str,
    latent: LatentArtifact,
) -> ComponentMemoryEstimate:
    weight_bytes = get_model_memory(controlnet_component)
    element_size = _activation_element_size(controlnet_component)
    activation_bytes, warning = _estimate_controlnet_activation_bytes(
        controlnet_component, pipeline_name, latent, element_size
    )
    return ComponentMemoryEstimate.create(
        component_name, ComponentRole.CONTROLNET, weight_bytes, activation_bytes, is_estimated=True, warning=warning
    )


def _estimate_component(
    component_name: str,
    component: torch.nn.Module,
    role: ComponentRole,
    pipe: DiffusionPipeline,
    pipeline_name: str,
    latent: LatentArtifact,
) -> ComponentMemoryEstimate:
    weight_bytes = get_model_memory(component)
    element_size = _activation_element_size(component)

    if role == ComponentRole.TEXT_ENCODER:
        activation_bytes = estimate_text_encoder_activation_bytes()
        return ComponentMemoryEstimate.create(component_name, role, weight_bytes, activation_bytes)

    if role == ComponentRole.VAE:
        try:
            activation_bytes = estimate_vae_activation_bytes(component, latent, element_size)
        except (AttributeError, KeyError, TypeError, IndexError, ValueError) as e:
            warning = f"Failed to estimate VAE activation memory: {e}. Showing weights only."
            return ComponentMemoryEstimate.create(
                component_name, role, weight_bytes, 0, is_estimated=True, warning=warning
            )
        vae_warning = vae_estimate_warning(component)
        return ComponentMemoryEstimate.create(
            component_name,
            role,
            weight_bytes,
            activation_bytes,
            is_estimated=vae_warning is not None,
            warning=vae_warning,
        )

    if role == ComponentRole.DENOISER:
        activation_bytes, warning = _estimate_denoiser_activation_bytes(
            component, pipe, pipeline_name, latent, element_size
        )
        return ComponentMemoryEstimate.create(
            component_name, role, weight_bytes, activation_bytes, is_estimated=warning is not None, warning=warning
        )

    if role == ComponentRole.CONTROLNET:
        return _estimate_controlnet_component(component_name, component, pipeline_name, latent)
    warning = f"Unrecognized component '{component_name}'; only weight memory could be estimated."
    return ComponentMemoryEstimate.create(component_name, role, weight_bytes, 0, is_estimated=True, warning=warning)


def _compute_peak_weight_topology(
    components: list[ComponentMemoryEstimate],
    offload_mode: str | None,
    denoiser_num_layers: int | None,
    resident_denoiser_bytes: int = 0,
) -> int:
    if not components:
        return 0
    if offload_mode is None:
        return sum(c.weight_bytes for c in components)
    if offload_mode == "model":
        return max(c.weight_bytes for c in components)
    # "sequential": approximate one denoiser layer's weight size, full weight for the
    # rest. Rough approximation, flagged for revisit if real-world accuracy misses target.
    largest = max(components, key=lambda c: c.weight_bytes)
    if largest.role == ComponentRole.DENOISER and denoiser_num_layers:
        base_denoiser_bytes = max(0, largest.weight_bytes - resident_denoiser_bytes)
        per_layer = base_denoiser_bytes // denoiser_num_layers
        others = [c.weight_bytes for c in components if c is not largest]
        return per_layer + max(others, default=0) + resident_denoiser_bytes
    if largest.role == ComponentRole.DENOISER:
        return max(c.weight_bytes for c in components)
    return max(c.weight_bytes for c in components) + resident_denoiser_bytes


def _compute_estimated_peak_bytes(peak_weight_bytes: int, components: list[ComponentMemoryEstimate]) -> int:
    """Combine the weight and activation terms into a peak VRAM figure.

    Models `torch.cuda.max_memory_allocated()`, then applies the single
    MEMORY_HEADROOM_FACTOR margin. Allocated is the portable target: it is fixed by the
    architecture and the resolution, so it reproduces exactly across runs and across GPUs.
    Everything above it -- CUDA context, cuBLAS/cuDNN workspaces, and the caching
    allocator's reserved-but-unused segments -- varies by card, driver and OS, and the
    allocator surrenders most of the last one under memory pressure anyway
    (`release_cached_blocks()` on allocation failure). Those belong in the blanket margin,
    not in a fitted constant.

    Activations are combined with `max`, not `sum`: text encoders run and are freed before
    denoise, and the denoiser's activations are freed before VAE decode. Those three
    phases never coexist, so summing them invented memory -- which happened to mask a much
    larger undercount in the VAE term until both were measured against a real run.
    """
    peak_activation_bytes = max((component.activation_bytes for component in components), default=0)
    return int((peak_weight_bytes + peak_activation_bytes) * MEMORY_HEADROOM_FACTOR)


def _controlnet_components(
    component_name: str,
    component: torch.nn.Module,
    pipeline_name: str,
    latent: LatentArtifact,
) -> list[ComponentMemoryEstimate]:
    """Return one ComponentMemoryEstimate per stacked ControlNet.

    Multi-ControlNet wrapper types (FluxMultiControlNetModel,
    QwenImageMultiControlNetModel, MultiControlNetModel (SDXL),
    SD3MultiControlNetModel) all expose `.nets: ModuleList[ControlNetModel]` --
    when present, each net is estimated individually so stacked ControlNets
    stay legible in the per-component breakdown; a single (non-stacked)
    ControlNet is estimated as one entry named "controlnet".
    """
    nets = getattr(component, "nets", None)
    if nets is None:
        return [_estimate_controlnet_component(component_name, component, pipeline_name, latent)]
    return [
        _estimate_controlnet_component(f"{component_name}_{index}", net, pipeline_name, latent)
        for index, net in enumerate(nets)
    ]


def _primary_denoiser_index(components: list[ComponentMemoryEstimate]) -> int | None:
    for preferred_name in ("transformer", "unet"):
        for index, component in enumerate(components):
            if component.component_name == preferred_name and component.role == ComponentRole.DENOISER:
                return index
    for index, component in enumerate(components):
        if component.role == ComponentRole.DENOISER:
            return index
    return None


def _lora_percentage(part: int, whole: int) -> str:
    if whole <= 0:
        return "N/A"
    return f"{part / whole * 100:.2f}%"


def _apply_lora_weights_to_components(
    components: list[ComponentMemoryEstimate], lora_adapters: list[LoraAdapterMemoryEstimate]
) -> tuple[list[ComponentMemoryEstimate], list[str], int]:
    if not lora_adapters:
        return components, [], 0

    adapter_warnings = [adapter.warning for adapter in lora_adapters if adapter.warning is not None]
    adapter_bytes = sum(adapter.weight_bytes for adapter in lora_adapters)
    denoiser_index = _primary_denoiser_index(components)
    if denoiser_index is None:
        warning = "Unfused LoRA weights could not be added because no transformer or UNet component was estimated."
        return components, [*adapter_warnings, warning], 0

    transformer = components[denoiser_index]
    base_weight_bytes = transformer.weight_bytes
    merged_weight_bytes = base_weight_bytes + adapter_bytes
    adapter_details = ", ".join(
        f"{adapter.adapter_name}: {to_human_readable_size(adapter.weight_bytes)}" for adapter in lora_adapters
    )
    tooltip = (
        f"Transformer weights: {to_human_readable_size(base_weight_bytes)}. "
        f"Unfused LoRA weights: {to_human_readable_size(adapter_bytes)} "
        f"({_lora_percentage(adapter_bytes, base_weight_bytes)} of base transformer weights; "
        f"{_lora_percentage(adapter_bytes, merged_weight_bytes)} of combined transformer weights). "
        f"Adapters: {adapter_details}."
    )
    updated_components = list(components)
    updated_components[denoiser_index] = replace(
        transformer,
        weight_bytes=merged_weight_bytes,
        total_bytes=merged_weight_bytes + transformer.activation_bytes,
        tooltip=tooltip,
    )
    return updated_components, adapter_warnings, adapter_bytes


def estimate_pipeline_memory(
    pipe: DiffusionPipeline,
    latent: LatentArtifact,
    pipeline_name: str,
    *,
    lora_adapters: list[LoraAdapterMemoryEstimate] | None = None,
) -> PipelineMemoryEstimate:
    """Estimate per-component memory usage for an already-loaded pipeline. Never executes the pipeline."""
    offload_mode = detect_offload_method(pipe)

    components: list[ComponentMemoryEstimate] = []
    for component_name, component in sorted(pipe.components.items()):
        if component is None or not hasattr(component, "parameters"):
            continue
        role = _classify_role(component_name)
        if role == ComponentRole.CONTROLNET:
            components.extend(_controlnet_components(component_name, component, pipeline_name, latent))
            continue
        components.append(_estimate_component(component_name, component, role, pipe, pipeline_name, latent))

    denoiser_num_layers = None
    driver_class = get_driver_class(pipeline_name)
    family = driver_class.memory_family if driver_class is not None else None
    if family is not None and family != MemoryFamily.UNET_SDPA:
        try:
            fields = get_denoiser_config_fields(pipeline_name, pipe)
        except (AttributeError, KeyError, TypeError, IndexError):
            fields = None
        if fields is not None:
            denoiser_num_layers = fields.num_layers

    peak_weight_bytes = _compute_peak_weight_topology(components, offload_mode, denoiser_num_layers)
    estimated_peak_bytes = _compute_estimated_peak_bytes(peak_weight_bytes, components)

    warnings = [c.warning for c in components if c.warning is not None]
    lora_adapters = lora_adapters or []
    components, lora_warnings, adapter_bytes = _apply_lora_weights_to_components(components, lora_adapters)
    warnings.extend(lora_warnings)
    peak_weight_bytes = _compute_peak_weight_topology(
        components, offload_mode, denoiser_num_layers, resident_denoiser_bytes=adapter_bytes
    )
    estimated_peak_bytes = _compute_estimated_peak_bytes(peak_weight_bytes, components)

    return PipelineMemoryEstimate(
        pipeline_name=pipeline_name,
        offload_mode=offload_mode,
        components=components,
        estimated_peak_bytes=estimated_peak_bytes,
        warnings=warnings,
        lora_adapters=lora_adapters,
    )


def estimate_component_memory(
    component: torch.nn.Module,
    component_name: str,
    role: ComponentRole | str,
    pipe: DiffusionPipeline,
    pipeline_name: str,
    latent: LatentArtifact,
) -> ComponentMemoryEstimate:
    """Estimate memory for a single component. Exposed independently of estimate_pipeline_memory
    to satisfy "estimate API for a configured pipeline or component" literally.
    """
    normalized_role = ComponentRole(role)
    return _estimate_component(component_name, component, normalized_role, pipe, pipeline_name, latent)


def _dtype_element_size(dtype: torch.dtype) -> int:
    return torch.zeros(1, dtype=dtype).element_size()


def resolve_pipeline_cls(artifact: DiffusionPipelineArtifact) -> type:
    """Resolve a registered Diffusers pipeline class without importing artifact data."""
    builder_module = artifact.builder_module
    builder_class_name = artifact.builder_class_name
    if not isinstance(builder_module, str) or not builder_module:
        msg = (
            f"Could not resolve pipeline class for pipeline '{artifact.pipeline_name}': "
            "the artifact is missing a valid builder_module or builder_class_name."
        )
        raise ValueError(msg)
    if not isinstance(builder_class_name, str) or not builder_class_name:
        msg = (
            f"Could not resolve pipeline class for pipeline '{artifact.pipeline_name}': "
            "the artifact is missing a valid builder_module or builder_class_name."
        )
        raise ValueError(msg)

    builder_identity = (builder_module, builder_class_name)
    builder_class = _TRUSTED_BUILDER_CLASSES.get(builder_identity)
    if builder_class is None:
        msg = (
            f"Could not resolve pipeline class for pipeline '{artifact.pipeline_name}': "
            f"builder identity '{builder_module}.{builder_class_name}' is not registered."
        )
        raise ValueError(msg)

    pipeline_cls = builder_class.pipeline_cls()
    if pipeline_cls.__name__ != artifact.pipeline_name:
        msg = (
            f"Could not resolve pipeline class for pipeline '{artifact.pipeline_name}': "
            f"builder identity '{builder_module}.{builder_class_name}' resolves to "
            f"'{pipeline_cls.__name__}'."
        )
        raise ValueError(msg)
    return pipeline_cls


def _module_component_slots(pipeline_cls: type) -> list[str]:
    component_slots, _ = pipeline_cls._get_signature_keys(pipeline_cls)  # noqa: SLF001
    signature_types = pipeline_cls._get_signature_types()  # noqa: SLF001
    module_slots = []
    for slot in component_slots:
        component_types = signature_types.get(slot, ())
        if any(
            isinstance(component_type, type) and issubclass(component_type, torch.nn.Module)
            for component_type in component_types
        ):
            module_slots.append(slot)
    return module_slots


def _load_base_model_index(repo_id: str, revision: str | None) -> dict[str, Any]:
    cached_path = try_to_load_from_cache(repo_id, filename="model_index.json", revision=revision)
    if not isinstance(cached_path, str):
        msg = (
            f"Attempted to estimate memory for base repo '{repo_id}'. "
            f"Failed because 'model_index.json' is not in the local HuggingFace cache for revision '{revision}'."
        )
        raise ComponentConfigNotCachedError(msg)

    model_index = try_load_json_dict(Path(cached_path))
    if model_index is None:
        msg = (
            f"Attempted to estimate memory for base repo '{repo_id}'. "
            f"Failed because cached 'model_index.json' at '{cached_path}' is not a valid JSON object."
        )
        raise ComponentConfigNotCachedError(msg)
    return model_index


def _model_index_has_component(model_index: dict[str, Any], slot: str) -> bool:
    component = model_index.get(slot)
    if not isinstance(component, (list, tuple)) or len(component) < 2:
        return False
    return component[0] is not None and component[1] is not None


def _weight_bearing_slots(
    pipeline_cls: type,
    component_overrides: dict[str, ComponentArtifact],
    *,
    all_overrides: bool,
    model_index: dict[str, Any] | None,
) -> list[str]:
    module_slots = _module_component_slots(pipeline_cls)
    if all_overrides:
        return sorted(set(module_slots) & set(component_overrides))

    optional_components = set(getattr(pipeline_cls, "_optional_components", []))
    return sorted(
        slot
        for slot in module_slots
        if slot in component_overrides
        or slot not in optional_components
        or (model_index is not None and _model_index_has_component(model_index, slot))
    )


def _stored_bytes_for_override(
    override: ComponentArtifact, effective_slot: str, pipeline_cls: type
) -> tuple[int, int, torch.nn.Module | None]:
    """Return (weight_bytes, stored_bytes_per_element, meta_component_or_None) for an
    override component, at its own declared dtype -- before any dynamic-quantization
    multiplier.

    `effective_slot` is the pipeline slot this override fills (the `_component_
    overrides` dict key), which takes precedence over `override.component` for class
    lookup -- mirrors `ComponentArtifact.materialize()` (component_artifact.py:106-119),
    since a generic override can be wired to a differently-named slot.

    Config is always attempted (best-effort) so activation formulas have shape data to
    read even for a SINGLE_FILE source. Weight bytes for a SINGLE_FILE source come from
    the file's on-disk size directly (Decision 5: it's one unsharded file, and for a
    GGUF file that size already reflects its final quantized bytes) rather than the
    meta-built component's shape-derived count.
    """
    if not isinstance(override, ModelComponentArtifact):
        msg = (
            f"Attempted to estimate memory for override component '{effective_slot}'. "
            f"Failed because '{type(override).__name__}' is not a ModelComponentArtifact."
        )
        raise TypeError(msg)
    torch_dtype = getattr(torch, override.torch_dtype)
    config_dict = override.try_read_config(pipeline_cls=pipeline_cls)
    meta_component = None
    if config_dict is not None:
        component_cls = get_component_class(pipeline_cls, effective_slot)
        meta_component = build_component_on_meta_device(component_cls, config_dict, torch_dtype)

    if override.source_type == ComponentSourceType.SINGLE_FILE:
        if not override.file_path:
            msg = (
                f"Attempted to estimate memory for override component '{override.component}'. "
                f"Failed because it has no file_path."
            )
            raise ValueError(msg)
        weight_bytes = Path(override.file_path).stat().st_size
        return weight_bytes, _dtype_element_size(torch_dtype), meta_component

    if meta_component is None:
        msg = (
            f"Attempted to estimate memory for override component '{override.component}'. "
            f"Failed because its config is not in the local HuggingFace cache."
        )
        raise ComponentConfigNotCachedError(msg)
    return get_model_memory(meta_component), _element_size(meta_component), meta_component


def _stored_bytes_for_base_component(
    slot: str,
    pipeline_cls: type,
    base_repo_id: str,
    base_revision: str | None,
    torch_dtype: torch.dtype,
) -> tuple[int, int, torch.nn.Module]:
    config_dict = resolve_base_component_config(base_repo_id, slot, base_revision)
    component_cls = get_component_class(pipeline_cls, slot)
    meta_component = build_component_on_meta_device(component_cls, config_dict, torch_dtype)
    return get_model_memory(meta_component), _element_size(meta_component), meta_component


def _controlnet_slot_estimates(
    artifact: ControlNetDiffusionPipelineArtifact,
    pipeline_name: str,
    latent: LatentArtifact,
    quantization_mode: str,
    transformer_layerwise_casting: bool,
) -> list[ComponentMemoryEstimate]:
    """Enumerate estimate entries for each stacked ControlNet on a not-yet-built pipeline.

    Reads `controlnet_models` (repo ids) directly off the artifact rather than
    `build_data` -- `ControlNetDiffusionPipelineArtifact.build_data` is inherited
    unmodified from the base artifact (`BaseDiffusionPipelineArtifact.__init__`), so
    it never contains ControlNet configuration.
    """
    driver_class = get_driver_class(pipeline_name)
    controlnet_cls = driver_class.controlnet_model_class if driver_class is not None else None
    if controlnet_cls is None:
        warning = (
            f"Attempted to estimate ControlNet memory for pipeline '{pipeline_name}'. "
            f"Failed because no ControlNet model class is registered for this pipeline."
        )
        return [
            ComponentMemoryEstimate.create(
                f"controlnet_{i}", ComponentRole.CONTROLNET, 0, 0, is_estimated=True, warning=warning
            )
            for i in range(len(artifact.controlnet_models))
        ]

    estimates: list[ComponentMemoryEstimate] = []
    controlnet_count = len(artifact.controlnet_models)
    for index, repo_id in enumerate(artifact.controlnet_models):
        component_name = "controlnet" if controlnet_count == 1 else f"controlnet_{index}"
        try:
            config_dict = resolve_base_component_config(repo_id, "controlnet")
        except ComponentConfigNotCachedError as e:
            estimates.append(
                ComponentMemoryEstimate.create(
                    component_name, ComponentRole.CONTROLNET, 0, 0, is_estimated=True, warning=str(e)
                )
            )
            continue

        meta_component = build_component_on_meta_device(controlnet_cls, config_dict, _BASE_PIPELINE_TORCH_DTYPE)
        stored_bytes_per_element = _element_size(meta_component)
        weight_bytes = get_model_memory(meta_component)
        effective_bytes_per_element = resolve_effective_bytes_per_element(
            stored_bytes_per_element,
            slot="controlnet",
            quantization_mode=quantization_mode,
            transformer_layerwise_casting=transformer_layerwise_casting,
            is_prequantized=artifact.is_prequantized,
            supports_layerwise_casting=artifact.supports_layerwise_casting,
        )
        if not artifact.is_prequantized and quantization_mode in QUANTIZATION_MODE_BYTES_PER_ELEMENT:
            weight_bytes = estimate_component_weight_bytes(
                meta_component,
                quantization_mode=quantization_mode,
                total_weight_bytes=weight_bytes,
            )
        else:
            weight_bytes = int(weight_bytes / stored_bytes_per_element * effective_bytes_per_element)
        # See the matching comment in the base-component loop below: activations are
        # never narrower than the compute floor, regardless of how the weights are stored.
        element_size = max(int(stored_bytes_per_element), _MIN_ACTIVATION_BYTES_PER_ELEMENT)

        activation_bytes, warning = _estimate_controlnet_activation_bytes(
            meta_component, pipeline_name, latent, element_size
        )
        estimates.append(
            ComponentMemoryEstimate.create(
                component_name,
                ComponentRole.CONTROLNET,
                weight_bytes,
                activation_bytes,
                is_estimated=True,
                warning=warning,
            )
        )
    return estimates


def estimate_pipeline_memory_from_build_data(
    artifact: DiffusionPipelineArtifact,
    latent: LatentArtifact,
    *,
    lora_adapters: list[LoraAdapterMemoryEstimate] | None = None,
) -> PipelineMemoryEstimate:
    """Estimate memory for a pipeline that has not been built yet.

    Never touches weight files or triggers a download: component configs are read from
    the warm HuggingFace cache and each component is built on the meta device
    (meta_device_builder.py) to get exact parameter shapes without allocating storage.
    See docs/spikes/memory_estimation_preload_api_plan.md.
    """
    build_data = artifact.build_data
    optimization_kwargs = artifact.optimization_kwargs
    pipeline_name = artifact.pipeline_name
    pipeline_cls = resolve_pipeline_cls(artifact)

    component_overrides: dict[str, ComponentArtifact] = build_data.get("_component_overrides", {})
    is_all_overrides = bool(build_data.get("_all_overrides"))
    base_repo_id = str(build_data.get("base_repo_id") or build_data.get("repo_id") or "")
    base_revision = build_data.get("base_revision") or build_data.get("revision")

    if not is_all_overrides and not base_repo_id:
        msg = (
            f"Attempted to estimate memory for pipeline '{pipeline_name}'. "
            f"Failed because build_data has no 'base_repo_id' and is not built purely from overrides."
        )
        raise ValueError(msg)

    model_index = None
    if not is_all_overrides:
        model_index = _load_base_model_index(base_repo_id, base_revision)
    slots = _weight_bearing_slots(
        pipeline_cls,
        component_overrides,
        all_overrides=is_all_overrides,
        model_index=model_index,
    )
    quantization_mode = optimization_kwargs.get("quantization_mode", "None")
    transformer_layerwise_casting = bool(optimization_kwargs.get("transformer_layerwise_casting", False))
    memory_optimization_strategy = optimization_kwargs.get("memory_optimization_strategy", "Manual")

    components: list[ComponentMemoryEstimate] = []
    denoiser_num_layers: int | None = None

    for slot in slots:
        role = _classify_role(slot)
        override = component_overrides.get(slot)
        try:
            if override is not None:
                weight_bytes, stored_bytes_per_element, meta_component = _stored_bytes_for_override(
                    override, slot, pipeline_cls
                )
            else:
                weight_bytes, stored_bytes_per_element, meta_component = _stored_bytes_for_base_component(
                    slot, pipeline_cls, base_repo_id, base_revision, _BASE_PIPELINE_TORCH_DTYPE
                )
        except ComponentConfigNotCachedError as e:
            components.append(ComponentMemoryEstimate.create(slot, role, 0, 0, is_estimated=True, warning=str(e)))
            continue

        effective_bytes_per_element = resolve_effective_bytes_per_element(
            stored_bytes_per_element,
            slot=slot,
            quantization_mode=quantization_mode,
            transformer_layerwise_casting=transformer_layerwise_casting,
            is_prequantized=artifact.is_prequantized,
            supports_layerwise_casting=artifact.supports_layerwise_casting,
        )
        if (
            not artifact.is_prequantized
            and meta_component is not None
            and not (transformer_layerwise_casting and slot == "transformer" and artifact.supports_layerwise_casting)
            and quantization_mode in QUANTIZATION_MODE_BYTES_PER_ELEMENT
        ):
            weight_bytes = estimate_component_weight_bytes(
                meta_component,
                quantization_mode=quantization_mode,
                total_weight_bytes=weight_bytes,
            )
        else:
            weight_bytes = int(weight_bytes / stored_bytes_per_element * effective_bytes_per_element)
        # Weights shrink under quantization; activations do not -- they are computed at the
        # compute dtype regardless. Deriving the activation width from
        # `effective_bytes_per_element` halved every activation term under fp8/int8 and, for
        # int4 (0.5 bytes), `int(0.5)` silently zeroed them.
        element_size = max(int(stored_bytes_per_element), _MIN_ACTIVATION_BYTES_PER_ELEMENT)

        if role == ComponentRole.TEXT_ENCODER:
            activation_bytes = estimate_text_encoder_activation_bytes()
            components.append(ComponentMemoryEstimate.create(slot, role, weight_bytes, activation_bytes))
            continue

        if role == ComponentRole.CONTROLNET:
            if meta_component is None:
                warning = "ControlNet config not available; showing weights only."
                components.append(
                    ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
                )
                continue
            activation_bytes, warning = _estimate_controlnet_activation_bytes(
                meta_component, pipeline_name, latent, element_size
            )
            components.append(
                ComponentMemoryEstimate.create(
                    slot, role, weight_bytes, activation_bytes, is_estimated=warning is not None, warning=warning
                )
            )
            continue

        if role == ComponentRole.VAE:
            if meta_component is None:
                warning = "VAE config not available; showing weights only."
                components.append(
                    ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
                )
                continue
            if memory_optimization_strategy == "Manual":
                apply_vae_memory_options(
                    meta_component,
                    slicing=bool(optimization_kwargs.get("vae_slicing", False)),
                    tiling=bool(optimization_kwargs.get("vae_tiling", False)),
                )
            else:
                # This matches the behaviour of _automatic_optimize_diffusion_pipeline on cuda (we ignore the behaviour on mps for now).
                apply_vae_memory_options(
                    meta_component,
                    slicing=True,
                    tiling=True,
                )
            try:
                activation_bytes = estimate_vae_activation_bytes(meta_component, latent, element_size)
            except (AttributeError, KeyError, TypeError, IndexError, ValueError) as e:
                warning = f"Failed to estimate VAE activation memory: {e}. Showing weights only."
                components.append(
                    ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
                )
                continue
            vae_warning = vae_estimate_warning(meta_component)
            components.append(
                ComponentMemoryEstimate.create(
                    slot,
                    role,
                    weight_bytes,
                    activation_bytes,
                    is_estimated=vae_warning is not None,
                    warning=vae_warning,
                )
            )
            continue

        if role == ComponentRole.OTHER:
            warning = f"Unrecognized component '{slot}'; only weight memory could be estimated."
            components.append(
                ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
            )
            continue

        # role == ComponentRole.DENOISER
        if meta_component is None:
            warning = "Denoiser config not available; showing weights only."
            components.append(
                ComponentMemoryEstimate.create(slot, role, weight_bytes, 0, is_estimated=True, warning=warning)
            )
            continue

        # Family adapters read the canonical `.transformer` / `.unet` attribute, so a second
        # denoiser slot such as WAN 2.2's `transformer_2` is exposed under its canonical name.
        # Family adapters read the canonical `.transformer` / `.unet` attribute, so a second
        # denoiser slot such as WAN 2.2's `transformer_2` is exposed under its canonical name.
        denoiser_pipe_stub = cast(Any, SimpleNamespace(**{slot_component_kind(slot): meta_component}))
        activation_bytes, warning = _estimate_denoiser_activation_bytes(
            meta_component, denoiser_pipe_stub, pipeline_name, latent, element_size
        )
        driver_class = get_driver_class(pipeline_name)
        family = driver_class.memory_family if driver_class is not None else None
        if family is not None and family != MemoryFamily.UNET_SDPA:
            try:
                fields = get_denoiser_config_fields(pipeline_name, denoiser_pipe_stub)
            except (AttributeError, KeyError, TypeError, IndexError):
                fields = None
            if fields is not None:
                denoiser_num_layers = fields.num_layers
        components.append(
            ComponentMemoryEstimate.create(
                slot, role, weight_bytes, activation_bytes, is_estimated=warning is not None, warning=warning
            )
        )

    if isinstance(artifact, ControlNetDiffusionPipelineArtifact):
        components.extend(
            _controlnet_slot_estimates(
                artifact, pipeline_name, latent, quantization_mode, transformer_layerwise_casting
            )
        )

    if memory_optimization_strategy == "Automatic":
        offload_mode = None
        automatic_warning = (
            "memory_optimization_strategy is 'Automatic'; the actual offload topology depends on free VRAM "
            "at build time and cannot be predicted before loading. Showing the upper bound (no offload "
            "assumed) -- actual usage after building will be equal to or lower than this. Set the strategy "
            "to Manual for an exact pre-load estimate."
        )
    else:
        # Translate the UI's cpu_offload_strategy vocabulary ("None"/"Model"/
        # "Sequential") into the same lowercase vocabulary detect_offload_method()
        # returns for the post-load path ("model"/"sequential"/None), so offload_mode
        # means the same thing regardless of which estimator produced it.
        cpu_offload_strategy = optimization_kwargs.get("cpu_offload_strategy", "None")
        offload_mode = {"None": None, "Model": "model", "Sequential": "sequential"}.get(cpu_offload_strategy)
        automatic_warning = None

    peak_weight_bytes = _compute_peak_weight_topology(components, offload_mode, denoiser_num_layers)
    estimated_peak_bytes = _compute_estimated_peak_bytes(peak_weight_bytes, components)

    warnings = [c.warning for c in components if c.warning is not None]
    if automatic_warning is not None:
        warnings.append(automatic_warning)
    lora_adapters = lora_adapters or []
    components, lora_warnings, adapter_bytes = _apply_lora_weights_to_components(components, lora_adapters)
    warnings.extend(lora_warnings)
    peak_weight_bytes = _compute_peak_weight_topology(
        components, offload_mode, denoiser_num_layers, resident_denoiser_bytes=adapter_bytes
    )
    estimated_peak_bytes = _compute_estimated_peak_bytes(peak_weight_bytes, components)

    return PipelineMemoryEstimate(
        pipeline_name=pipeline_name,
        offload_mode=offload_mode,
        components=components,
        estimated_peak_bytes=estimated_peak_bytes,
        warnings=warnings,
        lora_adapters=lora_adapters,
    )


def estimate_pipeline_memory_from_artifact(
    artifact: DiffusionPipelineArtifact,
    latent: LatentArtifact,
) -> PipelineMemoryEstimate:
    """Single public entry point: estimate memory for `artifact`, regardless of whether
    its pipeline has been built yet.

    Dispatches to the exact post-load estimator when the pipeline is already resident
    in model_cache under artifact.config_hash, otherwise to the pre-load, config-only
    estimator. See docs/spikes/memory_estimation_preload_api_plan.md, Decision 1.
    """
    lora_adapters = collect_runtime_lora_memory(artifact)
    cached_pipe = None
    if artifact.config_hash and model_cache.has_pipeline(artifact.config_hash):
        cached_pipe = model_cache.get_pipeline(artifact.config_hash)
    if cached_pipe is None:
        return estimate_pipeline_memory_from_build_data(artifact, latent, lora_adapters=lora_adapters)
    return estimate_pipeline_memory(
        cached_pipe,
        latent,
        artifact.pipeline_name,
        lora_adapters=lora_adapters,
    )
