"""Keep controlnet_registry.py's dicts in sync with each other and with the driver registry.

Mirrors test_memory_estimation_registry_consistency.py's rationale: these are
hand-maintained dicts keyed by pipeline-class-name strings, and nothing enforces
consistency at import time. A stale or missing entry silently downgrades a
ControlNet's memory estimate to weights-only instead of failing loudly.
"""

from __future__ import annotations

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY
from modular_diffusion_nodes_library.memory_estimation.controlnet_registry import (
    _CONTROLNET_ACTIVATION_FAMILY_BY_PIPELINE_NAME,
    _CONTROLNET_CLASS_BY_PIPELINE_NAME,
    _CONTROLNET_DENOISER_FIELD_ADAPTERS,
)
from modular_diffusion_nodes_library.memory_estimation.family_registry import MemoryFamily


def test_controlnet_class_and_activation_family_dicts_have_the_same_keys() -> None:
    class_keys = set(_CONTROLNET_CLASS_BY_PIPELINE_NAME)
    family_keys = set(_CONTROLNET_ACTIVATION_FAMILY_BY_PIPELINE_NAME)
    assert class_keys == family_keys, (
        f"_CONTROLNET_CLASS_BY_PIPELINE_NAME and _CONTROLNET_ACTIVATION_FAMILY_BY_PIPELINE_NAME "
        f"(controlnet_registry.py) must share the same pipeline keys. "
        f"Only in class dict: {sorted(class_keys - family_keys)}. "
        f"Only in family dict: {sorted(family_keys - class_keys)}."
    )


def test_every_controlnet_pipeline_has_a_driver() -> None:
    orphaned = set(_CONTROLNET_CLASS_BY_PIPELINE_NAME) - set(_DRIVER_REGISTRY)
    assert orphaned == set(), (
        f"These pipelines have a ControlNet class in _CONTROLNET_CLASS_BY_PIPELINE_NAME "
        f"(controlnet_registry.py) but no driver in _DRIVER_REGISTRY (driver_factory.py): "
        f"{sorted(orphaned)}."
    )


def test_every_non_unet_controlnet_pipeline_has_a_denoiser_field_adapter() -> None:
    """UNET_SDPA (SDXL) is the sole exception -- its ControlNet activation reads
    block_out_channels/transformer_layers_per_block directly, not through
    _CONTROLNET_DENOISER_FIELD_ADAPTERS."""
    missing = {
        pipeline_name
        for pipeline_name, family in _CONTROLNET_ACTIVATION_FAMILY_BY_PIPELINE_NAME.items()
        if family != MemoryFamily.UNET_SDPA and pipeline_name not in _CONTROLNET_DENOISER_FIELD_ADAPTERS
    }
    assert missing == set(), (
        f"These pipelines have a non-UNET_SDPA ControlNet activation family but no "
        f"_CONTROLNET_DENOISER_FIELD_ADAPTERS entry in controlnet_registry.py, so their ControlNet "
        f"activation memory silently falls back to weights-only: {sorted(missing)}."
    )


def test_no_orphaned_controlnet_denoiser_field_adapters() -> None:
    orphaned = set(_CONTROLNET_DENOISER_FIELD_ADAPTERS) - set(_CONTROLNET_CLASS_BY_PIPELINE_NAME)
    assert orphaned == set(), (
        f"These pipelines have a _CONTROLNET_DENOISER_FIELD_ADAPTERS entry in controlnet_registry.py "
        f"but no _CONTROLNET_CLASS_BY_PIPELINE_NAME entry: {sorted(orphaned)}."
    )
