"""Keep controlnet_registry.py's adapters in sync with the driver registry.

Mirrors test_memory_estimation_registry_consistency.py's rationale: the denoiser
config adapters are hand-maintained and keyed by pipeline-class-name strings, and
nothing enforces consistency at import time. A stale or missing entry silently
downgrades a ControlNet's memory estimate to weights-only instead of failing loudly.
"""

from __future__ import annotations

import pytest

pytest.importorskip("diffusers", reason="Loads real driver classes; run in the execution environment.")

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import (
    get_driver_class,
    supported_pipeline_classes,
)
from modular_diffusion_nodes_library.memory_estimation.controlnet_registry import _CONTROLNET_DENOISER_FIELD_ADAPTERS
from modular_diffusion_nodes_library.memory_family import MemoryFamily


def _registered_driver_classes():
    driver_classes = {}
    for pipeline_name in supported_pipeline_classes():
        driver_class = get_driver_class(pipeline_name)
        assert driver_class is not None, f"{pipeline_name}: no driver class is registered."
        driver_classes[pipeline_name] = driver_class
    return driver_classes


_DRIVER_CLASSES = _registered_driver_classes()
_CONTROLNET_PIPELINE_NAMES = {
    pipeline_name
    for pipeline_name, driver_class in _DRIVER_CLASSES.items()
    if driver_class.controlnet_model_class is not None
}


def test_every_non_unet_controlnet_pipeline_has_a_denoiser_field_adapter() -> None:
    """UNET_SDPA (SDXL) is the sole exception -- its ControlNet activation reads
    block_out_channels/transformer_layers_per_block directly, not through
    _CONTROLNET_DENOISER_FIELD_ADAPTERS."""
    missing = {
        pipeline_name
        for pipeline_name in _CONTROLNET_PIPELINE_NAMES
        if _DRIVER_CLASSES[pipeline_name].memory_family != MemoryFamily.UNET_SDPA
        and pipeline_name not in _CONTROLNET_DENOISER_FIELD_ADAPTERS
    }
    assert missing == set(), (
        f"These pipelines have a controlnet_model_class and a non-UNET_SDPA memory_family but no "
        f"_CONTROLNET_DENOISER_FIELD_ADAPTERS entry in controlnet_registry.py, so their ControlNet "
        f"activation memory silently falls back to weights-only: {sorted(missing)}."
    )


def test_no_orphaned_controlnet_denoiser_field_adapters() -> None:
    orphaned = set(_CONTROLNET_DENOISER_FIELD_ADAPTERS) - _CONTROLNET_PIPELINE_NAMES
    assert orphaned == set(), (
        f"These pipelines have a _CONTROLNET_DENOISER_FIELD_ADAPTERS entry in controlnet_registry.py "
        f"but their driver has no controlnet_model_class: {sorted(orphaned)}."
    )
