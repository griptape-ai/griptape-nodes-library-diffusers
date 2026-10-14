"""Keep driver memory-estimation metadata in sync with the estimator adapters.

Each registered driver declares its activation-memory family as a class variable.
The denoiser config adapters remain keyed by pipeline class name because they normalize
pipeline-specific config fields. These tests make missing metadata loud instead of
silently downgrading the estimate to weights-only.
"""

from __future__ import annotations

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY
from modular_diffusion_nodes_library.memory_estimation.family_registry import _DENOISER_FIELD_ADAPTERS
from modular_diffusion_nodes_library.memory_family import MemoryFamily


def test_every_driver_pipeline_has_a_memory_family() -> None:
    """Every registered driver must declare a family, or estimates degrade to weights-only."""
    missing = {
        pipeline_name for pipeline_name, driver_class in _DRIVER_REGISTRY.items() if driver_class.memory_family is None
    }
    assert missing == set(), (
        f"These pipelines are registered in _DRIVER_REGISTRY but their drivers have no memory_family: {sorted(missing)}."
    )


def test_every_non_unet_driver_pipeline_has_a_denoiser_adapter() -> None:
    """Every family except UNET_SDPA reads denoiser shape through `_DENOISER_FIELD_ADAPTERS`.

    UNET_SDPA (SDXL) is the sole documented exception: it reads shape via
    `get_sdxl_unet_levels` instead, because UNet2DConditionModel has no flat
    num_layers/hidden_dim. A pipeline in any other family with no adapter entry makes
    `get_denoiser_config_fields` return None, which silently falls back to "showing
    weights only" instead of raising.
    """
    missing = {
        pipeline_name
        for pipeline_name, driver_class in _DRIVER_REGISTRY.items()
        if driver_class.memory_family != MemoryFamily.UNET_SDPA and pipeline_name not in _DENOISER_FIELD_ADAPTERS
    }
    assert missing == set(), (
        f"These pipelines have a non-UNET_SDPA memory family but no _DENOISER_FIELD_ADAPTERS entry in "
        f"family_registry.py, so their denoiser activation memory silently falls back to weights-only: "
        f"{sorted(missing)}."
    )


def test_no_orphaned_denoiser_adapters() -> None:
    """An adapter for a pipeline with no driver is dead code that nothing ever calls."""
    orphaned = set(_DENOISER_FIELD_ADAPTERS) - set(_DRIVER_REGISTRY)
    assert orphaned == set(), (
        f"These pipelines have a _DENOISER_FIELD_ADAPTERS entry in family_registry.py but no driver in "
        f"_DRIVER_REGISTRY (driver_factory.py): {sorted(orphaned)}."
    )


def test_every_memory_family_is_used() -> None:
    """An unused MemoryFamily value suggests dead code or a forgotten wiring."""
    used_families = {driver_class.memory_family for driver_class in _DRIVER_REGISTRY.values()}
    unused = set(MemoryFamily) - used_families
    assert unused == set(), (
        f"These MemoryFamily values are never assigned to any pipeline: {sorted(family.value for family in unused)}."
    )
