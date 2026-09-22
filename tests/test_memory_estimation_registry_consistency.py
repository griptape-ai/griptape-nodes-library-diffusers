"""Keep the memory-estimation registries in sync with the driver registry.

`_FAMILY_BY_PIPELINE_NAME` and `_DENOISER_FIELD_ADAPTERS` (family_registry.py) are
hand-maintained dicts keyed by the same pipeline-class-name strings as
`_DRIVER_REGISTRY` (driver_factory.py), but nothing enforces that at import time. A new
provider or pipeline added to `_DRIVER_REGISTRY` without a matching family_registry.py
entry does not crash -- `_estimate_denoiser_activation_bytes` in
pipeline_memory_estimator.py catches the missing lookup and downgrades to a "showing
weights only" warning -- so the gap is a silent GPU-memory-estimation quality
regression rather than a loud failure. These tests make it loud.
"""

from __future__ import annotations

from modular_diffusion_nodes_library.latent_pipeline_drivers.driver_factory import _DRIVER_REGISTRY
from modular_diffusion_nodes_library.memory_estimation.family_registry import (
    _DENOISER_FIELD_ADAPTERS,
    _FAMILY_BY_PIPELINE_NAME,
    MemoryFamily,
)


def test_every_driver_pipeline_has_a_memory_family() -> None:
    """Every pipeline with a driver must also have a memory family, or its GPU memory
    estimate silently degrades to weights-only."""
    missing = set(_DRIVER_REGISTRY) - set(_FAMILY_BY_PIPELINE_NAME)
    assert missing == set(), (
        f"These pipelines are registered in _DRIVER_REGISTRY (driver_factory.py) but have no memory "
        f"family in _FAMILY_BY_PIPELINE_NAME (family_registry.py): {sorted(missing)}. Add an entry so "
        f"the pipeline's GPU memory estimate isn't silently downgraded to weights-only."
    )


def test_no_orphaned_memory_family_entries() -> None:
    """A family entry for a pipeline with no driver is dead/misleading: nothing ever builds that pipeline."""
    orphaned = set(_FAMILY_BY_PIPELINE_NAME) - set(_DRIVER_REGISTRY)
    assert orphaned == set(), (
        f"These pipelines have a memory family in _FAMILY_BY_PIPELINE_NAME (family_registry.py) but no "
        f"driver in _DRIVER_REGISTRY (driver_factory.py): {sorted(orphaned)}. Remove the stale entry or "
        f"register the missing driver."
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
        for pipeline_name, family in _FAMILY_BY_PIPELINE_NAME.items()
        if pipeline_name in _DRIVER_REGISTRY
        and family != MemoryFamily.UNET_SDPA
        and pipeline_name not in _DENOISER_FIELD_ADAPTERS
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
    used_families = set(_FAMILY_BY_PIPELINE_NAME.values())
    unused = set(MemoryFamily) - used_families
    assert unused == set(), (
        f"These MemoryFamily values are never assigned to any pipeline: {sorted(family.value for family in unused)}."
    )
