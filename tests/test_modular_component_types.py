"""Component classes and slot aliases for modular pipelines."""

from __future__ import annotations

import pytest


@pytest.mark.parametrize("pipeline_name", ["Flux3ModularPipeline", "MiniMaxH3ModularPipeline"])
def test_modular_component_lookup_uses_real_block_metadata(pipeline_name: str) -> None:
    diffusers = pytest.importorskip("diffusers")
    from modular_diffusion_nodes_library.component_loading.pipeline_components import pipeline_component_types
    from modular_diffusion_nodes_library.component_loading.pipeline_type_registry import get_component_class

    pipeline_cls = getattr(diffusers, pipeline_name)
    components = pipeline_component_types(pipeline_cls)

    assert "transformer" in components
    assert "vae" in components
    assert get_component_class(pipeline_cls, "transformer") is components["transformer"]


def test_audio_component_aliases_keep_model_and_scheduler_types() -> None:
    from modular_diffusion_nodes_library.component_loading.component_slots import (
        component_config_filename,
        slot_artifact_type_name,
    )
    from modular_diffusion_nodes_library.nodes.load_component_node import _LOADABLE_COMPONENTS

    assert _LOADABLE_COMPONENTS["Audio VAE"] == "audio_vae"
    assert slot_artifact_type_name("audio_vae") == "VaeComponentArtifact"
    assert slot_artifact_type_name("audio_scheduler") == "SchedulerComponentArtifact"
    assert component_config_filename("audio_scheduler") == "scheduler_config.json"
