"""A pipeline built purely from component overrides must not need `_pipeline_cls` in its build data.

The builder node omits that entry because resolving it imports diffusers on the orchestrator, so the
building process resolves the class from the params class instead.
"""

from __future__ import annotations

from typing import Any

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)


class _FakePipeline:
    def __init__(self, **components: Any) -> None:
        self.components = components


class _FakePipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    @classmethod
    def pipeline_cls(cls) -> type:  # type: ignore[override]
        return _FakePipeline


def test_overrides_only_build_resolves_pipeline_class_without_build_data_entry() -> None:
    build_data = {"_component_overrides": {}, "_all_overrides": True}

    pipe = _FakePipelineParameters._build_pipeline_from_overrides_only(build_data, {"vae": "the-vae"})  # noqa: SLF001

    assert isinstance(pipe, _FakePipeline)
    assert pipe.components == {"vae": "the-vae"}
