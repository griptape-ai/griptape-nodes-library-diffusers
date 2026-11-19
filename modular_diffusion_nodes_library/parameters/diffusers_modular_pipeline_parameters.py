from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

from modular_diffusion_nodes_library.component_loading.modular_pipeline_loader import (
    ModularPipelineLoader,
    ModularPipelineLoadOptions,
)
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

if TYPE_CHECKING:
    from diffusers.modular_pipelines.modular_pipeline import ModularPipeline


class DiffusersModularPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    """Integrate repository component loading without changing classic pipeline dispatch."""

    supported_workflows: ClassVar[tuple[str, ...]] = ()

    @classmethod
    @abstractmethod
    def get_modular_load_options(cls, build_data: dict[str, Any]) -> ModularPipelineLoadOptions:
        raise NotImplementedError

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> ModularPipeline:
        return cls._load_modular_pipeline(build_data)

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> ModularPipeline:
        return cls._load_modular_pipeline({**build_data, "_component_overrides": overrides})

    @classmethod
    def _load_modular_pipeline(cls, build_data: dict[str, Any]) -> ModularPipeline:
        loader = ModularPipelineLoader(
            pipeline_cls=cls.pipeline_cls(), options=cls.get_modular_load_options(build_data)
        )
        return loader.from_pretrained(
            build_data["base_repo_id"],
            revision=build_data.get("base_revision") or build_data.get("revision"),
            overrides=build_data.get("_component_overrides", {}),
        )
