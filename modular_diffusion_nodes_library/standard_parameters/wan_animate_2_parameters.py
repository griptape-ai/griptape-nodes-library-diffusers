from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

from griptape_nodes.exe_types.node_types import BaseNode

from modular_diffusion_nodes_library.parameters.connectable_huggingface_repo_parameter import (
    ConnectableHuggingFaceRepoParameter,
)
from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)
from modular_diffusion_nodes_library.utils.torch_utils import get_best_device

if TYPE_CHECKING:
    from diffusers import ModularPipeline  # type: ignore[reportMissingImports]

logger = logging.getLogger("modular_diffusers_nodes_library")


class WanAnimate2PipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls_path = "diffusers.modular_pipelines.wan_animate_2.modular_pipeline:WanAnimate2ModularPipeline"
    _component_slots: ClassVar[list[str]] = []
    _repo_ids: ClassVar[list[str]] = ["Wan-AI/Wan2.2-Animate-2-14B-Diffusers"]

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        """WanAnimate2ModularPipeline does not support overrides."""
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = ConnectableHuggingFaceRepoParameter(
            node,
            repo_ids=self._repo_ids,
            parameter_name="model",
            list_all_models=list_all_models,
        )

    def add_input_parameters(self) -> None:
        self._model_repo_parameter.add_input_parameters()

    def remove_input_parameters(self) -> None:
        self._model_repo_parameter.remove_input_parameters()

    def get_config_kwargs(self) -> dict:
        return {
            "model": self._node.get_parameter_value("model"),
        }

    def validate_before_node_run(self) -> list[Exception] | None:
        errors = []
        model_errors = self._model_repo_parameter.validate_before_node_run()
        if model_errors:
            errors.extend(model_errors)
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        repo_id, revision = self._model_repo_parameter.get_repo_revision()
        return {
            "repo_id": repo_id,
            "revision": revision,
        }

    def requires_device_map(self) -> bool:
        # Used here as an opt-out from post-hoc pipeline optimization, not literally to request an
        # accelerate device_map. On a ModularPipeline `optimize_diffusion_pipeline` would call
        # `.to(device)` while both `enable_*_cpu_offload` calls are hasattr-gated and silently skipped.
        # Placement is instead owned by `_load` via the ComponentsManager below.
        return True

    def is_prequantized(self) -> bool:
        # Suppresses quantization and layerwise casting, which fire before the requires_device_map
        # short-circuit. Neither is safe to apply on top of the ComponentsManager offload hooks.
        return True

    @classmethod
    def _build_pipeline_from_repo(cls, build_data: dict[str, Any], overrides: dict[str, Any]) -> ModularPipeline:  # type: ignore[reportAttributeAccessIssue]
        return cls._load(build_data)

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> ModularPipeline:
        return cls._load(build_data)

    @classmethod
    def _load(cls, build_data: dict[str, Any]) -> ModularPipeline:
        # `from_pretrained` reads `_class_name` from the repo's `modular_model_index.json`, so it returns
        # the standard or the distilled pipeline class to match the repo. It resolves the component specs
        # but loads no weights; `load_components` fetches them. The transformer is compiled later, by the
        # driver, on the first denoise.
        import torch  # type: ignore[reportMissingImports]
        from diffusers import ComponentsManager, ModularPipeline  # type: ignore[reportMissingImports]

        manager = ComponentsManager()
        pipe = ModularPipeline.from_pretrained(
            build_data["repo_id"],
            revision=build_data["revision"],
            components_manager=manager,
            local_files_only=True,
        )
        pipe.load_components(dtype=torch.bfloat16)
        manager.enable_auto_cpu_offload(device=get_best_device())
        return pipe  # type: ignore[reportReturnType]


class WanAnimate2DistilledPipelineParameters(WanAnimate2PipelineParameters):
    _pipeline_cls_path = (
        "diffusers.modular_pipelines.wan_animate_2.modular_pipeline:WanAnimate2DistilledModularPipeline"
    )
    _repo_ids: ClassVar[list[str]] = ["Wan-AI/Wan2.2-Animate-2-14B-Distilled-Diffusers"]
