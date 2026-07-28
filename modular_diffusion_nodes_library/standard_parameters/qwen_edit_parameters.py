import logging
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from modular_diffusion_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class QwenEditPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    _pipeline_cls = diffusers.QwenImageEditPipeline  # type: ignore[reportAttributeAccessIssue]

    @classmethod
    def supports_build_from_overrides_only(cls) -> bool:
        """QwenImageEditPipeline requires 'processor' which is not in ALLOWED_COMPONENT_SLOTS."""
        return False

    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "Qwen/Qwen-Image-Edit",
                "Qwen/Qwen-Image-Edit-2509",
                "Qwen/Qwen-Image-Edit-2511",
            ],
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
        errors = self._model_repo_parameter.validate_before_node_run()
        return errors or None

    def get_build_data(self) -> dict[str, Any]:
        base_repo_id, base_revision = self._resolve_repo(self._model_repo_parameter)
        return {
            "base_repo_id": base_repo_id,
            "base_revision": base_revision,
        }

    @classmethod
    def _build_pipeline_from_repo(
        cls, build_data: dict[str, Any], overrides: dict[str, Any]
    ) -> diffusers.QwenImageEditPipeline:  # type: ignore[reportAttributeAccessIssue]
        return cls._pipeline_cls.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=build_data["base_repo_id"],
            revision=build_data["base_revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            **overrides,
        )
