# Copied from diffusers_nodes_library/common/parameters/diffusion/wan/wan_parameters.py
import logging
from typing import Any

import diffusers  # type: ignore[reportMissingImports]
import torch  # type: ignore[reportMissingImports]
from griptape_nodes.exe_types.node_types import BaseNode
from griptape_nodes.exe_types.param_components.huggingface.huggingface_repo_parameter import HuggingFaceRepoParameter

from diffusers_nodes_library.parameters.modular_pipeline_type_parameters import (
    ModularDiffusionPipelineTypePipelineParameters,
)

logger = logging.getLogger("modular_diffusers_nodes_library")


class WanPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
                "Wan-AI/Wan2.1-T2V-14B-Diffusers",
                "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
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

    @property
    def pipeline_class(self) -> type:
        return diffusers.WanPipeline  # type: ignore[reportAttributeAccessIssue]

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
        # The 14B and A14B WAN T2V models (~28 GB in bfloat16) load the entire model
        # into CPU RAM before any CUDA placement, exhausting memory on many systems.
        # Using device_map lets accelerate stream each layer directly to the right
        # device during loading so we never materialize the full model on CPU.
        # The 1.3B model is small enough that this is harmless for it too.
        return True

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> diffusers.WanPipeline:  # type: ignore[reportAttributeAccessIssue]
        return diffusers.WanPipeline.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=build_data["repo_id"],
            revision=build_data["revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="balanced",
        )
