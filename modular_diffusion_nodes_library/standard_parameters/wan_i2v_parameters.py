# Copied from diffusers_nodes_library/common/parameters/diffusion/wan/img2vid_parameters.py
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


class WanImageToVideoPipelineParameters(ModularDiffusionPipelineTypePipelineParameters):
    def __init__(self, node: BaseNode, *, list_all_models: bool = False):
        super().__init__(node)
        self._model_repo_parameter = HuggingFaceRepoParameter(
            node,
            repo_ids=[
                "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers",
                "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
                "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
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
        return diffusers.WanImageToVideoPipeline  # type: ignore[reportAttributeAccessIssue]

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
        # WAN I2V models are 14B parameters (~28 GB in bfloat16). Loading the full
        # model into CPU RAM before CUDA placement exhausts memory on most systems.
        # Using device_map lets accelerate place each layer directly on the right
        # device during loading, avoiding the CPU-RAM spike entirely.
        return True

    @classmethod
    def build_pipeline_from_build_data(cls, build_data: dict[str, Any]) -> diffusers.WanImageToVideoPipeline:  # type: ignore[reportAttributeAccessIssue]
        return diffusers.WanImageToVideoPipeline.from_pretrained(  # type: ignore[reportAttributeAccessIssue]
            pretrained_model_name_or_path=build_data["repo_id"],
            revision=build_data["revision"],
            torch_dtype=torch.bfloat16,
            local_files_only=True,
            device_map="balanced",
        )
